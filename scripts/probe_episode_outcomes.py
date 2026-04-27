#!/usr/bin/env python3
"""Heuristically score each episode for "looks like a successful insertion".

We don't store an explicit success label in the parquet (the
``/scoring/insertion_event`` topic only fires during recording but
isn't dumped per-frame), so this script infers outcome from signals
that ARE in observation.state + action:

  - Final TCP z height vs. the per-episode min z reached (a real
    insertion ends with the gripper held at the seated depth, so the
    last ~30 frames sit within ~1 mm of the per-episode minimum).
  - Final action magnitude (a controlled stop has |a| ~= 0; a panic
    abort or a "ran out of time" termination usually has the operator
    still mid-motion).
  - End-of-episode wrench magnitude (post-2026-04-24 datasets carry
    tared F/T as the last 6 obs columns; a seated cable shows a
    sustained ~2-10 N reaction; an open-air hover shows ~0 N).

This is heuristic, not authoritative -- but for diagnosing "which one
of these N episodes did the operator forget to discard?" it's plenty.

Usage:
    pixi run python scripts/probe_episode_outcomes.py \
        teleop-dataset-gamepad/sfp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _read_columns(files: list[Path], cols: list[str]) -> dict[str, np.ndarray]:
    chunks: dict[str, list] = {c: [] for c in cols}
    for fp in files:
        tbl = pq.read_table(fp, columns=cols)
        for c in cols:
            arr = tbl.column(c).to_numpy(zero_copy_only=False)
            if arr.ndim == 1 and isinstance(arr[0], np.ndarray):
                arr = np.stack(list(arr))
            chunks[c].append(arr)
    return {c: np.concatenate(v, axis=0) for c, v in chunks.items()}


def _episode_index(files: list[Path]) -> np.ndarray:
    out = []
    for fp in files:
        tbl = pq.read_table(fp, columns=["episode_index"])
        out.append(tbl.column("episode_index").to_numpy())
    return np.concatenate(out, axis=0).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root (contains meta/ and data/)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    info = _load_info(root)
    state_names = info["features"]["observation.state"]["names"]

    files = sorted((root / "data").rglob("*.parquet"))
    cols = _read_columns(files, ["observation.state", "action"])
    state = cols["observation.state"]
    action = cols["action"]
    ep = _episode_index(files)

    # Find indices of the columns we care about
    def idx(name):
        return state_names.index(name) if name in state_names else None

    iz = idx("tcp_pose.position.z")
    iwf_x = idx("wrench.force.x")
    iwf_y = idx("wrench.force.y")
    iwf_z = idx("wrench.force.z")
    has_wrench = None not in (iwf_x, iwf_y, iwf_z)

    print(f"root: {root}")
    print(f"episodes: {info['total_episodes']}  frames: {info['total_frames']}  fps: {info['fps']}")
    print(f"observation.state columns: {len(state_names)}  has_wrench: {has_wrench}")
    print()

    rows = []
    for e in np.unique(ep):
        mask = ep == e
        T = int(mask.sum())
        z = state[mask, iz]
        a = action[mask]

        # Trajectory features
        z_min = float(z.min())
        z_max = float(z.max())
        z_end = float(z[-1])
        z_descent = z_max - z_min  # total z-range used during episode
        # Last 1s of frames: how close to the per-episode minimum is the
        # gripper sitting?  A real insertion ends "parked" near z_min;
        # a failed episode often ends mid-air with z >> z_min.
        last_n = min(int(info["fps"]), T)
        z_settle = float(z[-last_n:].mean())
        z_settle_above_min = z_settle - z_min

        # Final action magnitude (linear only -- yaw can be nonzero
        # during a held position).
        a_end_lin = float(np.linalg.norm(a[-last_n:, :3], axis=1).mean())

        # Wrench at end -- a seated cable produces sustained reaction
        # force.  Open-air hover should be near zero (post-tare).
        if has_wrench:
            f_end = float(np.linalg.norm(state[mask][-last_n:, [iwf_x, iwf_y, iwf_z]], axis=1).mean())
            f_max = float(np.linalg.norm(state[mask][:, [iwf_x, iwf_y, iwf_z]], axis=1).max())
        else:
            f_end = float("nan")
            f_max = float("nan")

        rows.append({
            "ep": int(e),
            "T": T,
            "dur_s": T / info["fps"],
            "z_descent_mm": z_descent * 1000.0,
            "z_settle_above_min_mm": z_settle_above_min * 1000.0,
            "a_end_lin": a_end_lin,
            "f_end_N": f_end,
            "f_max_N": f_max,
        })

    # Print a per-episode table sorted by episode number
    print(f"{'ep':>3} {'T':>5} {'dur_s':>6} "
          f"{'z_desc_mm':>10} {'z_settle+mm':>11} "
          f"{'|a_end|':>9} {'f_end_N':>8} {'f_max_N':>8}  HINT")
    rows.sort(key=lambda r: r["ep"])
    for r in rows:
        # Heuristic flag: insertion looks unlikely if the gripper
        # didn't settle near its descent minimum (>2 mm above) AND the
        # final wrench is essentially zero (cable not in port).  Loose
        # threshold; tighten as we collect ground truth.
        likely_failed = (
            r["z_settle_above_min_mm"] > 2.0
            and (np.isnan(r["f_end_N"]) or r["f_end_N"] < 0.5)
        )
        # Other suspicious patterns:
        bored_end = r["a_end_lin"] > 0.005   # operator was still moving at episode end
        no_descent = r["z_descent_mm"] < 5.0  # never went down meaningfully

        flags = []
        if likely_failed: flags.append("NO_SEAT")
        if bored_end:     flags.append("MOTION_AT_END")
        if no_descent:    flags.append("NO_DESCENT")

        print(f"{r['ep']:>3} {r['T']:>5} {r['dur_s']:>6.1f} "
              f"{r['z_descent_mm']:>10.1f} {r['z_settle_above_min_mm']:>11.1f} "
              f"{r['a_end_lin']:>9.4f} {r['f_end_N']:>8.2f} {r['f_max_N']:>8.2f}  "
              f"{' '.join(flags) if flags else 'ok'}")

    print()
    print("Column meanings:")
    print("  z_desc_mm     : peak-to-trough z range during the episode (gripper descent magnitude)")
    print("  z_settle+mm   : gripper height at end of episode minus its per-episode minimum.")
    print("                  ~0 mm => gripper finished AT its descent floor (= seated).")
    print("                  >2 mm => gripper finished hovering ABOVE where it had been.")
    print("                  This is the strongest signal that an insertion did NOT seat.")
    print("  |a_end|       : average linear-velocity magnitude over the last 1 s")
    print("                  (>0.005 means the operator never let go before saving)")
    print("  f_end_N       : average tared force magnitude over the last 1 s")
    print("                  (>~0.5 N during a hold = seated cable; ~0 = open air)")
    print("  f_max_N       : peak force magnitude during the episode")
    print()
    print("Flags:  NO_SEAT       insertion-shape signal absent (most useful)")
    print("        MOTION_AT_END operator hadn't stopped before pressing save")
    print("        NO_DESCENT    gripper never moved down meaningfully (probably wrong scene)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
