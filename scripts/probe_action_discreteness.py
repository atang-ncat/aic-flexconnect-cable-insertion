#!/usr/bin/env python3
"""Quick action-space discreteness probe for teleop-dataset-ft-v1.

We're suspicious that even after "smoothing", the dataset action values are
still bunched at {-0.1, 0, +0.1} rather than continuously distributed.  This
script answers, per action dimension:

  * What fraction of values land near {-0.1, 0, +0.1}?
  * What's the histogram of |action[t]| inside the active band (|a|>1e-3)?
  * What's the step-to-step delta distribution (does it jump 0 -> 0.1, or
    smoothly ramp through intermediate values)?
  * How many unique values do we actually see per dim, after rounding to a
    grid?
  * Are episodes really smoothed, or is each episode just zero / saturated?

Run from anywhere:

    cd src/aic && pixi run -- python /scratch2/atang/ws_aic/scripts/probe_action_discreteness.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _load_info(root: Path) -> dict:
    with open(root / "meta" / "info.json") as f:
        return json.load(f)


def _read_columns(files, cols):
    blocks = {c: [] for c in cols}
    for f in files:
        t = pq.read_table(f, columns=cols, use_threads=True)
        for c in cols:
            col_data = t[c]
            for i in range(t.num_rows):
                blocks[c].append(np.asarray(col_data[i].as_py(), dtype=np.float64))
    return {c: np.stack(blocks[c], axis=0) for c in cols}


def _read_episode_index(files) -> np.ndarray:
    parts: list[np.ndarray] = []
    for f in files:
        t = pq.read_table(f, columns=["episode_index"], use_threads=True)
        parts.append(np.asarray(t["episode_index"].to_pylist(), dtype=np.int64))
    return np.concatenate(parts, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", nargs="?", default=None,
        help="Dataset root (contains meta/ and data/)",
    )
    ap.add_argument(
        "--max-episodes", type=int, default=0,
        help="Limit to first N episodes (0 = all). Useful for big raw datasets.",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve() if args.root else repo / "teleop-dataset-ft-v1"
    info = _load_info(root)
    names = info["features"]["action"]["names"]

    files = sorted((root / "data").rglob("*.parquet"))
    print(f"root: {root}")
    print(f"files: {len(files)}  episodes: {info['total_episodes']}  frames: {info['total_frames']}")

    cols = _read_columns(files, ["action"])
    actions = cols["action"]  # (N, 6)
    ep = _read_episode_index(files)  # (N,)
    if args.max_episodes > 0:
        keep_mask = ep < args.max_episodes
        actions = actions[keep_mask]
        ep = ep[keep_mask]
        print(f"(limited to first {args.max_episodes} episodes -> {len(ep)} frames)")
    n, A = actions.shape
    assert A == len(names), (A, len(names))

    print(f"actions: shape {actions.shape}  dtype {actions.dtype}")
    print()

    # 1. Bucket each value into one of {neg-sat, near-zero, pos-sat, mid}.
    #    Saturation cutoff: > 0.95 * max abs (so 0.0998 still counts).
    print("=== Per-dim bucket fractions ===")
    sat_thr = 0.095   # absolute, since clip is 0.1
    zero_thr = 1e-3   # "essentially zero"
    print(f"{'dim':<12}  {'max|a|':>9}  {'%neg-sat':>9}  {'%pos-sat':>9}  {'%~zero':>9}  {'%mid':>9}  {'#unique@1e-4':>12}")
    for i, nm in enumerate(names):
        a = actions[:, i]
        max_abs = float(np.abs(a).max())
        neg_sat = float((a < -sat_thr).mean()) * 100
        pos_sat = float((a > sat_thr).mean()) * 100
        near_zero = float((np.abs(a) < zero_thr).mean()) * 100
        mid = 100.0 - neg_sat - pos_sat - near_zero
        # Quantize to 1e-4 grid to count effective discreteness, ignoring
        # tiny float-precision wiggles.
        n_unique = len(np.unique(np.round(a / 1e-4).astype(np.int64)))
        print(f"{nm:<12}  {max_abs:>9.5f}  {neg_sat:>8.2f}%  {pos_sat:>8.2f}%  {near_zero:>8.2f}%  {mid:>8.2f}%  {n_unique:>12d}")
    print()

    # 2. Mid-range (active but not saturated) magnitude histogram per dim.
    #    If smoothing produces a real ramp, mid values should be DENSE here.
    print("=== Mid-band coverage (zero_thr < |a| < sat_thr) ===")
    bins = np.array([1e-3, 5e-3, 1e-2, 2e-2, 5e-2, sat_thr])
    print(f"{'dim':<12}  " + "  ".join(f"<{b:.3f}".rjust(9) for b in bins[1:]) + f"  {'mid_total':>10}")
    for i, nm in enumerate(names):
        a = np.abs(actions[:, i])
        mid_mask = (a >= zero_thr) & (a < sat_thr)
        mid_total = int(mid_mask.sum())
        if mid_total == 0:
            print(f"{nm:<12}  " + "  ".join("       0%" for _ in bins[1:]) + f"  {mid_total:>10d}")
            continue
        cum = []
        for b in bins[1:]:
            cum.append(100.0 * float(((a >= zero_thr) & (a < b)).sum()) / mid_total)
        print(f"{nm:<12}  " + "  ".join(f"{v:>7.2f}%" for v in cum) + f"  {mid_total:>10d}")
    print()

    # 3. Step-to-step deltas WITHIN episode.  If smoothing was effective the
    #    deltas should be small and continuous; if actions are still discrete
    #    bangs, deltas will cluster at +/- 0.1 (transition into / out of saturation).
    print("=== Step-to-step delta distribution (intra-episode) ===")
    deltas = []
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        if len(idx) < 2:
            continue
        ep_a = actions[idx]
        d = np.diff(ep_a, axis=0)  # (Te-1, A)
        deltas.append(d)
    deltas = np.concatenate(deltas, axis=0)  # (sum, A)
    print(f"{'dim':<12}  {'%|d|<1e-4':>10}  {'%|d|>0.05':>10}  {'%|d|>0.09':>10}  {'max|d|':>9}  {'std(d)':>9}")
    for i, nm in enumerate(names):
        d = deltas[:, i]
        ad = np.abs(d)
        p_zero = 100.0 * float((ad < 1e-4).mean())
        p_big = 100.0 * float((ad > 0.05).mean())
        p_huge = 100.0 * float((ad > 0.09).mean())
        print(f"{nm:<12}  {p_zero:>9.2f}%  {p_big:>9.2f}%  {p_huge:>9.2f}%  {float(ad.max()):>9.5f}  {float(d.std()):>9.5f}")
    print()

    # 4. Per-episode summary: does each episode contain mid-range action values
    #    at all, or is it bang-bang only?  If most episodes have <1% mid-range
    #    samples, "smoothing" did nothing useful for the policy.
    print("=== Per-episode mid-range coverage (over ALL 6 dims combined) ===")
    eps = np.unique(ep)
    pct_mid = []
    pct_idle = []
    for e in eps:
        idx = np.nonzero(ep == e)[0]
        a = actions[idx]
        a_mid = (np.abs(a) >= zero_thr) & (np.abs(a) < sat_thr)
        pct_mid.append(100.0 * a_mid.any(axis=1).mean())  # per-step "any dim mid"
        pct_idle.append(100.0 * (np.linalg.norm(a, axis=1) < 1e-3).mean())
    pct_mid = np.asarray(pct_mid)
    pct_idle = np.asarray(pct_idle)
    print(f"  episodes: {len(eps)}")
    print(f"  per-ep %frames with at least one MID-RANGE action: "
          f"mean {pct_mid.mean():.2f}%  median {np.median(pct_mid):.2f}%  "
          f"min {pct_mid.min():.2f}%  max {pct_mid.max():.2f}%")
    print(f"  per-ep %IDLE frames (||a||<1e-3): "
          f"mean {pct_idle.mean():.2f}%  median {np.median(pct_idle):.2f}%")

    # 5. Overall smoothness diagnostic: ratio std(action) / std(diff(action)).
    #    For a smooth bandlimited signal at fps=20 this is >> 1.  For
    #    bang-bang teleop it's ~1 (each transition is a step of comparable
    #    size to the signal range).
    print()
    print("=== Smoothness ratio std(a) / std(d a)  (high = smooth) ===")
    print(f"{'dim':<12}  {'std(a)':>10}  {'std(da)':>10}  {'ratio':>8}")
    for i, nm in enumerate(names):
        sa = float(actions[:, i].std())
        sd = float(deltas[:, i].std())
        r = sa / max(sd, 1e-12)
        print(f"{nm:<12}  {sa:>10.5f}  {sd:>10.5f}  {r:>8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
