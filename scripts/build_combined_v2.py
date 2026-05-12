#!/usr/bin/env python3
"""Build a unified, task-tagged training dataset for v18 (May-11 recovery plan).

Combines three existing collected datasets into a single LeRobot v3.0
dataset at ``datasets/teleop-dataset-combined-v2/`` with a per-episode
``task_index`` column so the policy can condition on SFP vs SC.

Sources
-------
* ``datasets/teleop-dataset-gamepad/sfp``     — 52 episodes (gamepad v1 SFP)
* ``datasets/teleop-dataset-gamepad-v2/sfp``  — gamepad v2 SFP
* ``datasets/teleop-dataset-gamepad/sc``      — 60 episodes (gamepad v1 SC)
* ``datasets/teleop-dataset-gamepad-v2/sc``   — gamepad v2 SC (optional; skipped if not present)

All sources are 30 fps with the same 39-D ``observation.state`` schema and
6-D twist actions, so they concatenate without resampling.  Sources whose
directory does not exist yet (typically ``.../gamepad-v2/sc`` before the first
SC recording session) are skipped with a NOTE.

All three are 30 fps with the same 39-D ``observation.state`` schema and
6-D twist actions, so we can concatenate without resampling.  The merge
preserves the original .mp4 video files via symlink (no re-encode).

Task tagging
------------
Each episode gets one of two task strings:
* ``insert_sfp`` (task_index = 0)   for both gamepad-v1 sfp and gamepad-v2 sfp
* ``insert_sc``  (task_index = 1)   for gamepad-v1 sc and gamepad-v2 sc

The training script reads ``task_index`` per-batch and appends a 2-D
one-hot to ``observation.state`` -- see ``configs/act_sfp_v18.yaml`` and
the ``task_inject`` block in ``train_act.py``.

Failure filtering
-----------------
With ``--drop-failed``, each episode is heuristically scored using the
same logic as ``scripts/probe_episode_outcomes.py`` (z-settle near
per-episode min, sustained end-of-episode wrench).  Episodes flagged
``NO_SEAT`` or ``NO_DESCENT`` are excluded.  We emit the dropped list to
``<out>/dropped_episodes.json`` so you can review which ones were cut.

Usage
-----
    cd /scratch2/atang/ws_aic/src/aic
    pixi run python /scratch2/atang/ws_aic/scripts/build_combined_v2.py \\
        --out /scratch2/atang/ws_aic/datasets/teleop-dataset-combined-v2 \\
        --drop-failed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = [
    "observation.images.left_camera",
    "observation.images.center_camera",
    "observation.images.right_camera",
]

DEFAULT_TASKS = [
    "Insert SFP connector into SFP port on NIC card",  # task_index 0
    "Insert SC connector into SC port",                # task_index 1
]


# Resolve the workspace root from this script's own location so the merge
# works whether you're on the host (/scratch2/atang/ws_aic) or inside the
# AIC docker container (/run/host/scratch2/atang/ws_aic, where /scratch2 is
# bind-mounted under /run/host/).  ``scripts/build_combined_v2.py`` lives at
# ``<workspace>/scripts/`` so the workspace is one level up.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"


def _resolve_dataset_path(rel: str) -> Path:
    """Map a workspace-relative path like ``datasets/foo/bar`` to a real
    on-disk path, transparently picking up the ``/run/host`` prefix when
    we're running inside the AIC eval container.

    Tries, in order:
      1. ``WORKSPACE_ROOT / rel``
      2. ``Path("/run/host") / WORKSPACE_ROOT / rel``  (container fallback)
    Returns the first one that exists; otherwise returns option 1 so the
    caller's "missing source" error message points at the host-style path.
    """
    p1 = DATASETS_ROOT / rel
    if p1.is_dir():
        return p1
    # If we're already under /run/host, also try going *up* by stripping
    # the /run/host prefix, in case someone passes a /scratch2/... path.
    p2 = Path("/run/host") / DATASETS_ROOT.relative_to("/") / rel
    if p2.is_dir():
        return p2
    return p1


@dataclass(frozen=True)
class Source:
    label: str
    root: Path
    task_index: int           # 0 = sfp, 1 = sc
    expected_episodes: int    # sanity check; set to -1 to skip


SOURCES: list[Source] = [
    Source(
        label="gamepad-v1-sfp",
        root=_resolve_dataset_path("teleop-dataset-gamepad/sfp"),
        task_index=0,
        expected_episodes=52,
    ),
    Source(
        label="gamepad-v2-sfp",
        root=_resolve_dataset_path("teleop-dataset-gamepad-v2/sfp"),
        task_index=0,
        expected_episodes=71,
    ),
    Source(
        label="gamepad-v1-sc",
        root=_resolve_dataset_path("teleop-dataset-gamepad/sc"),
        task_index=1,
        expected_episodes=60,
    ),
    Source(
        label="gamepad-v2-sc",
        root=_resolve_dataset_path("teleop-dataset-gamepad-v2/sc"),
        task_index=1,
        expected_episodes=-1,
    ),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_data_parquet(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquets under {root}")
    dfs = [pq.read_table(p).to_pandas() for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _load_episodes_parquet(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episodes parquets under {root}")
    dfs = [pq.read_table(p).to_pandas() for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


# ---------------------------------------------------------------------------
# Failure heuristic — mirrors scripts/probe_episode_outcomes.py
# ---------------------------------------------------------------------------


def _state_idx(state_names: list[str], name: str) -> int | None:
    return state_names.index(name) if name in state_names else None


def flag_failed_episodes(
    df: pd.DataFrame,
    state_names: list[str],
    fps: int,
) -> set[int]:
    """Return the set of episode indices (in this source's local space) that
    look like failed insertions.

    Heuristic (NO_SEAT or NO_DESCENT):
      * NO_SEAT: gripper finishes >=2 mm above the per-episode min z AND
        end-of-episode mean force <0.5 N.  A successful insertion settles
        at the descent floor with sustained reaction force.
      * NO_DESCENT: total z range during the episode <5 mm.  Probably the
        operator aborted before the approach phase.
    """
    iz = _state_idx(state_names, "tcp_pose.position.z")
    iwx = _state_idx(state_names, "wrench.force.x")
    iwy = _state_idx(state_names, "wrench.force.y")
    iwz = _state_idx(state_names, "wrench.force.z")
    if iz is None:
        return set()  # nothing to do without z signal
    has_wrench = None not in (iwx, iwy, iwz)

    state = np.stack(df["observation.state"].values)
    eps = df["episode_index"].to_numpy()
    fi = df["frame_index"].to_numpy()
    bad: set[int] = set()
    for e in np.unique(eps):
        mask = eps == e
        order = np.argsort(fi[mask])
        z = state[mask, iz][order]
        last_n = min(int(fps), len(z))
        z_descent_mm = float((z.max() - z.min()) * 1000.0)
        z_settle_above_min_mm = float((z[-last_n:].mean() - z.min()) * 1000.0)

        if has_wrench:
            f_mag = np.linalg.norm(state[mask][:, [iwx, iwy, iwz]], axis=1)
            f_end = float(f_mag[order][-last_n:].mean())
        else:
            f_end = float("nan")

        no_seat = z_settle_above_min_mm > 2.0 and (
            np.isnan(f_end) or f_end < 0.5
        )
        no_descent = z_descent_mm < 5.0
        if no_seat or no_descent:
            bad.add(int(e))
    return bad


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output dataset root (must not already exist).",
    )
    ap.add_argument(
        "--drop-failed",
        action="store_true",
        help="Drop episodes flagged NO_SEAT or NO_DESCENT by the heuristic.",
    )
    ap.add_argument(
        "--include-task-indices",
        type=str,
        default=None,
        help="Comma-separated list of task_index values to include "
             "(e.g. '0' for SFP-only, '1' for SC-only, default = all).",
    )
    args = ap.parse_args()

    # Optionally restrict which source datasets to ingest.  Used by the
    # 'pure SFP' variant where we want only gamepad-v1/sfp + gamepad-v2/sfp
    # so the saved normalizer stats and recomputed image stats are SFP-only.
    if args.include_task_indices is not None:
        keep_set = {int(x.strip()) for x in args.include_task_indices.split(",") if x.strip()}
        sources = [s for s in SOURCES if s.task_index in keep_set]
        if not sources:
            print(
                f"ERROR: --include-task-indices={args.include_task_indices} "
                f"matched no sources",
                file=sys.stderr,
            )
            return 1
    else:
        sources = list(SOURCES)

    # Optional v2/SC tree may not exist until the first SC session — skip quietly.
    present: list[Source] = []
    skipped_labels: list[str] = []
    for s in sources:
        if not s.root.is_dir():
            skipped_labels.append(s.label)
            continue
        present.append(s)
    if skipped_labels:
        print(
            "NOTE: skipping sources (dataset root not found yet): "
            + ", ".join(skipped_labels)
        )
    sources = present
    if not sources:
        print(
            "ERROR: no source datasets found after filtering — create at least one "
            "under datasets/teleop-dataset-gamepad*/",
            file=sys.stderr,
        )
        return 1

    out: Path = args.out.resolve()
    # Container-aware --out path: if the parent directory of the requested
    # output doesn't exist (e.g. user passes /scratch2/... while inside the
    # AIC docker, where /scratch2 is mounted at /run/host/scratch2), try the
    # /run/host-prefixed variant.
    if not out.parent.is_dir():
        alt = Path("/run/host") / Path(str(out).lstrip("/"))
        if alt.parent.is_dir():
            print(f"NOTE: redirecting --out from {out} to {alt} "
                  f"(running inside container, /scratch2 is mapped under /run/host).")
            out = alt
    if out.exists():
        print(f"ERROR: output already exists: {out}", file=sys.stderr)
        print("Remove it first if you want to rebuild.", file=sys.stderr)
        return 1

    # ── Sanity-check sources ───────────────────────────────────────────
    for src in sources:
        if not src.root.is_dir():
            print(f"ERROR: missing source: {src.root}", file=sys.stderr)
            return 1
        info = _load_info(src.root)
        n = int(info["total_episodes"])
        if src.expected_episodes >= 0 and n != src.expected_episodes:
            print(
                f"WARNING: {src.label} has {n} episodes "
                f"(expected {src.expected_episodes}); continuing."
            )
        if int(info["fps"]) != 30:
            print(
                f"ERROR: {src.label} is {info['fps']} fps; expected 30. "
                "Concatenation assumes uniform fps.",
                file=sys.stderr,
            )
            return 1

    # ── Output directory layout ────────────────────────────────────────
    out.mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    for cam in CAMS:
        (out / "videos" / cam / "chunk-000").mkdir(parents=True)

    # ── Per-source load + failure filter ──────────────────────────────
    dropped_log: dict[str, list[int]] = {}
    per_src_data: list[tuple[Source, pd.DataFrame, pd.DataFrame, dict]] = []
    for src in sources:
        info = _load_info(src.root)
        df = _load_data_parquet(src.root)
        eps_meta = _load_episodes_parquet(src.root)
        state_names = info["features"]["observation.state"]["names"]

        bad: set[int] = set()
        if args.drop_failed:
            bad = flag_failed_episodes(df, state_names, fps=int(info["fps"]))
        dropped_log[src.label] = sorted(bad)

        if bad:
            keep_mask = ~df["episode_index"].isin(bad)
            df = df[keep_mask].reset_index(drop=True)
            eps_meta = eps_meta[~eps_meta["episode_index"].isin(bad)].reset_index(drop=True)

        n_eps = int(eps_meta["episode_index"].nunique())
        n_frames = int(len(df))
        print(
            f"[{src.label}] task={src.task_index}  "
            f"kept {n_eps} ep / {n_frames} frames "
            f"(dropped {len(bad)} ep)"
        )
        per_src_data.append((src, df, eps_meta, info))

    # ── Re-index episodes globally and stitch the data table ──────────
    all_data: list[pd.DataFrame] = []
    all_eps: list[pd.DataFrame] = []

    next_global_ep = 0
    next_global_idx = 0  # frame-level "index" column
    next_global_video_file_idx: dict[str, int] = {cam: 0 for cam in CAMS}

    for src, df_src, eps_src, info_src in per_src_data:
        # Sort each episode's rows by frame_index, then re-emit a contiguous
        # frame_index 0..N-1 per episode in the OUTPUT space.
        df_src = df_src.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        # Build local→global episode mapping (kept episodes only, in sorted order).
        kept_local_eps = sorted(df_src["episode_index"].unique())
        local_to_global = {
            int(e): next_global_ep + i for i, e in enumerate(kept_local_eps)
        }

        # ── DATA parquet rows ──────────────────────────────────────────
        new_ep = df_src["episode_index"].map(local_to_global).astype(np.int64)
        # Re-emit frame_index per episode as 0..n-1; recompute the global
        # "index" column as a simple monotonic counter across the merged
        # dataset.
        new_fi = np.empty(len(df_src), dtype=np.int64)
        new_index = np.empty(len(df_src), dtype=np.int64)
        new_ts = np.empty(len(df_src), dtype=np.float32)

        cursor = 0
        for local_e in kept_local_eps:
            mask = (df_src["episode_index"] == local_e).values
            n = int(mask.sum())
            new_fi[mask] = np.arange(n, dtype=np.int64)
            new_index[mask] = next_global_idx + np.arange(n, dtype=np.int64)
            new_ts[mask] = (np.arange(n, dtype=np.float32) / 30.0).astype(np.float32)
            next_global_idx += n
            cursor += n

        df_out = df_src.copy()
        df_out["episode_index"] = new_ep
        df_out["frame_index"] = new_fi
        df_out["index"] = new_index
        df_out["timestamp"] = new_ts
        df_out["task_index"] = np.int64(src.task_index)
        all_data.append(df_out)

        # ── EPISODES metadata rows ────────────────────────────────────
        eps_out = eps_src.copy()
        eps_out = eps_out[eps_out["episode_index"].isin(local_to_global.keys())]
        eps_out = eps_out.sort_values("episode_index").reset_index(drop=True)
        eps_out["episode_index"] = eps_out["episode_index"].map(local_to_global).astype(np.int64)

        # Recompute lengths + dataset_from/to_index for the merged frame-space.
        # We use the freshly emitted data table for ground truth.
        ep_lengths = (
            df_out.groupby("episode_index").size().reindex(eps_out["episode_index"]).values
        )
        eps_out["length"] = ep_lengths.astype(np.int64)
        # dataset_from/to_index in MERGED frame-index space (equal to
        # df_out["index"] min/max+1 per episode).
        from_idx = (
            df_out.groupby("episode_index")["index"].min().reindex(eps_out["episode_index"]).values
        )
        to_idx = from_idx + ep_lengths
        eps_out["dataset_from_index"] = from_idx.astype(np.int64)
        eps_out["dataset_to_index"] = to_idx.astype(np.int64)
        eps_out["data/chunk_index"] = np.int64(0)
        eps_out["data/file_index"] = np.int64(0)
        eps_out["meta/episodes/chunk_index"] = np.int64(0)
        eps_out["meta/episodes/file_index"] = np.int64(0)

        # Tag every episode with the right task string and (single-row) task_index.
        task_str = DEFAULT_TASKS[src.task_index]
        eps_out["tasks"] = [[task_str] for _ in range(len(eps_out))]

        # ── Symlink videos with re-indexed names ───────────────────────
        # Each source has chunk-000/file-XXX.mp4 (one episode per file).
        # Re-emit them as file-{global_file_idx}.mp4 in OUT.
        for cam in CAMS:
            cam_src_dir = src.root / "videos" / cam / "chunk-000"
            if not cam_src_dir.is_dir():
                raise FileNotFoundError(f"missing video dir: {cam_src_dir}")
            ordered_files = sorted(
                cam_src_dir.glob("*.mp4"),
                key=lambda p: int(p.stem.split("-")[1]),
            )
            new_video_file_idx_for_local: dict[int, int] = {}
            for local_e in kept_local_eps:
                # The source's video file index for episode i is normally
                # equal to i (one episode per file).  We assert that and
                # fall back to looking it up in eps_src if not.
                src_file_idx_col = f"videos/{cam}/file_index"
                if src_file_idx_col in eps_src.columns:
                    sub = eps_src[eps_src["episode_index"] == local_e]
                    if len(sub) >= 1:
                        local_file_idx = int(sub.iloc[0][src_file_idx_col])
                    else:
                        local_file_idx = local_e
                else:
                    local_file_idx = local_e
                src_path = cam_src_dir / f"file-{local_file_idx:03d}.mp4"
                if not src_path.is_file():
                    raise FileNotFoundError(f"missing video: {src_path}")
                new_idx = next_global_video_file_idx[cam]
                next_global_video_file_idx[cam] += 1
                new_video_file_idx_for_local[local_e] = new_idx
                dst = out / "videos" / cam / "chunk-000" / f"file-{new_idx:03d}.mp4"
                os.symlink(src_path.resolve(), dst)

            # Rewrite the per-episode video file_index in eps_out.
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col not in eps_out.columns:
                eps_out[chunk_col] = np.int64(0)
            else:
                eps_out[chunk_col] = np.int64(0)
            # Map each output-space episode index back to its local source
            # episode index, then to the new file index.
            global_to_local = {g: l for l, g in local_to_global.items()}
            new_file_indices = [
                new_video_file_idx_for_local[global_to_local[int(g)]]
                for g in eps_out["episode_index"].values
            ]
            eps_out[file_col] = np.asarray(new_file_indices, dtype=np.int64)

            # Per-episode video timestamps: 0 .. length/30 (each video is one episode).
            from_ts_col = f"videos/{cam}/from_timestamp"
            to_ts_col = f"videos/{cam}/to_timestamp"
            durations = eps_out["length"].values / 30.0
            eps_out[from_ts_col] = np.zeros(len(eps_out), dtype=np.float64)
            eps_out[to_ts_col] = durations.astype(np.float64)

        all_eps.append(eps_out)
        next_global_ep += len(kept_local_eps)

    # ── Concatenate ────────────────────────────────────────────────────
    df_final = pd.concat(all_data, ignore_index=True)
    eps_final = pd.concat(all_eps, ignore_index=True)
    print(f"\nCombined: {len(eps_final)} episodes, {len(df_final)} frames.")

    # Re-index eps_final to be 0..N-1 in case ordering changed.
    eps_final = eps_final.sort_values("episode_index").reset_index(drop=True)

    # ── Write data parquet ────────────────────────────────────────────
    pq.write_table(
        pa.Table.from_pandas(df_final, preserve_index=False),
        out / "data" / "chunk-000" / "file-000.parquet",
    )

    # ── Write episodes parquet ────────────────────────────────────────
    # Drop any old per-episode stats columns; they'll be recomputed downstream.
    eps_keep_cols = [c for c in eps_final.columns if not c.startswith("stats/")]
    eps_final_min = eps_final[eps_keep_cols].copy()
    pq.write_table(
        pa.Table.from_pandas(eps_final_min, preserve_index=False),
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    # ── Write tasks.parquet ───────────────────────────────────────────
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(DEFAULT_TASKS)))},
        index=DEFAULT_TASKS,
    )
    pq.write_table(
        pa.Table.from_pandas(tasks_df, preserve_index=True),
        out / "meta" / "tasks.parquet",
    )

    # ── Write stats.json (action + observation.state only) ────────────
    actions = np.stack(df_final["action"].values).astype(np.float64)
    states = np.stack(df_final["observation.state"].values).astype(np.float64)
    stats: dict = {}
    for feat_name, arr in [("action", actions), ("observation.state", states)]:
        stats[feat_name] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).clip(min=1e-8).tolist(),
            "count": [int(len(arr))],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    # Borrow image stats from the largest source so the file is complete.
    # (LeRobot will recompute these if needed; we just need them present.)
    biggest_info = max(
        (_load_info(s.root) for s in sources),
        key=lambda i: int(i["total_frames"]),
    )
    biggest_root = next(
        s.root for s in sources
        if int(_load_info(s.root)["total_frames"]) == int(biggest_info["total_frames"])
    )
    src_stats_path = biggest_root / "meta" / "stats.json"
    if src_stats_path.is_file():
        src_stats = json.loads(src_stats_path.read_text())
        for cam in CAMS:
            if cam in src_stats:
                stats[cam] = src_stats[cam]
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    # ── Write info.json ────────────────────────────────────────────────
    info_template = json.loads(
        (sources[0].root / "meta" / "info.json").read_text()
    )
    info_out = dict(info_template)
    info_out["total_episodes"] = int(len(eps_final))
    info_out["total_frames"] = int(len(df_final))
    info_out["total_tasks"] = len(DEFAULT_TASKS)
    info_out["fps"] = 30
    info_out["splits"] = {"train": f"0:{len(eps_final)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))

    # ── Write dropped-episode log so failures are auditable ───────────
    (out / "dropped_episodes.json").write_text(json.dumps(dropped_log, indent=2))

    # ── Final report ───────────────────────────────────────────────────
    n_sfp = int((eps_final["tasks"].apply(lambda xs: xs[0]) == DEFAULT_TASKS[0]).sum())
    n_sc = int(len(eps_final) - n_sfp)
    print(
        f"\nDONE  out={out}\n"
        f"   episodes : {len(eps_final)}  ({n_sfp} sfp / {n_sc} sc)\n"
        f"   frames   : {len(df_final)}\n"
        f"   fps      : 30\n"
        f"   tasks    : {DEFAULT_TASKS}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
