#!/usr/bin/env python3
"""Quick audit of the teleoperated SFP-insertion dataset.

Read-only, no cleaning.  Answers:

    1. How long are the episodes?  Any obvious outliers (too short = aborted;
       too long = operator stalled)?
    2. What does the 6-D action distribution look like per-dimension?
    3. How many frames are essentially "idle" (||action|| below threshold)?
       Where do they cluster -- at episode boundaries (pre-roll / post-roll)
       or mid-episode (operator hesitation)?
    4. Are there action outliers (spikes much larger than the typical value)
       that would indicate teleop glitches / dropped frames?

Runs on pure parquet metadata, no video decoding, takes < 10 seconds.
Prints a human-readable report and a single-line JSON summary (tagged
``AUDIT_SUMMARY``) that's easy to grep out of logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_data_files(root: Path) -> list[Path]:
    # LeRobot v3.0 layout: <root>/data/chunk-XXX/file-YYY.parquet (one file
    # per chunk, many episodes per file).  v2 layout:
    # <root>/data/chunk-XXX/episode-YYYYYY.parquet (one file per episode).
    candidates = sorted((root / "data").rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files under {root/'data'}")
    return candidates


def _histogram_text(values: np.ndarray, bins: int = 20, width: int = 40) -> str:
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts.max(), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / max_count)
        lines.append(f"  [{edges[i]:8.2f} .. {edges[i+1]:8.2f}]  {c:6d}  {bar}")
    return "\n".join(lines)


def audit(root: Path, idle_threshold: float, outlier_z: float) -> dict:
    print(f"=== Auditing teleop dataset at: {root} ===\n")

    # --- Episode-level stats from the episodes parquet(s) ---
    # In LeRobot v3 the episodes metadata is split across multiple files
    # (meta/episodes/chunk-XXX/file-YYY.parquet), each covering a few
    # episodes, so we must glob+concat rather than reading just the first.
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(
            f"Could not find episodes parquet under {root/'meta'/'episodes'}"
        )
    # Only pull the columns we need -- stats fields in v3 can explode the
    # parquet to hundreds of columns we don't care about here.
    ep_cols = ["episode_index", "length"]
    try:
        ep_df = pd.concat(
            [pd.read_parquet(f, columns=ep_cols) for f in ep_files],
            ignore_index=True,
        )
    except Exception:
        ep_df = pd.concat(
            [pd.read_parquet(f) for f in ep_files],
            ignore_index=True,
        )
    # Common column names across LeRobot versions: "length" or "frames_length".
    len_col = "length" if "length" in ep_df.columns else ("frames_length" if "frames_length" in ep_df.columns else None)
    if len_col is None:
        raise KeyError(f"No length column in episodes parquet; saw {list(ep_df.columns)}")

    lengths = ep_df[len_col].to_numpy()
    print(f"[EPISODES] n={len(lengths)}  total_frames={int(lengths.sum())}")
    print(f"  length  min / p5 / p50 / p95 / max  =  "
          f"{lengths.min()} / {np.percentile(lengths, 5):.0f} / "
          f"{np.percentile(lengths, 50):.0f} / {np.percentile(lengths, 95):.0f} / "
          f"{lengths.max()}")
    print(f"  length  mean={lengths.mean():.1f}  std={lengths.std():.1f}")
    print("  length histogram:")
    print(_histogram_text(lengths, bins=12, width=40))
    # Flag obvious outliers: <p10 or >p90 relative to median.
    short_cutoff = 0.5 * np.percentile(lengths, 50)
    long_cutoff = 2.0 * np.percentile(lengths, 50)
    n_short = int((lengths < short_cutoff).sum())
    n_long = int((lengths > long_cutoff).sum())
    print(f"  episodes shorter than 0.5x median  ({short_cutoff:.0f} frames) : {n_short}")
    print(f"  episodes longer  than 2.0x median  ({long_cutoff:.0f} frames) : {n_long}")

    # --- Frame-level stats from data parquets ---
    data_files = _find_data_files(root)
    print(f"\n[FRAMES] reading {len(data_files)} parquet file(s)...")
    dfs = []
    cols_needed = ["action", "episode_index", "frame_index"]
    for pf in data_files:
        try:
            dfs.append(pd.read_parquet(pf, columns=cols_needed))
        except Exception:
            dfs.append(pd.read_parquet(pf))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  loaded {len(df)} rows")

    action_mat = np.stack(df["action"].to_numpy())  # (N, action_dim)
    N, A = action_mat.shape
    print(f"  action tensor shape: {action_mat.shape}")

    # Per-dim stats
    print(f"\n[ACTION per dim]  (6-D Cartesian twist: lin-xyz, ang-xyz)")
    print(f"  {'dim':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  "
          f"{'std':>10}  {'|mean|':>10}")
    for d in range(A):
        col = action_mat[:, d]
        print(f"  {d:>4}  {col.min():>10.4f}  {col.max():>10.4f}  "
              f"{col.mean():>10.4f}  {col.std():>10.4f}  {np.abs(col).mean():>10.4f}")

    # Idle frames: ||action|| below threshold (in the raw action units, which
    # are cm/s linear and rad/s angular for our 6-DoF Cartesian twist).
    act_norm = np.linalg.norm(action_mat, axis=1)
    n_idle = int((act_norm < idle_threshold).sum())
    pct_idle = 100.0 * n_idle / N
    print(f"\n[IDLE FRAMES]  ||action||2 < {idle_threshold:.3f}")
    print(f"  count = {n_idle} / {N}  ({pct_idle:.1f}%)")

    # Where do idle frames cluster?  Check start-of-episode vs mid vs end.
    print(f"\n[IDLE FRAME POSITIONS] (is-idle vs frame_index-within-episode)")
    # Compute relative position within each episode (0 = first frame, 1 = last).
    # Since `frame_index` resets per episode in LeRobot, we just need episode length.
    df["is_idle"] = act_norm < idle_threshold
    by_ep = df.groupby("episode_index")
    rel_positions = []
    for eid, sub in by_ep:
        L = len(sub)
        if L <= 1:
            continue
        idle_mask = sub["is_idle"].to_numpy()
        rel_positions.append(np.where(idle_mask)[0] / max(L - 1, 1))
    if rel_positions:
        rel_positions = np.concatenate(rel_positions)
        first_10pct = int((rel_positions < 0.10).sum())
        last_10pct = int((rel_positions > 0.90).sum())
        middle = int(((rel_positions >= 0.10) & (rel_positions <= 0.90)).sum())
        total_idle = len(rel_positions)
        print(f"  first 10% of each ep: {first_10pct} ({100*first_10pct/max(total_idle,1):.1f}% of idle)")
        print(f"  last  10% of each ep: {last_10pct} ({100*last_10pct/max(total_idle,1):.1f}% of idle)")
        print(f"  middle 80%         : {middle} ({100*middle/max(total_idle,1):.1f}% of idle)")

    # Outliers: frames where any action dim is > outlier_z * std(dim).
    std = action_mat.std(axis=0, keepdims=True)
    mean = action_mat.mean(axis=0, keepdims=True)
    z = np.abs((action_mat - mean) / np.maximum(std, 1e-9))
    is_outlier = (z > outlier_z).any(axis=1)
    n_outlier = int(is_outlier.sum())
    print(f"\n[OUTLIERS]  any-dim |z| > {outlier_z}")
    print(f"  count = {n_outlier} / {N}  ({100*n_outlier/N:.2f}%)")

    # Jitter proxy: per-episode std of action differences (smaller = smoother).
    print(f"\n[JITTER]  per-episode std of consecutive action deltas (smaller = smoother)")
    jitters = []
    for eid, sub in by_ep:
        a = np.stack(sub["action"].to_numpy())
        if len(a) < 2:
            continue
        jitters.append(np.linalg.norm(np.diff(a, axis=0), axis=1).std())
    jitters = np.asarray(jitters)
    print(f"  min / p50 / max = {jitters.min():.4f} / {np.percentile(jitters, 50):.4f} / "
          f"{jitters.max():.4f}")
    # Episodes with unusually high jitter
    jitter_cutoff = np.percentile(jitters, 95)
    high_jitter_count = int((jitters > jitter_cutoff).sum())
    print(f"  episodes with jitter > p95 ({jitter_cutoff:.4f}): {high_jitter_count}")

    summary = {
        "n_episodes": int(len(lengths)),
        "total_frames": int(N),
        "length_min": int(lengths.min()),
        "length_median": float(np.percentile(lengths, 50)),
        "length_max": int(lengths.max()),
        "length_std": float(lengths.std()),
        "n_short_episodes": n_short,
        "n_long_episodes": n_long,
        "action_dim": int(A),
        "action_abs_mean_per_dim": [float(x) for x in np.abs(action_mat).mean(axis=0)],
        "action_std_per_dim": [float(x) for x in action_mat.std(axis=0)],
        "idle_threshold": idle_threshold,
        "idle_frames": n_idle,
        "idle_pct": pct_idle,
        "outlier_z": outlier_z,
        "outlier_frames": n_outlier,
    }
    print("\nAUDIT_SUMMARY", json.dumps(summary))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("/scratch2/atang/ws_aic/teleop-dataset"))
    p.add_argument(
        "--idle-threshold", type=float, default=0.01,
        help="||action||2 below this counts as 'idle'. Units: action native units.",
    )
    p.add_argument("--outlier-z", type=float, default=6.0)
    args = p.parse_args()
    audit(args.root, args.idle_threshold, args.outlier_z)


if __name__ == "__main__":
    main()
