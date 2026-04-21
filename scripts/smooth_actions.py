#!/usr/bin/env python3
"""Apply Savitzky-Golay smoothing to action data in a LeRobot dataset.

This script reads the raw keyboard-teleoperated action data, applies a
Savitzky-Golay filter per-episode and per-axis, and writes the result
to a new dataset directory. The original dataset is never modified.

The filter removes the step-function velocity profiles inherent to keyboard
teleoperation while preserving intentional direction changes and recovery
corrections.

What gets smoothed:
    * ``action`` columns only (6D Cartesian twist: linear xyz, angular xyz)

What is NOT touched:
    * ``observation.state`` — proprioceptive data comes from the sim, not keyboard
    * Videos / images — obviously
    * Metadata (info.json, tasks.parquet, episodes/) — copied as-is
    * All index columns (frame_index, episode_index, index, task_index, timestamp)

Usage:
    cd /scratch2/atang/ws_aic/src/aic && pixi run python \\
        /scratch2/atang/ws_aic/scripts/smooth_actions.py \\
        --input  /scratch2/atang/ws_aic/teleop-dataset \\
        --output /scratch2/atang/ws_aic/teleop-dataset-smoothed \\
        --window 9 --polyorder 3

    # Preview mode (prints stats, writes nothing):
    ... --preview

    # Override window per-axis (advanced):
    ... --window 9 --window-angular 7
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter


# ──────────────────────────────────────────────────────────────────────
# Smoothing core
# ──────────────────────────────────────────────────────────────────────


def smooth_episode_actions(
    actions: np.ndarray,
    window: int = 9,
    polyorder: int = 3,
    window_angular: int | None = None,
) -> np.ndarray:
    """Smooth a (T, 6) array of velocity actions per-axis with Savitzky-Golay.

    Args:
        actions: (T, 6) array — [linear.x, linear.y, linear.z,
                                   angular.x, angular.y, angular.z].
        window: Filter window length for linear axes (must be odd, >= polyorder+2).
        polyorder: Polynomial order for the filter.
        window_angular: Optional separate window for angular axes (indices 3-5).
                        Defaults to ``window`` if not provided.

    Returns:
        Smoothed (T, 6) array.
    """
    T = actions.shape[0]
    if window_angular is None:
        window_angular = window

    smoothed = np.copy(actions)

    for axis in range(6):
        w = window_angular if axis >= 3 else window

        # Savitzky-Golay requires window <= T and window must be odd.
        # For very short episodes, fall back to a smaller window.
        effective_w = min(w, T)
        if effective_w % 2 == 0:
            effective_w -= 1
        if effective_w < polyorder + 2:
            # Episode too short to filter — leave raw.
            continue

        smoothed[:, axis] = savgol_filter(actions[:, axis], effective_w, polyorder)

    return smoothed


# ──────────────────────────────────────────────────────────────────────
# Dataset I/O
# ──────────────────────────────────────────────────────────────────────

ACTION_COL = "action"
EPISODE_COL = "episode_index"


def process_parquet_file(
    src_path: Path,
    dst_path: Path,
    window: int,
    polyorder: int,
    window_angular: int | None,
    preview: bool,
) -> dict:
    """Read one parquet file, smooth actions per-episode, and write the result.

    Returns a dict of per-episode statistics for reporting.
    """
    table = pq.read_table(src_path)
    actions_col = table.column(ACTION_COL)
    episode_col = table.column(EPISODE_COL).to_pylist()

    # Convert action column to numpy: (N, 6)
    actions_np = np.array([row.as_py() for row in actions_col], dtype=np.float32)

    # Group rows by episode
    episodes_in_file: dict[int, list[int]] = {}
    for row_idx, ep in enumerate(episode_col):
        episodes_in_file.setdefault(ep, []).append(row_idx)

    stats = {}
    smoothed_actions = np.copy(actions_np)

    for ep_idx, row_indices in sorted(episodes_in_file.items()):
        rows = sorted(row_indices)
        ep_actions = actions_np[rows]  # (T_ep, 6)

        ep_smoothed = smooth_episode_actions(
            ep_actions, window=window, polyorder=polyorder,
            window_angular=window_angular,
        )
        smoothed_actions[rows] = ep_smoothed

        # Compute before/after statistics
        raw_jerk = np.diff(ep_actions, axis=0)
        smooth_jerk = np.diff(ep_smoothed, axis=0)
        stats[ep_idx] = {
            "n_frames": len(rows),
            "raw_jerk_std": float(np.std(raw_jerk)),
            "smooth_jerk_std": float(np.std(smooth_jerk)),
            "max_abs_change": float(np.max(np.abs(ep_smoothed - ep_actions))),
            "mean_abs_change": float(np.mean(np.abs(ep_smoothed - ep_actions))),
        }

    if preview:
        return stats

    # Rebuild the action column as a fixed-size list
    smoothed_list = smoothed_actions.tolist()
    new_action_col = pa.FixedSizeListArray.from_arrays(
        pa.array([v for row in smoothed_list for v in row], type=pa.float32()),
        list_size=6,
    )

    # Replace column in table
    col_idx = table.column_names.index(ACTION_COL)
    new_table = table.set_column(col_idx, table.schema.field(ACTION_COL), new_action_col)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)

    return stats


def recompute_stats(output_dir: Path) -> None:
    """Recompute min/max/mean/std/quantile stats for the smoothed dataset.

    This updates ``meta/stats.json`` so that normalization layers in
    the policy see the correct statistics for the smoothed actions.
    """
    stats_path = output_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    # Gather all action data from the smoothed parquet files
    data_dir = output_dir / "data"
    all_actions = []
    for pf in sorted(data_dir.rglob("*.parquet")):
        t = pq.read_table(pf, columns=[ACTION_COL])
        actions = np.array([row.as_py() for row in t.column(ACTION_COL)], dtype=np.float32)
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 6)

    # Recompute stats for the action feature
    stats["action"] = {
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "count": [len(all_actions)],
        "q01": np.quantile(all_actions, 0.01, axis=0).tolist(),
        "q10": np.quantile(all_actions, 0.10, axis=0).tolist(),
        "q50": np.quantile(all_actions, 0.50, axis=0).tolist(),
        "q90": np.quantile(all_actions, 0.90, axis=0).tolist(),
        "q99": np.quantile(all_actions, 0.99, axis=0).tolist(),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[✓] Recomputed action stats in {stats_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Savitzky-Golay smoothing to LeRobot dataset actions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the original dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the smoothed dataset.")
    parser.add_argument("--window", type=int, default=9,
                        help="Savitzky-Golay window length (must be odd). Default: 9.")
    parser.add_argument("--polyorder", type=int, default=3,
                        help="Polynomial order for the filter. Default: 3.")
    parser.add_argument("--window-angular", type=int, default=None,
                        help="Separate window for angular axes (3-5). Defaults to --window.")
    parser.add_argument("--preview", action="store_true",
                        help="Print before/after statistics without writing any files.")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")
    if output_dir.exists() and not args.preview:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Remove it first or choose a different path to avoid accidentally overwriting data."
        )

    # Validate window parameters
    if args.window % 2 == 0:
        raise ValueError(f"--window must be odd, got {args.window}")
    if args.window_angular is not None and args.window_angular % 2 == 0:
        raise ValueError(f"--window-angular must be odd, got {args.window_angular}")

    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Window:      {args.window} (linear), {args.window_angular or args.window} (angular)")
    print(f"Polyorder:   {args.polyorder}")
    print(f"Mode:        {'PREVIEW (no files written)' if args.preview else 'WRITE'}")
    print()

    # ── Copy everything except data/ first ──
    if not args.preview:
        # Copy meta/, videos/ and any other top-level files
        for item in input_dir.iterdir():
            dst = output_dir / item.name
            if item.name == "data":
                continue  # we'll write new parquet files
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
        print(f"[✓] Copied metadata and videos to {output_dir}")

    # ── Process parquet files ──
    data_dir = input_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"[…] Processing {len(parquet_files)} parquet files\n")

    all_stats: dict[int, dict] = {}

    for pf in parquet_files:
        rel = pf.relative_to(input_dir)
        dst = output_dir / rel
        file_stats = process_parquet_file(
            src_path=pf,
            dst_path=dst,
            window=args.window,
            polyorder=args.polyorder,
            window_angular=args.window_angular,
            preview=args.preview,
        )
        all_stats.update(file_stats)

    # ── Report ──
    print(f"{'Episode':>8} {'Frames':>7} {'Raw Jerk σ':>12} {'Smooth Jerk σ':>14} {'Reduction':>10} {'Max Δ':>10} {'Mean Δ':>10}")
    print("-" * 83)

    total_raw = 0.0
    total_smooth = 0.0
    for ep in sorted(all_stats):
        s = all_stats[ep]
        reduction = 1.0 - s["smooth_jerk_std"] / max(s["raw_jerk_std"], 1e-12)
        total_raw += s["raw_jerk_std"]
        total_smooth += s["smooth_jerk_std"]
        print(
            f"{ep:>8d} {s['n_frames']:>7d} "
            f"{s['raw_jerk_std']:>12.6f} {s['smooth_jerk_std']:>14.6f} "
            f"{reduction:>9.1%} {s['max_abs_change']:>10.6f} {s['mean_abs_change']:>10.6f}"
        )

    n = len(all_stats)
    avg_reduction = 1.0 - total_smooth / max(total_raw, 1e-12)
    print("-" * 83)
    print(f"{'AVG':>8s} {'':>7s} {total_raw/n:>12.6f} {total_smooth/n:>14.6f} {avg_reduction:>9.1%}")
    print()

    if not args.preview:
        # Recompute normalization stats for the smoothed actions
        recompute_stats(output_dir)
        print(f"\n[✓] Smoothed dataset written to: {output_dir}")
        print(f"    Update your training config to point at this directory:")
        print(f"      dataset:")
        print(f"        root: {output_dir.relative_to(input_dir.parent)}")
    else:
        print("[ℹ] Preview mode — no files were written.")


if __name__ == "__main__":
    main()
