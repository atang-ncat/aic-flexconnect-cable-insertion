#!/usr/bin/env python3
"""Lightweight LeRobot v3 data audit: action idle rate, per-dim coverage, F/T.

Usage:
  python scripts/audit_teleop_dataset.py /path/to/teleop-dataset-ft-v1
  python scripts/audit_teleop_dataset.py  # default: <repo>/teleop-dataset-ft-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _load_info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    with open(p) as f:
        return json.load(f)


def _wrench_indices(info: dict) -> tuple[int, int]:
    names = info["features"]["observation.state"]["names"]
    w0 = next(i for i, n in enumerate(names) if n.startswith("wrench."))
    w1 = len(names)
    return w0, w1


def _table_to_2d_list_column(t, col: str) -> np.ndarray:
    col_data = t[col]
    n = t.num_rows
    rows: list = []
    for i in range(n):
        rows.append(np.asarray(col_data[i].as_py(), dtype=np.float64))
    return np.stack(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root (contains meta/ and data/)",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve() if args.root else repo / "teleop-dataset-ft-v1"
    if not (root / "meta" / "info.json").is_file():
        print(f"error: {root} is not a LeRobot dataset", file=sys.stderr)
        return 1

    info = _load_info(root)
    w0, w1 = _wrench_indices(info)
    A = int(np.prod(info["features"]["action"]["shape"], dtype=int))

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        print("error: no data/**/*.parquet", file=sys.stderr)
        return 1

    A_blocks: list[np.ndarray] = []
    S_blocks: list[np.ndarray] = []

    for f in files:
        t = pq.read_table(
            f, columns=["action", "observation.state"], use_threads=True
        )
        A_blocks.append(_table_to_2d_list_column(t, "action"))
        S_blocks.append(_table_to_2d_list_column(t, "observation.state"))

    actions = np.concatenate(A_blocks, axis=0)
    states = np.concatenate(S_blocks, axis=0)
    n_frames = actions.shape[0]

    norms = np.linalg.norm(actions, axis=1)
    w = states[:, w0:w1]
    w_norms = np.linalg.norm(w, axis=1)

    print(f"root: {root}")
    print(f"episodes (meta): {info['total_episodes']}  frames (meta): {info['total_frames']}")
    print(f"parquet files: {len(files)}  rows scanned: {n_frames}")
    print()
    print("action L2 norm (per frame):")
    print(f"  mean {float(norms.mean()):.6f}  std {float(norms.std()):.6f}")
    print(f"  P(||a||<1e-3) idle_strict: {100.0 * float((norms < 1e-3).mean()):.2f}%")
    print(f"  P(||a||<1e-2) idle_loose: {100.0 * float((norms < 1e-2).mean()):.2f}%")
    names = info["features"]["action"]["names"]
    abs_m = np.abs(actions)
    for i, nm in enumerate(names):
        print(f"  |{nm}| max {float(abs_m[:, i].max()):.6f}  mean_abs {float(abs_m[:, i].mean()):.6f}")
    print()
    print("wrench (observation.state last 6 fields) L2 norm:")
    print(f"  mean {float(w_norms.mean()):.4f} (mixed F/T units in vector)")
    print(f"  P(||w||>0.01): {100.0 * float((w_norms > 0.01).mean()):.2f}%")
    print(f"  P(||w||>1.0):  {100.0 * float((w_norms > 1.0).mean()):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
