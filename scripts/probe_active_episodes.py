#!/usr/bin/env python3
"""Find the most active episodes (highest fraction of non-idle frames) so we
can probe where the motion actually lives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    files = sorted((args.root / "data").rglob("*.parquet"))
    info = json.load(open(args.root / "meta" / "info.json"))
    names = info["features"]["action"]["names"]
    A = []
    E = []
    for f in files:
        t = pq.read_table(f, columns=["action", "episode_index"], use_threads=True)
        ep = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
        a = np.stack(
            [np.asarray(t["action"][i].as_py(), dtype=np.float64) for i in range(t.num_rows)]
        )
        A.append(a)
        E.append(ep)
    A = np.concatenate(A)
    E = np.concatenate(E)

    n_act = (np.linalg.norm(A, axis=1) > 1e-3).astype(np.int64)
    n_sat = (np.abs(A) > 0.095).any(axis=1).astype(np.int64)

    rows = []
    for e in np.unique(E):
        idx = np.nonzero(E == e)[0]
        T = len(idx)
        rows.append((
            int(e), T,
            float(n_act[idx].mean()) * 100,
            float(n_sat[idx].mean()) * 100,
        ))
    rows.sort(key=lambda r: -r[2])  # most-active first

    print(f"{'ep':>5}  {'T':>5}  {'%act':>7}  {'%sat':>7}")
    for r in rows[:10]:
        print(f"{r[0]:>5d}  {r[1]:>5d}  {r[2]:>6.1f}%  {r[3]:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
