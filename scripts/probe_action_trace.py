#!/usr/bin/env python3
"""Print a compact ASCII trace of one episode's action sequence.

Helps eyeball whether actions ramp smoothly or bang between zero and ±0.1.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--axis", type=int, default=2, help="0=lin.x ... 5=ang.z")
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()

    info = json.load(open(args.root / "meta" / "info.json"))
    names = info["features"]["action"]["names"]
    files = sorted((args.root / "data").rglob("*.parquet"))

    ep_actions = None
    for f in files:
        t = pq.read_table(f, columns=["action", "episode_index"], use_threads=True)
        ep = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
        mask = ep == args.episode
        if not mask.any():
            continue
        a_col = t["action"]
        rows = []
        for i in range(t.num_rows):
            if mask[i]:
                rows.append(np.asarray(a_col[i].as_py(), dtype=np.float64))
        ep_actions = np.stack(rows, axis=0)
        break
    if ep_actions is None:
        print(f"Episode {args.episode} not found.")
        return 1

    a = ep_actions[: args.max_frames, args.axis]
    print(f"{args.root.name} ep={args.episode}  axis={names[args.axis]}  T={len(a)}")
    print(f"  range [{a.min():+.4f}, {a.max():+.4f}]  std {a.std():.4f}  "
          f"%saturated {100*((np.abs(a)>0.095)).mean():.1f}%  "
          f"%zero {100*((np.abs(a)<1e-3)).mean():.1f}%")
    print()

    # Map [-0.1, 0.1] -> [0, width-1].
    rng = 0.1
    w = args.width
    half = w // 2
    print(" " * half + "|")
    print(" " * half + "0" + (" " * (half - 5)) + f"({names[args.axis]})")
    chars = "._-=+*#"
    for v in a:
        col = int(round((v + rng) / (2 * rng) * (w - 1)))
        col = max(0, min(w - 1, col))
        line = list("." * w)
        line[half] = "|"
        # Fill from center to col with a track character.
        if col < half:
            for c in range(col, half):
                line[c] = "-"
        elif col > half:
            for c in range(half + 1, col + 1):
                line[c] = "-"
        line[col] = "*" if abs(v) > 0.095 else "+"
        print("".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
