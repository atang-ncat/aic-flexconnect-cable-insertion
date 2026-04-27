#!/usr/bin/env python3
"""Print the last-second wrench-magnitude trace for every episode.

A seated SFP cable shows a *steady* low-N force (~3-10 N) for the
final hold.  An aborted attempt shows either ~0 N (cable in air) or a
ramping/oscillating force (operator was still pushing).  This script
prints |F| sampled every 100 ms for the last 1 s of each episode so
the difference is immediately visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    info = json.loads((root / "meta" / "info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    fps = info["fps"]

    files = sorted((root / "data").rglob("*.parquet"))
    state_chunks, ep_chunks = [], []
    for fp in files:
        tbl = pq.read_table(fp, columns=["observation.state", "episode_index"])
        s = tbl.column("observation.state").to_numpy(zero_copy_only=False)
        if s.ndim == 1:
            s = np.stack(list(s))
        state_chunks.append(s)
        ep_chunks.append(tbl.column("episode_index").to_numpy())
    state = np.concatenate(state_chunks, axis=0)
    ep = np.concatenate(ep_chunks, axis=0).astype(np.int64)

    iwf = [state_names.index(n) for n in ("wrench.force.x", "wrench.force.y", "wrench.force.z")]
    iz = state_names.index("tcp_pose.position.z")

    print(f"Last 2.0 s of each episode, sampled every 0.1 s")
    print(f"  cols: t=seconds before episode end, |F|=N, dz_mm=mm above per-ep z-min")
    print()

    for e in np.unique(ep):
        m = ep == e
        st = state[m]
        f = np.linalg.norm(st[:, iwf], axis=1)
        z = st[:, iz]
        z_min = z.min()
        T = len(st)
        # Sample every 3 frames (= 0.1 s @ 30 fps) over the last 60 frames
        N = min(int(2.0 * fps), T)
        idx = np.arange(T - N, T, max(1, fps // 10))
        line = f"ep{int(e)} (T={T:>4}):  "
        cells = []
        for i in idx:
            t_before_end = (T - 1 - i) / fps
            cells.append(f"t-{t_before_end:>3.1f}s |F|={f[i]:>5.1f}N dz={1000*(z[i]-z_min):>5.1f}mm")
        print(line + "  ".join(cells))

    print()
    print("Look for: a steady ~3-10 N hold over the last 1 s = inserted.")
    print("           ramping / changing |F| or |F| < 1 N at end           = NOT inserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
