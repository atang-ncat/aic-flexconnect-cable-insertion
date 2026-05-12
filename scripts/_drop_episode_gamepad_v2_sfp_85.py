#!/usr/bin/env python3
"""One-off: drop episode 85 from teleop-dataset-gamepad-v2/sfp (static / zero-action).

Patches data parquet, trims file-009 MP4s via stream-copy concat, fixes meta/episodes
rows + indices/timestamps, updates meta/info.json and recomputes meta/stats.json (numeric).

Run: python scripts/_drop_episode_gamepad_v2_sfp_85.py

Then: pixi run python scripts/recompute_image_stats.py --root datasets/teleop-dataset-gamepad-v2/sfp
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DS = Path(__file__).resolve().parents[1] / "datasets" / "teleop-dataset-gamepad-v2" / "sfp"
DATA_CHUNK = DS / "data" / "chunk-000"
VID_ROOT = DS / "videos"
EP_CHUNK = DS / "meta" / "episodes" / "chunk-000"
FFMPEG = Path(__file__).resolve().parents[1] / "src" / "aic" / ".pixi" / "envs" / "default" / "bin" / "ffmpeg"

DROP_EP = 85
N_DROP = 1438  # frames
TIME_PART1 = 1420 / 30.0
TIME_PART2_START = 2858 / 30.0
DELTA_T = N_DROP / 30.0
GLOBAL_IDX_SPLIT = 44403  # first global index after removed segment


def feat_stats_vector(a: np.ndarray) -> dict:
    """Per-episode table uses 1-D ndarrays (not nested lists) for numeric stats."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    n = int(a.shape[0])

    def q(p):
        return np.quantile(a, p, axis=0).astype(np.float64)

    out = {
        "min": a.min(axis=0).astype(np.float64),
        "max": a.max(axis=0).astype(np.float64),
        "mean": a.mean(axis=0).astype(np.float64),
        "std": a.std(axis=0).astype(np.float64),
        "count": np.array([n], dtype=np.int64),
        "q01": q(0.01).astype(np.float64),
        "q10": q(0.10).astype(np.float64),
        "q50": q(0.50).astype(np.float64),
        "q90": q(0.90).astype(np.float64),
        "q99": q(0.99).astype(np.float64),
    }
    return out


def feat_stats_episode_index(ep: int, n: int) -> dict:
    v = float(ep)
    return {
        "min": np.array([ep], dtype=np.int64),
        "max": np.array([ep], dtype=np.int64),
        "mean": np.array([v], dtype=np.float64),
        "std": np.array([0.0], dtype=np.float64),
        "count": np.array([n], dtype=np.int64),
        "q01": np.array([v], dtype=np.float64),
        "q10": np.array([v], dtype=np.float64),
        "q50": np.array([v], dtype=np.float64),
        "q90": np.array([v], dtype=np.float64),
        "q99": np.array([v], dtype=np.float64),
    }


def feat_stats_task_index(ti: np.ndarray) -> dict:
    a = ti.astype(np.float64)
    v0 = float(np.min(a))
    return {
        "min": np.array([int(np.min(a))], dtype=np.int64),
        "max": np.array([int(np.max(a))], dtype=np.int64),
        "mean": np.array([float(np.mean(a))], dtype=np.float64),
        "std": np.array([float(np.std(a))], dtype=np.float64),
        "count": np.array([int(a.size)], dtype=np.int64),
        "q01": np.array([v0], dtype=np.float64),
        "q10": np.array([v0], dtype=np.float64),
        "q50": np.array([v0], dtype=np.float64),
        "q90": np.array([v0], dtype=np.float64),
        "q99": np.array([v0], dtype=np.float64),
    }


def patch_data_file_009():
    path = DATA_CHUNK / "file-009.parquet"
    t = pq.read_table(path)
    ei = t["episode_index"].to_numpy()
    n_bad = int((ei == DROP_EP).sum())
    if n_bad == 0:
        print(f"[data] skip {path.name} (no episode {DROP_EP})")
        return
    if n_bad != N_DROP:
        raise SystemExit(
            f"{path.name}: expected episode {DROP_EP} to have {N_DROP} frames, got {n_bad}. "
            "Refusing to patch (unexpected layout)."
        )
    idx = t["index"].to_numpy()
    mask = ei != DROP_EP
    t2 = t.take(np.nonzero(mask)[0])
    ei2 = t2["episode_index"].to_numpy()
    idx2 = t2["index"].to_numpy()
    new_ei = np.where(ei2 > DROP_EP, ei2 - 1, ei2).astype(np.int64)
    new_idx = np.where(idx2 >= GLOBAL_IDX_SPLIT, idx2 - N_DROP, idx2).astype(np.int64)
    j = t2.schema.get_field_index("episode_index")
    t2 = t2.set_column(j, "episode_index", pa.array(new_ei))
    j = t2.schema.get_field_index("index")
    t2 = t2.set_column(j, "index", pa.array(new_idx))
    pq.write_table(t2, path)
    print(f"[data] wrote {path.name} rows={len(t2)}")


def patch_data_file_010():
    path = DATA_CHUNK / "file-010.parquet"
    t = pq.read_table(path)
    ei0 = int(t["episode_index"][0].as_py())
    if ei0 == 95:
        print(f"[data] skip {path.name} (already renumbered)")
        return
    if ei0 != 96:
        raise SystemExit(f"{path.name}: expected first episode_index 96 before patch, got {ei0}")
    ei = t["episode_index"].to_numpy().astype(np.int64) - 1
    idx = t["index"].to_numpy().astype(np.int64) - N_DROP
    j = t.schema.get_field_index("episode_index")
    t = t.set_column(j, "episode_index", pa.array(ei))
    j = t.schema.get_field_index("index")
    t = t.set_column(j, "index", pa.array(idx))
    pq.write_table(t, path)
    print(f"[data] wrote {path.name} rows={len(t)}")


def trim_video_file_009(cam_folder: str):
    rel = Path(cam_folder) / "chunk-000" / "file-009.mp4"
    src = VID_ROOT / rel
    if src.with_suffix(src.suffix + ".bak_ep85").exists():
        print(f"[video] skip {rel} (backup .bak_ep85 exists — already trimmed)")
        return
    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found: {FFMPEG}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p1 = td / "p1.mp4"
        p2 = td / "p2.mp4"
        lst = td / "concat.txt"
        subprocess.run(
            [str(FFMPEG), "-y", "-i", str(src), "-t", str(TIME_PART1), "-c", "copy", str(p1)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                str(FFMPEG),
                "-y",
                "-ss",
                str(TIME_PART2_START),
                "-i",
                str(src),
                "-c",
                "copy",
                str(p2),
            ],
            check=True,
            capture_output=True,
        )
        lst.write_text(f"file '{p1}'\nfile '{p2}'\n")
        out = td / "out.mp4"
        subprocess.run(
            [str(FFMPEG), "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
            check=True,
            capture_output=True,
        )
        bak = src.with_suffix(src.suffix + ".bak_ep85")
        if bak.exists():
            bak.unlink()
        src.rename(bak)
        shutil.move(str(out), str(src))
    print(f"[video] trimmed {rel}")


def apply_feature_stats(df: pd.DataFrame, row_idx: int, prefix: str, stats: dict):
    for k, v in stats.items():
        col = f"{prefix}/{k}"
        df.at[row_idx, col] = v


def recompute_row_from_fulltable(df: pd.DataFrame, row_idx: int, ep: int, full: pa.Table, copy_img_from: pd.Series):
    ti = full["episode_index"].to_numpy()
    take = np.nonzero(ti == ep)[0]
    t_ep = full.take(take)
    n = len(t_ep)
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in t_ep["action"].to_pylist()])
    ostate = np.stack([np.asarray(x, dtype=np.float32) for x in t_ep["observation.state"].to_pylist()])
    ts = t_ep["timestamp"].to_numpy().astype(np.float64)
    fi = t_ep["frame_index"].to_numpy().astype(np.int64)
    gi = t_ep["index"].to_numpy().astype(np.int64)
    tk = t_ep["task_index"].to_numpy().astype(np.int64)

    apply_feature_stats(df, row_idx, "stats/action", feat_stats_vector(actions))
    apply_feature_stats(df, row_idx, "stats/observation.state", feat_stats_vector(ostate))
    apply_feature_stats(df, row_idx, "stats/timestamp", feat_stats_vector(ts))
    st_fi = feat_stats_vector(fi.astype(np.float64))
    st_fi["min"] = np.array([int(fi.min())], dtype=np.int64)
    st_fi["max"] = np.array([int(fi.max())], dtype=np.int64)
    apply_feature_stats(df, row_idx, "stats/frame_index", st_fi)

    st_gi = feat_stats_vector(gi.astype(np.float64))
    st_gi["min"] = np.array([int(gi.min())], dtype=np.int64)
    st_gi["max"] = np.array([int(gi.max())], dtype=np.int64)
    apply_feature_stats(df, row_idx, "stats/index", st_gi)
    apply_feature_stats(df, row_idx, "stats/episode_index", feat_stats_episode_index(ep, n))
    apply_feature_stats(df, row_idx, "stats/task_index", feat_stats_task_index(tk))

    for c in df.columns:
        if c.startswith("stats/observation.images."):
            df.at[row_idx, c] = copy_img_from[c]


def _coerce_numeric_stats_columns(df: pd.DataFrame) -> None:
    """Pandas assignment can squeeze length-1 arrays to scalars; keep LeRobot ndarray shapes."""
    for c in df.columns:
        if not c.startswith("stats/"):
            continue
        if "observation.images" in c and not c.endswith("/count"):
            continue
        fixed = []
        for x in df[c]:
            if isinstance(x, (int, np.integer)) and c.endswith("/count"):
                a = np.array([int(x)], dtype=np.int64)
            elif isinstance(x, np.ndarray):
                a = x
                if a.ndim == 0:
                    v = a.item()
                    if c.endswith("/count"):
                        a = np.array([int(v)], dtype=np.int64)
                    elif (
                        ("stats/episode_index/" in c or "stats/frame_index/" in c)
                        and c.endswith(("/min", "/max"))
                    ):
                        a = np.array([int(v)], dtype=np.int64)
                    elif c.startswith("stats/index/") and c.endswith(("/min", "/max")):
                        a = np.array([int(v)], dtype=np.int64)
                    else:
                        a = np.array([float(v)], dtype=np.float64)
                elif a.ndim > 1:
                    a = a.ravel()
            else:
                a = x
            fixed.append(a)
        df[c] = fixed


def patch_episodes_file_009(full: pa.Table):
    path = EP_CHUNK / "file-009.parquet"
    orig = pq.read_table(path)
    schema = orig.schema
    df_old = orig.to_pandas()
    df = df_old[df_old["episode_index"] != DROP_EP].reset_index(drop=True)

    for i in range(len(df)):
        ep = int(df.at[i, "episode_index"])
        if ep > DROP_EP:
            df.at[i, "episode_index"] = ep - 1
            df.at[i, "dataset_from_index"] -= N_DROP
            df.at[i, "dataset_to_index"] -= N_DROP
            for cam in ("left_camera", "center_camera", "right_camera"):
                base = f"videos/observation.images.{cam}/"
                ft = df.at[i, base + "from_timestamp"]
                if ft >= TIME_PART2_START - 1e-6:
                    df.at[i, base + "from_timestamp"] = ft - DELTA_T
                    df.at[i, base + "to_timestamp"] -= DELTA_T

    # Rows 0-2 unchanged stats; rows 3-12 (old ep 86-95) need stats + images copied from df_old rows 4-13
    for new_i in range(3, 13):
        old_i = new_i + 1
        ep = int(df.at[new_i, "episode_index"])
        recompute_row_from_fulltable(df, new_i, ep, full, df_old.iloc[old_i])

    _coerce_numeric_stats_columns(df)

    out = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(out, path)
    print(f"[episodes] wrote {path.name} rows={len(out)}")


def patch_episodes_file_010(full: pa.Table):
    path = EP_CHUNK / "file-010.parquet"
    orig = pq.read_table(path)
    schema = orig.schema
    df_old = orig.to_pandas()
    df = df_old.copy()
    for i in range(len(df)):
        df.at[i, "episode_index"] = int(df.at[i, "episode_index"]) - 1
        df.at[i, "dataset_from_index"] -= N_DROP
        df.at[i, "dataset_to_index"] -= N_DROP
        ep = int(df.at[i, "episode_index"])
        recompute_row_from_fulltable(df, i, ep, full, df_old.iloc[i])

    _coerce_numeric_stats_columns(df)

    out = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(out, path)
    print(f"[episodes] wrote {path.name} rows={len(out)}")


def recompute_dataset_stats_json():
    paths = sorted(DATA_CHUNK.glob("*.parquet"))
    full = pa.concat_tables([pq.read_table(p) for p in paths])
    n = len(full)
    assert n == 50758 - N_DROP, n

    stats_path = DS / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    ei = full["episode_index"].to_numpy()
    action = np.stack([np.asarray(x, dtype=np.float32) for x in full["action"].to_pylist()])
    ostate = np.stack([np.asarray(x, dtype=np.float32) for x in full["observation.state"].to_pylist()])

    def feat_json(name, arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return {
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "count": [int(a.shape[0])],
            "q01": np.quantile(a, 0.01, axis=0).tolist(),
            "q10": np.quantile(a, 0.10, axis=0).tolist(),
            "q50": np.quantile(a, 0.50, axis=0).tolist(),
            "q90": np.quantile(a, 0.90, axis=0).tolist(),
            "q99": np.quantile(a, 0.99, axis=0).tolist(),
        }

    stats["action"] = feat_json("action", action)
    stats["observation.state"] = feat_json("observation.state", ostate)
    for key, col in [
        ("timestamp", "timestamp"),
        ("frame_index", "frame_index"),
        ("episode_index", "episode_index"),
        ("index", "index"),
        ("task_index", "task_index"),
    ]:
        arr = full[col].to_numpy()
        stats[key] = feat_json(key, arr)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[meta] wrote stats.json ({n} frames)")


def patch_info_json():
    import pyarrow.compute as pc

    path = DS / "meta" / "info.json"
    paths = sorted(DATA_CHUNK.glob("*.parquet"))
    full = pa.concat_tables([pq.read_table(p) for p in paths])
    n_frames = len(full)
    n_eps = int(pc.max(full["episode_index"]).as_py()) + 1
    with open(path) as f:
        info = json.load(f)
    info["total_episodes"] = n_eps
    info["total_frames"] = n_frames
    info["splits"]["train"] = f"0:{n_eps}"
    with open(path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"[meta] info.json episodes={n_eps} frames={n_frames}")


def load_full_table() -> pa.Table:
    paths = sorted(DATA_CHUNK.glob("*.parquet"))
    return pa.concat_tables([pq.read_table(p) for p in paths])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--meta-only",
        action="store_true",
        help="Only patch meta/episodes, info.json, stats.json (data+video already done).",
    )
    args = ap.parse_args()

    if not DS.is_dir():
        raise SystemExit(f"Dataset not found: {DS}")

    if not args.meta_only:
        patch_data_file_009()
        patch_data_file_010()
        for cam in (
            "observation.images.left_camera",
            "observation.images.center_camera",
            "observation.images.right_camera",
        ):
            trim_video_file_009(cam)

    full = load_full_table()
    n_expect = 50758 - N_DROP
    if len(full) != n_expect:
        raise SystemExit(f"Expected {n_expect} frames after drop, got {len(full)}")

    patch_episodes_file_009(full)
    patch_episodes_file_010(full)
    patch_info_json()
    recompute_dataset_stats_json()
    print("Done. Run recompute_image_stats.py under pixi for camera mean/std in stats.json.")


if __name__ == "__main__":
    main()
