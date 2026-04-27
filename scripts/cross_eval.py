#!/usr/bin/env python3
"""Cross-evaluate all ACT runs against a common reference: RAW teleop-dataset val.

The per-run ``val/l1_loss`` and ``val/action_mse_norm`` metrics reported
during training are each computed in that run's own normalized action space
(using its training dataset's mean / std).  Raw-trained runs normalize by
raw stats; smoothed-trained runs normalize by smoothed stats.  The two
normalized spaces have different scales, so the raw cross-run comparison
is misleading.

This script instead computes every run's error in a common reference:
physical action units, against RAW val actions.  Steps per run:

    1. Load the run's best checkpoint (policy + preprocessor).
    2. Grab the run's action mean / std from its own training dataset
       metadata (we need these to UN-normalize the model's predictions).
    3. Feed a RAW val batch through the run's preprocessor (which normalizes
       obs using the run's own stats -- this is correct, since the model was
       trained with those stats).
    4. Call ``predict_action_chunk`` -> normalized pred.
    5. Un-normalize pred back to physical units: ``pred * std_run + mean_run``.
    6. Compare to the ORIGINAL RAW val action (physical units, untouched).

The reported ``l1_physical`` is "average absolute prediction error in action
units" (m/s for translation dims 0-2, rad/s for angular dims 3-5).  This IS
comparable across all runs.  We also report the same metric split by dim
family (trans vs angular) so a run that's great at translation but bad at
rotation stands out.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

from torch.utils.data import DataLoader, Subset  # noqa: E402

import json  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402


def _load_act_config(ckpt_dir: Path) -> ACTConfig:
    """Load ACTConfig from a checkpoint directory.

    We can't use ``ACTPolicy.from_pretrained`` without explicitly passing a
    config, because our saved ``config.json`` lacks the ``type`` dispatch
    key that PreTrainedConfig's draccus parser requires.  So we instantiate
    ACTConfig manually from the JSON.
    """
    with open(ckpt_dir / "config.json") as f:
        raw = json.load(f)

    def _to_feat(d: dict) -> PolicyFeature:
        return PolicyFeature(type=FeatureType[d["type"]], shape=tuple(d["shape"]))

    input_features = {k: _to_feat(v) for k, v in raw.pop("input_features").items()}
    output_features = {k: _to_feat(v) for k, v in raw.pop("output_features").items()}
    # Strip keys that aren't ACTConfig fields (PeFT / hub stuff).
    drop = {"use_peft", "push_to_hub", "repo_id", "private", "tags", "license"}
    raw = {k: v for k, v in raw.items() if k not in drop}
    # normalization_mapping values are strings in JSON; ACTConfig wants
    # NormalizationMode enum.  Convert.
    if "normalization_mapping" in raw:
        from lerobot.configs.types import NormalizationMode
        raw["normalization_mapping"] = {
            k: NormalizationMode[v] for k, v in raw["normalization_mapping"].items()
        }
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        **raw,
    )
    return cfg

log = logging.getLogger("cross_eval")

RAW_ROOT = "/scratch2/atang/ws_aic/teleop-dataset"
OUTPUTS_ROOT = Path("/scratch2/atang/ws_aic/outputs/act_sfp")
SEED = 42
N_VAL = 15  # all runs used the same split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_val_episodes(n_episodes: int, n_val: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    eps = list(range(n_episodes))
    rng.shuffle(eps)
    return sorted(eps[:n_val])


def load_run(run_name: str, device: torch.device) -> dict:
    run_dir = OUTPUTS_ROOT / run_name
    ckpt_dir = run_dir / "checkpoints" / "best"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No best checkpoint at {ckpt_dir}")
    # Resolve symlink for logging
    resolved = ckpt_dir.resolve().name

    # Read config.yaml to know which training dataset was used (we need its
    # action stats to un-normalize the model's output).
    with open(run_dir / "config.yaml") as f:
        run_cfg = yaml.safe_load(f)
    train_ds_root = run_cfg["dataset"]["root"]
    # Resolve relative paths relative to the workspace root so this script
    # is runnable from any cwd.
    if not os.path.isabs(train_ds_root):
        train_ds_root = str(Path("/scratch2/atang/ws_aic") / train_ds_root)
    cams_trained = list(run_cfg["dataset"]["cameras"])

    # Load the training dataset purely to get action mean/std (we don't
    # need its frames -- just meta.stats).  This is fast.
    train_meta_ds = LeRobotDataset(
        repo_id=run_cfg["dataset"]["repo_id"],
        root=train_ds_root,
        video_backend=run_cfg["dataset"].get("video_backend", "pyav"),
    )
    stats_action = train_meta_ds.meta.stats["action"]
    action_mean = torch.as_tensor(stats_action["mean"], dtype=torch.float32, device=device)
    action_std = torch.as_tensor(stats_action["std"], dtype=torch.float32, device=device)
    del train_meta_ds  # don't hold open the hf_dataset

    # Load the policy.  We must pass config= explicitly (see _load_act_config).
    policy_cfg = _load_act_config(ckpt_dir)
    policy_cfg.device = str(device)
    policy = ACTPolicy.from_pretrained(str(ckpt_dir), config=policy_cfg)
    policy.to(device)
    policy.eval()

    # Load the run's own preprocessor.  make_pre_post_processors with
    # pretrained_path=ckpt_dir loads the exact pipeline that was used at train.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(ckpt_dir)
    )

    return {
        "name": run_name,
        "ckpt_step": resolved,
        "policy": policy,
        "policy_cfg": policy_cfg,
        "preprocessor": preprocessor,
        "action_mean": action_mean,   # physical-units mean of THIS run's train set
        "action_std": action_std,     # physical-units std  of THIS run's train set
        "train_ds_root": train_ds_root,
        "cams_trained": cams_trained,
    }


def build_common_val_loader(
    cams_needed: list[str], chunk_size: int, batch_size: int, num_workers: int
) -> tuple[DataLoader, float]:
    """Build one val DataLoader against the RAW dataset, restricted to the
    cameras that every run-under-test needs decoded.
    """
    probe = LeRobotDataset(repo_id="local/teleop_sfp", root=RAW_ROOT, video_backend="pyav")
    n_ep = probe.num_episodes
    fps = probe.fps
    del probe

    val_eps = split_val_episodes(n_ep, N_VAL, SEED)
    log.info("RAW val episodes (seed=%d, n=%d): %s", SEED, len(val_eps), val_eps)

    dt = 1.0 / fps
    delta_ts = {"action": [i * dt for i in range(chunk_size)]}

    val_ds = LeRobotDataset(
        repo_id="local/teleop_sfp",
        root=RAW_ROOT,
        episodes=val_eps,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )
    # Restrict decoded videos to the union of cameras any run needs.
    keep = set(cams_needed)
    all_cams = [k for k, v in val_ds.meta.features.items() if v.get("dtype") == "video"]
    unused = [k for k in all_cams if k not in keep]
    if unused:
        orig_cls = type(val_ds.meta)
        orig_prop = orig_cls.video_keys

        class _FilteredMeta(orig_cls):
            @property
            def video_keys(self):
                return [k for k in orig_prop.fget(self) if k in keep]

        _FilteredMeta.__name__ = orig_cls.__name__ + "Filtered"
        val_ds.meta.__class__ = _FilteredMeta
        log.info("Val loader will decode only: %s (skipping %s)", sorted(keep), unused)

    return (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        fps,
    )


def evaluate_one(run: dict, val_loader: DataLoader, device: torch.device,
                 max_batches: int) -> dict[str, float]:
    """Run one model over the shared RAW val set, report metrics in physical units."""
    policy = run["policy"]
    preprocessor = run["preprocessor"]
    mean = run["action_mean"]  # (A,)
    std = run["action_std"]    # (A,)
    cams = run["cams_trained"]

    totals = {"l1_phys": 0.0, "mse_phys": 0.0, "l1_trans": 0.0, "l1_angular": 0.0}
    counts = {k: 0 for k in totals}

    policy.eval()
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        # Move to device.  Some cameras on the batch may be ones this run didn't
        # train on -- we need to drop those keys so the run's preprocessor /
        # normalizer doesn't complain about unexpected keys.  Keep state, action,
        # action_is_pad, task, and the cameras this run was trained with.
        clean: dict[str, Any] = {}
        for k, v in batch.items():
            if k.startswith("observation.images."):
                if k in cams:
                    clean[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                # else drop
            elif torch.is_tensor(v):
                clean[k] = v.to(device, non_blocking=True)
            else:
                clean[k] = v
        # Save physical-unit ground truth before preprocessor mutates the action.
        gt_phys = clean["action"].clone().float()  # (B, chunk, A)
        pad_mask = None
        if "action_is_pad" in clean:
            # mask[...,None] broadcasts over action dims.
            pad_mask = (~clean["action_is_pad"]).unsqueeze(-1).float()  # (B, chunk, 1)

        # Preprocessor normalizes obs + action to the run's own normalized space.
        clean = preprocessor(clean)

        with torch.no_grad():
            pred_norm = policy.predict_action_chunk(clean).float()  # (B, chunk, A)
        # Un-normalize pred to physical units using THIS run's stats.
        pred_phys = pred_norm * std + mean

        diff = pred_phys - gt_phys
        abs_diff = diff.abs()

        if pad_mask is not None:
            denom_all = pad_mask.sum().clamp_min(1.0) * pred_phys.shape[-1]
            l1 = (abs_diff * pad_mask).sum() / denom_all
            mse = (diff.pow(2) * pad_mask).sum() / denom_all
            # trans = dims 0..2, angular = dims 3..5 (by 6-DoF twist convention)
            denom_half = pad_mask.sum().clamp_min(1.0) * 3
            l1_t = (abs_diff[..., :3] * pad_mask).sum() / denom_half
            l1_a = (abs_diff[..., 3:] * pad_mask).sum() / denom_half
        else:
            l1 = abs_diff.mean()
            mse = diff.pow(2).mean()
            l1_t = abs_diff[..., :3].mean()
            l1_a = abs_diff[..., 3:].mean()

        for k, v in zip(
            ["l1_phys", "mse_phys", "l1_trans", "l1_angular"],
            [l1, mse, l1_t, l1_a],
        ):
            totals[k] += float(v.item())
            counts[k] += 1

    return {k: totals[k] / max(counts[k], 1) for k in totals}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run names (subdirs of outputs/act_sfp) to evaluate.")
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load each run once.  Collect the union of cameras across runs so the
    # shared val loader decodes exactly what's needed.
    runs = [load_run(name, device) for name in args.runs]
    cams_union = sorted({c for r in runs for c in r["cams_trained"]})
    log.info("Cameras union across runs: %s", cams_union)

    # chunk_size: every ACT run we care about used 100.
    chunk_size = int(runs[0]["policy_cfg"].chunk_size)
    for r in runs[1:]:
        assert int(r["policy_cfg"].chunk_size) == chunk_size, \
            f"{r['name']} has chunk_size={r['policy_cfg'].chunk_size}, not {chunk_size}"

    val_loader, fps = build_common_val_loader(
        cams_needed=cams_union,
        chunk_size=chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log.info("Val loader: fps=%g, batches=%d, bs=%d", fps, args.batches, args.batch_size)

    results = {}
    for r in runs:
        log.info("\n--- Evaluating %s (ckpt=%s) ---", r["name"], r["ckpt_step"])
        metrics = evaluate_one(r, val_loader, device, args.batches)
        results[r["name"]] = metrics
        log.info("  %s", {k: round(v, 6) for k, v in metrics.items()})

    # Pretty print ranking table (by l1_phys, lower = better).
    print("\n" + "=" * 80)
    print("CROSS-EVAL on RAW val, metrics in PHYSICAL action units")
    print("=" * 80)
    print(f"{'Run':<28s} {'l1_phys':>12s} {'mse_phys':>12s} "
          f"{'l1_trans':>12s} {'l1_ang':>12s}")
    print("-" * 80)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["l1_phys"])
    for name, m in ranked:
        print(f"{name:<28s} {m['l1_phys']:>12.6f} {m['mse_phys']:>12.6f} "
              f"{m['l1_trans']:>12.6f} {m['l1_angular']:>12.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
