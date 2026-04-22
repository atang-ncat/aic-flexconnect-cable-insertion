#!/usr/bin/env python3
"""Quick physical-unit evaluation for v5 runs, comparable to baselines_ols.py.

Handles both v5 variants:

  * v5_clean (regression, state-stripped)  -- predicts normalized actions
    in the NEW train-only-stats-14D-state normalizer, so we un-normalize
    with the same action mean/std that the run's preprocessor uses.
  * v5_cls   (classification, state-stripped) -- predicts physical actions
    directly from the argmax lookup; no un-normalization needed.

The held-out val split is the same seed=42 split used by baselines_ols.py
(15 eps, ~11.8k frames), so the reported l1_phys / motion_accuracy numbers
are directly comparable to:

    zero-predictor       l1_phys=0.006493  motion_acc= 0.00%
    OLS-clean (14-D)     l1_phys=0.009297  motion_acc=31.79%
    OLS-leaky (26-D)     l1_phys=0.003629  motion_acc=80.78%
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    import certifi  # noqa: F401
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import PolicyFeature

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_discrete_act import DiscreteACTPolicy, snap_to_class  # noqa: E402


def split_episodes(n_episodes: int, n_val: int, seed: int):
    rng = random.Random(seed)
    eps = list(range(n_episodes))
    rng.shuffle(eps)
    return sorted(eps[n_val:]), sorted(eps[:n_val])


def load_act_config_from_ckpt(ckpt_dir: Path) -> tuple[ACTConfig, dict]:
    with open(ckpt_dir / "config.json") as f:
        raw = json.load(f)

    def _to_feat(d: dict) -> PolicyFeature:
        return PolicyFeature(type=FeatureType[d["type"]], shape=tuple(d["shape"]))

    input_features = {k: _to_feat(v) for k, v in raw.pop("input_features").items()}
    output_features = {k: _to_feat(v) for k, v in raw.pop("output_features").items()}
    drop = {"use_peft", "push_to_hub", "repo_id", "private", "tags", "license"}
    raw = {k: v for k, v in raw.items() if k not in drop}
    if "normalization_mapping" in raw:
        raw["normalization_mapping"] = {
            k: NormalizationMode[v] for k, v in raw["normalization_mapping"].items()
        }
    cfg = ACTConfig(input_features=input_features, output_features=output_features, **raw)
    return cfg, raw


def get_state_keep_idx(run_cfg: dict, ds_meta) -> list[int] | None:
    """Resolve state_drop_prefixes from the run's saved config.yaml."""
    ds_cfg = run_cfg.get("dataset", {})
    prefixes = ds_cfg.get("state_drop_prefixes")
    if not prefixes:
        return None
    names = ds_meta.features["observation.state"].get("names") or []
    keep = [i for i, n in enumerate(names) if not any(n.startswith(p) for p in prefixes)]
    return keep if len(keep) != len(names) else None


def metrics(pred_phys: np.ndarray, target_phys: np.ndarray) -> dict:
    err = pred_phys - target_phys
    l1 = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    target_cls = np.clip(np.round(target_phys / 0.1), -1, 1).astype(np.int64) + 1
    pred_cls = np.clip(np.round(pred_phys / 0.1), -1, 1).astype(np.int64) + 1
    correct = (pred_cls == target_cls)
    overall_acc = float(correct.mean())
    motion_mask = (target_cls != 1)
    motion_acc = float(correct[motion_mask].mean()) if motion_mask.any() else float("nan")
    l1_trans = float(np.mean(np.abs(err[:, :3])))
    l1_angular = float(np.mean(np.abs(err[:, 3:])))
    return {
        "l1_phys": l1,
        "mse_phys": mse,
        "l1_trans": l1_trans,
        "l1_angular": l1_angular,
        "accuracy": overall_acc,
        "motion_accuracy": motion_acc,
    }


def evaluate_run(run_dir: Path, ckpt: str = "best", device: str = "cuda:0") -> dict:
    with open(run_dir / "config.yaml") as f:
        run_cfg = yaml.safe_load(f)

    ckpt_dir = run_dir / "checkpoints" / ckpt
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"{ckpt_dir} not found")
    # resolve symlink
    ckpt_dir = ckpt_dir.resolve()

    cfg, _ = load_act_config_from_ckpt(ckpt_dir)

    # Discrete?
    use_discrete = bool(run_cfg.get("policy", {}).get("discrete_action", {}).get("enable", False))

    # Build policy
    policy: Any
    if use_discrete:
        policy = DiscreteACTPolicy(cfg)
    else:
        policy = ACTPolicy(cfg)

    # Load weights. LeRobot saves under ``model.safetensors``.
    from safetensors.torch import load_file

    sd = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  (loaded with {len(missing)} missing / {len(unexpected)} unexpected keys)")

    policy.to(device).eval()

    # Build the RAW val dataset (same seed=42 split as training + baselines).
    ds_cfg = run_cfg["dataset"]
    raw_root = "/scratch2/atang/ws_aic/teleop-dataset"  # always raw for cross-eval
    # Need delta_timestamps so chunk dim aligns with policy expectations.
    probe = LeRobotDataset(repo_id=ds_cfg["repo_id"], root=raw_root, video_backend="pyav")
    _, val_eps = split_episodes(probe.num_episodes, ds_cfg["val_num_episodes"], run_cfg["seed"])
    fps = probe.fps
    dt = 1.0 / fps
    delta_ts = {"action": [i * dt for i in cfg.action_delta_indices]}
    del probe

    ds = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=raw_root,
        episodes=val_eps,
        delta_timestamps=delta_ts,
        video_backend="pyav",
    )

    # Resolve state keep idx for the run, and build a filtered-stats preprocessor.
    keep_idx = get_state_keep_idx(run_cfg, ds.meta)
    baseline_stats = ds.meta.stats
    if keep_idx is not None:
        # filter stats
        src = baseline_stats["observation.state"]
        idx_np = np.asarray(keep_idx, dtype=np.int64)
        new_state_stats = {}
        for sk, sv in src.items():
            arr = np.asarray(sv)
            if arr.ndim >= 1 and arr.shape[0] == len(src["mean"]):
                new_state_stats[sk] = arr[idx_np]
            else:
                new_state_stats[sk] = arr
        baseline_stats = {**baseline_stats, "observation.state": new_state_stats}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=None,
        dataset_stats=baseline_stats,
    )

    # Restrict cameras in meta so videos we don't use don't get decoded.
    cam_keep = set(cfg.image_features.keys())
    all_cams = [k for k, v in ds.meta.features.items() if v.get("dtype") == "video"]
    if not all(c in cam_keep for c in all_cams):
        orig_cls = type(ds.meta)

        class _FM(orig_cls):
            @property
            def video_keys(self):
                orig = orig_cls.video_keys.fget(self)
                return [k for k in orig if k in cam_keep]

        ds.meta.__class__ = _FM

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    keep_t = torch.as_tensor(keep_idx, dtype=torch.long, device=device) if keep_idx is not None else None

    # Gather predictions, compare to raw physical targets.
    all_pred = []
    all_target = []
    with torch.no_grad():
        for batch in loader:
            # Move to device
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            # Snapshot the raw target BEFORE preprocessor normalizes it.
            target_phys = batch["action"].clone().float()  # (B, T, A) raw
            # State filter
            if keep_t is not None:
                batch["observation.state"] = batch["observation.state"].index_select(dim=-1, index=keep_t)
            batch = preprocessor(batch)

            pred = policy.predict_action_chunk(batch).float()  # (B, T, A)

            if use_discrete:
                # already physical
                pred_phys = pred
            else:
                # un-normalize using the action stats the preprocessor was built from
                a_mean = torch.as_tensor(baseline_stats["action"]["mean"], device=device, dtype=pred.dtype)
                a_std = torch.as_tensor(baseline_stats["action"]["std"], device=device, dtype=pred.dtype)
                pred_phys = pred * a_std + a_mean

            # Mask pads
            mask = ~batch["action_is_pad"]  # (B, T)
            pred_flat = pred_phys[mask].cpu().numpy()
            tgt_flat = target_phys[mask].cpu().numpy()
            all_pred.append(pred_flat)
            all_target.append(tgt_flat)

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    return metrics(pred, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[
        "outputs/act_sfp/v5_clean_baseline",
        "outputs/act_sfp/v5_classification",
    ])
    p.add_argument("--ckpt", default="best")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workspace", default="/scratch2/atang/ws_aic")
    args = p.parse_args()

    print(f"{'run':<30s} {'ckpt':<20s} {'l1_phys':>10s} {'mse_phys':>11s} {'l1_trans':>10s} {'l1_ang':>10s} {'acc':>7s} {'motion_acc':>11s}")
    print("-" * 120)
    for rel in args.runs:
        run_dir = Path(args.workspace) / rel
        if not run_dir.exists():
            print(f"  (skip) {rel} -- not found")
            continue
        try:
            m = evaluate_run(run_dir, ckpt=args.ckpt, device=args.device)
            tag = f"{rel.split('/')[-1]}"
            # Read actual step the best ckpt points to
            ck = (run_dir / "checkpoints" / args.ckpt).resolve().name
            print(f"{tag:<30s} {ck:<20s} {m['l1_phys']:>10.6f} {m['mse_phys']:>11.8f} {m['l1_trans']:>10.6f} {m['l1_angular']:>10.6f} {m['accuracy']:>7.4f} {m['motion_accuracy']:>11.4f}")
        except Exception as e:
            print(f"  (error) {rel} -- {e!r}")

    # Reference line
    print()
    print("Baseline reference (from scripts/baselines_ols.py):")
    print(f"{'zero-predictor':<30s} {'-':<20s} {0.006493:>10.6f} {0.00064927:>11.8f} {0.005440:>10.6f} {0.007545:>10.6f} {0.9351:>7.4f} {0.0000:>11.4f}")
    print(f"{'OLS-clean (14-D state)':<30s} {'-':<20s} {0.009297:>10.6f} {0.00042673:>11.8f} {0.009381:>10.6f} {0.009213:>10.6f} {0.9556:>7.4f} {0.3179:>11.4f}")
    print(f"{'OLS-leaky (26-D state)':<30s} {'-':<20s} {0.003629:>10.6f} {0.00014570:>11.8f} {0.006011:>10.6f} {0.001248:>10.6f} {0.9830:>7.4f} {0.8078:>11.4f}")


if __name__ == "__main__":
    main()
