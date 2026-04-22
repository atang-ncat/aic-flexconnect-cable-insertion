#!/usr/bin/env python3
"""Non-vision baselines for the SFP teleop dataset.

This script establishes the honest "floor" that any ACT/VQ-BeT/diffusion
policy must clear to be doing real vision-based learning.

Three baselines are reported, all evaluated on the same held-out val split
used by ``scripts/train_act.py`` (``split_episodes(n_ep, n_val=15, seed=42)``
by default):

  * **zero-predictor**          : always predict action = 0.
  * **OLS(full-state)**         : linear regression from the 26-D
                                   ``observation.state`` to the 6-D action.
                                   Includes the leaky ``tcp_velocity.*`` /
                                   ``tcp_error.*`` channels; this is the
                                   baseline you've been implicitly competing
                                   with.  Expected ``val l1_phys ~= 0.00363``.
  * **OLS(leak-stripped)**      : same linear regression, but with the leaky
                                   state channels dropped (same 14-D input
                                   that ``act_sfp_v5_clean.yaml`` uses).
                                   THIS is the real non-vision baseline ACT
                                   has to beat in v5.
  * **Discrete-majority class** : for each dim, predict 0 (the majority class
                                   for every dim).  Reports per-dim accuracy
                                   so the classification run's numbers are
                                   directly comparable.

All metrics are reported in PHYSICAL units (same as ``cross_eval.py``) so the
numbers are directly comparable across runs, datasets, and normalization
schemes.  We also log to wandb (optional) under a dedicated project so every
training run's wandb workspace shows the floor alongside its own curve.

Usage::

    cd /scratch2/atang/ws_aic/src/aic && pixi run python \\
        /scratch2/atang/ws_aic/scripts/baselines_ols.py \\
        --dataset-root /scratch2/atang/ws_aic/teleop-dataset \\
        --val-num-episodes 15 --seed 42

Add ``--wandb`` to publish under project ``act_sfp_baselines``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Indices of leaky state channels in the 26-D observation.state (see
# teleop-dataset's info.json for the full list of column names).
LEAK_PREFIXES = ("tcp_velocity.", "tcp_error.")

CLASS_VALUES = np.array([-0.1, 0.0, 0.1])


def split_episodes(n_episodes: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    """Same split used by scripts/train_act.py for apples-to-apples comparison."""
    rng = random.Random(seed)
    eps = list(range(n_episodes))
    rng.shuffle(eps)
    val = sorted(eps[:n_val])
    train = sorted(eps[n_val:])
    return train, val


def state_keep_indices(state_names: list[str]) -> list[int]:
    """Return column indices that are NOT leaky (no tcp_velocity/tcp_error)."""
    return [i for i, n in enumerate(state_names) if not any(n.startswith(p) for p in LEAK_PREFIXES)]


def metrics_vs_raw(pred_phys: np.ndarray, target_phys: np.ndarray) -> dict:
    """Compute L1, MSE in physical units, plus per-dim and overall accuracy.

    ``pred_phys``, ``target_phys`` both have shape ``(N, A)`` in raw physical
    units.  Accuracy uses the {-0.1, 0, +0.1} snap of predictions against the
    (already discrete) targets.
    """
    err = pred_phys - target_phys
    l1 = np.mean(np.abs(err))
    mse = np.mean(err ** 2)

    target_cls = np.clip(np.round(target_phys / 0.1), -1, 1).astype(np.int64) + 1
    pred_cls = np.clip(np.round(pred_phys / 0.1), -1, 1).astype(np.int64) + 1
    correct = (pred_cls == target_cls)
    per_dim_acc = correct.mean(axis=0).tolist()
    overall_acc = correct.mean()

    motion_mask = (target_cls != 1)
    if motion_mask.any():
        motion_acc = correct[motion_mask].mean()
    else:
        motion_acc = float("nan")

    # Split linear / angular L1 for the error-budget table.
    l1_trans = np.mean(np.abs(err[:, :3]))
    l1_angular = np.mean(np.abs(err[:, 3:]))

    return {
        "l1_phys": float(l1),
        "mse_phys": float(mse),
        "l1_trans": float(l1_trans),
        "l1_angular": float(l1_angular),
        "accuracy": float(overall_acc),
        "motion_accuracy": float(motion_acc),
        "per_dim_accuracy": per_dim_acc,
    }


def fit_ols(X_tr: np.ndarray, Y_tr: np.ndarray) -> np.ndarray:
    """OLS with bias: returns W of shape (D+1, A)."""
    X_tr_b = np.concatenate([X_tr, np.ones((len(X_tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(X_tr_b, Y_tr, rcond=None)
    return W


def predict_ols(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X_b = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return X_b @ W


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--repo-id", default="local/teleop_sfp")
    p.add_argument("--val-num-episodes", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="Log to project act_sfp_baselines")
    args = p.parse_args()

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=str(args.dataset_root),
        video_backend="pyav",
    )
    state_names = ds.meta.features["observation.state"].get("names") or []

    train_eps, val_eps = split_episodes(ds.num_episodes, args.val_num_episodes, args.seed)
    print(f"Dataset: {args.dataset_root}")
    print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"  split (seed={args.seed}): {len(train_eps)} train / {len(val_eps)} val")

    # Load state + action for all frames.
    state = np.stack([np.asarray(s, dtype=np.float32) for s in ds.hf_dataset["observation.state"]])
    action = np.stack([np.asarray(a, dtype=np.float32) for a in ds.hf_dataset["action"]])
    ep_idx = np.asarray(ds.hf_dataset["episode_index"])

    val_eps_set = set(val_eps)
    train_mask = ~np.isin(ep_idx, list(val_eps_set))
    val_mask = np.isin(ep_idx, list(val_eps_set))

    # ---- Baseline 1: zero-predictor ----
    pred_zero = np.zeros_like(action[val_mask])
    m_zero = metrics_vs_raw(pred_zero, action[val_mask])

    # ---- Baseline 2: OLS on full state ----
    W_full = fit_ols(state[train_mask], action[train_mask])
    pred_full = predict_ols(state[val_mask], W_full)
    m_full = metrics_vs_raw(pred_full, action[val_mask])

    # ---- Baseline 3: OLS on leak-stripped state ----
    keep_idx = state_keep_indices(state_names)
    kept_names = [state_names[i] for i in keep_idx]
    dropped = [n for n in state_names if n not in set(kept_names)]
    state_clean = state[:, keep_idx]
    W_clean = fit_ols(state_clean[train_mask], action[train_mask])
    pred_clean = predict_ols(state_clean[val_mask], W_clean)
    m_clean = metrics_vs_raw(pred_clean, action[val_mask])

    # ---- Baseline 4: "always predict idle" classification baseline ----
    # Represented in physical space as zero, so identical to m_zero's L1,
    # but we rename the key set for clarity in the reported table.
    m_majority = dict(m_zero)

    # ---- Report ----
    print()
    print(f"Held-out val: {val_mask.sum()} frames across {len(val_eps)} episodes")
    print()
    print(f"  Leaky state columns dropped ({len(dropped)}): {dropped}")
    print(f"  Kept state columns ({len(keep_idx)}): {kept_names}")
    print()
    print(f"{'baseline':<35s} {'l1_phys':>10s} {'mse_phys':>11s} {'l1_trans':>10s} {'l1_ang':>10s} {'acc':>7s} {'motion_acc':>11s}")
    print("-" * 100)
    for name, m in [
        ("zero-predictor (=majority-class)", m_zero),
        ("OLS from full 26-D state (LEAKY)", m_full),
        ("OLS from 14-D state (leak-strip)", m_clean),
    ]:
        print(
            f"{name:<35s} {m['l1_phys']:>10.6f} {m['mse_phys']:>11.8f} "
            f"{m['l1_trans']:>10.6f} {m['l1_angular']:>10.6f} "
            f"{m['accuracy']:>7.4f} {m['motion_accuracy']:>11.4f}"
        )

    print()
    print("Reference for interpretation:")
    print("  * ACT v2_center_only (cross_eval) best: l1_phys = 0.003603")
    print("  * If leak-stripped OLS is close to ACT's old best, ACT was riding the leak.")
    print("  * Any ACT run trained with the v5_clean config should be compared")
    print("    against the LEAK-STRIP baseline, not the full-state one.")

    if args.wandb:
        import wandb

        run = wandb.init(
            project="act_sfp_baselines",
            name="ols_baselines",
            config={
                "dataset_root": str(args.dataset_root),
                "val_num_episodes": args.val_num_episodes,
                "seed": args.seed,
                "leaky_columns_dropped": dropped,
            },
        )
        log_payload = {}
        for name, m in [
            ("zero", m_zero),
            ("ols_full", m_full),
            ("ols_clean", m_clean),
            ("majority_class", m_majority),
        ]:
            for k, v in m.items():
                if isinstance(v, list):
                    for i, vi in enumerate(v):
                        log_payload[f"{name}/{k}_dim{i}"] = vi
                else:
                    log_payload[f"{name}/{k}"] = v
        run.log(log_payload)
        run.finish()


if __name__ == "__main__":
    main()
