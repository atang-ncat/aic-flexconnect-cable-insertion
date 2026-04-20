#!/usr/bin/env python3
"""Train a VQ-BeT policy on the teleoperated SFP-insertion dataset.

Design goals:
    * Explicit, readable training loop (no draccus / lerobot CLI indirection).
    * Re-use LeRobot's VQBeTConfig / VQBeTPolicy / pre-post processors so that
      checkpoints produced here are drop-in compatible with lerobot-eval or an
      inference-time ROS policy like `RunACT`/`RunVQBeT`.
    * Split the single-split teleop dataset into train (default 136 ep) and
      validation (default 15 ep) by episode index, seeded and deterministic.
    * VQ-BeT has a two-phase training schedule: for the first
      ``n_vqvae_training_steps`` updates the policy's ``forward`` trains only
      the Residual-VQ; afterwards it switches automatically to the BeT GPT +
      offset head. This script logs a ``phase`` metric so the transition is
      visible in WandB/TensorBoard.

Typical launch from the workspace root:

    cd /scratch2/atang/ws_aic/src/aic && pixi run python \\
        /scratch2/atang/ws_aic/scripts/train_vqbet.py \\
        --config /scratch2/atang/ws_aic/configs/vqbet_sfp.yaml

Any scalar in the YAML can be overridden on the CLI, e.g.

    --set training.batch_size=128 training.steps=50_000 wandb.enable=false
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Pixi's bundled Python doesn't point at a CA bundle by default, which breaks
# torchvision's ImageNet-weights download via urllib. Route SSL verification
# through the certifi bundle shipped in the env.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402

from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

log = logging.getLogger("train_vqbet")


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _parse_scalar(raw: str) -> Any:
    """Parse a CLI override value as YAML (so true/false/null/ints/lists work)."""
    return yaml.safe_load(raw)


def _apply_dotted_override(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override '{ov}' must be of the form key.subkey=value")
        k, v = ov.split("=", 1)
        _apply_dotted_override(cfg, k.strip(), _parse_scalar(v.strip()))

    # Resolve any relative paths against the workspace root (config's parent
    # directory's parent, since configs live under <workspace>/configs/).
    workspace_root = path.parent.parent
    for key in ("output_dir",):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((workspace_root / cfg[key]).resolve())
    ds_root = cfg.get("dataset", {}).get("root")
    if ds_root and not Path(ds_root).is_absolute():
        resolved = (workspace_root / ds_root).resolve()
        if not resolved.exists():
            # Fall back to cwd-relative if the workspace-relative path does not exist.
            resolved = Path(ds_root).resolve()
        cfg["dataset"]["root"] = str(resolved)
    return cfg


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def split_episodes(n_episodes: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically split episode indices into train / val lists."""
    rng = random.Random(seed)
    all_eps = list(range(n_episodes))
    rng.shuffle(all_eps)
    val = sorted(all_eps[:n_val])
    train = sorted(all_eps[n_val:])
    return train, val


def build_delta_timestamps(fps: int, policy_cfg: VQBeTConfig) -> dict[str, list[float]]:
    """Time-window indices -> timestamps (in seconds) for the LeRobotDataset sampler."""
    dt = 1.0 / fps
    obs_ts = [i * dt for i in policy_cfg.observation_delta_indices]
    act_ts = [i * dt for i in policy_cfg.action_delta_indices]
    delta_ts: dict[str, list[float]] = {"action": act_ts, "observation.state": obs_ts}
    for img_key in policy_cfg.image_features:
        delta_ts[img_key] = obs_ts
    return delta_ts


def select_policy_features(
    ds_meta, camera_keys: list[str]
) -> tuple[dict, dict]:
    """Return (input_features, output_features) filtered to the requested cameras.

    This is what `make_policy` normally does from the dataset metadata, except
    here we explicitly drop the camera keys that are not in ``camera_keys``
    (needed for VQ-BeT in LeRobot 0.4.3, whose validator rejects >1 image).
    """
    all_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: v for k, v in all_features.items() if v.type is FeatureType.ACTION}
    input_features = {}
    for k, v in all_features.items():
        if k in output_features:
            continue
        if v.type is FeatureType.VISUAL and k not in camera_keys:
            continue
        input_features[k] = v
    missing = [c for c in camera_keys if c not in input_features]
    if missing:
        raise ValueError(f"Requested cameras not found in dataset: {missing}")
    return input_features, output_features


def _restrict_video_keys(meta, keep: list[str]) -> None:
    """Restrict which camera videos a ``LeRobotDataset`` decodes per sample.

    ``LeRobotDatasetMetadata.video_keys`` is a ``@property`` computed from
    ``info["features"]``. Mutating the underlying features dict would also
    perturb normalization stats, policy feature inference, and other code
    paths that rely on the full feature list. Instead we subclass the meta
    object in place and override just the property so decoding iterates a
    subset of keys while everything else sees the original layout.

    The relevant decode paths are ``_get_query_timestamps`` (L973) and
    ``_query_videos`` (L1022-L1030) in ``lerobot.datasets.lerobot_dataset``,
    both of which iterate over ``self.meta.video_keys``. Dropped cameras are
    therefore never loaded into a batch.
    """
    keep_set = set(keep)
    orig_cls = type(meta)
    orig_prop = orig_cls.video_keys  # @property descriptor on the original class

    class _FilteredMeta(orig_cls):
        @property
        def video_keys(self):  # type: ignore[override]
            return [k for k in orig_prop.fget(self) if k in keep_set]

    _FilteredMeta.__name__ = orig_cls.__name__ + "WithFilteredVideoKeys"
    _FilteredMeta.__qualname__ = orig_cls.__qualname__ + "WithFilteredVideoKeys"
    meta.__class__ = _FilteredMeta


def make_datasets(cfg: dict, policy_cfg: VQBeTConfig) -> tuple[LeRobotDataset, LeRobotDataset]:
    ds_cfg = cfg["dataset"]
    # First pass: load a metadata-only copy to discover num_episodes / fps.
    probe = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=ds_cfg["root"],
        video_backend=ds_cfg.get("video_backend", "pyav"),
    )
    fps = probe.fps
    n_episodes = probe.num_episodes
    # Stash selected features on the policy config before we tear down the probe,
    # so downstream `make_policy` does not re-infer (and re-introduce) the cams.
    input_features, output_features = select_policy_features(
        probe.meta, ds_cfg["cameras"]
    )
    policy_cfg.input_features = input_features
    policy_cfg.output_features = output_features
    del probe

    train_eps, val_eps = split_episodes(n_episodes, ds_cfg["val_num_episodes"], cfg["seed"])
    log.info("Split %d episodes -> %d train / %d val (seed=%d)", n_episodes, len(train_eps), len(val_eps), cfg["seed"])

    delta_ts = build_delta_timestamps(fps, policy_cfg)

    train_ds = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=ds_cfg["root"],
        episodes=train_eps,
        delta_timestamps=delta_ts,
        video_backend=ds_cfg.get("video_backend", "pyav"),
    )
    val_ds = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=ds_cfg["root"],
        episodes=val_eps,
        delta_timestamps=delta_ts,
        video_backend=ds_cfg.get("video_backend", "pyav"),
    )

    # --- Skip decoding cameras the policy doesn't use.
    # This cuts per-step data-loader wait proportionally (3 cameras -> 1 camera
    # means ~3x faster pyav decoding). The policy features have already been
    # filtered above, so dropped cameras won't appear in the batch anyway.
    all_cams = [k for k, v in train_ds.meta.features.items() if v.get("dtype") == "video"]
    unused = [k for k in all_cams if k not in set(ds_cfg["cameras"])]
    if unused:
        _restrict_video_keys(train_ds.meta, ds_cfg["cameras"])
        _restrict_video_keys(val_ds.meta, ds_cfg["cameras"])
        log.info(
            "Skipping unused cameras in dataloader: %s (keeping: %s)",
            unused, ds_cfg["cameras"],
        )

    return train_ds, val_ds


def cycle(iterable: Iterable):
    while True:
        yield from iter(iterable)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


@dataclass
class CheckpointBookkeeper:
    root: Path
    keep_last_n: int = 3
    recent: list[Path] = field(default_factory=list)
    best_val: float = float("inf")
    best_dir: Path | None = None

    def save(
        self,
        step: int,
        policy,
        preprocessor,
        postprocessor,
        optimizer,
        scheduler,
        is_best: bool,
        val_loss: float | None,
    ) -> Path:
        tag = f"step_{step:09d}"
        ckpt_dir = self.root / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # HF-style policy weights + config + preprocessor/postprocessor bundles.
        policy.save_pretrained(ckpt_dir)
        preprocessor.save_pretrained(ckpt_dir)
        postprocessor.save_pretrained(ckpt_dir)

        # Training state (so we can resume).
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.random.get_rng_state(),
                "rng_torch_cuda": torch.cuda.get_rng_state_all(),
                "best_val": self.best_val,
            },
            ckpt_dir / "training_state.pt",
        )
        (self.root / "checkpoints" / "last").unlink(missing_ok=True)
        try:
            (self.root / "checkpoints" / "last").symlink_to(tag, target_is_directory=True)
        except OSError:
            pass

        self.recent.append(ckpt_dir)
        while len(self.recent) > self.keep_last_n:
            old = self.recent.pop(0)
            if old != self.best_dir and old.exists():
                shutil.rmtree(old, ignore_errors=True)

        if is_best and val_loss is not None:
            # Point "best" to the new dir; previous best stays on disk if still
            # within the last-N window, otherwise it's already been pruned.
            self.best_val = val_loss
            self.best_dir = ckpt_dir
            best_link = self.root / "checkpoints" / "best"
            best_link.unlink(missing_ok=True)
            try:
                best_link.symlink_to(tag, target_is_directory=True)
            except OSError:
                pass

        return ckpt_dir


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    policy,
    preprocessor,
    postprocessor,
    dataloader: DataLoader,
    max_batches: int,
    device: torch.device,
    autocast_ctx,
) -> dict[str, float]:
    policy.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = preprocessor(batch)

        with autocast_ctx():
            loss, info = policy.forward(batch)

        # Action-space MSE in original (unnormalized) units using postprocessor.
        try:
            # VQBeTPolicy.predict_action_chunk expects queues; for batch eval
            # we call the underlying VQBeTModel directly in rollout mode.
            # That path only works once VQ-VAE has been discretized.
            if policy.vqbet.action_head.vqvae_model.discretized.item():
                model_out = policy.vqbet(
                    {
                        **batch,
                        "observation.images": torch.stack(
                            [batch[k] for k in policy.config.image_features], dim=-4
                        ),
                    },
                    rollout=True,
                )
                pred = model_out[:, : policy.config.action_chunk_size]  # (B, chunk, action_dim) normalized
                target = batch["action"][:, : policy.config.action_chunk_size]
                totals["val/action_mse_norm"] = totals.get("val/action_mse_norm", 0.0) + float(
                    torch.nn.functional.mse_loss(pred, target).item()
                )
                counts["val/action_mse_norm"] = counts.get("val/action_mse_norm", 0) + 1
        except Exception as e:
            log.debug("rollout-mode MSE skipped: %s", e)

        totals["val/loss"] = totals.get("val/loss", 0.0) + float(loss.item())
        counts["val/loss"] = counts.get("val/loss", 0) + 1
        for k, v in info.items():
            if torch.is_tensor(v):
                v = float(v.item())
            key = f"val/{k}"
            totals[key] = totals.get(key, 0.0) + float(v)
            counts[key] = counts.get(key, 0) + 1

    policy.train()
    return {k: totals[k] / max(counts[k], 1) for k in totals}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_backbone_weights(policy, backbone_cfg: dict | None, device: torch.device) -> None:
    """Override the VQ-BeT vision backbone's weights.

    Must run after ``make_policy`` (so the module exists) and before the
    optimizer is built (so the new parameters are registered in the optim
    param groups).

    ``source`` can be one of:
        * ``null`` / missing   - do nothing (random init, or whatever
          ``pretrained_backbone_weights`` loaded inside LeRobot already).
        * ``imagenet``         - re-load torchvision's ImageNet-1K-V1 weights
          into the backbone. Useful if the policy config set
          ``pretrained_backbone_weights: null`` for some reason.
        * ``r3m``              - Robotics-pretrained ResNet-50 from Nair et
          al., hosted on HF Hub at ``hf_repo/hf_filename``. Weights are a
          flat torchvision-compatible state_dict under a ``convnet.`` prefix,
          with ``fc.*`` absent (we discard it anyway).

    For ``r3m`` we build a fresh ``torchvision.models.<backbone>`` with the
    downloaded weights, then swap out ``policy.vqbet.rgb_encoder.backbone``
    (an ``nn.Sequential`` with avgpool+fc already stripped) for the first
    N-2 children of that reference network. This keeps the SpatialSoftmax
    pool and the subsequent Linear/ReLU intact: their input shapes were
    computed at policy-construction time from the random-init backbone's
    feature-map shape, which is determined by architecture alone.
    """
    if not backbone_cfg:
        return
    source = backbone_cfg.get("source")
    if source in (None, "", "random"):
        return

    backbone_name = policy.config.vision_backbone
    if source == "imagenet":
        log.info("Loading torchvision ImageNet-1K-V1 weights into %s backbone", backbone_name)
        ref_net = getattr(torchvision.models, backbone_name)(weights="DEFAULT")
    elif source == "r3m":
        if not backbone_name.startswith("resnet50"):
            raise ValueError(
                f"backbone_init.source=r3m requires vision_backbone=resnet50, got {backbone_name!r}"
            )
        from huggingface_hub import hf_hub_download

        log.info(
            "Downloading R3M weights from hf://%s/%s",
            backbone_cfg["hf_repo"], backbone_cfg["hf_filename"],
        )
        ckpt_path = hf_hub_download(
            repo_id=backbone_cfg["hf_repo"],
            filename=backbone_cfg["hf_filename"],
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prefix = backbone_cfg.get("strip_prefix", "convnet.")
        if prefix:
            sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
        ref_net = torchvision.models.resnet50(weights=None)
        missing, unexpected = ref_net.load_state_dict(sd, strict=False)
        non_fc_missing = [k for k in missing if not k.startswith("fc.")]
        if non_fc_missing:
            raise RuntimeError(f"R3M load is missing non-fc keys: {non_fc_missing}")
        if unexpected:
            log.warning("R3M load has unexpected keys (ignored): %s", unexpected[:10])
    else:
        raise ValueError(f"Unknown backbone_init.source: {source!r}")

    new_backbone = nn.Sequential(*list(ref_net.children())[:-2])
    policy.vqbet.rgb_encoder.backbone = new_backbone.to(device)
    log.info("Installed %s-initialized backbone (%s)", source, backbone_name)


def build_vqbet_config(policy_overrides: dict, device: str) -> VQBeTConfig:
    kwargs = dict(policy_overrides)
    # YAML gives lists for tuple-typed fields; cast.
    if "crop_shape" in kwargs and isinstance(kwargs["crop_shape"], list):
        kwargs["crop_shape"] = tuple(kwargs["crop_shape"])
    if "optimizer_betas" in kwargs and isinstance(kwargs["optimizer_betas"], list):
        kwargs["optimizer_betas"] = tuple(kwargs["optimizer_betas"])
    cfg = VQBeTConfig(**kwargs)
    cfg.device = device
    return cfg


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "train.log"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def init_wandb(cfg: dict, run_dir: Path, run_name: str):
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enable", False):
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb.enable=true but wandb is not installed; skipping.")
        return None
    run = wandb.init(
        project=wb_cfg.get("project"),
        entity=wb_cfg.get("entity"),
        name=run_name,
        dir=str(run_dir),
        tags=wb_cfg.get("tags") or None,
        config=cfg,
        save_code=False,
    )
    return run


def train(cfg: dict) -> None:
    # --- Run dir ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or f"vqbet_sfp_{timestamp}"
    run_dir = Path(cfg["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    log.info("Run dir: %s", run_dir)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Reproducibility ---
    seed = int(cfg["seed"])
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(cfg["device"])

    # --- Policy config ---
    policy_cfg = build_vqbet_config(cfg["policy"], device=cfg["device"])

    # --- Datasets ---
    train_ds, val_ds = make_datasets(cfg, policy_cfg)
    log.info(
        "Train: %d ep / %d frames | Val: %d ep / %d frames",
        train_ds.num_episodes, train_ds.num_frames, val_ds.num_episodes, val_ds.num_frames,
    )

    # --- Policy + processors ---
    # make_policy fills in input/output features from ds_meta and builds VQBeTPolicy.
    policy = make_policy(cfg=policy_cfg, ds_meta=train_ds.meta)
    policy.to(device)
    policy.train()

    # Swap in a robotics-pretrained (or other) backbone before the optimizer is
    # built, so new parameters are tracked in the optim param groups.
    load_backbone_weights(policy, cfg.get("backbone_init"), device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        dataset_stats=train_ds.meta.stats,
    )

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info("Policy params: total=%.1fM trainable=%.1fM", n_total / 1e6, n_trainable / 1e6)

    # --- Optimizer + scheduler (use VQ-BeT presets from the config) ---
    optimizer = policy_cfg.get_optimizer_preset().build(policy.get_optim_params())
    sched_preset = policy_cfg.get_scheduler_preset()
    lr_scheduler = sched_preset.build(optimizer, cfg["training"]["steps"]) if sched_preset else None

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"] and device.type == "cuda",
        drop_last=True,
        prefetch_factor=cfg["training"]["prefetch_factor"] if cfg["training"]["num_workers"] > 0 else None,
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=max(2, cfg["training"]["num_workers"] // 2),
        pin_memory=cfg["training"]["pin_memory"] and device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(train_loader)

    # --- AMP context ---
    amp_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    amp_dtype = amp_dtype_map.get(cfg["training"].get("amp_dtype", "bfloat16"), torch.bfloat16)

    def autocast_ctx():
        if cfg["training"]["use_amp"] and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    # --- WandB ---
    wandb_run = init_wandb(cfg, run_dir, run_name)

    # --- Bookkeeping ---
    ckpt = CheckpointBookkeeper(root=run_dir, keep_last_n=cfg["logging"]["keep_last_n_checkpoints"])
    log_freq = int(cfg["logging"]["log_freq"])
    save_freq = int(cfg["logging"]["save_freq"])
    eval_freq = int(cfg["evaluation"]["eval_freq"])
    total_steps = int(cfg["training"]["steps"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    n_vqvae_steps = int(policy_cfg.n_vqvae_training_steps)

    log.info(
        "Training for %d steps | VQ-VAE phase: first ~%d steps | batch=%d | lr=%g",
        total_steps, n_vqvae_steps, cfg["training"]["batch_size"], policy_cfg.optimizer_lr,
    )

    # --- Loop ---
    step = 0
    running = {"loss": 0.0, "grad_norm": 0.0, "count": 0, "dl_s": 0.0, "step_s": 0.0}
    t_start = time.perf_counter()

    while step < total_steps:
        t_dl0 = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        t_dl = time.perf_counter() - t_dl0

        t_step0 = time.perf_counter()
        with autocast_ctx():
            loss, info = policy.forward(batch)

        loss.backward()
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            lr_scheduler.step()

        t_step = time.perf_counter() - t_step0
        step += 1

        running["loss"] += float(loss.item())
        running["grad_norm"] += float(grad_norm.item())
        running["dl_s"] += t_dl
        running["step_s"] += t_step
        running["count"] += 1

        if step % log_freq == 0:
            n = max(running["count"], 1)
            phase = "vqvae" if not policy.vqbet.action_head.vqvae_model.discretized.item() else "bet"
            lr = optimizer.param_groups[0]["lr"]
            scalar = {
                "train/loss": running["loss"] / n,
                "train/grad_norm": running["grad_norm"] / n,
                "train/lr": lr,
                "train/phase_is_bet": 1.0 if phase == "bet" else 0.0,
                "train/dataloading_s": running["dl_s"] / n,
                "train/step_s": running["step_s"] / n,
                "train/samples_per_s": cfg["training"]["batch_size"] * n / max(running["step_s"], 1e-9),
                "step": step,
            }
            # VQ-VAE diagnostics (when policy returns them) or BeT auxiliary losses.
            for k, v in info.items():
                if torch.is_tensor(v):
                    v = float(v.item())
                scalar[f"train/{k}"] = float(v)

            log.info(
                "step %d/%d | phase=%s | loss=%.4f | grad=%.2f | lr=%.2e | dl=%.3fs | step=%.3fs",
                step, total_steps, phase,
                scalar["train/loss"], scalar["train/grad_norm"], lr,
                scalar["train/dataloading_s"], scalar["train/step_s"],
            )
            if wandb_run is not None:
                wandb_run.log(scalar, step=step)
            running = {k: 0.0 for k in running}
            running["count"] = 0

        do_eval = eval_freq > 0 and (step % eval_freq == 0 or step == total_steps)
        do_save = save_freq > 0 and (step % save_freq == 0 or step == total_steps)

        if do_eval:
            t0 = time.perf_counter()
            val_metrics = evaluate(
                policy, preprocessor, postprocessor,
                val_loader, cfg["evaluation"]["eval_batches"], device, autocast_ctx,
            )
            val_metrics["val/eval_s"] = time.perf_counter() - t0
            log.info("eval @ step %d: %s", step, {k: round(v, 5) for k, v in val_metrics.items()})
            if wandb_run is not None:
                wandb_run.log({**val_metrics, "step": step}, step=step)

        if do_save:
            is_best = False
            val_loss = None
            if do_eval and "val/loss" in val_metrics:
                val_loss = val_metrics["val/loss"]
                is_best = val_loss < ckpt.best_val
            path = ckpt.save(
                step=step,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                is_best=is_best,
                val_loss=val_loss,
            )
            log.info("saved checkpoint: %s%s", path, " (new best)" if is_best else "")

    elapsed = time.perf_counter() - t_start
    log.info("Training complete. %d steps in %.1fs (%.2f steps/s).", total_steps, elapsed, total_steps / max(elapsed, 1e-9))

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    p.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        metavar="key.subkey=value",
        help="Dotted-path overrides for any YAML scalar (values parsed as YAML).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
