#!/usr/bin/env python3
"""Train an ACT policy on the teleoperated SFP-insertion dataset.

Sibling of ``train_vqbet.py``; follows the same structure on purpose so that
a single call-site (launcher shell scripts, wandb comparisons, post-hoc
eval) can handle both.  Key differences from ``train_vqbet.py``:

    * No two-phase VQ-VAE pretraining, just a single-phase L1-loss objective
      (plus CVAE KLD if ``use_vae=true``).
    * Supports arbitrarily many ``observation.images.*`` keys (no feature
      pruning required; ACTConfig.validate_features accepts them all).
    * Custom in-loop image augmentation (ColorJitter + RandomErasing) applied
      to training batches *before* the policy preprocessor so augmentation
      happens in [0, 1] image space rather than on mean-std-centered tensors.
    * ``best`` symlink tracks the lowest ``val/l1_loss`` we have observed.

Typical launch (from the workspace root):

    cd /scratch2/atang/ws_aic/src/aic && pixi run python \\
        /scratch2/atang/ws_aic/scripts/train_act.py \\
        --config /scratch2/atang/ws_aic/configs/act_sfp.yaml

Overrides use the same ``--set`` syntax as train_vqbet.py, e.g.:

    --set training.batch_size=48 training.steps=10_000 wandb.enable=false
"""

from __future__ import annotations

import argparse
import logging
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

# Route SSL verification through certifi's bundle (pixi's bundled Python has
# no default CA path, which breaks torchvision's ImageNet-weights download).
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import torch.nn.functional as F  # noqa: E402
import torchvision.transforms.v2 as T  # noqa: E402

from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

log = logging.getLogger("train_act")


# ---------------------------------------------------------------------------
# Config handling  (mirrors train_vqbet.py so both scripts feel identical)
# ---------------------------------------------------------------------------


def _parse_scalar(raw: str) -> Any:
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

    # Resolve workspace-relative paths.  Configs live at <workspace>/configs/.
    workspace_root = path.parent.parent
    for key in ("output_dir",):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((workspace_root / cfg[key]).resolve())
    ds_root = cfg.get("dataset", {}).get("root")
    if ds_root and not Path(ds_root).is_absolute():
        resolved = (workspace_root / ds_root).resolve()
        if not resolved.exists():
            resolved = Path(ds_root).resolve()
        cfg["dataset"]["root"] = str(resolved)
    return cfg


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def split_episodes(n_episodes: int, n_val: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    all_eps = list(range(n_episodes))
    rng.shuffle(all_eps)
    val = sorted(all_eps[:n_val])
    train = sorted(all_eps[n_val:])
    return train, val


def build_delta_timestamps(fps: int, policy_cfg: ACTConfig) -> dict[str, list[float]]:
    """LeRobotDataset delta_timestamps for ACT.

    ACT has ``observation_delta_indices = None`` (single-frame obs) and
    ``action_delta_indices = [0, 1, ..., chunk_size-1]``.  Returning a dict
    with only ``action`` means observations stay scalar-in-time (shape
    (B, C, H, W) rather than (B, 1, C, H, W)).
    """
    dt = 1.0 / fps
    act_ts = [i * dt for i in policy_cfg.action_delta_indices]
    return {"action": act_ts}


def select_policy_features(ds_meta, camera_keys: list[str]) -> tuple[dict, dict]:
    """Filter dataset features to the cameras requested by ``camera_keys``.

    ACT happily accepts multi-image inputs, but we still explicitly trim to
    the cameras listed in the config so that later cam-ablation runs don't
    re-introduce dropped views via ``make_policy``'s auto-inference.
    """
    all_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: v for k, v in all_features.items() if v.type is FeatureType.ACTION}
    input_features: dict = {}
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
    """Prune the set of videos a ``LeRobotDataset`` decodes per sample.

    Identical to the VQ-BeT script's helper; see that file for the rationale
    (subclass ``meta`` in-place so ``video_keys`` returns a filtered view
    without touching feature metadata used for normalization).
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


def make_datasets(cfg: dict, policy_cfg: ACTConfig) -> tuple[LeRobotDataset, LeRobotDataset]:
    ds_cfg = cfg["dataset"]
    probe = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=ds_cfg["root"],
        video_backend=ds_cfg.get("video_backend", "pyav"),
    )
    fps = probe.fps
    n_episodes = probe.num_episodes
    input_features, output_features = select_policy_features(probe.meta, ds_cfg["cameras"])
    policy_cfg.input_features = input_features
    policy_cfg.output_features = output_features
    del probe

    train_eps, val_eps = split_episodes(n_episodes, ds_cfg["val_num_episodes"], cfg["seed"])
    log.info(
        "Split %d episodes -> %d train / %d val (seed=%d)",
        n_episodes, len(train_eps), len(val_eps), cfg["seed"],
    )

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
# Image augmentation (train-only, applied BEFORE the preprocessor's norm)
# ---------------------------------------------------------------------------


class ImageAugmenter:
    """ColorJitter + RandomErasing applied in-place on a LeRobotDataset batch.

    The LeRobot preprocessor expects images in ``[0, 1]`` float space and then
    (for ACT) applies MEAN_STD normalization.  We therefore run our transforms
    on the raw batch before the preprocessor, so both ColorJitter's brightness
    / contrast semantics and RandomErasing's ``value=0.0`` fill are meaningful.

    Images arrive with shape ``(B, C, H, W)`` when ``observation_delta_indices``
    is ``None`` (ACT's case).  If a future policy variant uses a time dim, the
    ``(B, T, C, H, W)`` case is handled by flattening and restoring the leading
    two dims.
    """

    def __init__(self, cfg: dict, image_keys: list[str]):
        self.enabled = bool(cfg.get("enable", True))
        self.image_keys = list(image_keys)
        cj = cfg.get("color_jitter") or {}
        self._cj_p = float(cj.get("p", 0.8))
        self._cj = T.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0),
        )
        re = cfg.get("random_erasing") or {}
        self._re = T.RandomErasing(
            p=float(re.get("p", 0.0)),
            scale=tuple(re.get("scale", (0.02, 0.1))),
            ratio=tuple(re.get("ratio", (0.3, 3.3))),
            value=re.get("value", 0.0),
        )

    def __call__(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        for k in self.image_keys:
            if k not in batch:
                continue
            img = batch[k]
            orig_shape = img.shape
            if img.ndim == 5:  # (B, T, C, H, W) — flatten time into batch
                b, t, c, h, w = img.shape
                img = img.reshape(b * t, c, h, w)
            # ColorJitter is expensive; gate it behind an independent Bernoulli
            # per batch to save CPU.  RandomErasing is cheap so we always call
            # it and let its own ``p`` sample internally.
            if self._cj_p > 0 and torch.rand(()) < self._cj_p:
                img = self._cj(img)
            img = self._re(img)
            if img.shape != orig_shape:
                img = img.reshape(orig_shape)
            batch[k] = img
        return batch


# ---------------------------------------------------------------------------
# Action-weighted L1 loss  (v3 intervention for teleop idle-frame bias)
# ---------------------------------------------------------------------------


class ActionWeighter:
    """Per-step loss weight based on the raw target action magnitude.

    Why this exists (see docs/ audits: ~60% of the teleop SFP dataset frames
    have ``||action||2 < 0.01``, and 95% of each episode's last-10%-of-frames
    is pure post-roll stillness).  Under an unweighted L1 objective the model
    is rewarded for predicting zero most of the time -- which is why v1/v2
    ACT converged to a good-looking train loss but weak insertion behavior.

    The fix is to down-weight (not mask out) frames whose *target* action is
    near zero so that mid-episode hesitation and post-roll stillness
    contribute less loss than the actual insertion motion.  We compute the
    weight from the raw (un-normalized) action so that "near zero" matches
    physical reality:

        w[t] = clip( ||action[t]||2 / scale , min=floor, max=1.0 )

    * ``scale``  : a reference action norm.  Actions of this magnitude or
      larger get weight 1.0.  Pick roughly the median motion norm over
      non-idle frames -- 0.03 works well for the SFP 6-D twist dataset.
    * ``floor``  : minimum per-step weight (how much a pure-idle frame still
      contributes).  Keeping ``floor>0`` regularizes the prediction (the
      model still learns "output zero when idle") but without letting those
      frames dominate.  We use 0.1, which makes motion frames contribute
      10x as much as idle frames to the loss.

    The weights are computed BEFORE ``preprocessor`` normalizes the batch,
    because the preprocessor shifts the action mean and would turn "raw
    zero" into a non-zero normalized number.
    """

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enable", False))
        self.scale = float(cfg.get("scale", 0.03))
        self.floor = float(cfg.get("floor", 0.1))
        self.ceil = float(cfg.get("ceil", 1.0))

    @torch.no_grad()
    def weights_from_raw_batch(self, batch: dict) -> torch.Tensor | None:
        """Compute per-step weights from ``batch["action"]`` (shape (B, T, A))."""
        if not self.enabled:
            return None
        action = batch[ACTION]  # (B, T, A), raw units (e.g., twist cm/s, rad/s)
        norm = action.float().norm(dim=-1)  # (B, T)
        w = (norm / max(self.scale, 1e-12)).clamp(min=self.floor, max=self.ceil)
        return w


def weighted_act_forward(
    policy,
    batch: dict,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    """ACT forward + loss with optional per-step L1 weighting.

    Mirrors ``ACTPolicy.forward`` from ``modeling_act.py`` but:

    1. Uses ``reduction='none'`` so we can apply per-step weights.
    2. Falls back to the vanilla (un-weighted) mean when ``weights`` is
       ``None``, so training with ``action_weighting.enable=false`` is
       bitwise identical to the stock implementation.
    3. Exposes diagnostic scalars (``train/l1_loss_weighted``,
       ``train/mean_action_weight``) via the ``info`` dict.

    KLD regularization is left unchanged -- it penalizes the CVAE latent,
    not the per-step action prediction, so per-step weighting doesn't apply.
    """
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

    actions_hat, (mu_hat, log_sigma_x2_hat) = policy.model(batch)

    # (B, T, A) per-element L1 error, with pad masking
    err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
    pad_mask = (~batch["action_is_pad"]).unsqueeze(-1).float()  # (B, T, 1)
    per_step = (err * pad_mask).mean(dim=-1)  # (B, T), avg over action dims

    loss_dict: dict[str, float] = {}
    if weights is None:
        # Same as upstream: mean over (B*T) valid entries.
        # (Upstream actually uses a simple ``.mean()`` over the masked
        # tensor, which divides by B*T*A regardless of pad; we retain that
        # behavior when weighting is off so that loss scales stay identical
        # to stock runs.)
        l1_loss = (err * pad_mask).mean()
    else:
        # weights: (B, T), pad mask collapsed to (B, T)
        mask = pad_mask.squeeze(-1)
        denom = (weights * mask).sum().clamp_min(1e-6)
        l1_loss = (per_step * weights * mask).sum() / denom
        loss_dict["l1_loss_weighted"] = l1_loss.item()
        loss_dict["mean_action_weight"] = (weights * mask).sum().item() / mask.sum().clamp_min(1.0).item()

    # Expose the unweighted L1 too so wandb plots stay comparable across v2/v3.
    with torch.no_grad():
        unweighted = (err * pad_mask).sum() / pad_mask.sum().clamp_min(1.0) / err.shape[-1]
    loss_dict["l1_loss"] = float(unweighted.item())

    if policy.config.use_vae:
        mean_kld = (
            (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
            .sum(-1).mean()
        )
        loss_dict["kld_loss"] = mean_kld.item()
        loss = l1_loss + mean_kld * policy.config.kl_weight
    else:
        loss = l1_loss

    return loss, loss_dict


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

        policy.save_pretrained(ckpt_dir)
        preprocessor.save_pretrained(ckpt_dir)
        postprocessor.save_pretrained(ckpt_dir)

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
    """Compute val/l1_loss + val/action_mse_norm on ``max_batches`` batches.

    We deliberately do NOT call ``policy.forward`` here because ACT's forward
    pass requires the VAE encoder's latent outputs to compute the KLD term,
    and the VAE encoder is only run when ``policy.training`` is True.  In
    ``.eval()`` the VAE head short-circuits to ``(None, None)`` which trips
    ``1 + None`` in the KLD formula (see modeling_act.py:155).

    Using ``predict_action_chunk`` instead gives us a clean, deterministic
    forward through the transformer decoder.  We then compute L1 and MSE
    manually against the normalized ground-truth chunk, masking out padded
    positions with ``action_is_pad`` so short trajectories don't skew the
    metric.

    * val/l1_loss         -- same objective ACT trains on (ignoring KLD),
                             directly comparable to train/l1_loss.
    * val/action_mse_norm -- MSE in the normalized action space, same metric
                             we tracked for VQ-BeT, so wandb plots line up.
    """
    policy.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = preprocessor(batch)

        with autocast_ctx():
            pred = policy.predict_action_chunk(batch)  # (B, chunk, A), normalized
        # Cast back to float32 for metric computation (autocast may emit bf16).
        pred = pred.float()
        target = batch["action"].float()

        if "action_is_pad" in batch:
            mask = (~batch["action_is_pad"]).unsqueeze(-1).float()
            denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
            l1 = ((pred - target).abs() * mask).sum() / denom
            mse = ((pred - target).pow(2) * mask).sum() / denom
        else:
            l1 = (pred - target).abs().mean()
            mse = (pred - target).pow(2).mean()

        totals["val/l1_loss"] = totals.get("val/l1_loss", 0.0) + float(l1.item())
        counts["val/l1_loss"] = counts.get("val/l1_loss", 0) + 1
        totals["val/action_mse_norm"] = (
            totals.get("val/action_mse_norm", 0.0) + float(mse.item())
        )
        counts["val/action_mse_norm"] = counts.get("val/action_mse_norm", 0) + 1

    # predict_action_chunk doesn't mutate the internal action queue (it rebuilds
    # it), but select_action's queue may have leftover entries.  Reset to be safe
    # in case a downstream callback calls select_action later.
    policy.reset()
    policy.train()
    return {k: totals[k] / max(counts[k], 1) for k in totals}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_act_config(policy_overrides: dict, device: str) -> ACTConfig:
    cfg = ACTConfig(**policy_overrides)
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
    return wandb.init(
        project=wb_cfg.get("project"),
        entity=wb_cfg.get("entity"),
        name=run_name,
        dir=str(run_dir),
        tags=wb_cfg.get("tags") or None,
        config=cfg,
        save_code=False,
    )


def train(cfg: dict) -> None:
    # --- Run dir ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or f"act_sfp_{timestamp}"
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
    policy_cfg = build_act_config(cfg["policy"], device=cfg["device"])

    # --- Datasets ---
    train_ds, val_ds = make_datasets(cfg, policy_cfg)
    log.info(
        "Train: %d ep / %d frames | Val: %d ep / %d frames",
        train_ds.num_episodes, train_ds.num_frames, val_ds.num_episodes, val_ds.num_frames,
    )

    # --- Policy + processors ---
    policy = make_policy(cfg=policy_cfg, ds_meta=train_ds.meta)
    policy.to(device)
    policy.train()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        dataset_stats=train_ds.meta.stats,
    )

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info("Policy params: total=%.1fM trainable=%.1fM", n_total / 1e6, n_trainable / 1e6)

    # --- Optimizer (ACTConfig only exposes AdamW) ---
    optimizer = policy_cfg.get_optimizer_preset().build(policy.get_optim_params())
    sched_preset = policy_cfg.get_scheduler_preset()
    lr_scheduler = sched_preset.build(optimizer, cfg["training"]["steps"]) if sched_preset else None

    # --- Augmenter (train-only) ---
    aug = ImageAugmenter(
        cfg.get("augmentation") or {},
        image_keys=list(policy_cfg.image_features),
    )
    if aug.enabled:
        log.info("Train augmentation: ColorJitter (p=%g) + RandomErasing on %s",
                 aug._cj_p, list(policy_cfg.image_features))

    # --- Action-weighted L1 (v3 intervention; see ActionWeighter docstring) ---
    action_weighter = ActionWeighter(cfg.get("action_weighting"))
    if action_weighter.enabled:
        log.info(
            "Action-weighted L1 loss ENABLED: scale=%g floor=%g ceil=%g",
            action_weighter.scale, action_weighter.floor, action_weighter.ceil,
        )

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

    # --- AMP ---
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

    log.info(
        "Training ACT for %d steps | batch=%d | lr=%g | weight_decay=%g | cams=%d",
        total_steps, cfg["training"]["batch_size"],
        policy_cfg.optimizer_lr, policy_cfg.optimizer_weight_decay,
        len(policy_cfg.image_features),
    )

    # --- Loop ---
    step = 0
    running = {"loss": 0.0, "grad_norm": 0.0, "count": 0, "dl_s": 0.0, "step_s": 0.0}
    t_start = time.perf_counter()

    while step < total_steps:
        t_dl0 = time.perf_counter()
        batch = next(dl_iter)
        # Augment BEFORE the preprocessor so [0,1]-space transforms make sense.
        batch = aug(batch)
        # Action weights are computed from RAW (un-normalized) actions, so
        # "zero motion" means exactly ||action||=0.  The preprocessor below
        # would shift the action mean and ruin that semantic, so we snapshot
        # the weights here.
        action_weights = action_weighter.weights_from_raw_batch(batch)
        batch = preprocessor(batch)
        if action_weights is not None:
            action_weights = action_weights.to(batch[ACTION].device)
        t_dl = time.perf_counter() - t_dl0

        t_step0 = time.perf_counter()
        with autocast_ctx():
            if action_weighter.enabled:
                loss, info = weighted_act_forward(policy, batch, action_weights)
            else:
                loss, info = policy.forward(batch)

        loss.backward()
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
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
            lr = optimizer.param_groups[0]["lr"]
            scalar = {
                "train/loss": running["loss"] / n,
                "train/grad_norm": running["grad_norm"] / n,
                "train/lr": lr,
                "train/dataloading_s": running["dl_s"] / n,
                "train/step_s": running["step_s"] / n,
                "train/samples_per_s": cfg["training"]["batch_size"] * n / max(running["step_s"], 1e-9),
                "step": step,
            }
            for k, v in info.items():
                if torch.is_tensor(v):
                    v = float(v.item())
                scalar[f"train/{k}"] = float(v)

            log.info(
                "step %d/%d | loss=%.4f | grad=%.2f | lr=%.2e | dl=%.3fs | step=%.3fs",
                step, total_steps,
                scalar["train/loss"], scalar["train/grad_norm"], lr,
                scalar["train/dataloading_s"], scalar["train/step_s"],
            )
            if wandb_run is not None:
                wandb_run.log(scalar, step=step)
            running = {k: 0.0 for k in running}
            running["count"] = 0

        do_eval = eval_freq > 0 and (step % eval_freq == 0 or step == total_steps)
        do_save = save_freq > 0 and (step % save_freq == 0 or step == total_steps)

        val_metrics: dict[str, float] = {}
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
            # We rank checkpoints by val/l1_loss because it is the actual
            # training objective (minus the CVAE KLD).  val/loss includes
            # kl_weight * kld which is an additional regularizer term.
            primary_metric = val_metrics.get("val/l1_loss", val_metrics.get("val/loss"))
            if primary_metric is not None:
                val_loss = float(primary_metric)
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
    log.info(
        "Training complete. %d steps in %.1fs (%.2f steps/s).",
        total_steps, elapsed, total_steps / max(elapsed, 1e-9),
    )

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
