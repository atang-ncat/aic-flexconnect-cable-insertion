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
    * **GPU-side augmentation**: the batch is moved to ``device`` immediately
      after ``next(dl_iter)``, so ColorJitter / RandomErasing and the mean-std
      Normalizer all run on the GPU.  This frees the CPU dataloader workers
      to do only the thing they're good at (video decode -> uint8 tensor), and
      removes the CPU-aug contention we observed when running multiple ACT
      jobs in parallel (load avg 300+ with 4 jobs, ~60 with GPU aug).
    * ``best`` symlink tracks the lowest ``val/l1_loss`` we have observed.
    * ``run_dir_collision`` (default ``auto``): if ``output_dir/run_name`` already
      contains a prior run (``train.log`` or ``checkpoints/``), a new suffix
      ``_{timestamp}`` is used so two jobs with the same ``run_name`` do not
      overwrite checkpoints.  Use ``error`` to fail fast, or ``overwrite`` to
      allow reuse (not safe for parallel runs).

Typical launch (from the workspace root):

    ./scripts/launch_train_act_ft_v1.sh   # F/T dataset v1 on GPU 1 (see script for CUDA_DEVICE_ID)

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

from torch.utils.data import Subset  # noqa: E402

from lerobot.configs.types import FeatureType, NormalizationMode  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

# Local import; sibling file in this same scripts/ directory.
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_discrete_act import DiscreteACTPolicy  # noqa: E402

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


def build_delta_timestamps(
    fps: int,
    policy_cfg: ACTConfig,
    include_prev_action: bool = False,
) -> dict[str, list[float]]:
    """LeRobotDataset delta_timestamps for ACT.

    ACT has ``observation_delta_indices = None`` (single-frame obs) and
    ``action_delta_indices = [0, 1, ..., chunk_size-1]``.  Returning a dict
    with only ``action`` means observations stay scalar-in-time (shape
    (B, C, H, W) rather than (B, 1, C, H, W)).

    v8 prev_action: prepend a single ``-1/fps`` past step to the action
    delta list, so each batch carries ``action[t-1]`` as the 0-th slot
    of its action tensor.  The training loop splits this off and concats
    it into ``observation.state`` before the policy sees the batch.  At
    episode start LeRobot marks ``action_is_pad[:, 0] = True`` because
    the negative delta falls before episode 0.
    """
    dt = 1.0 / fps
    idxs = list(policy_cfg.action_delta_indices)
    if include_prev_action:
        idxs = [-1] + idxs
    return {"action": [i * dt for i in idxs]}


def select_policy_features(
    ds_meta,
    camera_keys: list[str],
    state_keep_idx: list[int] | None = None,
    prev_action_dim: int = 0,
) -> tuple[dict, dict]:
    """Filter dataset features to the cameras requested by ``camera_keys``.

    ACT happily accepts multi-image inputs, but we still explicitly trim to
    the cameras listed in the config so that later cam-ablation runs don't
    re-introduce dropped views via ``make_policy``'s auto-inference.

    If ``state_keep_idx`` is provided, the ``observation.state`` feature's
    shape is rewritten to ``(len(state_keep_idx),)`` so the policy builds a
    state projection sized for the filtered state.  The actual channel
    selection on each batch happens via ``_apply_state_filter_to_batch`` in
    the train/eval loops.

    If ``prev_action_dim > 0`` (v8 prev_action feature), further extend the
    state shape by that many channels.  The additional channels are populated
    at batch time by ``apply_prev_action_to_batch`` from ``action[t-1]``,
    concatenated onto the already state-filtered observation.state.  ACT
    builds a single ``Linear(state_dim, dim_model)`` projection for the
    state token, so widening that vector is the cleanest way to give the
    policy an extra input without modifying the encoder architecture.
    """
    from dataclasses import replace

    all_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: v for k, v in all_features.items() if v.type is FeatureType.ACTION}
    input_features: dict = {}
    for k, v in all_features.items():
        if k in output_features:
            continue
        if v.type is FeatureType.VISUAL and k not in camera_keys:
            continue
        # Rewrite observation.state shape for state-filter / prev_action.
        if k == "observation.state":
            base = len(state_keep_idx) if state_keep_idx is not None else v.shape[0]
            new_dim = base + int(prev_action_dim)
            if new_dim != v.shape[0]:
                v = replace(v, shape=(new_dim,))
        input_features[k] = v
    missing = [c for c in camera_keys if c not in input_features]
    if missing:
        raise ValueError(f"Requested cameras not found in dataset: {missing}")
    return input_features, output_features


def compute_state_keep_indices(
    ds_meta,
    drop_prefixes: list[str] | None,
    keep_names: list[str] | None = None,
) -> list[int] | None:
    """Return the indices of ``observation.state`` columns to keep.

    Priority: if ``keep_names`` is given, keep exactly those (error on miss).
    Otherwise, if ``drop_prefixes`` is non-empty, drop any column whose name
    starts with one of those prefixes.  Returns ``None`` to indicate "no
    filtering" (keep all channels), so upstream code can short-circuit.

    The raw teleop dataset declares 26 state columns; we strip
    ``tcp_velocity.*`` (cols 7-12) and ``tcp_error.*`` (cols 13-18) because
    they are near-copies of the action target (validated with corr >=0.97
    for ``tcp_velocity.angular.x`` vs ``action.angular.x``).  Leaving them
    in means the policy can achieve val/l1 matching a 157-param OLS from
    state alone -- i.e. it never needs to learn vision at all.
    """
    state_feat = ds_meta.features.get("observation.state")
    if state_feat is None:
        return None
    names = state_feat.get("names")
    if not names:
        if drop_prefixes or keep_names:
            raise ValueError(
                "dataset lacks observation.state 'names' metadata but a "
                "state filter was requested; cannot resolve indices."
            )
        return None

    if keep_names is not None:
        idx: list[int] = []
        missing: list[str] = []
        for want in keep_names:
            if want in names:
                idx.append(names.index(want))
            else:
                missing.append(want)
        if missing:
            raise ValueError(f"state_keep_names not found in dataset: {missing}")
        return idx

    drop_prefixes = drop_prefixes or []
    if not drop_prefixes:
        return None
    idx = [i for i, n in enumerate(names) if not any(n.startswith(p) for p in drop_prefixes)]
    if len(idx) == len(names):
        return None  # nothing actually dropped
    return idx


def filter_state_stats(stats: dict, keep_idx: list[int]) -> dict:
    """Return a new stats dict with ``observation.state`` channels sliced.

    Stats dicts arrive with ``min/max/mean/std/count/q01/q10/q50/q90/q99``
    per feature.  We slice every per-channel array, leave scalars alone.
    """
    if "observation.state" not in stats:
        return stats
    out = {k: v for k, v in stats.items()}
    src = stats["observation.state"]
    new: dict = {}
    idx_np = np.asarray(keep_idx, dtype=np.int64)
    for sk, sv in src.items():
        arr = np.asarray(sv)
        if arr.ndim >= 1 and arr.shape[0] == len(src["mean"]):
            new[sk] = arr[idx_np]
        else:
            new[sk] = arr
    out["observation.state"] = new
    return out


def apply_state_filter_to_batch(
    batch: dict, keep_idx: torch.Tensor | None
) -> dict:
    """Slice ``observation.state`` on the LAST axis to the kept indices.

    Handles both ``(B, D)`` and ``(B, T, D)`` shapes.  No-op when
    ``keep_idx`` is None.  Returns the same batch dict (mutated) for
    convenience.
    """
    if keep_idx is None:
        return batch
    key = "observation.state"
    if key not in batch:
        return batch
    s = batch[key]
    # keep_idx lives on the same device as s for efficient index_select.
    if keep_idx.device != s.device:
        keep_idx = keep_idx.to(s.device)
    batch[key] = s.index_select(dim=-1, index=keep_idx)
    return batch


def apply_prev_action_to_batch(batch: dict, enabled: bool) -> dict:
    """Pop ``action[t-1]`` off the front of the action chunk and concat it
    into ``observation.state`` (v8 prev_action input).

    Precondition
    ------------
    ``make_datasets`` was called with ``include_prev_action=True``, so the
    dataset's ``delta_timestamps["action"]`` includes a ``-1`` prepended
    index and each batch carries an action tensor of shape
    ``(B, chunk_size + 1, A)`` rather than the usual ``(B, chunk_size, A)``.

    Transform
    ---------
    * ``prev_action = action[:, 0, :]``  (B, A)
    * Zero ``prev_action`` on rows where ``action_is_pad[:, 0]`` is True
      (LeRobot marks the slot as pad when the -1 offset falls before the
      episode start).  This matches how the OLS(prev_a) baseline was
      computed: zero-fill at episode boundaries.
    * Truncate ``batch["action"]`` and ``batch["action_is_pad"]`` back to
      their nominal chunk-only shapes ``(B, chunk_size, A)`` / ``(B, chunk_size)``
      so the policy's head, loss, and eval metrics see the same shapes
      they did pre-v8.
    * Append ``prev_action`` to the end of ``observation.state`` on the
      last axis.  The state projection was widened by ``A`` dims in
      ``select_policy_features`` so it can absorb this.

    Ordering matters: run this AFTER state filtering (so the leaky
    state channels are gone before prev_action is appended) and BEFORE
    the preprocessor (so prev_action gets normalized with state stats).

    No-op when ``enabled=False`` or ``action`` has only the chunk size.
    """
    if not enabled:
        return batch
    action = batch.get("action")
    if action is None or action.dim() != 3 or action.shape[1] < 2:
        return batch
    is_pad = batch.get("action_is_pad")
    prev_action = action[:, 0, :]  # (B, A)
    if is_pad is not None:
        prev_is_pad = is_pad[:, 0].unsqueeze(-1)  # (B, 1)
        prev_action = torch.where(
            prev_is_pad, torch.zeros_like(prev_action), prev_action
        )
        batch["action_is_pad"] = is_pad[:, 1:].contiguous()
    batch["action"] = action[:, 1:, :].contiguous()
    state = batch.get("observation.state")
    if state is not None:
        batch["observation.state"] = torch.cat([state, prev_action], dim=-1)
    return batch


def extend_state_stats_with_prev_action(stats: dict, action_dim: int) -> dict:
    """Widen ``observation.state`` stats by ``action_dim`` trailing channels
    populated from ``action`` stats.

    Called once at train-time setup when ``previous_action.enable=true``.
    After ``filter_state_stats`` has reduced state stats to ``D`` rows,
    we append ``action_dim`` more rows so the NormalizerProcessorStep's
    state mean/std vectors line up with the post-concat tensor shape.
    We reuse the action stats as a reasonable proxy for prev_action stats
    (prev_action values are drawn from the same action distribution, just
    shifted back one frame).
    """
    if action_dim <= 0:
        return stats
    act = stats.get("action")
    state = stats.get("observation.state")
    if act is None or state is None:
        return stats
    out_state: dict = {}
    for k, v in state.items():
        arr = np.asarray(v)
        if k == "count":
            out_state[k] = v  # unchanged
            continue
        av = act.get(k)
        if av is None or arr.ndim < 1:
            out_state[k] = v
            continue
        av_arr = np.asarray(av)
        if av_arr.shape[0] != action_dim:
            out_state[k] = v
            continue
        out_state[k] = np.concatenate([arr, av_arr.astype(arr.dtype)], axis=0)
    return {**stats, "observation.state": out_state}


def compute_train_only_stats(
    train_ds: LeRobotDataset,
    baseline_stats: dict,
    feature_keys: Iterable[str] = ("observation.state", "action"),
) -> dict:
    """Recompute stats for ``feature_keys`` over the TRAIN episodes only.

    Why: ``train_ds.meta.stats`` is loaded from ``meta/stats.json`` which
    was computed at dataset-build time over ALL 151 episodes, including
    the 15 we hold out for val.  The normalizer ends up using val-set
    moments, a real if mild leak.

    We only recompute per-frame continuous features (state, action); video
    stats stay on baseline_stats unchanged because recomputing them would
    cost hours of video decode, and image normalization in ACT is
    ImageNet-constant anyway (see ACTConfig.normalization_mapping).
    """
    out = {k: v for k, v in baseline_stats.items()}
    for key in feature_keys:
        if key not in baseline_stats:
            continue
        col = train_ds.hf_dataset[key]
        arr = np.stack([np.asarray(x, dtype=np.float64) for x in col])
        new: dict = {
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0).clip(min=1e-8),
            "count": np.asarray([len(arr)], dtype=np.int64),
        }
        for q, qv in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
            new[qv] = np.quantile(arr, q, axis=0)
        out[key] = new
    return out


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


def trim_episode_tails(
    ds: LeRobotDataset, tail_frac: float, min_keep: int = 1
) -> Subset:
    """Return a ``Subset`` of ``ds`` that drops the last ``tail_frac`` of frames
    of each episode.

    Why: the audit (``scripts/audit_teleop_data.py``) showed that ~60% of frames
    are near-idle, and most of those idle frames cluster at the END of each
    episode (post-insertion stillness after the operator stopped driving).
    Training through those frames biases the model toward "predict zero" and
    inflates the effective training set without adding useful signal.

    We filter at the FRAME-INPUT level, not at the action-target level -- an
    unfiltered frame whose chunk extends into the trimmed tail still uses the
    tail's actions as targets (LeRobotDataset handles that via its delta-index
    padding).  That's intentional: we want the model to still learn "hold
    still" once insertion is done, just not be dominated by it.

    ``min_keep`` guards against degenerate-short episodes (e.g. teleop aborted
    early): we keep at least ``min_keep`` frames per episode no matter what.
    """
    if tail_frac <= 0.0:
        return Subset(ds, list(range(len(ds))))
    ep_idx = np.asarray(ds.hf_dataset["episode_index"])
    kept: list[int] = []
    dropped = 0
    unique_eps = np.unique(ep_idx)
    for e in unique_eps:
        frame_positions = np.nonzero(ep_idx == e)[0]
        n = len(frame_positions)
        keep_n = max(min_keep, int(round(n * (1.0 - tail_frac))))
        kept.extend(frame_positions[:keep_n].tolist())
        dropped += n - keep_n
    log.info(
        "Boundary trim: tail_frac=%.2f -> kept %d / %d frames (dropped %d, "
        "~%.1f%% of %d episodes)",
        tail_frac, len(kept), len(ds), dropped, 100.0 * dropped / max(len(ds), 1),
        len(unique_eps),
    )
    return Subset(ds, kept)


def make_datasets(
    cfg: dict, policy_cfg: ACTConfig
) -> tuple[LeRobotDataset, LeRobotDataset, list[int] | None, bool]:
    """Build train/val datasets and return them along with:
    * ``state_keep_idx``: indices of observation.state channels to keep
      (or ``None`` if no state filter).
    * ``use_prev_action``: True when ``dataset.previous_action.enable: true``;
      each batch's action tensor will carry an extra past step at slot 0
      that the training/eval loops pop off via ``apply_prev_action_to_batch``.

    The caller threads both flags through the train/eval loops.
    """
    ds_cfg = cfg["dataset"]
    prev_cfg = ds_cfg.get("previous_action") or {}
    use_prev_action = bool(prev_cfg.get("enable", False))

    probe = LeRobotDataset(
        repo_id=ds_cfg["repo_id"],
        root=ds_cfg["root"],
        video_backend=ds_cfg.get("video_backend", "pyav"),
    )
    fps = probe.fps
    n_episodes = probe.num_episodes
    state_keep_idx = compute_state_keep_indices(
        probe.meta,
        drop_prefixes=ds_cfg.get("state_drop_prefixes"),
        keep_names=ds_cfg.get("state_keep_names"),
    )
    if state_keep_idx is not None:
        names = probe.meta.features["observation.state"].get("names") or []
        kept = [names[i] for i in state_keep_idx]
        dropped = [n for n in names if n not in set(kept)]
        log.info(
            "State filter: keeping %d/%d columns -- kept=%s dropped=%s",
            len(kept), len(names), kept, dropped,
        )

    action_dim = int(probe.meta.features["action"]["shape"][0])
    prev_action_dim = action_dim if use_prev_action else 0
    if use_prev_action:
        log.info(
            "Previous-action input ENABLED: extending observation.state by %d dims "
            "(action[t-1], zero-filled at episode boundary).",
            prev_action_dim,
        )

    input_features, output_features = select_policy_features(
        probe.meta,
        ds_cfg["cameras"],
        state_keep_idx=state_keep_idx,
        prev_action_dim=prev_action_dim,
    )
    policy_cfg.input_features = input_features
    policy_cfg.output_features = output_features
    del probe

    train_eps, val_eps = split_episodes(n_episodes, ds_cfg["val_num_episodes"], cfg["seed"])
    log.info(
        "Split %d episodes -> %d train / %d val (seed=%d)",
        n_episodes, len(train_eps), len(val_eps), cfg["seed"],
    )

    delta_ts = build_delta_timestamps(
        fps, policy_cfg, include_prev_action=use_prev_action
    )

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

    return train_ds, val_ds, state_keep_idx, use_prev_action


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
            # ColorJitter is still expensive (per-pixel); gate it behind an
            # independent Bernoulli per batch.  RandomErasing is cheap (a few
            # rectangles) so we always call it and let its own ``p`` sample
            # internally.  Both transforms dispatch to the tensor's device
            # (GPU or CPU) -- no explicit .to() needed.
            if self._cj_p > 0 and torch.rand(()) < self._cj_p:
                img = self._cj(img)
            img = self._re(img)
            if img.shape != orig_shape:
                img = img.reshape(orig_shape)
            batch[k] = img
        return batch


def batch_to_device(
    batch: dict, device: torch.device, non_blocking: bool = True
) -> dict:
    """Move every tensor in a LeRobot batch dict to ``device``.

    Non-tensor entries (e.g. string task keys) are passed through.  ``non_blocking``
    is a no-op unless the source tensors live in pinned memory -- in our setup
    they do (``pin_memory=True`` in the dataloader), so this overlaps H->D transfer
    with the previous step's GPU work.
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


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
        # v5 fix: divide by valid-element count, not B*T*A.  Upstream's
        # ``(err * pad_mask).mean()`` divides by the total tensor size and
        # silently shrinks the loss for pad-heavy chunks (end-of-episode
        # samples with ``chunk_size=100`` pad ~12% of positions).  That
        # mis-scales the GRADIENT on those chunks and makes
        # weighted/unweighted comparisons cross-cell dishonest.  The logged
        # ``train/l1_loss`` diagnostic at line ~476 already uses this
        # valid-count denominator; now the optimizer does too.
        l1_loss = (err * pad_mask).sum() / pad_mask.sum().clamp_min(1.0) / err.shape[-1]
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
            # When we promote a new best, the previous best_dir stops being
            # special.  If it's outside the keep_last_n window, delete it now
            # so we don't leak one ckpt dir per "new-best" event over a long
            # run (seen in v4: aggressive save-on-best without this cleanup
            # would bloat ~200MB per event).
            prev_best = self.best_dir
            self.best_val = val_loss
            self.best_dir = ckpt_dir
            best_link = self.root / "checkpoints" / "best"
            best_link.unlink(missing_ok=True)
            try:
                best_link.symlink_to(tag, target_is_directory=True)
            except OSError:
                pass
            if (
                prev_best is not None
                and prev_best != ckpt_dir
                and prev_best not in self.recent
                and prev_best.exists()
            ):
                shutil.rmtree(prev_best, ignore_errors=True)

        return ckpt_dir


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopper:
    """Patience-based early stopping on a configurable val metric.

    v2/v3 runs showed that ACT's val/l1 plateaus well before the 50k-step
    schedule completes (best usually in the 20k-30k range) and then mildly
    overfits for the rest of training.  Burning compute past that plateau
    only yielded regressions, so v4 introduces early stopping -- let each
    run decide its own budget instead of forcing a fixed wall-clock.

    v6 generalizes this stopper to an arbitrary metric + direction.  v5_cls
    motivated the change: training-time ``val/l1_loss`` plateaued at step
    1000 (0.00431) but ``val/motion_accuracy`` kept climbing through step
    10000.  Stopping on l1 threw away 3x of motion-accuracy headroom.

    Rules:
      * ``enabled`` off   -> stopper is a no-op (matches legacy v2/v3 behavior).
      * ``enabled`` on    -> once ``step >= min_steps``, stop when we've gone
        ``patience_steps`` consecutive training steps since we last saw an
        improvement of at least ``min_delta`` (absolute) in the tracked
        metric.
      * Direction is set via ``mode``: ``"minimize"`` (default; e.g., l1,
        mse, cross-entropy) or ``"maximize"`` (e.g., motion_accuracy,
        accuracy, task success).

    ``is_improvement`` is exposed separately (without the min_delta gate) so
    the checkpoint bookkeeper can promote on any strict improvement while
    the stopper itself waits for a min_delta improvement before resetting
    patience.
    """

    enabled: bool = False
    patience_steps: int = 5_000
    min_delta: float = 0.0005
    min_steps: int = 10_000
    metric: str = "val/l1_loss"
    mode: str = "minimize"  # or "maximize"
    best: float = field(init=False)
    best_step: int = 0
    last_improve_step: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("minimize", "maximize"):
            raise ValueError(f"early_stopping.mode must be minimize|maximize, got {self.mode!r}")
        self.best = float("inf") if self.mode == "minimize" else float("-inf")

    # ---- comparison helpers --------------------------------------------
    def _is_strict_improvement(self, val: float) -> bool:
        return val < self.best if self.mode == "minimize" else val > self.best

    def _is_delta_improvement(self, val: float) -> bool:
        if self.mode == "minimize":
            return val + self.min_delta < self.best
        return val - self.min_delta > self.best

    def is_improvement(self, val: float | None) -> bool:
        """Strict improvement test used by the checkpoint bookkeeper."""
        if val is None:
            return False
        return self._is_strict_improvement(val)

    # ---- main stopper hook ---------------------------------------------
    def update(self, step: int, value: float | None) -> bool:
        """Return ``True`` iff training should stop at / after ``step``.

        Rules as of v6:
          * Any STRICT improvement updates ``self.best`` + ``best_step``.
          * Only a MIN_DELTA improvement (vs. the PRIOR best) resets the
            patience counter.

        v6 bugfix: in the first v6 run the two checks shared ``self.best``
        state -- by the time ``_is_delta_improvement`` ran, ``self.best``
        had already been set to ``value`` by the strict-improvement branch,
        so the delta check always read ``value - min_delta > value`` and
        ``last_improve_step`` never advanced.  That caused the stopper to
        fire the instant ``step >= min_steps``.  We now snapshot the prior
        best and run BOTH checks against it, which is the correct
        semantics.
        """
        if value is None:
            return False
        prior_best = self.best
        # Strict improvement: promote best + best_step (used by the
        # checkpoint bookkeeper to advance the `best` symlink).
        if self._mode_cmp_strict(value, prior_best):
            self.best = value
            self.best_step = step
        # Delta improvement: reset patience.  Compared against PRIOR best.
        if self._mode_cmp_delta(value, prior_best) or step == 0:
            self.last_improve_step = step
        if not self.enabled:
            return False
        if step < self.min_steps:
            return False
        return (step - self.last_improve_step) >= self.patience_steps

    # Internal: comparison against an arbitrary reference (not self.best).
    def _mode_cmp_strict(self, val: float, ref: float) -> bool:
        return val < ref if self.mode == "minimize" else val > ref

    def _mode_cmp_delta(self, val: float, ref: float) -> bool:
        if self.mode == "minimize":
            return val + self.min_delta < ref
        return val - self.min_delta > ref


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    policy,
    preprocessor,
    postprocessor,
    dataloader: DataLoader,
    max_batches: int | None,
    device: torch.device,
    autocast_ctx,
    state_keep_t: torch.Tensor | None = None,
    action_dim_names: list[str] | None = None,
    use_prev_action: bool = False,
) -> dict[str, float]:
    """Compute val/l1_loss + val/action_mse_norm on (up to) ``max_batches`` batches.

    Pass ``max_batches=None`` (or ``<=0``) to iterate the FULL val loader.
    v5 default is full-val so that run-to-run comparisons aren't biased by
    whichever 1600 samples happen to sort first.

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

    v6 pooling fix
    --------------
    Pre-v6 this function averaged per-batch fractions unweighted (``sum(frac)
    / n_batches``), which is a **mean-of-means** estimator.  For metrics
    whose denominator varies across batches (in particular
    ``val/motion_accuracy``: motion frames are sparse, so per-batch fractions
    have very different denominators), that underestimates the pooled number
    by up to 2x.  We confirmed this on v5_cls:

        training-time val/motion_accuracy @ step 10000 = 0.2650  (mean-of-means)
        pooled physical-unit cross-eval  @ step 10000 = 0.5690  (correct)

    v6 accumulates numerator and denominator globally per metric and divides
    once at the end.

    * val/l1_loss         -- same objective ACT trains on (ignoring KLD).
    * val/action_mse_norm -- MSE in the normalized action space.
    * val/accuracy        -- fraction of (dim, pos) slots where the argmax
                             class matches the target class (discrete head).
    * val/motion_accuracy -- same, restricted to positions whose target is
                             non-idle.  THIS is the signal for discrete runs.
    """
    policy.eval()
    # (numerator, denominator) accumulator per metric.
    acc: dict[str, tuple[float, float]] = {}

    def _add(key: str, num: float, den: float) -> None:
        n, d = acc.get(key, (0.0, 0.0))
        acc[key] = (n + num, d + den)

    use_full = max_batches is None or max_batches <= 0
    for i, batch in enumerate(dataloader):
        if not use_full and i >= max_batches:
            break
        # Mirror the training loop: move to GPU before preprocessor so the
        # normalizer runs on device.
        batch = batch_to_device(batch, device)
        batch = apply_state_filter_to_batch(batch, state_keep_t)
        # v8: teacher-forced eval protocol -- always consume the ground-truth
        # action[t-1] off the dataset's action tensor.  See the train-loop
        # comment for the exposure-bias caveat.
        batch = apply_prev_action_to_batch(batch, use_prev_action)
        batch = preprocessor(batch)

        with autocast_ctx():
            pred = policy.predict_action_chunk(batch)  # (B, chunk, A), normalized
        # Cast back to float32 for metric computation (autocast may emit bf16).
        pred = pred.float()
        target = batch["action"].float()

        A = pred.shape[-1]
        if "action_is_pad" in batch:
            mask = (~batch["action_is_pad"]).unsqueeze(-1).float()  # (B, T, 1)
            n_valid = float(mask.sum().item())  # positions (not * A)
        else:
            mask = torch.ones_like(pred[..., :1])
            n_valid = float(pred.numel() / A)

        # l1 / mse denominators are n_valid * A (one sample per (pos, dim)).
        _add("val/l1_loss",
             float(((pred - target).abs() * mask).sum().item()),
             n_valid * A)
        _add("val/action_mse_norm",
             float(((pred - target).pow(2) * mask).sum().item()),
             n_valid * A)

        # Classification-head diagnostics.  We key on ``n_classes`` rather
        # than isinstance(DiscreteACTPolicy) so this works if the policy is
        # wrapped/compiled.
        if hasattr(policy, "n_classes"):
            from modeling_discrete_act import snap_to_class

            target_cls = snap_to_class(target)
            pred_cls = snap_to_class(pred)
            correct = (target_cls == pred_cls).float()  # (B, T, A)
            # per-(pos, dim) accuracy over all non-pad positions.
            _add("val/accuracy",
                 float((correct * mask).sum().item()),
                 n_valid * A)
            motion_cells = (target_cls != 1).float() * mask  # (B, T, A)
            motion_denom = float(motion_cells.sum().item())
            _add("val/motion_accuracy",
                 float((correct * motion_cells).sum().item()),
                 motion_denom)

            # Per-dim breakdown: extremely useful for locating which action
            # axis the policy is failing on.  For sfp teleop, angular.y is
            # ~99.96% zero in the whole dataset, so val/motion_accuracy/ang.y
            # is typically undefined (NaN) or trivially 1.0, while the other
            # five dims carry the real signal.  Each dim contributes n_valid
            # positions to the overall accuracy and a variable number of
            # motion positions.
            names = action_dim_names if action_dim_names is not None else [f"dim{a}" for a in range(A)]
            for a in range(A):
                n = names[a] if a < len(names) else f"dim{a}"
                corr_a = correct[..., a] * mask.squeeze(-1)  # (B, T)
                _add(f"val/accuracy/{n}",
                     float(corr_a.sum().item()),
                     n_valid)
                mot_a = motion_cells[..., a]  # (B, T)
                mot_denom_a = float(mot_a.sum().item())
                _add(f"val/motion_accuracy/{n}",
                     float((correct[..., a] * mot_a).sum().item()),
                     mot_denom_a)

    # predict_action_chunk doesn't mutate the internal action queue (it rebuilds
    # it), but select_action's queue may have leftover entries.  Reset to be safe
    # in case a downstream callback calls select_action later.
    policy.reset()
    policy.train()
    return {k: (num / den if den > 0 else 0.0) for k, (num, den) in acc.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_act_config(policy_overrides: dict, device: str) -> ACTConfig:
    # ``discrete_action`` is a custom flag consumed by ``train()``; pop it
    # before ACTConfig sees kwargs it doesn't understand.
    policy_overrides = dict(policy_overrides)
    policy_overrides.pop("discrete_action", None)
    # Allow the config to specify normalization_mapping as
    # ``{"VISUAL": "MEAN_STD", ...}`` (string-valued, so YAML-friendly); map
    # strings to the NormalizationMode enum here.
    nm = policy_overrides.get("normalization_mapping")
    if isinstance(nm, dict):
        policy_overrides["normalization_mapping"] = {
            k: (NormalizationMode[v] if isinstance(v, str) else v) for k, v in nm.items()
        }
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


def _run_dir_looks_in_use(path: Path) -> bool:
    """True if path exists and already holds a run (log or checkpoints)."""
    if not path.is_dir():
        return False
    return (path / "train.log").is_file() or (path / "checkpoints").is_dir()


def resolve_run_dir_and_name(
    output_dir: Path, run_name: str, timestamp: str, mode: str
) -> tuple[Path, str]:
    """Pick a unique run subdir so two jobs with the same run_name do not clobber.

    * ``auto`` (default): if ``output_dir/run_name`` is already a run, use
      ``{run_name}_{timestamp}``, then ``{run_name}_{timestamp}_2``, etc.
    * ``error``: raise if the directory is already in use.
    * ``overwrite``: use ``output_dir/run_name`` as before (can corrupt parallel runs).
    """
    m = (mode or "auto").lower()
    if m not in ("auto", "error", "overwrite"):
        raise ValueError(
            f"run_dir_collision must be auto|error|overwrite, got {mode!r}"
        )
    run_dir = output_dir / run_name
    if not _run_dir_looks_in_use(run_dir):
        return run_dir, run_name
    if m == "overwrite":
        log.warning(
            "run_dir_collision=overwrite: reusing in-use run dir %s (unsafe if another job is writing).",
            run_dir,
        )
        return run_dir, run_name
    if m == "error":
        raise FileExistsError(
            f"Run directory already in use: {run_dir} "
            f"(remove it, pick a different run_name, or set run_dir_collision: auto in the config)."
        )
    # auto: disambiguate with timestamp, then _2, _3, ...
    candidate = f"{run_name}_{timestamp}"
    run_dir = output_dir / candidate
    n = 2
    while _run_dir_looks_in_use(run_dir):
        candidate = f"{run_name}_{timestamp}_{n}"
        run_dir = output_dir / candidate
        n += 1
    log.info(
        "Run name %r was already in use; using %r to avoid clobbering outputs.",
        run_name,
        candidate,
    )
    return run_dir, candidate


def train(cfg: dict) -> None:
    # --- Run dir (avoid two concurrent jobs with the same run_name sharing one folder) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or f"act_sfp_{timestamp}"
    collision = str(cfg.get("run_dir_collision", "auto"))
    output_root = Path(cfg["output_dir"])
    run_dir, run_name = resolve_run_dir_and_name(
        output_root, run_name, timestamp, collision
    )
    cfg["run_name"] = run_name
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
    train_ds, val_ds, state_keep_idx, use_prev_action = make_datasets(cfg, policy_cfg)
    log.info(
        "Train: %d ep / %d frames | Val: %d ep / %d frames",
        train_ds.num_episodes, train_ds.num_frames, val_ds.num_episodes, val_ds.num_frames,
    )
    # Short labels for per-dim val metrics (wandb-friendly).  Falls back to
    # dim{i} if the dataset doesn't carry feature names.
    _raw_act_names = train_ds.meta.features.get("action", {}).get("names") or []
    action_dim_names = [
        n.replace("linear.", "lin.").replace("angular.", "ang.")
        for n in _raw_act_names
    ]

    # Materialize state_keep_idx as a tensor once; moved to device lazily per-batch.
    state_keep_t: torch.Tensor | None = (
        torch.as_tensor(state_keep_idx, dtype=torch.long)
        if state_keep_idx is not None
        else None
    )

    # --- Policy + processors ---
    #
    # Classification variant: bypass LeRobot's make_policy so we can
    # instantiate DiscreteACTPolicy with the config already filled in by
    # ``make_datasets``.  make_policy does some normalization-mapping
    # validation we still want, so we only branch when the flag is on.
    discrete_cfg = (cfg.get("policy") or {}).get("discrete_action") or {}
    use_discrete = bool(discrete_cfg.get("enable", False))
    if use_discrete:
        if policy_cfg.normalization_mapping.get(FeatureType.ACTION) is not NormalizationMode.IDENTITY:
            log.warning(
                "discrete_action is ON but normalization_mapping.ACTION=%s; "
                "forcing IDENTITY so the class-snap receives raw physical values.",
                policy_cfg.normalization_mapping.get(FeatureType.ACTION),
            )
            policy_cfg.normalization_mapping[FeatureType.ACTION] = NormalizationMode.IDENTITY
        policy = DiscreteACTPolicy(policy_cfg)
        log.info(
            "Using DiscreteACTPolicy: %d dims x %d classes (values={-0.1, 0, +0.1})",
            policy.action_dim, policy.n_classes,
        )
    else:
        policy = make_policy(cfg=policy_cfg, ds_meta=train_ds.meta)
    policy.to(device)
    policy.train()

    # --- Backbone freezing (v11 intervention for small datasets) ---
    # When optimizer_lr_backbone == 0, freeze all backbone parameters entirely.
    # This prevents the ImageNet-pretrained ResNet from overfitting on <50k frames
    # and removes ~11M params from the optimizer state (saves GPU memory too).
    if policy_cfg.optimizer_lr_backbone == 0:
        n_frozen = 0
        for n, p in policy.named_parameters():
            if n.startswith("model.backbone"):
                p.requires_grad_(False)
                n_frozen += p.numel()
        log.info(
            "Backbone FROZEN: set requires_grad=False on %d params (%.1fM) "
            "because optimizer_lr_backbone=0.",
            n_frozen, n_frozen / 1e6,
        )

    # Stats for the preprocessor:
    #   1. Baseline = train_ds.meta.stats (from meta/stats.json at build time).
    #   2. If dataset.stats_train_only, recompute state+action stats from
    #      train episodes only (closes a mild val-leak in normalization).
    #   3. If state_keep_idx is set, slice the state stats to the kept cols so
    #      the normalizer matches the filtered input shape.
    baseline_stats = train_ds.meta.stats
    if bool(cfg.get("dataset", {}).get("stats_train_only", False)):
        baseline_stats = compute_train_only_stats(train_ds, baseline_stats)
        log.info("Dataset stats recomputed on train episodes only.")
    if state_keep_idx is not None:
        baseline_stats = filter_state_stats(baseline_stats, state_keep_idx)
    if use_prev_action:
        action_dim = int(policy_cfg.action_feature.shape[0])
        baseline_stats = extend_state_stats_with_prev_action(baseline_stats, action_dim)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        dataset_stats=baseline_stats,
    )

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info("Policy params: total=%.1fM trainable=%.1fM", n_total / 1e6, n_trainable / 1e6)

    # --- Optimizer (ACTConfig only exposes AdamW) ---
    optimizer = policy_cfg.get_optimizer_preset().build(policy.get_optim_params())

    # --- LR scheduler (v11: cosine warmup+decay for small datasets) ---
    # ACTConfig.get_scheduler_preset() returns None (no scheduler at all).
    # When ``lr_schedule`` is present in the config, we build a warmup+cosine
    # schedule ourselves.  Backward-compatible: configs without ``lr_schedule``
    # get a constant LR as before.
    lr_scheduler = None
    lr_sched_cfg = cfg.get("lr_schedule")
    if lr_sched_cfg and lr_sched_cfg.get("enable", True):
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )
        warmup_steps = int(lr_sched_cfg.get("warmup_steps", 500))
        total = int(cfg["training"]["steps"])
        min_lr_ratio = float(lr_sched_cfg.get("min_lr_ratio", 0.01))
        # LinearLR ramps from start_factor → end_factor over total_iters.
        warmup = LinearLR(
            optimizer, start_factor=min_lr_ratio, end_factor=1.0,
            total_iters=warmup_steps,
        )
        # CosineAnnealingLR decays from current LR to eta_min over T_max.
        base_lr = policy_cfg.optimizer_lr
        cosine = CosineAnnealingLR(
            optimizer, T_max=total - warmup_steps,
            eta_min=base_lr * min_lr_ratio,
        )
        lr_scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        log.info(
            "LR schedule: linear warmup %d steps -> cosine decay to %.1e over %d steps",
            warmup_steps, base_lr * min_lr_ratio, total - warmup_steps,
        )
    else:
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
    if action_weighter.enabled and use_discrete:
        log.warning(
            "action_weighting.enable=True is incompatible with discrete_action "
            "(weights are defined on an L1 regression objective, not CE).  "
            "Disabling action weighting for this run."
        )
        action_weighter.enabled = False
    elif action_weighter.enabled:
        log.info(
            "Action-weighted L1 loss ENABLED: scale=%g floor=%g ceil=%g",
            action_weighter.scale, action_weighter.floor, action_weighter.ceil,
        )

    # --- Boundary-trim (v4 intervention; leaves val set untouched) ---
    tail_frac = float(cfg.get("dataset", {}).get("trim_tail_fraction", 0.0) or 0.0)
    train_src: Any = train_ds
    if tail_frac > 0.0:
        train_src = trim_episode_tails(train_ds, tail_frac=tail_frac)

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_src,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"] and device.type == "cuda",
        drop_last=True,
        prefetch_factor=cfg["training"]["prefetch_factor"] if cfg["training"]["num_workers"] > 0 else None,
        persistent_workers=cfg["training"]["num_workers"] > 0,
    )
    val_nw = max(2, cfg["training"]["num_workers"] // 2)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["evaluation"]["eval_batch_size"],
        shuffle=False,
        num_workers=val_nw,
        pin_memory=cfg["training"]["pin_memory"] and device.type == "cuda",
        drop_last=False,
        # v5: keep val workers alive between evals.  Without this we pay
        # ~15-30s of worker spin-up EVERY eval, which at eval_freq=1000
        # steps and eval_batches=null (full val) dominates wall-clock.
        persistent_workers=val_nw > 0,
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

    # --- Early stopping (v4 intervention; no-op if disabled) ---
    es_cfg = cfg.get("early_stopping") or {}
    # v6: metric + mode are configurable.  Default stays backward-compatible
    # with v2-v5 (minimize val/l1_loss).
    stopper = EarlyStopper(
        enabled=bool(es_cfg.get("enable", False)),
        patience_steps=int(es_cfg.get("patience_steps", 5_000)),
        min_delta=float(es_cfg.get("min_delta", 0.0005)),
        min_steps=int(es_cfg.get("min_steps", 10_000)),
        metric=str(es_cfg.get("metric", "val/l1_loss")),
        mode=str(es_cfg.get("mode", "minimize")),
    )
    # Keep the checkpoint bookkeeper's "best" comparison aligned with the
    # stopper's metric + direction so ``best`` symlink tracks the metric the
    # run actually cares about.
    ckpt.best_val = float("-inf") if stopper.mode == "maximize" else float("inf")
    if stopper.enabled:
        log.info(
            "Early stopping ENABLED: metric=%s (%s), patience=%d steps, min_delta=%.4f, min_steps=%d",
            stopper.metric, stopper.mode,
            stopper.patience_steps, stopper.min_delta, stopper.min_steps,
        )

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
        # Move the batch to the training device BEFORE aug / norm so both
        # ColorJitter and the mean-std Normalizer run on GPU (freeing the CPU
        # dataloader workers to do only video decode).  The Normalizer auto-
        # migrates its stats to match input device on first call, so we don't
        # need to .to(device) it explicitly -- see normalize_processor.py:317.
        batch = batch_to_device(batch, device)
        # v5: strip leaky state channels (tcp_velocity, tcp_error) BEFORE the
        # preprocessor and action-weighter see the state.  No-op if the
        # dataset.state_drop_prefixes config key is unset.
        batch = apply_state_filter_to_batch(batch, state_keep_t)
        # v8: pop the extra past-step off the action chunk and append it to
        # observation.state, zeroed at episode boundaries.  No-op when
        # previous_action.enable is false (action shape is already chunk_size).
        batch = apply_prev_action_to_batch(batch, use_prev_action)
        # Augment BEFORE the preprocessor so [0,1]-space transforms make sense.
        batch = aug(batch)
        # Action weights are computed from RAW (un-normalized) actions, so
        # "zero motion" means exactly ||action||=0.  The preprocessor below
        # would shift the action mean and ruin that semantic, so we snapshot
        # the weights here.
        action_weights = action_weighter.weights_from_raw_batch(batch)
        batch = preprocessor(batch)
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
            # v5: ``eval_batches: null`` (or <=0) runs the full val set.
            eval_batches_cfg = cfg["evaluation"].get("eval_batches")
            val_metrics = evaluate(
                policy, preprocessor, postprocessor,
                val_loader, eval_batches_cfg, device, autocast_ctx,
                state_keep_t=state_keep_t,
                action_dim_names=action_dim_names,
                use_prev_action=use_prev_action,
            )
            val_metrics["val/eval_s"] = time.perf_counter() - t0
            log.info("eval @ step %d: %s", step, {k: round(v, 5) for k, v in val_metrics.items()})
            if wandb_run is not None:
                wandb_run.log({**val_metrics, "step": step}, step=step)

        # Compute is_best on EVERY eval (not just on save_freq boundaries) so we
        # never miss a best ckpt that happens at a non-save step.  v4 runs hit
        # this exact bug: best val/l1 was at step 16000 but save_freq=5000, so
        # the `best` symlink pointed to step 20000 (a worse ckpt).
        #
        # v6: pull the primary metric from the stopper config so "best" tracks
        # the metric the run cares about (e.g. val/motion_accuracy maximized
        # for classification runs).
        val_loss = None
        is_best = False
        if do_eval:
            primary_metric = val_metrics.get(
                stopper.metric,
                val_metrics.get("val/l1_loss", val_metrics.get("val/loss")),
            )
            if primary_metric is not None:
                val_loss = float(primary_metric)
                is_best = stopper.is_improvement(val_loss)

        # Save whenever:
        #   a) we're on a save_freq boundary (periodic last-N retention), OR
        #   b) this eval is a new best (so `best` is always the real best).
        # The bookkeeper dedupes step_X directories, so saving on the same step
        # twice in a row is harmless.
        if do_save or is_best:
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
            log.info(
                "saved checkpoint: %s%s%s",
                path,
                " (new best)" if is_best else "",
                " (on-best only, not save_freq)" if is_best and not do_save else "",
            )

        # Early stopping is evaluated AFTER do_save so the best-checkpoint
        # symlink is always up-to-date when we exit the loop.  We feed the
        # stopper every eval (not every save), because eval_freq is typically
        # finer-grained than save_freq.
        if do_eval:
            primary = val_metrics.get(
                stopper.metric,
                val_metrics.get("val/l1_loss", val_metrics.get("val/loss")),
            )
            should_stop = stopper.update(step, float(primary) if primary is not None else None)
            if wandb_run is not None and stopper.enabled:
                wandb_run.log(
                    {
                        # v6: report under a neutral name so wandb panels don't
                        # assume l1 -- the stopper's metric is configurable.
                        "val/best_tracked": stopper.best,
                        "val/steps_since_improvement": step - stopper.last_improve_step,
                        "step": step,
                    },
                    step=step,
                )
            if should_stop:
                log.info(
                    "Early stop @ step %d: no improvement in %d steps "
                    "(best=%.5f @ step %d)",
                    step, step - stopper.last_improve_step,
                    stopper.best, stopper.best_step,
                )
                # Ensure we have a checkpoint saved at the stop point if we
                # haven't just saved one.  This makes the `last` symlink land
                # on the final state.
                if not do_save:
                    ckpt.save(
                        step=step,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        is_best=False,
                        val_loss=float(primary) if primary is not None else None,
                    )
                break

    elapsed = time.perf_counter() - t_start
    log.info(
        "Training complete. %d/%d steps in %.1fs (%.2f steps/s).",
        step, total_steps, elapsed, step / max(elapsed, 1e-9),
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
