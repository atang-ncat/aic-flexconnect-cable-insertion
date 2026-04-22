"""Discrete (3-class per dim) ACT policy head for the SFP teleop dataset.

Why this exists
---------------
Empirical audit of ``teleop-dataset/data/``:

    every action dim      takes exactly one of  {-0.1, 0, +0.1}
    ``angular.y``         is 99.9% zero (effectively a constant)
    59.5%                 of all frames are all-zero (idle)
    max deviation from    |k * 0.1|  for any frame is  0.0

This is not a continuous-control dataset. It's **6x3-way categorical** (really
5x3-way, because ``angular.y`` is a constant). L1 / MSE regression on it is a
spectacularly bad fit: the Bayes-optimal L1 predictor for a distribution that
is 90-99% at zero is just "output zero", which is what every ACT run we have
learned to do. Cross-evaluation in physical action units shows that a 157-
parameter OLS from state alone matches our 32M-parameter ACT to within 1%.

This module replaces ACT's per-dim regression head with 6 independent 3-way
softmax heads trained with cross-entropy, and decodes at inference via argmax
back to ``{-0.1, 0, +0.1}`` physical values. The transformer backbone, CVAE,
image encoders, and training-loop plumbing are unchanged.

Usage
-----
In ``scripts/train_act.py``, when ``cfg["policy"]["discrete_action"]["enable"]
== True``, we bypass LeRobot's ``make_policy`` and build a ``DiscreteACTPolicy``
directly. The config MUST also set the action normalization mode to
``IDENTITY`` (``policy.normalization_mapping.ACTION: IDENTITY``) so the target
arrives in raw physical units, ready to be snapped to the 0.1 grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES


# The three permitted raw-physical action values, in class-index order.
# Index 0 -> -0.1, index 1 -> 0.0, index 2 -> +0.1.
CLASS_VALUES = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)
N_CLASSES = 3
ACTION_STEP = 0.1  # the quantum; the dataset is exactly multiples of this


def snap_to_class(raw_action: torch.Tensor) -> torch.Tensor:
    """Map raw action in {-0.1, 0, +0.1} -> class index in {0, 1, 2}.

    We round first to absorb any float32 drift (the dataset has zero
    deviation from the grid, but normalized pipelines could introduce tiny
    noise in principle).
    """
    return torch.round(raw_action / ACTION_STEP).long().clamp(-1, 1) + 1


class DiscreteACTPolicy(ACTPolicy):
    """ACT with a per-dim 3-way classification head.

    We keep the parent ``__init__`` so buffers, image processors, and the
    ``PolicyFeature`` plumbing come through unchanged.  Right after the parent
    constructs ``self.model``, we:

    * Replace ``self.model.action_head`` (``Linear(D, A)``) with
      ``Linear(D, A*C)`` so the output is per-dim class logits.
    * Register the class value lookup as a buffer so device transfers follow
      the module.

    Loss: mean cross-entropy across all valid (non-pad) positions and dims,
    + KLD if ``use_vae=True`` (unchanged).

    Inference: argmax over class dim, lookup into CLASS_VALUES.  Output has
    the same shape/semantics as regression ACT (``(B, T, A)`` raw physical),
    so the rest of the training pipeline and downstream cross-eval scripts
    don't need to know they're talking to a classification model.
    """

    def __init__(self, config: ACTConfig, **kwargs):
        # ACTPolicy.__init__ signature is ``(config, **kwargs)`` (no
        # positional dataset_stats since LeRobot v2.x).  Forward kwargs.
        super().__init__(config, **kwargs)

        self.action_dim = int(config.action_feature.shape[0])
        self.n_classes = N_CLASSES

        old_head = self.model.action_head
        in_features = old_head.in_features
        # Fresh initialization is fine; the old head's weights were sized for
        # regression. We use the default Linear init.
        self.model.action_head = nn.Linear(in_features, self.action_dim * self.n_classes)

        self.register_buffer("_class_values", CLASS_VALUES.clone(), persistent=False)

    # ---- core helpers -------------------------------------------------

    def _logits_from_model(self, batch: dict) -> tuple[torch.Tensor, tuple]:
        """Run the transformer and reshape the flat head output to class logits.

        Returns ``(logits, (mu, log_sigma_x2))`` where ``logits`` has shape
        ``(B, T, A, C)``.
        """
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        flat, (mu, log_sigma_x2) = self.model(batch)
        # flat: (B, T, A*C)
        B, T, AC = flat.shape
        logits = flat.view(B, T, self.action_dim, self.n_classes)
        return logits, (mu, log_sigma_x2)

    def _decode(self, logits: torch.Tensor) -> torch.Tensor:
        """argmax -> physical action, shape (B, T, A)."""
        cls = logits.argmax(dim=-1)  # (B, T, A)
        return self._class_values[cls]

    # ---- overrides for train / eval ----------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        """Same contract as ACTPolicy.predict_action_chunk but returns raw
        physical action values decoded from the classification head."""
        self.eval()
        logits, _ = self._logits_from_model(batch)
        return self._decode(logits)

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Cross-entropy + optional KLD.

        Expects ``batch[ACTION]`` to be in RAW physical units (i.e., the
        preprocessor is set to IDENTITY for the ACTION normalization mode).
        """
        logits, (mu_hat, log_sigma_x2_hat) = self._logits_from_model(batch)
        B, T, A, C = logits.shape

        raw_action = batch[ACTION]  # (B, T, A), physical
        target_cls = snap_to_class(raw_action)  # (B, T, A) in {0, 1, 2}

        # (B*T*A, C) logits against (B*T*A,) targets, masked by action_is_pad.
        ce_per_pos = F.cross_entropy(
            logits.reshape(-1, C),
            target_cls.reshape(-1),
            reduction="none",
        ).view(B, T, A)

        pad_mask = (~batch["action_is_pad"]).unsqueeze(-1).float()  # (B, T, 1)
        denom = pad_mask.sum().clamp_min(1.0) * A
        ce_loss = (ce_per_pos * pad_mask).sum() / denom

        # Diagnostics: per-dim accuracy, plus "non-idle" accuracy (only the
        # positions where target != 0).  The non-idle accuracy is the real
        # signal -- "always predict 0" trivially gets 80-99.9% per-dim acc.
        with torch.no_grad():
            pred_cls = logits.argmax(dim=-1)
            correct = (pred_cls == target_cls).float()
            acc = (correct * pad_mask).sum() / denom

            idle_mask = (target_cls == 1).float() * pad_mask
            motion_mask = (target_cls != 1).float() * pad_mask
            motion_denom = motion_mask.sum().clamp_min(1.0)
            motion_acc = (correct * motion_mask).sum() / motion_denom

            # Fraction of predictions that are "idle" -- a model collapsing
            # to the majority class will push this toward 1.0.
            pred_idle_rate = ((pred_cls == 1).float() * pad_mask).sum() / pad_mask.sum().clamp_min(1.0)

            # Also report L1 in physical units so wandb plots line up with
            # regression runs and cross_eval.py.
            phys_pred = self._class_values[pred_cls]
            l1_phys = ((phys_pred - raw_action).abs() * pad_mask).sum() / denom

        loss_dict = {
            "ce_loss": ce_loss.item(),
            "accuracy": acc.item(),
            "motion_accuracy": motion_acc.item(),
            "pred_idle_rate": pred_idle_rate.item(),
            "l1_phys": l1_phys.item(),
            # Legacy key so the existing training logger ("loss") picks it up.
            "l1_loss": ce_loss.item(),
        }
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = ce_loss + mean_kld * self.config.kl_weight
        else:
            loss = ce_loss

        return loss, loss_dict
