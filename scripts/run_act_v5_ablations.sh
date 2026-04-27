#!/usr/bin/env bash
# Launch the three v5 ablation cells in parallel on GPUs 0/1/2.
#
# Design rationale lives at the top of configs/act_sfp_v5.yaml.
#
#   GPU 0  v5_seed_variance  : v2 recipe (raw, 1 cam, no weight, no trim),
#                              seed=7 instead of 42.  Measures run-to-run
#                              noise -- the ruler for every future claim.
#   GPU 1  v5_strong_aug     : v2 recipe + stronger regularization
#                              (brightness/contrast 0.5, erasing p=0.4,
#                              dropout 0.15).  Attacks the overfitting
#                              plateau Antigravity flagged.
#   GPU 2  v5_smoothed_combo : smoothed + mild_weight (floor=0.5) +
#                              trim=0.10 + all 3 cameras.  Ceiling probe
#                              for the smoothed family; smoothed outputs
#                              are still a better choice for hardware
#                              rollout safety, so we want its best version.
#
# All three inherit v5's early-stop config (5k patience, 12k warmup).
# Expected wall-clock: 1-3 hours each depending on when they converge;
# they will self-budget.

set -euo pipefail

WS=/scratch2/atang/ws_aic
CONFIG=${WS}/configs/act_sfp_v5.yaml
LOG_DIR=${WS}/logs
mkdir -p "${LOG_DIR}"

PROJECT=${WS}/src/aic

launch () {
  local name="$1"
  local gpu="$2"
  shift 2
  local log="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"
  echo "[launch] ${name} on GPU ${gpu}  -> ${log}"
  cd "${PROJECT}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup pixi run python \
    "${WS}/scripts/train_act.py" \
    --config "${CONFIG}" \
    --set run_name="${name}" wandb.tags="[act,sfp,teleop-only,v5,${name}]" "$@" \
    > "${log}" 2>&1 &
  echo $! > "${pid_file}"
  sleep 2
}

# ---------------------------------------------------------------------------
# Cell A: seed-variance probe (v2 recipe, seed=7).
# ---------------------------------------------------------------------------
launch v5_seed_variance 0 \
  seed=7

# ---------------------------------------------------------------------------
# Cell B: stronger regularization (v2 recipe + bigger aug + dropout 0.15).
# ColorJitter hue stays small (0.05) because large hue shifts break the
# assumption that color is a useful cue for part identification.
# ---------------------------------------------------------------------------
launch v5_strong_aug 1 \
  augmentation.color_jitter.brightness=0.5 \
  augmentation.color_jitter.contrast=0.5 \
  augmentation.color_jitter.saturation=0.5 \
  augmentation.random_erasing.p=0.4 \
  augmentation.random_erasing.scale='[0.02,0.18]' \
  policy.dropout=0.15

# ---------------------------------------------------------------------------
# Cell C: smoothed kitchen-sink.  3 cams + mild weight + trim=0.10.
# ---------------------------------------------------------------------------
launch v5_smoothed_combo 2 \
  dataset.root=teleop-dataset-smoothed \
  'dataset.cameras=[observation.images.center_camera,observation.images.left_camera,observation.images.right_camera]' \
  dataset.trim_tail_fraction=0.10 \
  action_weighting.enable=true \
  action_weighting.floor=0.5

echo
echo "All three v5 runs launched.  Monitor with:"
echo "  tail -f ${LOG_DIR}/v5_*.log"
echo "PID files:  ${LOG_DIR}/v5_*.pid"
