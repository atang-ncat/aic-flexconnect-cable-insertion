#!/usr/bin/env bash
# Launch the three ACT-v4 ablation cells in parallel on GPUs 0/1/2.
#
# All three share the v4 base config (smoothed data, 1-cam, ImageNet ResNet-18,
# GPU-side aug, unweighted L1, early stopping).  Each flips exactly one
# variable:
#
#   GPU 0  v4_multicam_smooth  : adds the two side cameras back in.
#   GPU 1  v4_boundary_trim    : drops the last 10% of each train episode.
#   GPU 2  v4_mild_weight      : re-enables weighted L1 at floor=0.5 (2x).
#
# Early stopping (5k patience, 12k min_steps) applies to all three, so each
# run self-budgets -- wall-clock varies but usually lands in 20k-30k steps.
#
# Logs go to ``logs/v4_*.log``; wandb names match the cell name.

set -euo pipefail

WS=/scratch2/atang/ws_aic
CONFIG=${WS}/configs/act_sfp_v4.yaml
LOG_DIR=${WS}/logs
mkdir -p "${LOG_DIR}"

# cwd for pixi must be the pixi-project root, not the ws root.
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
    --set run_name="${name}" wandb.tags="[act,sfp,teleop-only,v4,${name}]" "$@" \
    > "${log}" 2>&1 &
  echo $! > "${pid_file}"
  sleep 2
}

launch v4_multicam_smooth 0 \
  'dataset.cameras=[observation.images.center_camera,observation.images.left_camera,observation.images.right_camera]'

launch v4_boundary_trim 1 \
  dataset.trim_tail_fraction=0.10

launch v4_mild_weight 2 \
  action_weighting.enable=true \
  action_weighting.floor=0.5

echo
echo "All three v4 runs launched.  Monitor with:"
echo "  tail -f ${LOG_DIR}/v4_*.log"
echo "PID files:  ${LOG_DIR}/v4_*.pid"
