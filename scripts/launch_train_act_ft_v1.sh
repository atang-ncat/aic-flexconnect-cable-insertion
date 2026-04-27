#!/usr/bin/env bash
# Train ACT (act_sfp_v10) on teleop-dataset-ft-v1 on a single GPU.
# Default: physical GPU 1 (set CUDA_DEVICE_ID=0 to use GPU 0, etc.)
#
# Usage:
#   ./scripts/launch_train_act_ft_v1.sh
#   CONFIG=/path/to/custom.yaml ./scripts/launch_train_act_ft_v1.sh
#   EXTRA="--set run_name=my_run training.steps=20_000" ./scripts/launch_train_act_ft_v1.sh
#
# Run from anywhere; paths are fixed to this repo.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src/aic"

# Visible GPU index: only one device seen as cuda:0 by PyTorch
export CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/act_sfp_v10.yaml}"
EXTRA="${EXTRA:-}"

exec pixi run python "${REPO_ROOT}/scripts/train_act.py" \
  --config "${CONFIG}" \
  ${EXTRA}
