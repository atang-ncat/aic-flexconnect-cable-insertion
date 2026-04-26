#!/usr/bin/env bash
# Train ACT v11 (anti-overfitting: cosine LR, frozen backbone, higher regularization)
# Default: physical GPU 1 (set CUDA_DEVICE_ID=0 to use GPU 0, etc.)
#
# Usage:
#   ./scripts/launch_train_act_v11.sh
#   CUDA_DEVICE_ID=0 ./scripts/launch_train_act_v11.sh
#   EXTRA="--set training.steps=10_000" ./scripts/launch_train_act_v11.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src/aic"

export CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/act_sfp_v11.yaml}"
EXTRA="${EXTRA:-}"

exec pixi run python "${REPO_ROOT}/scripts/train_act.py" \
  --config "${CONFIG}" \
  ${EXTRA}
