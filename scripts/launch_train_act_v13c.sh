#!/usr/bin/env bash
# Train ACT v13c (disable CVAE entirely -- pure conditional regression).
# Default: physical GPU 2 (set CUDA_DEVICE_ID=N to use a different GPU).
# v13c isolates the use_vae: true -> false change.  Single lever for clean
# attribution against v12 and the parallel v13 / v13b experiments.
#
# Usage:
#   ./scripts/launch_train_act_v13c.sh
#   CUDA_DEVICE_ID=3 ./scripts/launch_train_act_v13c.sh
#   EXTRA="--set training.steps=10_000" ./scripts/launch_train_act_v13c.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src/aic"

export CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/act_sfp_v13c.yaml}"
EXTRA="${EXTRA:-}"

exec pixi run python "${REPO_ROOT}/scripts/train_act.py" \
  --config "${CONFIG}" \
  ${EXTRA}
