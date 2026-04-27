#!/usr/bin/env bash
# Train ACT v13b (lower KL weight + commit to longer action chunk at deploy).
# Default: physical GPU 0 (set CUDA_DEVICE_ID=1 to use GPU 1, etc.)
# v13b is the complement to v13: keeps action_weighting at v12 settings and
# instead loosens the CVAE KL term and bumps n_action_steps.
#
# Usage:
#   ./scripts/launch_train_act_v13b.sh
#   CUDA_DEVICE_ID=1 ./scripts/launch_train_act_v13b.sh
#   EXTRA="--set training.steps=10_000" ./scripts/launch_train_act_v13b.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src/aic"

export CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/act_sfp_v13b.yaml}"
EXTRA="${EXTRA:-}"

exec pixi run python "${REPO_ROOT}/scripts/train_act.py" \
  --config "${CONFIG}" \
  ${EXTRA}
