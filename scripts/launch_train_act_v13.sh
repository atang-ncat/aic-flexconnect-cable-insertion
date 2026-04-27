#!/usr/bin/env bash
# Launch v13 training (fixed img norm + aggressive action weighting + head trim)
#
# Usage:
#   ./scripts/launch_train_act_v13.sh          # default GPU 0
#   CUDA_DEVICE_ID=1 ./scripts/launch_train_act_v13.sh   # use GPU 1
set -euo pipefail

GPU="${CUDA_DEVICE_ID:-0}"
CONFIG="/scratch2/atang/ws_aic/configs/act_sfp_v13.yaml"
LOGDIR="/scratch2/atang/ws_aic/outputs/act_sfp/sweep_logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/v13_${TIMESTAMP}.log"

echo "Starting v13 on GPU ${GPU} → ${LOGFILE}"
cd /scratch2/atang/ws_aic/src/aic

CUDA_VISIBLE_DEVICES="$GPU" \
  nohup pixi run python /scratch2/atang/ws_aic/scripts/train_act.py \
    --config "$CONFIG" \
    > "$LOGFILE" 2>&1 &

echo "PID: $!"
echo "Tail: tail -f ${LOGFILE}"
