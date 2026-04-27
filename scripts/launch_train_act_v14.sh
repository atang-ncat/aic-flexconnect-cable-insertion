#!/usr/bin/env bash
# Launch v14 (GPU 0) and v14b (GPU 1) in parallel
set -euo pipefail
SCRIPT=/scratch2/atang/ws_aic/scripts/train_act.py
LOGDIR=/scratch2/atang/ws_aic/outputs/act_sfp/sweep_logs
mkdir -p "$LOGDIR"

echo "=== Launching v14 (40x weighting) on GPU 0 ==="
cd /scratch2/atang/ws_aic/src/aic && \
  CUDA_VISIBLE_DEVICES=0 pixi run python "$SCRIPT" \
    --config /scratch2/atang/ws_aic/configs/act_sfp_v14.yaml \
  > "${LOGDIR}/v14_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID1=$!
echo "  PID=$PID1"

sleep 5

echo "=== Launching v14b (20x weighting + low KL) on GPU 1 ==="
cd /scratch2/atang/ws_aic/src/aic && \
  CUDA_VISIBLE_DEVICES=1 pixi run python "$SCRIPT" \
    --config /scratch2/atang/ws_aic/configs/act_sfp_v14b.yaml \
  > "${LOGDIR}/v14b_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID2=$!
echo "  PID=$PID2"

echo ""
echo "Both running. Monitor:"
echo "  tail -f ${LOGDIR}/v14_*.log"
echo "  tail -f ${LOGDIR}/v14b_*.log"
echo ""
echo "wait $PID1 $PID2"
wait $PID1 $PID2
echo "=== Both done ==="
