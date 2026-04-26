#!/usr/bin/env bash
# Launch all v11 ablation experiments in parallel across 4 GPUs.
# Each run takes ~15-20 min with 15k steps on RTX 6000 Ada.
#
# Usage:
#   ./scripts/launch_v11_sweep.sh        # run all 4
#   ./scripts/launch_v11_sweep.sh v11    # run just one
#   ./scripts/launch_v11_sweep.sh v11 v11c  # run a subset

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/scripts/train_act.py"
LOG_DIR="${REPO_ROOT}/outputs/act_sfp/sweep_logs"
mkdir -p "${LOG_DIR}"

# Experiment → GPU assignment
declare -A GPU_MAP=(
  [v11]=0
  [v11b]=1
  [v11c]=2
  [v11d]=3
)

# Default: run all experiments; or specify which ones on the command line.
if [ $# -gt 0 ]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(v11 v11b v11c v11d)
fi

PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
  config="${REPO_ROOT}/configs/act_sfp_${exp}.yaml"
  gpu="${GPU_MAP[$exp]:-0}"
  logfile="${LOG_DIR}/${exp}_$(date +%Y%m%d_%H%M%S).log"

  if [ ! -f "${config}" ]; then
    echo "ERROR: Config not found: ${config}" >&2
    continue
  fi

  echo "Starting ${exp} on GPU ${gpu} → ${logfile}"
  (
    cd "${REPO_ROOT}/src/aic"
    CUDA_VISIBLE_DEVICES="${gpu}" pixi run python "${SCRIPT}" \
      --config "${config}" \
      2>&1 | tee "${logfile}"
  ) &
  PIDS+=($!)
  # Stagger starts by 5s to avoid dataset-loading thundering herd
  sleep 5
done

echo ""
echo "=== All ${#PIDS[@]} experiments launched ==="
echo "PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/v11*.log"
echo "  watch 'nvidia-smi'"
echo ""
echo "Waiting for all to finish..."
wait "${PIDS[@]}"
echo "=== All experiments complete ==="
