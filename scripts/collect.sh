#!/usr/bin/env bash
# Operator wrapper for auto_collect.py — avoids long-command paste issues.
#
# Usage (inside the aic_eval distrobox, with Gazebo already running
# with ground_truth:=true):
#
#   bash /run/host/scratch2/atang/ws_aic/scripts/collect.sh dryrun-sfp
#   bash /run/host/scratch2/atang/ws_aic/scripts/collect.sh sfp 30
#   bash /run/host/scratch2/atang/ws_aic/scripts/collect.sh sc 20
#   bash /run/host/scratch2/atang/ws_aic/scripts/collect.sh dryrun-sc
#
# Any extra flags after the mode get passed through to auto_collect.py.
# E.g.:
#   bash scripts/collect.sh sfp 30 --resume
#   bash scripts/collect.sh sfp 30 --discard-high-force

set -euo pipefail

# Resolve the workspace root from this script's location, so the wrapper
# works whether invoked by absolute path, relative path, or via symlink.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AIC_DIR="$WS_ROOT/src/aic"
AUTO_COLLECT="$WS_ROOT/scripts/auto_collect.py"
DATA_ROOT_BASE="$WS_ROOT/teleop-automated-dataset-ft"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <mode> [num_episodes] [extra auto_collect.py flags...]" >&2
    echo "  mode: sfp | sc | dryrun-sfp | dryrun-sc" >&2
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    dryrun-sfp)
        EXTRA=(--plug-type sfp --dry-run)
        ;;
    dryrun-sc)
        EXTRA=(--plug-type sc --dry-run)
        ;;
    sfp|sc)
        if [[ $# -lt 1 ]]; then
            echo "Usage: $0 $MODE <num_episodes> [extra flags...]" >&2
            exit 1
        fi
        N="$1"
        shift
        EXTRA=(
            --plug-type "$MODE"
            --dataset-root "$DATA_ROOT_BASE/$MODE"
            --num-episodes "$N"
            --max-attempts "$((N * 2))"
        )
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Expected: sfp | sc | dryrun-sfp | dryrun-sc" >&2
        exit 1
        ;;
esac

cd "$AIC_DIR"
echo "[collect.sh] cd $AIC_DIR"
echo "[collect.sh] invoking auto_collect.py ${EXTRA[*]} $*"
exec pixi run python3 "$AUTO_COLLECT" "${EXTRA[@]}" "$@"
