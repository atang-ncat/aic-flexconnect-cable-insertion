#!/usr/bin/env bash
# collect_many.sh — single-config multi-episode wrapper for auto_collect.py
#
# Launches Gazebo, runs auto_collect.py with --exit-on-success, kills
# Gazebo, and loops N times. Each iteration produces at most one
# saved episode. The first episode creates the dataset; subsequent
# episodes pass --resume to append to it.
#
# Why a wrapper: once the cable is latched into the port, further
# attempts in the same Gazebo session would yank it back out. The
# supported way to collect N episodes is N fresh Gazebo launches.
# This script automates that loop.
#
# Usage:
#   source ~/lab/ws_aic/setup_dev.sh     # once per terminal
#   scripts/collect_many.sh \
#       --episodes 3 \
#       --config "nic_card_mount_0_present:=true" \
#       --port-frame "task_board/nic_card_mount_0/sfp_port_0_link"
#
# Notes:
#   - Requires ros2 + pixi on PATH (source setup_dev.sh first).
#   - Gazebo is launched in its own process group; on cleanup we SIGTERM
#     the group, then SIGKILL any survivors, then pkill as a safety net.
#   - Readiness is detected by polling /joint_states. Default timeout 60s.
#   - Success is confirmed by reading total_episodes from meta/info.json.
#   - Ctrl-C is trapped so Gazebo gets killed even on an early exit.

set -u

# -----------------------------------------------------------------------------
# Defaults (can be overridden via CLI flags)
# -----------------------------------------------------------------------------
EPISODES=1
CONFIG="nic_card_mount_0_present:=true"
PORT_FRAME="task_board/nic_card_mount_0/sfp_port_0_link"
PLUG_FRAME=""   # empty => let auto_collect.py use its default
DATASET="/run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp"
REPO_ID="atang/aic_sfp_auto"
MAX_ATTEMPTS=8
READY_TIMEOUT=60
POST_KILL_SLEEP=3
GAZEBO_LOG="/tmp/collect_many_gazebo.log"
AUTO_COLLECT="/run/host/scratch2/atang/ws_aic/scripts/auto_collect.py"
# Directory containing the pixi.toml that auto_collect.py needs.
# `pixi run` must be invoked from here (or with --manifest-path); running
# from the workspace root fails with "could not find pixi.toml".
PIXI_DIR="/run/host/scratch2/atang/ws_aic/src/aic"

usage() {
    cat <<'EOF'
Usage: collect_many.sh [options]

Options:
  --episodes N              Number of episodes to collect (default: 1)
  --config "ARGS"           Gazebo launch argument string for this config
                            (default: "nic_card_mount_0_present:=true")
  --port-frame FRAME        TF frame of the target SFP port
                            (default: task_board/nic_card_mount_0/sfp_port_0_link)
  --plug-frame FRAME        TF frame of the plug tip (omit to use script default)
  --dataset-root DIR        Where the LeRobot dataset lives
  --repo-id REPO            LeRobot repo id for the dataset
  --max-attempts N          Max auto_collect attempts per episode (default: 8)
  --ready-timeout SECONDS   How long to wait for Gazebo readiness (default: 60)
        --gazebo-log FILE         Where to capture Gazebo stdout/stderr
                            (default: /tmp/collect_many_gazebo.log)
  --pixi-dir DIR            Directory containing pixi.toml to run from
                            (default: /run/host/scratch2/atang/ws_aic/src/aic)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)        EPISODES="$2"; shift 2 ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --port-frame)      PORT_FRAME="$2"; shift 2 ;;
        --plug-frame)      PLUG_FRAME="$2"; shift 2 ;;
        --dataset-root)    DATASET="$2"; shift 2 ;;
        --repo-id)         REPO_ID="$2"; shift 2 ;;
        --max-attempts)    MAX_ATTEMPTS="$2"; shift 2 ;;
        --ready-timeout)   READY_TIMEOUT="$2"; shift 2 ;;
        --gazebo-log)      GAZEBO_LOG="$2"; shift 2 ;;
        --pixi-dir)        PIXI_DIR="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# -----------------------------------------------------------------------------
# Environment sanity checks
# -----------------------------------------------------------------------------
if ! command -v ros2 >/dev/null; then
    echo "[wrapper] ros2 not on PATH. Did you source setup_dev.sh?" >&2
    exit 3
fi
if ! command -v pixi >/dev/null; then
    echo "[wrapper] pixi not on PATH." >&2
    exit 3
fi
if [[ ! -f "$AUTO_COLLECT" ]]; then
    echo "[wrapper] auto_collect.py not found at $AUTO_COLLECT" >&2
    exit 3
fi
if [[ ! -f "$PIXI_DIR/pixi.toml" ]]; then
    echo "[wrapper] No pixi.toml at $PIXI_DIR (pass --pixi-dir to override)" >&2
    exit 3
fi

# -----------------------------------------------------------------------------
# State + cleanup
# -----------------------------------------------------------------------------
GZ_PID=""

cleanup_gazebo() {
    # Kill everything in Gazebo's process group, then belt-and-suspenders pkill
    # in case anything escaped (e.g. bridges that reparented to init).
    if [[ -n "$GZ_PID" ]]; then
        # With setsid, the child's PGID equals its PID.
        kill -TERM -"$GZ_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$GZ_PID" 2>/dev/null || true
        GZ_PID=""
    fi
    pkill -f 'gz sim' 2>/dev/null || true
    pkill -f 'ros2 launch.*aic_gz_bringup' 2>/dev/null || true
    pkill -f 'ros_gz_bridge' 2>/dev/null || true
    pkill -f 'parameter_bridge' 2>/dev/null || true
    sleep "$POST_KILL_SLEEP"
}

on_interrupt() {
    echo ""
    echo "[wrapper] Caught signal, cleaning up Gazebo..."
    cleanup_gazebo
    exit 130
}

trap on_interrupt INT TERM

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
count_saved_episodes() {
    # total_episodes from the LeRobot dataset's meta/info.json, or 0.
    local info="$DATASET/meta/info.json"
    if [[ ! -f "$info" ]]; then
        echo 0
        return
    fi
    python3 - "$info" <<'PY' 2>/dev/null || echo 0
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("total_episodes", 0))
PY
}

launch_gazebo() {
    local args="$1"
    echo "[wrapper] Launching Gazebo (log: $GAZEBO_LOG)"
    echo "[wrapper]   args: $args"
    : > "$GAZEBO_LOG"
    # setsid puts Gazebo in its own process group so we can kill the whole tree.
    setsid bash -c "exec ros2 launch aic_bringup aic_gz_bringup.launch.py \
        ground_truth:=true start_aic_engine:=false \
        spawn_task_board:=true spawn_cable:=true \
        attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
        $args" </dev/null >>"$GAZEBO_LOG" 2>&1 &
    GZ_PID=$!
    disown 2>/dev/null || true
}

wait_for_ready() {
    # Returns 0 once a /joint_states message arrives, 1 on timeout.
    local timeout=$1
    local t0=$SECONDS
    echo "[wrapper] Waiting for /joint_states (up to ${timeout}s)..."
    while (( SECONDS - t0 < timeout )); do
        if timeout 3 ros2 topic echo /joint_states --once \
                >/dev/null 2>&1; then
            echo "[wrapper] Gazebo ready after $((SECONDS - t0))s"
            return 0
        fi
        # Poll every ~2s; don't hammer ros2 daemon
        sleep 2
    done
    echo "[wrapper] Timed out after ${timeout}s waiting for Gazebo" >&2
    return 1
}

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
started_at=$SECONDS
initial_saved=$(count_saved_episodes)
saved_this_run=0
failed_launches=0
failed_collections=0

echo "[wrapper] === collect_many.sh ==="
echo "[wrapper] target:      $EPISODES episode(s)"
echo "[wrapper] config:      $CONFIG"
echo "[wrapper] port frame:  $PORT_FRAME"
echo "[wrapper] dataset:     $DATASET"
echo "[wrapper] starting with $initial_saved existing episode(s) in dataset"
echo ""

for ((ep=1; ep<=EPISODES; ep++)); do
    echo "=================================================================="
    echo "[wrapper] Episode $ep / $EPISODES   (saved so far: $saved_this_run)"
    echo "=================================================================="

    pre_count=$(count_saved_episodes)

    launch_gazebo "$CONFIG"
    if ! wait_for_ready "$READY_TIMEOUT"; then
        echo "[wrapper] Gazebo failed to come up; tail of $GAZEBO_LOG:"
        tail -n 20 "$GAZEBO_LOG" | sed 's/^/[gazebo] /'
        cleanup_gazebo
        failed_launches=$((failed_launches + 1))
        continue
    fi

    resume_flag=""
    if [[ "$pre_count" -gt 0 ]]; then
        resume_flag="--resume"
    fi

    extra_plug=""
    if [[ -n "$PLUG_FRAME" ]]; then
        extra_plug="--plug-frame $PLUG_FRAME"
    fi

    echo "[wrapper] Running auto_collect.py (resume='$resume_flag')"
    # `pixi run` requires the working directory to contain pixi.toml, so
    # cd into the pixi project in a subshell. The subshell isolates the
    # cd so $PWD is unchanged afterwards.
    # shellcheck disable=SC2086
    (
        cd "$PIXI_DIR" && \
        pixi run python3 "$AUTO_COLLECT" \
            --dataset-root "$DATASET" \
            --repo-id "$REPO_ID" \
            --port-frame "$PORT_FRAME" \
            $extra_plug \
            --num-episodes 1 --max-attempts "$MAX_ATTEMPTS" \
            --exit-on-success $resume_flag
    )
    rc=$?

    post_count=$(count_saved_episodes)
    cleanup_gazebo

    if (( post_count > pre_count )); then
        saved_this_run=$((saved_this_run + 1))
        echo "[wrapper] Episode $ep: SAVED (dataset now has $post_count total)"
    else
        failed_collections=$((failed_collections + 1))
        echo "[wrapper] Episode $ep: NO SAVE (rc=$rc, dataset still at $post_count)"
    fi
done

elapsed=$((SECONDS - started_at))
final_count=$(count_saved_episodes)

echo ""
echo "=================================================================="
echo "[wrapper] SUMMARY"
echo "=================================================================="
echo "[wrapper]   saved this run:     $saved_this_run / $EPISODES"
echo "[wrapper]   dataset episodes:   $initial_saved -> $final_count"
echo "[wrapper]   launch failures:    $failed_launches"
echo "[wrapper]   collection failures:$failed_collections"
printf "[wrapper]   elapsed wall time:  %dm %02ds\n" $((elapsed/60)) $((elapsed%60))
echo "=================================================================="

# Non-zero exit if we didn't get everything we asked for
if (( saved_this_run < EPISODES )); then
    exit 1
fi
exit 0
