#!/usr/bin/env bash
# Re-tare the wrist F/T sensor by calling the controller's tare service.
#
# Why this exists: the lerobot driver writes wrench - controller_state.fts_tare_offset
# into observation.state.  The controller only updates fts_tare_offset when this
# service is called.  Without a per-session re-tare, the recorded "contact force"
# slowly accumulates a gravity-projection error as the gripper rotates over the
# session, and the policy can correlate that drift with task progress (a leak).
#
# Operator workflow during teleop recording:
#   1. Move the arm to a free-space pose with the plug in hand and no contact.
#   2. From any terminal inside the aic_eval distrobox, run:
#        bash /run/host/scratch2/atang/ws_aic/scripts/tare.sh
#   3. Confirm the line "success: True ... Successfully tared" appears.
#   4. Begin the episode (lerobot-record continues running through this).
#
# Recommended cadence:
#   * Once at session start (mandatory).
#   * Whenever the operator picks the plug up from a fresh location.
#   * Whenever the gripper has been rotated noticeably (> ~10 deg cumulative)
#     since the last tare.
#
# For SFP, where rotation is minimal, one tare per session is usually enough.
# For SC and any future task with substantial rotation, re-tare more often.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AIC_DIR="$WS_ROOT/src/aic"

cd "$AIC_DIR"
echo "[tare.sh] calling /aic_controller/tare_force_torque_sensor (std_srvs/Trigger)"
exec pixi run ros2 service call \
    /aic_controller/tare_force_torque_sensor \
    std_srvs/srv/Trigger
