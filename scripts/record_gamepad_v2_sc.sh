#!/usr/bin/env bash
# Record gamepad teleop into gamepad-v2 SC dataset (separate tree from SFP).
# Launch Gazebo with the SC scene first — see docs/teleop_guide.md §1 (reversed cable).
set -euo pipefail
cd /run/host/scratch2/atang/ws_aic/src/aic
exec pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_gamepad_ee --teleop.id=aic \
  --teleop.stick_deadzone=0.02 \
  --teleop.stick_expo=1.0 \
  --teleop.trigger_deadzone=0.01 \
  --teleop.trigger_expo=1.0 \
  --teleop.low_command_scaling=0.04 \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=atang/aic_sc_demos_gamepad_v2 \
  --dataset.root=/run/host/scratch2/atang/ws_aic/datasets/teleop-dataset-gamepad-v2/sc \
  --dataset.single_task="Insert SC connector into SC port" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true \
  "$@"
