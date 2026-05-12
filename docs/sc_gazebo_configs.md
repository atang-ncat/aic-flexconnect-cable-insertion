# SC Port Gazebo Scene Configs — FlexConnect Cable Insertion

All commands run in T2 (Gazebo terminal) after:

```bash
distrobox enter -r aic_eval
source ~/aic-flexconnect-cable-insertion/setup_dev.sh
```

Workflow for switching scenes:
1. Ctrl+C the current scene in T2
2. Wait 3-5 seconds for cleanup
3. Paste new launch command
4. Wait for Gazebo GUI to fully reload
5. Tare F/T sensor in T5: `ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger`
6. T6 keeps running with `--resume=true` — no need to restart recording

---

## Suggested Recording Rotation

| Episodes | Config | Why |
|---|---|---|
| 1-15 ✅ | DONE — default (port=-0.04, rail=-0.09) | baseline data |
| 16-25 | Config 2 (port=-0.02, rail=-0.05) | small shift |
| 26-35 | Config 3 (port=0.0, rail=0.0) | centered |
| 36-45 | Config 4 (port=+0.02, rail=+0.03) | opposite shift |
| 46-55 | Config 5 (port=-0.023, rail=-0.07) | eval edge |
| 56-65 | Config 6 (default + yaw=3.05) | rotation |
| 66-75 | Config 7 (default + task_board_y=-0.15) | shifted board |
| 76+ | Combinations | hardest cases |

---

## Default — what you've been using (Episodes 1-15)

```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## SC Port Translation Variations

### Config 2 — Slightly left
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.05
```

### Config 3 — Centered
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.0
```

### Config 4 — Slightly right
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.03
```

### Config 5 — Eval edge (max left -0.023)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.023 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.07
```

### Config 5b — Eval edge (max right +0.023)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.023 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## Rail Mount Variations

### Rail center (default)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.0
```

### Rail shifted +0.05
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.05
```

---

## Task Board Position Variations (Bigger Workspace Diversity)

Default: `task_board_x:=0.15`, `task_board_y:=-0.20`

### Config 7a — Board closer to robot (x=0.13)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.13
```

### Config 7b — Board further from robot (x=0.17)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.17
```

### Config 7c — Board shifted right (y=-0.15)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_y:=-0.15
```

### Config 7d — Board shifted left (y=-0.25)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_y:=-0.25
```

---

## Board Yaw Variations (Rotated Task Board)

Default yaw ≈ π (3.14). Vary ±0.1 rad (~6°).

### Config 6a — Yaw 3.05 (~5° CW)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_yaw:=3.05
```

### Config 6b — Yaw 3.24 (~6° CCW)
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_yaw:=3.24
```

---

## Combination Variations (Hardest Cases)

### Combo 1 — Port shift + yaw
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.05 \
  task_board_yaw:=3.05
```

### Combo 2 — Eval edge + board further
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.023 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.07 \
  task_board_x:=0.17
```

### Combo 3 — Centered + board shifted right
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=0.0 \
  task_board_y:=-0.15
```

### Combo 4 — All variations combined
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.015 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.04 \
  task_board_x:=0.16 task_board_y:=-0.18 task_board_yaw:=3.18
```

---

## Reference — What NOT to Change

| Argument | Value | Why keep it |
|---|---|---|
| `cable_type` | `sfp_sc_cable_reversed` | Specific to SC port task |
| `ground_truth` | `true` | Needed for recording (and tare service) |
| `start_aic_engine` | `false` | Recording doesn't need the engine |
| `spawn_task_board` | `true` | Required for scene |
| `spawn_cable` | `true` | Required for scene |
| `attach_cable_to_gripper` | `true` | Required for teleop start state |
| `sc_port_0_present` | `true` | The thing you're inserting into |
| `sc_mount_rail_0_present` | `true` | The mount holding the SC port |

---

## Argument Reference Table

| Argument | Default | Range to vary | Effect |
|---|---|---|---|
| `sc_port_0_translation` | varies | -0.04 to +0.04 m | SC port lateral position on rail |
| `sc_mount_rail_0_translation` | varies | -0.09 to +0.09 m | Rail position on task board |
| `task_board_x` | 0.15 | 0.13 to 0.17 m | Board forward/back from robot |
| `task_board_y` | -0.20 | -0.15 to -0.25 m | Board left/right |
| `task_board_yaw` | ~3.14 | 3.0 to 3.28 rad | Board rotation around vertical axis |

---

## Per-Episode Checklist

Before EACH episode (regardless of config):
1. [ ] Gazebo loaded fully (you can see arm + cable + task board)
2. [ ] Tare F/T sensor in T5
3. [ ] T4 force monitor reads ~0 N after tare
4. [ ] T6 lerobot-record is running (waiting for input)
5. [ ] Gamepad responding (test left stick if first episode of session)

Per-config switching (every 8-10 episodes):
1. [ ] Note last episode number in info.json
2. [ ] Ctrl+C T2 Gazebo, wait 5 sec
3. [ ] Launch new config in T2
4. [ ] Wait for GUI fully loaded
5. [ ] Tare in T5
6. [ ] Continue T6 — episode count keeps incrementing

---

## Quick Verify After Scene Change

```bash
# Check the scene is running
ros2 topic list | grep -E "image|wrench|joint"

# Should see /center_camera/image, /left_camera/image, /right_camera/image,
# /fts_broadcaster/wrench, /joint_states
```

If topics missing, the scene didn't launch correctly — Ctrl+C and retry.
