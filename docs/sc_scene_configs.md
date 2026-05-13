# SC plug insertion — scene configurations for data collection

> **Goal:** Record diverse **SC → SC port** demonstrations (gamepad v2 lives under `datasets/teleop-dataset-gamepad-v2/sc`).  
> **Companion:** SFP/NIC-focused launch grids are in [`sfp_scene_configs.md`](sfp_scene_configs.md).

---

## What to vary

| Dimension | Launch knobs | Why it matters |
|-----------|----------------|----------------|
| **SC port slot on board** | `sc_port_0_translation`, optional `sc_port_1_present` | Lateral target moves — same idea as shifting SFP NIC rails |
| **SC mount rail** | `sc_mount_rail_0_translation`, second rail | Body the port sits on shifts in the slot |
| **Board pose** | `task_board_x`, `task_board_y`, `task_board_yaw` | Reach distance, perspective, approach angle |
| **Port orientation** | `sc_port_0_yaw` / `_roll` / `_pitch` (small) | Teaches alignment under slight cavity misalignment |
| **Clutter** | Enable NIC + `sc_port_1` / rails | Visual distractors (eval-like boards) |

**Fixed for this task:** use the **reversed** cable so the gripper holds the **SC** end:

- `cable_type:=sfp_sc_cable_reversed`

---

## Quick-start

### Terminal 1 — launch the scene

Pick a block from [Full Copy-Paste Launch Commands](#full-copy-paste-launch-commands) below.

### Terminal 2 — record (gamepad v2, SC dataset root)

**Helper script (recommended):**

```bash
bash /run/host/scratch2/atang/ws_aic/scripts/record_gamepad_v2_sc.sh --resume=true
```

**Full `pixi` one-liner (when you want to override flags or aren't using the helper):**

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_gamepad_ee --teleop.id=aic \
  --teleop.stick_deadzone=0.02 --teleop.stick_expo=1.0 \
  --teleop.trigger_deadzone=0.01 --teleop.trigger_expo=1.0 \
  --teleop.low_command_scaling=0.04 \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=atang/aic_sc_demos_gamepad_v2 \
  --dataset.root=/run/host/scratch2/atang/ws_aic/datasets/teleop-dataset-gamepad-v2/sc \
  --dataset.single_task="Insert SC connector into SC port" \
  --dataset.push_to_hub=false --dataset.private=true \
  --play_sounds=false --display_data=true --resume=true
```

> **First session on an empty root:** omit `--resume=true` once so LeRobot can create the dataset; every session after, keep it on.

See also: [`gamepad_v2_recording.md`](gamepad_v2_recording.md) for the dataset-root convention and the SFP equivalent.

### Terminal 3 (optional)

```bash
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

---

## Baseline SC scene (sanity check)

Matches [`teleop_guide.md`](teleop_guide.md) §1 — same as **SC-1.1** below.

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## Group 1 — SC port lateral translation

Vary where the **opening** sits along the board. Aim for **~5 demos per row** when building a dense set.

| Config | `sc_port_0_translation` | Notes |
|--------|-------------------------|--------|
| **SC-1.1** | `-0.04` | Default “centered” |
| **SC-1.2** | `-0.02` | Shifted — tighter lateral margin |
| **SC-1.3** | `0.0` | Other end of travel |
| **SC-1.4** | `0.03` | Positive shift (if your xacro allows) |

→ [Copy-paste blocks](#full-copy-paste-launch-commands) (Group 1)

---

## Group 2 — SC mount rail translation

| Config | `sc_mount_rail_0_translation` |
|--------|----------------------------------|
| **SC-2.1** | `-0.09` (default) |
| **SC-2.2** | `-0.06` |
| **SC-2.3** | `-0.12` |

→ [Copy-paste blocks](#full-copy-paste-launch-commands) (Group 2)

---

## Group 3 — Board pose (reach & yaw)

| Config | Board pose |
|--------|------------|
| **SC-3.1** | default (no extra `task_board_*`) |
| **SC-3.2** | `task_board_x:=0.13 task_board_y:=-0.2` |
| **SC-3.3** | `task_board_x:=0.17 task_board_y:=-0.2` |
| **SC-3.4** | `task_board_x:=0.15 task_board_y:=-0.15 task_board_yaw:=3.05` |

→ [Copy-paste blocks](#full-copy-paste-launch-commands) (Group 3)

---

## Group 4 — Small port orientation offsets

| Config | Change |
|--------|--------|
| **SC-4.1** | `sc_port_0_yaw:=0.05` |
| **SC-4.2** | `sc_port_0_yaw:=-0.05` |

→ [Copy-paste blocks](#full-copy-paste-launch-commands) (Group 4)

---

## Group 5 — Clutter (distractors)

| Config | Idea |
|--------|------|
| **SC-5.1** | NIC + **two** SC ports (task port + distractor) |
| **SC-5.2** | SC port + board shifted (like SFP group 8.4) |
| **SC-5.3** | Slot-1 style clutter + board yaw |
| **SC-5.4** | SC port shifted + distractor port |

→ [Copy-paste blocks](#full-copy-paste-launch-commands) (Group 5)

Tune `sc_port_1_*` if your board layout differs — see `spawn_task_board.launch.py`.

---

## Episode count summary

| Group | Focus | Configs | Suggested episodes |
|-------|-------|---------|-------------------|
| 1 | Port lateral | 4 | 20 |
| 2 | Rail translation | 3 | 15 |
| 3 | Board pose | 4 | 20 |
| 4 | Port yaw | 2 | 10 |
| 5 | Clutter | 4 | 20 |
| **Total** | | **17** | **~85** |

---

## Recommended session plan

| Session | Configs | Focus |
|---------|---------|-------|
| **1** | SC-1.1–1.4 | Port translation sweep |
| **2** | SC-2.1–2.3, SC-4.1–4.2 | Rail + small yaw |
| **3** | SC-3.1–3.4 | Board pose |
| **4** | SC-5.1–5.4 | Distractors |

---

## Full Copy-Paste Launch Commands

All commands assume:

```bash
source ~/lab/ws_aic/setup_dev.sh
```

omitted in each block below for brevity — add it once per terminal session.

### Group 1 — SC port lateral translation

**SC-1.1 — translation −0.04 (default)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.2 — translation −0.02**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.3 — translation 0.0**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.4 — translation +0.03**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.03 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

### Group 2 — SC mount rail translation

**SC-2.1 — rail −0.09 (default)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-2.2 — rail −0.06**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.06
```

**SC-2.3 — rail −0.12**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.12
```

### Group 3 — Board pose

**SC-3.1 — default board (no `task_board_*`; same as SC-2.1 geometry)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-3.2 — board closer (x=0.13)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.13 task_board_y:=-0.2
```

**SC-3.3 — board further (x=0.17)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.17 task_board_y:=-0.2
```

**SC-3.4 — board y + yaw**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.15 task_board_y:=-0.15 task_board_yaw:=3.05
```

### Group 4 — Port yaw offsets

**SC-4.1 — sc_port_0 yaw +0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 sc_port_0_yaw:=0.05 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-4.2 — sc_port_0 yaw −0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 sc_port_0_yaw:=-0.05 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

### Group 5 — Clutter (distractors)

**SC-5.1 — NIC + task SC port + second SC port**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sc_port_1_present:=true sc_port_1_translation:=0.03 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-5.2 — SC ports + board position shift**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.01 \
  sc_port_1_present:=true sc_port_1_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_x:=0.17 task_board_y:=-0.15
```

**SC-5.3 — NIC slot 1 + SC + board yaw**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  nic_card_mount_1_present:=true \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_port_1_present:=true sc_port_1_translation:=-0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  task_board_yaw:=3.1
```

**SC-5.4 — SC port + shifted second port**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.02 \
  sc_port_1_present:=true sc_port_1_translation:=0.03 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## Merge for training

SC v2 recordings should live in **`datasets/teleop-dataset-gamepad-v2/sc`**, then rebuild the unified dataset:

```bash
cd /scratch2/atang/ws_aic/src/aic
pixi run python /scratch2/atang/ws_aic/scripts/build_combined_v2.py \
  --out /scratch2/atang/ws_aic/datasets/teleop-dataset-combined-v2 \
  --drop-failed
```

(`build_combined_v2.py` ingests **`gamepad-v2/sc`** as `task_index=1` when that directory exists.)

---

## Progress checklist

### Group 1 — Port translation
- [ ] **SC-1.1** trans=−0.04 (ep ___–___)
- [ ] **SC-1.2** trans=−0.02 (ep ___–___)
- [ ] **SC-1.3** trans=0.0 (ep ___–___)
- [ ] **SC-1.4** trans=+0.03 (ep ___–___)

### Group 2 — Rail translation
- [ ] **SC-2.1** rail=−0.09 (ep ___–___)
- [ ] **SC-2.2** rail=−0.06 (ep ___–___)
- [ ] **SC-2.3** rail=−0.12 (ep ___–___)

### Group 3 — Board pose
- [ ] **SC-3.1** default (ep ___–___)
- [ ] **SC-3.2** x=0.13 (ep ___–___)
- [ ] **SC-3.3** x=0.17 (ep ___–___)
- [ ] **SC-3.4** y/yaw (ep ___–___)

### Group 4 — Port yaw
- [ ] **SC-4.1** yaw=+0.05 (ep ___–___)
- [ ] **SC-4.2** yaw=−0.05 (ep ___–___)

### Group 5 — Clutter
- [ ] **SC-5.1** NIC + 2× SC (ep ___–___)
- [ ] **SC-5.2** SC + board shift (ep ___–___)
- [ ] **SC-5.3** slot1 + yaw (ep ___–___)
- [ ] **SC-5.4** two ports shifted (ep ___–___)
