# SFP Insertion — Scene Configurations for Data Collection

> **Goal:** 200 episodes across ~40 configurations for a robust SFP insertion dataset.
> Record **5 episodes per configuration**, then move to the next one.

---

## What Varies During Evaluation (and therefore in our data)

Based on `qualification_phase.md` and `sample_config.yaml`, the eval randomizes:

| Dimension | What changes | Eval range | Why it matters |
|-----------|-------------|------------|----------------|
| **NIC rail slot** | Which of the 5 Y-positions (0–4) the NIC card sits on | Any of `nic_card_mount_0` through `_4` | Completely different approach trajectory for each slot |
| **NIC translation** | Slides NIC card along X on its rail | [-0.0215, 0.0234] m | Changes lateral alignment target |
| **NIC yaw** | Rotates NIC card on its rail | Small offsets (sample uses 0.0) | Changes insertion angle |
| **Board x** | Shifts board forward/back from robot | ~0.13–0.17 m (sample: 0.15) | Changes reach distance |
| **Board y** | Shifts board left/right | ~-0.25 to 0.0 m (sample: -0.2, 0.0) | Changes lateral approach |
| **Board yaw** | Rotates entire board | ~2.9–3.3 rad (sample: 3.0, 3.1415) | Changes approach angle |
| **Distractors** | SC ports, mount rails, extra NIC cards | Various | Visual clutter the policy must ignore |
| **Grasp noise** | Small perturbations in cable grasp | ~2mm, ~0.04 rad | Natural sim variance handles this |

**Fixed (does not vary):** cameras, robot home pose, cable type (`sfp_sc_cable`).

---

## Quick-Start: Recording Workflow

For full details see [`teleop_guide.md`](teleop_guide.md). The essentials are below.

### Terminal 1 — Launch the scene

Replace `<CONFIG_PARAMS>` with the parameters from each config section below.

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  <CONFIG_PARAMS>
```

### Terminal 2 — Record episodes

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic/
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=atang/aic_sfp_demos \
  --dataset.root=/run/host/scratch2/atang/ws_aic/teleop-dataset/sfp \
  --dataset.single_task="Insert SFP connector into SFP port on NIC card" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true \
  --resume=true
```

> **First time only:** remove `--resume=true` to create the dataset. Every session after, keep it.

### Terminal 3 (optional) — Monitor insertion & force

```bash
ros2 topic echo /scoring/insertion_event
```

```bash
source /ws_aic/install/setup.bash
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

### Keyboard controls (essentials)

| Key | Action |
|-----|--------|
| `W/S` | Move forward / backward (X) |
| `A/D` | Move left / right (Y) |
| `Q/E` | Move up / down (Z) |
| `I/K` | Rotate pitch |
| `J/L` | Rotate yaw |
| `U/O` | Rotate roll |
| `Right Arrow` | **Save episode** and start next |
| `Backspace` | Discard current episode |
| `Escape` | Stop recording and exit |
| `1` | Switch to low-speed mode (fine insertion) |
| `2` | Switch to high-speed mode (approach) |

---

## Group 1 — NIC Rail Slot Variation (25 episodes)

The NIC card can sit on any of 5 rail slots (different Y positions). This is the **biggest** visual change between trials — the card appears at completely different locations on the board.

All use default board pose (x=0.15, y=-0.2, yaw=3.1415) and center NIC translation (0.0).

| Config | NIC slot | Params | Episodes |
|--------|----------|--------|----------|
| **1.1** | Slot 0 | `nic_card_mount_0_present:=true` | 5 |
| **1.2** | Slot 1 | `nic_card_mount_1_present:=true` | 5 |
| **1.3** | Slot 2 | `nic_card_mount_2_present:=true` | 5 |
| **1.4** | Slot 3 | `nic_card_mount_3_present:=true` | 5 |
| **1.5** | Slot 4 | `nic_card_mount_4_present:=true` | 5 |

---

## Group 2 — NIC Translation on Slot 0 (30 episodes)

Slot 0 with the NIC card slid along its rail. Covers the full eval clamp range and some beyond.

| Config | Translation | Difficulty | Params | Episodes |
|--------|-------------|-----------|--------|----------|
| **2.1** | 0.0 (center) | Easy | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0` | 5 |
| **2.2** | +0.01 | Easy | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01` | 5 |
| **2.3** | -0.01 | Easy | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01` | 5 |
| **2.4** | +0.023 (eval max) | Medium | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023` | 5 |
| **2.5** | -0.021 (eval min) | Medium | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021` | 5 |
| **2.6** | +0.036 (xacro max) | Hard | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036` | 5 |

---

## Group 3 — NIC Translation on Other Slots (30 episodes)

Same translation variation but on slots 1 and 2 to combine slot + translation diversity.

| Config | Slot | Translation | Params | Episodes |
|--------|------|-------------|--------|----------|
| **3.1** | 1 | +0.015 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.015` | 5 |
| **3.2** | 1 | -0.015 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015` | 5 |
| **3.3** | 1 | +0.023 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.023` | 5 |
| **3.4** | 2 | 0.0 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.0` | 5 |
| **3.5** | 2 | +0.02 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.02` | 5 |
| **3.6** | 2 | -0.02 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.02` | 5 |

---

## Group 4 — Board Position Variation (30 episodes)

The entire board shifts in x (forward/back) and y (left/right). Changes how far the robot must reach and the camera perspective.

All use slot 0 with centered NIC translation.

| Config | Board x | Board y | Board yaw | Params | Episodes |
|--------|---------|---------|-----------|--------|----------|
| **4.1** | 0.15 | -0.2 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.2` | 5 |
| **4.2** | 0.13 | -0.2 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.13 task_board_y:=-0.2` | 5 |
| **4.3** | 0.17 | -0.2 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.17 task_board_y:=-0.2` | 5 |
| **4.4** | 0.15 | -0.15 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.15` | 5 |
| **4.5** | 0.15 | -0.25 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.25` | 5 |
| **4.6** | 0.17 | -0.15 | 3.1415 | `nic_card_mount_0_present:=true task_board_x:=0.17 task_board_y:=-0.15` | 5 |

---

## Group 5 — Board Yaw Variation (25 episodes)

Board rotated around Z axis. Combined with different slots and translations.

| Config | Slot | NIC trans | Board yaw | Params | Episodes |
|--------|------|-----------|-----------|--------|----------|
| **5.1** | 0 | 0.0 | 3.05 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 task_board_yaw:=3.05` | 5 |
| **5.2** | 0 | 0.0 | 3.24 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 task_board_yaw:=3.24` | 5 |
| **5.3** | 0 | +0.02 | 3.0 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 task_board_yaw:=3.0` | 5 |
| **5.4** | 1 | 0.0 | 3.05 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.0 task_board_yaw:=3.05` | 5 |
| **5.5** | 1 | -0.015 | 3.24 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015 task_board_yaw:=3.24` | 5 |

---

## Group 6 — NIC Yaw Offset (20 episodes)

The NIC card itself is slightly rotated on its rail. Changes the insertion angle.

| Config | Slot | NIC trans | NIC yaw | Params | Episodes |
|--------|------|-----------|---------|--------|----------|
| **6.1** | 0 | 0.0 | +0.05 | `nic_card_mount_0_present:=true nic_card_mount_0_yaw:=0.05` | 5 |
| **6.2** | 0 | 0.0 | -0.05 | `nic_card_mount_0_present:=true nic_card_mount_0_yaw:=-0.05` | 5 |
| **6.3** | 0 | +0.015 | +0.08 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 nic_card_mount_0_yaw:=0.08` | 5 |
| **6.4** | 1 | 0.0 | +0.05 | `nic_card_mount_1_present:=true nic_card_mount_1_yaw:=0.05` | 5 |

---

## Group 7 — Combined Variation (20 episodes)

Multiple dimensions varying at once — the hardest, most realistic configs. These mimic what eval actually looks like.

| Config | Slot | NIC trans | NIC yaw | Board x | Board y | Board yaw | Params | Episodes |
|--------|------|-----------|---------|---------|---------|-----------|--------|----------|
| **7.1** | 0 | +0.02 | +0.05 | 0.15 | -0.2 | 3.05 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 nic_card_mount_0_yaw:=0.05 task_board_yaw:=3.05` | 5 |
| **7.2** | 1 | -0.015 | -0.05 | 0.17 | -0.15 | 3.24 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015 nic_card_mount_1_yaw:=-0.05 task_board_x:=0.17 task_board_y:=-0.15 task_board_yaw:=3.24` | 5 |
| **7.3** | 2 | +0.01 | 0.0 | 0.13 | -0.25 | 3.1 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 task_board_x:=0.13 task_board_y:=-0.25 task_board_yaw:=3.1` | 5 |
| **7.4** | 3 | -0.01 | +0.03 | 0.16 | -0.18 | 3.2 | `nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.01 nic_card_mount_3_yaw:=0.03 task_board_x:=0.16 task_board_y:=-0.18 task_board_yaw:=3.2` | 5 |

---

## Group 8 — With Distractors (20 episodes)

Other board components present as visual distractors. During eval, the board often has SC ports, mount rails, and even extra NIC cards visible.

| Config | Slot | NIC trans | Distractors | Params | Episodes |
|--------|------|-----------|-------------|--------|----------|
| **8.1** | 0 | 0.005 | SC port | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 sc_port_0_present:=true sc_port_0_translation:=-0.04` | 5 |
| **8.2** | 0 | +0.02 | SC port shifted | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 sc_port_0_present:=true sc_port_0_translation:=0.03` | 5 |
| **8.3** | 1 | 0.0 | SC port + yaw | `nic_card_mount_1_present:=true sc_port_0_present:=true sc_port_0_translation:=-0.02 task_board_yaw:=3.1` | 5 |
| **8.4** | 0 | -0.01 | SC + board shift | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01 sc_port_0_present:=true sc_port_0_translation:=0.0 task_board_x:=0.17 task_board_y:=-0.15` | 5 |

---

## Episode Count Summary

| Group | Focus | Configs | Episodes |
|-------|-------|---------|----------|
| 1 — Rail slots | Which slot (0–4) | 5 | 25 |
| 2 — NIC translation (slot 0) | Lateral shift | 6 | 30 |
| 3 — NIC translation (slots 1–2) | Slot + shift combo | 6 | 30 |
| 4 — Board position | x, y variation | 6 | 30 |
| 5 — Board yaw | Rotation | 5 | 25 |
| 6 — NIC yaw | Card orientation | 4 | 20 |
| 7 — Combined | Multi-dimension | 4 | 20 |
| 8 — Distractors | Visual clutter | 4 | 20 |
| **Total** | | **40** | **200** |

---

## Priority Order for Recording

If you can't finish all 200, here's what matters most:

1. **Groups 1 + 2** (55 episodes) — Covers all 5 slots and the full NIC translation range. **Absolute minimum viable dataset.**
2. **Group 3** (30 episodes) — Slot + translation combos. Strongly recommended.
3. **Groups 4 + 5** (55 episodes) — Board pose variation. Important for real eval.
4. **Groups 6 + 7 + 8** (60 episodes) — NIC yaw, combined, and distractors. Polish for robustness.

---

## Recommended Session Plan

Each session is ~20 minutes of active recording (plus breaks).

| Session | Configs | Episodes | Focus |
|---------|---------|----------|-------|
| **1** | 1.1–1.5 | 25 | All 5 rail slots |
| **2** | 2.1–2.6 | 30 | NIC translation range |
| **3** | 3.1–3.6 | 30 | Translation on other slots |
| **4** | 4.1–4.6 | 30 | Board position |
| **5** | 5.1–5.5 | 25 | Board yaw |
| **6** | 6.1–6.4 | 20 | NIC yaw |
| **7** | 7.1–7.4, 8.1–8.4 | 40 | Combined + distractors |

> **Break 5 minutes between sessions.** Don't record more than 2 sessions in a row without a longer break.

---

## Full Copy-Paste Launch Commands

### Group 1 — Rail Slots

**1.1 — Slot 0**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true
```

**1.2 — Slot 1**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true
```

**1.3 — Slot 2**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true
```

**1.4 — Slot 3**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true
```

**1.5 — Slot 4**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true
```

### Group 2 — NIC Translation (Slot 0)

**2.1 — Center**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0
```

**2.2 — Right +0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01
```

**2.3 — Left -0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01
```

**2.4 — Eval max +0.023**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023
```

**2.5 — Eval min -0.021**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021
```

**2.6 — Beyond eval +0.036**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036
```

### Group 3 — NIC Translation (Slots 1–2)

**3.1 — Slot 1, +0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.015
```

**3.2 — Slot 1, -0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015
```

**3.3 — Slot 1, +0.023**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.023
```

**3.4 — Slot 2, center**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.0
```

**3.5 — Slot 2, +0.02**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.02
```

**3.6 — Slot 2, -0.02**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.02
```

### Group 4 — Board Position

**4.1 — Default pose**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.2
```

**4.2 — Board closer (x=0.13)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.13 task_board_y:=-0.2
```

**4.3 — Board further (x=0.17)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.17 task_board_y:=-0.2
```

**4.4 — Board shifted right (y=-0.15)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.15
```

**4.5 — Board shifted left (y=-0.25)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.25
```

**4.6 — Board further + shifted right**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.17 task_board_y:=-0.15
```

### Group 5 — Board Yaw

**5.1 — Yaw 3.05 (~5° CW)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_yaw:=3.05
```

**5.2 — Yaw 3.24 (~6° CCW)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_yaw:=3.24
```

**5.3 — Yaw 3.0 + NIC shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
  task_board_yaw:=3.0
```

**5.4 — Slot 1, yaw 3.05**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true \
  task_board_yaw:=3.05
```

**5.5 — Slot 1, yaw 3.24 + NIC shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015 \
  task_board_yaw:=3.24
```

### Group 6 — NIC Yaw

**6.1 — NIC yaw +0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_yaw:=0.05
```

**6.2 — NIC yaw -0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_yaw:=-0.05
```

**6.3 — NIC yaw + translation**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 nic_card_mount_0_yaw:=0.08
```

**6.4 — Slot 1, NIC yaw**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_yaw:=0.05
```

### Group 7 — Combined

**7.1**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 nic_card_mount_0_yaw:=0.05 \
  task_board_yaw:=3.05
```

**7.2**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015 nic_card_mount_1_yaw:=-0.05 \
  task_board_x:=0.17 task_board_y:=-0.15 task_board_yaw:=3.24
```

**7.3**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 \
  task_board_x:=0.13 task_board_y:=-0.25 task_board_yaw:=3.1
```

**7.4**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true nic_card_mount_3_translation:=-0.01 nic_card_mount_3_yaw:=0.03 \
  task_board_x:=0.16 task_board_y:=-0.18 task_board_yaw:=3.2
```

### Group 8 — Distractors

**8.1 — SC port present**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```

**8.2 — SC port shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
  sc_port_0_present:=true sc_port_0_translation:=0.03
```

**8.3 — Slot 1 + SC port + board yaw**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  task_board_yaw:=3.1
```

**8.4 — SC port + board position shift**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01 \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  task_board_x:=0.17 task_board_y:=-0.15
```

---

## Progress Checklist

### Group 1 — Rail Slots
- [ ] **1.1** Slot 0, center (ep ___–___)
- [ ] **1.2** Slot 1, center (ep ___–___)
- [ ] **1.3** Slot 2, center (ep ___–___)
- [ ] **1.4** Slot 3, center (ep ___–___)
- [ ] **1.5** Slot 4, center (ep ___–___)

### Group 2 — NIC Translation (Slot 0)
- [ ] **2.1** trans=0.0 (ep ___–___)
- [ ] **2.2** trans=+0.01 (ep ___–___)
- [ ] **2.3** trans=-0.01 (ep ___–___)
- [ ] **2.4** trans=+0.023 eval max (ep ___–___)
- [ ] **2.5** trans=-0.021 eval min (ep ___–___)
- [ ] **2.6** trans=+0.036 beyond eval (ep ___–___)

### Group 3 — NIC Translation (Slots 1–2)
- [ ] **3.1** slot 1, +0.015 (ep ___–___)
- [ ] **3.2** slot 1, -0.015 (ep ___–___)
- [ ] **3.3** slot 1, +0.023 (ep ___–___)
- [ ] **3.4** slot 2, center (ep ___–___)
- [ ] **3.5** slot 2, +0.02 (ep ___–___)
- [ ] **3.6** slot 2, -0.02 (ep ___–___)

### Group 4 — Board Position
- [ ] **4.1** default pose (ep ___–___)
- [ ] **4.2** x=0.13 closer (ep ___–___)
- [ ] **4.3** x=0.17 further (ep ___–___)
- [ ] **4.4** y=-0.15 right (ep ___–___)
- [ ] **4.5** y=-0.25 left (ep ___–___)
- [ ] **4.6** x=0.17 y=-0.15 (ep ___–___)

### Group 5 — Board Yaw
- [ ] **5.1** yaw=3.05 (ep ___–___)
- [ ] **5.2** yaw=3.24 (ep ___–___)
- [ ] **5.3** yaw=3.0 + NIC shift (ep ___–___)
- [ ] **5.4** slot 1, yaw=3.05 (ep ___–___)
- [ ] **5.5** slot 1, yaw=3.24 + NIC shift (ep ___–___)

### Group 6 — NIC Yaw
- [ ] **6.1** NIC yaw=+0.05 (ep ___–___)
- [ ] **6.2** NIC yaw=-0.05 (ep ___–___)
- [ ] **6.3** NIC yaw + translation (ep ___–___)
- [ ] **6.4** slot 1, NIC yaw (ep ___–___)

### Group 7 — Combined
- [ ] **7.1** slot 0 multi-var (ep ___–___)
- [ ] **7.2** slot 1 multi-var (ep ___–___)
- [ ] **7.3** slot 2 multi-var (ep ___–___)
- [ ] **7.4** slot 3 multi-var (ep ___–___)

### Group 8 — Distractors
- [ ] **8.1** SC port (ep ___–___)
- [ ] **8.2** SC port shifted (ep ___–___)
- [ ] **8.3** slot 1 + SC + yaw (ep ___–___)
- [ ] **8.4** SC + board shift (ep ___–___)
