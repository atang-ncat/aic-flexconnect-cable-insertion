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

Use one of the copy-paste blocks in [Group 1](#group-1--sc-port-lateral-translation) or later, or the **baseline** below.

### Terminal 2 — record (gamepad v2, SC dataset root)

```bash
bash /run/host/scratch2/atang/ws_aic/scripts/record_gamepad_v2_sc.sh --resume=true
```

First session on an empty root: run the same script **without** `--resume=true` once.  
Full paths and one-liner: [`gamepad_v2_recording.md`](gamepad_v2_recording.md).

### Terminal 3 (optional)

```bash
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

---

## Baseline SC scene (single session sanity check)

Matches [`teleop_guide.md`](teleop_guide.md) §1 — good default before sweeping parameters.

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
| **SC-1.1** | `-0.04` | Default “centered” (baseline above) |
| **SC-1.2** | `-0.02` | Shifted — tighter lateral margin |
| **SC-1.3** | `0.0` | Other extreme of travel |
| **SC-1.4** | `0.03` | Positive shift (if your xacro allows) |

**SC-1.1**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.2**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.3**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-1.4**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=0.03 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## Group 2 — SC mount rail translation

Keeps the port present but slides the **rail** the assembly rides on (composes with port translation).

| Config | `sc_mount_rail_0_translation` |
|--------|----------------------------------|
| **SC-2.1** | `-0.09` (default) |
| **SC-2.2** | `-0.06` |
| **SC-2.3** | `-0.12` |

**SC-2.2 example**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.06
```

---

## Group 3 — Board pose (reach & yaw)

Same **`task_board_*`** knobs as SFP collection — changes distance and viewing angle to the blue SC housing.

| Config | Board pose |
|--------|------------|
| **SC-3.1** | default (omit extra args) |
| **SC-3.2** | `task_board_x:=0.13 task_board_y:=-0.2` |
| **SC-3.3** | `task_board_x:=0.17 task_board_y:=-0.2` |
| **SC-3.4** | `task_board_x:=0.15 task_board_y:=-0.15 task_board_yaw:=3.05` |

**SC-3.3 example**
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

---

## Group 4 — Small port orientation offsets

Use **small** angles so physics stays stable; focus is learnable alignment diversity.

**SC-4.1 — yaw +0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 sc_port_0_yaw:=0.05 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

**SC-4.2 — yaw −0.05 rad**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  sc_port_0_present:=true sc_port_0_translation:=-0.04 sc_port_0_yaw:=-0.05 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09
```

---

## Group 5 — Clutter (second SC port + NIC visible)

Rough analog of SFP “distractors” — board looks busier; policy must ignore the non-task port.

**SC-5.1 — NIC slot 0 + SC port shifted + second port**
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

Tune `sc_port_1_*` if your board layout uses different defaults — see `spawn_task_board.launch.py` for the full argument list.

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

## Episode checklist (lightweight)

| Group | Configs | Purpose |
|-------|---------|---------|
| 1 | SC-1.1–1.4 | Port lateral sweep |
| 2 | SC-2.1–2.3 | Rail translation |
| 3 | SC-3.1–3.4 | Board pose |
| 4 | SC-4.1–4.2 | Port yaw |
| 5 | SC-5.1+ | Clutter |

Log episode ranges in your lab notebook the same way as [`sfp_scene_configs.md`](sfp_scene_configs.md).
