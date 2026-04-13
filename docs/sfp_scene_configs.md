# SFP Insertion — Scene Configurations for Data Collection

> **Goal:** Collect a diverse, comprehensive dataset for the SFP-to-NIC insertion task.
> Record **4–5 episodes per configuration**, then move to the next one.
> All configs use `nic_card_mount_0` (slot 0, the default evaluation slot).

---

## Quick Reference

| Parameter | What it does | Range |
|-----------|-------------|-------|
| `nic_card_mount_0_translation` | Slides NIC card along X axis (left/right on board) | [-0.048, 0.036] m (xacro); eval clamps to [-0.0215, 0.0234] |
| `task_board_yaw` | Rotates entire board around Z axis | Default 3.1415 rad; small variations ±0.15 rad (~±9°) |
| `sc_port_0_present` / `sc_port_0_translation` | Adds SC port as visual distractor | [-0.055, 0.055] m |

---

## Base Command Template

All configs use this base. Replace `<PARAMS>` with the config-specific lines.

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  <PARAMS>
```

---

## Configurations

### Group A — Nominal / Centered (warm-up configs)

These are the "easy" configs. Use them first to build muscle memory and get clean baseline demos.

#### A1: Dead center, default yaw
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0
```
**Episodes: 5** | Difficulty: Easy | The default setup.

#### A2: Slight right shift, default yaw
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005
```
**Episodes: 4** | Difficulty: Easy | Barely different from center — good for consistency.

#### A3: Slight left shift, default yaw
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.005
```
**Episodes: 4** | Difficulty: Easy | Mirror of A2.

---

### Group B — Moderate Variation (within eval range)

The eval engine clamps NIC translation to [-0.0215, 0.0234]. These configs cover that range.

#### B1: Moderate right shift
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015
```
**Episodes: 5** | Difficulty: Medium

#### B2: Moderate left shift
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.015
```
**Episodes: 5** | Difficulty: Medium

#### B3: Right edge of eval range
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023
```
**Episodes: 5** | Difficulty: Medium-Hard | Near the eval clamp limit.

#### B4: Left edge of eval range
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021
```
**Episodes: 5** | Difficulty: Medium-Hard | Near the eval clamp limit.

---

### Group C — Board Yaw Variation

Same NIC translations but with the board rotated. Teaches the policy to handle approach angle changes.

#### C1: Board rotated clockwise, center NIC
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
  task_board_yaw:=3.05
```
**Episodes: 4** | Difficulty: Medium | ~5° clockwise from default.

#### C2: Board rotated counter-clockwise, center NIC
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
  task_board_yaw:=3.24
```
**Episodes: 4** | Difficulty: Medium | ~5° counter-clockwise from default.

#### C3: Board rotated clockwise + right shift
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 \
  task_board_yaw:=3.05
```
**Episodes: 4** | Difficulty: Medium-Hard | Combined variation.

#### C4: Board rotated counter-clockwise + left shift
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.015 \
  task_board_yaw:=3.24
```
**Episodes: 4** | Difficulty: Medium-Hard | Combined variation.

---

### Group D — Edge Cases (beyond eval range, for robustness)

These push past the eval clamp limits. The policy may never see these exact values during scoring, but training on extremes improves generalization at the boundaries.

#### D1: Far right (xacro max)
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036
```
**Episodes: 4** | Difficulty: Hard | Max rightward shift.

#### D2: Far left (xacro limit)
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.04
```
**Episodes: 4** | Difficulty: Hard | Near max leftward shift.

#### D3: Far right + board rotated
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036 \
  task_board_yaw:=3.0
```
**Episodes: 4** | Difficulty: Hard | Worst-case combined variation.

---

### Group E — With Distractors

Same insertion task, but the SC port is also present on the board. Teaches the policy not to get confused by extra geometry.

#### E1: Center NIC + SC port present
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```
**Episodes: 4** | Difficulty: Medium | Visual distractor.

#### E2: Shifted NIC + SC port present
```
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
  sc_port_0_present:=true sc_port_0_translation:=0.0
```
**Episodes: 4** | Difficulty: Medium-Hard | Port and distractor in different positions.

---

## Episode Count Summary

| Group | Configs | Episodes/Config | Total |
|-------|---------|----------------|-------|
| A — Nominal | 3 | 4–5 | ~13 |
| B — Moderate | 4 | 5 | 20 |
| C — Yaw variation | 4 | 4 | 16 |
| D — Edge cases | 3 | 4 | 12 |
| E — Distractors | 2 | 4 | 8 |
| **Total** | **16** | | **~69** |

This gives ~69 episodes. You can trim Group D or E if pressed for time, but try to get at least all of Groups A, B, and C (~49 episodes). The eval-range configs (A + B) are the highest priority.

---

## Recommended Recording Order

1. **A1** — Start here to warm up. Get 5 clean demos.
2. **A2, A3** — Still easy, slight variation. Build confidence.
3. **B1, B2** — Moderate shifts. Take your time.
4. **C1, C2** — Yaw changes. The approach feels different — practice a bit.
5. **B3, B4** — Edge of eval range. These matter for robustness.
6. **C3, C4** — Combined variation. The hardest "realistic" configs.
7. **E1, E2** — Distractor configs. Same skill, different scenery.
8. **D1, D2, D3** — Extreme edge cases. Do these last — they push your limits.

> **Take a 5-minute break after every 15–20 episodes.** Fatigue makes demos sloppy.

---

## Full Copy-Paste Commands

For convenience, here is every launch command ready to paste.

### A1
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0
```

### A2
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005
```

### A3
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.005
```

### B1
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015
```

### B2
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.015
```

### B3
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023
```

### B4
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021
```

### C1
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
  task_board_yaw:=3.05
```

### C2
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0 \
  task_board_yaw:=3.24
```

### C3
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 \
  task_board_yaw:=3.05
```

### C4
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.015 \
  task_board_yaw:=3.24
```

### D1
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036
```

### D2
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.04
```

### D3
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.036 \
  task_board_yaw:=3.0
```

### E1
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```

### E2
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.02 \
  sc_port_0_present:=true sc_port_0_translation:=0.0
```

---

## Tracking Your Progress

Use this checklist to track which configs you've completed. Mark the checkbox and note the episode range.

- [ ] **A1** — center, default yaw (episodes ___–___)
- [ ] **A2** — +0.005, default yaw (episodes ___–___)
- [ ] **A3** — -0.005, default yaw (episodes ___–___)
- [ ] **B1** — +0.015, default yaw (episodes ___–___)
- [ ] **B2** — -0.015, default yaw (episodes ___–___)
- [ ] **B3** — +0.023, eval edge (episodes ___–___)
- [ ] **B4** — -0.021, eval edge (episodes ___–___)
- [ ] **C1** — center, yaw 3.05 (episodes ___–___)
- [ ] **C2** — center, yaw 3.24 (episodes ___–___)
- [ ] **C3** — +0.015, yaw 3.05 (episodes ___–___)
- [ ] **C4** — -0.015, yaw 3.24 (episodes ___–___)
- [ ] **D1** — +0.036, far right (episodes ___–___)
- [ ] **D2** — -0.04, far left (episodes ___–___)
- [ ] **D3** — +0.036, yaw 3.0 (episodes ___–___)
- [ ] **E1** — +0.005, SC distractor (episodes ___–___)
- [ ] **E2** — +0.02, SC distractor (episodes ___–___)
