# SFP Insertion — Scene Configurations for **Automated** Data Collection

> **Goal:** ~120 episodes across **40 configurations** collected by `scripts/auto_collect.py`.
> **Rule:** 3 episodes per configuration. Each episode = one Gazebo launch (see workflow below).
>
> Complementary to the 200-episode teleop set in [`sfp_scene_configs.md`](sfp_scene_configs.md).
> Mirrors the same variation axes but tuned for what the auto script can reliably solve.

---

## Why a different config set from teleop?

The automated script (`scripts/auto_collect.py`) is **not a vision policy** — it uses ground-truth TF frames and a CheatCode-style P+I controller. That changes which configs are valuable:

| Dimension | Teleop (human) | Auto (script) |
|-----------|----------------|----------------|
| **Visual distractors** | Challenging to ignore | Free — script uses TF, not vision. **Include more of them.** |
| **Extreme NIC translation** (`±0.036`) | Good skill-test | Unreliable — clipped to eval range (`±0.023`). |
| **Large NIC yaw** (`±0.08`) | Good skill-test | Unreliable — orientation alignment has limits. Dropped. |
| **Combined heavy variations** | Great realism | Failure-prone. Kept only moderate combos. |
| **Cable grasp noise** | Natural sim variance | Handled by P+I integrator — still useful for diversity. |

Net effect: **drop the extreme/beyond-eval configs**, **add more distractor clutter** (eval-realistic busy boards), and keep slot + translation + board-pose coverage intact.

---

## What varies (recap)

Same eval axes as the teleop doc — `nic_card_mount_{0..4}_{present,translation,yaw}`, `task_board_{x,y,yaw}`, plus distractor flags (`sc_port_{0,1}_present`, `sfp_mount_rail_{0,1}_present`, `lc_mount_rail_{0,1}_present`, `sc_mount_rail_{0,1}_present`).

**Fixed:** cameras, robot home pose, cable type (`sfp_sc_cable`).

---

## Recording Workflow — One Episode Per Gazebo Launch

See [`auto_collection_guide.md`](auto_collection_guide.md) for full details.

### Terminal 1 — Launch the scene

Replace `<CONFIG_PARAMS>` with the per-config parameters in the tables below.

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  <CONFIG_PARAMS>
```

### Terminal 2 — Run the auto collector (FIRST episode of the dataset)

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success
```

### Terminal 2 — EVERY subsequent episode (add `--resume`)

```bash
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success --resume
```

> **After each successful episode:** Ctrl-C Gazebo, change `<CONFIG_PARAMS>` (or keep them to collect another ep of the same config), relaunch, rerun the `--resume` command.

### Terminal 3 (optional) — Watch insertion + force

```bash
ros2 topic echo /scoring/insertion_event
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

---

## Group A — NIC Rail Slots (15 episodes)

Coverage of all 5 Y-positions. The biggest visual change between configs — the card appears in a completely different place.

All use default board pose (`x=0.15, y=-0.2, yaw=3.1415`) and centered NIC translation.

| Config | NIC slot | Params | Episodes |
|--------|----------|--------|----------|
| **A.1** | Slot 0 | `nic_card_mount_0_present:=true` | 3 |
| **A.2** | Slot 1 | `nic_card_mount_1_present:=true` | 3 |
| **A.3** | Slot 2 | `nic_card_mount_2_present:=true` | 3 |
| **A.4** | Slot 3 | `nic_card_mount_3_present:=true` | 3 |
| **A.5** | Slot 4 | `nic_card_mount_4_present:=true` | 3 |

---

## Group B — NIC Translation on Slot 0 (15 episodes)

Slot 0, sliding along the rail. Full **eval clamp** range, no beyond-eval torture tests.

| Config | Translation | Params | Episodes |
|--------|-------------|--------|----------|
| **B.1** | 0.0 (center) | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0` | 3 |
| **B.2** | +0.01 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01` | 3 |
| **B.3** | -0.01 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01` | 3 |
| **B.4** | +0.023 (eval max) | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023` | 3 |
| **B.5** | -0.021 (eval min) | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021` | 3 |

---

## Group C — NIC Translation on Slots 1–4 (18 episodes)

Cross-slot translation diversity.

| Config | Slot | Translation | Params | Episodes |
|--------|------|-------------|--------|----------|
| **C.1** | 1 | +0.015 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.015` | 3 |
| **C.2** | 1 | -0.015 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015` | 3 |
| **C.3** | 2 | +0.015 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015` | 3 |
| **C.4** | 2 | -0.015 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.015` | 3 |
| **C.5** | 3 | +0.01 | `nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.01` | 3 |
| **C.6** | 4 | -0.01 | `nic_card_mount_4_present:=true nic_card_mount_4_translation:=-0.01` | 3 |

---

## Group D — Board Position (18 episodes)

Board shifted in X (forward/back) and Y (left/right). Same ranges as eval.

| Config | Slot | Board x | Board y | Params | Episodes |
|--------|------|---------|---------|--------|----------|
| **D.1** | 0 | 0.15 | -0.2 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.2` | 3 |
| **D.2** | 0 | 0.13 | -0.2 | `nic_card_mount_0_present:=true task_board_x:=0.13 task_board_y:=-0.2` | 3 |
| **D.3** | 0 | 0.17 | -0.2 | `nic_card_mount_0_present:=true task_board_x:=0.17 task_board_y:=-0.2` | 3 |
| **D.4** | 0 | 0.15 | -0.15 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.15` | 3 |
| **D.5** | 0 | 0.15 | -0.25 | `nic_card_mount_0_present:=true task_board_x:=0.15 task_board_y:=-0.25` | 3 |
| **D.6** | 1 | 0.16 | -0.18 | `nic_card_mount_1_present:=true task_board_x:=0.16 task_board_y:=-0.18` | 3 |

---

## Group E — Board Yaw (12 episodes)

Board rotated on Z. Moderate yaw only — auto script orientation tracking handles ~±0.1 rad from nominal `3.1415` reliably.

| Config | Slot | Board yaw | Params | Episodes |
|--------|------|-----------|--------|----------|
| **E.1** | 0 | 3.05 | `nic_card_mount_0_present:=true task_board_yaw:=3.05` | 3 |
| **E.2** | 0 | 3.24 | `nic_card_mount_0_present:=true task_board_yaw:=3.24` | 3 |
| **E.3** | 1 | 3.10 | `nic_card_mount_1_present:=true task_board_yaw:=3.10` | 3 |
| **E.4** | 2 | 3.15 | `nic_card_mount_2_present:=true task_board_yaw:=3.15` | 3 |

---

## Group F — Combined (moderate) (15 episodes)

Multi-axis variation, but **moderate** — stays within what the P+I controller can reliably track. No extreme combos.

| Config | Slot | NIC trans | Board x | Board y | Board yaw | Params | Episodes |
|--------|------|-----------|---------|---------|-----------|--------|----------|
| **F.1** | 0 | +0.01 | 0.15 | -0.2 | 3.10 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01 task_board_yaw:=3.10` | 3 |
| **F.2** | 1 | -0.01 | 0.16 | -0.22 | 3.1415 | `nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.01 task_board_x:=0.16 task_board_y:=-0.22` | 3 |
| **F.3** | 2 | +0.015 | 0.14 | -0.2 | 3.2 | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015 task_board_x:=0.14 task_board_yaw:=3.2` | 3 |
| **F.4** | 3 | 0.0 | 0.16 | -0.18 | 3.1415 | `nic_card_mount_3_present:=true task_board_x:=0.16 task_board_y:=-0.18` | 3 |
| **F.5** | 4 | -0.005 | 0.15 | -0.2 | 3.10 | `nic_card_mount_4_present:=true nic_card_mount_4_translation:=-0.005 task_board_yaw:=3.10` | 3 |

---

## Group G — Distractors / Eval-Like Clutter (27 episodes)

**This is the biggest group on purpose.** Distractors cost nothing for the auto script (TF-driven) but are the #1 reason a trained vision policy will struggle at eval — real eval boards are busy. Every episode here is a "free" robustness sample.

| Config | Slot | NIC trans | Distractors | Params | Ep |
|--------|------|-----------|-------------|--------|----|
| **G.1** | 0 | 0.0 | SC port 0 at -0.04 | `nic_card_mount_0_present:=true sc_port_0_present:=true sc_port_0_translation:=-0.04` | 3 |
| **G.2** | 0 | +0.01 | SC port 0 at +0.03 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01 sc_port_0_present:=true sc_port_0_translation:=0.03` | 3 |
| **G.3** | 1 | 0.0 | SC port 0 + SFP rail 0 | `nic_card_mount_1_present:=true sc_port_0_present:=true sc_port_0_translation:=-0.02 sfp_mount_rail_0_present:=true` | 3 |
| **G.4** | 2 | 0.0 | LC mount rail 0 | `nic_card_mount_2_present:=true lc_mount_rail_0_present:=true` | 3 |
| **G.5** | 0 | 0.0 | SC ports 0 + 1 | `nic_card_mount_0_present:=true sc_port_0_present:=true sc_port_1_present:=true sc_port_1_translation:=0.04` | 3 |
| **G.6** | 3 | 0.0 | SC mount rail 0 + SFP rail 1 | `nic_card_mount_3_present:=true sc_mount_rail_0_present:=true sfp_mount_rail_1_present:=true` | 3 |
| **G.7** | 0 | +0.015 | Full clutter: SC×2, SFP rail 0, LC rail 0 | `nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 sc_port_0_present:=true sc_port_1_present:=true sc_port_1_translation:=0.04 sfp_mount_rail_0_present:=true lc_mount_rail_0_present:=true` | 3 |
| **G.8** | 1 | 0.0 | Full clutter + board yaw 3.1 | `nic_card_mount_1_present:=true sc_port_0_present:=true sc_mount_rail_0_present:=true sfp_mount_rail_1_present:=true task_board_yaw:=3.10` | 3 |
| **G.9** | 2 | +0.01 | Full clutter + board shift | `nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 sc_port_0_present:=true sc_port_1_present:=true lc_mount_rail_0_present:=true sfp_mount_rail_0_present:=true task_board_x:=0.16 task_board_y:=-0.18` | 3 |

---

## Episode Count Summary

| Group | Focus | Configs | Episodes |
|-------|-------|---------|----------|
| A — Rail slots | Which slot (0–4) | 5 | 15 |
| B — NIC translation (slot 0) | Lateral shift | 5 | 15 |
| C — NIC translation (slots 1–4) | Slot + shift combos | 6 | 18 |
| D — Board position | x, y variation | 6 | 18 |
| E — Board yaw | Moderate rotation | 4 | 12 |
| F — Combined | Moderate multi-var | 5 | 15 |
| G — Distractors | Eval-like clutter | 9 | 27 |
| **Total** | | **40** | **120** |

---

## Priority Order

If you can't finish all 120, here's what matters most:

1. **Groups A + B** (30 ep) — all 5 slots + full eval translation range. **Minimum viable dataset.**
2. **Group C** (18 ep) — cross-slot translation coverage.
3. **Groups D + E** (30 ep) — board pose coverage.
4. **Group G** (27 ep) — distractors. **Critical** for a vision policy that will see busy eval boards.
5. **Group F** (15 ep) — combined moderate variation.

---

## Session Plan

Each session is **~10 minutes per episode** end-to-end (Gazebo launch ~40s, insertion ~30s, save ~10s, kill + relaunch). For 120 episodes, budget roughly 20 hours total — split across multiple sittings.

| Session | Configs | Episodes | Focus |
|---------|---------|----------|-------|
| 1 | A.1–A.5 | 15 | All 5 rail slots |
| 2 | B.1–B.5 | 15 | Slot 0 translation range |
| 3 | C.1–C.6 | 18 | Cross-slot translation |
| 4 | D.1–D.6 | 18 | Board position |
| 5 | E.1–E.4, F.1–F.5 | 27 | Board yaw + moderate combined |
| 6 | G.1–G.9 | 27 | Distractors (the big one) |

> If auto_collect fails on a config after 5 attempts, skip it and note it in the tracker — we'll review those edge cases manually.

---

## Full Copy-Paste Launch Commands

All commands share the Terminal-2 `auto_collect.py` line from the workflow above. Only the Terminal-1 `ros2 launch` command changes.

### Group A — Rail Slots

**A.1 — Slot 0**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true
```

**A.2 — Slot 1**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true
```

**A.3 — Slot 2**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true
```

**A.4 — Slot 3**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true
```

**A.5 — Slot 4**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true
```

### Group B — NIC Translation (Slot 0)

**B.1 — Center**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.0
```

**B.2 — Right +0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01
```

**B.3 — Left -0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.01
```

**B.4 — Eval max +0.023**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.023
```

**B.5 — Eval min -0.021**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.021
```

### Group C — NIC Translation (Slots 1–4)

**C.1 — Slot 1, +0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=0.015
```

**C.2 — Slot 1, -0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.015
```

**C.3 — Slot 2, +0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015
```

**C.4 — Slot 2, -0.015**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=-0.015
```

**C.5 — Slot 3, +0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.01
```

**C.6 — Slot 4, -0.01**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true nic_card_mount_4_translation:=-0.01
```

### Group D — Board Position

**D.1 — Default pose**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.2
```

**D.2 — Closer (x=0.13)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.13 task_board_y:=-0.2
```

**D.3 — Further (x=0.17)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.17 task_board_y:=-0.2
```

**D.4 — Right (y=-0.15)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.15
```

**D.5 — Left (y=-0.25)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  task_board_x:=0.15 task_board_y:=-0.25
```

**D.6 — Slot 1, shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true \
  task_board_x:=0.16 task_board_y:=-0.18
```

### Group E — Board Yaw

**E.1 — Yaw 3.05 (Slot 0)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true task_board_yaw:=3.05
```

**E.2 — Yaw 3.24 (Slot 0)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true task_board_yaw:=3.24
```

**E.3 — Yaw 3.10 (Slot 1)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true task_board_yaw:=3.10
```

**E.4 — Yaw 3.15 (Slot 2)**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true task_board_yaw:=3.15
```

### Group F — Combined (moderate)

**F.1 — Slot 0, trans +0.01, yaw 3.10**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01 \
  task_board_yaw:=3.10
```

**F.2 — Slot 1, trans -0.01, board shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.01 \
  task_board_x:=0.16 task_board_y:=-0.22
```

**F.3 — Slot 2, trans +0.015, board yaw 3.2**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.015 \
  task_board_x:=0.14 task_board_yaw:=3.2
```

**F.4 — Slot 3, board shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true \
  task_board_x:=0.16 task_board_y:=-0.18
```

**F.5 — Slot 4, trans -0.005, yaw 3.10**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_4_present:=true nic_card_mount_4_translation:=-0.005 \
  task_board_yaw:=3.10
```

### Group G — Distractors / Eval-Like Clutter

**G.1 — Slot 0 + SC port 0**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```

**G.2 — Slot 0, trans +0.01, SC port shifted**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.01 \
  sc_port_0_present:=true sc_port_0_translation:=0.03
```

**G.3 — Slot 1 + SC port + SFP rail**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true \
  sc_port_0_present:=true sc_port_0_translation:=-0.02 \
  sfp_mount_rail_0_present:=true
```

**G.4 — Slot 2 + LC mount rail**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true \
  lc_mount_rail_0_present:=true
```

**G.5 — Slot 0 + both SC ports**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true \
  sc_port_0_present:=true \
  sc_port_1_present:=true sc_port_1_translation:=0.04
```

**G.6 — Slot 3 + SC rail + SFP rail 1**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_3_present:=true \
  sc_mount_rail_0_present:=true \
  sfp_mount_rail_1_present:=true
```

**G.7 — Slot 0 + full clutter**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.015 \
  sc_port_0_present:=true \
  sc_port_1_present:=true sc_port_1_translation:=0.04 \
  sfp_mount_rail_0_present:=true \
  lc_mount_rail_0_present:=true
```

**G.8 — Slot 1 + full clutter + yaw 3.10**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_1_present:=true \
  sc_port_0_present:=true \
  sc_mount_rail_0_present:=true \
  sfp_mount_rail_1_present:=true \
  task_board_yaw:=3.10
```

**G.9 — Slot 2 + full clutter + board shift**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.01 \
  sc_port_0_present:=true \
  sc_port_1_present:=true \
  lc_mount_rail_0_present:=true \
  sfp_mount_rail_0_present:=true \
  task_board_x:=0.16 task_board_y:=-0.18
```

---

## Progress Checklist (mirrors the GitHub tracker issue)

### Group A — Rail Slots
- [ ] **A.1** Slot 0 (ep ___–___)
- [ ] **A.2** Slot 1 (ep ___–___)
- [ ] **A.3** Slot 2 (ep ___–___)
- [ ] **A.4** Slot 3 (ep ___–___)
- [ ] **A.5** Slot 4 (ep ___–___)

### Group B — NIC Translation, Slot 0
- [ ] **B.1** trans=0.0 (ep ___–___)
- [ ] **B.2** trans=+0.01 (ep ___–___)
- [ ] **B.3** trans=-0.01 (ep ___–___)
- [ ] **B.4** trans=+0.023 (ep ___–___)
- [ ] **B.5** trans=-0.021 (ep ___–___)

### Group C — NIC Translation, Slots 1–4
- [ ] **C.1** slot 1, +0.015 (ep ___–___)
- [ ] **C.2** slot 1, -0.015 (ep ___–___)
- [ ] **C.3** slot 2, +0.015 (ep ___–___)
- [ ] **C.4** slot 2, -0.015 (ep ___–___)
- [ ] **C.5** slot 3, +0.01 (ep ___–___)
- [ ] **C.6** slot 4, -0.01 (ep ___–___)

### Group D — Board Position
- [ ] **D.1** default pose (ep ___–___)
- [ ] **D.2** x=0.13 closer (ep ___–___)
- [ ] **D.3** x=0.17 further (ep ___–___)
- [ ] **D.4** y=-0.15 right (ep ___–___)
- [ ] **D.5** y=-0.25 left (ep ___–___)
- [ ] **D.6** slot 1 + shift (ep ___–___)

### Group E — Board Yaw
- [ ] **E.1** slot 0, yaw=3.05 (ep ___–___)
- [ ] **E.2** slot 0, yaw=3.24 (ep ___–___)
- [ ] **E.3** slot 1, yaw=3.10 (ep ___–___)
- [ ] **E.4** slot 2, yaw=3.15 (ep ___–___)

### Group F — Combined (moderate)
- [ ] **F.1** slot 0 multi (ep ___–___)
- [ ] **F.2** slot 1 multi (ep ___–___)
- [ ] **F.3** slot 2 multi (ep ___–___)
- [ ] **F.4** slot 3 multi (ep ___–___)
- [ ] **F.5** slot 4 multi (ep ___–___)

### Group G — Distractors
- [ ] **G.1** SC port 0 (ep ___–___)
- [ ] **G.2** SC port shifted (ep ___–___)
- [ ] **G.3** slot 1 + SC + SFP rail (ep ___–___)
- [ ] **G.4** slot 2 + LC rail (ep ___–___)
- [ ] **G.5** both SC ports (ep ___–___)
- [ ] **G.6** slot 3 + SC rail + SFP rail 1 (ep ___–___)
- [ ] **G.7** full clutter slot 0 (ep ___–___)
- [ ] **G.8** full clutter slot 1 + yaw (ep ___–___)
- [ ] **G.9** full clutter slot 2 + shift (ep ___–___)
