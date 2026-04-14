# Automated Demo Collection Guide

## What This Is

`scripts/auto_collect.py` is an automated data collection script that generates LeRobot-compatible training demos **without a human operator**. It replicates the targeting logic from CheatCode (the ground-truth expert policy) but outputs velocity commands — the same action format that `lerobot-record` produces during manual teleoperation.

The result: a dataset that is **byte-for-byte compatible** with our teleop dataset and can be merged for training.

## How It Works

### Architecture

```
Ground-truth TF frames          Proportional Controller           Robot
─────────────────────           ───────────────────────           ─────
Port position (from /tf)  ───►  target_pose = port + offset  ──►  velocity = K × (target - current)
Plug position (from /tf)  ───►  orientation alignment        ──►  6D twist command (MODE_VELOCITY)
                                                                     │
                                                                     ▼
                                                              LeRobot Dataset
                                                              (same format as teleop)
```

1. **TF lookups** — uses ground-truth frames (`ground_truth:=true`) to find the exact port and plug positions, same as CheatCode
2. **Proportional controller** — converts the position error into a velocity command: `velocity = gain × (target - current)`, clamped to safe limits
3. **Recording** — captures observations (26-dim state + 3 camera images) and actions (6D velocity) using the same `LeRobotDataset` API that `lerobot-record` uses internally
4. **Success filtering** — monitors `/scoring/insertion_event` and only saves episodes where insertion actually succeeded
5. **Safety monitoring** — tracks force, collisions, and stuck conditions to discard bad episodes

### Insertion Phases

Each episode runs through 4 phases mimicking human teleoperation:

| Phase | Duration | Speed | What happens |
|-------|----------|-------|-------------|
| **1. Approach** | ~8s | 0.04 m/s | Move above the port, align orientation |
| **2. Fine align** | ~4s | 0.02 m/s | Lower closer, correct lateral errors |
| **3. Insert** | ~12s | 0.008 m/s | Slow descent along insertion axis |
| **4. Hold** | ~3s | 0 m/s | Wait for insertion event confirmation |

### Episode Discard Conditions

Not all attempts succeed. The script automatically discards bad episodes:

| Condition | Action | Why |
|-----------|--------|-----|
| No insertion event detected | Discard | Failed attempt |
| Force > 30 N | Abort + discard | Dangerously high, would damage real hardware |
| Force > 20 N for > 1s | Discard (optional) | Would get −12 penalty during eval |
| Off-limit collision | Abort + discard | Would get −24 penalty during eval |
| Robot stuck for > 2s | Abort + discard | Controller diverged or jammed |
| Episode < 10 frames | Discard | Too short to be useful |

## Prerequisites

- Gazebo must be running with **`ground_truth:=true`** (the script needs TF frames for port/plug positions)
- **`start_aic_engine:=false`** (we don't need the scoring engine, just the cable plugin)
- A cable must be spawned and attached to the gripper

## How to Run

### Terminal 1 — Launch Gazebo scene

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true
```

Change `nic_card_mount_0_present:=true` to any config from [`sfp_scene_configs.md`](sfp_scene_configs.md).

### Terminal 2 — Run auto collection

**First run (creates dataset):**

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-dataset/sfp_auto \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 10 \
  --max-attempts 20
```

**Subsequent runs (resume existing dataset):**

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-dataset/sfp_auto \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 10 \
  --max-attempts 20 \
  --resume
```

### Quick test (2 episodes, verify everything works)

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-dataset/sfp_auto_test \
  --repo-id atang/aic_sfp_auto_test \
  --num-episodes 2 \
  --max-attempts 5
```

After verifying the test works, delete it:

```bash
rm -rf /scratch2/atang/ws_aic/teleop-dataset/sfp_auto_test
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | (required) | Full path where dataset is stored |
| `--repo-id` | `atang/aic_sfp_auto` | Dataset identifier |
| `--num-episodes` | 20 | Target number of successful episodes |
| `--max-attempts` | 50 | Max total attempts before giving up |
| `--fps` | 30 | Recording frame rate |
| `--port-frame` | `task_board/nic_card_mount_0/sfp_port_0_link` | TF frame for the target port |
| `--plug-frame` | `cable_0/sfp_tip_link` | TF frame for the cable plug tip |
| `--resume` | false | Resume an existing dataset instead of creating new |
| `--vcodec` | `libsvtav1` | Video codec (`libsvtav1` or `h264`) |
| `--max-episode-time` | 60.0 | Max seconds per attempt |
| `--discard-high-force` | false | Discard episodes where force > 20 N for > 1s |

## Changing the Target Port or NIC Slot

The TF frame names follow this pattern:

- **Port:** `task_board/{module_name}/{port_name}_link`
- **Plug:** `cable_0/{plug_name}_link`

For SFP on different NIC slots:

| NIC Slot | `--port-frame` value |
|----------|---------------------|
| Slot 0 | `task_board/nic_card_mount_0/sfp_port_0_link` |
| Slot 1 | `task_board/nic_card_mount_1/sfp_port_0_link` |
| Slot 2 | `task_board/nic_card_mount_2/sfp_port_0_link` |
| Slot 3 | `task_board/nic_card_mount_3/sfp_port_0_link` |
| Slot 4 | `task_board/nic_card_mount_4/sfp_port_0_link` |

Example for slot 2:

```bash
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-dataset/sfp_auto \
  --repo-id atang/aic_sfp_auto \
  --port-frame "task_board/nic_card_mount_2/sfp_port_0_link" \
  --num-episodes 10 --resume
```

Make sure the Gazebo scene has the matching NIC mount present (e.g., `nic_card_mount_2_present:=true`).

## Tuning the Controller

If the robot doesn't behave well, adjust these parameters in the script (search for `ControllerGains`):

| Symptom | Parameter to change | Direction |
|---------|-------------------|-----------|
| Robot overshoots / oscillates | `kp_linear` | Reduce (try 1.0) |
| Robot moves too slowly | `max_linear_vel` | Increase (try 0.06) |
| Insertion never succeeds | z_offset final value | Make more negative (try -0.02) |
| Force readings too high | `insertion_linear_vel` | Reduce (try 0.005) |
| Orientation never aligns | `kp_angular` | Increase (try 3.0) |

## Output Format

The script produces a standard LeRobot dataset identical to `lerobot-record` output:

```
teleop-dataset/sfp_auto/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── stats.json         # Feature statistics
│   ├── tasks.parquet      # Task descriptions
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # State + action data (26-dim obs + 6-dim action)
└── videos/
    ├── observation.images.left_camera/
    │   └── chunk-000/file-000.mp4
    ├── observation.images.center_camera/
    │   └── chunk-000/file-000.mp4
    └── observation.images.right_camera/
        └── chunk-000/file-000.mp4
```

## Merging with Teleop Data

The automated dataset can be merged with manual teleop data for training. Both datasets have identical features and formats. See LeRobot documentation for merging datasets, or simply point `lerobot-train` at both dataset paths.

## How This Differs from Manual Teleop

| Aspect | Manual Teleop | Auto Collection |
|--------|--------------|----------------|
| **Approach diversity** | High — humans vary paths naturally | Low — same proportional controller path |
| **Recovery behaviors** | Yes — humans correct mistakes | No — controller never makes mistakes |
| **Speed** | ~30-40 demos/hour per person | Hundreds/hour, unattended |
| **Force awareness** | Human feels resistance and backs off | Controller pushes until abort threshold |
| **Coverage** | Limited by operator time | Can run overnight across many configs |

**Best practice:** use both. Teleop demos teach diversity and recovery. Auto demos provide broad coverage across scene configurations.

## Troubleshooting

**"TF frames not found"**
- Is `ground_truth:=true` set in the launch command?
- Is the NIC card mount actually present in the scene? (e.g., `nic_card_mount_0_present:=true`)
- Wait a few seconds after Gazebo starts before running the script.

**"Timed out waiting for controller_state"**
- Is the AIC controller active? The robot arm must be spawned and the controller running.

**"Timed out waiting for cameras"**
- Camera image topics use `lazy: true` bridging. The bridge starts publishing once something subscribes. Give it 10-15 seconds.

**Low success rate (<50%)**
- The proportional controller gains may need tuning for your specific scene config.
- Try increasing `max_episode_time` to give more time for insertion.

**All episodes discarded for high force**
- Remove `--discard-high-force` to keep them anyway (force data is still logged).
- Or reduce `insertion_linear_vel` in the script for gentler descent.
