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
2. **Proportional controller + lateral P+I** — converts position error into a velocity command: `velocity = gain × (target - current)`, clamped to safe limits. During the fine-align and insert phases we additionally apply a CheatCode-style integrator on the plug's lateral (port-local XY) error to eliminate steady-state offset — without this, descent often jams at the port mouth.
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

The supported workflow is **one episode per Gazebo launch**, repeated until
you have enough episodes. See the "Recommended workflow" section further
down for a full explanation of why. Two terminals, one loop:

### Terminal 1 — Launch Gazebo (one launch = one episode attempt)

```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true
```

Swap NIC mount / cable type / port arguments as needed between launches.
See [`sfp_scene_configs.md`](sfp_scene_configs.md) for the full config
matrix.

### Terminal 2 — Run auto collection (once per launch)

**Very first episode (creates the dataset — do NOT use `--resume`):**

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success
```

This runs up to 5 attempts within the current Gazebo session until one
insertion succeeds, then exits. If none of the 5 attempts succeed, it
stops and nothing is written to the dataset for this launch.

**Every episode after that (always use `--resume` to append):**

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success --resume
```

With `--resume`, existing episodes are **never touched** — the script
validates the dataset schema and appends the new one. Without `--resume`,
if the dataset root already exists the script exits with a clear error
rather than overwriting.

### Loop for an evening of collection

1. Launch Gazebo (Terminal 1) with the config you want for this episode.
2. Run the collector (Terminal 2) with `--exit-on-success` (and `--resume`
   on every run except the very first).
3. When the script exits, `Ctrl-C` Gazebo in Terminal 1.
4. Relaunch Gazebo with the next config (or the same one — your call).
5. Go to step 2. Keep going until you have your target episode count.

### Quick smoke test (separate dataset, same workflow)

Use a throwaway dataset path so it doesn't mix with your real one:

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp_test \
  --repo-id atang/aic_sfp_auto_test \
  --num-episodes 1 --max-attempts 5 --exit-on-success
```

After verifying, delete it before starting real collection:

```bash
rm -rf /scratch2/atang/ws_aic/teleop-automated-dataset/sfp_test
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
| `--exit-on-success` | false | Stop as soon as one episode is saved. Recommended for the per-launch workflow below. |
| `--noise-scale` | 1.0 | Scales per-episode target-pose noise. See [Target-pose noise](#target-pose-noise). Pass `0` to disable. |
| `--seed` | (none) | Optional RNG seed for reproducible noise profiles. |
| `--reset-scene` | false | **No-op / broken.** Calling `/gz_server/reset_simulation` on this build crashes the `ros_gz_container` (ros2_control reloads inside the same process and segfaults). Flag is accepted for backward compatibility only. |

## Target-pose noise

Without any noise, the CheatCode expert produces trajectories that are nearly
identical episode-to-episode: the plug descends in a clean straight line with
minimal lateral action. That's mechanically perfect but makes the resulting
dataset *too sterile* — a policy trained only on these clean runs has never
seen a micro-correction and gets brittle when it ends up in a state slightly
off the demonstrated trajectory.

`--noise-scale` (default **1.0**) injects small per-phase perturbations into
the commanded target pose. Each episode samples one offset per phase, holds
it constant through that phase, and logs the values at the start of the
attempt. The lateral P+I integrator still references the **true** port
position, so it naturally corrects against the biased target — producing
realistic micro-corrections in the recorded velocity actions.

Ranges at `--noise-scale 1.0`:

| Phase | Lateral XY | Yaw (around insertion axis) |
|-------|------------|-----------------------------|
| Approach | ±10 mm | ±2° |
| Fine align | ±3 mm | ±1° |
| Insertion | ±1 mm | 0 |
| Hold | 0 | 0 |

Values are small enough that insertion still succeeds (well within the port's
mechanical tolerance), but large enough that each trajectory looks subtly
different and the recorded `linear.x / linear.y / angular.z` actions have
visible variance instead of being dead-zero most of the time.

Pass `--noise-scale 0` to reproduce the old deterministic behavior (useful for
debugging). Pass `--seed 42` along with it to get a reproducible profile
across identical configs.

## Recommended workflow: one episode per Gazebo launch

The scene only spawns **one** cable (`cable_0`). Once that cable is latched
into the port, any additional attempts in the same Gazebo session must either
rip it back out or wedge the arm — both produce garbage data. So the
supported unattended-ish workflow is **one episode per launch**, repeated
until you've collected the target number of episodes:

1. Launch Gazebo with your config:

   ```bash
   ros2 launch aic_bringup aic_gz_bringup.launch.py \
     ground_truth:=true start_aic_engine:=false \
     spawn_task_board:=true spawn_cable:=true \
     attach_cable_to_gripper:=true cable_type:=sfp_sc_cable \
     nic_card_mount_0_present:=true
   ```
2. In another terminal, run the collector in "grab one" mode. First time:

   ```bash
   cd /run/host/scratch2/atang/ws_aic/src/aic
   pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
     --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
     --repo-id atang/aic_sfp_auto \
     --num-episodes 1 --max-attempts 5 --exit-on-success
   ```

   The script retries up to 5 times until one successful insertion is captured,
   then exits. If none succeed, it stops after 5 attempts with nothing saved.
3. Kill Gazebo (`Ctrl-C`). Optionally change configs (different NIC slot,
   cable type, etc.).
4. Relaunch Gazebo. Rerun the script with `--resume` to append to the same
   dataset:

   ```bash
   pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
     --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
     --repo-id atang/aic_sfp_auto \
     --num-episodes 1 --max-attempts 5 --exit-on-success --resume
   ```
5. Repeat steps 3–4 until you have N episodes.

This also lets you naturally vary configs between episodes (just change the
launch args) without any scripting gymnastics — the dataset keeps growing
across launches thanks to `--resume`.

### Why not `--reset-scene`?

`/gz_server/reset_simulation` *does* rewind the Gazebo world, but on this
build the reset path unloads and reloads the entire ros2_control stack
inside the same container, which segfaults shortly after (container exits
with code −11). The script now warns and no-ops the flag.

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

Example for slot 2 (remember: one episode per launch, so `--num-episodes 1`):

```bash
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --port-frame "task_board/nic_card_mount_2/sfp_port_0_link" \
  --num-episodes 1 --max-attempts 5 --exit-on-success --resume
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
teleop-automated-dataset/sfp/
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
| **Speed** | ~30-40 demos/hour per person | ~20-60/hour with per-launch workflow; bound by Gazebo launch time |
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
