# Teleop Recording — Step-by-Step Procedure

> **Audience:** The teammate running the actual teleop sessions, and anyone verifying the dataset afterward.
>
> **Purpose:** Make sure every new recording session produces a dataset that (a) has F/T sensor data, (b) has continuous-valued actions, and (c) is not contaminated by the schema of the old 201-episode dataset. Follow this procedure exactly for every session.
>
> If you want to understand the *technique* of teleoperating (controls, drills, what to keep vs. discard), read `teleop_guide.md`. This document is just the mechanical procedure.

---

## What changed from the old procedure

Two driver-level changes were shipped on 2026-04-24:

1. **The lerobot driver now records the F/T sensor.** Every frame's `observation.state` has 6 extra columns (`wrench.force.{x,y,z}`, `wrench.torque.{x,y,z}`) with the tare already subtracted. The old 201-episode dataset does not have these columns. New datasets do.

2. **The keyboard teleop driver now EMA-smooths the published velocity.** You operate the keyboard exactly as before, but instead of stepping the output between 0 and 0.1 instantly, it ramps over ~80 ms. Recorded actions are now continuous-valued, which the policy can learn to emit smoothly. You may notice the robot feels very slightly less snappy — that is the smoothing at work.

**What this means for you:** The way you *teleoperate* is unchanged. The way you *set up and record* has a few extra checks, detailed below.

---

## Step 0 — One-time setup (skip if already done)

```bash
# Confirm the driver edits are in your pixi env
pixi run python -c "from lerobot_robot_aic.aic_robot_aic_controller import ObservationState; \
  print(len(ObservationState.__annotations__), 'fields; wrench in schema:', \
  any('wrench' in k for k in ObservationState.__annotations__))"
```

Expected output:
```
32 fields; wrench in schema: True
```

If you see `26 fields` or `wrench in schema: False`, the driver edits are not active in your environment. Re-clone or re-sync the repo and rerun the symlink step from the commit messages — see `critical_issues.md` entry #2.

---

## Step 1 — Launch Gazebo

> **Paste tip:** copy each command as **one single line** — do not use backslash line continuations when pasting into the terminal. Some terminals insert invisible trailing whitespace after `\` which breaks the continuation and causes errors like `malformed launch argument ' '`. If a command is shown here with `\` continuations for readability, join them into one line before pasting.
>
> **Path note:** inside the `aic_eval` distrobox your prompt looks like `atang@aic_eval:...`. Inside the container, `/scratch2/...` does not exist — use `/run/host/scratch2/...` instead. All shell commands below assume the distrobox.

Terminal 1, SFP scene (single line):

```bash
source ~/lab/ws_aic/setup_dev.sh && ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true start_aic_engine:=false spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true cable_type:=sfp_sc_cable nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 sc_port_0_present:=true sc_port_0_translation:=-0.04
```

For SC, swap `cable_type:=sfp_sc_cable_reversed` and the SC rail/mount args — see `teleop_guide.md` section 1 for the full SC command.

Wait for Gazebo to fully settle. You should see the robot arm, the task board, and the plug grasped by the gripper.

---

## Step 2 — Pre-flight sanity checks

**Do not skip these.** They take 60 seconds total and catch the two failure modes that produce silently-broken datasets.

Terminal 2, all in the same environment (`source setup_dev.sh` first):

```bash
# Check 1: F/T sensor is actually publishing
ros2 topic hz /fts_broadcaster/wrench
```

Expected: `average rate: 250-500`. If it says `no new messages`, the sensor is dead — kill Gazebo and relaunch. If you record without this working, every wrench column in your dataset will be zero.

```bash
# Check 2: The tare offset is established (non-zero, finite)
ros2 topic echo /aic_controller/controller_state --once \
  | grep -A5 fts_tare_offset
```

Expected: you should see `force: x: <something>, y: <something>, z: <something>` with non-zero values (the gripper+plug gravity load, typically 5–15 N in z). If it's all zeros, the controller hasn't tared yet — wait 5 seconds and re-run.

```bash
# Check 3: Confirm the ros topic activity looks right
ros2 topic list | grep -E "fts_broadcaster|controller_state|joint_states"
```

Expected: all three present. Ctrl+C out when confirmed.

---

## Step 3 — Pick a fresh dataset path

**Critical rule: do not resume the old `teleop-dataset/` directory.** That dataset was recorded with the 26-D schema (no wrench). If you point the new driver at it, lerobot will throw a schema-mismatch error *or* silently mix schemas (depending on lerobot version). Either way the result is bad.

Use a new path. Recommended convention:

```bash
DATASET_ROOT=/scratch2/atang/ws_aic/teleop-dataset-ft-v1
DATASET_REPO=local/teleop_sfp_ft
```

- `-ft-v1` signals "F/T-enabled, version 1 of the new-driver era"
- Increment to `v2` if you ever change `ObservationState` again

Confirm the directory does not already exist:

```bash
ls "$DATASET_ROOT" 2>/dev/null && echo "EXISTS — pick a different path or delete first" \
                               || echo "OK — fresh path"
```

---

## Step 4 — Start lerobot-record

Terminal 2 (after sanity checks passed). Copy-paste the single-line form below. Do **not** use the readable version's backslash line continuations — some terminals insert trailing whitespace that breaks them:

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && pixi run lerobot-record --robot.type=aic_controller --robot.id=aic --teleop.type=aic_keyboard_ee --teleop.id=aic --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link --dataset.repo_id=local/teleop_sfp_ft --dataset.root=/run/host/scratch2/atang/ws_aic/teleop-dataset-ft-v1 --dataset.single_task="insert SFP" --dataset.push_to_hub=false --dataset.private=true --play_sounds=false --display_data=true
```

Readable version (for reference only):

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic

pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=local/teleop_sfp_ft \
  --dataset.root=/run/host/scratch2/atang/ws_aic/teleop-dataset-ft-v1 \
  --dataset.single_task="insert SFP" \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --play_sounds=false \
  --display_data=true
```

**Things to notice about this command versus the old one:**

| Part | Why it matters |
|---|---|
| `--dataset.root=...-ft-v1` | New path. Do not resume old dataset. |
| `--dataset.single_task="insert SFP"` | Tags every episode with `task_index=0` for SFP. Use `"insert SC"` for SC sessions — that tag goes into `meta/tasks.jsonl` and becomes the input to task-conditioning later. |
| No `--resume=true` on first session | Omitted so lerobot creates the dataset fresh. If you come back to this dataset tomorrow to add more episodes, add `--resume=true`. |

If you get `FileExistsError`, the dataset path already has stuff in it. Either add `--resume=true` (if you want to continue that dataset) or pick a different path.

---

## Step 5 — Record ONE test episode and verify

Do one episode before committing to a long session. Just drive the plug around for 5–10 seconds, doesn't have to be a successful insertion. Press **Right Arrow** to save.

Then Ctrl+C to stop lerobot-record. In a third terminal:

```bash
pixi run python3 -c "
import pandas as pd, numpy as np, glob
p = '/scratch2/atang/ws_aic/teleop-dataset-ft-v1'
parquets = sorted(glob.glob(f'{p}/data/chunk-000/*.parquet'))
assert parquets, 'No parquet files written — dataset is empty'
df = pd.read_parquet(parquets[0])
state = np.stack(df['observation.state'])
acts  = np.stack(df['action'])
print('frames:', len(df))
print('state shape:', state.shape, '(expect (_, 32))')
print('--- wrench (last 6 state cols) ---')
print('  mean:', state[:, -6:].mean(axis=0).round(3))
print('  std :', state[:, -6:].std(axis=0).round(3))
print('  max abs:', np.abs(state[:, -6:]).max(axis=0).round(3))
print('--- actions ---')
for d in range(6):
    u = np.unique(np.round(acts[:, d], 4))
    print(f'  dim {d}: n_unique={len(u)}, range=[{acts[:,d].min():.3f}, {acts[:,d].max():.3f}]')
"
```

What you want to see:

| Check | Good | Bad — stop and diagnose |
|---|---|---|
| `state shape: (_, 32)` | Yes | Shows 26 → driver edits not active |
| Wrench columns `std` > 0.1 on at least one axis | Yes | All zeros → F/T not publishing |
| Wrench columns `max abs` in [0.5, 10.0] range | Yes | All near 0 or > 50 → tare broken |
| Action `n_unique` > 10 per actively-used dim | Yes | Only 3 values → smoothing not active |

Only proceed to full recording if **all four checks pass.** If any fail, re-verify the driver install (Step 0) and the Gazebo sensor publishing (Step 2). Do not record a full session on a broken setup.

---

## Step 6 — Full recording session

Re-launch lerobot-record with `--resume=true` appended to the command from Step 4 (the dataset now exists, so lerobot will continue adding episodes).

Follow the teleoperation technique in `teleop_guide.md` sections 2–10 for the actual driving. Key reminders from that doc:

- Press `k` for slow mode when within 3 cm of the port
- Press **Right Arrow** as soon as the insertion succeeds — do not wait for the 60 s timeout
- Press **Left Arrow** during reset to discard a bad demo
- Keep demos with corrections and recoveries — they are the most valuable training signal
- Take a 5-minute break every 20 minutes

**Session targets:**

| Plug | Episodes | Scene variation |
|---|---|---|
| SFP pilot | 30–50 | 3–4 different `nic_card_mount_0_translation` values, 2–3 `yaw` values |
| SFP full | 100+ | Rotate through 10+ scene configs per `sfp_scene_configs.md` |
| SC pilot | 30–50 | Same idea for SC, using `insert SC` task tag |
| SC full | 50+ | Focus on precision: SC port has no guide rails |

Between scene changes: Ctrl+C lerobot-record, kill Gazebo, relaunch Gazebo with new params, wait, re-run lerobot-record with `--resume=true`.

---

## Step 7 — Post-session verification

After closing lerobot-record:

```bash
# Episode count and basic stats
pixi run python3 -c "
import json
info = json.load(open('/scratch2/atang/ws_aic/teleop-dataset-ft-v1/meta/info.json'))
print('total episodes:', info['total_episodes'])
print('total frames:', info['total_frames'])
print('state dim:', info['features']['observation.state']['shape'])
print('has wrench:', any('wrench' in n for n in info['features']['observation.state']['names']))
"
```

```bash
# Spot-check a random late episode still has F/T data
pixi run python3 -c "
import pandas as pd, numpy as np, glob
p = '/scratch2/atang/ws_aic/teleop-dataset-ft-v1'
df = pd.read_parquet(sorted(glob.glob(f'{p}/data/chunk-000/*.parquet'))[-1])
import random; random.seed(0)
ep = df.groupby('episode_index').size().index[-1]  # last episode
sub = df[df['episode_index'] == ep]
state = np.stack(sub['observation.state'])
print(f'episode {ep}: {len(sub)} frames')
print(f'wrench std:  {state[:, -6:].std(axis=0).round(3)}')
print(f'wrench max:  {np.abs(state[:, -6:]).max(axis=0).round(3)}')
"
```

If the last episode's wrench still looks alive (non-zero std, realistic magnitudes), your session is good. File location and schema are documented in `teleop_guide.md` section 12.

---

## Step 8 — Hand off for training

Message the team (Slack/Discord) with:

- Dataset path (`/scratch2/atang/ws_aic/teleop-dataset-ft-v1`)
- Episode count
- Plug type (SFP / SC / mixed)
- Which scene configs were used, roughly
- Any issues noticed during the session

That way whoever retrains can sanity-check before committing to a long run.

---

## Common failure modes (read before your first session)

| Symptom | Cause | Fix |
|---|---|---|
| `FileExistsError` on start | Dataset path already exists and you didn't pass `--resume=true` | Either add `--resume=true` to continue, or pick a new path |
| `wrench std` all zeros | F/T topic not publishing when you started recording | Kill lerobot, restart Gazebo, wait longer for sensors to come up, re-do Step 2 |
| `state shape: (_, 26)` | Driver edits not active (pixi cache is still using old install) | Re-apply the symlink from critical_issues.md; re-run Step 0 verification |
| Actions still show only 3 unique values per dim | Teleop driver edits not active | Same as above — re-verify `aic_teleoperation/cartesian_keyboard_teleop.py` has the EMA code |
| Wrench max abs > 50 N | Tare not subtracted (stale tare, or gripper grasping when tare was set) | Restart Gazebo; the controller tares at startup before the plug is grasped |
| Robot feels slightly sluggish | EMA smoothing doing its job | Normal. If it genuinely hurts your precision, tweak `ACTION_SMOOTHING_ALPHA` at the top of `cartesian_keyboard_teleop.py` (higher alpha = less smoothing). 0.5 default, 0.7 if you want snappier. |

---

## One rule to remember

**Never resume the old `teleop-dataset/` with the new driver.** The schemas are incompatible. New path, new name, new session. Everything else in this doc is just making sure that rule sticks.
