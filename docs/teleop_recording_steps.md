# Teleop Recording — Step-by-Step Procedure

> **Audience:** The teammate running the actual teleop sessions, and anyone verifying the dataset afterward.
>
> **Purpose:** Make sure every new recording session produces a dataset that (a) has F/T sensor data, (b) has continuous-valued actions, and (c) is not contaminated by the schema of the old 201-episode dataset. Follow this procedure exactly for every session.
>
> If you want to understand the *technique* of teleoperating (controls, drills, what to keep vs. discard), read `teleop_guide.md`. This document is just the mechanical procedure.

---

## What changed from the old procedure

Driver-level changes shipped between 2026-04-24 and 2026-04-25:

1. **The lerobot driver now records the F/T sensor.** Every frame's `observation.state` has 6 extra columns (`wrench.force.{x,y,z}`, `wrench.torque.{x,y,z}`) with the tare already subtracted. The old 201-episode dataset does not have these columns. New datasets do.

2. **The lerobot driver now records joint velocities.** Seven new columns (`joint_velocities.0` .. `joint_velocities.6`) are pulled from the same `/joint_states` message that already provided positions. Free dynamics signal for ACT/Diffusion-Policy.

3. **The lerobot teleop path now EMA-smooths the published velocity.** Note: this fix lives in `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py` — the file the `--teleop.type=aic_keyboard_ee` route actually loads. (The earlier patch on `cartesian_keyboard_teleop.py` was on a different, unused code path.) You operate the keyboard exactly as before; the published twist now ramps over ~167 ms instead of stepping. Recorded actions become continuous-valued. The robot feels very slightly less snappy — that is the smoothing at work.

4. **Per-session F/T re-tare.** A small wrapper at `scripts/tare.sh` calls the controller's `~/tare_force_torque_sensor` service. Run it once per session (and after any noticeable gripper rotation) to keep the wrench baseline consistent across the dataset.

**Total `observation.state` is now 39-D:** 7 tcp_pose + 6 tcp_velocity + 6 tcp_error + 7 joint_positions + 7 joint_velocities + 6 wrench. Training configs typically drop tcp_velocity and tcp_error (state-leakage prevention), leaving 27 useful dims.

**What this means for you:** The way you *teleoperate* is unchanged. The way you *set up and record* has a few extra checks, detailed below.

---

## Step 0 — One-time setup (skip if already done)

```bash
# Confirm the driver edits are in your pixi env
pixi run python -c "from lerobot_robot_aic.aic_robot_aic_controller import ObservationState; \
  ann = ObservationState.__annotations__; \
  print(len(ann), 'fields; wrench:', any('wrench' in k for k in ann), \
        '; joint_velocities:', any('joint_velocities' in k for k in ann))"

pixi run python -c "from lerobot_robot_aic.aic_teleop import ACTION_SMOOTHING_ALPHA; \
  print('teleop EMA alpha =', ACTION_SMOOTHING_ALPHA)"
```

Expected output:
```
39 fields; wrench: True ; joint_velocities: True
teleop EMA alpha = 0.5
```

If you see `26 fields` / `32 fields` / `wrench: False` / `joint_velocities: False`, or `lerobot-record` fails at startup with `Could not import third-party plugin: lerobot_robot_aic` / `ModuleNotFoundError: No module named 'lerobot_robot_aic.aic_robot_aic_controller'`, the driver edits are not active in your environment. The fix is to symlink the source files into the pixi env using **relative** paths so the symlinks resolve correctly both on the host and inside the `aic_eval` distrobox (where `/scratch2/...` is remounted at `/run/host/scratch2/...`).

Run from the **host** shell (not inside the distrobox), so `/scratch2/...` paths resolve:

```bash
cd /scratch2/atang/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/lerobot_robot_aic
rm -f aic_teleop.py aic_robot_aic_controller.py
ln -srv /scratch2/atang/ws_aic/src/aic/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py aic_teleop.py
ln -srv /scratch2/atang/ws_aic/src/aic/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot_aic_controller.py aic_robot_aic_controller.py
rm -rf __pycache__
```

The `-r` flag converts the absolute path to a relative one (`../../../../../../../aic_utils/...`), which works under any mount point. **Do not use `ln -s` without `-r`** — absolute symlinks made on the host point at `/scratch2/...` paths that don't exist inside the distrobox, and `lerobot-record` will fail to import the plugin.

Then re-run the verification block above.

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

## Step 4 — Tare the F/T sensor (once per session, then again per episode)

The tare offset is published live on `/aic_controller/controller_state` and the lerobot driver subtracts it on every recorded frame. Re-taring instantly shifts what gets written into the parquet — no need to stop `lerobot-record`. Just keep one terminal open with this command available:

```bash
bash /run/host/scratch2/atang/ws_aic/scripts/tare.sh
```

Expected output ends with `success=True, message='Successfully tared force torque sensor.'`.

**When to run it:**

1. **Once at session start**, while the arm is in its launch pose with the plug grasped — this gives `force_monitor.py` a clean zero to display from.
2. **Before pressing Right Arrow on every episode**, while the arm is back at the start pose (no contact). One half-second extra per episode.
3. **After any scene change** (cable repositioned on the board, controller restart, gripper rotated > ~10° from the last tare).

**Empirical evidence this matters:** in early test recordings, mean `force_z` drifted from -3 N (ep 0) to +18 N (ep 1, 2) — a 21 N spread across just 3 episodes when no per-episode tare was done. With the per-episode tare protocol above, a later 5-episode batch held mean `force_z` between -1.0 N and -1.9 N — **a 1 N spread**. The protocol is what makes the wrench column trainable.

**Do not run `tare.sh` mid-episode** — it would inject a step discontinuity into the recorded wrench. Only run it during the reset window between episodes, while the arm is in free space.

---

## Step 5 — Start lerobot-record

Terminal 2 (after sanity checks and tare passed). Copy-paste the single-line form below. Do **not** use the readable version's backslash line continuations — some terminals insert trailing whitespace that breaks them:

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && pixi run lerobot-record --robot.type=aic_controller --robot.id=aic --teleop.type=aic_keyboard_ee --teleop.id=aic --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link --dataset.repo_id=local/teleop_sfp_ft --dataset.root=/run/host/scratch2/atang/ws_aic/teleop-dataset-ft-v1 --dataset.single_task="insert SFP" --dataset.fps=20 --dataset.push_to_hub=false --dataset.private=true --play_sounds=false --display_data=true
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
  --dataset.fps=20 \
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
| `--dataset.fps=20` | Matches the actual camera publish rate (configured in `aic_robot.py` at `fps=20`). The old 201-ep dataset was recorded at 30 Hz with 20 Hz cameras, which meant ~33% of frames had a stale image. Recording at 20 Hz keeps every frame's image and action fresh. |
| No `--resume=true` on first session | Omitted so lerobot creates the dataset fresh. If you come back to this dataset tomorrow to add more episodes, add `--resume=true`. |

If you get `FileExistsError`, the dataset path already has stuff in it. Either add `--resume=true` (if you want to continue that dataset) or pick a different path.

---

## Step 6 — Record ONE test episode and verify

Do one episode before committing to a long session. Run the 5-axis wiggle drill (see Step 7's "Action-axis coverage drill" below) at the start, then drive the plug around. Doesn't have to be a successful insertion. Press **Right Arrow** to save.

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
print('state shape:', state.shape, '(expect (_, 39))')
print('--- joint velocities (state cols 20:27) ---')
print('  std :', state[:, 20:27].std(axis=0).round(3))
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
| `state shape: (_, 39)` | Yes | 26 → old driver. 32 → joint_velocities patch not active. |
| Joint velocity columns `std` > 0 on at least 5 of 7 joints | Yes | All zeros → `/joint_states` lacks velocity, or driver fallback hit. (`jv.1` near zero is expected — that's the gripper joint under alphabetical sort.) |
| Wrench columns `std` > 0.1 on at least one axis | Yes | All zeros → F/T not publishing |
| Wrench `mean fz` in [-3, +3] N | Yes | \|mean fz\| > 5 N → stale tare (re-run `tare.sh` and re-record). High `max abs` (>20 N) is fine — that's real contact signal during jam attempts. |
| Action `n_unique` > 10 per actively-used dim, **and all 6 dims appear in [-, +] range** | Yes | Only 3 values per dim → EMA not active in `aic_teleop.py`. Any dim only one-sided → wiggle drill skipped that key. |

Only proceed to full recording if **all five checks pass.** If any fail, re-verify the driver install (Step 0), the Gazebo sensor publishing (Step 2), the tare protocol (Step 4), and that you actually ran the 5-axis wiggle drill. Do not record a full session on a broken setup.

---

## Step 7 — Full recording session

Re-launch lerobot-record with `--resume=true` appended to the command from Step 5 (the dataset now exists, so lerobot will continue adding episodes).

Follow the teleoperation technique in `teleop_guide.md` sections 2–10 for the actual driving. Key reminders from that doc:

- Press `k` for slow mode when within 3 cm of the port
- Press **Right Arrow** as soon as the insertion succeeds — do not wait for the 60 s timeout
- Press **Left Arrow** during reset to discard a bad demo
- Keep demos with corrections and recoveries — they are the most valuable training signal
- Take a 5-minute break every 20 minutes

### Action-axis coverage drill (mandatory at the start of every episode)

**Why this exists:** the previous 201-episode corpus had `angular.y` literally 0 in 100% of frames, `angular.z` 0 in 97.8%, `linear.x` 0 in 96.5%, and `linear.y` 0 in 95.7% — i.e. the dataset basically only contained `linear.z` (descent) and a little `angular.x`. A policy trained on that has no learned response for the other four axes. The first batches of the new dataset still showed `angular.y` at 0% until the wiggle drill below was made mandatory.

**The 5-axis wiggle (≈3 seconds at episode start, before any insertion attempt):**

| Axis | Negative key | Positive key | Hold each |
|---|---|---|---|
| `linear.x` | `a` | `d` | ~0.3 s |
| `linear.y` | `w` | `s` | ~0.3 s |
| `angular.x` (roll) | `Shift+S` | `Shift+W` | ~0.3 s |
| `angular.y` (**pitch — most skipped**) | `Shift+A` | `Shift+D` | ~0.3 s |
| `angular.z` (**yaw — second most skipped**) | `q` | `e` | ~0.3 s |

Skip `linear.z` from the wiggle — you'll exercise it during the descent itself.

The motion per tap is small (≤ ~5° rotation, ≤ ~5 mm translation) — enough to register in the recorded action stream, not enough to disturb the start pose. EMA smoothing turns each 0.3 s tap into a brief ramp, which is even better learning signal than a single-step impulse.

**The two keys most operators forget:** `Shift+A`/`Shift+D` for pitch (because the keyboard combo is unusual) and `q`/`e` for yaw (because it feels redundant when the actual insertion doesn't need yaw). Empirically, even after operators are reminded about pitch they still skip yaw — when you finish the pitch wiggle, **deliberately tap `q` then `e` before starting the descent**. Ten extra key presses per episode in exchange for a policy that can command yaw at all.

**During the actual insertion attempt (after the wiggle):**

- Approach the port on a deliberately offset trajectory, then correct with `a/d` and `w/s`.
- Vary which axis you descend on: pure `f` half the time, mixed with light `a/s` corrections the rest.
- Use `k` for slow mode within ~3 cm of the port.

A messy demo that hits all axes is more useful than a perfect demo that only descends.

### Re-tare cadence

Per Step 4: re-tare **before pressing Right Arrow on every episode**. Quick checklist for the per-episode reset window:

1. Press Right Arrow to save the previous episode.
2. Move the arm back to the start pose, plug in hand, no contact.
3. Run `bash /run/host/scratch2/atang/ws_aic/scripts/tare.sh` from your tare terminal.
4. Glance at `force_monitor.py` — the displayed force should snap to ~0 N. That's your in-loop confirmation the tare succeeded. If it doesn't snap to zero, the gripper was probably still touching the receptacle — back off and re-tare.
5. Press Right Arrow to start the next episode.
6. Run the 5-axis wiggle drill above for the first ~3 seconds of the episode.
7. Then attempt the insertion.

`lerobot-record` continues running through all of this — the tare is a single ROS service call that takes ~half a second and is picked up by the driver on the next frame.

**Session targets:**

| Plug | Episodes | Scene variation |
|---|---|---|
| SFP pilot | 30–50 | 3–4 different `nic_card_mount_0_translation` values, 2–3 `yaw` values |
| SFP full | 100+ | Rotate through 10+ scene configs per `sfp_scene_configs.md` |
| SC pilot | 30–50 | Same idea for SC, using `insert SC` task tag |
| SC full | 50+ | Focus on precision: SC port has no guide rails |

Between scene changes: Ctrl+C lerobot-record, kill Gazebo, relaunch Gazebo with new params, wait, **re-run `tare.sh`**, then re-run lerobot-record with `--resume=true`.

---

## Step 8 — Post-session verification

After closing lerobot-record:

```bash
# Episode count and basic stats
pixi run python3 -c "
import json
info = json.load(open('/scratch2/atang/ws_aic/teleop-dataset-ft-v1/meta/info.json'))
names = info['features']['observation.state']['names']
print('total episodes:', info['total_episodes'])
print('total frames:', info['total_frames'])
print('fps:', info['fps'])
print('state dim:', info['features']['observation.state']['shape'])
print('has wrench:', any('wrench' in n for n in names))
print('has joint_velocities:', any('joint_velocities' in n for n in names))
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

## Step 9 — Hand off for training

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
| `state shape: (_, 26)` or `(_, 32)` | Driver edits not active (pixi cache is still using old install) | Re-apply the symlinks from Step 0; re-run Step 0 verification |
| Actions still show only 3 unique values per dim | Teleop EMA not active in `aic_teleop.py` | Re-run Step 0 verification — the relevant file is `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py`, NOT `aic_teleoperation/cartesian_keyboard_teleop.py` (that file is a separate standalone ROS node and not used by `lerobot-record`). |
| Wrench max abs > 50 N | Stale or wrong tare | Run `bash scripts/tare.sh` and verify with `ros2 topic echo /aic_controller/controller_state --once \| grep -A5 fts_tare_offset` |
| Wrench drifts across episodes within a session | Tare is only set once per Gazebo launch; gripper rotation since then has projected gravity differently onto the FTS frame | Re-run `tare.sh` periodically — at minimum on every scene change |
| Robot feels slightly sluggish | EMA smoothing doing its job | Normal. If it genuinely hurts your precision, tweak `ACTION_SMOOTHING_ALPHA` at the top of `aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_teleop.py` (higher alpha = less smoothing). 0.5 default, 0.7 if you want snappier. |

---

## One rule to remember

**Never resume the old `teleop-dataset/` with the new driver.** The schemas are incompatible. New path, new name, new session. Everything else in this doc is just making sure that rule sticks.
