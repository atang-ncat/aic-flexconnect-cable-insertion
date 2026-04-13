# Teleoperation Guide — Mastering Robot Control for Data Collection

> **Purpose:** Teach you how to teleoperate the UR5e arm in the AIC Gazebo simulation to collect high-quality demonstration data for training visuomotor insertion policies.
>
> **Time to competency:** ~30 minutes of practice before recording real demos.

---

## 1. Prerequisites

Before starting teleoperation, you need the Gazebo simulation running with a scene.

**Terminal 1 — Launch Gazebo with an SFP scene:**
```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```

**Terminal 2 — Launch Gazebo with an SC scene (reversed cable):**
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

**Terminal 2 — Start lerobot-record (records data + gives you keyboard teleop):**
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

> **Note:** The `UV_CACHE_DIR`, `RATTLER_CACHE_DIR`, and `XDG_CACHE_HOME` environment variables are already set in `~/.bashrc`, so you don't need to export them manually.

> **First time only:** If you have never recorded any episodes yet (the dataset directory does not exist), remove `--resume=true` from the command above. It will create a fresh dataset. Every session after that, keep `--resume=true` to continue adding episodes — without it, lerobot crashes with `FileExistsError`.

**Terminal 2 (alternative) — Standalone teleop for practice (no recording):**
```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

**Terminal 3 (optional) — Monitor insertion success:**
```bash
ros2 topic echo /scoring/insertion_event
```

**Terminal 4 (optional) — Monitor force during teleoperation:**
```bash
source /ws_aic/install/setup.bash
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

Shows a live dashboard with the tared force magnitude, a color-coded bar, and cumulative time above the 20 N scoring threshold. Prints a session summary when you Ctrl+C.

---

## 2. Keyboard Controls — Complete Reference

### Linear Movement (translation)

| Key | Axis | Direction (`base_link` frame) |
|:---:|:----:|-------------------------------|
| `d` | +X | Right |
| `a` | -X | Left |
| `s` | +Y | Away from you (forward) |
| `w` | -Y | Toward you (backward) |
| `f` | +Z | **UP** (away from table — retract/back off) |
| `r` | -Z | **DOWN** (toward table — insertion direction) |

> **Critical:** `r` pushes DOWN into the port. `f` pulls UP away. Do not confuse them.

### Angular Movement (rotation)

| Key | Axis | Motion |
|:---:|:----:|--------|
| `Shift+w` | +Angular X | Tilt forward (roll) |
| `Shift+s` | -Angular X | Tilt backward (roll) |
| `Shift+d` | +Angular Y | Tilt right (pitch) |
| `Shift+a` | -Angular Y | Tilt left (pitch) |
| `e` | +Angular Z | Rotate counter-clockwise (yaw) |
| `q` | -Angular Z | Rotate clockwise (yaw) |

> **Shift key release order:** Let go of the letter key FIRST, then release Shift. If you release Shift first, the lowercase letter registers and the robot translates instead of rotating.

### Mode and Frame Controls

| Key | Function |
|:---:|----------|
| `k` | **Slow mode** — 0.02 m/s linear, 0.02 rad/s angular |
| `l` | **Fast mode** — 0.1 m/s linear, 0.1 rad/s angular |
| `m` | Switch to `base_link` frame (world-fixed axes) |
| `n` | Switch to `gripper/tcp` frame (gripper-relative axes) |
| `ESC` | Quit teleoperation |

---

## 3. Understanding the Two Reference Frames

### `base_link` frame (press `m`) — World Perspective

Axes are fixed to the robot's base on the table. They **never change** regardless of how the gripper is rotated.

```
         -Y (w)
          ↑
          |
-X (a) ←-·-→ +X (d)
          |
          ↓
         +Y (s)

+Z (f) = UP      -Z (r) = DOWN
```

**When to use:** Coarse approach. When the connector is far from the port and you need predictable, consistent directions. "Press `d` to go right" always means the same thing in the room.

### `gripper/tcp` frame (press `n`) — Gripper's Perspective

Axes are attached to the gripper tip and **rotate with it**. If the gripper is tilted, all directions tilt with it.

**When to use:** Fine alignment and final insertion. Once you are close to the port and the gripper is oriented for insertion, the gripper's local axes align with the port's geometry. Small lateral corrections in this frame move the connector precisely in the plane you need to align.

### Recommended Workflow

1. Start in `base_link` frame (`m`) for the approach
2. Switch to `gripper/tcp` frame (`n`) once you are close and aligned
3. If you get disoriented, press `m` to go back to world frame

---

## 4. Speed Modes — When to Use Each

| Mode | Speed | Use For |
|------|-------|---------|
| **Fast** (`l`) | 0.1 m/s | Moving across the workspace to get near the port. Never use fast mode within 2cm of the port. |
| **Slow** (`k`) | 0.02 m/s | Fine alignment and insertion. This is 5x slower. Always switch to slow before the connector is near the port opening. |

> **Rule of thumb:** If you can see the port in the camera and the connector is close, you should already be in slow mode.

---

## 5. The Five Phases of a Good Demo

Each demonstration episode follows this natural progression:

### Phase 1: Orient (1–2 seconds)

Before moving anything, look at the Gazebo camera views. Identify your target:
- **SFP port:** The rectangular slot on top of the green NIC card (PCB)
- **SC port:** The rectangular opening on top of the small blue box

Know where you need to go before you touch a key.

### Phase 2: Coarse Approach (3–5 seconds)

- Frame: `base_link` (`m`)
- Speed: Fast (`l`)
- Keys: `a`/`d` (left/right), `w`/`s` (forward/back)

Slide the arm laterally until the connector is roughly above the target port. Do NOT descend yet. Get the XY alignment approximately right first.

### Phase 3: Fine Alignment (3–5 seconds)

- Frame: `base_link` (`m`) or `gripper/tcp` (`n`)
- Speed: **Slow** (`k`) — switch now
- Keys: Small taps on `a`/`d`/`w`/`s`

Center the connector precisely over the port opening. Watch the center camera feed. The connector tip should be directly above the port center. Take your time — precision here determines whether insertion succeeds.

### Phase 4: Insertion (3–5 seconds)

- Speed: Slow (`k`)
- Key: `r` (descend)

Press `r` in short, gentle taps. Watch what happens:
- **If the connector slides straight in:** Keep going with `r` until it seats fully.
- **If the connector stops or deflects sideways:** STOP descending. Nudge laterally with `a`/`d`/`w`/`s` to realign, then try `r` again.
- **If you hear/see collision or high forces:** Press `f` to back up slightly, realign, and retry.

### Phase 5: Confirm (1–2 seconds)

Hold position briefly. Check Terminal 4 for a `/scoring/insertion_event` message. If it fires, the insertion succeeded. If not, try gentle additional descent with `r` or slight lateral adjustments.

---

## 6. Practice Drills

Complete these drills in order before recording real demos. Each builds on the previous one.

### Drill 1: Basic Movement (5 minutes)

Goal: Get comfortable with all 6 axes of movement.

1. Press `l` (fast mode), `m` (base_link frame)
2. Press `d` for 1 second, then `a` for 1 second — watch the arm slide left and right
3. Press `s` for 1 second, then `w` for 1 second — forward and back
4. Press `r` for 1 second, then `f` for 1 second — down and up
5. Repeat until the directions feel intuitive

### Drill 2: Speed Control (3 minutes)

Goal: Feel the difference between fast and slow.

1. Press `l` (fast), then tap `d` — notice how quickly it moves
2. Press `k` (slow), then tap `d` — notice the difference
3. Practice switching between fast and slow while moving

### Drill 3: Approach and Hover (5 minutes)

Goal: Move the arm from the starting position to hover directly above a port.

1. Start in fast mode, base_link frame
2. Use `a`/`d`/`w`/`s` to position the connector above the SFP port on the NIC card
3. Switch to slow mode
4. Fine-tune until the connector is centered above the port
5. Do NOT descend yet — just practice the approach
6. Repeat 5 times from different starting positions (reset the sim between attempts)

### Drill 4: Insertion Attempt (10 minutes)

Goal: Complete a full insertion.

1. Approach and hover above the port (Drill 3)
2. In slow mode, press `r` gently to descend
3. If the connector deflects, press `f` to back up, adjust laterally, try again
4. Keep going until `/scoring/insertion_event` fires or you run out of patience
5. Repeat 5 times. Your success rate will improve with each attempt.

### Drill 5: SC Port (5 minutes)

Goal: Adapt to the SC port (smaller target, no guide rails).

1. Relaunch Gazebo with the SC scene (reversed cable)
2. Repeat Drills 3 and 4 targeting the blue SC port
3. SC requires more precise alignment before descending — spend extra time on Phase 3

### Drill 6: Frame Switching (5 minutes)

Goal: Get comfortable switching between base_link and gripper/tcp frames.

1. Approach a port in base_link frame
2. Switch to gripper/tcp frame (`n`)
3. Notice how `a`/`d`/`w`/`s` now move relative to the gripper
4. Practice fine alignment in gripper frame
5. If you get lost, press `m` to go back to base_link

---

## 7. Golden Rules for Demo Quality

These rules apply once you start recording actual demonstrations for training data.

### 1. Slow is fast
Gentle, deliberate movements produce better training signal than fast, jerky ones. The policy learns your speed profile — if you rush, it rushes and overshoots.

### 2. Include corrections — do NOT discard imperfect demos
If you overshoot and then correct, that correction is the **most valuable** part of the demo. The policy needs to learn what "being slightly wrong and fixing it" looks like. A dataset of only perfect, straight-line approaches teaches the policy nothing about recovery.

### 3. Vary your approach path
Do not always follow the exact same trajectory. Sometimes approach from the left, sometimes from the right. Sometimes linger longer over the port before descending. This diversity is what makes the policy robust to different starting positions during evaluation.

### 4. One axis at a time
Move in one direction at a time. Press `d` and wait, then press `r` and wait. Do not press `d` and `r` simultaneously. Sequential movements are cleaner for the policy to learn than diagonal ones.

### 5. Consistent start state
Reset the simulation between episodes so each demo starts from the same robot configuration. But vary the scene configuration (board pose, rail translations) across recording sessions.

### 6. Record 5 throwaway demos first
Your first few demos will be terrible. That is normal. Get the muscle memory before recording for real.

### 7. Take breaks
Fatigue kills demo quality. Record in 20-minute sessions with 5-minute breaks between them. A tired operator produces sloppy demos that hurt the trained policy.

### 8. Both success and near-success are valuable
A demo where you get close to the port but don't quite insert is still useful — the approach trajectory teaches the policy how to navigate toward the target. Only discard demos where you obviously go to the completely wrong port or make wild, random movements.

---

## 7b. What to Keep vs. What to Discard

ACT, Diffusion Policy, VQ-BeT, and other behavior cloning architectures learn to imitate **every frame** of your dataset equally. They don't distinguish "good" from "bad" episodes. This means the composition of your dataset directly shapes the policy's behavior.

### Keep these episodes (press Right Arrow)

- **Overshoot then correct:** You go past the port, realize it, adjust laterally, and insert. This is the **most valuable** type of demo — the policy learns what "being off target" looks like and how to recover. A dataset of only perfect approaches produces a brittle policy that can't handle the slightest misalignment during eval.
- **Wrong angle, then adjust:** You approach at a slight angle, notice the connector isn't aligned, rotate or nudge, and complete the insertion. Same logic — the correction IS the training signal.
- **Bump and retry:** You descend too early, the connector hits the port lip, you back up with `f`, realign, and retry successfully. Teaches the policy contact recovery.
- **Near-miss without insertion:** You get close, align well, but the episode ends before full insertion. The approach trajectory is still valuable data for learning how to navigate toward the target.

### Discard these episodes (press Left Arrow)

- **Completely wrong location:** You navigate to the wrong side of the board, nowhere near the target port.
- **Random key mashing:** You lost orientation and pressed keys randomly with no meaningful intent.
- **Gave up and stopped:** You stopped moving halfway through with no attempt to complete the task. The policy would learn "stop moving" as a valid action.

### Why imperfect demos matter more than you think

If 90% of your demos are clean approaches and 10% have corrections and overshoots that eventually succeed, the policy learns that the dominant behavior is a smooth approach, but it **also** learns what recovery looks like from the 10%. This is exactly what you want — a policy that usually goes straight to the target but knows how to fix itself when it's off.

If you only collect flawless, identical-looking demos, the policy becomes fragile. The moment the eval board is in a slightly different position than anything in training, the policy has no concept of "I'm off, how do I correct?" and fails.

**Bottom line:** keep every episode where you made a meaningful attempt at the task, regardless of how messy the path was. The messy ones with successful corrections are worth more than the clean ones.

---

## 8. Monitoring and Debugging

### Watch the robot state in real time

```bash
# TCP pose (where the gripper is)
ros2 topic echo /aic_controller/controller_state --field tcp_pose

# Force/torque readings (detect contact)
ros2 topic echo /fts_broadcaster/wrench

# Check what commands are being sent
ros2 topic echo /aic_controller/pose_commands --field velocity
```

### Check insertion success

```bash
ros2 topic echo /scoring/insertion_event
```

This only fires when the connector is properly seated in the correct port. If you descend fully and this doesn't fire, the insertion failed — back up and try again.

### Monitor force (avoid the -12 pt penalty)

```bash
source /ws_aic/install/setup.bash
python3 /run/host/scratch2/atang/ws_aic/scripts/force_monitor.py
```

This runs the force monitor script which shows a live, color-coded dashboard:
- **Green** — tared force is well below 20 N, you're safe
- **Yellow** — approaching the 20 N threshold, ease up
- **Red** — above 20 N, cumulative time is counting toward the 1 s limit

The scoring penalty is **-12 points** if your tared force magnitude exceeds **20 N** for more than **1 second cumulative** across the entire run. The monitor tracks this in real time and shows the running total. When you Ctrl+C, it prints a session summary.

### Reset the simulation

If the robot gets into a bad state or you want a fresh start, kill the Gazebo launch (Ctrl+C in Terminal 1) and relaunch it. There is no soft reset — you need a full relaunch.

---

## 9. Common Mistakes and How to Avoid Them

| Mistake | What happens | Fix |
|---------|-------------|-----|
| Descending in fast mode | Connector slams into the board, huge force spike | Always switch to slow (`k`) before descending |
| Descending before aligning | Connector hits the port lip and deflects sideways | Spend more time on Phase 3 (fine alignment) |
| Confusing `r` and `f` | You go up when you meant to go down, or vice versa | `r` = down (toward table), `f` = up (away from table). Mnemonic: "**r**am it down" |
| Pressing multiple keys at once | Diagonal movement, hard for policy to learn | One key at a time. Lift finger before pressing next key |
| Not switching to slow mode | Overshooting the port during fine alignment | Press `k` as soon as you are within 3cm of the port |
| Getting disoriented in gripper frame | Moving in unexpected directions after frame switch | Press `m` to return to base_link frame when confused |
| Recording demos while fatigued | Sloppy, inconsistent trajectories | Take breaks every 20 minutes |
| Only recording perfect demos | Policy never learns recovery behaviors | Keep demos with corrections — they are the most valuable |

---

## 10. Connector-Specific Tips

### SFP Module into NIC Card (Trials 1 and 2)

- The SFP port has **metal guide rails** inside. Once you get the connector within about 3mm of center, the rails will physically guide it in during the final descent.
- You can be slightly less precise with lateral alignment — the guide rails forgive 2–3mm of error.
- The NIC card is the **green PCB** on the board. The SFP port is the rectangular slot on top.
- The connector is metallic and rectangular. It drops in vertically.

### SC Plug into SC Port (Trial 3)

- The SC port has **no guide rails**. It is an open rectangular cavity. The connector must be precisely centered before you descend, or it will catch on the lip.
- Spend extra time on Phase 3 (fine alignment). You need sub-millimeter precision.
- The SC port is the small **blue box** on the board. Its opening faces upward.
- The connector is white/blue with a rectangular profile.
- If the connector catches on the lip, back up with `f`, nudge laterally, and try again. Do not force it — that creates high forces and loses scoring points.

---

## 11. Scene Configurations for Practice

Vary these parameters across sessions to build a diverse dataset:

```bash
# SFP scene with different NIC card positions
nic_card_mount_0_translation:=0.005    # centered
nic_card_mount_0_translation:=0.02     # shifted right
nic_card_mount_0_translation:=-0.01    # shifted left

# SC scene with different port positions
sc_port_0_translation:=-0.04           # centered
sc_port_0_translation:=-0.02           # shifted
sc_port_0_translation:=0.0             # other end

# Board yaw variation (add to any scene)
yaw:=3.1415                            # default (facing robot)
yaw:=3.0                               # slightly rotated
yaw:=2.9                               # more rotated
```

---

## 12. Recording with lerobot-record — Full Workflow

### How it works

`lerobot-record` does two things simultaneously:
1. **Gives you keyboard teleop** (same keys as `cartesian_keyboard_teleop`)
2. **Records everything** — joint states, actions, and all 3 camera feeds at 30 fps

The Gazebo scene runs separately in its own terminal. `lerobot-record` connects to the running ROS 2 system by subscribing to `/observations`, camera topics, etc.

### Episode management hotkeys

| Key | Action |
|-----|--------|
| **Right Arrow** | **Save** the current episode and start recording the next one |
| **Left Arrow** | **Discard** the current episode (use during the "Reset" phase after a bad demo) |
| **Ctrl+C** | **Stop** recording entirely and exit. The dataset is saved automatically. |

### Step-by-step recording workflow

1. **Launch Gazebo** in terminal 1 with your scene configuration
2. **Launch lerobot-record** in terminal 2 (see command in Section 1)
3. You will see `Recording episode 0` — start teleoperating immediately
4. After a successful insertion, press **Right Arrow** to save the episode
5. During the "Reset" phase (encoding video, ~10–20 seconds), you can:
   - Kill Gazebo in terminal 1 and relaunch with a new scene configuration
   - Wait for `Recording episode N` to appear before teleoperating again
6. If the demo was bad (wrong port, wild movements), press **Left Arrow** to discard
7. Repeat steps 3–6 for each episode
8. When done collecting, press **Ctrl+C** to stop

### Timing: don't let episodes time out

Episodes auto-save after 60 seconds (`episode_time_s: 60`). If you don't press Right Arrow within 60 seconds, the episode saves automatically — including all the idle time at the end. This pads your data with useless zero-velocity frames.

**Always press Right Arrow** as soon as the insertion succeeds (or as soon as you decide the attempt is done). A good SFP demo is 10–20 seconds, not 60.

### Where is the dataset saved?

The dataset is saved locally to:

```
/scratch2/atang/ws_aic/teleop-dataset/sfp/
```

This is set by `--dataset.root` in the recording command. Note: `--dataset.root` is the **full path** to the dataset directory — lerobot does NOT append `repo_id` to it. The `repo_id` is only used for Hub identification. Keeping datasets inside the workspace (rather than `/tmp`) ensures they persist across container restarts.

The directory structure:

```
teleop-dataset/sfp/
├── meta/
│   ├── info.json          # dataset metadata (fps, features, etc.)
│   ├── episodes.jsonl     # per-episode metadata
│   └── tasks.jsonl        # task description
├── data/
│   └── train-00000-of-00001.parquet   # numerical data (joints, actions)
└── videos/
    ├── observation.images.left_camera/
    │   ├── episode_000000.mp4
    │   ├── episode_000001.mp4
    │   └── ...
    ├── observation.images.center_camera/
    │   └── ...
    └── observation.images.right_camera/
        └── ...
```

### Resuming a previous session

The main recording command in Section 1 already includes `--resume=true`. Just copy-paste it and you'll pick up from the last episode number. This is the normal workflow — you launch Gazebo, run the command, record a few episodes, Ctrl+C, and come back later to record more.

> **What `--resume=true` does:** It opens the existing dataset and appends new episodes after the last one. Without it, lerobot tries to create the dataset directory from scratch and **crashes** with `FileExistsError`.

### Starting fresh (deleting an existing dataset)

If you want to discard all previously recorded episodes and start over from episode 0, delete the dataset directory (the path from `--dataset.root`):

```bash
# Delete the SFP dataset and start over
rm -rf /scratch2/atang/ws_aic/teleop-dataset/sfp

# Delete the SC dataset and start over
rm -rf /scratch2/atang/ws_aic/teleop-dataset/sc
```

After deleting, run the recording command from Section 1 **without** `--resume=true` — it will create a fresh dataset from scratch.

### The Rerun viewer window

When `--display_data=true`, a Rerun visualization window pops up showing:
- **Time-series plots** of all joint states, velocities, and action commands
- **Live camera feeds** from all three cameras
- **Timeline strip** of recorded frames

This is purely a monitoring tool — it does NOT control the robot. It is useful for verifying that all cameras are capturing and that action data looks reasonable. You can safely minimize it while teleoperating, or pass `--display_data=false` to disable it entirely.

---

## 13. Collecting a Full Dataset — Practical Checklist

### SFP dataset (Trials 1 and 2)

| Step | Details |
|------|---------|
| **Target** | 50+ episodes |
| **repo_id** | `atang/aic_sfp_demos` |
| **task** | `"Insert SFP connector into SFP port on NIC card"` |
| **Scene variation** | Rotate through 3–4 different `nic_card_mount_0_translation` values and 2–3 `yaw` values |
| **Episode length** | Aim for 10–20 seconds per episode |

### SC dataset (Trial 3)

| Step | Details |
|------|---------|
| **Target** | 50+ episodes |
| **repo_id** | `atang/aic_sc_demos` |
| **task** | `"Insert SC plug into SC port"` |
| **cable_type** | `sfp_sc_cable_reversed` |
| **Scene variation** | Rotate through `sc_port_0_translation` and `yaw` values |
| **Episode length** | Aim for 15–30 seconds per episode (SC is harder) |

### Session rhythm

1. Pick a scene configuration, launch Gazebo
2. Record 5–8 episodes with that configuration
3. Kill Gazebo, change the configuration, relaunch
4. Repeat until you have 50+ good episodes
5. Take a 5-minute break every 20 minutes

---

## 14. Next Steps After Collection

Once you have 50+ SFP and 50+ SC demos:

1. **Train your policy** — use `lerobot-train` with the ACT architecture
2. **Evaluate** — run the trained policy without ground truth and check scores
3. **Iterate** — identify failure modes, collect targeted demos for weak areas, retrain
