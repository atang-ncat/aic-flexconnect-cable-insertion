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

**Terminal 3 — Start teleoperation:**
```bash
source ~/lab/ws_aic/setup_dev.sh
ros2 run aic_teleoperation cartesian_keyboard_teleop
```

**Terminal 4 (optional) — Monitor insertion success:**
```bash
ros2 topic echo /scoring/insertion_event
```

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

## 12. Next Steps After Practice

Once you are comfortable with teleoperation (can reliably insert SFP in <15 seconds and SC in <25 seconds):

1. **Set up recording** — either via `lerobot-record` (if pixi works) or a custom rosbag recording script
2. **Collect SFP demos** — target 50+ episodes across varied scene configs
3. **Collect SC demos** — target 50+ episodes across varied scene configs
4. **Train your policy** — use `lerobot-train` with the ACT architecture
5. **Evaluate** — run the trained policy without ground truth and check scores
