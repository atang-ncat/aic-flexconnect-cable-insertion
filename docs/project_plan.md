# 🏆 Project FlexConnect — AIC Competition Plan

> **Team Apex Autonomy** · AI for Industry Challenge  
> Organized by Intrinsic / Google DeepMind / NVIDIA  
> **Qualification Deadline: May 15, 2026**

---

## 1. Problem Description

### What is the AIC?

The AI for Industry Challenge (AIC) is a **precision fiber-optic cable insertion** task. A UR5e robot arm (equipped with a Robotiq Hand-E gripper, Axia80 force-torque sensor, and 3 wrist cameras) must insert cable connectors into target ports on a randomized task board.

> [!IMPORTANT]
> **The robot starts with the plug already grasped**, within centimeters of the port. There is **no grasping problem** to solve. The entire challenge is:
> 1. **Visually locate** the target port from camera images
> 2. **Align** the connector with sub-millimeter precision
> 3. **Execute** a smooth, gentle insertion without excessive force or collisions

### Connector Types & Port Identification

The competition uses a **dual-ended cable** (`sfp_sc_cable`) with two different connectors:

| End | Connector | Target Port | Visual ID on Task Board |
|-----|-----------|-------------|------------------------|
| **SFP Module** | SFP transceiver (rectangular metallic) | SFP port on **NIC Card** (green PCB) | The **green circuit board** with a rectangular slot on top |
| **SC Plug** | SC fiber-optic connector (white rectangular) | **SC Port** (blue housing) | The small **blue box** mounted on a rail on the task board |

From the simulation screenshots:
- The **NIC Card** is the green PCB (printed circuit board) visible on the task board — its SFP port is the rectangular slot on top where the SFP transceiver drops in vertically
- The **SC Port** is the small **blue box** mounted on the dark task board surface — it has a rectangular opening on its top face

### Insertion Direction — Both Are Vertical (Top-Down)

> [!IMPORTANT]
> **Important correction:** Based on our hands-on testing, **both SFP and SC insertions are approximately vertical (top-down)** when the ports are mounted on the task board in the default configuration. The SC port, despite being a "horizontal" connector in real-world fiber optics, is mounted on the task board with its opening **facing upward**, making insertion a downward motion — same general direction as SFP into the NIC card.

The key difference is precision, not direction:
- **SFP port** has tight guide rails that physically correct small misalignments → forgiving
- **SC port** has wider tolerance with no self-correction → requires precise alignment from the policy

The cable type in the scene determines which end faces forward:
- `sfp_sc_cable` → SFP end is the insertion tip
- `sfp_sc_cable_reversed` → SC end is the insertion tip

### Qualification Trials

> [!IMPORTANT]
> **You do NOT choose which port type to target.** The evaluation engine runs ALL three trials automatically against your single submitted policy. Your policy receives a `Task` message specifying the cable, plug, port, and target module — it must handle both SFP and SC insertions.

| Trial | Cable Type | Plug | Target Port | Randomized Parameters |
|:-----:|-----------|------|-------------|----------------------|
| **1** | `sfp_sc_cable` | SFP Module (`sfp_tip`) | SFP port on NIC Card (`nic_card_mount_0`) | Board pose, NIC rail translation |
| **2** | `sfp_sc_cable` | SFP Module (`sfp_tip`) | SFP port on NIC Card (`nic_card_mount_1`) | Board pose, NIC rail translation (different slot) |
| **3** | `sfp_sc_cable_reversed` | SC Plug (`sc_tip`) | SC Port (`sc_port_1`) | Board pose, board yaw (3.0 rad), SC rail translation |

Key observations from the sample config:
- Trials 1 & 2 use the **same cable orientation** but different NIC card slots
- Trial 3 uses the **reversed cable** and a different board yaw angle
- The board pose, rail translations, and yaw all vary between trials
- Multiple distractor ports (SFP mounts, SC mounts, LC mounts) are present — the policy must target the **correct one**
- The `Task` message tells the policy exactly which port to target

### What Your Policy Sees During Evaluation

> [!WARNING]
> Evaluation runs with `ground_truth:=false`. Your policy has **NO access** to TF frames for ports or plugs.

| Available ✅ | NOT Available ❌ |
|---|---|
| 3 camera images (left / center / right) | Port / plug TF frames |
| Joint states (6 arms + 2 gripper) | Ground truth object poses |
| Force / torque sensor readings (tared at startup) | Gazebo internal state |
| Controller state (TCP pose, velocity, error) | `/scoring/*` topics |
| Robot TF frames (`base_link` → `gripper/tcp`) | Any simulation backdoor |
| `Task` message (cable name, plug name, port name, target module) | |

Your policy must **locate the port using only cameras + proprioception** and then perform the insertion.

### Scoring Breakdown (100 pts max per trial, 300 total)

| Tier | Category | Points | Notes |
|------|----------|:------:|-------|
| **1** | Model validity | **1** | Policy loads and responds to action requests |
| **2** | Task duration | **0–12** | ≤ 5s → max, ≥ 60s → 0 |
| **2** | Trajectory smoothness | **0–6** | Low jerk (Savitzky-Golay filtered) |
| **2** | Trajectory efficiency | **0–6** | Short path length |
| **2** | Force penalty | **0 to −12** | >20N for >1s |
| **2** | Collision penalty | **0 to −24** | Any contact with off-limit areas |
| **3** | **Insertion success** | **75** | Correct port. **Wrong port = −12** |
| **3** | Partial insertion | **38–50** | Plug inside port bounding box, proportional to depth |
| **3** | Proximity | **0–25** | Closer to port = more points |

> [!IMPORTANT]
> **75% of the score is insertion success.** Even getting close to the port (proximity) scores up to 25 pts. A policy that reliably approaches the target port without inserting can still score ~78 pts across 3 trials.

**Tier 2 points are only awarded if Tier 3 > 0** — the plug must at least be in proximity to the target port.

---

## 2. Strategic Approach — End-to-End Visuomotor Policies

### Why End-to-End Over Modular Pipelines?

We advocate strongly for **end-to-end visuomotor policies** over the traditional perception → planning → control pipeline.

| Factor | End-to-End 🏆 | Modular (Perception + Planning + Control) |
|--------|:-----------:|:---------------------------------------:|
| **Error propagation** | None — single model | Errors compound across modules |
| **Integration effort** | One model to train | 3+ components to integrate & debug |
| **Adaptability** | Learns subtle visual-motor correlations | Each module has independent failure modes |
| **Force feedback** | Naturally integrates F/T with vision | Requires explicit fusion logic |
| **Compliance** | Learns gentle insertion from demos | Requires separate impedance controller tuning |
| **Speed to iterate** | Retrain one model | Debug across multiple modules |

End-to-end policies learn the **implicit mapping** from raw sensor inputs (cameras, joint states, F/T) directly to motor commands. This is exactly what ACT, Diffusion Policy, and HIL-SERL provide — they see what the robot sees and produce what the robot does, with no hand-engineered perception or planning in between.

### Policy Candidates

We will explore **multiple policy architectures** in parallel, leveraging our 5-member team:

#### 1. ACT — Action Chunking with Transformers (Primary)
- **What:** Transformer-based imitation learning that predicts sequences (chunks) of future actions
- **Strengths:** Proven on this exact task (organizer baseline exists), temporally consistent, fast inference
- **Data needs:** ~100–200 demonstrations
- **Reference:** LeRobot toolkit, `grkw/aic_act_policy` pre-trained model

#### 2. Diffusion Policy (Parallel Track)
- **What:** Models the action distribution as a conditional denoising diffusion process
- **Strengths:** Excels at multimodal action distributions (critical when multiple approach strategies are valid), handles high-dimensional action spaces, consistently outperforms prior IL methods by 40%+ on benchmarks
- **Data needs:** ~100–200 demonstrations (same dataset as ACT)
- **Inference speed concern:** Multiple denoising steps can be slow; mitigated by OneDP (single-step distillation) or DDIM sampling with fewer steps
- **Reference:** Chi et al. 2023, LeRobot has built-in support

#### 3. RL Fine-Tuning — Optional for Either Team

> [!NOTE]
> **RL fine-tuning (e.g., HIL-SERL) is an optional stretch goal** that either team can explore for final-stage policy refinement. This is **not** a core deliverable — it's available to whichever team gets their baseline working first and wants to push scores higher.

- **What:** Human-in-the-loop sample-efficient RL that fine-tunes a policy with online interventions
- **Strengths:** Achieves >95% success on insertion tasks within 1–2.5 hours of training
- **When to use:** After ACT or Diffusion Policy reaches ~80%+ success rate
- **Compute:** ✅ We have the GPU compute to handle RL training. **We do NOT need Isaac Lab to parallelize training** — our hardware is sufficient for the RL loop in Gazebo directly.
- **Reference:** Luo et al. 2024, `rail-berkeley/hil-serl`

> [!TIP]
> **Our recommended progression:** Train ACT + Diffusion Policy in parallel → pick best performer → optionally fine-tune with RL for the final push to >95% success rate.

### Is Using CheatCode Demos Legal?

**Yes — 100%.** From `challenge_rules.md`, section 2c:

> *"During training, participants may use all internal state information, including ground truth data available over the `/tf` topic."*

Ground truth is simply unavailable during evaluation — the organizers control the flag.

### Can CheatCode Be Submitted Directly?

**No.** CheatCode depends on TF frames that only exist when `ground_truth:=true`. During evaluation, these don't exist and CheatCode crashes.

---

## 3. Team Structure & Assignments

### Active Team Members (5)

| Member | Subteam |
|--------|:-------:|
| **Emiralp** | Team Alpha |
| **Lahari Sri Kari** | Team Alpha |
| **Andrews** | Team Alpha |
| **Vijay** | Team Beta |
| **Devi** | Team Beta |

### Subteam Responsibilities

#### Team Alpha (3 members) — ACT Route
- **Data collection path:** Automated SFP demo collection via CheatCode pipeline
- ACT policy training, evaluation, and iteration
- Submission packaging (Docker, Lifecycle compliance)

#### Team Beta (2 members) — Diffusion Policy Route
- **Data collection path:** Teleoperation demos (SC port + supplementary SFP)
- Diffusion Policy training, evaluation, and iteration

### Shared Responsibilities
- **Dataset is shared** — both teams use the same combined SFP + SC demo dataset
- **Data augmentation & domain randomization** — both teams contribute (can be applied during training or as a separate dataset expansion step)
- **Inference speed optimization** — both teams explore techniques (DDIM sampling, OneDP distillation, etc.)
- **Evaluation protocol is unified** — same 3-trial benchmark for comparing policies
- **Best-performing policy gets submitted** — healthy competition, best model wins
- **RL fine-tuning is optional** — either team can explore this once their baseline is working

> [!NOTE]
> **On RL fine-tuning:** We have the compute resources to handle RL training workloads. There is **no need for Isaac Lab** or parallel simulation environments — we can run the RL fine-tuning loop directly in Gazebo on our hardware. This is a bonus optimization, not a dependency.

---

## 4. Data Collection Plan

### Critical Constraint: Action Space Mismatch

> [!CAUTION]
> **CheatCode and the training pipeline use DIFFERENT action spaces.** You cannot simply record CheatCode's output as training data.
>
> | Component | Action Mode | Action Format |
> |-----------|------------|---------------|
> | **CheatCode** → `set_pose_target()` | `MODE_POSITION` | Target Pose (position + orientation) |
> | **RunACT** → `set_cartesian_twist_target()` | `MODE_VELOCITY` | 6D Cartesian Twist (linear.xyz + angular.xyz) |
> | **lerobot-record** / `AICRobotAICController` | `MODE_VELOCITY` | 6D Cartesian Twist |
>
> If you record CheatCode's position commands and train a policy on them, at inference time RunACT would interpret the outputs as velocities. **The robot would do something completely wrong.**

### What's Ready to Use Today

| Tool | Status | Purpose |
|------|:------:|---------|
| `lerobot-record` | ✅ Ready | Records teleop demos in native LeRobot format (keyboard/spacemouse) |
| `lerobot-train` | ✅ Ready | Trains ACT / Diffusion Policy from LeRobot datasets |
| `grkw/aic_act_policy` | ✅ Ready | Pre-trained ACT baseline on HuggingFace — free insurance |
| `lerobot-teleoperate` | ✅ Ready | Practice teleop before recording |
| CheatCode (fixed) | ✅ Works for SFP+SC | But outputs position actions — **incompatible with training format** |

### What Does NOT Exist Yet

- ❌ No CheatCode velocity-mode wrapper/recorder
- ❌ No batch/randomized scene launcher script
- ❌ No rosbag-to-LeRobot conversion script
- ❌ No committed datasets

### Dual-Track Collection Strategy

```
┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│ TRACK A: Manual Teleop via lerobot-record│  │ TRACK B: CheatCode Velocity Wrapper      │
│ (PRIMARY — works TODAY)                  │  │ (PARALLEL — needs engineering)            │
│                                          │  │                                          │
│ • Keyboard/spacemouse teleop             │  │ • CheatCode computes target pose         │
│ • Correct velocity action format         │  │ • Wrapper converts to velocity commands  │
│ • All 5 team members can contribute      │  │ • Records via AICRobotAICController      │
│ • Diverse approaches + recovery behavior │  │ • Filter by /scoring/insertion_event     │
│ • ~30-40 demos/hour per operator         │  │ • Scalable to hundreds of demos          │
│ • Target: 50+ SFP, 50+ SC demos         │  │ • Target: 100+ automated demos           │
└────────────────────┬─────────────────────┘  └────────────────────┬─────────────────────┘
                     │                                             │
                     └─────────────────┬───────────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │  UNIFIED TRAINING DATASET  │
                         │  LeRobot HDF5 format       │
                         │  6D velocity actions       │
                         │  ~200+ demos total         │
                         └─────────┬─────────────────┘
                                   │
                     ┌─────────────┼─────────────────┐
                     ▼             ▼                 ▼
                ┌────────┐   ┌──────────────┐  ┌──────────┐
                │  ACT   │   │  Diffusion   │  │ RL Tuning│
                │ Policy │   │   Policy     │  │(optional)│
                └────────┘   └──────────────┘  └──────────┘
```

### Track A: Manual Teleop via lerobot-record (Primary — All Team Members)

**This is the primary data source. It works today and produces correctly formatted data.**

```bash
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=${HF_USER}/aic_sfp_demos \
  --dataset.single_task="insert sfp module into nic card port" \
  --dataset.push_to_hub=false --display_data=true --play_sounds=false
```

**Key guidelines:**
- Rotate scene configs between sessions (vary board pose, rail translations)
- Each demo takes ~60-90 seconds including reset
- Record diverse approaches: fast/slow, different angles, include recovery moments
- **Imperfect demos are valuable** — slight wobbles and corrections teach the policy to recover
- All 5 team members can contribute immediately

### Track B: CheatCode Velocity Wrapper (Parallel — 1-2 engineers)

**Engineering task:** Write a script that uses CheatCode's `calc_gripper_pose` logic to compute target poses, then converts to velocity commands compatible with the LeRobot pipeline.

The core conversion:
```python
# CheatCode computes a target pose
target_pose = calc_gripper_pose(port_transform, z_offset=z_offset)

# Convert position target to velocity command
current_pose = observation["tcp_pose"]
linear_vel = K_p * (target_position - current_position)
angular_vel = K_r * orientation_error(target_orientation, current_orientation)

# Record as 6D velocity action (compatible with LeRobot)
action = [linear_vel.x, linear_vel.y, linear_vel.z,
          angular_vel.x, angular_vel.y, angular_vel.z]
```

**Automation loop:**
1. Launch Gazebo with randomized config
2. Connect via `AICRobotAICController.get_observation()` and `send_action()`
3. Run CheatCode velocity logic, recording each `(observation, action)` pair
4. Monitor `/scoring/insertion_event` for success filtering
5. Save successful episodes as LeRobot dataset
6. Reset and repeat

### Why Teleop Demos Are Better Than CheatCode-Only

| What teleop provides | What CheatCode-only would lack |
|---------------------|-------------------------------|
| **Diverse approach paths** | Same mathematical trajectory every time |
| **Recovery behaviors** | Never encounters or recovers from misalignment |
| **Force-aware insertion** | Ignores F/T sensor entirely |
| **Broad state distribution** | Narrow — always visits same states |
| **Natural speed variation** | Fixed 10mm/s descent |

### Teleoperation Guide — How to Record High-Quality Demos

> [!TIP]
> **Read this before touching the keyboard.** The quality of your demos directly determines the quality of your policy. A dataset of 50 excellent demos will outperform 200 sloppy ones.

#### Setup

```bash
# Terminal 1: Launch Gazebo with ground truth enabled (for scoring feedback)
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005

# Terminal 2: Practice teleop BEFORE recording
cd ~/ws_aic/src/aic
pixi run lerobot-teleoperate \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --display_data=true

# Terminal 3: Monitor insertion success
ros2 topic echo /scoring/insertion_event
```

#### Keyboard Controls (Cartesian mode, `base_link` frame)

| Key | Motion | Think of it as... |
|-----|--------|-------------------|
| `a` / `d` | -X / +X (left / right) | Sliding the arm left-right |
| `w` / `s` | -Y / +Y (forward / back) | Pushing arm forward or pulling back |
| `r` / `f` | -Z / +Z (up / down) | **This is your insertion axis** — `f` pushes DOWN into the port |
| `q` / `e` | -Yaw / +Yaw | Rotating the connector clockwise/counter-clockwise |
| `Shift+w` / `Shift+s` | +Roll / -Roll | Tilting the connector forward/back |
| `Shift+a` / `Shift+d` | -Pitch / +Pitch | Tilting the connector left/right |
| `t` | Toggle slow/fast mode | **Start in slow mode. Always.** |

> [!IMPORTANT]
> **Shift+key release order matters:** Let go of the letter key BEFORE releasing Shift. Otherwise the robot keeps rotating after you release both keys.

#### Recording Controls (during `lerobot-record`)

| Key | Function |
|-----|----------|
| Right Arrow | Finish current episode, start next |
| Left Arrow | Discard current episode, re-record |
| ESC | Stop recording session |

#### The 5 Phases of a Good Demo

**Phase 1: Orient (1-2 seconds).** Before moving, look at the camera feeds (`--display_data=true`). Identify the target port. For SFP, find the green NIC card. For SC, find the blue box.

**Phase 2: Coarse approach (3-5 seconds).** Use `a`/`d` and `w`/`s` to slide the arm laterally until the connector is roughly above the target port. Use fast mode (`t`) for this. Do NOT descend yet.

**Phase 3: Fine alignment (3-5 seconds).** Switch to slow mode (`t`). Use small taps on `a`/`d`/`w`/`s` to center the connector precisely over the port opening. Watch the center camera feed — the connector tip should be directly above the port center.

**Phase 4: Insertion (3-5 seconds).** Press `f` (descend) in slow mode. Use short, gentle taps. Watch the force readings if visible. If you feel resistance (the connector stops moving down), STOP. Nudge laterally with `a`/`d`/`w`/`s` to realign, then continue descending.

**Phase 5: Confirmation (1-2 seconds).** Hold position briefly. Check terminal 3 for `/scoring/insertion_event`. If it fires, press Right Arrow to save. If not, try gentle adjustments or press Left Arrow to discard and retry.

#### Golden Rules for Demo Quality

1. **Slow is fast.** Gentle, deliberate movements produce better training signal than fast, jerky ones. The policy learns your speed profile — if you rush, it rushes.

2. **Include corrections.** If you overshoot, correct it visibly. Do NOT discard the demo. The policy needs to learn what "being slightly wrong and fixing it" looks like. These recovery moments are the most valuable training data.

3. **Vary your approach.** Do not always follow the exact same path. Sometimes approach from the left, sometimes from the right. Sometimes go slow, sometimes slightly faster. This diversity is what makes the policy robust.

4. **One thing at a time.** Move in one axis at a time. Don't press `d` and `f` simultaneously. Sequential movements are easier for the policy to learn than diagonal ones.

5. **Start every demo from the same initial state** (reset the sim between episodes). But vary the scene config across recording sessions.

6. **Record 5 practice demos first and discard them.** Your first few will be terrible. That's normal. Get comfortable with the controls before recording for real.

7. **Take breaks.** Fatigue kills demo quality. Record in 20-minute sessions with 5-minute breaks.

### Alternative Data Collection — What If We Had No CheatCode and No Teleop?

> [!NOTE]
> This section discusses hypothetical alternatives. We have both CheatCode and teleop available — these are fallback ideas or supplementary strategies worth knowing about.

Even without a ground-truth teacher policy or human teleop, there are several creative approaches to collecting training demonstrations:

#### 1. Scripted Trajectory Generation
Write a simple Python script that generates smooth waypoint trajectories toward the port based on the `Task` message fields. At training time, use ground-truth TF frames to compute the target position. Generate trajectories as sequences of velocity commands:

```python
# Compute a straight-line velocity trajectory from current TCP to target
direction = normalize(target_pos - current_tcp_pos)
velocity = direction * speed  # constant speed along the line
```

Unlike CheatCode, this would output velocity actions (compatible with training). The trajectories would be geometrically simple (straight lines or splines) but still provide useful approach demonstrations. This is essentially a "dumb CheatCode" that does not need a PI controller.

#### 2. DAgger with the Pre-Trained Baseline
Run the pre-trained `grkw/aic_act_policy` in the sim. When it succeeds, keep the episode. When it fails, record the failure trajectory anyway (the approach portion is still useful), and optionally have a human correct the final alignment via teleop. This is a form of Dataset Aggregation (DAgger) — using the current policy to generate data, then augmenting with corrections.

#### 3. Replay-Based Data Collection (Isaac Lab Path)
The workspace includes Isaac Lab demo recording and replay scripts at `src/aic/aic_utils/aic_isaac/aic_isaaclab/scripts/record_demos.py`. Isaac Lab supports massive parallelization (hundreds of environments simultaneously), randomization events, and HDF5 output. If you train an RL policy in Isaac Lab first (even a rough one), you can record its rollouts as demonstrations, then convert to LeRobot format for ACT training. This is a "sim-to-sim" transfer approach.

#### 4. Inverse Kinematics Trajectory Sampling
Using the known port position (from ground-truth TF at training time) and the robot's URDF, compute IK solutions for a sequence of waypoints along the insertion path. Sample multiple approach angles and speeds. Convert joint trajectories to Cartesian velocities via the Jacobian. This produces diverse, physically valid demonstrations without any controller or human input.

#### 5. Motion Planning + Noise Injection
Use a standard motion planner (MoveIt, OMPL) to generate collision-free paths from the starting configuration to the port. Add Gaussian noise to the planned trajectories to create diversity. Record the noisy trajectories as demonstrations. The noise teaches the policy to handle imperfect states.

#### 6. Cross-Simulator Transfer
Train a basic policy in MuJoCo (which has a mirror environment at `src/aic/aic_utils/aic_mujoco/`) where physics may differ. Record successful trajectories. Use these as demonstrations for Gazebo training. The physics mismatch acts as implicit domain randomization.

The key insight across all these approaches: **you do not need a human to demonstrate the task if you have ground-truth state at training time.** Any method that can generate trajectories ending with the plug in the port produces valid training data — the learned policy will then reproduce these trajectories using only camera observations.

### Data Augmentation (Both Teams)

Once the base dataset is collected:
- **Color / brightness jitter** for visual robustness
- **Random cropping** to simulate camera viewpoint variation
- **Temporal augmentation** (speed variation) for trajectory diversity
- **Synthetic noise injection** on F/T readings

---

## 5. Training Plan

### Policy Architecture Comparison

| Feature | ACT | Diffusion Policy | RL Fine-Tuning (Optional) |
|---------|-----|-----------------|------------|
| **Type** | Imitation Learning | Imitation Learning | RL (fine-tuning) |
| **Action representation** | Chunk of K future actions | Denoised action sequence | Single-step policy |
| **Multimodal handling** | VAE latent space | Naturally multimodal (diffusion process) | Standard policy gradient |
| **Inference speed** | Fast (~50Hz) | Slower (10–20 denoising steps) | Fast |
| **Data needs** | 100–200 demos | 100–200 demos | Pre-trained policy + 1–2 hrs online |
| **Best for** | Baseline, quick results | Complex alignment strategies | Pushing from 80% → 95%+ |

### Training Configuration

| Parameter | ACT | Diffusion Policy |
|-----------|-----|-----------------|
| Framework | LeRobot | LeRobot |
| Chunk size | 100 | 16–32 |
| Image resolution | 288x256 (3 cameras, 0.25x scale) | 288x256 (3 cameras, 0.25x scale) |
| Learning rate | 1e-5 | 1e-4 |
| Batch size | 8–16 | 8–16 |
| Training steps | 100K–300K | 100K–300K |
| Task conditioning | Embed plug_type + port_type | Embed plug_type + port_type |

### Evaluation Protocol (Shared Across Teams)

After training, evaluate in Gazebo with `ground_truth:=false`:
1. Run all 3 trial types from the sample config
2. Record total score across trials
3. Identify failure modes (which trial fails, why)
4. Collect targeted demos for failure cases and retrain
5. **Compare ACT vs Diffusion Policy scores** → submit the better one

---

## 6. Submission Strategy

### Multi-Submission Approach

We are targeting **4 total submissions** leading up to the deadline, ensuring progressive improvement with feedback from the evaluation system.

> [!IMPORTANT]
> **Key assumption:** The competition portal allows multiple submissions per day, and each submission returns scores that help us gauge improvement. If this is not the case, we need to verify with the organizers immediately and adjust our strategy.

| Submission | Target Date | Goal | Expected State |
|:----------:|:-----------:|------|----------------|
| **#1 — Baseline** | **Apr 20** (end of Week 2) | First working policy that can perform insertion | Baseline ACT or Diffusion Policy; scores > 0 on all trials |
| **#2 — Improved** | **Apr 27** (end of April) | Refined policy after one week of iteration | Higher success rate, better scores from targeted fixes |
| **#3 — Competitive** | **May 4** (first week of May) | Polished policy with optimizations | Strong insertion rates, Tier 2 optimizations started |
| **#4 — Final** | **May 12–14** (before deadline) | Best possible policy with all refinements | Maximum scores, RL fine-tuning if applicable |

> [!TIP]
> Each submission gives us score feedback. Use this feedback loop to prioritize what to fix next — don't guess, measure.

### Submission Feedback Loop

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Train   │ ──► │  Submit  │ ──► │  Get Scores  │ ──► │ Analyze  │
│  Policy  │     │  Docker  │     │  Per Trial   │     │ Failures │
└──────────┘     └──────────┘     └──────────────┘     └────┬─────┘
     ▲                                                      │
     └──────────────────────────────────────────────────────┘
                    Iterate & Improve
```

---

## 7. Task Breakdown

### Phase 0: Environment Setup ✅ DONE
- [x] Get Gazebo eval container running in distrobox
- [x] Run CheatCode for successful SFP insertion
- [x] Confirm `/scoring/insertion_event` topic is available
- [x] Set up git repo and push to GitHub
- [x] Verify CheatCode behavior across different port types (SFP ✅, SC ✅ after fix)
- [x] Add orientation-aware CheatCode patch for reference
- [x] Fix CheatCode kinematic targeting bug (plug-to-gripper offset)

---

### Phase 1: Pipeline Validation + Data Collection — Week 1 (Apr 7–13)

> [!CAUTION]
> **Day 1 is critical.** Validate the full end-to-end pipeline (record → train → deploy) BEFORE investing in bulk data collection. The most dangerous mistake is spending all week building infrastructure and ending up with zero trained policies.

**Day 1 — Pipeline Validation (Both Teams):**
- [ ] Test pre-trained `grkw/aic_act_policy` baseline on all 3 trial types (free scores!)
- [ ] Test `lerobot-record` end-to-end with keyboard teleop (3-5 throwaway demos)
- [ ] Verify dataset saves correctly and can be loaded for training
- [ ] Test `lerobot-teleoperate` for practice

**Days 2-7 — Parallel Data Collection:**

**Track A — Manual Teleop (All Team Members, PRIMARY):**
- [ ] Record **50+ SFP insertion demos** via `lerobot-record`
- [ ] Record **50+ SC insertion demos** via `lerobot-record`
- [ ] Rotate scene configs between sessions (vary board pose, rail translations)
- [ ] Include diverse approaches: fast/slow, different angles, recovery moments

**Track B — CheatCode Velocity Wrapper (1-2 engineers, PARALLEL):**
- [ ] Write `cheatcode_velocity_wrapper.py`:
  - [ ] Use CheatCode's `calc_gripper_pose` to compute target poses
  - [ ] Convert position targets to 6D velocity commands
  - [ ] Record via `AICRobotAICController.get_observation()` + `send_action()`
  - [ ] Monitor `/scoring/insertion_event` for success filtering
  - [ ] Save as LeRobot dataset format
- [ ] Collect **100+ automated demos** with randomization (SFP + SC)

**Shared:**
- [ ] Merge Track A + Track B demos into **unified dataset** (~200+ demos)
- [ ] Dataset quality audit: remove corrupted / poor-quality demos
- [ ] Validate dataset format works with both ACT and Diffusion Policy training

---

### Phase 2: ACT vs Diffusion Policy — Parallel Training — Week 2 (Apr 14–20)

> [!IMPORTANT]
> **Both teams train in parallel on the same shared dataset.** By the close of this week, we should have **baseline policies that can perform the insertion task.** This is also our **Submission #1** deadline.

**Team Alpha — ACT:**
- [ ] Train ACT on combined SFP + SC dataset with task conditioning
- [ ] Evaluate on all 3 trial types
- [ ] Record baseline scores
- [ ] Identify failure modes (which trial fails, why)

**Team Beta — Diffusion Policy:**
- [ ] Train Diffusion Policy on same combined dataset
- [ ] Evaluate on all 3 trial types
- [ ] Compare scores vs ACT
- [ ] Optimize inference speed if needed (DDIM, fewer steps)

**Shared:**
- [ ] **Policy comparison meeting**: which performs better on which trials?
- [ ] Package best-performing policy into Docker
- [ ] 🚀 **SUBMISSION #1** — Baseline policy (target: Apr 20)

---

### Phase 3: Iteration, Efficiency & Optimization — Week 3 (Apr 21–27)

> [!IMPORTANT]
> **Focus: make it better.** Use the scores from Submission #1 to drive targeted improvements.

- [ ] Analyze Submission #1 scores per trial — identify weakest points
- [ ] Collect more targeted demos for failure cases
- [ ] Hyperparameter tuning (chunk size, learning rate, augmentation, denoising steps)
- [ ] Increase domain randomization coverage
- [ ] Stress test across 20+ randomized configs
- [ ] Optimize Tier 2 scores (smoothness, speed, trajectory efficiency)
- [ ] Fine-tune force control to avoid penalties
- [ ] **Optional (either team):** Begin RL fine-tuning exploration on best-performing policy
- [ ] 🚀 **SUBMISSION #2** — Improved policy (target: Apr 27, end of month)

---

### Phase 4: Competitive Polish — Week 4 (Apr 28–May 4)

- [ ] Continue iteration based on Submission #2 feedback
- [ ] Push insertion success rate higher on all 3 trials
- [ ] Ensure no collisions (−24 pts) or excessive force (−12 pts)
- [ ] **Optional (either team):** RL fine-tuning if baseline is at ~80%+ success
- [ ] Test submission Docker in full local evaluation (mimicking cloud pipeline)
- [ ] 🚀 **SUBMISSION #3** — Competitive policy (target: May 4, first week of May)

---

### Phase 5: Final Touches — Weeks 5–6 (May 5–14)

> [!CAUTION]
> **Do NOT introduce major changes this late.** Focus on stability, edge cases, and Tier 2 score optimization.

- [ ] Analyze Submission #3 scores — close remaining gaps
- [ ] Final hyperparameter sweep (small adjustments only)
- [ ] Verify ROS 2 Lifecycle compliance is rock-solid:
  - `unconfigured` → `configured` → `active` → `deactivate` → `cleanup` → `shutdown`
- [ ] Run 50+ randomized evaluation trials locally — confirm consistency
- [ ] 🚀 **SUBMISSION #4 — FINAL** (target: May 12–14, before May 15 deadline)

---

## 8. RL Fine-Tuning — Optional Track

> [!NOTE]
> **This section is for either team.** RL fine-tuning is entirely optional and should only be pursued once a team has a working baseline policy with reasonable success rates (~80%+).

### When to Consider RL

| Condition | Action |
|-----------|--------|
| Baseline policy < 50% success | ❌ Don't bother with RL — fix data & architecture first |
| Baseline policy 50–80% success | ⚠️ Maybe — focus on more demos and tuning first |
| Baseline policy > 80% success | ✅ RL fine-tuning can push to 95%+ |

### Compute & Infrastructure

- ✅ **We have the GPU compute** to run RL training workloads
- ✅ **No need for Isaac Lab** to parallelize training — our hardware handles the RL loop directly in Gazebo
- ✅ **HIL-SERL** is designed for sample-efficient fine-tuning (1–2 hours of online interaction)

### Approach

1. Take the best-performing IL policy (ACT or Diffusion Policy)
2. Use HIL-SERL for online fine-tuning with human corrections
3. Focus on the specific trial types where the IL policy struggles
4. Validate improvement on the full 3-trial benchmark before submitting

---

## 9. Why End-to-End Wins for This Task

Traditional modular approaches to robot manipulation build separate components:

```
Camera → [Perception: Detect port] → [Planning: Plan path] → [Control: Execute motion]
```

Each module introduces failure points, requires hand-tuned interfaces, and cannot adapt holistically. For the AIC insertion task specifically:

1. **Sub-millimeter alignment is visuomotor** — the last 2mm of insertion requires simultaneous visual feedback and force modulation. A separate perception module can't provide the closed-loop visual feedback needed at this resolution.

2. **Force-vision coupling is critical** — feeling resistance while watching the connector approach teaches the policy when to push harder vs. realign. Modular pipelines lose this coupling.

3. **The same dataset trains everything** — end-to-end policies learn perception, planning, and control from the same demonstrations. No need to label port locations, plan collision-free paths, or tune impedance controllers separately.

4. **Generalization is built-in** — visual features that matter for insertion (port edges, metallic reflections, connector alignment) are learned in the context of motor actions, not in isolation.

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| CheatCode success rate too low | Slow SFP data collection | Filter by `/scoring/insertion_event`; ~60-75% success rate = 100 demos from ~150 runs |
| SC teleoperation too difficult | Can't collect SC demos | Focus SFP first (200 pts from Trials 1 & 2); train SC later |
| ACT doesn't generalize | Low eval scores | Diffusion Policy as backup; different inductive bias may work better |
| Diffusion Policy too slow | Can't run at real-time | DDIM with fewer steps, or OneDP distillation |
| Grasp variation breaks insertion | Failed trials at eval | Add ±2mm gripper offset perturbations to demos |
| Submission Docker doesn't build | Can't submit | Test submission pipeline starting Week 2, not last minute |
| Proximity-only scores too low | Fail to qualify | Even 25 pts/trial × 3 = 75 pts; enough to qualify |
| One subteam's policy clearly better | Wasted effort | Not wasted — second policy provides fallback and comparison data |
| Competition doesn't allow multiple daily submissions | Can't iterate fast | Verify submission rules early; adjust cadence accordingly |

---

## 11. Execution Timeline Summary

```
TODAY: April 6, 2026 (Sunday)

Week 1 (Apr 7–13):   Phase 1 — Data infrastructure + full dataset curation (COMBINED)
Week 2 (Apr 14–20):  Phase 2 — ACT vs Diffusion Policy parallel training
                      ┗━━ 🚀 SUBMISSION #1 — Baseline policies (Apr 20)
Week 3 (Apr 21–27):  Phase 3 — Iteration, efficiency, optimization
                      ┗━━ 🚀 SUBMISSION #2 — Improved policy (Apr 27)
Week 4 (Apr 28–May 4): Phase 4 — Competitive polish
                      ┗━━ 🚀 SUBMISSION #3 — Competitive policy (May 4)
Weeks 5-6 (May 5–14): Phase 5 — Final touches & stability
                      ┗━━ 🚀 SUBMISSION #4 — FINAL (May 12–14)

                      ━━━ QUALIFICATION DEADLINE: MAY 15 ━━━
```

> [!CAUTION]
> **Submission #1 at end of Week 2 is non-negotiable.** Having a baseline that scores > 0 proves the full pipeline (data → train → Docker → submit → score) works end-to-end. Everything after that is improvement.

---

## 12. Success Criteria

| Milestone | Target | When | Owner |
|-----------|--------|------|-------|
| Pre-trained baseline tested | `grkw/aic_act_policy` scores on all 3 trials known | Day 1 (Apr 7) | Both teams |
| `lerobot-record` pipeline validated | Record, save, and load a demo end-to-end | Day 1 (Apr 7) | Both teams |
| Manual teleop demos flowing | 20+ SFP + 20+ SC demos recorded | Mid-Week 1 | Both teams |
| CheatCode velocity wrapper operational | Automated demo collection running | Mid-Week 1 | 1-2 engineers |
| Full dataset ready | 200+ demos (SFP + SC, merged) | End of Week 1 | Both teams |
| ACT baseline trained & evaluated | Score > 0 on all 3 trials | Mid-Week 2 | Team Alpha |
| Diffusion Policy trained & evaluated | Score > 0 on all 3 trials | Mid-Week 2 | Team Beta |
| **Submission #1** | **Baseline policy submitted** | **Apr 20** | **Both** |
| **Submission #2** | **Improved policy, higher scores** | **Apr 27** | **Both** |
| **Submission #3** | **Competitive policy** | **May 4** | **Both** |
| **Submission #4 -- FINAL** | **Best possible policy** | **May 12-14** | **All** |
| **Qualification** | **>= 150 pts total** | **May 15** | **All** |

---

## 13. Open Questions

> [!WARNING]
> **Verify these with the competition organizers ASAP:**
> 1. **Does the competition allow multiple submissions per day?** Our strategy assumes we can submit frequently and iterate on scores.
> 2. **Do we receive per-trial scores after each submission?** We need this feedback to know which trials are failing and prioritize fixes.
> 3. **Is there a cooldown between submissions?** If so, we need to be more strategic about when we submit.
