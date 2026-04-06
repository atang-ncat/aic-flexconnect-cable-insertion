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

### Connector Types

The competition uses a **dual-ended cable** (`sfp_sc_cable`) with two different connectors:

| End | Connector | Target Port | Insertion Direction | Physical Character |
|-----|-----------|-------------|--------------------|--------------------|
| **SFP Module** | SFP transceiver | SFP port on **NIC Card** | **Vertical (top-down)** | Tight slot with physical guide rails — self-correcting once aligned |
| **SC Plug** | SC fiber-optic | **SC Port** (standalone, yellow housing) | **Horizontal** | Wider tolerance, no self-correction — requires precise alignment |

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
- Multiple distractor ports (SFP mounts, SC mounts, LC mounts) are present on the board — the policy must target the correct one

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
> **75% of the score is insertion success.** Even getting close to the port (proximity) scores up to 25 pts. A policy that reliably approaches the target port without inserting can still qualify with ~78 pts across 3 trials.

### Tier 2 Scoring — Only Awarded If Close

Tier 2 points (duration, smoothness, efficiency) are **only awarded if Tier 3 > 0** — meaning the plug must at least be in proximity to the target port. If the plug ends up far from the port, Tier 2 = 0.

---

## 2. Strategic Approach — Imitation Learning (ACT)

### Why Imitation Learning?

| Factor | ACT (Imitation Learning) | Pure RL (Isaac Lab) |
|--------|:------------------------:|:-------------------:|
| Demo availability | ♾️ unlimited via CheatCode | N/A |
| Cable physics required | No — learns from Gazebo directly | Yes — Isaac Lab has none |
| Sim-to-sim transfer risk | **None** (train & eval both in Gazebo) | **High** (Isaac → Gazebo) |
| Sample efficiency | ~100–200 demos | Millions of steps |
| Time to first result | ~2–3 weeks | ~5+ weeks |
| Proven on this task | ✅ organizer baseline exists | ❌ reward engineering needed |

The organizers **designed this pipeline intentionally**: CheatCode is the "teacher," ACT is the "student," and the LeRobot toolchain connects them.

### Is This Legal?

**Yes — 100%.** From `challenge_rules.md`, section 2c:

> *"During training, participants may use all internal state information, including ground truth data available over the `/tf` topic."*

This explicitly permits running CheatCode with `ground_truth:=true` to generate training data. Ground truth is simply unavailable during evaluation — the organizers control the flag.

### Can CheatCode Be Submitted Directly?

**No.** CheatCode depends on TF frames (`task_board/{module}/{port}_link`) that are only published when `ground_truth:=true`. During evaluation, these frames don't exist and CheatCode crashes.

### The Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TRAINING  (ground_truth:=true)                        │
│                                                                          │
│  SFP Demos:  CheatCode auto-inserts → filter by /scoring/insertion_event │
│  SC Demos:   Human teleoperation → manual quality check                  │
│                                                                          │
│  Record: 3 cameras + joint states + F/T sensor + TCP pose + actions      │
│  Vary: board pose, rail translations, grasp offsets (~2mm / ~0.04 rad)   │
│                                                                          │
│  Train ACT: policy maps [cameras, joints, F/T, Task] → twist commands   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   EVALUATION  (ground_truth:=false)                      │
│                                                                          │
│  ACT policy receives: 3 cameras + joints + F/T + Task message            │
│  ACT policy outputs: Twist velocity commands to aic_controller           │
│  No TF frames needed — the policy learned visual servoing from demos     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Collection Plan

### SFP Demos — Automated via CheatCode

**Method:** Run CheatCode teacher policy across randomized NIC card configurations. Filter successful insertions using `/scoring/insertion_event` topic as ground-truth oracle.

**Why this works:** CheatCode reliably inserts SFP connectors ~60–75% of the time. Failed attempts are automatically discarded. We only need ~100–200 successful demos.

**Automation loop:**
1. Launch Gazebo with randomized board pose / NIC rail translation
2. Start rosbag recording (cameras + joints + F/T + actions)
3. Run CheatCode via `InsertCable` action goal
4. Monitor `/scoring/insertion_event` — if received, keep bag; else discard
5. Reset simulation, repeat with new randomization

**Randomization ranges:**
- Board X: 0.12 – 0.20m
- Board Y: -0.25 – 0.05m
- Board yaw: 2.8 – 3.5 rad
- NIC rail translation: -0.0215 – 0.0234m
- Cable gripper offset Z: ±0.003m (grasp variation)

### SC Demos — Manual Teleoperation

**Method:** Human teleoperation using the provided `aic_teleoperation` tools.

**Why manual:** Our testing confirmed CheatCode's PI controller cannot handle SC port insertion due to oscillation and lack of physical self-correction in the port geometry. Teleoperation produces high-quality demos.

**Target:** 50–100 SC insertion demos with similar randomization.

### Data Format

**Option A (Primary):** Record rosbags, then convert to LeRobot HDF5 format using a custom conversion script.

**Option B (Backup):** If `lerobot-record` can be adapted to work alongside CheatCode (non-neural-network policy), use it directly.

> [!NOTE]
> `lerobot-record` expects a `PreTrainedPolicy` (neural network). CheatCode is a ROS Lifecycle node, not a LeRobot policy. We will need either a wrapper or the rosbag → HDF5 conversion path.

---

## 4. Training Plan

### Policy Architecture — ACT (Action Chunking with Transformers)

ACT is the recommended architecture from the organizers. Key features:
- **Chunk prediction:** Predicts sequences of future actions (not single-step), improving temporal consistency
- **Multi-modal input:** Fuses camera images + joint states + F/T readings
- **Task conditioning:** The `Task` message (plug type, port name, target module) is encoded as a conditioning signal so one policy handles both SFP and SC

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Policy type | `act` | Proven for manipulation, organizer baseline exists |
| Chunk size | 100 | Standard for ACT, sufficient for insertion trajectory |
| Image resolution | 480×640 (3 cameras) | Native camera resolution |
| Learning rate | 1e-5 | Standard ACT default |
| Batch size | 8–16 | GPU memory dependent |
| Training steps | 100K–300K | Typical for 100–200 demos |
| Augmentation | Color jitter, random crop | Domain randomization for visual robustness |
| Task conditioning | Embed `plug_type` + `port_type` as categorical token | Single policy for SFP + SC |

### Evaluation Protocol

After training, evaluate in Gazebo with `ground_truth:=false`:
1. Run all 3 trial types from the sample config
2. Record total score across trials
3. Identify failure modes (which trial fails, why)
4. Collect targeted demos for failure cases and retrain

---

## 5. Task Breakdown

### Phase 0: Environment Setup ✅ DONE
- [x] Get Gazebo eval container running in distrobox
- [x] Run CheatCode for successful SFP insertion
- [x] Confirm `/scoring/insertion_event` topic is available
- [x] Set up git repo and push to GitHub
- [x] Verify CheatCode behavior across different port types
- [x] Add orientation-aware CheatCode patch for reference

### Phase 1: Automated Data Collection Pipeline (1 week)
- [ ] Write Python orchestration script (`collect_sfp_demos.py`):
  - [ ] Randomize scene parameters (board pose, rail translations)
  - [ ] Launch Gazebo programmatically with randomized config
  - [ ] Start rosbag recording
  - [ ] Trigger CheatCode policy (lifecycle + action goal)
  - [ ] Monitor `/scoring/insertion_event` for success
  - [ ] Save successful bags, discard failures
  - [ ] Reset and loop
- [ ] Write rosbag → LeRobot HDF5 conversion script
- [ ] Collect **100+ successful SFP insertion demos**
- [ ] Validate dataset: spot-check 10 demos visually

### Phase 2: SC Demo Collection via Teleoperation (3–5 days)
- [ ] Set up teleoperation environment
- [ ] Practice SC insertion manually (5–10 warmup runs)
- [ ] Collect **50+ successful SC insertion demos**
- [ ] Apply same randomization as SFP (board pose, rail translation)
- [ ] Convert to same HDF5 format as SFP demos
- [ ] Merge SFP + SC into unified dataset

### Phase 3: ACT Training (1 week)
- [ ] Configure ACT training with LeRobot
- [ ] Train initial model on SFP-only dataset
- [ ] Evaluate SFP-only model in Gazebo (Trials 1 & 2)
- [ ] Train combined SFP + SC model with task conditioning
- [ ] Evaluate combined model on all 3 trials
- [ ] Identify failure modes and collect targeted demos

### Phase 4: Iteration & Hardening (1–2 weeks)
- [ ] Collect more demos for failure cases
- [ ] Hyperparameter tuning (chunk size, learning rate, augmentation)
- [ ] Increase domain randomization coverage
- [ ] Stress test across 20+ randomized configs
- [ ] Target: **≥75 pts on Trials 1 & 2, ≥50 pts on Trial 3**

### Phase 5: Docker & Submission (1 week)
- [ ] Package policy into Docker following `submission.md`
- [ ] Verify ROS 2 Lifecycle compliance:
  - `unconfigured` → `configured` → `active` → `deactivate` → `cleanup` → `shutdown`
- [ ] Test full local evaluation mimicking cloud pipeline
- [ ] Confirm policy handles all 3 trials end-to-end
- [ ] Submit to evaluation portal
- [ ] Iterate based on leaderboard results

### Phase 6: Polish & Buffer (remaining time)
- [ ] Optimize Tier 2 scores (smoothness, speed, efficiency)
- [ ] Fine-tune force control to avoid penalties
- [ ] Final submission before deadline

---

## 6. Key Technical Decisions

### Why Rosbags Over lerobot-record?

`lerobot-record` is designed for neural network policies (expects `PreTrainedPolicy`). CheatCode is a ROS node that receives action goals. Rather than building a complex wrapper, we record standard rosbags containing all sensor data + robot commands, then convert to LeRobot's HDF5 format. This is:
- **Simpler** — no adapter needed
- **Debuggable** — can replay bags with `ros2 bag play`
- **Flexible** — same pipeline works for both CheatCode and teleoperation demos

### Why Not Pure RL?

- Isaac Lab has **no cable physics** — the deformable cable is Gazebo-specific
- Training RL in Gazebo is too slow (no GPU-accelerated parallel envs)
- Sim-to-sim transfer (Isaac → Gazebo) is an unsolved problem for deformable objects
- Reward engineering for sub-millimeter insertion is extremely difficult

### Why Not Residual RL Fine-Tuning?

Only worth considering if already scoring 85+ consistently. The engineering cost of building an RL fine-tuning pipeline on top of ACT outweighs the potential 5–10 point improvement from Tier 2 bonuses.

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| CheatCode success rate too low | Slow data collection | Filter by `/scoring/insertion_event`; only need ~100 successes out of ~200 runs |
| SC teleoperation too difficult | Can't collect SC demos | Focus SFP first (200 pts from Trials 1 & 2 alone); train SC later |
| ACT doesn't generalize to new board poses | Low eval scores | Include 15+ different board configurations in training data |
| Grasp variation breaks insertion | Failed trials at eval | Add ±2mm gripper offset perturbations to demo collection |
| Submission Docker doesn't build | Can't submit | Start Docker packaging in Phase 5, not last minute |
| Proximity-only scores too low | Fail to qualify | Even 25 pts/trial × 3 = 75 pts; combined with validity = 78 pts minimum |

---

## 8. Timeline Summary

```
Week 1 (Apr 7–13):   Phase 1 — Automated SFP collection pipeline + collect 100 demos
Week 2 (Apr 14–20):  Phase 2 — SC teleoperation demos + merge dataset
Week 3 (Apr 21–27):  Phase 3 — ACT training + initial evaluation
Week 4 (Apr 28–May 4): Phase 4a — Iterate on failures, collect more demos
Week 5 (May 5–11):   Phase 4b + Phase 5 — Docker packaging + local testing
Week 6 (May 12–15):  Phase 6 — Submit + polish + buffer
```

> [!CAUTION]
> **Do NOT leave submission packaging to the last week.** Docker/container bugs are the #1 reason teams fail competitions. Build and test the container in Week 5.

---

## 9. Success Criteria

| Milestone | Target | When |
|-----------|--------|------|
| SFP demo collection working | 100+ successful bags | End of Week 1 |
| SC demos collected | 50+ demos | End of Week 2 |
| ACT model trained and evaluated | Score > 0 on all 3 trials | End of Week 3 |
| Consistent scoring | ≥75 pts on Trials 1 & 2 | End of Week 4 |
| Submission working end-to-end | Docker passes local eval | End of Week 5 |
| **Qualification** | **≥ 150 pts total** | **May 15** |
