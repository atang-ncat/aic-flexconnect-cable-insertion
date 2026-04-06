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

#### 3. HIL-SERL — RL Fine-Tuning (Stretch Goal)
- **What:** Human-in-the-loop sample-efficient RL that fine-tunes a policy with online interventions
- **Strengths:** Achieves >95% success on insertion tasks within 1–2.5 hours of training, can refine IL policies beyond demonstration quality
- **When to use:** After ACT or Diffusion Policy reaches ~80%+ success — HIL-SERL can push to near-perfect by RL fine-tuning with human corrections
- **Data needs:** Pre-trained IL policy + ~1–2 hrs of online interaction with human corrections
- **Reference:** Luo et al. 2024, `rail-berkeley/hil-serl`

> [!TIP]
> **Our recommended progression:** Train ACT + Diffusion Policy in parallel → pick best performer → optionally fine-tune with HIL-SERL for the final push to >95% success rate.

### Is Using CheatCode Demos Legal?

**Yes — 100%.** From `challenge_rules.md`, section 2c:

> *"During training, participants may use all internal state information, including ground truth data available over the `/tf` topic."*

Ground truth is simply unavailable during evaluation — the organizers control the flag.

### Can CheatCode Be Submitted Directly?

**No.** CheatCode depends on TF frames that only exist when `ground_truth:=true`. During evaluation, these don't exist and CheatCode crashes.

---

## 3. Team Structure & Assignments

### Active Team Members (5)

| Member | Subteam | Primary Role |
|--------|---------|-------------|
| **Andrews** | Team Alpha | Lead: CheatCode automation, data pipeline, infrastructure |
| **Lahari Sri Kari** | Team Alpha | ACT training, hyperparameter tuning, evaluation |
| **Vijay** | Team Alpha | ACT training, data augmentation, domain randomization |
| **Devi** | Team Beta | Diffusion Policy training, architecture experiments |
| **Emiralp** | Team Beta | Teleoperation demos (SC port), Diffusion Policy evaluation |

### Subteam Responsibilities

#### Team Alpha (3 members) — ACT Route
- Automated SFP demo collection via CheatCode pipeline
- ACT policy training, evaluation, and iteration
- Submission packaging (Docker, Lifecycle compliance)

#### Team Beta (2 members) — Diffusion Policy Route
- SC demo collection via teleoperation
- Diffusion Policy training, evaluation, and iteration
- Inference speed optimization (DDIM sampling, OneDP distillation)

### Shared Responsibilities
- **Dataset is shared** — both teams use the same combined SFP + SC demo dataset
- **Evaluation protocol is unified** — same 3-trial benchmark for comparing policies
- **Best-performing policy gets submitted** — healthy competition, best model wins
- If time permits, explore **HIL-SERL fine-tuning** on the best-performing policy

---

## 4. Data Collection Plan

### Dual-Track Collection Strategy

Data collection is split across both subteams, with datasets merged for all policy training:

```
┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐
│    TEAM ALPHA: Automated Pipeline   │    │      TEAM BETA: Teleoperation       │
│                                     │    │                                     │
│  • CheatCode teacher for SFP        │    │  • Human teleop for SC port         │
│  • Randomized scene configs         │    │  • Human teleop for SFP (backup)    │
│  • Filter by /scoring/insertion_    │    │  • Manual quality check per demo    │
│    event (auto success detection)   │    │  • Diverse approach strategies      │
│  • Target: 100+ SFP demos          │    │  • Target: 50+ SC, 30+ SFP demos   │
└─────────────────┬───────────────────┘    └─────────────────┬───────────────────┘
                  │                                          │
                  └────────────┬─────────────────────────────┘
                               ▼
                 ┌─────────────────────────────┐
                 │   UNIFIED TRAINING DATASET  │
                 │                             │
                 │  ~180+ demos total          │
                 │  Both SFP and SC            │
                 │  Diverse randomization      │
                 │  LeRobot HDF5 format        │
                 └──────────┬──────────────────┘
                            │
              ┌─────────────┼─────────────────┐
              ▼             ▼                 ▼
         ┌────────┐   ┌──────────────┐  ┌──────────┐
         │  ACT   │   │  Diffusion   │  │ HIL-SERL │
         │ Policy │   │   Policy     │  │ (later)  │
         └────────┘   └──────────────┘  └──────────┘
```

### SFP Demos — Automated via CheatCode (Team Alpha)

**Method:** Run CheatCode across randomized NIC card configurations. Filter by `/scoring/insertion_event` topic (binary ground-truth oracle).

**Automation loop:**
1. Launch Gazebo with randomized board pose / NIC rail translation
2. Start rosbag recording (cameras + joints + F/T + actions)
3. Run CheatCode via `InsertCable` action goal
4. Monitor `/scoring/insertion_event` — if received, keep bag; else discard
5. Reset simulation, repeat with new randomization

**Randomization ranges:**
- Board X: 0.12 – 0.20m, Y: -0.25 – 0.05m
- Board yaw: 2.8 – 3.5 rad
- NIC rail translation: -0.0215 – 0.0234m
- Cable gripper offset Z: ±0.003m (grasp variation)

### SC & Supplementary Demos — Teleoperation (Team Beta)

**Method:** Human teleoperation using the provided `aic_teleoperation` tools.

**Why manual for SC:** Our testing confirmed CheatCode's PI controller cannot handle SC port insertion due to oscillation and lack of physical self-correction.

**Additional value:** Teleoperated demos capture diverse approach strategies (different paths, speeds, corrections) that automated demos can't provide. This diversity improves policy generalization.

### Data Format & Conversion

**Primary path:** Record rosbags → convert to LeRobot HDF5 format via custom script.

> [!NOTE]
> `lerobot-record` expects a `PreTrainedPolicy` (neural network). Since CheatCode is a ROS node, we use rosbags + conversion.

### Future: Data Augmentation

Once the base dataset is collected, we may explore:
- **Color / brightness jitter** for visual robustness
- **Random cropping** to simulate camera viewpoint variation
- **Temporal augmentation** (speed variation) for trajectory diversity
- **Synthetic noise injection** on F/T readings

---

## 5. Training Plan

### Policy Architecture Comparison

| Feature | ACT | Diffusion Policy | HIL-SERL |
|---------|-----|-----------------|----------|
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
| Image resolution | 480×640 (3 cameras) | 480×640 (3 cameras) |
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

## 6. Task Breakdown

### Phase 0: Environment Setup ✅ DONE
- [x] Get Gazebo eval container running in distrobox
- [x] Run CheatCode for successful SFP insertion
- [x] Confirm `/scoring/insertion_event` topic is available
- [x] Set up git repo and push to GitHub
- [x] Verify CheatCode behavior across different port types (SFP ✅, SC ❌)
- [x] Add orientation-aware CheatCode patch for reference

---

### Phase 1: Data Collection Infrastructure (Week 1)

**Team Alpha — Automated Pipeline:**
- [ ] Write `collect_sfp_demos.py` orchestration script:
  - [ ] Randomize scene parameters (board pose, rail translations)
  - [ ] Launch Gazebo programmatically with randomized config
  - [ ] Start rosbag recording
  - [ ] Trigger CheatCode (lifecycle + action goal)
  - [ ] Monitor `/scoring/insertion_event` for success
  - [ ] Save successful bags, discard failures
  - [ ] Reset and loop
- [ ] Write rosbag → LeRobot HDF5 conversion script
- [ ] Collect first batch of **50 successful SFP demos**

**Team Beta — Teleoperation Setup:**
- [ ] Set up teleoperation environment
- [ ] Practice SC insertion manually (5–10 warmup runs)
- [ ] Document teleoperation workflow for the team
- [ ] Collect first batch of **20 SC demos**

---

### Phase 2: Full Dataset Collection (Week 2)

**Team Alpha:**
- [ ] Scale SFP collection to **100+ successful demos**
- [ ] Validate dataset: spot-check 10 demos visually
- [ ] Begin ACT training on SFP-only dataset (early results)

**Team Beta:**
- [ ] Collect **50+ SC insertion demos**
- [ ] Collect **30+ supplementary SFP demos via teleoperation** (diverse strategies)
- [ ] Convert all demos to HDF5 format

**Shared:**
- [ ] Merge all demos into **unified dataset** (~180+ demos)
- [ ] Dataset quality audit: remove corrupted / poor-quality demos

---

### Phase 3: Parallel Policy Training (Week 3)

**Team Alpha — ACT:**
- [ ] Train ACT on combined SFP + SC dataset with task conditioning
- [ ] Evaluate on all 3 trial types
- [ ] Record baseline scores
- [ ] Identify failure modes

**Team Beta — Diffusion Policy:**
- [ ] Configure Diffusion Policy training in LeRobot
- [ ] Train on same combined dataset
- [ ] Evaluate on all 3 trial types
- [ ] Compare scores vs ACT

**Shared:**
- [ ] **Policy comparison meeting**: which performs better on which trials?
- [ ] Decide whether both tracks continue or one is prioritized

---

### Phase 4: Iteration & Hardening (Weeks 4–5)

- [ ] Collect more targeted demos for failure cases
- [ ] Hyperparameter tuning (chunk size, learning rate, augmentation, denoising steps)
- [ ] Increase domain randomization coverage
- [ ] Stress test across 20+ randomized configs
- [ ] **Optional:** HIL-SERL fine-tuning on best-performing policy
- [ ] **Target: ≥75 pts on Trials 1 & 2, ≥50 pts on Trial 3**

---

### Phase 5: Docker & Submission (Week 5)

- [ ] Package best policy into Docker following `submission.md`
- [ ] Verify ROS 2 Lifecycle compliance:
  - `unconfigured` → `configured` → `active` → `deactivate` → `cleanup` → `shutdown`
- [ ] Test full local evaluation mimicking cloud pipeline
- [ ] Confirm policy handles all 3 trials end-to-end
- [ ] Submit to evaluation portal
- [ ] Iterate based on leaderboard results

---

### Phase 6: Polish & Buffer (Week 6)

- [ ] Optimize Tier 2 scores (smoothness, speed, efficiency)
- [ ] Fine-tune force control to avoid penalties
- [ ] Second submission attempt if time permits
- [ ] Final submission before May 15 deadline

---

## 7. Why End-to-End Wins for This Task

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

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| CheatCode success rate too low | Slow SFP data collection | Filter by `/scoring/insertion_event`; ~60-75% success rate = 100 demos from ~150 runs |
| SC teleoperation too difficult | Can't collect SC demos | Focus SFP first (200 pts from Trials 1 & 2); train SC later |
| ACT doesn't generalize | Low eval scores | Diffusion Policy as backup; different inductive bias may work better |
| Diffusion Policy too slow | Can't run at real-time | DDIM with fewer steps, or OneDP distillation |
| Grasp variation breaks insertion | Failed trials at eval | Add ±2mm gripper offset perturbations to demos |
| Submission Docker doesn't build | Can't submit | Start Docker packaging in Phase 5, not last minute |
| Proximity-only scores too low | Fail to qualify | Even 25 pts/trial × 3 = 75 pts; enough to qualify |
| One subteam's policy clearly better | Wasted effort | Not wasted — second policy provides fallback and comparison data |

---

## 9. Timeline Summary

```
Week 1 (Apr 7–13):   Phase 1 — Data infrastructure + first demos (both teams)
Week 2 (Apr 14–20):  Phase 2 — Full dataset + early ACT training
Week 3 (Apr 21–27):  Phase 3 — Parallel training (ACT vs Diffusion Policy)
Week 4 (Apr 28–May 4): Phase 4a — Iterate, collect more demos, compare policies
Week 5 (May 5–11):   Phase 4b + 5 — Docker + local testing + first submission
Week 6 (May 12–15):  Phase 6 — Polish + final submission
```

> [!CAUTION]
> **Do NOT leave submission packaging to the last week.** Docker/container bugs are the #1 reason teams fail competitions.

---

## 10. Success Criteria

| Milestone | Target | When | Owner |
|-----------|--------|------|-------|
| SFP auto-collection working | 100+ successful bags | End of Week 2 | Team Alpha |
| SC demos collected | 50+ demos | End of Week 2 | Team Beta |
| ACT model evaluated | Score > 0 on all 3 trials | End of Week 3 | Team Alpha |
| Diffusion Policy evaluated | Score > 0 on all 3 trials | End of Week 3 | Team Beta |
| Best policy selected | ≥75 pts on Trials 1 & 2 | End of Week 4 | Both |
| Submission working | Docker passes local eval | End of Week 5 | Andrews |
| **Qualification** | **≥ 150 pts total** | **May 15** | **All** |
