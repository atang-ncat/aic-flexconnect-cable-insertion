# 🏆 AIC Competition — Development Strategy

> **Team Apex Autonomy** · AI for Industry Challenge (Intrinsic / Google DeepMind / NVIDIA)
>
> Cable connector insertion · UR5e + Robotiq Hand-E · Qualification deadline: **May 15, 2026**

---

## 1. Competition Overview

The AIC is a **precision cable insertion** task. A UR5e robot equipped with a Robotiq Hand-E gripper, Axia80 F/T sensor, and 3 wrist cameras must insert fiber optic connectors into target ports on a randomized task board.

> [!IMPORTANT]
> **The robot starts with the plug already in-hand, within centimeters of the target port.**
> There is **no grasping problem** to solve. The entire challenge is:
> 1. Visually identify the target port using camera images
> 2. Align the connector with sub-millimeter precision
> 3. Execute a smooth, gentle insertion

### Qualification Trials

| Trial | Plug Type | Target Port | Randomization |
|:-----:|-----------|-------------|---------------|
| 1 | SFP Module | NIC Card SFP Port | Board pose, NIC rail position & yaw |
| 2 | SFP Module | NIC Card SFP Port | Different board pose, different NIC position |
| 3 | SC Plug | SC Port | Board pose, SC rail position |

> [!IMPORTANT]
> A **single submitted policy** must handle **both** plug types (SFP and SC) across all three trials. The `Task` message specifies which port to target — the policy must condition on this.

### Scoring Breakdown (100 pts max per trial)

| Category | Points | Notes |
|----------|:------:|-------|
| Model validity | 1 | Loads and responds to action requests |
| **Insertion success** | **75** | Correct port · Wrong port = **−12** |
| Partial proximity | 0–50 | Closer to port = more points (generous fallback) |
| Task duration | 0–12 | ≤ 5 s → max · ≥ 60 s → 0 |
| Trajectory smoothness | 0–6 | Low jerk (Savitzky-Golay filtered) |
| Trajectory efficiency | 0–6 | Short path length |
| Force penalty | 0 to −12 | > 20 N for > 1 s |
| Collision penalty | 0 to −24 | Any contact with off-limit areas |

**75% of the score is insertion success. Everything else is secondary.**

### Evaluation Environment

> [!WARNING]
> Evaluation runs in **Gazebo** with `ground_truth:=false`. Your policy receives **only**:
>
> | Available ✅ | NOT Available ❌ |
> |---|---|
> | 3 camera images (left / center / right) | Port / plug TF frames |
> | Joint states | Object poses |
> | Force / torque sensor (tared at startup) | Gazebo internal state |
> | Controller state (TCP pose, velocity, error) | Scoring topics |
> | Robot TF frames (base_link → gripper/tcp) | Ground truth |
>
> **Your policy must locate the port using only cameras + proprioception.**

---

## 2. Strategic Approach — Imitation Learning (ACT)

### Why ACT Wins

| Factor | ACT (Imitation Learning) | Pure RL |
|--------|:------------------------:|:-------:|
| Demo availability | ♾️ unlimited via CheatCode | N/A |
| Cable physics required | No (learns from Gazebo directly) | Yes — Isaac Lab has none |
| Sim-to-sim transfer risk | None (train & eval both in Gazebo) | High (Isaac Sim → Gazebo) |
| Sample efficiency | ~100 demos sufficient | Millions of steps |
| Time to first result | ~2 weeks | ~4–5 weeks |
| Proven on this task | ✅ organizer baseline exists | ❌ reward engineering needed |

**The organizers designed this pipeline intentionally** — CheatCode is the "teacher," ACT is the "student," and the LeRobot toolchain connects them.

### Is Using CheatCode for Demos Legal?

**Yes — 100% legal.** From `challenge_rules.md`, section 2c:

> *"During training, participants may use all internal state information, including ground truth data available over the `/tf` topic."*

This explicitly permits:
- ✅ Running CheatCode with `ground_truth:=true` during training
- ✅ Recording demonstrations via `lerobot-record`
- ✅ Training a learned policy (ACT) on those demos
- ✅ Submitting the learned policy (which uses only cameras / joints / F-T)

Ground truth is simply unavailable during evaluation — the organizers control the `ground_truth` flag.

### Can You Submit CheatCode Directly?

**No.** CheatCode depends on TF frames (`task_board/{module}/{port}_link`) that are **only published when `ground_truth:=true`**. During evaluation, these frames don't exist and CheatCode crashes at `_wait_for_tf()`.

---

## 3. Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRAINING  (ground_truth:=true)                    │
│                                                                      │
│  1. Gazebo eval with ground_truth:=true                              │
│  2. CheatCode auto-inserts using TF frames (perfect demos)           │
│  3. lerobot-record captures: cameras + joints + F/T + actions        │
│  4. Vary board pose / rail position across episodes                  │
│  5. Vary grasp pose slightly (~2mm, ~0.04 rad) to match eval noise   │
│  6. Collect demos for BOTH SFP and SC plug types                     │
│  7. lerobot-train produces ACT policy                                │
│                                                                      │
│  INPUT:  Ground truth TF frames (legal during training)              │
│  OUTPUT: ACT model mapping [cameras, joints, F/T, task] → actions    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   EVALUATION  (ground_truth:=false)                  │
│                                                                      │
│  ACT policy receives: 3 camera images + joints + F/T + Task msg      │
│  ACT policy outputs: Twist velocity commands to aic_controller       │
│  No TF frames needed — the policy learned visual servoing from demos │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Robustness Considerations

> [!IMPORTANT]
> **Grasp pose variation**: The qualification docs confirm ~2 mm / ~0.04 rad deviations in the plug-to-TCP grasp across trials. Demos should include slight grasp variations to ensure the policy doesn't overfit to a single grasp.

> [!IMPORTANT]
> **Both plug types, one policy**: The same submitted policy handles SFP (Trials 1 & 2) and SC (Trial 3). The demo dataset must include both, and the policy must condition on the `Task` message to know which port type to target.

---

## 4. Available Tooling

| Tool | Status | Role |
|------|:------:|------|
| Gazebo eval container | ✅ | Primary training & evaluation environment |
| CheatCode baseline | ✅ | Automated demo generation with ground truth TFs |
| LeRobot (`lerobot-record` / `lerobot-train`) | ✅ | Data collection + ACT training pipeline |
| ACT baseline on HuggingFace (`grkw/aic_act_policy`) | ✅ | Pre-trained reference model for pipeline validation |
| Isaac Lab RL (`rsl_rl/train.py`) | ⏸️ | Deprioritized — no cable physics, sim-transfer risk |
| MuJoCo integration | ⏸️ | Only if specific failure modes need diverse physics |

---

## 5. Execution Timeline (7 weeks → May 15)

### Week 1 — Pipeline Validation ⚡ CRITICAL

> [!CAUTION]
> **This week gates everything.** If the eval container doesn't work, nothing else matters.

| Step | Action | Success Criteria |
|:----:|--------|------------------|
| 1 | Get Gazebo eval container running cleanly | `distrobox enter -r aic_eval -- /entrypoint.sh` starts without errors |
| 2 | Run CheatCode for one successful insertion | Score > 0 on a single trial |
| 3 | Run the **pre-trained ACT baseline** (`grkw/aic_act_policy`) in Gazebo | Baseline scores > 0, confirming the full submission pipeline works |

```bash
# Start Gazebo eval
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Run pre-trained ACT baseline
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT
```

### Week 2 — Demo Collection

| Step | Action | Target |
|:----:|--------|--------|
| 1 | Automate CheatCode demo collection with `lerobot-record` | 50–100 SFP demos |
| 2 | Collect SC plug demos | 50–100 SC demos |
| 3 | Vary board pose, rail position, and grasp offset across episodes | Domain randomization |

```bash
# Run CheatCode as automated demonstrator
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true

pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode

# Record with lerobot
cd ~/ws_aic/src/aic
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --dataset.repo_id=apex_autonomy/aic_demos \
  --dataset.single_task="insert cable" \
  --dataset.push_to_hub=false \
  --display_data=true
```

### Week 3 — Train & Baseline

| Step | Action | Target |
|:----:|--------|--------|
| 1 | Train ACT on collected demos | First custom model |
| 2 | Evaluate in Gazebo across all 3 trial types | Scoring baseline |
| 3 | Identify failure modes (which trials fail, why) | Targeted improvement plan |

```bash
pixi run lerobot-train \
  --dataset.repo_id=apex_autonomy/aic_demos \
  --policy.type=act \
  --output_dir=outputs/train/act_aic \
  --policy.device=cuda
```

### Week 4–5 — Iterate & Harden

- Collect more targeted demos for failure cases
- Hyperparameter tuning (chunk size, learning rate, augmentation)
- Increase domain randomization (more board poses, grasp variations)
- Test across many randomized trials to measure consistency

### Week 6 — Docker & Submission ⚡ CRITICAL

> [!CAUTION]
> **Do NOT leave submission packaging to Week 7.** Submission bugs are the #1 reason teams fail competitions. Build and test the Docker image early.

- Package policy into Docker following [submission.md](../../src/aic/docs/submission.md)
- Run full local evaluation mimicking the cloud pipeline
- Verify ROS 2 Lifecycle compliance (unconfigured → configured → active → deactivate → cleanup → shutdown)
- Confirm the policy handles all three trials

### Week 7 — Buffer & Polish

- Fix any remaining submission issues
- Final tuning pass
- **Submit early, iterate if time permits**

---

## 6. What to Deprioritize

| Item | Reason |
|------|--------|
| **Pure RL in Isaac Lab** | No cable physics, rewards don't match the insertion task, sim-transfer risk to Gazebo |
| **Residual RL fine-tuning** | Only worthwhile if already scoring 85+ consistently — the 18 Tier 2 bonus points aren't worth the engineering cost at lower scores |
| **MuJoCo cross-training** | Only if ACT shows specific failure modes that diverse physics would help; Gazebo-only training is fine since eval IS in Gazebo |
| **Tier 2 optimization** (smoothness, speed) | Focus on insertion success first — partial proximity alone scores up to 50 pts |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Eval container dependency issues | Dedicate Week 1 entirely to this; recent Docker fixes should persist |
| ACT doesn't generalize to SC plugs | Ensure 50+ SC demos in dataset; condition policy on Task message |
| Grasp pose variation causes failures | Include grasp perturbations in demo collection |
| Submission Docker doesn't work | Test submission pipeline in Week 6, not Week 7 |
| Low insertion success rate | Even proximity-only scores 25 pts/trial — a policy that gets close still qualifies |

---

## 8. Summary

```
STRATEGY:  Imitation Learning (ACT) trained on CheatCode demos
WHY:       Proven pipeline, organizer-designed, no sim-transfer gap
PRIORITY:  Insertion success (75 pts) >>> everything else
TIMELINE:  Pipeline first (Wk 1) → Demos (Wk 2) → Train (Wk 3) → Iterate (Wk 4-5) → Submit (Wk 6) → Buffer (Wk 7)
SKIP:      Pure RL, Isaac Lab, MuJoCo (unless needed), Tier 2 optimization
```
