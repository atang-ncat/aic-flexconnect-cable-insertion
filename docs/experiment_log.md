# ACT Policy Experiment Log — SFP Cable Insertion

> **Last updated:** 2026-04-27  
> **Dataset:** `teleop-dataset-ft-v1` (115 episodes, EMA-smoothed keyboard teleop)  
> **Task:** SFP cable insertion into NIC card ports (AIC FlexConnect challenge)  
> **Architecture:** ACT (Action Chunking with Transformers) via LeRobot  
> **Hardware:** 4× NVIDIA RTX 6000 Ada (48 GB each)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scoring System](#scoring-system)
3. [Complete Results Table](#complete-results-table)
4. [Evolution of Techniques](#evolution-of-techniques)
   - [Phase 1: Architecture & Data Pipeline (v3–v9)](#phase-1-architecture--data-pipeline-v3v9)
   - [Phase 2: Regularization & Overfitting (v10–v11)](#phase-2-regularization--overfitting-v10v11)
   - [Phase 3: Action Weighting Breakthrough (v12–v13)](#phase-3-action-weighting-breakthrough-v12v13)
   - [Phase 4: Exhausting Hyperparameters (v13b–v14b)](#phase-4-exhausting-hyperparameters-v13bv14b)
5. [Deploy-Time Experiments](#deploy-time-experiments)
6. [Key Findings](#key-findings)
7. [The Data Ceiling](#the-data-ceiling)
8. [Next Steps: Breaking Through](#next-steps-breaking-through)

---

## Executive Summary

After 15+ training runs and dozens of deployment evaluations, we established that:

- **Action weighting is the only training-time lever that moved deployment scores.** Going from ~10× to ~20× upweighting of non-idle frames increased the score from 43 → 68 (a 58% improvement).
- **All other hyperparameter changes — KL weight, VAE removal, image normalization, chunk size, deploy-time gain — produced no improvement** beyond the noise band established by v13.
- **The model consistently converges to a final plug-port distance of ~0.06m on trial 2**, regardless of any change we make. This 6cm gap represents a **data ceiling** — the 115-episode keyboard teleop dataset does not contain enough precision insertion demonstrations to teach the model the final approach phase.

**Current best: v13 at 68.02 / 300 (22.7%).** A perfect score requires three successful insertions.

---

## Scoring System

**Maximum score per trial: 100 points. Three trials = 300 points max.**

Each evaluation runs 3 trials:
- **Trial 1 & 2:** SFP plug → SFP port (two different NIC card positions)
- **Trial 3:** SC plug → SC port (different cable type, never in training data)

### Tier 1: Model Validity (0–1 point)

Sanity check that the submission loads and runs. Pass = 1, Fail = 0.

### Tier 2: Performance & Convergence (−36 to +24 points)

| Category | Range | Details |
|----------|-------|---------|
| Trajectory smoothness | 0–6 | Inversely proportional to avg jerk (0 m/s³ = 6 pts, ≥50 m/s³ = 0 pts) |
| Task duration | 0–12 | Inversely proportional to time (≤5s = 12 pts, ≥60s = 0 pts) |
| Trajectory efficiency | 0–6 | Inversely proportional to path length vs initial distance |
| Insertion force penalty | 0 to −12 | Penalty if force >20N for >1s |
| Off-limit contact penalty | 0 to −24 | Penalty for collisions with enclosure/task board |

> **Note:** Smoothness, duration, and efficiency are only awarded if the plug is within the max bounding radius of the port (tier 3 > 0). Otherwise they score 0.

### Tier 3: Task Success (−12 to 75 points)

| Outcome | Score |
|---------|-------|
| Correct port insertion | **75** |
| Wrong port insertion | −12 |
| Partial insertion (inside port bounding box) | 38–50 (proportional to depth) |
| Proximity (near port but not inserted) | 0–25 (inversely proportional to distance) |

The proximity score uses the max acceptable distance = half the initial plug-port distance. At the port entrance = 25 pts. Beyond max distance = 0 pts.

---

## Complete Results Table

| Run | Key Change | val/l1 | SFP Score | Trial 1 dist | Trial 2 dist | Verdict |
|-----|-----------|--------|-----------|--------------|--------------|---------|
| v3–v5 | Initial pipeline, data fixes | — | ~3 | idle | idle | ❌ Mean collapse |
| v6 | 3 cameras, 201 episodes | — | ~3 | idle | idle | ❌ Mean collapse |
| v10 | F/T sensor, 27-d state, trainable backbone | 0.255 | ~35 | 0.13m | 0.14m | ✓ First movement |
| v11 | Frozen backbone, cosine LR, regularization | 0.260 | ~38 | 0.12m | 0.12m | ✓ Slight improvement |
| v11c | Frozen + augmentation + weight decay | 0.244 | ~3 | idle | idle | ❌ Best val loss but collapsed at deploy |
| **v12** | **10× action weighting** | **0.244** | **43.52** | **0.09m** | **0.11m** | **✓ Weighting works** |
| **v13** | **20× action weighting + head trim** | **0.257** | **68.02** | **0.09m** | **0.06m** | **🏆 BEST** |
| v13b | Weak weight + low KL (1.0) | 0.243 | 3.0 | idle | idle | ❌ Collapsed |
| v13c | Weak weight + no VAE | — | 3.0 | idle | idle | ❌ Collapsed |
| v14 | 40× action weighting | 0.257 | — | — | — | Same val loss as v13 |
| v14b | 20× weight + low KL (1.0) | 0.256 | 66.13 | 0.09m | 0.06m | ≈ v13 (noise band) |

---

## Evolution of Techniques

### Phase 1: Architecture & Data Pipeline (v3–v9)

**Goal:** Get the robot to move at all.

| Technique | What We Tried | Result |
|-----------|--------------|--------|
| Single camera → 3 cameras | Added left, center, right camera inputs | No effect on mean collapse |
| Data smoothing (Savitzky-Golay) | Post-processed keyboard teleop to smooth jerky velocity profiles | Improved data quality but didn't fix deployment |
| State representation | Experimented with TCP pose, joint positions, quaternion representation | Settled on 27-d: TCP pose (7) + joints (6) + joint velocities (6) + F/T wrench (6) + gripper (2) |
| VQ-BeT | Tested discrete action tokenization as an alternative architecture | Abandoned — harder to tune, didn't outperform ACT |
| Previous action input | Fed the previous action as additional input to the policy | No measurable improvement |

**Key learning:** The mean-collapse problem (model predicts ~zero action for all inputs) dominated everything. Data pipeline correctness and architecture didn't matter until this was solved.

---

### Phase 2: Regularization & Overfitting (v10–v11)

**Goal:** Get consistent, non-zero action predictions at deployment.

| Technique | What We Tried | Result |
|-----------|--------------|--------|
| Trainable → frozen backbone | `optimizer_lr_backbone: 0.0` (ResNet18 frozen at ImageNet weights) | ✓ Prevented backbone overfitting, model started moving |
| Cosine LR schedule | Warmup 500 steps → cosine decay to 1% of peak | ✓ Stabilized training, prevented late-stage divergence |
| Dropout 0.1 | Applied to transformer encoder/decoder | Mild regularization, no dramatic effect |
| Weight decay 5e-4 | L2 penalty on all parameters | Helped prevent overfitting |
| Color jitter augmentation | brightness/contrast/saturation/hue + random erasing | Improved generalization to lighting variation |
| F/T sensor data | Added 6-d tared force/torque wrench to state vector | Provided contact feedback signal |

**Key learning:** Freezing the backbone was the critical fix. With a trainable backbone, the ResNet memorized the training images and produced degenerate features at deployment. With a frozen backbone, features generalized.

**Paradox discovered:** v11c achieved the lowest validation loss (0.244) of any run but **completely collapsed at deployment** — predicting near-zero actions. Low val loss ≠ good deployment performance. The model was fitting the idle frames perfectly (which dominate the dataset) while learning nothing about the movement frames.

---

### Phase 3: Action Weighting Breakthrough (v12–v13)

**Goal:** Fix the idle-frame dominance problem identified by v11c's paradox.

#### The Core Problem

In our 115-episode dataset, **88–98% of frames are idle** (zero or near-zero action). The standard L1 loss treats all frames equally, so the model learns to predict "don't move" — which is correct for most of the data but useless for the actual task.

#### Action Weighting

We implemented a per-sample loss weighting scheme based on action magnitude:

```
weight(action) = max(floor, min(ceil, exp(-||action|| / scale)))
```

| Parameter | v12 | v13 | Effect |
|-----------|-----|-----|--------|
| `scale` | 0.03 | 0.02 | Sharper weighting curve |
| `floor` | 0.1 | 0.05 | Idle frames get less weight |
| Effective upweighting | ~10× | ~20× | How much more the loss cares about moving frames |

**v12 result:** Score jumped from ~38 → 43.52. First time the plug got within 0.09m.

**v13 result:** Score jumped from 43.52 → **68.02**. Trial 2 reached **0.06m** — inside the bounding radius for the first time, unlocking tier_2 and tier_3 bonus points.

#### Head Trimming (v13)

Episodes start with long idle periods (simulation settling, initial positioning). We added `trim_head_fraction: 0.10` to cut the first 10% of each episode, removing ~5,600 useless zero-action frames. Combined with the existing `trim_tail_fraction: 0.05`, this removed ~15% of total frames.

#### Image Normalization Experiment (v13)

We set `VISUAL: IDENTITY` in the normalization mapping, intending to pass raw [0,1] pixels to the frozen ResNet backbone instead of the "broken" per-image-mean normalization (which produced inputs in the range [-30, +24]).

**Result:** The `IDENTITY` config was a **no-op during training** — LeRobot's normalizer module still computed and applied the stored mean/std statistics regardless. The saved normalizer checkpoint proves this: the image stats show the same std≈0.018 values as v12.

We confirmed this empirically at deployment:
- Deploy with stored stats ([-30, +24] inputs): **68.02**
- Deploy with identity (raw [0,1] inputs): **3.0**

The model was trained on [-30, +24] inputs and performs best when deployment matches. The "broken" normalization is actually the model's expected input distribution.

---

### Phase 4: Exhausting Hyperparameters (v13b–v14b)

**Goal:** Find additional levers beyond action weighting to push past 0.06m.

#### v13b: Lower KL Weight (10 → 1)

**Hypothesis:** The CVAE prior (KL divergence term) pulls the latent distribution toward a zero-mean Gaussian, which might bias the decoder toward producing zero actions. Reducing KL weight from 10 to 1 would let the CVAE commit to more decisive action distributions.

**Result:** Score = **3.0** (complete mean collapse). The model used v12-level action weighting (scale=0.03, floor=0.1), so the reduced KL couldn't compensate for weak loss balance.

**Learning:** The CVAE prior is not the cause of action timidity. It actually provides useful regularization — without strong KL, the model collapsed more easily.

#### v13c: No VAE (use_vae: false)

**Hypothesis:** Remove the variational bottleneck entirely — use a deterministic encoder instead of the CVAE.

**Result:** Score = **3.0** (same collapse). Also used v12-level action weighting.

**Learning:** Removing the VAE doesn't help. The CVAE architecture is fine; the loss balance is what matters.

#### v14: Stronger Action Weighting (40×)

**Hypothesis:** If 20× worked, maybe 40× works better. Push `scale=0.015, floor=0.025`.

**Result:** val/l1 = 0.257 (identical to v13). Early-stopped at step 5000. Not deployed because val loss showed no improvement.

**Learning:** Diminishing returns on action weighting. Going from 10× → 20× helped dramatically, but 20× → 40× added nothing. The model already learned to care about movement frames; making it care even more doesn't improve trajectory precision.

#### v14b: Strong Weighting + Low KL

**Hypothesis:** v13b tested low KL with weak weighting (failed). Maybe low KL helps *on top of* strong weighting?

**Result:** Score = **66.13** (same noise band as v13). Trial 2 still at 0.06m.

**Learning:** Low KL is neither helpful nor harmful when combined with strong weighting. The score is statistically indistinguishable from v13.

---

## Deploy-Time Experiments

### AIC_ACT_GAIN: Action Magnitude Scaling

With v13 locked to 0.06m and actions at 12–17 mm/s, we tested multiplying the unnormalized actions by a constant gain factor at deployment time.

| Gain | Trial 1 dist | Trial 2 dist | Total Score |
|------|-------------|-------------|-------------|
| 1.0 (baseline) | 0.10m | 0.06m | 61.82 |
| 1.3 | 0.11m | 0.06m | 59.43 |
| 1.5 | 0.10m | 0.06m | 63.72 |

**Result:** No improvement. Trial 2 locks to 0.06m regardless of gain. The problem is trajectory accuracy (the model aims at the wrong spot), not action magnitude (it moves fast enough to get there).

### IDENTITY Normalization Patch

Patched `RunACT.py` to skip image mean/std normalization when the config says `VISUAL: IDENTITY`.

**Result:** Score dropped from 68 → **3.0**. The model was trained with the "broken" normalization and requires it at deployment. Reverted immediately.

---

## Key Findings

### What Works ✅

1. **Action weighting (20×)** — The single most impactful change. Increased score from 43 → 68 by making the loss focus on movement frames instead of idle frames.
2. **Frozen backbone** — Essential for preventing overfitting. Trainable backbones memorize training images.
3. **Head/tail trimming** — Removes useless idle frames from episode boundaries.
4. **Cosine LR schedule** — Stabilizes training and prevents late-stage divergence.
5. **Color jitter + random erasing** — Improves robustness to visual variation.

### What Doesn't Work ❌

1. **Lower KL weight** — Doesn't improve action decisiveness. The CVAE prior is already well-calibrated.
2. **Removing the VAE** — The variational bottleneck is actually useful regularization.
3. **Stronger action weighting (>20×)** — Diminishing returns. 40× gives the same val loss as 20×.
4. **Deploy-time gain scaling** — Doesn't change where the trajectory converges, only how fast it gets there.
5. **IDENTITY image normalization** — Was a no-op during training. Applying it at deploy time destroyed the model.
6. **Longer action chunks (n_action_steps=25)** — Tested in v13b alongside other changes. Didn't help.

### Surprising Discoveries 🔍

1. **Low val loss ≠ good deployment:** v11c had the lowest val/l1 (0.244) but scored 3.0 at deployment. The model was fitting idle frames perfectly.
2. **"Broken" normalization works fine:** The per-image-mean normalization produces inputs in [-30, +24] — far outside ImageNet range. Yet the frozen ResNet's FrozenBatchNorm2d partially compensates, and the transformer decoder adapts to whatever features the backbone produces. Consistency between train and deploy matters more than "correctness."
3. **The CVAE helps, not hurts:** We initially suspected the KL term of suppressing action magnitude. Empirically, both reducing KL and removing the VAE made things worse (when action weighting was weak) or had no effect (when action weighting was strong).

---

## The Data Ceiling

### Evidence

All models trained on `teleop-dataset-ft-v1` (115 episodes) converge to:
- **val/l1 ≈ 0.256–0.257** regardless of hyperparameters
- **Trial 2 final distance ≈ 0.06m** regardless of hyperparameters or deploy-time hacks
- **Trial 1 final distance ≈ 0.09–0.10m** consistently

No training hyperparameter change moves these numbers. The model has learned everything it can from this dataset.

### Root Cause Analysis

1. **Keyboard teleop lacks precision:** The demonstrations were collected via keyboard control, which produces jerky, imprecise trajectories. Even after Savitzky-Golay smoothing, the data lacks the sub-centimeter precision needed for the final insertion phase.

2. **Idle frame dominance:** 88–98% of frames in each episode are idle. Even with 20× action weighting, the model sees relatively few examples of the critical final approach. The ~2–12% of frames that actually matter for insertion may not contain enough diversity of approach angles and positions.

3. **No successful insertions in data:** The training data contains approaches toward the port but very few (if any) actual successful insertions. The model literally has never seen what a successful insertion looks like — it can only extrapolate from approach trajectories.

4. **Single port position variance:** The training data covers a limited range of port positions. The model generalizes reasonably within that range (trial 1 and 2 use different NIC card positions) but cannot extrapolate to the SC port (trial 3 always scores 0).

### The 0.06m Wall

The model's trajectory converges to a point ~6cm from the port on trial 2. This is likely the "average endpoint" of the training demonstrations — the point where most teleop trajectories stopped or slowed down significantly. The model has learned "go toward the port and stop around here" but not "align precisely and push in."

---

## Next Steps: Breaking Through

### Priority 1: Collect Better Data

The only way to break the 0.06m ceiling is with higher-quality training data:

1. **Precision teleop demonstrations** — Use spacemouse or joystick control (if available) instead of keyboard. Focus specifically on the final 10cm approach and insertion phase.

2. **Successful insertion examples** — The model needs to see complete insertions. Even 10–20 episodes with successful insertions could teach the final approach behavior.

3. **Guided demonstrations** — Slow, deliberate insertions with consistent alignment. Quality over quantity — 20 precise insertions are worth more than 100 more keyboard demos.

4. **Port position diversity** — Collect demos at multiple NIC card positions to improve generalization across trial 1 and trial 2.

5. **SC cable demos** — Trial 3 (SC cable) scores zero because the model has never seen SC cables. Even 5–10 SC insertion demos could unlock significant points.

### Priority 2: Data Augmentation Strategies

If collecting new data is slow, these approaches may help stretch existing data:

1. **Trajectory replay with noise** — Replay recorded trajectories with small random perturbations to increase diversity.

2. **Domain randomization** — Randomize visual properties (lighting, textures) during training to improve visual generalization.

3. **Hindsight relabeling** — For episodes that nearly inserted, artificially extend the trajectory toward the port to create synthetic insertion examples.

### Priority 3: Architecture Changes (Lower Priority)

These may help but are unlikely to break the ceiling without better data:

1. **Diffusion Policy** — May model the multimodal action distribution better than ACT's CVAE. Worth testing on the same dataset as a comparison.

2. **Larger backbone (ResNet34/50)** — More visual capacity, but risky with only 115 episodes (overfitting).

3. **Force-guided refinement** — Use the F/T sensor data to implement a residual policy that adjusts the last few cm based on contact forces. This could bridge the 0.06m gap without new visual data.

---

## Appendix: Training Configuration Reference

### v13 (Current Best) — Key Parameters

```yaml
# Action weighting — THE critical parameter
action_weighting:
  enable: true
  scale: 0.02       # ~20× upweighting of movement frames
  floor: 0.05
  ceil: 1.0

# Dataset trimming
trim_tail_fraction: 0.05
trim_head_fraction: 0.10

# Architecture
policy:
  chunk_size: 50
  n_action_steps: 1
  vision_backbone: resnet18
  pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"
  dim_model: 256
  n_heads: 8
  use_vae: true
  kl_weight: 10.0
  dropout: 0.1
  optimizer_lr: 1.0e-5
  optimizer_lr_backbone: 0.0      # frozen backbone
  temporal_ensemble_coeff: 0.01

# Training
training:
  steps: 15_000
  batch_size: 32
  grad_clip_norm: 10.0
  use_amp: true
  amp_dtype: bfloat16

lr_schedule:
  enable: true
  warmup_steps: 500
  min_lr_ratio: 0.01

early_stopping:
  enable: true
  patience_steps: 3_000
```

### Deploy Configuration

```bash
# Default (v13, gain=1.0)
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunACT

# With specific checkpoint
AIC_ACT_POLICY_PATH=/path/to/checkpoints/best \
  pixi run ros2 run aic_model aic_model ...

# With action gain (experimental, doesn't help)
AIC_ACT_GAIN=1.3 pixi run ros2 run aic_model aic_model ...
```
