# Policy Architecture Analysis & Execution Plan

> **Team Apex Autonomy** · AI for Industry Challenge
> **Date:** April 11, 2026 · **Deadline:** May 15, 2026 (34 days remaining)

---

## 1. The Keyboard Teleop Problem (Data Quality)

### The Core Issue

All demo data is collected via **keyboard teleoperation**, which produces **discrete, step-function velocity profiles** — not smooth, continuous trajectories:

```
Keyboard input:     ___┌──┐___┌──┐___┌──────┐___
                    0  0.02  0  0.02  0   0.02   0

Ideal trajectory:   ___╱──╲___╱──╲___╱──────╲___
                    smooth acceleration/deceleration
```

The keyboard produces only ~13 unique action patterns (6 axes × 2 directions + zero), with the majority being zeros (key released). This creates **infinite jerk** at every key press/release boundary.

### Why This Matters

Both ACT and Diffusion Policy are **imitation learners** — their entire objective is to reproduce the statistical distribution of the training data. If the data is jerky, the policy will be jerky.

- **AIC scores trajectory smoothness** (0–6 pts per trial, measured via Savitzky-Golay filter)
- **Jerk causes force spikes** which can trigger the >20N penalty (0 to −12 pts)
- **Jerky insertion attempts** increase collision risk (0 to −24 pts)

### How Each Architecture Handles Jerky Data

| Architecture | How it handles step-function velocities | Result |
|---|---|---|
| **ACT** | Temporal ensembling blends overlapping chunks → some smoothing. CVAE captures style variation. But underlying model still learned jerkiness. | Attenuated jitter — neither clean 0 nor clean 0.02 |
| **Diffusion Policy** | Iterative denoising converges toward training distribution. If distribution IS jerky, denoised output IS jerky. | Learns to produce the most statistically likely stutter |
| **VQ-BeT** | Codebook absorbs keyboard patterns into clean motion primitives. Classification picks from clean dictionary. | Clean discrete transitions — but quantization risk at insertion |

---

## 2. Post-Processing: Smoothing Keyboard Demos

**Apply offline, after recording but before training.** This cleans the data without changing the collection workflow.

### Recommended: Savitzky-Golay Filter

Chosen because the AIC scoring system itself uses Savitzky-Golay to measure trajectory smoothness — aligning training data with the scoring function.

```python
from scipy.signal import savgol_filter
import numpy as np

def smooth_actions(actions: np.ndarray, window: int = 9, polyorder: int = 3) -> np.ndarray:
    """
    Smooth recorded velocity actions per axis.
    
    actions: shape (T, 6) — 6D velocity commands over T timesteps
    window: must be odd. 9 at 4Hz ≈ 2.25s sliding window.
    polyorder: 3 = cubic, preserves intentional direction changes.
    """
    smoothed = np.zeros_like(actions)
    for axis in range(6):
        smoothed[:, axis] = savgol_filter(actions[:, axis], window, polyorder)
    return smoothed
```

### Alternative: Exponential Moving Average

```python
def ema_smooth(actions: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    smoothed = np.zeros_like(actions)
    smoothed[0] = actions[0]
    for t in range(1, len(actions)):
        smoothed[t] = alpha * actions[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed
```

### Alternative: Cubic Spline Re-interpolation

```python
from scipy.interpolate import CubicSpline

def spline_smooth(actions: np.ndarray, subsample: int = 3) -> np.ndarray:
    T = len(actions)
    t_full = np.arange(T)
    t_knots = t_full[::subsample]
    smoothed = np.zeros_like(actions)
    for axis in range(6):
        cs = CubicSpline(t_knots, actions[t_knots, axis])
        smoothed[:, axis] = cs(t_full)
    return smoothed
```

> [!WARNING]
> **Don't over-smooth.** Corrections and recovery behaviors are the most valuable training data. Use window=7 or 9 (Savitzky-Golay) as the starting point — removes keyboard discretization artifacts while preserving intentional direction changes.

### Where to Apply

Dataset location: `~/.cache/huggingface/lerobot/atang/aic_sfp_demos/data/train-*.parquet`

1. Load the parquet file
2. Group by episode
3. Apply smoother to action columns per episode
4. Save as new dataset (or overwrite)

---

## 3. Policy Architecture Comparison

### Tier 1: Drop-In (Already in LeRobot)

#### ACT — Action Chunking with Transformers (Current Plan, Team Alpha)

- **How:** Predicts a chunk of K=100 future actions in one forward pass via CVAE + Transformer
- **Strengths:** Proven baseline exists from organizers, temporally consistent, fast inference (~50Hz)
- **Weakness:** CVAE averages keyboard jitter rather than eliminating it
- **Risk:** Bad CVAE z-sample → commits to wrong trajectory for entire chunk
- **LeRobot:** `--policy.type=act`

#### Diffusion Policy (Current Plan, Team Beta)

- **How:** Iteratively denoises random noise into action sequences (10-20 steps)
- **Strengths:** Naturally multimodal, excellent motion quality, 40-60% better than prior IL on benchmarks
- **Weakness:** Slower inference (5-10Hz), learns to reproduce keyboard stutter if that's the data
- **Risk:** Inference latency — may need DDIM for real-time
- **LeRobot:** `--policy.type=diffusion`

#### VQ-BeT — Vector Quantized Behavior Transformer (New Addition)

- **How:** Learns a codebook of ~512 motion primitives via VQ-VAE, then classifies which primitive to use + small continuous residual
- **Strengths:** Naturally handles discrete keyboard data, classification avoids continuous averaging, fast inference
- **Weakness:** Codebook quantization may lack vocabulary for sub-millimeter insertion corrections
- **Risk:** If codebook is too coarse for insertion phase, residual head can't compensate
- **LeRobot:** `--policy.type=vqbet`

**VQ-BeT Architecture:**

```
Stage 1 — Codebook Learning (offline, action data only):
  All demo action chunks → VQ-VAE → 512 codebook entries (motion primitives)

Stage 2 — Policy Training:
  Observations (cameras + state) → Transformer → Classification Head (which code?)
                                               → Residual Head (small offset)
  
  Final action = codebook[selected_code] + residual_offset
  Loss = CrossEntropy(code prediction) + L1(residual prediction)
```

**Why VQ-BeT fits keyboard data:** Prediction is a classification problem (pick 1 of 512 tokens), not continuous regression. The codebook entries ARE clean patterns. Sharp transitions between key press/release become transitions between discrete states — what transformers excel at.

**Quantization risk mitigation:**
- Oversample insertion-phase action chunks 8x during VQ-VAE training
- Increase codebook size to 1024-2048 for finer granularity
- Temporal loss weighting: 5x weight on insertion-phase timesteps

---

### Tier 2: Higher Ceiling, More Setup

#### π₀ (Pi-Zero) / Flow Matching

- **How:** Uses flow matching (straight-line noise→action transport) instead of iterative diffusion. Pre-trained on massive cross-embodiment data.
- **Strengths:** Smoothest action generation (~50Hz), one/two-step inference, pre-trained foundation
- **Weakness:** ~3B parameters, GPU-hungry, fine-tuning gap uncertain for AIC
- **In LeRobot:** Yes (VLA policy type)
- **Verdict:** High ceiling but setup/compute cost may not fit timeline

#### SmolVLA — Small Vision-Language-Action Model

- **How:** Lightweight (~450M param) VLA. Takes images + language task description → actions.
- **Strengths:** Natural language task conditioning ("insert SFP" vs "insert SC"), fine-tunable on single GPU
- **Weakness:** Relatively new, less battle-tested
- **Verdict:** Interesting for task conditioning but probably not a priority with only 2 plug types

#### Consistency Policy

- **How:** Distills a trained Diffusion Policy into a single-step generator
- **Strengths:** 10x faster inference than Diffusion with competitive success rates
- **Weakness:** Two-stage process (train Diffusion first, then distill). Your 4Hz control loop is already slower than both architectures.
- **Verdict:** Nice optimization but inference speed isn't the bottleneck

---

### Tier 3: Not Recommended

#### SARNN — Spatial Attention Recurrent Neural Network

- **How:** RNN (LSTM) with spatial attention heatmaps over camera images + image reconstruction
- **Strengths:** Interpretable (visualize where model looks), good temporal memory, lightweight
- **Weakness:** Deterministic regression — averages multiple valid approaches (worst multimodality handling of all candidates). Single-step prediction — no action chunking. 2022-era, consistently outperformed by ACT/Diffusion/VQ-BeT.
- **Not in LeRobot** — requires separate RoboManipBaselines framework
- **Verdict:** ❌ Skip. Strictly worse than available alternatives for this task.

#### DP3 — 3D Diffusion Policy

- **How:** Replaces 2D images with 3D point clouds from depth cameras. Same diffusion backbone.
- **Strengths:** Explicit 3D spatial reasoning — ideal for sub-mm insertion. Viewpoint/appearance invariant. 10-40 demos sufficient. Best theoretical architecture for precision insertion.
- **Weakness:** Requires depth cameras (not in default AIC camera config). Not in LeRobot — custom pipeline. Data format conversion needed.
- **Blocking question:** Are depth topics available in the evaluation environment?
- **Verdict:** ⭐ Highest theoretical ceiling for insertion, but **practical barriers are too high** for 34-day timeline unless depth data is already available.

```bash
# Check if depth is available before investing any time:
ros2 topic list | grep -i depth
```

---

## 4. The Hybrid Architecture (Future Optimization)

> [!NOTE]
> **This is a Phase 3+ optimization.** Do NOT implement before having a baseline score from a vanilla policy.

### Concept: VQ-BeT for Approach, ACT for Insertion

```python
class HybridPolicy:
    def __init__(self):
        self.vqbet_policy = load_vqbet("outputs/train/vqbet_v1/best.pt")
        self.act_policy = load_act("outputs/train/act_v1/best.pt")
        self.force_threshold = 3.0  # Newtons
        self.phase = "approach"
    
    def predict(self, observation):
        fz = abs(observation["force_torque"][2])  # Z-axis force
        
        if self.phase == "approach" and fz > self.force_threshold:
            self.phase = "insertion"
        
        if self.phase == "approach":
            return self.vqbet_policy.predict(observation)
        else:
            return self.act_policy.predict(observation)
```

**Rationale:**
- **VQ-BeT (approach):** Clean discrete motion from keyboard data. Classification handles multiple approach paths. No jitter.
- **ACT (insertion):** Continuous regression for sub-millimeter, force-responsive adjustments. Temporal ensembling for smoothness.
- **F/T sensor as phase detector:** Z-force >3N = contact made → switch.

**When to implement:** Only after (1) both VQ-BeT and ACT are independently trained and scored, (2) analysis shows VQ-BeT wins on approach but ACT wins on insertion, and (3) there is >1 week remaining before deadline.

---

## 5. Anti-Memorization Protocol (All Architectures)

These apply to ALL policy training, not just VQ-BeT.

### Domain Randomization (Data Collection)

Vary board/rail positions on a continuous range every session:

```bash
nic_card_mount_0_translation:=$(python3 -c "import random; print(f'{random.uniform(-0.02, 0.05):.4f}')")
sc_port_0_translation:=$(python3 -c "import random; print(f'{random.uniform(-0.06, 0.02):.4f}')")
yaw:=$(python3 -c "import random; print(f'{random.uniform(2.8, 3.2):.4f}')")
```

### Image Augmentation (Training)

```python
import torchvision.transforms as T

train_transforms = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])
```

**Do NOT use random crops** — spatial layout must match the robot's actual camera view.

### Temporal Loss Weighting (Training)

Weight the loss 5x higher on the last 30% of each episode (insertion phase):

```python
def compute_temporal_weight(timestep, episode_length):
    progress = timestep / episode_length
    return 5.0 if progress > 0.7 else 1.0
```

---

## 6. Execution Plan (Grounded in Reality)

> [!CAUTION]
> **As of April 11, 2026:**
> - Zero demos collected
> - Zero policies trained
> - Zero submissions made
> - Submission #1 is due April 20 (9 days)
>
> **The most impactful thing right now is NOT architecture selection — it's collecting demos and training a baseline.**

### Phase 1: Data + Baseline (Apr 11 → Apr 20) ⚡ CRITICAL

**Everyone records demos. No exceptions. No architecture research.**

| Day | Action | Owner |
|:---:|--------|-------|
| **Apr 11-13** | Record 50+ SFP demos via `lerobot-record` | All 5 members |
| **Apr 11-13** | Record 50+ SC demos via `lerobot-record` | All 5 members |
| **Apr 13** | Apply Savitzky-Golay smoothing to action data | 1 person |
| **Apr 14-15** | Train vanilla ACT on smoothed dataset | Team Alpha |
| **Apr 14-15** | Train vanilla Diffusion Policy on same dataset | Team Beta |
| **Apr 16-17** | Evaluate both in Gazebo (`ground_truth:=false`) | Both teams |
| **Apr 18** | Package best-performing policy into Docker | Both teams |
| **Apr 19-20** | 🚀 **SUBMISSION #1 — Baseline** | All |

**Training commands:**

```bash
# Team Alpha — ACT
pixi run lerobot-train \
  --dataset.repo_id=atang/aic_sfp_demos \
  --policy.type=act \
  --output_dir=outputs/train/act_v1 \
  --policy.device=cuda

# Team Beta — Diffusion Policy
pixi run lerobot-train \
  --dataset.repo_id=atang/aic_sfp_demos \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_v1 \
  --policy.device=cuda
```

### Phase 2: Experiment + Iterate (Apr 20 → May 4)

**Only after Submission #1 scores are in.**

| Action | When | Condition |
|--------|------|-----------|
| Train VQ-BeT on same dataset (`--policy.type=vqbet`) | Apr 21-23 | Always — zero extra effort |
| Compare ACT vs Diffusion vs VQ-BeT scores | Apr 23 | Always |
| Collect targeted demos for failure cases | Apr 24-27 | Based on score analysis |
| Apply image augmentation to training | Apr 24 | Always |
| Increase domain randomization coverage | Apr 24+ | Always |
| 🚀 **SUBMISSION #2** | Apr 27 | |
| Explore codebook engineering for VQ-BeT | Apr 28+ | Only if VQ-BeT shows promise |
| 🚀 **SUBMISSION #3** | May 4 | |

### Phase 3: Polish + Final (May 5 → May 14)

| Action | When | Condition |
|--------|------|-----------|
| Explore hybrid VQ-BeT + ACT architecture | May 5-8 | Only if data shows VQ-BeT wins approach + ACT wins insertion |
| RL fine-tuning (HIL-SERL) | May 5-10 | Only if best policy >80% success |
| Final hyperparameter sweep | May 8-10 | Small adjustments only |
| Verify Docker + Lifecycle compliance | May 10-12 | Non-negotiable |
| 🚀 **SUBMISSION #4 — FINAL** | May 12-14 | |

---

## 7. Decision Framework

### Which Policy to Submit?

```
After training all three (ACT, Diffusion, VQ-BeT) on the same dataset:

1. Compare total scores across all 3 trial types
2. Identify per-trial performance:
   - Which policy wins Trial 1 (SFP, slot 0)?
   - Which policy wins Trial 2 (SFP, slot 1)?
   - Which policy wins Trial 3 (SC)?
3. If one policy dominates all trials → submit it
4. If different policies win different trials → consider hybrid approach
5. If scores are similar → submit the one with lowest variance (most consistent)
```

### When to Invest in Architecture Modifications?

| Baseline Score | Action |
|:-:|---|
| **< 30 pts** | Fix data. More demos. Better demo quality. Architecture doesn't matter yet. |
| **30-60 pts** | Policy approaches port but fails insertion. Try VQ-BeT, tune augmentation. |
| **60-80 pts** | Policy inserts sometimes. Targeted demos for failure cases. Codebook engineering. |
| **> 80 pts** | Consider hybrid architecture or RL fine-tuning for the final push. |

---

## 8. Key Takeaways

1. **The keyboard teleop jerkiness problem is real** — ACT and Diffusion will reproduce step-function velocities from the training data. Post-process demos with Savitzky-Golay before training.

2. **VQ-BeT is the best architectural fit for keyboard data** — classification over a clean codebook naturally handles discrete inputs. But it carries quantization risk for sub-mm insertion.

3. **SARNN is not worth pursuing** — older architecture, worse multimodality handling, not in LeRobot.

4. **DP3 has the highest ceiling for insertion** — but requires depth cameras and a custom pipeline. Only viable if depth data is already available.

5. **The hybrid VQ-BeT + ACT approach is elegant** — but it's a Phase 3 optimization, not a Phase 1 priority.

6. **Data quality beats architecture** — 200 excellent demos with a vanilla ACT will outperform 50 mediocre demos with a bespoke hybrid. Collect first, optimize second.

7. **Submit early, iterate on scores** — a baseline score from the evaluation system is worth more than all the architectural analysis combined.
