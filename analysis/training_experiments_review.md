# Critical Review of All AIC Training Experiments

> **Task:** SFP cable insertion via imitation learning  
> **Dataset:** 151 episodes, 119k frames @ 30 Hz, 3 cameras (left/center/right), 6-DoF Cartesian twist actions  
> **Framework:** LeRobot (v0.4.3 for VQ-BeT, custom script for ACT v2/v3)

---

## Experiment Inventory

| # | Run Name | Policy | Backbone | Cameras | Data | Aug | Loss | Steps | Status |
|---|----------|--------|----------|---------|------|-----|------|-------|--------|
| 1 | ab_act_baseline | ACT | ResNet18 (ImageNet) | center | raw | ✗ | L1 | 85K+ | ✅ Completed (~50 epochs) |
| 2 | ab_imagenet_r50 | VQ-BeT | ResNet50 (ImageNet) | center | raw | ✗ | VQ-BeT | 154K/200K | ⏸ Stopped (~77%) |
| 3 | ab_r3m_r50 | VQ-BeT | ResNet50 (R3M) | center | raw | ✗ | VQ-BeT | 154K/200K | ⏸ Stopped (~77%) |
| 4 | ab_imagenet_r50_v2 | VQ-BeT | ResNet50 (ImageNet) | center | raw | ✗ | VQ-BeT | 3K/200K | ❌ OOM crash |
| 5 | ab_r3m_r50_v2 | VQ-BeT | ResNet50 (R3M) | center | raw | ✗ | VQ-BeT | 3K/200K | ❌ OOM crash |
| 6 | act_v2_center_only | ACT | ResNet18 (ImageNet) | center | raw | ✅ ColorJ+Erase | L1 | 45K/50K | ⏸ Stopped early |
| 7 | act_v2_noaug_control | ACT | ResNet18 (ImageNet) | 3-cam | raw | ✗ | L1 | 50K | ✅ Completed |
| 8 | act_v2_multicam_aug | ACT | ResNet18 (ImageNet) | 3-cam | raw | ✅ ColorJ+Erase | L1 | 17K/50K | ⏸ Still running |
| 9 | act_v2_rn50 | ACT | ResNet50 (ImageNet) | 3-cam | raw | ✅ ColorJ+Erase | L1 | 17K/50K | ⏸ Still running |
| 10 | v3_smooth_only | ACT | ResNet18 (ImageNet) | center | **smoothed** | ✅ ColorJ+Erase | L1 | 50K | ✅ Completed |
| 11 | v3_weight_only | ACT | ResNet18 (ImageNet) | center | raw | ✅ ColorJ+Erase | **Weighted L1** | 50K | ✅ Completed |
| 12 | v3_both_smooth_weight | ACT | ResNet18 (ImageNet) | center | **smoothed** | ✅ ColorJ+Erase | **Weighted L1** | 50K | ✅ Completed |

---

## Generation 1: VQ-BeT Backbone Ablation (Runs 2–5)

### Key Metric: `val/action_mse_norm`

| Run | Best eval step | val/action_mse_norm | val/action_mse_error | val/loss |
|-----|---------------|---------------------|---------------------|----------|
| ImageNet R50 | 130K | **0.02339** | **0.02339** | 236.63 |
| R3M R50 | 115K | 0.02387 | 0.02387 | 240.63 |

### Honest Assessment

> [!WARNING]
> **The VQ-BeT backbone experiment was a dead end.**

1. **ImageNet vs R3M → Negligible difference.** The delta in `val/action_mse_error` between ImageNet (0.0234) and R3M (0.0239) is **2%** — well within noise. You could not reliably distinguish these two backbones. The R3M pretrained features, specifically designed for robotics, provided no advantage over generic ImageNet features for this task.

2. **Severe plateau after step ~15K.** Both runs hit a `val/action_mse_error` floor of ~0.0235–0.0240 by step 15K and then flat-lined for the remaining 135K+ steps. You burned about **10× the useful compute** for zero marginal improvement. The VQ-BeT loss (classification + offset) tells the same story: the val/classification_loss barely moved from step 60K to 150K.

3. **Single-camera limitation.** VQ-BeT v0.4.3's validator accepts only ONE camera input. You had 3 cameras worth of data sitting unused, which was a significant missed opportunity.

4. **V2 OOM crashes.** The second attempt at both ImageNet and R3M ResNet-50 runs crashed at step 3K with `torch.OutOfMemoryError` on a 48GB GPU. This wasted further time and suggests the batch configuration was untested before launch.

5. **GPU utilization was abysmal.** Per your own config notes, VQ-BeT was running at ~5-10% GPU utilization, meaning the GPU was mostly idle while the dataloader struggled. ACT at ~100% was a far better use of the hardware.

**Verdict:** The backbone choice was the wrong variable to ablate. The bottleneck was data and augmentation, not model capacity.

---

## Generation 1.5: ACT Baseline (Run 1)

### Key Metric: Training Loss (no val eval was configured)

| Metric | Early (step 2K) | Mid (step 20K) | Final (step 85K) |
|--------|-----------------|----------------|------------------|
| train/loss | 0.362 | ~0.180 | **0.155** |
| grad_norm | 12.9 | ~4.5 | 1.7 |

### Assessment

> [!IMPORTANT]
> **This was a useful baseline but had critical gaps: no validation evaluation, no multi-cam input, no augmentation.**

- The ACT baseline ran through the stock LeRobot CLI (`lerobot train`), so it used the LeRobot framework's default settings — single center camera, no augmentation, and critically **no offline validation evaluation** was logged.
- Train loss dropped from 0.37 → 0.155 over 85K steps (~50 epochs). The loss was essentially flat after epoch ~20–25 (step ~35K), indicating overfitting onset.
- Without val metrics, you couldn't detect overfitting vs genuine improvement. This was the correct thing to fix in v2 — and you did.
- Grad norm dropped 12.9 → 1.7, indicating the model fully converged. No training issues.

**Verdict:** Served its purpose as a sanity check. Correctly identified that ACT converges much faster and more efficiently than VQ-BeT on this data.

---

## Generation 2: ACT Multi-Variable Ablation (Runs 6–9)

### Key Metric: `val/l1_loss` (lower is better) and `val/action_mse_norm`

| Run | Best val/l1_loss | Best step | val/action_mse_norm @ best | Final l1_loss | Status |
|-----|-----------------|-----------|---------------------------|---------------|--------|
| **center_only** (1-cam, aug) | **0.14093** | 26K | **0.5713** | 0.14482 (45K) | Stopped |
| **noaug_control** (3-cam, no aug) | 0.14369 | 31K | 0.57575 | 0.15315 (50K) | ✅ Done |
| **multicam_aug** (3-cam, aug) | 0.13807 | 14K | 0.55992 | — (17K, still running) | ⏸ Running |
| **rn50** (3-cam, aug, RN50) | 0.14125 | 17K | 0.56909 | — (17K, still running) | ⏸ Running |

### Honest Assessment

> [!NOTE]
> **Generation 2 was well-designed and produced your most informative results to date.**

1. **Augmentation helps: center_only vs noaug_control.** Comparing single-camera + aug (center_only) to 3-cam + no-aug (noaug_control), the augmented single-cam model got a *better* best val/l1_loss (0.1409 vs 0.1437). This tells you augmentation is doing real work — the model is better at generalizing even with 1/3 the visual information.

2. **Multi-cam + aug is the clear winner (so far).** `multicam_aug` (3-cam + augmentation) achieved the best val/l1_loss of **0.13807** at only step 14K, and is still improving. At step 17K it's at 0.14351, which is a minor uptick — may be early signs of overfitting, or just noise.

3. **ResNet50 vs ResNet18 on ACT: still marginal.** The RN50 run's best (0.14125 at step 17K) is not meaningfully better than multicam_aug's 0.13807 — and it costs significantly more compute per step (~2.2s/step vs ~0.28s/step for center_only). The backbone upgrade is not paying for itself.

4. **Overfitting is real.** The noaug_control's val/l1_loss traces a classic overfitting arc: best at 0.1437 @ step 31K, then climbing to 0.1532 by step 50K — that's +7% degradation. Training should have stopped at ~31K. The center_only run shows the same pattern after step 26K.

5. **multicam_aug and rn50 are still running but already showing the overfitting signature.** I note both are at step 17K. If the multicam_aug continues the trend from 14K→17K (l1 went from 0.138 to 0.144), you may want early stopping around step 20–25K.

> [!CAUTION]
> **Two runs (multicam_aug and rn50) are still in progress at only ~34% completion.** They converge much more slowly due to 3-cam input feeding 3× the data per step (dataloader time is ~2.7s vs ~0.3s). At current pace, multicam_aug won't finish for another ~30+ hours.

---

## Generation 3: Data Quality Experiments (Runs 10–12)

### Key Metric: `val/l1_loss` and `val/action_mse_norm`

| Run | Treatment | Best val/l1_loss | Best step | val/action_mse_norm @ best | Final l1_loss (50K) |
|-----|-----------|-----------------|-----------|---------------------------|---------------------|
| **v3_smooth_only** | Smoothed data, standard L1 | **0.15559** | 14K | **0.55444** | 0.1693 |
| **v3_weight_only** | Raw data, weighted L1 | 0.23871 | 35K | 0.80322 | 0.2641 |
| **v3_both** | Smoothed data + weighted L1 | 0.18859 | 47K | 0.62146 | 0.2286 |

### Honest Assessment

> [!WARNING]
> **The v3 data-quality experiments have serious problems. The weighted L1 loss is actively harmful.**

1. **Smoothed data helps modestly — but only without weighting.** `v3_smooth_only` (smoothed data, standard L1) achieved val/l1 = 0.1556, which is *worse* than the v2 center_only run (0.1409). However, the `val/action_mse_norm` of 0.5544 is actually *competitive* with the multicam_aug run (0.5599). This is interesting: smoothing reduces action variance, so the l1_loss is inherently measured against a "calmer" target space. It's probably learning something useful despite the higher l1 number.

2. **Weighted L1 loss is catastrophically bad.** `v3_weight_only` has val/l1 = 0.2387, which is **70% worse** than center_only. The model's val/action_mse_norm oscillates wildly between 0.80 and 1.19 in the 30K-50K range — it is numerically *unstable*. The weighting scheme (floor=0.1, ceil=1.0, scale=0.03) is essentially telling the model to ignore 60% of the dataset (the idle frames), but those frames are the ones that keep the robot from drifting when it should be still. The model ends up mispredicting both idle AND motion frames.

3. **The combination (both) is worse than smooth alone.** `v3_both` shows extreme instability: l1 jumps from 0.189 at step 47K to 0.293 at step 49K — a wildly erratic loss landscape. The weighted loss dominates and poisons the smooth data benefit. Final l1 of 0.229 is nearly 2× worse than center_only.

4. **All v3 runs used only center camera.** This is a step backwards from the v2 multi-cam runs and makes comparison across generations harder. The v3 experiments would have been cleaner if they used the same 3-cam setup as v2 multicam_aug.

---

## Cross-Generation Ranking (Best Val/l1_loss)

| Rank | Run | val/l1_loss | val/action_mse_norm | Key Insight |
|------|-----|------------|---------------------|-------------|
| 🥇 | **multicam_aug** (3-cam, aug, RN18) | **0.13807** | **0.55992** | Multi-cam + augmentation wins; still improving |
| 🥈 | center_only (1-cam, aug) | 0.14093 | 0.57130 | Augmentation matters more than extra cameras |
| 🥉 | rn50 (3-cam, aug, RN50) | 0.14125 | 0.56909 | Bigger backbone gives marginal gain |
| 4 | noaug_control (3-cam, no aug) | 0.14369 | 0.57575 | Proves augmentation is needed |
| 5 | v3_smooth_only (1-cam, smoothed) | 0.15559 | 0.55444 | Smoothing helps `mse_norm` but hurts `l1` |
| 6 | ACT baseline (1-cam, CLI) | ~0.155* | N/A | No val eval; inferred from train loss |
| 7 | v3_both (smooth + weighted) | 0.18859 | 0.62146 | Weighted loss dominates and hurts |
| 8 | VQ-BeT ImageNet R50 | N/A | 0.02339** | Different metric space; not comparable |
| 9 | VQ-BeT R3M R50 | N/A | 0.02387** | Same as above |
| 10 | v3_weight_only | 0.23871 | 0.80322 | Weighted L1 is actively harmful |

*\*Estimated from training loss convergence*  
*\*\*VQ-BeT reports a different `action_mse_norm` because the action space is normalized differently in the VQ codebook*

---

## Key Findings & My Honest Take

### What Worked
1. **Switching from VQ-BeT to ACT was correct.** ACT converges in ~15-25K steps vs VQ-BeT's >150K, uses the GPU far more efficiently, and natively supports multiple cameras. This was the right pivot.
2. **Image augmentation (ColorJitter + RandomErasing) is your most impactful intervention.** It consistently improved validation metrics vs the no-aug baseline by ~3-5%.
3. **Multi-camera input helps**, but the effect is smaller than augmentation. The multicam_aug run benefits from both simultaneously.

### What Didn't Work
1. **Backbone ablation (ImageNet vs R3M, ResNet18 vs ResNet50) is a distraction.** Across all experiments, backbone choice explains <2% of performance variance. The bottleneck is and has always been data quality and augmentation.
2. **Action-weighted L1 loss is harmful in its current form.** The floor=0.1 setting aggressively down-weights idle frames, but these frames teach the robot WHEN NOT TO MOVE — which is arguably the hardest part of cable insertion. This needs to be rethought or abandoned.
3. **The VQ-BeT v2 OOM crashes represent wasted time.** Always smoke-test configs on a small subset first.

### What's Concerning

> [!CAUTION]
> **The fundamental problem hasn't changed: val/l1_loss plateaus around 0.138–0.141 and then overfits.**

All experiments hit a floor in the 0.138–0.155 range. No combination of backbone, augmentation, or loss function has broken through this barrier. This suggests:
- **The dataset itself may be the ceiling.** 151 episodes of keyboard-teleoperated data with inherent jitter may simply not contain enough signal for a policy that needs sub-millimeter insertion precision.
- **The action space (6-DoF Cartesian twist) at 30Hz may be too fine-grained** for the level of precision in the demonstrations.
- **No experiment has been tested in closed-loop rollout yet.** Val/l1_loss is a proxy metric — the only thing that matters is whether the robot can actually insert the cable. A model with l1_loss=0.14 could be functionally useless if the errors compound during rollout.

---

## Concrete Recommendations

### Immediate (this week)
1. **Let multicam_aug finish to ~25K steps, then stop.** It's your best model. Watch for val/l1 > 0.145 as the overfitting signal.
2. **Kill the rn50 run.** It's burning 3× the compute for marginal gain. The backbone is not the bottleneck.
3. **Test the multicam_aug best checkpoint in rollout.** Deploy the step 14K or 15K checkpoint in simulation and measure success rate on the actual insertion task. This is the most important thing you can do right now — all the offline metrics are academic until validated in closed-loop.

### Short Term
4. **Re-run v3_smooth_only with 3 cameras + augmentation.** Smoothed data showed a genuinely good `action_mse_norm`, but the single-camera setup handicapped it. A proper comparison would be: multicam_aug (raw data) vs multicam_aug_smooth (smoothed data), holding everything else constant.
5. **Add early stopping.** Your models consistently overfit after step 25-30K. Implement patience-based early stopping keyed on val/l1_loss with patience=5K steps.

### If Val Performance Plateaus
6. **More data.** 151 episodes is on the low end for visuomotor policies. The original ACT paper used 50 demos but for a much simpler task. Cable insertion demands more demonstrations with higher quality.
7. **Consider spacemouse/gamepad teleoperation** instead of keyboard. The smoothing experiment showed that cleaner trajectory data matters — collecting cleaner data at the source is better than post-hoc filtering.
8. **Try action space redesign.** If the twist actions are noisy, consider delta-position targets or a lower control frequency (15Hz instead of 30Hz) to reduce the policy's prediction burden.
