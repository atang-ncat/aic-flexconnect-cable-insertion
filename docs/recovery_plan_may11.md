# AIC Recovery Plan — May 11 → May 15

> ## 🛑 STOP — current standings (May 12, 07:39)
>
> | Run | Trial 1 SFP | Trial 2 SFP | Trial 3 SC | **Total** |
> |---|---:|---:|---:|---:|
> | **A — pure ACT, no tricks** | 43.28 (final 0.05 m) | 43.17 (final 0.05 m) | **19.30** (final 0.17 m) | **🥇 105.74** |
> | A' — agg recovery (F=6, lift=1.6s) | 43.18 (final 0.05 m) | 42.79 (final 0.05 m) | 1.00 (out-of-bounds) | 86.97 |
> | B — A' + SEATING | 43.94 (final 0.03 m, tier3=25 max) | 44.12 (final 0.04 m, tier3=25 max) | 1.00 (out-of-bounds) | 89.06 |
> | C — mod recovery + tuned SEATING | 22.44 (false-success @ z=0.30 m) | 20.43 (false-success) | 1.00 (out-of-bounds) | 43.87 ← regression |
> | D — agg recovery + SEATING + new guards | **6.54** (gripper crash, contacts:-24) | 42.36 | **−11.00** (96 N press → -12 force penalty) | **37.90** ← worst yet |
> | E1 — "pure ACT" + force-safety cap (BUGGY) | 39.57 (LIFT mid-approach) | 28.79 (LIFT mid-approach) | 7.52 (cap fired but force still hit 97 N) | **75.88** |
>
> **Run E1 exposed two bugs in our code, both now fixed (07:42):**
> 1. **`AIC_ACT_RECOVERY` defaulted to `"1"` (ON).** `unset AIC_ACT_RECOVERY` did not disable it. Combined with leftover `AIC_ACT_RECOVERY_F=6`, `_LIFT_S=1.6` from prior shell sessions, E1 was actually running Run A''s aggressive recovery, knocking SFP plugs ~14 cm off course mid-approach. **Default changed to `"0"` (OFF).** Now `unset` actually disables recovery.
> 2. **Force-safety cap ZEROED velocity but force still climbed to 97 N.** The impedance controller's position target was already integrated 50+ mm downward from past commands, so `F = K·Δx` kept rising even with zero velocity command. **Fix:** safety cap now commands `+5 mm/s upward retreat` instead of zero, actively pulling the position target back. New env: `AIC_ACT_FORCE_SAFETY_RETREAT_VZ` (default 0.005 m/s).
>
> **Run A is STILL the current best by 16+ points after 5 runs.**
>
> **Run D taught us two new failure modes — fixed in code at 07:30:**
> 1. SEATING entered on REAL contact (3.4 N) but admittance pressed for the full 10 s timeout, descending 53 mm (cap was at 150 mm theoretical). The gripper finger crashed into nic_card_mount → −24 contact penalty → tier_2 went **negative**.
>    **Fix:** new `AIC_ACT_SEAT_MAX_DZ=0.020` (default 20 mm) bails to LIFT past this depth without SUCCESS.
> 2. Pure ACT on SC trial 3 is high-variance. Run A got 19.30 (lucky); Run D got -11 (96 N press for 5 s → -12 force penalty). Same code, different stochastic outcome.
>    **Fix:** new universal `AIC_ACT_FORCE_SAFETY_F=15.0` (N) that freezes ALL motion (linear+angular) when force exceeds the cap for 0.3 s. Applies regardless of trick gating. Bounds the worst-case crash damage.
>
> ### THREE-OPTION decision matrix
>
> | If you want to… | Do this |
> |---|---|
> | Stop now and bank a result | Submit **Run A** (no tricks, no SEATING). 105.74 confirmed. |
> | Add ONE more eval to make Run A more reliable | **Run E1** (below): pure ACT + force-safety cap only. Should match Run A while bounding the SC crash variance. **Recommended.** |
> | Try to improve over 105.74 | **Run E2** (below): Run B-style + ALL safety guards. Higher upside, still defended against catastrophic crashes. |
>
> **Always run with `AIC_ACT_SFP_ONLY_TRICKS=1` and `AIC_ACT_FORCE_SAFETY_F=15`.** These are pure safety nets — they cost nothing on good cases and bound the bad ones.

## Run E1' — pure ACT + safety nets (RECOMMENDED next eval)

Goal: confirm we have a reproducible 105+ floor, with crash bounded.
**Fully decontaminated env — clears every leftover `AIC_ACT_*` from prior shell sessions.**

```bash
# Sync patched RunACT.py into the pixi env (now with default RECOVERY=OFF and active force-retreat)
cp /run/host/scratch2/atang/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACT.py \
   /run/host/scratch2/atang/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/RunACT.py

# 🔥 NUKE every AIC_ACT_* env var from prior shell sessions (this is what bit Run E1).
for v in $(env | grep -oE '^AIC_ACT_[A-Z_]+' | sort -u); do unset "$v"; done

# Set ONLY what we want
export AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best
export AIC_ACT_SFP_ONLY_TRICKS=1            # belt: SC → pure ACT regardless
export AIC_ACT_FORCE_SAFETY_F=15.0          # SAFETY NET: retreat at +5mm/s when |F|>15N for 0.3s
export AIC_ACT_FORCE_SAFETY_HOLD_S=0.3
export AIC_ACT_FORCE_SAFETY_RETREAT_VZ=0.005

# Sanity-check before launching — should print exactly these vars and nothing else
env | grep -E '^AIC_ACT_' | sort

# Then in the model terminal:
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunACT
```

**Expected Run E1' outcome:**
- Model startup logs should now show:
  - `Recovery wrapper DISABLED` (NOT `ENABLED`!).
  - `FORCE SAFETY CAP ENABLED: |F|>15.0N for 0.30s → retreat at +5.0mm/s`
- SFP trials: ~43+43 = 86 (matches Run A).
- SC trial: 5–25. Force cap should now keep `|F|` bounded near 15 N (was 97 N in E1).
- **Total: 95–115. Reliable floor at ~95.**

If E1' ≥ 105 → that's the submission. We're done.

## Run E2 — SEATING with all guards (if E1 ≥ 105 and you have 30 min)

Goal: try to add SFP tier_3=25 ceiling without crashing.

```bash
# (sync RunACT.py same as above, then:)
unset AIC_ACT_GAIN AIC_ACT_RESCALE AIC_ACT_Z_FLOOR AIC_ACT_SPIRAL

export AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best
export AIC_ACT_SFP_ONLY_TRICKS=1            # SC → pure ACT
export AIC_ACT_FORCE_SAFETY_F=15.0          # SAFETY NET
export AIC_ACT_FORCE_SAFETY_HOLD_S=0.3

# Recovery — gentler, applies only to SFP
export AIC_ACT_RECOVERY=1
export AIC_ACT_RECOVERY_F=8.0               # Run A's level (less aggressive)
export AIC_ACT_RECOVERY_F_HOLD_S=0.5
export AIC_ACT_RECOVERY_LIFT_S=0.8          # 56 mm lift — enough to retry, not enough to drift far
export AIC_ACT_RECOVERY_LIFT_VZ=0.07

# SEATING with all guards
export AIC_ACT_SEATING=1
export AIC_ACT_SPIRAL_TRIGGER_F=1.5
export AIC_ACT_SPIRAL_HOLD_S=0.3
export AIC_ACT_SEAT_BAIL_F=12.0
export AIC_ACT_SEAT_MIN_CONTACT_F=2.5       # Real-contact guard (no false success)
export AIC_ACT_SEAT_MAX_DZ=0.020            # NEW: cap descent at 20 mm (was unbounded → 53 mm)
export AIC_ACT_SEAT_FZ=2.0
export AIC_ACT_SEAT_KZ=0.005
export AIC_ACT_SEAT_R=0.0015
export AIC_ACT_SEAT_FREQ=1.0
export AIC_ACT_SEAT_DONE_Z=0.005
export AIC_ACT_SEAT_DONE_F=1.0
export AIC_ACT_SEAT_DONE_HOLD_S=0.3
export AIC_ACT_SEAT_MAX_S=4.0               # shortened: 4 s × 5 mm/s = 20 mm theoretical, matches DZ cap

cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunACT
```

**If E2 < E1, revert to E1 (or A) for submission.** No more tinkering.
>
> ## Confirmed scores (May 12, 06:39):
> - `v13 (no recovery)`              → **68.02 / 300** (historical).
> - `v13 + recovery_F=12N`           → **50.64 / 300** (03:18).
> - `v18_sfp_only + recovery_F=12N`  → **26.09 / 300** (03:43).
> - `v19_sfp_pure (Run A, no tricks, no recovery)` → **105.74 / 300** (06:39). 🎉
>   - trial_1 SFP: 43.28 (final 0.05 m — closest near-miss yet, tier_3=24.21)
>   - trial_2 SFP: 43.17 (final 0.05 m, but stuck at |F|>12N, ran out of trial time)
>   - trial_3 SC : 19.30 (final 0.17 m, tier_3=0.37 — expected, no SC training)
>   - **iter-0 unnorm Z action ≈ -15 mm/s on every trial** → no more hedging. Image-stats fix worked.
> - `v19_sfp_pure (Run A', recovery_F=6 hold=0.4 lift=1.6s @0.07m/s)` → **86.97 / 300** (06:51).
>   - trial_1 SFP: 43.18 (final 0.05 m, tier_3=24.08)
>   - trial_2 SFP: 42.79 (final 0.05 m, tier_3=23.92)
>   - trial_3 SC : 1.00 (final 0.27 m — tier_2 zero'd, plug pushed *out of bounding radius* by aggressive LIFT)
>   - aggressive recovery makes SC much WORSE (no training → wandering → LIFT pushes it further away).
> - `v19_sfp_pure (Run B, A' recovery + SEATING)` → **89.06 / 300** (06:58). 🎯
>   - trial_1 SFP: **43.94** (final **0.03 m**, **tier_3=25.00 ← max for no-insertion**)
>   - trial_2 SFP: **44.12** (final **0.04 m**, **tier_3=25.00 ← max for no-insertion**)
>   - trial_3 SC : 1.00 (same SC regression as A')
>   - SEATING fired but bailed within ~0.8 s. Three lessons from the logs:
>     1. SEATING entered at **|F|=5.9N** (with `recovery_F=6`) → no headroom before the 6N bail.
>     2. SEATING bailed sharing the recovery threshold; bail at 9N happened in 0.8 s, only Δz=-0.3mm.
>     3. The 23.6 mm/s spiral peak velocity slaps the plug into the port wall, spiking force fast.
>   - But the **5 mm closer final distance** (0.05 → 0.03/0.04) and tier_3 ceiling proves SEATING is the right idea — it just needs more headroom and a softer spiral.
>
> **🔴 ROOT-CAUSE BUG FOUND (03:43): the entire `v18*` family was poisoned by image stats.**
> `teleop-dataset-combined-v2` borrowed its image normalizer from the largest source
> (`teleop-dataset-gamepad/sc`, 82k frames). That SC dataset has
> `right_camera mean=0.35, std=0.018` — but at deploy the right camera always sees
> mean ≈ 0.59, which is **13σ out of distribution** through the policy's frozen
> ResNet18. RunACT.py *always* applies the saved normalizer (proven by the v13
> IDENTITY-vs-MEAN_STD experiment that swung 3 → 68), so v18, v18b, and
> v18_sfp_only **all carry the broken normalizer baked into their safetensors.**
> Those three checkpoints are unrecoverable — DO NOT keep evaluating them.
>
> The fix is exactly what we were already doing: train on
> `teleop-dataset-sfp-pure`, whose stats come from `gamepad-v1-sfp` (right_camera
> mean=0.58, std=0.034) — i.e. matched to deploy.
>
> **Training status (May 12, 03:46):**
> - `v18_combined_taskcond`            : DEAD (broken stats)
> - `v18b_combined_taskbalanced`       : DEAD (broken stats)
> - `v18_sfp_only`                     : DEAD (broken stats — confirmed 26.09)
> - `v19_sfp_pure` (clean stats)       : RUNNING on GPU 1, step 1500/18000 @ 03:46 (~5:10 AM ETA), val/l1=0.30 @ step 1000 already
>
> **Order of operations:**
> 1. Wait for v19 to finish (~5:10 AM). It is the only model in the pipeline
>    with correct image stats; it should behave like v13 (committed actions).
> 2. Test `v19_sfp_pure` + recovery_F=12N. Compare to v13's 50.64 baseline.
> 3. If v19 ≥ 60: it's our new SFP baseline; ship it as primary.
>    If v19 < 50: ship v13 (no recovery) as the safe 68. Done.
> 4. **SC trial 3 is permanently locked** for tonight. The SC source dataset
>    (`teleop-dataset-gamepad/sc`) was deleted from host disk at some point;
>    `combined-v2`'s 60 SC video symlinks are dangling and point at
>    `/run/host/.../teleop-dataset-gamepad/sc/...` which doesn't resolve from
>    either the host or the docker container today. There is no way to retrain
>    on SC without re-collecting that data. Accept the 75-pt SC ceiling.
>
> **Sub-takeaway: do NOT attempt to retrain anything on `teleop-dataset-combined-v2`** —
> all 60 SC episode videos in it are broken symlinks. Anything training on it will
> either silently skip SC frames (poisoning state/action stats again) or crash on read.

---

## Deploy-time tricks now in `RunACT.py` (May 12, 04:15)

Two new env-flagged behaviours, both default OFF so v13 runs are unchanged.

### Z-descent floor (`AIC_ACT_Z_FLOOR`)

If the policy hedges (commands |vz| < floor) and there's no contact yet,
override z-velocity with a fixed minimum descent. Forces the robot to reach
port height instead of crawling and timing out. Disabled when in LIFT or
when |F| ≥ `AIC_ACT_Z_FLOOR_F_MAX`.

| env var | default | meaning |
|---|---|---|
| `AIC_ACT_Z_FLOOR` | `0.0` (off) | Min descent speed (m/s). Set `0.006` to enable a 6 mm/s floor. |
| `AIC_ACT_Z_FLOOR_F_MAX` | `1.5` | Don't apply floor when contact force exceeds this (N). |

### Spiral lateral search (`AIC_ACT_SPIRAL`)

Classic peg-in-hole trick: when light contact is detected (1.5N < |F| < f_thresh)
for `AIC_ACT_SPIRAL_HOLD_S` seconds, override XY action with a constant-radius
circular sweep so the cable scans across the port mouth. Spiral exits after
`AIC_ACT_SPIRAL_MAX_S` seconds OR when force escalates above f_thresh
(LIFT preempts SEARCH). Z-action stays untouched (or floored) so we keep pressing.

| env var | default | meaning |
|---|---|---|
| `AIC_ACT_SPIRAL` | `0` (off) | `1` to enable. |
| `AIC_ACT_SPIRAL_TRIGGER_F` | `1.5` | Force threshold to start (N). |
| `AIC_ACT_SPIRAL_HOLD_S` | `0.3` | Must hold trigger force this long (s). |
| `AIC_ACT_SPIRAL_R` | `0.004` | Lateral peak displacement (m, default 4mm). |
| `AIC_ACT_SPIRAL_FREQ` | `1.0` | Revolutions per second (Hz). |
| `AIC_ACT_SPIRAL_MAX_S` | `4.0` | Max spiral duration before reverting (s). |

### SEATING (force-controlled insertion, `AIC_ACT_SEATING`) — NEW

The proper deterministic seating module. On the same light-contact trigger as
SPIRAL, switches to:
* **XY**: small fast spiral (default 2.5 mm @ 1.5 Hz) to scan for the chamfer.
* **Z**: admittance loop `vz = -K_z·(|Fz_target| - |Fz_meas|)` — robot becomes
  "soft", presses with constant target Fz, follows the plug down when it slides
  into the channel. Press capped at `AIC_ACT_SEAT_VZ_PRESS`, back-off capped
  asymmetrically tighter at `AIC_ACT_SEAT_VZ_BACK`.
* **rot**: zeroed (no drift during seating).
* **Success**: Δz ≥ `AIC_ACT_SEAT_DONE_Z` AND |F| < `AIC_ACT_SEAT_DONE_F`,
  held for `AIC_ACT_SEAT_DONE_HOLD_S`. Transitions to terminal SEATED
  state (zero motion); we hold position so the cable stays seated.
* **Bail**: |F| > f_thresh OR seat_max_s elapsed → LIFT (existing recovery)
  → re-approach via ACT → seating fires again on next contact.
* **Uses |Fz| for the admittance loop**, |F| total only for triggers. This
  prevents the spiral's lateral side-loads from confusing the press signal.
* If `AIC_ACT_SEATING=1` and recovery is off, recovery is implicitly enabled
  so bail-to-LIFT works.

| env var | default | meaning |
|---|---|---|
| `AIC_ACT_SEATING` | `0` (off) | `1` to enable. Supersedes `AIC_ACT_SPIRAL`. |
| `AIC_ACT_SEAT_FZ` | `2.0` | Target |Fz| during press (N). |
| `AIC_ACT_SEAT_KZ` | `0.005` | Admittance gain (m/s per N). |
| `AIC_ACT_SEAT_VZ_PRESS` | `0.015` | Max press speed (m/s). |
| `AIC_ACT_SEAT_VZ_BACK` | `0.008` | Max back-off speed (m/s). |
| `AIC_ACT_SEAT_R` | `0.0025` | Seating spiral radius (m, 2.5mm). |
| `AIC_ACT_SEAT_FREQ` | `1.5` | Seating spiral freq (Hz). |
| `AIC_ACT_SEAT_DONE_Z` | `0.005` | Min Δz progress for success (m). |
| `AIC_ACT_SEAT_DONE_F` | `1.0` | Max |F| during success window (N). |
| `AIC_ACT_SEAT_DONE_HOLD_S` | `0.3` | Hold success window this long (s). |
| `AIC_ACT_SEAT_MAX_S` | `8.0` | Timeout per attempt before LIFT (s). |

---

## v19 eval commands (paste when training finishes ~5:00 AM)

In the **eval container** terminal:

```bash
# 0. Sync patched RunACT.py into pixi env site-packages (one-time per session)
cp /run/host/scratch2/atang/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACT.py \
   /run/host/scratch2/atang/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/RunACT.py

# 0b. Verify checkpoint stats are healthy BEFORE wasting eval cycles
pixi run python /run/host/scratch2/atang/ws_aic/scripts/verify_checkpoint_stats.py \
    /run/host/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best
# If it says FAIL: stop. Ship v13. Don't waste trials.
```

### Eval matrix — run them in this order until something beats 50.64

```bash
# Common setup
export AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best
unset AIC_ACT_GAIN AIC_ACT_RESCALE
cd /run/host/scratch2/atang/ws_aic/src/aic
LAUNCH='pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.RunACT'

# Run A: v19 NO recovery, NO tricks (apples-to-apples vs v13's 68)
unset AIC_ACT_RECOVERY AIC_ACT_RECOVERY_F AIC_ACT_RECOVERY_LIFT_S AIC_ACT_RECOVERY_LIFT_VZ
unset AIC_ACT_Z_FLOOR AIC_ACT_SPIRAL
$LAUNCH

# Run B: v19 + recovery_F=12 (same wrapper that v18-sfp-only used)
export AIC_ACT_RECOVERY=1 AIC_ACT_RECOVERY_F=12.0 AIC_ACT_RECOVERY_LIFT_S=2.0 AIC_ACT_RECOVERY_LIFT_VZ=0.05
unset AIC_ACT_Z_FLOOR AIC_ACT_SPIRAL
$LAUNCH

# Run C: v19 + recovery + Z-floor (force descent to port height)
export AIC_ACT_RECOVERY=1 AIC_ACT_RECOVERY_F=12.0 AIC_ACT_RECOVERY_LIFT_S=2.0 AIC_ACT_RECOVERY_LIFT_VZ=0.05
export AIC_ACT_Z_FLOOR=0.008 AIC_ACT_Z_FLOOR_F_MAX=1.5
unset AIC_ACT_SPIRAL
$LAUNCH

# Run D: v19 + recovery + Z-floor + spiral search (kitchen sink, no seating)
export AIC_ACT_RECOVERY=1 AIC_ACT_RECOVERY_F=12.0 AIC_ACT_RECOVERY_LIFT_S=2.0 AIC_ACT_RECOVERY_LIFT_VZ=0.05
export AIC_ACT_Z_FLOOR=0.008 AIC_ACT_Z_FLOOR_F_MAX=1.5
export AIC_ACT_SPIRAL=1 AIC_ACT_SPIRAL_TRIGGER_F=1.5 AIC_ACT_SPIRAL_R=0.004 AIC_ACT_SPIRAL_FREQ=1.0 AIC_ACT_SPIRAL_MAX_S=4.0
unset AIC_ACT_SEATING
$LAUNCH

# Run E: v19 + recovery + Z-floor + SEATING (the architecture we agreed on)
# This is the one we expect to actually insert.
export AIC_ACT_RECOVERY=1 AIC_ACT_RECOVERY_F=12.0 AIC_ACT_RECOVERY_LIFT_S=2.0 AIC_ACT_RECOVERY_LIFT_VZ=0.05
export AIC_ACT_Z_FLOOR=0.008 AIC_ACT_Z_FLOOR_F_MAX=1.5
unset AIC_ACT_SPIRAL  # SEATING supersedes SPIRAL
export AIC_ACT_SEATING=1
# (SEAT_* defaults are good first guesses; tune after first run)
$LAUNCH
```

### Decision tree

- **Run A** (v19 raw): tells us whether v19 deploys cleanly. We expect ~68 if image-stats fix worked.
- **Run B** (recovery only): apples-to-apples vs v18-sfp-only's 26.09. We expect ~50–80.
- **Run C** (+ Z-floor): catches the case where policy hedges; floors descent. Should improve trial_1.
- **Run D** (+ spiral, no seating): adds the lateral search. Might bump trial_2.
- **Run E** (+ SEATING): the real architecture. Force-controlled insertion. Target: **≥ 150**.
- If E hits ≥ 150 → that's submission v1. Then Wednesday spends on tuning E's thresholds for max score.
- If E < 100 → debug seating logs, look for `SEATING[N] entered/SUCCESS/TIMEOUT/HARD-CONTACT` lines, tune the threshold that's failing.

### What "good" looks like in iter 0 logs

- `|UNNORM action|_inf ≥ 8e-3` (8 mm/s) — v13 had 1.3e-2 and committed; v18-sfp-only had 4.85e-3 and hedged.
- All three RAW image means within ±0.05 of [0.59, 0.61, 0.59] — confirms image stats match deploy.
- During Run D, look for `SPIRAL[1] SEARCH triggered` lines as the cable contacts the port lip.
- During Run E, look for `SEATING[1] entered: |F|=X.XN z0=YYY.Ymm` and ideally `SEATING[1] SUCCESS: Δz=+X.Xmm`. If you see `TIMEOUT` or `HARD-CONTACT`, those tell you which threshold to tune.

### Tuning hints for Run E (in order of likely-needed)

| Symptom in logs | Likely fix |
|---|---|
| `SEATING entered` but `TIMEOUT` with Δz~0 | `Kz` too low or `Fz_target` too low — increase one |
| `HARD-CONTACT` immediately after entering | `Fz_target` too high; lower it; or `vz_press_max` too high |
| Many SEATING attempts, never SUCCESS | spiral radius probably wrong; try `AIC_ACT_SEAT_R=0.005` (5mm) |
| Almost SUCCESS then timeout | `seat_done_f` too tight; try `AIC_ACT_SEAT_DONE_F=1.5` |
| SUCCESS but trial scores low | `seat_done_z` too small; try `AIC_ACT_SEAT_DONE_Z=0.008` |

> **Where you are tonight:** v13 (68/300) is still the best policy. Everything since
> (v14 → v17c) has either matched or undershot it on val/l1, and none has unlocked
> insertion success. You have 4 days, 3 datasets that haven't been used together,
> and one clear hypothesis about why training keeps stalling.
>
> **This document is deliberately short.** Don't add anything to it tonight. Read it
> tomorrow morning, do step 1, then step 2. We submit on May 14 and treat May 15
> as the buffer.

---

## What is actually wrong (one paragraph)

You have been iterating *training hyperparameters* on the **wrong dataset
shape** for the last three weeks. The core problems are not action weighting,
KL weight, chunk size, or backbone:

1. **The policy has no way to tell SFP from SC.** Trial 3 (worth 75 pts) is
   guaranteed to score zero unless we condition the model on the task.
2. **The 60-episode gamepad SC dataset has never been used for training.**
   That is your single biggest unused resource. (`teleop-dataset-gamepad/sc`)
3. **The 71-episode gamepad-v2 SFP and the 52-episode gamepad-v1 SFP have not
   been combined.** Training has been running on one or the other, or on the
   keyboard data, but never on the full gamepad pool with task tags.
4. **The deployed model has no recovery behaviour** because the demos do not
   contain miss-and-retry. The fastest fix is *not* more demos — it is a
   thin force-feedback wrapper around `RunACT.py` that backs the gripper off
   when it detects a crashed insertion and re-queries the policy.

Everything below addresses these four things in order.

---

## The plan, day by day

### Day 1 — Tue May 12 — One training run, one config, no improvising

Goal: launch a single training job that uses every demo we have, with task
conditioning, and let it run overnight.

**1.1 Build the unified, task-tagged dataset.** The new script
`scripts/build_combined_v2.py` (committed alongside this plan) merges:

| Source | Episodes | Task tag |
|--------|---------:|----------|
| `datasets/teleop-dataset-gamepad/sfp` | 52 | `insert_sfp` |
| `datasets/teleop-dataset-gamepad-v2/sfp` | 71 | `insert_sfp` |
| `datasets/teleop-dataset-gamepad/sc` | 60 | `insert_sc` |
| **Total** | **183** | |

The script writes a LeRobot v3.0 dataset to
`datasets/teleop-dataset-combined-v2/` with:

* `task_index` column populated per-episode (0 = SFP, 1 = SC).
* `meta/tasks.parquet` with `{0: "insert_sfp", 1: "insert_sc"}`.
* All cameras and the 39-D `observation.state` preserved as-is.
* Failed episodes flagged via `probe_episode_outcomes.py` are dropped (we
  emit a `dropped_episodes.json` next to the dataset for traceability).

Run it once:

```bash
cd /scratch2/atang/ws_aic/src/aic
pixi run python /scratch2/atang/ws_aic/scripts/build_combined_v2.py \
    --out /scratch2/atang/ws_aic/datasets/teleop-dataset-combined-v2 \
    --drop-failed
```

Expected runtime: ~5–8 minutes (it's mostly symlinking videos, no re-encode).

**1.2 Train v18 with task conditioning.** The new config
`configs/act_sfp_v18.yaml` is v13's recipe applied to the new dataset, with a
small `extra_state_inject` block that tells `train_act.py` to append a 2-D
one-hot to `observation.state` based on `task_index`. State goes 27 → 29
dims; the policy's state projection auto-resizes.

```bash
cd /scratch2/atang/ws_aic/src/aic && CUDA_VISIBLE_DEVICES=0 \
    pixi run python /scratch2/atang/ws_aic/scripts/train_act.py \
    --config /scratch2/atang/ws_aic/configs/act_sfp_v18.yaml
```

Expected: val/l1 in the 0.16–0.20 band (gamepad data has lower variance than
keyboard), early-stops around step 8–12k, ~3 hours wallclock on one RTX 6000.

> **Do not start a second config tonight... unless you have spare GPUs.**
> v17a/b/c showed more ablations don't help when the data shape is wrong,
> but the data shape is now fixed. With multi-GPU, two careful parallel
> ablations are worth queuing — see the next subsection.

#### 1.3 Optional: two parallel ablations on spare GPUs

If you have free GPUs sitting idle while v18 runs on GPU 0, queue these
two **fire-and-forget** runs. By morning you'll have v18 + 2 ablations
to choose from at eval time. Both reuse the same combined-v2 dataset; no
extra data prep needed.

**v18b — task-balanced sampling** (GPU 1).
The combined-v2 dataset is 123 SFP : 60 SC episodes (~2:1). v18 random-
samples frames, so each batch is ~2:1 SFP/SC and gradients are SFP-
dominated. v18b enables `dataset.task_balance.enable=true`, which builds
a `WeightedRandomSampler` with per-frame weights `1/freq[task_index]`.
Each batch is now ~50/50 SFP/SC, which should make SC training stronger
without giving up SFP. **Hypothesis to confirm tomorrow:** `val/l1_sc`
(once we add per-task logging) should drop more than v18's, and the
overall `val/l1` should be similar or slightly higher (that's fine; the
balanced average is the meaningful number for the eval mix).

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && CUDA_VISIBLE_DEVICES=1 \
    pixi run python /run/host/scratch2/atang/ws_aic/scripts/train_act.py \
    --config /run/host/scratch2/atang/ws_aic/configs/act_sfp_v18b.yaml
```

**v18-sfp-only — pure SFP baseline** (GPU 3).
Trains an otherwise identical ACT on **only** the 123 SFP episodes (no
task conditioning, no SC). Two purposes:

1. **Sanity check on task conditioning.** If v18-sfp-only's val/l1 is
   meaningfully lower than v18's val/l1 on the same SFP val episodes,
   we've measured the cost of mixing SC into the same network — and can
   decide to ship two task-specific checkpoints instead of one.
2. **Submission backup for SFP.** If everything else goes sideways, this
   is the cleanest possible SFP-only model on gamepad data.

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && CUDA_VISIBLE_DEVICES=3 \
    pixi run python /run/host/scratch2/atang/ws_aic/scripts/train_act.py \
    --config /run/host/scratch2/atang/ws_aic/configs/act_sfp_v18_sfponly.yaml
```

**What to look for in the startup logs of each:**

* v18b: `Task-balanced sampling ENABLED: WeightedRandomSampler will
  weight each frame by 1/freq[task_index].` and `Split 183 episodes ->
  165 train / 18 val ... per-task: train={0: 111, 1: 54} val={0: 12, 1: 6}`.
* v18-sfp-only: `Episode task filter ACTIVE: keeping 123/183 episodes
  with task_index in [0]` and `Split 123 episodes -> 105 train / 18 val
  ... per-task: train={0: 105} val={0: 18}`.

If either startup log doesn't show those lines, kill that run — it means
the new config knob didn't take effect.

### Day 2 — Wed May 13 — Eval and the force-feedback wrapper

**2.1 Gazebo-eval v18.** `RunACT.py` is already patched for v18:

* `__init__` auto-detects task conditioning from the saved `state_dim`
  (`28/29/30` → `num_tasks = state_dim − 27`). Override with
  `AIC_ACT_NUM_TASKS=2` if it ever miscounts.
* `prepare_observations` builds the 27-D base state and appends the task
  one-hot before normalization.
* `insert_cable` reads `task.plug_type` / `task.port_type` (both checked
  for the substring `"sc"`) to pick `task_index ∈ {0=SFP, 1=SC}`.
* The default checkpoint path now points to
  `outputs/act_sfp/v18_combined_taskcond/checkpoints/best`. To run the old
  v13 without task conditioning, set
  `AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v13_fixed_img_norm/checkpoints/best`
  (auto-detection sees state_dim=27 and disables the one-hot).

Expected outcome: trial 1 distance ≤ v13's 0.09 m, trial 2 ≤ 0.06 m, trial 3
goes from "policy doesn't move" to "policy approaches SC port". If trial 3
just gets the plug into the bounding radius, it's worth 25–50 pts of
proximity score that v13 currently has zero of.

**2.2 Force-feedback recovery wrapper (already in `RunACT.py`).** Wraps
`insert_cable()` in a `NORMAL → LIFT → NORMAL` state machine that monitors
contact force magnitude on the wrist. When `|F| > AIC_ACT_RECOVERY_F` for
`AIC_ACT_RECOVERY_F_HOLD_S` consecutive seconds, the wrapper:

1. Calls `self.policy.reset()` to flush the temporal-ensemble buffer.
2. Commands a constant `+Z` lift at `AIC_ACT_RECOVERY_LIFT_VZ` for
   `AIC_ACT_RECOVERY_LIFT_S` seconds (XY/rot zeroed).
3. Returns to `NORMAL` so the policy plans a fresh approach on the
   newly-observed scene.

Defaults (env-overridable):

| Env var | Default | Notes |
|---|---|---|
| `AIC_ACT_RECOVERY` | `1` | Set to `0` to disable the wrapper. |
| `AIC_ACT_RECOVERY_F` | `8.0` N | Force threshold to trigger recovery. |
| `AIC_ACT_RECOVERY_F_HOLD_S` | `0.6` s | Duration force must be sustained. |
| `AIC_ACT_RECOVERY_LIFT_S` | `0.6` s | Time to spend lifting. |
| `AIC_ACT_RECOVERY_LIFT_VZ` | `0.05` m/s | Lift velocity (50 mm/s). |

Two behaviours fall out of this for free:

* **Failed insertion → policy retries.** The plug lifts ~30 mm in 0.6 s,
  the policy is fed a fresh observation, and it gets another shot at
  alignment. v13's main failure mode (slide down past the port, never look
  up) goes away.
* **Hard contact protection.** Anything pushing > 8 N for more than 0.6 s
  triggers the lift, which kills most of the >20 N force-penalty risk.

This is deploy-time only — works for both v13 and v18 checkpoints, and is
turned on by default.

### Day 3 — Thu May 14 — Submit, do not iterate

**3.1 Pick the best of {v13 + recovery wrapper, v18, v18 + recovery wrapper}**
based on the Gazebo eval numbers from Day 2. If v18 doesn't clearly beat v13
on insertion success, **submit v13 + recovery wrapper**. v13 has the most
real-world hours behind it; v18 is only one training run old.

**3.2 Build the submission Docker** following
`src/aic/docs/submission.md`. Test the lifecycle compliance locally
(unconfigured → configured → active → deactivate → cleanup → shutdown).

**3.3 Submit.** Late afternoon at the latest.

### Day 4 — Fri May 15 — Buffer only

Do **not** plan new work for this day. It exists to absorb a packaging bug
or a re-submission. If everything went well on Day 3, take the day off and
read.

---

## What we are explicitly *not* doing

These were tempting and they are wrong calls right now:

| Tempting thing | Why we're skipping it |
|---|---|
| Diffusion Policy / VQ-BeT swap | A new architecture costs more than a week to debug. ACT v13 already scores 68; we don't need a new architecture, we need recoveries. |
| Re-collecting more demos | 183 episodes is enough if the data pipeline is correct. We have spent five weeks proving that more poorly-conditioned demos do not help. |
| Bigger backbone (RN50, RN34) | The cross-experiment review (`analysis/training_experiments_review.md`) showed <2% delta. This is a known dead end. |
| Stronger action weighting (≥40×) | v14 already showed diminishing returns. |
| Action rescaling experiments | v15/v15b were a workaround for keyboard binarity. Gamepad data is continuous; rescaling actively blunts it. |
| Hybrid VQ-BeT + ACT | A Phase-3 luxury we no longer have time for. |

---

## How to know if v18 is better than v13 *before* the Gazebo eval

Two cheap signals during the v18 run:

* **val/l1 should drop below 0.20.** v17 plateaued at 0.21 on a slightly
  different mix; the SC episodes plus the task one-hot should give us
  another step down. If v18 stalls above 0.22, something is wrong with the
  task injection — open `outputs/act_sfp/v18_combined_taskcond/train.log`
  and look for `"Task injection ENABLED: state 27 → 29 dims"`. If that line
  is missing, the config didn't take effect.
* **val/action_mse_norm in the 0.55–0.65 band.** v17 sat at 0.81 — that's
  the "everything averages to zero" regime. Anything below 0.7 means the
  model is committing to actions instead of hedging.

If both signals are in the right band by step 5000, let the run finish
overnight. If either is wrong, **stop the run and revisit before
investing eval time.**

---

## Files this plan adds (small, all reviewed in the same PR)

* `scripts/build_combined_v2.py` — dataset merge with task tagging.
* `configs/act_sfp_v18.yaml` — single training config for v18.
* `configs/act_sfp_v18b.yaml` — task-balanced sampling ablation.
* `configs/act_sfp_v18_sfponly.yaml` — pure-SFP baseline ablation.
* (modifies) `scripts/train_act.py` — adds:
  - `task_inject` config block (one-hot `task_index` appended to state).
  - `task_balance` config block (WeightedRandomSampler).
  - `episode_task_filter` config block (drop episodes by task_index).
  - `load_episode_task_index` and `split_episodes_balanced` helpers.
* (modifies) `src/aic/aic_example_policies/aic_example_policies/ros/RunACT.py`
  — task-conditioning auto-detect from `state_dim`, task one-hot append in
  `prepare_observations`, and the force-feedback recovery state machine in
  `insert_cable`.

That's it. No other repo changes tonight.

---

## One final thing

You're tired and the deadline is real. The plan above is sized for one
person doing one focused day at a time. If something on Day 2 takes longer
than expected, **collapse Day 3 into "submit v13 with the recovery
wrapper"**. The recovery wrapper alone is probably worth +20 points across
the three trials, and it's a 30-line edit. Do not let the perfect block
the deadline.

---

## Post-Run-C analysis (May 12, 07:10)

### What Run C taught us

The model log was lying. SEATING SUCCESS fired on every trial, but the
robot froze 100 mm above the port. The root cause: SEATING entered in
**free space** because we lowered `AIC_ACT_SPIRAL_TRIGGER_F` to 0.8 N (well
within the gripper's 0.9-1.2 N inertial transient as it accelerates from
home). Once entered, admittance descended at ~10 mm/s and the success
criterion (`Δz≥5mm AND |F|<1N`) is **trivially satisfied in free space**:
no force ever builds up because we are not touching anything.

### Two code fixes in `RunACT.py`

**1. Per-task gating (kills the SC regression).** Tricks now check
`current_task_index`. The default (`AIC_ACT_SFP_ONLY_TRICKS=1`) disables
recovery, Z-floor, spiral, and SEATING for SC trials. SC always runs pure
ACT → SC trial 3 will reliably score ~19 pts (Run A's level), no more 1.0
collapses.

**2. SEATING SUCCESS guard.** Added `AIC_ACT_SEAT_MIN_CONTACT_F` (default
2.5 N): SUCCESS requires `seat_max_f_seen ≥ MIN_CONTACT_F`. A free-space
trigger that never builds force will time out and bail to LIFT instead of
falsely succeeding. Real contact (typically 3-10 N when pressing the port
plate) easily clears the guard.

### Run D parameters (recommended next eval)

The goal of Run D is to RECOVER Run B's tier_3=25 ceiling on SFP without
the SC regression and without the false-success bug.

```bash
# Sync patched RunACT.py into the pixi env
cp /run/host/scratch2/atang/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunACT.py \
   /run/host/scratch2/atang/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/RunACT.py

# Clean env
unset AIC_ACT_GAIN AIC_ACT_RESCALE AIC_ACT_Z_FLOOR AIC_ACT_SPIRAL

export AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best

# Per-task gating (default ON; just be explicit). SC trials → pure ACT.
export AIC_ACT_SFP_ONLY_TRICKS=1

# Recovery — Run B's aggressive values (only applies to SFP now)
export AIC_ACT_RECOVERY=1
export AIC_ACT_RECOVERY_F=6.0
export AIC_ACT_RECOVERY_F_HOLD_S=0.4
export AIC_ACT_RECOVERY_LIFT_S=1.6
export AIC_ACT_RECOVERY_LIFT_VZ=0.07

# SEATING — Run B-style (1.5N trigger above gripper inertia, 0.3s hold)
export AIC_ACT_SEATING=1
export AIC_ACT_SPIRAL_TRIGGER_F=1.5    # back to 1.5N (above 1.1N inertial noise)
export AIC_ACT_SPIRAL_HOLD_S=0.3       # back to 0.3s (must sustain for 3 control cycles)
export AIC_ACT_SEAT_BAIL_F=12.0        # SEATING tolerates |F|<12N before LIFT
export AIC_ACT_SEAT_MIN_CONTACT_F=2.5  # NEW: must observe ≥2.5N during seating to declare SUCCESS
export AIC_ACT_SEAT_FZ=2.0
export AIC_ACT_SEAT_KZ=0.005
export AIC_ACT_SEAT_R=0.0015           # small spiral (1.5mm radius)
export AIC_ACT_SEAT_FREQ=1.0           # 1.0Hz → peak vxy=9.4mm/s
export AIC_ACT_SEAT_DONE_Z=0.005
export AIC_ACT_SEAT_DONE_F=1.0
export AIC_ACT_SEAT_DONE_HOLD_S=0.3
export AIC_ACT_SEAT_MAX_S=10.0

cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunACT
```

### Expected Run D outcome

- **Trial 3 (SC):** pure ACT (tricks gated off) → ~19 pts. **Locked.**
- **Trial 1, 2 (SFP):** SEATING fires only on real contact (≥1.5N for 0.3s),
  with proper bail and proper success guard. Should match Run B's tier_3=25
  ceiling without false-locking high. Each trial expected 43-45 pts.
- **Total Run D expected: 105-115 pts.**

If trial 1 and 2 again log `SEATING SUCCESS` but tier_3 < 20:
- Read the SUCCESS line — what's `maxF=`? If it's 2.5-3 N, we're still
  locking too high. Raise `AIC_ACT_SEAT_MIN_CONTACT_F=4.0`.
- The honest fix is a much larger spiral. The plug is laterally misaligned
  by ~5 cm at the contact point; a 1.5 mm spiral cannot find the chamfer.

### Run E (only if Run D ≥ Run A baseline)

Try a **growing-radius spiral** in the SEATING phase to actually search the
5 cm misalignment region. Pseudo-code:
```python
r_t = seat_amp_m * (1.0 + elapsed_seat * spiral_growth_per_s)
phase = seat_omega * elapsed_seat
action[0] = -seat_omega * r_t * np.sin(phase)
action[1] =  seat_omega * r_t * np.cos(phase)
```
With `r_t` growing 1 mm/s, after 8 s the radius reaches 9.5 mm. Combined
with the chamfer reaction force, the plug self-aligns. (This is a 5-line
edit; only attempt if there's >2 hrs left.)

### Submission floor

If Run D regresses below 100, **revert to Run A configuration**
(everything off):
```bash
unset AIC_ACT_RECOVERY AIC_ACT_SEATING AIC_ACT_SPIRAL AIC_ACT_Z_FLOOR
unset AIC_ACT_GAIN AIC_ACT_RESCALE
export AIC_ACT_POLICY_PATH=/scratch2/atang/ws_aic/outputs/act_sfp/v19_sfp_pure/checkpoints/best
```
That delivers the 105.74 we already have in hand.
