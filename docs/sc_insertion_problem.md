# SC Port Insertion Problem — Technical Description

## Context

We are working on the AI for Industry Challenge (AIC) where a UR5e robot must insert fiber-optic cable connectors into ports on a task board. The `CheatCode` policy is a reference implementation that uses **ground-truth TF frames** (available only during training) and a **pure integral (I-only) lateral controller** to guide the connector into the port.

**CheatCode works reliably for SFP insertion into NIC card ports (~60-75% success rate)** but **consistently fails for SC connector insertion into SC ports**.

The modified CheatCode source is at:
- Source: `src/aic/aic_example_policies/aic_example_policies/ros/CheatCode.py`
- Runtime copy: `src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/CheatCode.py`

## The Setup

- **Cable type:** `sfp_sc_cable_reversed` (SC end faces forward)
- **Plug frame:** `sc_tip` (TF frame at the tip of the SC connector, attached to gripper)
- **Port frame:** `sc_port_base` (TF frame at the base/bottom of the SC port)
- **Port entrance frame:** `sc_port_base_link_entrance` (TF frame at the entrance of the SC port, offset -0.01564m along port's local Z from `sc_port_base`)
- **Target module:** `sc_port_0` or `sc_port_1`

The SC port is the **small blue box** on the task board. When mounted on the board in the default configuration, its opening faces **upward**, making the insertion direction approximately **vertical (top-down)** — same general direction as SFP insertion into the NIC card. However, the board can have yaw offsets (Trial 3 uses board yaw=3.0 rad instead of π≈3.14, i.e. ~8° rotation), so it is not exactly vertical — the orientation-aware code handles this correctly.

## How CheatCode Works

CheatCode operates in two phases:

### Phase 1: Approach (pfrac 0.0 → 1.0)
- Linearly interpolates from current gripper position to the target port position + a 0.2m offset along the port's insertion axis (local Z)
- No integrator correction during this phase — **the integrator is explicitly reset to zero every tick** (`reset_xy_integrator=True`)
- Takes about 15 seconds

### Phase 2: Insertion (z_offset 0.2 → -0.015)
- Steps z_offset down by 0.0005m per tick along the port's insertion axis
- Descent speed: 0.0005m / 0.05s = **10mm/s** (same for both SFP and SC)
- A **pure integral (I-only) controller** corrects lateral misalignment:
  ```python
  # There is NO proportional term — only integrator * gain
  target = (
      port_pos
      + insertion_axis * (z_offset - insertion_offset)
      + lateral_u * i_gain * self._tip_x_error_integrator
      + lateral_v * i_gain * self._tip_y_error_integrator
  )
  ```
- The integrators accumulate u/v errors each tick and are clamped to `±_max_integrator_windup`

**Key controller parameters:**
- `i_gain = 0.15`
- `_max_integrator_windup = 0.05`
- **Max lateral correction = 0.15 × 0.05 = 0.0075m = 7.5mm**

### Why "I-Only" Matters

The controller has **no proportional (P) term** — no `Kp * current_error` applied directly. The consequences:
- Correction **ramps up slowly** as the integrator accumulates (no immediate response to error)
- Once correction kicks in, there's **no proportional damping** to prevent overshoot
- After overshooting, the integrator must **slowly unwind** before correcting in the other direction
- This creates the classic I-controller oscillation pattern: slow ramp → overshoot → slow unwind → overshoot opposite → repeat

The feedforward positioning (targeting `port_pos`) provides some implicit positioning, but it is **not** a P-term in the control loop — it doesn't scale with current lateral error.

## What Happens During SC Insertion

### Phase 1 (Approach) — Works Fine
The arm smoothly approaches the SC port. By pfrac=1.0, the connector is positioned ~0.2m above the port. However, the uv_error at the end of approach is typically:
```
uv_error: 0.0201 -0.00181   (20mm offset in u, 1.8mm in v)
```

**Critical issue:** Because the integrator is reset during the entire approach phase, this 20mm lateral offset is not being corrected during approach. When the descent phase begins, the integrator starts from zero — so the first few steps of descent have **near-zero lateral correction**, exactly when initial alignment matters most for SC.

### Phase 2 (Insertion) — Fails
As z_offset decreases from 0.2 toward -0.015:

1. **Slow integrator ramp-up:** The integrator starts from zero and must accumulate error over many ticks before generating meaningful correction. Meanwhile, the arm is already descending at 10mm/s toward the port.

2. **Persistent lateral offset:** The u_error stays at ~0.008-0.02m (8-20mm) through much of the descent. The integrator saturates at 0.05 and the max correction (7.5mm) is insufficient to fully close the gap.

3. **Partial insertion:** Around z_offset ~0.05-0.03, the connector reaches the SC port. Because of the lateral offset, **one side of the connector enters the port but the other side hangs outside**. The connector rests on the lip of the port at an angle.

4. **Contact destabilization (no force feedback):** As the connector makes contact with the port edge, contact forces push the plug sideways. **The F/T sensor is available** (`/fts_broadcaster/wrench`) **but never consulted** — the descent is purely position-based. The u/v errors start oscillating:
```
z_offset: 0.035   uv_error:  0.00637 -0.0034
z_offset: 0.031   uv_error:  0.00136 -0.00702
z_offset: 0.029   uv_error: -0.00101 -0.00753
z_offset: 0.025   uv_error: -0.00302 -0.00574
```

5. **Arm retracts:** When z_offset reaches -0.015, CheatCode declares "done" regardless of actual insertion success. The policy enters a "stabilization" phase where it stops sending commands, and the arm returns to a neutral position — **pulling the connector away from the port**.

6. **False positive:** `insert_cable()` returns `True` even though insertion did not occur. The `/scoring/insertion_event` topic does NOT fire — confirming no actual insertion happened.

## Why SFP Works But SC Doesn't

| Factor | SFP (NIC Card) | SC Port |
|--------|----------------|---------|
| Port width | ~13mm (tight slot) | ~9mm (open cavity) |
| Physical guide rails | **Yes** — rails mechanically constrain and redirect | **No** — plug rebounds freely |
| Max lateral correction (7.5mm) relative to port width | ~58% of port width — rails absorb the rest | ~83% of port width — can push plug across entire opening |
| Integrator overshoots | Rails absorb overshoot, hold plug in place | Plug oscillates freely in opposite direction |
| Error reversal after contact | Doesn't matter — rails hold alignment | Starts oscillation cycle |
| Insertion without precise alignment | Possible — rails guide connector in | Impossible — must be centered within ~1mm |

**The max lateral correction of 7.5mm relative to the SC port width of ~9mm means the I-controller can push the plug across nearly the entire port opening, guaranteeing oscillation when there's no physical damping.**

## What We Tried

### Attempt 1: Generalized Insertion Axis
**Change:** Replaced hardcoded world-Z descent with port-orientation-aware insertion. Uses the port's quaternion to compute the actual insertion axis from its rotation matrix.

**Result:** The arm still descends approximately vertically (because the SC port's local Z IS approximately vertical in the default task board config). Same lateral offset problem persists. The generalization is correct and useful for non-default board orientations, but it doesn't fix the lateral alignment issue.

### Attempt 2: Increased I Gains
**Change:** `_max_integrator_windup: 0.05 → 0.15`, `i_gain: 0.15 → 0.25`
**Max correction:** 0.25 × 0.15 = 0.0375m = 37.5mm (5× more authority)

**Result:** Made things **significantly worse**. The increased correction authority caused the controller to **overshoot violently**. The connector swings past the port, the integrator winds up the other way, swings back — creating a growing oscillation. The integrators bounce between +0.15 and -0.15:
```
z_offset: 0.006   uv_error:  0.0209 -0.0279   integrators: -0.128,  0.113
z_offset: 0.003   uv_error:  0.0275 -0.0343   integrators:  0.0288, -0.0852
z_offset: -0.005  uv_error: -0.0143  0.0155   integrators:  0.135,  -0.131
z_offset: -0.011  uv_error: -0.0332  0.0305   integrators: -0.15,    0.15   ← full saturation
```

The connector swings back and forth across the port like a pendulum. **Reverted gains back to original values.**

## Root Cause Summary

The failure has **five compounding factors**:

1. **Pure I-controller with no P-term** — Correction ramps up slowly, overshoots, unwinds slowly, overshoots opposite. A proportional term would provide immediate correction and damping.

2. **SC port has no physical self-correction** — Unlike SFP's guide rails, the SC port is a simple rectangular cavity that can't mechanically redirect a misaligned connector.

3. **Integrator reset during approach** — The ~20mm lateral offset accumulated during approach is discarded when the descent phase begins. The controller starts from zero at the worst possible time.

4. **No force feedback** — The F/T sensor data is available but ignored. Contact forces from hitting the port lip push the connector sideways, but the controller can't detect or react to this.

5. **Descent speed too fast for I-only control** — 10mm/s descent doesn't give the integrator enough time to converge for SC's tight tolerance. Same speed works for SFP because the rails compensate.

## Potential Fixes (Not Yet Attempted)

1. **Add a proportional term** — `correction = Kp * current_error + Ki * integrator` would give immediate correction and reduce overshoot
2. **Pre-seed the integrator** — Instead of resetting to zero at descent start, initialize the integrator from the final approach error
3. **Add force feedback** — Monitor `/fts_broadcaster/wrench` during descent; pause/backoff when contact force exceeds threshold
4. **Reduce descent speed for SC** — Slower descent gives the controller more time to converge laterally
5. **Hybrid approach for data collection** — Use CheatCode for the approach phase (getting close), then switch to human teleoperation for the final alignment + insertion

## What Remains Unsolved

- The SC connector **can** reach the SC port with the right general trajectory
- The lateral alignment at the point of insertion is consistently off by 8-20mm
- An I-only controller cannot achieve the sub-millimeter alignment needed for SC insertion
- A learned visuomotor policy (ACT, Diffusion Policy) should handle this — it implicitly learns visual-force alignment that a simple controller cannot express
- For data collection: human teleoperation or hybrid CheatCode-approach + human-insertion

## Files & Commands for Reproduction

**Launch SC port scene:**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true \
  cable_type:=sfp_sc_cable_reversed \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
  sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08 \
  sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
  sc_port_0_present:=true sc_port_0_translation:=-0.04
```

**Run CheatCode for SC insertion:**
```bash
source setup_dev.sh
ros2 run aic_model aic_model --ros-args \
  -p policy:=aic_example_policies.ros.CheatCode -p use_sim_time:=true

# In another terminal:
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate
ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
  "{task: {cable_name: 'cable_0', plug_name: 'sc_tip', \
  port_name: 'sc_port_base', target_module_name: 'sc_port_0'}}"
```

**Monitor insertion success:**
```bash
ros2 topic echo /scoring/insertion_event
```
