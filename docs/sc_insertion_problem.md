# SC Port Insertion Problem — Technical Description

## Context

We are working on the AI for Industry Challenge (AIC) where a UR5e robot must insert fiber-optic cable connectors into ports on a task board. The `CheatCode` policy is a reference implementation that uses **ground-truth TF frames** (available only during training) and a simple **PI controller** to guide the connector into the port.

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

The SC port is the **small blue box** on the task board. When mounted on the board in the default configuration, its opening faces **upward**, making the insertion direction approximately **vertical (top-down)** — same general direction as SFP insertion into the NIC card.

## How CheatCode Works

CheatCode operates in two phases:

### Phase 1: Approach (pfrac 0.0 → 1.0)
- Linearly interpolates from current gripper position to the target port position + a 0.2m offset along the port's insertion axis (local Z)
- No integrator correction during this phase
- Takes about 15 seconds

### Phase 2: Insertion (z_offset 0.2 → -0.015)
- Steps z_offset down by 0.0005m per tick along the port's insertion axis
- A PI controller corrects lateral misalignment (perpendicular to the insertion axis)
- The PI controller measures `u_error` and `v_error` (lateral offsets in the port's local X and Y axes)
- Integrators accumulate the errors and apply corrections: `correction = i_gain * integrator`
- Integrators are clamped to `±_max_integrator_windup`

**Key PI parameters:**
- `i_gain = 0.15`
- `_max_integrator_windup = 0.05`
- **Max lateral correction = 0.15 × 0.05 = 0.0075m = 7.5mm**

## What Happens During SC Insertion

### Phase 1 (Approach) — Works Fine
The arm smoothly approaches the SC port. By pfrac=1.0, the connector is positioned ~0.2m above the port. The uv_error at the end of approach is typically:
```
uv_error: 0.0201 -0.00181   (20mm offset in u, 1.8mm in v)
```

### Phase 2 (Insertion) — Fails
As z_offset decreases from 0.2 toward -0.015:

1. **Persistent lateral offset:** The u_error stays stubbornly at ~0.008-0.02m (8-20mm) throughout the entire descent. The integrator saturates at 0.05 and the correction (7.5mm) is insufficient to close the gap.

2. **Partial insertion:** Around z_offset ~0.05-0.03, the connector reaches the SC port. Because of the lateral offset, **one side of the connector enters the port but the other side hangs outside**. The connector is essentially resting on the lip of the port at an angle.

3. **Contact destabilization:** As the connector makes contact with the port edge, contact forces push the plug sideways. The u/v errors start oscillating:
```
z_offset: 0.035   uv_error:  0.00637 -0.0034
z_offset: 0.031   uv_error:  0.00136 -0.00702
z_offset: 0.029   uv_error: -0.00101 -0.00753
z_offset: 0.025   uv_error: -0.00302 -0.00574
```

4. **Arm retracts:** When z_offset reaches -0.015, CheatCode declares "done" regardless of actual insertion success. The policy enters a "stabilization" phase where it stops sending commands, and the arm returns to a neutral position — **pulling the connector away from the port**.

5. **False positive:** `insert_cable()` returns `True` even though insertion did not occur. The `/scoring/insertion_event` topic does NOT fire — confirming no actual insertion happened.

## What We Tried

### Attempt 1: Generalized Insertion Axis
**Change:** Replaced hardcoded world-Z descent with port-orientation-aware insertion. Uses the port's quaternion to compute the actual insertion axis from its rotation matrix.

**Result:** The arm still descends vertically (because the SC port's local Z IS approximately vertical on the default task board). Same lateral offset problem persists.

### Attempt 2: Increased PI Gains
**Change:** `_max_integrator_windup: 0.05 → 0.15`, `i_gain: 0.15 → 0.25`  
**Max correction:** 0.25 × 0.15 = 0.0375m = 37.5mm (5× more authority)

**Result:** Made things worse. The increased correction authority caused the controller to **overshoot**. The connector now swings past the port, the integrator winds up the other way, swings back — creating a growing oscillation. The integrators bounce between +0.15 and -0.15:
```
z_offset: 0.006   uv_error:  0.0209 -0.0279   integrators: -0.128,  0.113
z_offset: 0.003   uv_error:  0.0275 -0.0343   integrators:  0.0288, -0.0852
z_offset: -0.005  uv_error: -0.0143  0.0155   integrators:  0.135,  -0.131
z_offset: -0.011  uv_error: -0.0332  0.0305   integrators: -0.15,    0.15   ← full saturation
```

The connector literally swings back and forth across the port like a pendulum. **Reverted gains back to original values.**

## Root Cause Analysis

The fundamental issue is **a mismatch between the PI controller's correction model and the SC port's geometry**:

1. **SFP port has physical guide rails** — even with 8mm lateral offset, the SFP port's tight metal slot physically nudges the connector into alignment during descent. The PI controller just needs to get "close enough" and the port geometry does the rest.

2. **SC port has no self-correction** — the SC port is an open rectangular cavity with wider tolerance. If the connector isn't precisely centered before insertion begins, there's nothing to guide it in. The connector either enters cleanly or catches on the lip.

3. **The PI controller can't converge** — with small gains, the correction authority is insufficient (7.5mm max vs 8-20mm error). With large gains, the controller oscillates. There's no stable gain setting that reliably achieves sub-millimeter alignment for SC.

4. **Possible TF frame offset** — the `sc_port_base` frame may not be perfectly centered on the port opening's physical center. Any systematic offset between the TF frame and the actual insertion point compounds the alignment problem.

## What Remains Unsolved

- The SC connector **can** reach the SC port with the right general trajectory
- The lateral alignment at the point of insertion is consistently off by 8-20mm
- A PI controller alone cannot achieve the sub-millimeter alignment needed for SC insertion
- A learned visuomotor policy (ACT, Diffusion Policy) should handle this — it can learn the subtle visual-force alignment that the PI controller cannot express
- Alternatively, improved teleoperation demos (human-collected) capture the correct insertion strategy

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
