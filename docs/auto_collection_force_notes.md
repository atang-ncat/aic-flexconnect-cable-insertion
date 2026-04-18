# Auto-collection force monitoring — investigation notes

_Last updated: 2026-04-13._

This document captures what we know (and what we tried) about the
Tier-2 scoring force penalty that the automated collection pipeline
(`scripts/auto_collect.py`) currently incurs. It is intended as a
handoff so the investigation can resume on a fresh branch later
without re-doing the diagnostics.

## TL;DR

- The automated pipeline currently produces LeRobot-compatible
  episodes at ~50–80% success rate but **every saved episode gets a
  -12 Tier-2 penalty** (force above 20 N sustained for >1 s during
  the Approach phase).
- The tared-force calculation in `auto_collect.py` matches the
  scoring script and `scripts/force_monitor.py` — **the number we
  report is the right number**. There is no offset/tare bug.
- Two Cartesian-path attempts to lift the plug clear of the card
  before horizontal travel (world +z and port +z L-paths) **both
  regressed success to 0%**. Pure Cartesian path shaping is not the
  answer.
- The leading candidate fix is **a joint-space "go-home" motion
  before each Approach attempt** so the starting TCP pose is
  deterministic rather than whatever the previous retreat left us
  in. This has not been implemented yet.
- Until the fix lands, we are shipping the baseline (straight-line
  Approach) and accepting the -12/episode penalty, per explicit user
  decision.

## Scoring rule (recap)

Tier-2 force penalty, as implemented in the challenge scoring
script:

- `FORCE_THRESHOLD = 20.0 N` (tared)
- `DURATION_THRESHOLD = 1.0 s` (cumulative time above threshold)
- Exceeding the cumulative threshold → **-12 points / episode**.
- `FORCE_ABORT = 30.0 N` → immediate episode abort (safety).

The tared force we compute in both the monitor and auto-collect is:

```
tared_force = raw_wrench.force - fts_tare_offset
```

where `fts_tare_offset` is the zero sampled from
`/aic_controller/controller_state` just after the FT sensor is
re-tared.

## What the diagnostic wrapper showed

`scripts/collect_many.sh` now parses the
`SUCCESS! Saving ... max_force=X.XN, force_above_threshold=Y.YYs`
line from each episode's log and builds a per-run summary.

Representative numbers from the baseline (straight-line Approach)
run on `try-automated-dataset-recording`:

- Teleop reference (human): max_force ~28 N, dwell ≈ 0.08 s → **no
  penalty**.
- Auto baseline: max_force ~22–28 N, dwell ≈ 2.5–4.5 s per episode
  → **every saved episode penalized**.

So it isn't a peak-force problem (we spike about as high as teleop
does); it's a **sustained-force / drag** problem. During Approach
the plug gets pushed through the card volume while the gripper
transitions from "behind the card" to "in front of the port", and
spends several seconds with the plug body wedged against the card
edge.

## Why straight-line Approach produces the drag

Starting TCP pose after `setup_environment` / retreat is not
repeatable: the gripper can end up anywhere from "well above the
table" to "partway behind the card". The final Approach target is
at `z_offset = 0.22 m` in front of the port. A single linear
interpolation from start → target means that, whenever the start
point is already near or behind the card plane, the straight-line
motion **drags the plug through the card** during the ramp instead
of going around it.

## What we tried

### C-alt-1a: World +z L-path (lift_over_world_z_m)

- **Idea:** split Approach into two legs: first lift the TCP +25 cm
  in world z (gravity up), then translate to the Approach target.
- **Result:** 0/3 success.
- **Diagnostic finding:** `plug_axial` moved from -137.8 mm to
  -438.6 mm — the plug went *further behind the card*. Reason: in
  the AIC scene, the port frame is tilted such that the port's +z
  axis (insertion direction) points strongly *downward* in world
  coordinates. Lifting in world +z therefore pulls the plug away
  from the port face instead of clearing it.
- **Conclusion:** World-frame intuition does not match the scene.
  Don't use `lift_over_world_z_m`.

### C-alt-1b: Port +z L-path (lift_over_port_z_m)

- **Idea:** same two-leg split, but lift along the port's +z axis
  in world (i.e. along the insertion direction, which points
  "downward" in world). This should move the gripper *away from*
  the card face regardless of scene tilt.
- **Result:** 0/3 success. Failures were a mix of "off-limit
  collision", "alignment not converged", and "force too high"
  aborts.
- **Diagnostic finding:** a 0.25 m lift along port +z from a
  shallow start pose drove the gripper too low and into
  table/card geometry. Smaller lifts didn't clear the card; larger
  lifts hit workspace limits.
- **Conclusion:** A fixed-offset L-path cannot accommodate the
  variance in the post-retreat starting pose. If we ever revisit
  L-paths we need to **gate the lift on the current pose** — e.g.
  only lift if `plug_axial` is already behind the card plane, and
  size the lift from the measured offset rather than a constant.

### C-alt-2 (not attempted yet): Joint-space go-home before Approach

- **Idea:** before each Approach attempt, command the robot to a
  known joint-space home configuration (well above the card,
  repeatable, no cable interaction) using the controller's
  joint-space mode. Then start the Cartesian Approach from that
  deterministic pose. This turns "Approach from wherever we ended
  up" into "Approach from a known safe pose", which is the actual
  fix — the drag is a symptom of the unknown starting pose, not of
  the straight-line interpolation itself.
- **Status:** this is the recommended next step. Not yet
  implemented.

### C-alt-3 (current state): Accept the -12 penalty

- Ship the straight-line baseline that achieves ~50–80% success
  rate on its own.
- Proceed with multi-config collection.
- Accept the known -12/episode Tier-2 penalty until C-alt-2 lands.

## Current code state on `try-automated-dataset-recording`

- `scripts/auto_collect.py` — L-path infrastructure (`lift_over_*`
  params, waypoint precompute, two-leg interp) has been **removed**.
  `run_phase` is back to its single straight-line interpolation. A
  comment in the `run_phase` docstring points at this doc so the
  history is not lost.
- `scripts/collect_many.sh` — keeps the per-episode force
  diagnostics (max_force / dwell / would-pass-scoring) added during
  the investigation, plus the orphan-empty-dataset cleanup fix.

## When this work resumes

Recommended order of operations:

1. Branch off `try-automated-dataset-recording` (e.g.
   `try-automated-dataset-force-fix`).
2. Implement C-alt-2 (joint-space home before Approach) as the
   primary fix.
3. Re-run `scripts/collect_many.sh -n 5` and confirm dwell drops
   below ~0.5 s per episode without regressing success rate.
4. Only after C-alt-2 is proven, consider adding a *gated* L-path
   that triggers only when the measured starting pose warrants it.

