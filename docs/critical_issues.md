# Critical Issues Found in the AIC Pipeline

> **Date:** 2026-04-24
> **Status of report:** Issues confirmed by direct inspection of the workspace. Some are fixed; others are pending. See the "Status" column.

This is the reference index of every meaningful problem identified in our SFP/SC cable insertion pipeline as of late April 2026, why it matters for the May 15 qualification, and where we stand on each.

---

## Severity legend

| Tag | Meaning |
|---|---|
| `BLOCKING` | Without this, we cannot produce a usable submission. |
| `MAJOR` | Will cost a large fraction of competition points. |
| `MINOR` | Quality-of-life or marginal points. |

---

## 1. We have never measured insertion success in Gazebo `BLOCKING`

**What's wrong.** The training stack has been tracking `val/motion_accuracy` (how often the policy guesses the right keyboard button on held-out frames). Our best model, v6b, plateaus at 0.52. That number does not measure whether the cable inserts. Insertion success is the only metric that matters (75 of 100 points per trial), and we have zero data points on it.

**Why it matters.** Every "this version is better" claim across v2 → v8 is built on a proxy that has no validated relationship to insertion success. We could already have a working policy and not know it, or we could have spent months optimizing the wrong number.

**Status.** Pending. The blocker is `RunACT.py` — see issue #6.

---

## 2. The lerobot driver never recorded the F/T sensor `BLOCKING for SC, MAJOR for SFP`

**What's wrong.** The lerobot recording driver (`aic_robot_aic_controller.py`) subscribed only to `controller_state` (TCP pose, velocity, error) and `joint_states`. It never subscribed to `/fts_broadcaster/wrench`. The 32-D `Observation.msg` available at evaluation time exposes `wrist_wrench`, but our entire 201-episode training dataset has zero force/torque columns.

**Why it matters.** Cable insertion is a contact task. The F/T spike from the connector touching the port lip is the highest-value single signal for triggering corrections during the final mm of descent. Without it the policy cannot learn force-aware behavior, which is exactly what SC needs (no guide rails, oscillates freely on contact).

**Status.** **FIXED in driver, requires re-collection.** The driver now subscribes to `/fts_broadcaster/wrench`, exposes 6 wrench fields in `ObservationState`, and subtracts the tare from `controller_state.fts_tare_offset` so what reaches the dataset is real contact force. New recording sessions automatically include F/T.

---

## 3. The action space is keyboard-quantized to {-0.1, 0, +0.1} `MAJOR`

**What's wrong.** The keyboard teleop driver published step-function velocities. Every recorded action is exactly one of three values per dimension. ~73% of frames are all-zero. Sub-mm alignment requires sub-mm velocity resolution; 0.1 m/s × 1/30 s = 3.3 mm per frame is far too coarse.

**Why it matters.** A model trained on these actions emits the same step functions at inference. That causes oscillation around the target, jerk-penalty point loss (up to -6 per trial), and contact force spikes. The discrete classification head in v8 was a clever workaround for the *statistics* of this data but not for the *physics* — the executed trajectories still jerk.

**Status.** **FIXED in teleop driver.** EMA smoothing (alpha=0.5) baked into `cartesian_keyboard_teleop.py`. Operator presses keys normally; published velocity ramps over ~2-3 ticks (~80 ms). Recorded actions become continuous values like 0.05, 0.075, 0.087 rather than {-0.1, 0, +0.1}. New recording sessions automatically have continuous actions; no SavGol post-pass needed.

---

## 4. Zero SC port demonstrations `BLOCKING for Trial 3`

**What's wrong.** We have 201 SFP episodes and **0 SC episodes**. CheatCode itself fails on SC (documented in `sc_insertion_problem.md` — pure I-controller oscillates because the SC port has no guide rails to absorb overshoot).

**Why it matters.** Trial 3 is worth 75 points for insertion alone. With no SC training data, the policy will make SFP-style attempts on the SC port and almost certainly fail. Even a partially-successful SFP-only policy caps at ~150 of the 225 insertion points across the three trials.

**Status.** Pending. Requires (a) human teleop SC demos using the now-smoothed keyboard driver, and (b) a fix for CheatCode SC if we want automated SC demos later.

---

## 5. No task conditioning for SFP vs SC `MAJOR`

**What's wrong.** A single submitted policy must handle both plug types. The `Task` message specifies which port to target. Our policy receives no encoding of this — it sees only images + state and predicts actions, with no way to know whether it should head for the green NIC card or the blue SC box.

**Why it matters.** Even if we collect SC data and merge it with SFP data, the policy has no signal to choose between the two behaviors. It will average the SFP-style and SC-style trajectories from the dataset, producing something that fits neither.

**Status.** Pending. Two options once we have SC data: (a) tag each episode with `--dataset.single_task="insert SFP"` vs `"insert SC"` so lerobot writes a `task_index` column, and add a learned task-token to the ACT encoder; or (b) add an explicit one-hot field to `ObservationState`.

---

## 6. RunACT.py cannot load any v5+ checkpoint `BLOCKING`

**What's wrong.** The inference policy at `src/aic/aic_example_policies/aic_example_policies/ros/RunACT.py` is hard-coded to a v2-era checkpoint and assembles a 26-D state vector. Every v5/v6/v6b/v8 model trains with a 14-D filtered state (drops `tcp_velocity.*` and `tcp_error.*`). Loading any of those checkpoints into the current `RunACT.py` would crash at the state-projection layer.

**Why it matters.** This is what gates issue #1. Until `RunACT.py` knows how to apply the state filter, decode the discrete classification head, and (eventually) consume F/T input, no recent model can be Gazebo-evaluated.

**Status.** Pending. Next concrete code task.

---

## 7. Temporal ensembling is disabled `MINOR-MAJOR`

**What's wrong.** Every config sets `temporal_ensemble_coeff: null`. The ACT paper documents temporal ensembling as a high-value default — it blends predictions from overlapping past chunks, smoothing the executed trajectory. Without it, sharp velocity discontinuities appear at every chunk boundary (every 100 frames = 3.3 s).

**Why it matters.** Discontinuities cost smoothness points and cause force spikes during contact phases. Temporal ensembling is a config-only change; cost is zero.

**Status.** **FIXED in v9 config.** `temporal_ensemble_coeff: 0.01` (paper default), paired with `chunk_size: 50` for reactive re-planning. Future runs inherit this.

---

## 8. Chunk size of 100 frames (3.3 s) is too long `MINOR-MAJOR`

**What's wrong.** ACT commits to 100 future actions from a single observation. If the gripper is 2 mm off when the chunk starts, the next 3.3 s execute that mistake with no correction. Cable insertion needs reactive correction at the contact boundary.

**Why it matters.** With chunk_size=100 the policy is essentially open-loop within each chunk. With temporal ensembling enabled and chunk_size=50, the policy effectively re-plans every frame.

**Status.** **FIXED in v9 config.** `chunk_size: 50`. Future runs inherit this.

---

## 9. Automated CheatCode dataset is nearly empty (and the script had real bugs) `MAJOR`

**What's wrong.** `auto_collect.py` produced only 12 SFP + 5 SC-diag episodes before stalling. On inspection the script had several silent failure modes: wall-clock `time.sleep(1/fps)` on a sim-time node (frame-rate drift), image decoding that assumed naked RGB bytes regardless of `msg.encoding` (would silently write BGR as RGB on Gazebo configs that publish bgr8), recording zero-action frames on transient TF failures (pollutes training distribution), zero-pixel frames when a camera stalled (silent dataset corruption), and 26-D state output incompatible with the new F/T-enabled teleop schema.

**Why it matters.** CheatCode demos are smooth proportional-controller output — strictly cleaner than keyboard even with EMA. More demos per operator-hour, diverse board poses. But only if the script actually produces valid data.

**Status.** **FIXED.** `auto_collect.py` rewrite:
- Sim-time control loop (`create_rate` + `rate.sleep`) — frame rate locks to `--fps` regardless of Gazebo speed.
- Encoding-aware image decoding (rgb8/bgr8/rgba8/bgra8 via `msg.encoding`).
- 32-D state with tared wrench, matching the new teleop schema.
- Transient TF failures hold last-action instead of recording a zero frame.
- Camera stalls drop the frame instead of writing black pixels.
- New `--plug-type {sfp,sc}` selects gains + frames + task string per profile.
- New `--dry-run` runs one episode without writing a dataset — catches bringup/TF/camera/F/T problems in 30 s before committing to a long session.
- Pre-flight F/T publishing + tare sanity checks at startup with actionable error messages.

See `docs/auto_collection_guide.md` for updated commands.

---

## 10. The smoothed dataset only covers 151 of 201 episodes `MINOR (now obsolete)`

**What's wrong.** `teleop-dataset-smoothed/` was created by running `scripts/smooth_actions.py` on the original 151-episode dataset. The 50 additional episodes collected later (bringing the total to 201) were never smoothed. v6/v6b/v8 had to use the raw 201-episode set to access all the data.

**Why it matters.** Mostly historical now — with EMA smoothing baked into the teleop driver, future recordings are smooth at source. The SavGol post-pass only matters if we want to retroactively smooth the existing 201 raw episodes.

**Status.** Deprioritized. Run `smooth_actions.py` on the full 201 only if you want one more comparison data point against v9-on-fresh-data.

---

## What changed in this fix-up cycle

| File | Change | Purpose |
|---|---|---|
| `src/aic/aic_utils/lerobot_robot_aic/lerobot_robot_aic/aic_robot_aic_controller.py` | Subscribe to `/fts_broadcaster/wrench`; 6 wrench fields in `ObservationState`; tare subtraction from `controller_state.fts_tare_offset` in `get_observation()` | Issue #2 |
| `src/aic/aic_utils/aic_teleoperation/aic_teleoperation/cartesian_keyboard_teleop.py` | EMA smoothing (alpha=0.5) on the published twist with a small deadband | Issue #3 |
| `configs/act_sfp_v9.yaml` | New training config: smoothed dataset, F/T-aware state filter, temporal ensemble, chunk_size=50, continuous regression | Issues #7, #8 |
| `docs/teleop_recording_steps.md` | Step-by-step recording procedure for the new driver | Operator-facing |

---

## Order of operations for recovering the timeline

1. **Re-record a small F/T-enabled SFP pilot dataset** (30–50 episodes) using the new driver. ~1 day.
2. **Fix `RunACT.py`** so it can load v5+ checkpoints with the state filter and (optionally) F/T. ~1 day.
3. **Train v9 on the existing 201 episodes** (no F/T, but proves the pipeline is correct end-to-end), Gazebo-eval it. Establish the first real insertion-success number. ~1 day.
4. **Train v10 on the F/T pilot dataset.** Compare v9 vs v10 in Gazebo to measure F/T's contribution. ~1 day.
5. **Decide based on v10 result:** if F/T is helping, scale up to 100+ F/T episodes; if not, focus elsewhere.
6. **Collect SC demos with the same setup,** add task tagging via `--dataset.single_task`, train multi-task v11.
7. **Submission Docker dry-run** by week of May 8 latest. The May 15 deadline is non-negotiable.

This sequencing keeps us anchored to insertion-success measurements at every step. The proxy-metric era is over.
