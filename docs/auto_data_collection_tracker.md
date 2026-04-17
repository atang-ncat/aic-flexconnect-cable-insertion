# SFP Auto Data Collection Tracker

> **Copy the content below into a new GitHub Issue** at:
> https://github.com/atang-ncat/aic-flexconnect-cable-insertion/issues/new
>
> Title: `SFP Auto Data Collection Tracker — 120 Episodes / 40 Configs`
>
> Then paste everything below the line.

---

## SFP Insertion Auto Data Collection Progress

**Goal:** 120 episodes across **40 configurations** collected with `scripts/auto_collect.py`.
**Rule:** 3 episodes per config, **one episode per Gazebo launch** (see [`docs/auto_collection_guide.md`](docs/auto_collection_guide.md)). When a config is done, check the box and fill in your name, episode range, and date.

**Format:** `- [x] **A.1** Slot 0 — @yourname, ep 0–2, Apr 14`

> **Companion to the teleop tracker** ([#1](https://github.com/atang-ncat/aic-flexconnect-cable-insertion/issues/1)). This one is for the **automated** collector, which uses ground-truth TF frames and a P+I controller — different config priorities, different failure modes.

> **Tip:** Check/uncheck boxes directly in GitHub (just click them). Add details by editing the issue body (pencil icon).

> **If a config consistently fails** after 5 `--max-attempts` across 2–3 relaunches, mark it as `SKIPPED` and note the reason. We'll triage edge cases manually — don't keep burning attempts on a brittle config.

---

### Group A — NIC Rail Slot Variation (15 episodes)

All use default board pose (x=0.15, y=-0.2, yaw=3.1415), centered NIC translation.

- [ ] **A.1** Slot 0 — _name, ep range, date_
- [ ] **A.2** Slot 1 — _name, ep range, date_
- [ ] **A.3** Slot 2 — _name, ep range, date_
- [ ] **A.4** Slot 3 — _name, ep range, date_
- [ ] **A.5** Slot 4 — _name, ep range, date_

**Subtotal: 0 / 15**

---

### Group B — NIC Translation, Slot 0 (15 episodes)

Eval clamp range only — no beyond-eval values (auto script doesn't reliably reach ±0.036).

- [ ] **B.1** trans=0.0 (center) — _name, ep range, date_
- [ ] **B.2** trans=+0.01 — _name, ep range, date_
- [ ] **B.3** trans=-0.01 — _name, ep range, date_
- [ ] **B.4** trans=+0.023 (eval max) — _name, ep range, date_
- [ ] **B.5** trans=-0.021 (eval min) — _name, ep range, date_

**Subtotal: 0 / 15**

---

### Group C — NIC Translation, Slots 1–4 (18 episodes)

Cross-slot translation coverage.

- [ ] **C.1** Slot 1, trans=+0.015 — _name, ep range, date_
- [ ] **C.2** Slot 1, trans=-0.015 — _name, ep range, date_
- [ ] **C.3** Slot 2, trans=+0.015 — _name, ep range, date_
- [ ] **C.4** Slot 2, trans=-0.015 — _name, ep range, date_
- [ ] **C.5** Slot 3, trans=+0.01 — _name, ep range, date_
- [ ] **C.6** Slot 4, trans=-0.01 — _name, ep range, date_

**Subtotal: 0 / 18**

---

### Group D — Board Position (18 episodes)

- [ ] **D.1** x=0.15, y=-0.2 (default) — _name, ep range, date_
- [ ] **D.2** x=0.13, y=-0.2 (closer) — _name, ep range, date_
- [ ] **D.3** x=0.17, y=-0.2 (further) — _name, ep range, date_
- [ ] **D.4** x=0.15, y=-0.15 (right) — _name, ep range, date_
- [ ] **D.5** x=0.15, y=-0.25 (left) — _name, ep range, date_
- [ ] **D.6** Slot 1, x=0.16, y=-0.18 — _name, ep range, date_

**Subtotal: 0 / 18**

---

### Group E — Board Yaw (12 episodes)

Moderate yaw only (±0.1 rad from nominal 3.1415). Larger yaw breaks the orientation tracker in the auto script.

- [ ] **E.1** Slot 0, yaw=3.05 — _name, ep range, date_
- [ ] **E.2** Slot 0, yaw=3.24 — _name, ep range, date_
- [ ] **E.3** Slot 1, yaw=3.10 — _name, ep range, date_
- [ ] **E.4** Slot 2, yaw=3.15 — _name, ep range, date_

**Subtotal: 0 / 12**

---

### Group F — Combined, Moderate (15 episodes)

Multi-axis variation but kept within the P+I controller's reliable envelope.

- [ ] **F.1** Slot 0, trans=+0.01, yaw=3.10 — _name, ep range, date_
- [ ] **F.2** Slot 1, trans=-0.01, board x=0.16, y=-0.22 — _name, ep range, date_
- [ ] **F.3** Slot 2, trans=+0.015, board x=0.14, yaw=3.2 — _name, ep range, date_
- [ ] **F.4** Slot 3, board x=0.16, y=-0.18 — _name, ep range, date_
- [ ] **F.5** Slot 4, trans=-0.005, yaw=3.10 — _name, ep range, date_

**Subtotal: 0 / 15**

---

### Group G — Distractors / Eval-Like Clutter (27 episodes)

**The biggest group.** Visual distractors are free for the script (TF-driven) but critical for training a vision policy that generalizes to busy eval boards.

- [ ] **G.1** Slot 0, SC port 0 at -0.04 — _name, ep range, date_
- [ ] **G.2** Slot 0, trans=+0.01, SC port 0 at +0.03 — _name, ep range, date_
- [ ] **G.3** Slot 1, SC port 0 + SFP rail 0 — _name, ep range, date_
- [ ] **G.4** Slot 2, LC mount rail 0 — _name, ep range, date_
- [ ] **G.5** Slot 0, both SC ports — _name, ep range, date_
- [ ] **G.6** Slot 3, SC rail + SFP rail 1 — _name, ep range, date_
- [ ] **G.7** Slot 0, full clutter (SC×2 + SFP rail + LC rail) — _name, ep range, date_
- [ ] **G.8** Slot 1, full clutter + yaw=3.10 — _name, ep range, date_
- [ ] **G.9** Slot 2, trans=+0.01, full clutter + board shift — _name, ep range, date_

**Subtotal: 0 / 27**

---

## Overall Progress

| Group | Focus | Configs | Done | Total Episodes |
|-------|-------|---------|------|----------------|
| A | Rail slots | 5 | 0/5 | 0 / 15 |
| B | NIC translation (slot 0) | 5 | 0/5 | 0 / 15 |
| C | NIC translation (slots 1–4) | 6 | 0/6 | 0 / 18 |
| D | Board position | 6 | 0/6 | 0 / 18 |
| E | Board yaw (moderate) | 4 | 0/4 | 0 / 12 |
| F | Combined (moderate) | 5 | 0/5 | 0 / 15 |
| G | Distractors / clutter | 9 | 0/9 | 0 / 27 |
| **Total** | | **40** | **0/40** | **0 / 120** |

## Priority Order

If time is limited, complete in this order:
1. **Groups A + B** (30 ep) — all 5 slots + full eval translation range. Minimum viable.
2. **Group C** (18 ep) — cross-slot translation.
3. **Groups D + E** (30 ep) — board pose coverage.
4. **Group G** (27 ep) — distractors. **Critical** for vision policy generalization.
5. **Group F** (15 ep) — moderate combined variation.

## Recommended Session Plan

| Session | Configs | Episodes | Focus |
|---------|---------|----------|-------|
| 1 | A.1–A.5 | 15 | All 5 rail slots |
| 2 | B.1–B.5 | 15 | Slot 0 translation range |
| 3 | C.1–C.6 | 18 | Cross-slot translation |
| 4 | D.1–D.6 | 18 | Board position |
| 5 | E.1–E.4 + F.1–F.5 | 27 | Yaw + moderate combined |
| 6 | G.1–G.9 | 27 | Distractors (the big one) |

Each episode takes ~10 minutes end-to-end (Gazebo launch ~40s + insertion ~30s + save ~10s + kill + relaunch). Budget ~20 hours total split across sittings.

## Per-Launch Commands (quick reference)

**First episode of the dataset (NO `--resume`):**
```bash
cd /run/host/scratch2/atang/ws_aic/src/aic
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success
```

**Every subsequent episode (ADD `--resume`):**
```bash
pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
  --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
  --repo-id atang/aic_sfp_auto \
  --num-episodes 1 --max-attempts 5 --exit-on-success --resume
```

## Quick Reference

- **Scene configs & launch commands:** see [`docs/sfp_auto_scene_configs.md`](docs/sfp_auto_scene_configs.md)
- **Auto collection guide:** see [`docs/auto_collection_guide.md`](docs/auto_collection_guide.md)
- **Companion teleop tracker:** [#1](https://github.com/atang-ncat/aic-flexconnect-cable-insertion/issues/1)
- **Force monitor:** `python3 scripts/force_monitor.py`

## Notes

- When you finish a config, **check the box AND add your name + episode range + date**.
- Episode ranges should match what `auto_collect.py` reports on save (e.g., `ep 10–12` for 3 consecutive episodes).
- **Episodes may not be contiguous** if you interleave configs. That's fine — just log each config's episode indices.
- If an episode was auto-aborted (stuck / force / no insertion), it's not saved, so just count successful saves.
- If auto_collect.py fails on a config after several relaunches, add a `⚠️ SKIPPED` note with the failure symptom (stuck, force too high, no insertion detected).
