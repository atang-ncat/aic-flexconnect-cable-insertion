# SFP Data Collection Tracker

> **Copy the content below into a new GitHub Issue** at:
> https://github.com/atang-ncat/aic-flexconnect-cable-insertion/issues/new
>
> Title: `SFP Data Collection Tracker — 200 Episodes / 40 Configs`
>
> Then paste everything below the line.

---

## SFP Insertion Data Collection Progress

**Goal:** 200 episodes across 40 configurations.
**Rule:** 5 episodes per config. When you finish a config, check the box and fill in your name, episode range, and date.

**Format:** `- [x] **1.1** Slot 0, center — @yourname, ep 0–4, Apr 14`

> **Tip:** You can check/uncheck boxes directly in GitHub (just click them). Add your name + details by editing the issue body (pencil icon).

---

### Group 1 — NIC Rail Slot Variation (25 episodes)

All use default board pose (x=0.15, y=-0.2, yaw=3.1415), centered NIC translation.

- [ ] **1.1** Slot 0 — _name, ep range, date_
- [ ] **1.2** Slot 1 — _name, ep range, date_
- [ ] **1.3** Slot 2 — _name, ep range, date_
- [ ] **1.4** Slot 3 — _name, ep range, date_
- [ ] **1.5** Slot 4 — _name, ep range, date_

**Subtotal: 0 / 25**

---

### Group 2 — NIC Translation, Slot 0 (30 episodes)

- [ ] **2.1** trans=0.0 (center) — _name, ep range, date_
- [ ] **2.2** trans=+0.01 — _name, ep range, date_
- [ ] **2.3** trans=-0.01 — _name, ep range, date_
- [ ] **2.4** trans=+0.023 (eval max) — _name, ep range, date_
- [ ] **2.5** trans=-0.021 (eval min) — _name, ep range, date_
- [ ] **2.6** trans=+0.036 (beyond eval) — _name, ep range, date_

**Subtotal: 0 / 30**

---

### Group 3 — NIC Translation, Slots 1–2 (30 episodes)

- [ ] **3.1** Slot 1, trans=+0.015 — _name, ep range, date_
- [ ] **3.2** Slot 1, trans=-0.015 — _name, ep range, date_
- [ ] **3.3** Slot 1, trans=+0.023 — _name, ep range, date_
- [ ] **3.4** Slot 2, trans=0.0 (center) — _name, ep range, date_
- [ ] **3.5** Slot 2, trans=+0.02 — _name, ep range, date_
- [ ] **3.6** Slot 2, trans=-0.02 — _name, ep range, date_

**Subtotal: 0 / 30**

---

### Group 4 — Board Position (30 episodes)

All use Slot 0 with centered NIC translation.

- [ ] **4.1** x=0.15, y=-0.2 (default) — _name, ep range, date_
- [ ] **4.2** x=0.13, y=-0.2 (closer) — _name, ep range, date_
- [ ] **4.3** x=0.17, y=-0.2 (further) — _name, ep range, date_
- [ ] **4.4** x=0.15, y=-0.15 (shifted right) — _name, ep range, date_
- [ ] **4.5** x=0.15, y=-0.25 (shifted left) — _name, ep range, date_
- [ ] **4.6** x=0.17, y=-0.15 (further + right) — _name, ep range, date_

**Subtotal: 0 / 30**

---

### Group 5 — Board Yaw (25 episodes)

- [ ] **5.1** Slot 0, yaw=3.05 — _name, ep range, date_
- [ ] **5.2** Slot 0, yaw=3.24 — _name, ep range, date_
- [ ] **5.3** Slot 0, NIC trans=+0.02, yaw=3.0 — _name, ep range, date_
- [ ] **5.4** Slot 1, yaw=3.05 — _name, ep range, date_
- [ ] **5.5** Slot 1, NIC trans=-0.015, yaw=3.24 — _name, ep range, date_

**Subtotal: 0 / 25**

---

### Group 6 — NIC Yaw Offset (20 episodes)

- [ ] **6.1** Slot 0, NIC yaw=+0.05 — _name, ep range, date_
- [ ] **6.2** Slot 0, NIC yaw=-0.05 — _name, ep range, date_
- [ ] **6.3** Slot 0, NIC trans=+0.015, yaw=+0.08 — _name, ep range, date_
- [ ] **6.4** Slot 1, NIC yaw=+0.05 — _name, ep range, date_

**Subtotal: 0 / 20**

---

### Group 7 — Combined Variation (20 episodes)

Multiple dimensions varying at once — mimics real eval conditions.

- [ ] **7.1** Slot 0, NIC trans=+0.02, yaw=+0.05, board yaw=3.05 — _name, ep range, date_
- [ ] **7.2** Slot 1, NIC trans=-0.015, yaw=-0.05, board x=0.17, y=-0.15, yaw=3.24 — _name, ep range, date_
- [ ] **7.3** Slot 2, NIC trans=+0.01, board x=0.13, y=-0.25, yaw=3.1 — _name, ep range, date_
- [ ] **7.4** Slot 3, NIC trans=-0.01, yaw=+0.03, board x=0.16, y=-0.18, yaw=3.2 — _name, ep range, date_

**Subtotal: 0 / 20**

---

### Group 8 — With Distractors (20 episodes)

SC ports and other board components present as visual distractors.

- [ ] **8.1** Slot 0, NIC trans=0.005, SC port at -0.04 — _name, ep range, date_
- [ ] **8.2** Slot 0, NIC trans=+0.02, SC port at +0.03 — _name, ep range, date_
- [ ] **8.3** Slot 1, SC port at -0.02, board yaw=3.1 — _name, ep range, date_
- [ ] **8.4** Slot 0, NIC trans=-0.01, SC port at 0.0, board x=0.17, y=-0.15 — _name, ep range, date_

**Subtotal: 0 / 20**

---

## Overall Progress

| Group | Focus | Configs | Done | Total Episodes |
|-------|-------|---------|------|----------------|
| 1 | Rail slots | 5 | 0/5 | 0 / 25 |
| 2 | NIC translation (slot 0) | 6 | 0/6 | 0 / 30 |
| 3 | NIC translation (slots 1–2) | 6 | 0/6 | 0 / 30 |
| 4 | Board position | 6 | 0/6 | 0 / 30 |
| 5 | Board yaw | 5 | 0/5 | 0 / 25 |
| 6 | NIC yaw | 4 | 0/4 | 0 / 20 |
| 7 | Combined | 4 | 0/4 | 0 / 20 |
| 8 | Distractors | 4 | 0/4 | 0 / 20 |
| **Total** | | **40** | **0/40** | **0 / 200** |

## Priority Order

If time is limited, complete in this order:
1. **Groups 1 + 2** (55 ep) — absolute minimum viable dataset
2. **Group 3** (30 ep) — slot + translation combos
3. **Groups 4 + 5** (55 ep) — board pose variation
4. **Groups 6 + 7 + 8** (60 ep) — polish for robustness

## Quick Reference

- **Scene configs & launch commands:** see [`docs/sfp_scene_configs.md`](docs/sfp_scene_configs.md)
- **Teleoperation guide:** see [`docs/teleop_guide.md`](docs/teleop_guide.md)
- **Force monitor:** `python3 scripts/force_monitor.py`

## Notes

- When you finish a config, **check the box AND add your name + episode range + date**.
- Episode ranges should match what lerobot reports (e.g., ep 4–8 means episodes 4, 5, 6, 7, 8).
- If an episode was discarded, note it (e.g., "ep 4–9, discarded ep 7").
- Keep corrections/overshoots — see "What to Keep vs. Discard" in the teleop guide.
