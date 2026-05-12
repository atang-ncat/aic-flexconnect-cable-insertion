# Gamepad v2 — SFP vs SC dataset roots

Use **separate** LeRobot dataset directories so `scripts/build_combined_v2.py` can assign the correct `task_index` per source (SFP = 0, SC = 1). Do **not** record SC demos under `.../gamepad-v2/sfp`.

| Task | `dataset.root` (host) | `dataset.root` (aic_eval container) |
|------|------------------------|----------------------------------------|
| **SFP** | `datasets/teleop-dataset-gamepad-v2/sfp` | `/run/host/scratch2/atang/ws_aic/datasets/teleop-dataset-gamepad-v2/sfp` |
| **SC** | `datasets/teleop-dataset-gamepad-v2/sc` | `/run/host/scratch2/atang/ws_aic/datasets/teleop-dataset-gamepad-v2/sc` |

## Gazebo

- **SFP:** this doc and [`sfp_scene_configs.md`](sfp_scene_configs.md).
- **SC:** **[`sc_scene_configs.md`](sc_scene_configs.md)** — reversed cable + `sc_port_*` / rail / board pose sweeps.

## Record (helpers from repo root)

```bash
# SFP — first session: no --resume. Later sessions: add --resume=true
bash /scratch2/atang/ws_aic/scripts/record_gamepad_v2_sfp.sh --resume=true
```

```bash
# SC — same resume rule
bash /scratch2/atang/ws_aic/scripts/record_gamepad_v2_sc.sh --resume=true
```

## `dataset.single_task` strings (must match merge script)

These match `DEFAULT_TASKS` in `scripts/build_combined_v2.py`:

- **SFP:** `Insert SFP connector into SFP port on NIC card`
- **SC:** `Insert SC connector into SC port`

## Merge into training dataset

After you have episodes under both v2 roots (and the v1 sources you still use), rebuild `datasets/teleop-dataset-combined-v2`:

```bash
cd /scratch2/atang/ws_aic/src/aic
pixi run python /scratch2/atang/ws_aic/scripts/build_combined_v2.py \
  --out /scratch2/atang/ws_aic/datasets/teleop-dataset-combined-v2 \
  --drop-failed
```

Remove the old `teleop-dataset-combined-v2` directory first if it already exists.

**Training** still points at `datasets/teleop-dataset-combined-v2` (e.g. `configs/act_sfp_v18.yaml`); no separate ACT config is required just for SC recording.

## One-line `pixi` (SC, copy-paste)

```bash
cd /run/host/scratch2/atang/ws_aic/src/aic && pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_gamepad_ee --teleop.id=aic \
  --teleop.stick_deadzone=0.02 --teleop.stick_expo=1.0 \
  --teleop.trigger_deadzone=0.01 --teleop.trigger_expo=1.0 \
  --teleop.low_command_scaling=0.04 \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=base_link \
  --dataset.repo_id=atang/aic_sc_demos_gamepad_v2 \
  --dataset.root=/run/host/scratch2/atang/ws_aic/datasets/teleop-dataset-gamepad-v2/sc \
  --dataset.single_task="Insert SC connector into SC port" \
  --dataset.push_to_hub=false --dataset.private=true \
  --play_sounds=false --display_data=true --resume=true
```

First time this root is empty, **omit** `--resume=true` once so LeRobot can create the dataset.
