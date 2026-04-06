# AIC Development: Status, Commands & Plan

## What You Have Ready

| Environment | Status | Purpose |
|-------------|--------|---------|
| **Gazebo** (distrobox `aic_eval`) | ✅ Working | Evaluation, testing policies, data collection |
| **Isaac Lab** (Docker `isaac-lab-base`) | ✅ Working | RL training, teleoperation, demo recording |
| **CheatCode policy** | ✅ Tested (~89-90 pts) | Reference for ground-truth insertion |
| **WaveArm policy** | ✅ Available | Minimal policy for setup verification |
| **RunACT policy** | ✅ Available | ACT imitation learning baseline |

---

## Command Reference

### Environment A: Gazebo

> [!IMPORTANT]
> Use `ros2 launch` for **development** (full scene, topics discoverable, no timeout).
> Use `/entrypoint.sh` only for **evaluation** (when a policy is ready).

#### Development Mode (recommended for daily work)
```bash
# Terminal 1: Enter distrobox and launch the full scene
distrobox enter aic_eval
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  spawn_task_board:=true \
  spawn_cable:=true \
  attach_cable_to_gripper:=true \
  nic_card_mount_0_present:=true \
  nic_card_mount_0_translation:=0.036 \
  sfp_mount_rail_0_present:=true \
  sfp_mount_rail_0_translation:=0.03 \
  sc_port_0_present:=true \
  sc_port_0_translation:=0.042

# Terminal 2: Teleoperate, inspect topics, etc.
distrobox enter aic_eval
source ~/ws_aic/setup_dev.sh              # One-time setup (system ROS + pixi packages)
ros2 topic list                           # Works! Full topic discovery
ros2 run aic_teleoperation cartesian_keyboard_teleop
ros2 run aic_teleoperation joint_keyboard_teleop
```

#### Evaluation Mode (scored trials — requires a running policy)
```bash
# Terminal 1: Launch sim + engine (waits for aic_model node, times out at ~60s/trial)
distrobox enter aic_eval
/entrypoint.sh ground_truth:=true start_aic_engine:=true

# Terminal 2: Run your policy IMMEDIATELY after engine starts looking
distrobox enter aic_eval
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args \
  -p policy:=aic_example_policies.ros.CheatCode \
  -p use_sim_time:=true
```

#### Example Policies
```bash
# Wave arm (basic test)
pixi run ros2 run aic_model aic_model --ros-args \
  -p policy:=aic_example_policies.ros.WaveArm \
  -p use_sim_time:=true

# CheatCode (ground truth insertion — needs ground_truth:=true)
pixi run ros2 run aic_model aic_model --ros-args \
  -p policy:=aic_example_policies.ros.CheatCode \
  -p use_sim_time:=true

# RunACT (learned policy)
pixi run ros2 run aic_model aic_model --ros-args \
  -p policy:=aic_example_policies.ros.RunACT \
  -p use_sim_time:=true
```

#### Data Collection & Training
```bash
# Collect training data with LeRobot
pixi run lerobot-record \
  --robot.type=aic_controller --robot.id=aic \
  --teleop.type=aic_keyboard_ee --teleop.id=aic \
  --robot.teleop_target_mode=cartesian --robot.teleop_frame_id=gripper/tcp \
  --dataset.repo_id=atang/aic_insertion_v1 \
  --dataset.single_task="Insert cable into port" \
  --dataset.push_to_hub=false --dataset.private=true \
  --play_sounds=false --display_data=true

# Train ACT from collected data
pixi run lerobot-train \
  --dataset.repo_id=atang/aic_insertion_v1 \
  --policy.type=act --policy.device=cuda \
  --output_dir=outputs/train/act_insertion_v1
```

#### Inspect Topics (from Terminal 2 after `source ~/ws_aic/setup_dev.sh`)
```bash
ros2 topic list                                   # All topics
ros2 topic echo /fts_broadcaster/wrench           # Force/torque
ros2 topic echo /joint_states                     # Joint positions
ros2 topic echo /aic_controller/controller_state  # TCP pose, error
```

---

### Environment B: Isaac Lab (RL Training)

```bash
# 1. Start & enter the container (from HOST, NOT distrobox)
cd ~/IsaacLab
./docker/container.py start base
./docker/container.py enter base

# 2. First-time setup each session (inside container — required every time you enter)
bash /workspace/isaaclab/aic/setup_isaac_aic.sh

# 3. Teleop (explore the scene with keyboard)
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/teleop.py \
  --task AIC-Task-v0 --num_envs 1 --teleop_device keyboard --enable_cameras

# 4. Record demonstrations for imitation learning
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/record_demos.py \
  --task AIC-Task-v0 --teleop_device keyboard --enable_cameras \
  --dataset_file ./datasets/dataset.hdf5 --num_demos 10

# 5. Replay recorded demos
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/replay_demos.py \
  --dataset_file ./datasets/dataset.hdf5

# 6. Train RL policy with PPO
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/train.py \
  --task AIC-Task-v0 --num_envs 1 --enable_cameras

# 7. Evaluate trained RL policy
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/play.py \
  --task AIC-Task-v0 --num_envs 1 --enable_cameras

# 8. Run a random agent (sanity check)
isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/random_agent.py \
  --task AIC-Task-v0 --num_envs 1 --enable_cameras

# 9. Stop the container (from HOST)
cd ~/IsaacLab
./docker/container.py stop base
```

**Isaac Lab Teleop Keys:** W/S (X), A/D (Y), Q/E (Z), Z/X (Rot X), T/G (Rot Y), C/V (Rot Z), K (gripper), R (reset)

---

## Development Plan

### Phase 1: Build Intuition (Now → Day 2)
- [ ] Teleoperate in **both** Gazebo and Isaac Lab — understand workspace limits, cable physics, insertion feel
- [ ] Watch force/torque readings during manual insertion attempts
- [ ] Run CheatCode with **different task board positions** to understand scoring variability

### Phase 2: Collect Data & Train Baseline (Day 3-7)
- [ ] Record **50-100 teleoperation demos** in Gazebo via LeRobot with varied board positions
- [ ] Train ACT policy on collected demos
- [ ] Evaluate ACT in Gazebo with engine — get a baseline score

### Phase 3: RL Training in Isaac Lab (Day 7-14)
- [ ] Start RL training with existing PPO + reward functions
- [ ] Tune rewards in `rewards.py` — add insertion depth reward
- [ ] Scale to multiple parallel envs (`--num_envs 64`) for faster training
- [ ] Transfer trained policy to Gazebo for evaluation

### Phase 4: Hybrid Policy & Polish (Day 14+)
- [ ] Combine learned approach with force-feedback for fine insertion
- [ ] Optimize for speed (≤5s = 12 bonus pts) and smoothness (6 pts)
- [ ] Ensure no collisions (-24 pts) or excessive force (-12 pts)
- [ ] Build submission container

### Scoring Reminder (100 pts max/trial)

| Component | Points | Key Threshold |
|-----------|--------|---------------|
| Correct insertion | **75** | Contact sensor verified |
| Partial/proximity | 0-50 | Even getting close scores |
| Duration | 0-12 | ≤5s = max |
| Smoothness | 0-6 | Low jerk = max |
| Efficiency | 0-6 | Direct path = max |
| Collision penalty | 0 to **-24** | Avoid at all costs |
| Force penalty | 0 to -12 | Keep <20N |

**Qualification deadline: May 15, 2026** — Evaluation: May 18-27
