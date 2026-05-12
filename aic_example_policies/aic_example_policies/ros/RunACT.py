#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Callable, Dict, Any, List
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Wrench

# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------------------------------------------------------
        # 1. Configuration & Weights Loading
        # -------------------------------------------------------------------------
        # v13: aggressive action weighting (scale=0.02, floor=0.05) + head trim.
        # BEST SCORE: 68.02. All v14 variants score in same ~62-68 band.
        # v18: task-conditioned ACT on combined gamepad SFP+SC dataset.
        #      State is 27 + num_tasks (=29 for SFP|SC).  We auto-detect
        #      task conditioning from the saved state_dim below.
        # Override with env AIC_ACT_POLICY_PATH=/path/to/checkpoints/best
        _default_ckpt = (
            "/scratch2/atang/ws_aic/outputs/act_sfp/"
            "v18_combined_taskcond/checkpoints/best"
        )
        policy_path = Path(os.environ.get("AIC_ACT_POLICY_PATH", _default_ckpt))
        # If running inside the docker container, /scratch2 is mapped to /run/host/scratch2
        if not policy_path.exists() and Path("/run/host" + str(policy_path)).exists():
            policy_path = Path("/run/host" + str(policy_path))

        # Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)

        # ── Deploy-time action-selection overrides ────────────────────────
        # AIC_ACT_ENSEMBLE_COEFF: override temporal_ensemble_coeff.
        #   Default = use trained value (0.01 for v13).
        #   "none"  = disable temporal ensemble, use n_action_steps queue.
        #   "0.1"   = less smoothing (new chunk gets ~10% weight).
        #   "0.5"   = much less smoothing.
        ensemble_env = os.environ.get("AIC_ACT_ENSEMBLE_COEFF", "")
        if ensemble_env.lower() == "none":
            config.temporal_ensemble_coeff = None
            self.get_logger().info(
                "AIC_ACT_ENSEMBLE_COEFF=none → temporal ensemble DISABLED"
            )
        elif ensemble_env:
            config.temporal_ensemble_coeff = float(ensemble_env)
            self.get_logger().info(
                f"AIC_ACT_ENSEMBLE_COEFF={config.temporal_ensemble_coeff}"
            )

        # AIC_ACT_N_STEPS: how many actions from each chunk to execute
        #   (only used when temporal ensemble is disabled).
        #   Default = 1 (re-predict every step). Try 5-10 for more open-loop.
        nsteps_env = os.environ.get("AIC_ACT_N_STEPS", "")
        if nsteps_env:
            config.n_action_steps = int(nsteps_env)
            self.get_logger().info(
                f"AIC_ACT_N_STEPS={config.n_action_steps}"
            )

        # Load Policy Architecture & Weights
        self.policy = ACTPolicy(config)
        model_weights_path = policy_path / "model.safetensors"
        self.policy.load_state_dict(load_file(model_weights_path))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

        # -------------------------------------------------------------------------
        # 2. Normalization Stats Loading
        # -------------------------------------------------------------------------
        norm_candidates = sorted(
            policy_path.glob("*normalizer_processor.safetensors")
        )
        if not norm_candidates:
            raise FileNotFoundError(
                f"No *normalizer_processor.safetensors under {policy_path}"
            )
        # Prefer the preprocessor normalizer over the postprocessor unnormalizer
        pre = [p for p in norm_candidates if "preprocessor" in p.name]
        stats_path = pre[0] if pre else norm_candidates[0]
        if len(norm_candidates) > 1:
            self.get_logger().info(
                f"Found {len(norm_candidates)} normalizer files; using {stats_path.name}"
            )
        stats = load_file(stats_path)

        # Helper to extract and shape stats for broadcasting
        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        # NOTE: always apply MEAN_STD from the saved normalizer, even when the
        # config says VISUAL: IDENTITY.  Empirically verified: the LeRobot
        # normalizer module applies stored stats during training regardless of
        # the IDENTITY flag, so deployment must match.  v13 with IDENTITY at
        # deploy scored 3.0; with MEAN_STD it scored 68.02.

        # Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }
        print(f"Image stats: {self.img_stats}")

        # Robot state stats — dim matches training (26 legacy, 27 for v10_ft: no
        # tcp_vel/tcp_error; includes joint velocities + tared wrench).
        # v18: state_dim = base + num_tasks (e.g. 29 for 27-D base + 2 one-hot).
        self.state_dim = int(stats["observation.state.mean"].shape[0])
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        print(f"Robot state dim={self.state_mean.shape[-1]} mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

        # ── Task-conditioned model auto-detection (v18) ─────────────────
        # The combined-v2 dataset uses 2 tasks (0=sfp, 1=sc) and base state 27.
        # If the trained checkpoint's state_dim matches 27 + num_tasks for some
        # small num_tasks, assume task conditioning is on.  Override with env
        # AIC_ACT_NUM_TASKS to be explicit.
        env_num_tasks = os.environ.get("AIC_ACT_NUM_TASKS", "")
        if env_num_tasks:
            self.num_tasks = int(env_num_tasks)
            self.base_state_dim = self.state_dim - self.num_tasks
        elif self.state_dim in (28, 29, 30):  # 27 + 1/2/3 tasks
            self.num_tasks = self.state_dim - 27
            self.base_state_dim = 27
        else:
            self.num_tasks = 0
            self.base_state_dim = self.state_dim
        if self.num_tasks > 0:
            self.get_logger().info(
                f"Task conditioning ENABLED: base state {self.base_state_dim} + "
                f"{self.num_tasks}-D one-hot. Default task index = 0 (SFP)."
            )
        # Will be set per-call from Task.plug_type in insert_cable().
        self.current_task_index = 0

        # Action Stats (1, 7) - Used for Un-normalization
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))
        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        # Config
        self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

        self.get_logger().info("Normalization statistics loaded successfully.")

    @staticmethod
    def _ros_image_to_rgb(raw_img) -> np.ndarray:
        """Decode a sensor_msgs/Image into an HxWx3 uint8 RGB ndarray.

        We avoid ``cv_bridge`` here because the system-ROS build links against
        NumPy 1.x while the pixi env ships NumPy >= 2 (Pixi env Python is what
        runs this code), which makes any ``cv_bridge`` import crash with a
        ``_multiarray_umath`` ABI error. The ``sensor_msgs/Image`` wire format
        is trivial to parse manually for the small set of encodings Gazebo
        publishes, so we just do that.
        """
        enc = raw_img.encoding.lower()
        channels_map = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }
        if enc not in channels_map:
            raise ValueError(f"Unsupported image encoding: {raw_img.encoding!r}")
        c = channels_map[enc]

        h, w, step = raw_img.height, raw_img.width, raw_img.step
        raw = np.frombuffer(raw_img.data, dtype=np.uint8)
        if step == w * c:
            arr = raw.reshape(h, w, c) if c > 1 else raw.reshape(h, w)
        else:
            arr = raw.reshape(h, step)[:, : w * c]
            arr = arr.reshape(h, w, c) if c > 1 else arr.reshape(h, w)

        if enc == "rgb8":
            return arr.copy()
        if enc == "bgr8":
            return arr[..., ::-1].copy()
        if enc == "rgba8":
            return arr[..., :3].copy()
        if enc == "bgra8":
            return arr[..., 2::-1].copy()
        # mono8
        return np.stack([arr, arr, arr], axis=-1)

    @staticmethod
    def _img_to_tensor(
        raw_img,
        device: torch.device,
        scale: float,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Converts ROS Image -> Resized -> Permuted -> Normalized Tensor."""
        # Always decode to rgb8 since the LeRobot dataset expects standard RGB tensors
        img_np = RunACT._ros_image_to_rgb(raw_img)

        # 2. Resize
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        # 3. To Tensor -> Permute (HWC -> CHW) -> Float -> Div(255) -> Batch Dim
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

        # 4. Normalize (Apply Mean/Std)
        # Formula: (x - mean) / std
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        """Convert ROS Observation message into dictionary of normalized tensors."""

        # --- Process Cameras ---
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        # --- Process Robot State ---
        # Order must match teleop / LeRobot dataset (see aic_robot_aic_controller.py).
        tcp_pose = obs_msg.controller_state.tcp_pose

        # The ACT dataset was collected using /joint_states natively, which alphabetizes joints.
        # But aic_adapter explicitly reorders Observation.joint_states to structural order.
        # We must re-alphabetize here to guarantee correct neural net feature matching.
        target_joint_names = [
            "elbow_joint",
            "gripper",  # aic_adapter renamed it from 'gripper/left_finger_joint' to 'gripper'
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        jm = obs_msg.joint_states
        joint_dict = dict(zip(jm.name, jm.position))
        vel_seq = list(jm.velocity) if jm.velocity else []
        vel_dict = {
            n: float(vel_seq[i]) if i < len(vel_seq) else 0.0
            for i, n in enumerate(jm.name)
        }

        # Recover the raw prismatic value (aic_adapter divided it by 2.0)
        joint_dict["gripper"] *= 2.0

        ordered_joints = [joint_dict[name] for name in target_joint_names]
        ordered_vels = [float(vel_dict.get(name, 0.0)) for name in target_joint_names]

        # Tared wrench (same convention as dataset: raw wrist - controller tare)
        _w = obs_msg.wrist_wrench.wrench
        _t = obs_msg.controller_state.fts_tare_offset.wrench
        wrench = (
            _w.force.x - _t.force.x,
            _w.force.y - _t.force.y,
            _w.force.z - _t.force.z,
            _w.torque.x - _t.torque.x,
            _w.torque.y - _t.torque.y,
            _w.torque.z - _t.torque.z,
        )

        # Use base_state_dim (without task one-hot) to choose the state
        # construction branch.  Task one-hot is appended after, below.
        base_dim = self.base_state_dim

        if base_dim == 27:
            # v10_ft / v13 / v18: train_act drops tcp_velocity.* and tcp_error.*.
            state_np = np.array(
                [
                    tcp_pose.position.x,
                    tcp_pose.position.y,
                    tcp_pose.position.z,
                    tcp_pose.orientation.x,
                    tcp_pose.orientation.y,
                    tcp_pose.orientation.z,
                    tcp_pose.orientation.w,
                    *ordered_joints,
                    *ordered_vels,
                    *wrench,
                ],
                dtype=np.float32,
            )
        elif base_dim == 26:
            tcp_vel = obs_msg.controller_state.tcp_velocity
            state_np = np.array(
                [
                    tcp_pose.position.x,
                    tcp_pose.position.y,
                    tcp_pose.position.z,
                    tcp_pose.orientation.x,
                    tcp_pose.orientation.y,
                    tcp_pose.orientation.z,
                    tcp_pose.orientation.w,
                    tcp_vel.linear.x,
                    tcp_vel.linear.y,
                    tcp_vel.linear.z,
                    tcp_vel.angular.x,
                    tcp_vel.angular.y,
                    tcp_vel.angular.z,
                    *obs_msg.controller_state.tcp_error,
                    *ordered_joints,
                ],
                dtype=np.float32,
            )
        elif base_dim == 39:
            # Full 39-D state: tcp_pose(7) + tcp_velocity(6) + tcp_error(6) +
            # joint_positions(7) + joint_velocities(7) + wrench(6).
            # Used by stock lerobot checkpoints (v18, v19) trained on
            # unfiltered observation.state.
            tcp_vel = obs_msg.controller_state.tcp_velocity
            tcp_err = list(obs_msg.controller_state.tcp_error)
            state_np = np.array(
                [
                    tcp_pose.position.x,
                    tcp_pose.position.y,
                    tcp_pose.position.z,
                    tcp_pose.orientation.x,
                    tcp_pose.orientation.y,
                    tcp_pose.orientation.z,
                    tcp_pose.orientation.w,
                    tcp_vel.linear.x,
                    tcp_vel.linear.y,
                    tcp_vel.linear.z,
                    tcp_vel.angular.x,
                    tcp_vel.angular.y,
                    tcp_vel.angular.z,
                    *tcp_err,             # 6 dims
                    *ordered_joints,      # 7 dims
                    *ordered_vels,        # 7 dims
                    *wrench,              # 6 dims
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"Unsupported observation.state base dim {base_dim} in checkpoint stats; "
                "expected 26 (legacy), 27 (v10_ft / v18), or 39 (full/stock-lerobot)."
            )

        # v18: append task one-hot if the policy was trained task-conditioned.
        if self.num_tasks > 0:
            one_hot = np.zeros(self.num_tasks, dtype=np.float32)
            t_idx = max(0, min(self.current_task_index, self.num_tasks - 1))
            one_hot[t_idx] = 1.0
            state_np = np.concatenate([state_np, one_hot], axis=0).astype(np.float32)
        if state_np.shape[0] != self.state_dim:
            raise RuntimeError(
                f"Built state of dim {state_np.shape[0]} but checkpoint expects "
                f"{self.state_dim} (base={self.base_state_dim} + tasks={self.num_tasks})."
            )

        # Normalize State
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std

        return obs

    def _debug_log_observation(
        self,
        obs_msg: Observation,
        obs_tensors: Dict[str, torch.Tensor],
        normalized_action: torch.Tensor,
        unnormalized_action: torch.Tensor,
    ) -> None:
        """One-shot diagnostic dump of inputs and outputs.

        Prints, for the first iteration of an insert_cable rollout:

          * Per-camera raw RGB stats (in [0,1]) so we can compare to the
            training-time camera stats logged at policy load.
          * Per-camera post-normalization tensor stats (should be roughly
            zero-mean, unit-std-ish if the input distribution matches
            training; values like |mean| > 5 indicate severe distribution
            shift that will dominate the policy output).
          * Raw state vector and its z-score against the saved training
            stats (max |z| > 3 means at least one state component is OOD).
          * Normalized action chunk[0] and the un-normalized command.
        """
        log = self.get_logger()
        log.info("=" * 70)
        log.info("RunACT diagnostic dump (iter=0)")
        log.info("=" * 70)

        cams = [
            ("left", obs_msg.left_image, "observation.images.left_camera"),
            ("center", obs_msg.center_image, "observation.images.center_camera"),
            ("right", obs_msg.right_image, "observation.images.right_camera"),
        ]
        for name, raw, key in cams:
            rgb = self._ros_image_to_rgb(raw).astype(np.float32) / 255.0
            log.info(
                f"  {name:6s} RAW : enc={raw.encoding} "
                f"shape={rgb.shape} "
                f"mean=[{rgb[..., 0].mean():.3f},{rgb[..., 1].mean():.3f},{rgb[..., 2].mean():.3f}] "
                f"std=[{rgb[..., 0].std():.3f},{rgb[..., 1].std():.3f},{rgb[..., 2].std():.3f}] "
                f"min={rgb.min():.3f} max={rgb.max():.3f}"
            )
            t = obs_tensors[key][0].detach().cpu()  # (3, H, W)
            log.info(
                f"  {name:6s} NORM: shape={tuple(t.shape)} "
                f"mean=[{t[0].mean():.2f},{t[1].mean():.2f},{t[2].mean():.2f}] "
                f"std=[{t[0].std():.2f},{t[1].std():.2f},{t[2].std():.2f}] "
                f"min={t.min():.2f} max={t.max():.2f}"
            )

        state_norm = obs_tensors["observation.state"][0].detach().cpu().numpy()
        state_mean = self.state_mean[0].detach().cpu().numpy()
        state_std = self.state_std[0].detach().cpu().numpy()
        state_raw = state_norm * state_std + state_mean
        np_opts = dict(precision=3, suppress_small=True, max_line_width=200)
        d = state_raw.shape[0]
        log.info(f"  state RAW  ({d}): {np.array2string(state_raw, **np_opts)}")
        log.info(f"  state NORM ({d}): {np.array2string(state_norm, **np_opts)}")
        log.info(
            f"  state max |z|: {np.abs(state_norm).max():.2f}  "
            f"(>3 => at least one state component is OOD)"
        )

        na = normalized_action[0].detach().cpu().numpy()
        ua = unnormalized_action[0].detach().cpu().numpy()
        log.info(f"  action NORM (6): {np.array2string(na, **np_opts)}")
        log.info(f"  action UNNORM(6): {np.array2string(ua, **np_opts)}")
        log.info(
            f"  |UNNORM action|_inf = {np.max(np.abs(ua)):.2e} "
            f"({'~zero (model is predicting idle)' if np.max(np.abs(ua)) < 1e-3 else 'non-trivial motion'})"
        )
        log.info("=" * 70)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

        # ── Pick the task one-hot from the Task message (v18) ───────────
        # Map the dispatcher's plug_type/port_type strings to our training
        # task ids: 0 = SFP, 1 = SC.  Defaults to 0 (SFP) on unknown types,
        # since SFP demos dominate the dataset.
        plug = (getattr(task, "plug_type", "") or "").strip().lower()
        port = (getattr(task, "port_type", "") or "").strip().lower()
        if "sc" in plug or "sc" in port:
            self.current_task_index = 1
        else:
            self.current_task_index = 0
        if self.num_tasks > 0:
            self.get_logger().info(
                f"insert_cable: plug_type={plug!r} port_type={port!r} -> "
                f"task_index={self.current_task_index} "
                f"({'SC' if self.current_task_index == 1 else 'SFP'})"
            )

        # ── Force-feedback recovery wrapper (v18) ───────────────────────
        # Symptom we're trying to fix: when the policy misses the port, it
        # keeps driving down (or laterally) into the housing with rising
        # contact force and never retries.  Wrap the policy command with a
        # small state machine:
        #   * NORMAL : pass policy actions through.
        #   * LIFT   : ignore policy XY/down-Z; command a constant +Z lift.
        #              Triggered when wrench force magnitude stays above
        #              RECOVERY_F_THRESH for RECOVERY_F_HOLD_S consecutive
        #              seconds.  Lifts for RECOVERY_LIFT_S, then resets the
        #              policy's temporal-ensemble buffer and returns to
        #              NORMAL so the policy can plan a fresh approach.
        # All thresholds env-overridable so we can tune in Gazebo.
        f_thresh = float(os.environ.get("AIC_ACT_RECOVERY_F", "8.0"))     # N
        f_hold_s = float(os.environ.get("AIC_ACT_RECOVERY_F_HOLD_S", "0.6"))
        lift_s = float(os.environ.get("AIC_ACT_RECOVERY_LIFT_S", "0.6"))
        lift_vz = float(os.environ.get("AIC_ACT_RECOVERY_LIFT_VZ", "0.05"))  # m/s
        # Default OFF (was "1") because Run E1 (May-12 07:34) showed that
        # `unset AIC_ACT_RECOVERY` left the recovery wrapper active with
        # leftover aggressive params from prior shell sessions, knocking
        # SFP trials 14+cm off course. Now: if you don't explicitly set
        # AIC_ACT_RECOVERY=1, recovery is OFF.
        recovery_enabled = os.environ.get("AIC_ACT_RECOVERY", "0") == "1"

        # Per-task gating: when AIC_ACT_SFP_ONLY_TRICKS=1 (default), the
        # recovery wrapper, Z-floor, spiral, and SEATING modules only apply
        # to SFP trials. SC trials run pure ACT — important because the
        # current v19 model has zero SC training data, and any LIFT or
        # SEATING action on the SC plug pushes it out of the bounding
        # radius (May-12 Run B/C: SC tier_3 collapsed to 1.0).
        sfp_only_tricks = os.environ.get("AIC_ACT_SFP_ONLY_TRICKS", "1") != "0"
        is_sc_trial = self.current_task_index == 1
        tricks_disabled_for_task = sfp_only_tricks and is_sc_trial
        if tricks_disabled_for_task:
            self.get_logger().warn(
                f"AIC_ACT_SFP_ONLY_TRICKS=1 and task=SC: disabling recovery, "
                f"Z-floor, spiral, and SEATING for this trial. Pure ACT only."
            )
            recovery_enabled = False

        if recovery_enabled:
            self.get_logger().info(
                f"Recovery wrapper ENABLED: f_thresh={f_thresh}N "
                f"hold={f_hold_s}s lift={lift_s}s vz={lift_vz}m/s"
            )
        else:
            self.get_logger().info("Recovery wrapper DISABLED (set AIC_ACT_RECOVERY=1 to enable).")

        # ── Z-descent floor (deploy-time fix for hedging policies) ────────
        # Policies trained on poisoned image stats (v18 family) under-commit
        # actions at deploy: |z_action| ≈ 4 mm/s instead of the 8–13 mm/s
        # seen in training data. The robot then runs out of trial time still
        # at z ~ 0.12 m above the port. Solution: when the policy commands a
        # weak descent AND there's no contact yet AND we're not in recovery,
        # override z_action with a fixed minimum descent velocity.
        # Default disabled (0.0). Set AIC_ACT_Z_FLOOR=0.006 to enable a
        # 6 mm/s descent floor.
        z_floor = float(os.environ.get("AIC_ACT_Z_FLOOR", "0.0"))            # m/s magnitude
        z_floor_f_max = float(os.environ.get("AIC_ACT_Z_FLOOR_F_MAX", "1.5"))  # N
        z_floor_enabled = z_floor > 0.0 and not tricks_disabled_for_task
        if z_floor_enabled:
            self.get_logger().info(
                f"Z-descent floor ENABLED: |vz|<{z_floor*1000:.1f}mm/s and "
                f"|F|<{z_floor_f_max:.1f}N → override vz=-{z_floor*1000:.1f}mm/s"
            )

        # ── Spiral lateral search (peg-in-hole classic) ───────────────────
        # When the gripper makes light contact (1.5N < |F| < f_thresh) but
        # isn't seating, overlay a small lateral spiral onto the policy's
        # XY action so we sweep the cable across the port mouth and find
        # the opening. Helps when the SFP plug is laterally misaligned by
        # a few millimetres at the contact point.
        # Default disabled. Set AIC_ACT_SPIRAL=1 to enable.
        # NOTE: SEATING below is a strict superset of this; if both are set,
        # SEATING takes precedence at the trigger.
        spiral_enabled = (
            os.environ.get("AIC_ACT_SPIRAL", "0") != "0"
            and not tricks_disabled_for_task
        )
        spiral_trigger_f = float(os.environ.get("AIC_ACT_SPIRAL_TRIGGER_F", "1.5"))  # N
        spiral_hold_s = float(os.environ.get("AIC_ACT_SPIRAL_HOLD_S", "0.3"))         # s
        spiral_amp_m = float(os.environ.get("AIC_ACT_SPIRAL_R", "0.004"))             # m peak
        spiral_freq_hz = float(os.environ.get("AIC_ACT_SPIRAL_FREQ", "1.0"))          # Hz
        spiral_max_s = float(os.environ.get("AIC_ACT_SPIRAL_MAX_S", "4.0"))           # s
        spiral_omega = 2.0 * np.pi * spiral_freq_hz
        spiral_peak_vxy = spiral_omega * spiral_amp_m  # m/s peak lateral velocity
        if spiral_enabled:
            self.get_logger().info(
                f"Spiral search ENABLED: trigger=|F|>{spiral_trigger_f:.1f}N "
                f"for {spiral_hold_s:.2f}s, R={spiral_amp_m*1000:.1f}mm "
                f"@ {spiral_freq_hz:.2f}Hz (peak vxy={spiral_peak_vxy*1000:.1f}mm/s) "
                f"max_dur={spiral_max_s:.1f}s"
            )

        # ── SEATING phase (force-controlled insertion) ────────────────────
        # The proper deterministic seating module: when light contact is
        # detected, switch to admittance-controlled descent + a finer
        # spiral.  Designed for chamfer-guided SFP insertion:
        #   * XY: small fast spiral (default 2.5 mm @ 1.5 Hz) so we keep
        #     scanning across the port mouth.
        #   * Z : admittance vz = -K_z * (|Fz_target| - |Fz_meas|), so we
        #     press with constant |Fz| (default 2 N). When the plug slides
        #     into the port channel |Fz| collapses, vz becomes more
        #     negative, and we follow the cable down.
        #   * Success: Δz ≥ AIC_ACT_SEAT_DONE_Z (default 5 mm) of progress
        #     AND |F_total| < AIC_ACT_SEAT_DONE_F (default 1 N) AND held
        #     for AIC_ACT_SEAT_DONE_HOLD_S (default 0.3 s). On success,
        #     transition to terminal SEATED state (zero motion).
        #   * Bail to LIFT on |F| > f_thresh OR after seat_max_s (default
        #     8 s) without success. LIFT then resumes NORMAL → ACT
        #     re-approaches and SEATING fires again on next contact.
        # We use |Fz| (compressive component) for the admittance loop and
        # |F| total only for the bail/trigger thresholds — this prevents
        # spiral side-loads from corrupting the admittance press signal.
        seating_enabled = (
            os.environ.get("AIC_ACT_SEATING", "0") != "0"
            and not tricks_disabled_for_task
        )
        seat_fz = float(os.environ.get("AIC_ACT_SEAT_FZ", "2.0"))                  # N target
        seat_kz = float(os.environ.get("AIC_ACT_SEAT_KZ", "0.005"))                # m/s per N
        seat_vz_press_max = float(os.environ.get("AIC_ACT_SEAT_VZ_PRESS", "0.015"))# m/s
        seat_vz_back_max = float(os.environ.get("AIC_ACT_SEAT_VZ_BACK", "0.008")) # m/s
        seat_amp_m = float(os.environ.get("AIC_ACT_SEAT_R", "0.0025"))             # m peak
        seat_freq_hz = float(os.environ.get("AIC_ACT_SEAT_FREQ", "1.5"))           # Hz
        seat_done_z = float(os.environ.get("AIC_ACT_SEAT_DONE_Z", "0.005"))        # m
        seat_done_f = float(os.environ.get("AIC_ACT_SEAT_DONE_F", "1.0"))          # N
        seat_done_hold_s = float(os.environ.get("AIC_ACT_SEAT_DONE_HOLD_S", "0.3"))# s
        seat_max_s = float(os.environ.get("AIC_ACT_SEAT_MAX_S", "8.0"))            # s
        # Separate bail threshold for SEATING (defaults to f_thresh if unset).
        # The May-12 06:55 trial showed SEATING entering at |F|≈5.9N (with
        # recovery_F=6N) and bailing within 0.8s — no headroom. Setting
        # AIC_ACT_SEAT_BAIL_F=12 lets SEATING tolerate a transient force spike
        # while the spiral finds the chamfer.
        seat_bail_f = float(os.environ.get("AIC_ACT_SEAT_BAIL_F", str(f_thresh)))  # N
        # Minimum force that must have been observed during this seating
        # attempt before SUCCESS is declared. Prevents the May-12 07:05 false
        # positive where SEATING fired in free space on a 0.9-1.1N inertial
        # transient, descended via admittance, and "succeeded" 100mm above
        # the port. Real contact with the port plate produces ≥2-3N before
        # the plug slips into the chamfer.
        seat_min_contact_f = float(os.environ.get("AIC_ACT_SEAT_MIN_CONTACT_F", "2.5"))  # N
        seat_omega = 2.0 * np.pi * seat_freq_hz
        seat_peak_vxy = seat_omega * seat_amp_m
        if seating_enabled:
            if not recovery_enabled:
                self.get_logger().warn(
                    "AIC_ACT_SEATING=1 implicitly enables the recovery "
                    "wrapper so seating-bail can LIFT and retry."
                )
                recovery_enabled = True
            self.get_logger().info(
                f"SEATING ENABLED: Fz_target={seat_fz:.1f}N "
                f"Kz={seat_kz:.4f}m/s/N "
                f"vz_press≤{seat_vz_press_max*1000:.1f}mm/s "
                f"vz_back≤{seat_vz_back_max*1000:.1f}mm/s "
                f"spiral={seat_amp_m*1000:.2f}mm@{seat_freq_hz:.2f}Hz "
                f"(peak vxy={seat_peak_vxy*1000:.1f}mm/s) "
                f"done@(Δz≥{seat_done_z*1000:.1f}mm,|F|<{seat_done_f:.1f}N "
                f"for {seat_done_hold_s:.2f}s,maxF≥{seat_min_contact_f:.1f}N) "
                f"bail_F={seat_bail_f:.1f}N timeout={seat_max_s:.1f}s"
            )

        # ── SEATING max-descent cap (defense against plow-through) ──────
        # Run D trial 1 (May-12 07:24): SEATING entered on real contact
        # (|F|=3.4N at z0=235mm), but admittance kept pressing down for 10s
        # until force *decreased* (lateral chamfer slip), descending 53mm
        # total. The gripper finger crashed into nic_card_mount mid-press,
        # earning a -24 contact penalty. Cap the descent so we bail to LIFT
        # before plowing into housing.
        seat_max_dz = float(os.environ.get("AIC_ACT_SEAT_MAX_DZ", "0.020"))  # m
        if seating_enabled:
            self.get_logger().info(
                f"  SEATING max-descent cap: Δz_max={seat_max_dz*1000:.1f}mm "
                f"(bails to LIFT past this without SUCCESS)"
            )

        # ── Universal force safety cap (defense against runaway press) ──
        # Run D trial 3 (May-12 07:25): pure ACT on SC (no training data)
        # drove the plug into the port at 96.07N for 5s, earning a -12
        # insertion-force penalty (tier_2 went to -12). When |F| exceeds
        # this threshold for FORCE_SAFETY_HOLD_S, freeze translation/
        # rotation. Applies to ALL states and ALL trials regardless of
        # trick gating — pure safety net.
        force_safety_f = float(os.environ.get("AIC_ACT_FORCE_SAFETY_F", "15.0"))      # N
        force_safety_hold_s = float(os.environ.get("AIC_ACT_FORCE_SAFETY_HOLD_S", "0.3"))  # s
        # Run E1 (May-12 07:34) found that ZEROING the commanded velocity
        # didn't reduce force — the impedance controller's position target
        # was already integrated 50+mm downward from past commands, so
        # F=K·Δx kept rising even with zero velocity command. Fix: command
        # an UPWARD retreat velocity to actively pull the position target
        # back. Default 5 mm/s — gentle enough not to push the SC plug
        # out of the bounding radius (max ~30 mm of retreat over a 6 s
        # trial-end window).
        force_safety_retreat_vz = float(os.environ.get("AIC_ACT_FORCE_SAFETY_RETREAT_VZ", "0.005"))  # m/s upward
        force_safety_enabled = force_safety_f > 0.0
        if force_safety_enabled:
            self.get_logger().info(
                f"FORCE SAFETY CAP ENABLED: |F|>{force_safety_f:.1f}N for "
                f"{force_safety_hold_s:.2f}s → retreat at +{force_safety_retreat_vz*1000:.1f}mm/s "
                f"(universal, applies to NORMAL/SEARCH/SEATING/LIFT)"
            )

        start_time = self.get_clock().now()
        # `RunACT` is a plain Policy (not an rclpy Node) so we can't use
        # ``self.create_rate``. Use ``Policy.sleep_for`` (sim-time aware) at the
        # bottom of the loop and a Duration-based deadline check here.
        control_period_s = 1.0 / 30.0
        duration_to_run = Duration(seconds=30.0)

        recovery_state = "NORMAL"
        recovery_high_force_t0 = None  # rclpy Time when force first exceeded
        recovery_lift_t0 = None        # rclpy Time when lift started
        n_recoveries = 0
        # Spiral state — only used when AIC_ACT_SPIRAL=1
        spiral_low_force_t0 = None     # rclpy Time when light-contact first appeared
        spiral_t0 = None               # rclpy Time when spiral SEARCH entered
        n_spirals = 0
        # Seating state — only used when AIC_ACT_SEATING=1
        seat_t0 = None                 # rclpy Time when SEATING entered
        seat_z0 = None                 # tcp.position.z captured at SEATING entry (m)
        seat_done_t0 = None            # rclpy Time when success criteria first met
        seat_max_f_seen = 0.0          # peak |F| observed inside this seating attempt (N)
        n_seats = 0
        seated = False                 # latched True after first successful seat
        # Force-safety cap state (universal, applies to all states)
        force_safety_t0 = None         # rclpy Time when |F| first crossed cap
        force_safety_active = False    # True while motion is currently frozen
        n_force_safety = 0

        iteration = 0
        # Run inference for 30 seconds (sim time)
        while (self.get_clock().now() - start_time) < duration_to_run:

            # 1. Get & Process Observation
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # 2. Model Inference
            with torch.inference_mode():
                # returns shape [1, 6] (first action of chunk)
                normalized_action = self.policy.select_action(obs_tensors)

            # 3. Un-normalize Action
            # Formula: (norm * std) + mean
            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean

            # 4. Extract and Command
            # AIC_ACT_RESCALE: inverse of the training-time action_rescale.
            # If training used action_rescale=0.3 (actions *= 0.3), set
            # AIC_ACT_RESCALE=0.3 here.  The code divides by the factor to
            # recover physical velocity units.
            rescale = float(os.environ.get("AIC_ACT_RESCALE", "1.0"))
            # AIC_ACT_GAIN env var multiplies the unnormalized action before sending.
            # Use it to test deploy-time magnitude scaling without retraining.
            # Default 1.0 = no change; common sweep values 1.3, 1.5, 2.0.
            gain = float(os.environ.get("AIC_ACT_GAIN", "1.0"))
            action = raw_action_tensor[0].cpu().numpy() / rescale * gain
            if iteration == 0 and rescale != 1.0:
                self.get_logger().info(f"AIC_ACT_RESCALE={rescale} (dividing actions to recover original units)")
            if iteration == 0 and gain != 1.0:
                self.get_logger().info(f"AIC_ACT_GAIN={gain} (action magnitudes scaled by this factor)")

            # One-shot input/output diagnostic on the first iteration so we can
            # tell apart "policy is genuinely predicting idle" from "policy is
            # being fed OOD / wrong-shaped observations".
            if iteration == 0:
                self._debug_log_observation(
                    observation_msg, obs_tensors, normalized_action, raw_action_tensor
                )

            # ── Recovery state machine ────────────────────────────────
            # Compute current contact force magnitude from the tared wrench
            # on the wrist (raw - controller tare), same convention used
            # for the state vector.
            now_ros = self.get_clock().now()
            _w = observation_msg.wrist_wrench.wrench
            _t = observation_msg.controller_state.fts_tare_offset.wrench
            fx = _w.force.x - _t.force.x
            fy = _w.force.y - _t.force.y
            fz = _w.force.z - _t.force.z
            f_mag = float(np.sqrt(fx * fx + fy * fy + fz * fz))

            override_action = None
            # Force-feedback recovery state machine. Order matters:
            # SEARCH escalates to LIFT on hard contact, NORMAL escalates to
            # SEARCH on light contact (if spiral enabled), and LIFT always
            # wins (it's the safety state).
            if recovery_enabled:
                if recovery_state == "NORMAL":
                    if f_mag > f_thresh:
                        if recovery_high_force_t0 is None:
                            recovery_high_force_t0 = now_ros
                        elapsed = (now_ros - recovery_high_force_t0).nanoseconds / 1e9
                        if elapsed >= f_hold_s:
                            recovery_state = "LIFT"
                            recovery_lift_t0 = now_ros
                            n_recoveries += 1
                            self.policy.reset()  # flush temporal-ensemble buffer
                            self.get_logger().warn(
                                f"RECOVERY[{n_recoveries}] LIFT triggered: "
                                f"|F|={f_mag:.2f}N held >{f_hold_s:.2f}s "
                                f"(thresh={f_thresh:.1f}N)"
                            )
                    else:
                        recovery_high_force_t0 = None

                if recovery_state == "LIFT":
                    elapsed_lift = (now_ros - recovery_lift_t0).nanoseconds / 1e9
                    if elapsed_lift >= lift_s:
                        recovery_state = "NORMAL"
                        recovery_high_force_t0 = None
                        spiral_low_force_t0 = None
                        self.get_logger().info(
                            f"RECOVERY[{n_recoveries}] LIFT done after "
                            f"{elapsed_lift:.2f}s; resuming policy."
                        )
                    else:
                        # Constant +Z lift, zero on all other axes.
                        override_action = np.array(
                            [0.0, 0.0, lift_vz, 0.0, 0.0, 0.0],
                            dtype=np.float32,
                        )

            # ── Light-contact trigger: enter SEATING (preferred) or SEARCH ──
            # Both seating and spiral use the same hold-timer logic; the
            # *destination* state depends on which env flag is set.
            # SEATING > SEARCH if both are set.
            in_normal = recovery_state == "NORMAL"
            light_contact = spiral_trigger_f <= f_mag < f_thresh
            if (
                (seating_enabled or spiral_enabled)
                and in_normal
                and override_action is None
                and not seated
                and light_contact
            ):
                if spiral_low_force_t0 is None:
                    spiral_low_force_t0 = now_ros
                elapsed_low = (now_ros - spiral_low_force_t0).nanoseconds / 1e9
                if elapsed_low >= spiral_hold_s:
                    if seating_enabled:
                        recovery_state = "SEATING"
                        seat_t0 = now_ros
                        seat_z0 = float(observation_msg.controller_state.tcp_pose.position.z)
                        seat_done_t0 = None
                        seat_max_f_seen = f_mag
                        n_seats += 1
                        self.policy.reset()  # ACT idle during seating
                        self.get_logger().warn(
                            f"SEATING[{n_seats}] entered: |F|={f_mag:.2f}N "
                            f"z0={seat_z0*1000:.1f}mm"
                        )
                    else:  # spiral_enabled only
                        recovery_state = "SEARCH"
                        spiral_t0 = now_ros
                        n_spirals += 1
                        self.get_logger().warn(
                            f"SPIRAL[{n_spirals}] SEARCH triggered: "
                            f"|F|={f_mag:.2f}N held >{spiral_hold_s:.2f}s"
                        )
            elif (
                (seating_enabled or spiral_enabled)
                and in_normal
                and f_mag < spiral_trigger_f * 0.5
            ):
                # Force dropped well below trigger → reset hold timer.
                spiral_low_force_t0 = None

            # ── SEARCH state (spiral-only, no admittance) ──────────────────
            if recovery_state == "SEARCH":
                elapsed_search = (now_ros - spiral_t0).nanoseconds / 1e9
                if elapsed_search >= spiral_max_s or f_mag > f_thresh:
                    self.get_logger().info(
                        f"SPIRAL[{n_spirals}] done after {elapsed_search:.2f}s "
                        f"(|F|={f_mag:.2f}N)"
                    )
                    recovery_state = "NORMAL"
                    spiral_low_force_t0 = None
                    self.policy.reset()
                else:
                    phase = spiral_omega * elapsed_search
                    action[0] = -spiral_peak_vxy * np.sin(phase)
                    action[1] =  spiral_peak_vxy * np.cos(phase)

            # ── SEATING state (admittance Z + spiral XY + insertion check)──
            if recovery_state == "SEATING":
                elapsed_seat = (now_ros - seat_t0).nanoseconds / 1e9
                tcp_z = float(observation_msg.controller_state.tcp_pose.position.z)
                z_progress = seat_z0 - tcp_z  # positive = descended below entry
                fz_press = abs(fz)  # compressive component, isolates press signal
                # Track peak force during this attempt — used to gate SUCCESS so
                # that free-space false-triggers (gripper inertia → 0.8-1.2N
                # transients followed by 0N) cannot trivially pass the
                # |F|<seat_done_f criterion.
                if f_mag > seat_max_f_seen:
                    seat_max_f_seen = f_mag

                # Bail conditions (LIFT and retry)
                if elapsed_seat > seat_max_s:
                    self.get_logger().warn(
                        f"SEATING[{n_seats}] TIMEOUT @ {elapsed_seat:.1f}s "
                        f"(Δz={z_progress*1000:+.1f}mm |F|={f_mag:.2f}N) → LIFT"
                    )
                    recovery_state = "LIFT"
                    recovery_lift_t0 = now_ros
                    n_recoveries += 1
                    spiral_low_force_t0 = None
                    self.policy.reset()
                elif z_progress > seat_max_dz:
                    # Plow-through guard: if we've descended past the cap
                    # without satisfying SUCCESS, the chamfer search has
                    # failed and we're driving into housing. Bail now.
                    self.get_logger().warn(
                        f"SEATING[{n_seats}] MAX-DESCENT Δz={z_progress*1000:+.1f}mm "
                        f"> {seat_max_dz*1000:.1f}mm cap (|F|={f_mag:.2f}N "
                        f"maxF={seat_max_f_seen:.2f}N) @ {elapsed_seat:.1f}s → LIFT"
                    )
                    recovery_state = "LIFT"
                    recovery_lift_t0 = now_ros
                    n_recoveries += 1
                    spiral_low_force_t0 = None
                    self.policy.reset()
                elif f_mag > seat_bail_f:
                    self.get_logger().warn(
                        f"SEATING[{n_seats}] HARD-CONTACT |F|={f_mag:.2f}N "
                        f"> bail={seat_bail_f:.1f}N "
                        f"@ {elapsed_seat:.1f}s (Δz={z_progress*1000:+.1f}mm) → LIFT"
                    )
                    recovery_state = "LIFT"
                    recovery_lift_t0 = now_ros
                    n_recoveries += 1
                    spiral_low_force_t0 = None
                    self.policy.reset()
                # Success requires THREE conditions:
                #   1. We descended at least seat_done_z (insertion progress)
                #   2. Force has dropped below seat_done_f (we slipped IN, not stuck)
                #   3. We previously saw |F|≥seat_min_contact_f during this
                #      attempt (proves the descent ended a real contact event,
                #      not just a free-space drift after a transient trigger)
                elif (
                    z_progress >= seat_done_z
                    and f_mag < seat_done_f
                    and seat_max_f_seen >= seat_min_contact_f
                ):
                    if seat_done_t0 is None:
                        seat_done_t0 = now_ros
                    held = (now_ros - seat_done_t0).nanoseconds / 1e9
                    if held >= seat_done_hold_s:
                        seated = True
                        recovery_state = "SEATED"
                        self.get_logger().warn(
                            f"SEATING[{n_seats}] SUCCESS: "
                            f"Δz={z_progress*1000:+.1f}mm |F|={f_mag:.2f}N "
                            f"maxF={seat_max_f_seen:.2f}N "
                            f"held {held:.2f}s @ {elapsed_seat:.1f}s"
                        )
                else:
                    # Not currently in success window; reset success-hold timer.
                    seat_done_t0 = None

                # Compute seating action (only if we're still in SEATING).
                if recovery_state == "SEATING":
                    # Spiral XY
                    phase = seat_omega * elapsed_seat
                    action[0] = -seat_peak_vxy * np.sin(phase)
                    action[1] =  seat_peak_vxy * np.cos(phase)
                    # Admittance Z: target |Fz| = seat_fz.
                    # vz = -K*(target - meas):
                    #   meas < target  →  err > 0  →  vz < 0  (press more)
                    #   meas > target  →  err < 0  →  vz > 0  (back off)
                    f_err = seat_fz - fz_press
                    vz_admit = -seat_kz * f_err
                    if vz_admit < -seat_vz_press_max:
                        vz_admit = -seat_vz_press_max
                    elif vz_admit > +seat_vz_back_max:
                        vz_admit = +seat_vz_back_max
                    action[2] = vz_admit
                    # Zero rotational drift during seating.
                    action[3] = 0.0
                    action[4] = 0.0
                    action[5] = 0.0

            # ── SEATED terminal: hold position so the cable stays seated ──
            if recovery_state == "SEATED":
                action[:] = 0.0

            # Z-descent floor: when policy hedges and we're not yet in contact
            # and not in LIFT/SEATING/SEATED, force a minimum descent so the
            # robot reaches port height.  SEARCH still respects this; SEATING
            # has its own admittance-controlled Z and shouldn't be overridden.
            if (
                z_floor_enabled
                and override_action is None
                and recovery_state in ("NORMAL", "SEARCH")
                and f_mag < z_floor_f_max
                and action[2] > -z_floor  # commanded descent weaker than floor
            ):
                action[2] = -z_floor

            if override_action is not None:
                action = override_action

            # ── Universal force-safety cap (last-line defense) ─────────────
            # Active retreat: when |F| exceeds the cap for force_safety_hold_s,
            # zero XY/rotation and command an UPWARD Z velocity to back off.
            # Run E1 showed that just zeroing velocity didn't reduce force
            # because the impedance controller's position target was already
            # integrated down — F=K·Δx kept climbing. Active retreat actually
            # pulls the position target back.
            if force_safety_enabled and f_mag > force_safety_f:
                if force_safety_t0 is None:
                    force_safety_t0 = now_ros
                elapsed_safety = (now_ros - force_safety_t0).nanoseconds / 1e9
                if elapsed_safety > force_safety_hold_s:
                    if not force_safety_active:
                        n_force_safety += 1
                        force_safety_active = True
                        self.get_logger().error(
                            f"FORCE SAFETY[{n_force_safety}] |F|={f_mag:.1f}N > "
                            f"{force_safety_f:.1f}N for {elapsed_safety:.2f}s "
                            f"in state={recovery_state} → retreat "
                            f"+{force_safety_retreat_vz*1000:.1f}mm/s"
                        )
                    # Active retreat: zero lateral and rotation, lift Z.
                    action[0] = 0.0
                    action[1] = 0.0
                    action[2] = +force_safety_retreat_vz
                    action[3] = 0.0
                    action[4] = 0.0
                    action[5] = 0.0
            else:
                force_safety_t0 = None
                if force_safety_active:
                    self.get_logger().warn(
                        f"FORCE SAFETY[{n_force_safety}] released "
                        f"(|F|={f_mag:.2f}N < {force_safety_f:.1f}N)"
                    )
                    force_safety_active = False

            if iteration % 30 == 0:
                self.get_logger().info(
                    f"Iteration {iteration} | "
                    f"SimTime: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.2f}s | "
                    f"|F|={f_mag:.2f}N state={recovery_state} | "
                    f"Unnormalized Act: {action[:3]}"
                )

            iteration += 1

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion_update = self.set_cartesian_twist_target(twist)
            move_robot(motion_update=motion_update)

            send_feedback("in progress...")

            self.sleep_for(control_period_s)

        self.get_logger().info("RunACT.insert_cable() done.")
        return True

    def set_cartesian_twist_target(self, twist: Twist, frame_id: str = "base_link"):
        motion_update_msg = MotionUpdate()
        motion_update_msg.velocity = twist
        motion_update_msg.header.frame_id = frame_id
        motion_update_msg.header.stamp = self.get_clock().now().to_msg()

        motion_update_msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update_msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        motion_update_msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)
        )

        motion_update_msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update_msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return motion_update_msg
