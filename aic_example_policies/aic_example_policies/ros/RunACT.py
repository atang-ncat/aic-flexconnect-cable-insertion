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
        # v12: frozen backbone + cosine LR + v10-level regularization.
        # best val/l1=0.244 @ step 3500. weight_norm=1.296 (vs v11c's 0.722).
        # Override with env AIC_ACT_POLICY_PATH=/path/to/checkpoints/best
        _default_ckpt = (
            "/scratch2/atang/ws_aic/outputs/act_sfp/"
            "v12_cosine_lr_frozen_bb_20260426_172303/checkpoints/best"
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
        self.state_dim = int(stats["observation.state.mean"].shape[0])
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        print(f"Robot state dim={self.state_mean.shape[-1]} mean: {self.state_mean}")
        print(f"Robot state std: {self.state_std}")

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

        if self.state_dim == 27:
            # v10_ft: train_act drops tcp_velocity.* and tcp_error.* from state.
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
        elif self.state_dim == 26:
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
        else:
            raise ValueError(
                f"Unsupported observation.state dim {self.state_dim} in checkpoint stats; "
                "expected 26 (legacy) or 27 (v10_ft)."
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

        start_time = self.get_clock().now()
        # `RunACT` is a plain Policy (not an rclpy Node) so we can't use
        # ``self.create_rate``. Use ``Policy.sleep_for`` (sim-time aware) at the
        # bottom of the loop and a Duration-based deadline check here.
        control_period_s = 1.0 / 30.0
        duration_to_run = Duration(seconds=30.0)

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
            action = raw_action_tensor[0].cpu().numpy()

            # One-shot input/output diagnostic on the first iteration so we can
            # tell apart "policy is genuinely predicting idle" from "policy is
            # being fed OOD / wrong-shaped observations".
            if iteration == 0:
                self._debug_log_observation(
                    observation_msg, obs_tensors, normalized_action, raw_action_tensor
                )

            if iteration % 30 == 0:
                self.get_logger().info(f"Iteration {iteration} | SimTime: {(self.get_clock().now() - start_time).nanoseconds / 1e9:.2f}s | Unnormalized Act: {action[:3]}")

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
