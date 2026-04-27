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

"""Deploy the v6b 3-class discrete-action ACT policy.

The v6 family (see ``scripts/modeling_discrete_act.py``) replaces ACT's
per-dimension regression head with 6 independent 3-way softmax heads. At
inference the model decodes argmax -> {-0.1, 0, +0.1} per action dim, so
the output is already in raw physical units and there is no
``unnormalize`` step. This sidesteps the action-mean attractor that the
regression-head ACT collapsed into during sim deployment.

Two key differences vs. ``RunACT``:

* ``DiscreteACTPolicy`` (subclass of ``ACTPolicy``) is constructed and
  has its weights loaded from the checkpoint. The transformer backbone
  is identical; only the action head differs.
* ``observation.state`` is 14-D, not 26-D: the run was trained with
  ``state_drop_prefixes: [tcp_velocity., tcp_error.]``. We construct
  the 14-D vector directly from the ``Observation`` message (TCP pose
  + 7 alphabetized joint positions).
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Dict
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
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# DiscreteACTPolicy lives in the workspace's ``scripts/`` dir (it's the
# training-time architecture override that produced the v6b checkpoint).
# Make it importable from inside the pixi env.
# ---------------------------------------------------------------------------
_SCRIPTS_CANDIDATES = [
    "/scratch2/atang/ws_aic/scripts",
    "/run/host/scratch2/atang/ws_aic/scripts",
]
for _p in _SCRIPTS_CANDIDATES:
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)
        break

from modeling_discrete_act import DiscreteACTPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# State layout: the 26-D Observation state vector emitted by aic_adapter is
#
#   [0..6]   tcp_pose (xyz + xyzw quat)
#   [7..12]  tcp_velocity (linear xyz, angular xyz)         <-- DROPPED
#   [13..18] tcp_error (xyz, rxryrz)                        <-- DROPPED
#   [19..25] joint_positions (alphabetized 7-vec)
#
# The v6b run was trained on the 14-D subvector [0..6, 19..25].
# ---------------------------------------------------------------------------
_KEEP_STATE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
]


class RunACTClassification(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Resolve checkpoint path (with /run/host fallback for the eval container)
        policy_path = Path(
            "/scratch2/atang/ws_aic/outputs/act_sfp/v6b_3cam_170_30/checkpoints/best"
        )
        if not policy_path.exists() and Path("/run/host" + str(policy_path)).exists():
            policy_path = Path("/run/host" + str(policy_path))

        # 2. Load ACTConfig (strip the LeRobot 'type' tag draccus doesn't recognize).
        with open(policy_path / "config.json", "r") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]
        config = draccus.decode(ACTConfig, config_dict)

        # 3. Build DiscreteACTPolicy and load weights.
        self.policy = DiscreteACTPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(
            f"DiscreteACT (v6b classification) policy loaded on {self.device} from {policy_path}"
        )

        # 4. Load normalization stats from the same checkpoint's preprocessor file.
        #    Action stats are NOT needed (head outputs physical actions directly via argmax).
        stats_path = (
            policy_path
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        stats = load_file(stats_path)

        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        # Image stats: (1, 3, 1, 1) for broadcasting against (B, C, H, W).
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
        # State stats: shape (14,) per the v6b state_drop_prefixes config.
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        if self.state_mean.shape[1] != len(_KEEP_STATE_NAMES):
            raise RuntimeError(
                f"v6b expects {len(_KEEP_STATE_NAMES)}-D state, but checkpoint stats "
                f"have shape {tuple(self.state_mean.shape)}"
            )

        self.image_scaling = 0.25  # Must match aic_adapter's down-sampling factor.

        self.get_logger().info(
            "Image / state stats loaded; action head is IDENTITY (physical units)."
        )
        # Print so we can compare against the live diagnostic dump on iter 0.
        print(f"State mean (14): {self.state_mean.cpu().numpy()}")
        print(f"State std  (14): {self.state_std.cpu().numpy()}")

    # ------------------------------------------------------------------
    # Image decoding (same as RunACT, kept here so we have one self-
    # contained file)
    # ------------------------------------------------------------------
    @staticmethod
    def _ros_image_to_rgb(raw_img) -> np.ndarray:
        """Decode a sensor_msgs/Image into an HxWx3 uint8 RGB ndarray.

        We avoid ``cv_bridge`` because the system-ROS build links against
        NumPy 1.x while the pixi env ships NumPy >= 2; importing
        ``cv_bridge`` from the pixi env crashes with a ``_multiarray_umath``
        ABI error.
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
        """ROS Image -> resized -> permuted -> normalized tensor (1, 3, H, W)."""
        img_np = RunACTClassification._ros_image_to_rgb(raw_img)
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation) -> Dict[str, torch.Tensor]:
        """Convert ROS Observation message into the 14-D state + 3 image tensors."""

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

        tcp_pose = obs_msg.controller_state.tcp_pose

        # The teleop-dataset stored joints in alphabetical order (LeRobot quirk).
        # aic_adapter emits structural order, so re-alphabetize here.
        target_joint_names = [
            "elbow_joint",
            "gripper",  # aic_adapter renamed gripper/left_finger_joint -> gripper
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        joint_dict = dict(
            zip(obs_msg.joint_states.name, obs_msg.joint_states.position)
        )
        # aic_adapter divides the prismatic gripper by 2.0; undo that to match training.
        joint_dict["gripper"] *= 2.0
        ordered_joints = [joint_dict[name] for name in target_joint_names]

        # Build the 14-D state vector directly (matches v6b's state_drop_prefixes).
        state_np = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # Joint Positions (7)
                *ordered_joints,
            ],
            dtype=np.float32,
        )
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std
        return obs

    # ------------------------------------------------------------------
    # Diagnostics: same single-shot dump shape as RunACT, adapted for the
    # 14-D state and physical action output (no un-normalize step).
    # ------------------------------------------------------------------
    def _debug_log_observation(
        self,
        obs_msg: Observation,
        obs_tensors: Dict[str, torch.Tensor],
        action_phys: torch.Tensor,
    ) -> None:
        log = self.get_logger()
        log.info("=" * 70)
        log.info("RunACTClassification diagnostic dump (iter=0)")
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
                f"min={rgb.min():.3f} max={rgb.max():.3f}"
            )
            t = obs_tensors[key][0].detach().cpu()
            log.info(
                f"  {name:6s} NORM: shape={tuple(t.shape)} "
                f"mean=[{t[0].mean():.2f},{t[1].mean():.2f},{t[2].mean():.2f}] "
                f"min={t.min():.2f} max={t.max():.2f}"
            )

        state_norm = obs_tensors["observation.state"][0].detach().cpu().numpy()
        state_mean = self.state_mean[0].detach().cpu().numpy()
        state_std = self.state_std[0].detach().cpu().numpy()
        state_raw = state_norm * state_std + state_mean
        np_opts = dict(precision=3, suppress_small=True, max_line_width=200)
        log.info(f"  state RAW  (14): {np.array2string(state_raw, **np_opts)}")
        log.info(f"  state NORM (14): {np.array2string(state_norm, **np_opts)}")
        log.info(
            f"  state max |z|: {np.abs(state_norm).max():.2f}  "
            f"(>3 => at least one state component is OOD)"
        )

        ap = action_phys[0].detach().cpu().numpy()
        log.info(f"  action PHYSICAL(6): {np.array2string(ap, **np_opts)}  "
                 f"(should be in {{-0.1, 0, +0.1}})")
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
        self.get_logger().info(
            f"RunACTClassification.insert_cable() enter. Task: {task}"
        )

        start_time = self.get_clock().now()
        control_period_s = 1.0 / 30.0
        duration_to_run = Duration(seconds=30.0)

        iteration = 0
        while (self.get_clock().now() - start_time) < duration_to_run:
            observation_msg = get_observation()
            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            with torch.inference_mode():
                # DiscreteACTPolicy.predict_action_chunk returns physical
                # actions; ACTPolicy.select_action manages the chunk queue
                # and pops one (1, 6) action per call. Since no
                # unnormalize_outputs processor is registered, the value
                # passes through unchanged.
                action_phys = self.policy.select_action(obs_tensors)

            action = action_phys[0].cpu().numpy()

            if iteration == 0:
                self._debug_log_observation(observation_msg, obs_tensors, action_phys)

            if iteration % 30 == 0:
                self.get_logger().info(
                    f"Iteration {iteration} | SimTime: "
                    f"{(self.get_clock().now() - start_time).nanoseconds / 1e9:.2f}s | "
                    f"Phys Act: {action}"
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

        self.get_logger().info("RunACTClassification.insert_cable() done.")
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
