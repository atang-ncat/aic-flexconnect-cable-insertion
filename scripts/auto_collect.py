#!/usr/bin/env python3
"""
Automated SFP/SC demo collection using CheatCode-style ground-truth
targeting with velocity commands compatible with the LeRobot training
pipeline.

Architecture:
    - Uses the same MotionUpdate/velocity interface as lerobot-record
    - Uses tf2_ros for ground-truth port/plug frame lookups
    - Proportional controller converts pose targets -> 6D velocity actions
    - Records directly into LeRobot dataset format via LeRobotDataset API
    - Filters episodes by /scoring/insertion_event success
    - Monitors force and collisions to discard bad episodes
    - Records F/T (tared) as 6 extra columns in observation.state, so the
      output dataset is schema-compatible with post-2026-04-24 teleop data
    - Sim-time control loop (create_rate + rate.sleep) so dataset frame
      rate stays locked to ``fps`` regardless of Gazebo's wall-clock speed
    - Encoding-aware image decoding (rgb8 / bgr8 / rgba8 / bgra8)
    - No zero-action frames on transient TF failures (holds last command
      instead) and no zero-frame frames on camera stalls (drops the tick)

Usage (inside distrobox, with Gazebo + ground_truth:=true running):

    # Smoke test first — runs one episode without creating a dataset:
    cd /run/host/scratch2/atang/ws_aic/src/aic
    pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
        --plug-type sfp --dry-run

    # Real collection:
    pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
        --plug-type sfp \
        --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset-ft/sfp \
        --num-episodes 30

    # SC collection uses a different plug profile (lower gains, slower descent):
    pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
        --plug-type sc \
        --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset-ft/sc \
        --num-episodes 30

The dataset it produces has 32-D observation.state (same as the F/T-
enabled teleop driver), so auto-collected and teleop demos can be
trained on together.
"""

import argparse
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
)
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Twist, Vector3, Wrench, WrenchStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

try:
    from ros_gz_interfaces.msg import Contacts
    HAS_CONTACTS_MSG = True
except ImportError:
    HAS_CONTACTS_MSG = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quaternion helpers (w,x,y,z convention matching CheatCode)
# ---------------------------------------------------------------------------

def quat_to_rotmat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quat_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_to_axis_angle(q_wxyz):
    """Convert quaternion to axis-angle. Returns (axis, angle)."""
    w, x, y, z = q_wxyz
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * math.acos(abs(w))
    s = math.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([x / s, y / s, z / s])
    if w < 0:
        axis = -axis
    return axis, angle


def orientation_error_as_angular_vel(q_current_wxyz, q_target_wxyz, gain=2.0, max_vel=0.3):
    """Compute angular velocity to rotate from current to target orientation."""
    q_error = quat_multiply(q_target_wxyz, quat_conjugate(q_current_wxyz))
    if q_error[0] < 0:
        q_error = tuple(-x for x in q_error)
    axis, angle = quat_to_axis_angle(q_error)
    angular_vel = gain * angle * axis
    mag = np.linalg.norm(angular_vel)
    if mag > max_vel:
        angular_vel = angular_vel * (max_vel / mag)
    return angular_vel


# ---------------------------------------------------------------------------
# Safety thresholds (matching ScoringTier2.cc)
# ---------------------------------------------------------------------------
FORCE_THRESHOLD = 20.0   # N — scoring penalty above this
FORCE_WARN = 15.0        # N — log a warning before hitting penalty
FORCE_ABORT = 30.0       # N — abort episode immediately


# ---------------------------------------------------------------------------
# ROS2 interface node
# ---------------------------------------------------------------------------

class AutoCollectNode(Node):
    def __init__(self):
        super().__init__("auto_collect_node", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True),
        ])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_controller_state = None
        self.last_joint_states = None
        self.last_wrench = None          # WrenchStamped — for observation.state
        self.insertion_detected = Event()
        self.collision_detected = Event()

        # Force monitoring state.  _tare_force/_tare_torque are the tare
        # offset from the controller; _current_wrench is the TARED wrench
        # used both for safety aborts and for the observation.state record.
        self._force_lock = Lock()
        self._tare_force = np.zeros(3)
        self._tare_torque = np.zeros(3)
        self._current_force_mag = 0.0
        self._max_force_mag = 0.0
        self._time_above_threshold = 0.0
        self._last_wrench_time = None
        self._current_wrench_tared = np.zeros(6)   # [fx, fy, fz, tx, ty, tz]

        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._controller_state_cb, 10,
        )
        self.create_subscription(
            JointState, "/joint_states",
            self._joint_states_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/scoring/insertion_event",
            self._insertion_cb, 10,
        )
        if HAS_CONTACTS_MSG:
            self.create_subscription(
                Contacts, "/aic/gazebo/contacts/off_limit",
                self._collision_cb, 10,
            )
        else:
            logger.warning("ros_gz_interfaces not available — collision detection disabled")
        self.create_subscription(
            WrenchStamped, "/fts_broadcaster/wrench",
            self._wrench_cb, qos_profile_sensor_data,
        )

        # Ground-truth TF bootstrap.  The AIC bringup has a
        # `topic_tools/relay` node forwarding /scoring/tf -> /tf with
        # `lazy: True`.  Latched static messages (specifically
        # /task_board/pose_static, which is published ONCE at startup via
        # TRANSIENT_LOCAL QoS) get lost because the relay only subscribes
        # to its input after /tf has a subscriber, which happens too late
        # to catch the initial latched message.  Dynamic frames like
        # cable_0 still come through because they're re-published every
        # tick.  This results in cable_0/* frames being available but
        # task_board/* frames never appearing on /tf.
        #
        # Fix: subscribe directly to /scoring/tf ourselves with matching
        # TRANSIENT_LOCAL QoS so we receive the latched static message,
        # then feed each transform into our TF buffer via
        # set_transform_static.  Also listen with default QoS for any
        # late static updates (scene re-spawn etc).  The regular
        # TransformListener still handles dynamic /tf transforms, so
        # cable_0/* remains correctly updated live.
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10,
        )
        self.create_subscription(
            TFMessage, "/scoring/tf",
            self._scoring_tf_static_cb, qos_latched,
        )

        self.motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10,
        )

    def _scoring_tf_static_cb(self, msg: TFMessage):
        """Treat every transform on /scoring/tf (transient_local) as static.

        The bridge publishes task_board/* here with TRANSIENT_LOCAL QoS and
        never updates them (the task board doesn't move).  Feeding into
        the buffer as static means they won't expire from the time-windowed
        cache during long sessions.
        """
        for tf in msg.transforms:
            try:
                self.tf_buffer.set_transform_static(tf, "ground_truth_bootstrap")
            except Exception as e:
                logger.debug(f"set_transform_static skipped {tf.child_frame_id}: {e}")

    def _controller_state_cb(self, msg):
        self.last_controller_state = msg
        tare_f = msg.fts_tare_offset.wrench.force
        tare_t = msg.fts_tare_offset.wrench.torque
        with self._force_lock:
            self._tare_force = np.array([tare_f.x, tare_f.y, tare_f.z])
            self._tare_torque = np.array([tare_t.x, tare_t.y, tare_t.z])

    def _joint_states_cb(self, msg):
        self.last_joint_states = msg

    def _insertion_cb(self, msg):
        logger.info(f"  [EVENT] Insertion detected: {msg.data}")
        self.insertion_detected.set()

    def _collision_cb(self, msg):
        if len(msg.contacts) > 0:
            logger.warning("  [EVENT] Off-limit collision detected!")
            self.collision_detected.set()

    def _wrench_cb(self, msg):
        self.last_wrench = msg
        raw_f = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        raw_t = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        with self._force_lock:
            tared_f = raw_f - self._tare_force
            tared_t = raw_t - self._tare_torque
            self._current_wrench_tared = np.concatenate([tared_f, tared_t])
            mag = float(np.linalg.norm(tared_f))
            self._current_force_mag = mag
            if mag > self._max_force_mag:
                self._max_force_mag = mag
            now = time.monotonic()
            if self._last_wrench_time is not None and mag > FORCE_THRESHOLD:
                self._time_above_threshold += now - self._last_wrench_time
            self._last_wrench_time = now

    def get_tared_wrench(self) -> np.ndarray:
        """Return the latest tared wrench (6-D: force xyz, torque xyz)."""
        with self._force_lock:
            return self._current_wrench_tared.copy()

    def reset_episode_monitors(self):
        """Reset per-episode monitoring state."""
        self.insertion_detected.clear()
        self.collision_detected.clear()
        with self._force_lock:
            self._current_force_mag = 0.0
            self._max_force_mag = 0.0
            self._time_above_threshold = 0.0
            self._last_wrench_time = None
            # Do NOT reset _current_wrench_tared or _tare_force/_tare_torque;
            # those are maintained by the callbacks across episodes.

    def get_force_stats(self):
        with self._force_lock:
            return {
                "current": self._current_force_mag,
                "max": self._max_force_mag,
                "time_above_threshold": self._time_above_threshold,
            }

    def is_force_abort(self):
        with self._force_lock:
            return self._current_force_mag > FORCE_ABORT

    def wait_for_state(self, timeout=10.0, require_wrench=True):
        """Block until all required ROS state has arrived.

        Gating on wrench matters because the first few ticks of every
        recorded episode would otherwise carry zero F/T, which leaks a
        dead signal into the training distribution.
        """
        t0 = time.monotonic()
        while True:
            missing = []
            if self.last_controller_state is None:
                missing.append("controller_state")
            if self.last_joint_states is None:
                missing.append("joint_states")
            if require_wrench and self.last_wrench is None:
                missing.append("fts_broadcaster/wrench")
            if not missing:
                return
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"Timed out waiting for: {missing}")
            time.sleep(0.1)

    def lookup_tf(self, target, source):
        return self.tf_buffer.lookup_transform(target, source, Time())

    def send_velocity(self, linear, angular, frame_id="base_link"):
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.velocity = Twist(
            linear=Vector3(x=float(linear[0]), y=float(linear[1]), z=float(linear[2])),
            angular=Vector3(x=float(angular[0]), y=float(angular[1]), z=float(angular[2])),
        )
        msg.target_stiffness = np.diag([85.0]*6).flatten()
        msg.target_damping = np.diag([75.0]*6).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0]*6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self.motion_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity([0, 0, 0], [0, 0, 0])

    def get_tcp_pose(self):
        cs = self.last_controller_state
        p = cs.tcp_pose
        return (
            np.array([p.position.x, p.position.y, p.position.z]),
            (p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
        )

    def get_observation_dict(self):
        """Build observation dict matching AICRobotAICController (32-D).

        Layout mirrors the teleop driver exactly after the 2026-04-24 F/T
        update: 26 legacy fields + 6 wrench fields (force xyz + torque xyz),
        all with the controller's tare subtracted.  This keeps
        auto-collected demos schema-compatible with F/T-enabled teleop
        data so they can be trained on together.
        """
        cs = self.last_controller_state
        js = self.last_joint_states
        p = cs.tcp_pose
        v = cs.tcp_velocity
        e = cs.tcp_error
        w = self.get_tared_wrench()
        return {
            "tcp_pose.position.x": p.position.x,
            "tcp_pose.position.y": p.position.y,
            "tcp_pose.position.z": p.position.z,
            "tcp_pose.orientation.x": p.orientation.x,
            "tcp_pose.orientation.y": p.orientation.y,
            "tcp_pose.orientation.z": p.orientation.z,
            "tcp_pose.orientation.w": p.orientation.w,
            "tcp_velocity.linear.x": v.linear.x,
            "tcp_velocity.linear.y": v.linear.y,
            "tcp_velocity.linear.z": v.linear.z,
            "tcp_velocity.angular.x": v.angular.x,
            "tcp_velocity.angular.y": v.angular.y,
            "tcp_velocity.angular.z": v.angular.z,
            "tcp_error.x": e[0],
            "tcp_error.y": e[1],
            "tcp_error.z": e[2],
            "tcp_error.rx": e[3],
            "tcp_error.ry": e[4],
            "tcp_error.rz": e[5],
            "joint_positions.0": js.position[0],
            "joint_positions.1": js.position[1],
            "joint_positions.2": js.position[2],
            "joint_positions.3": js.position[3],
            "joint_positions.4": js.position[4],
            "joint_positions.5": js.position[5],
            "joint_positions.6": js.position[6],
            "wrench.force.x": float(w[0]),
            "wrench.force.y": float(w[1]),
            "wrench.force.z": float(w[2]),
            "wrench.torque.x": float(w[3]),
            "wrench.torque.y": float(w[4]),
            "wrench.torque.z": float(w[5]),
        }


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

class CameraCapture:
    """Subscribes to ROS image topics and caches the latest frame per camera.

    Image handling is encoding-aware: Gazebo typically publishes ``rgb8``,
    ``bgr8``, or ``rgba8`` depending on sensor config.  We detect via
    ``msg.encoding`` and convert to a canonical ``rgb8`` (H, W, 3) uint8
    layout — same as what the lerobot driver records — so auto-collected
    frames match teleop frames pixel-for-pixel through the training
    pipeline.  Silently assuming bytes-as-RGB (previous behavior) would
    produce a BGR-trained policy that fails at eval time.
    """

    _SUPPORTED = {
        # encoding -> (channels, swap_rb_to_rgb)
        "rgb8": (3, False),
        "bgr8": (3, True),
        "rgba8": (4, False),
        "bgra8": (4, True),
    }

    def __init__(self, node, cameras, scale=0.25, out_hw=(256, 288)):
        self.frames = {}
        self.scale = scale
        self.out_hw = out_hw
        self._warned_encodings = set()
        for name, topic in cameras.items():
            self.frames[name] = None
            node.create_subscription(
                Image, topic,
                lambda msg, n=name: self._img_cb(n, msg),
                qos_profile_sensor_data,
            )

    def _decode(self, msg):
        enc = msg.encoding
        if enc not in self._SUPPORTED:
            if enc not in self._warned_encodings:
                logger.warning(f"Unexpected image encoding '{enc}'; treating as rgb8 bytes. "
                               f"If colors look wrong, add this encoding to CameraCapture._SUPPORTED.")
                self._warned_encodings.add(enc)
            channels, swap_rb = 3, False
        else:
            channels, swap_rb = self._SUPPORTED[enc]
        # Use step for correct stride handling instead of assuming
        # packed rows (Gazebo does pack, but robustness is free).
        expected_size = msg.height * msg.step
        data = np.frombuffer(msg.data, dtype=np.uint8, count=expected_size)
        arr = data.reshape(msg.height, msg.step // channels, channels)[:, :msg.width, :]
        if channels == 4:
            arr = arr[:, :, :3]
        if swap_rb:
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    def _img_cb(self, name, msg):
        try:
            arr = self._decode(msg)
            if self.scale != 1.0:
                arr = cv2.resize(arr, None, fx=self.scale, fy=self.scale,
                                 interpolation=cv2.INTER_AREA)
            self.frames[name] = arr
        except Exception as e:
            logger.error(f"Image decode failed for {name} (encoding={msg.encoding}): {e}")

    def get_images(self):
        """Return the latest frames.  Raises if any camera has no data —
        the old fallback of substituting zeros was silently polluting the
        dataset with black frames when a camera topic stalled."""
        missing = [k for k, v in self.frames.items() if v is None]
        if missing:
            raise RuntimeError(f"No image received yet from cameras: {missing}")
        return {k: v.copy() for k, v in self.frames.items()}

    def wait_for_images(self, timeout=10.0):
        t0 = time.monotonic()
        while any(v is None for v in self.frames.values()):
            if time.monotonic() - t0 > timeout:
                missing = [k for k, v in self.frames.items() if v is None]
                logger.warning(f"Timed out waiting for cameras: {missing}")
                return False
            time.sleep(0.1)
        return True


# ---------------------------------------------------------------------------
# Proportional controller
# ---------------------------------------------------------------------------

@dataclass
class ControllerGains:
    kp_linear: float = 1.5
    kp_angular: float = 2.0
    max_linear_vel: float = 0.05
    max_angular_vel: float = 0.3
    insertion_linear_vel: float = 0.01


ZERO_ACTION = {
    "linear.x": 0.0, "linear.y": 0.0, "linear.z": 0.0,
    "angular.x": 0.0, "angular.y": 0.0, "angular.z": 0.0,
}


def compute_velocity_action(
    current_pos, current_quat_wxyz,
    target_pos, target_quat_wxyz,
    gains: ControllerGains,
    insertion_phase: bool = False,
):
    pos_error = target_pos - current_pos
    dist = np.linalg.norm(pos_error)
    max_v = gains.insertion_linear_vel if insertion_phase else gains.max_linear_vel

    linear_vel = gains.kp_linear * pos_error
    lin_mag = np.linalg.norm(linear_vel)
    if lin_mag > max_v:
        linear_vel = linear_vel * (max_v / lin_mag)

    angular_vel = orientation_error_as_angular_vel(
        current_quat_wxyz, target_quat_wxyz,
        gain=gains.kp_angular, max_vel=gains.max_angular_vel,
    )

    return {
        "linear.x": float(linear_vel[0]),
        "linear.y": float(linear_vel[1]),
        "linear.z": float(linear_vel[2]),
        "angular.x": float(angular_vel[0]),
        "angular.y": float(angular_vel[1]),
        "angular.z": float(angular_vel[2]),
    }, dist


# ---------------------------------------------------------------------------
# Target pose computation (from CheatCode logic)
# ---------------------------------------------------------------------------

def compute_target_pose(node, port_frame, plug_frame, z_offset=0.1):
    """
    Compute gripper target pose that aligns plug with port,
    offset along the port's insertion axis by z_offset.
    """
    port_tf = node.lookup_tf("base_link", port_frame)
    plug_tf = node.lookup_tf("base_link", plug_frame)
    gripper_tf = node.lookup_tf("base_link", "gripper/tcp")

    port_pos = np.array([
        port_tf.transform.translation.x,
        port_tf.transform.translation.y,
        port_tf.transform.translation.z,
    ])
    plug_pos = np.array([
        plug_tf.transform.translation.x,
        plug_tf.transform.translation.y,
        plug_tf.transform.translation.z,
    ])
    gripper_pos = np.array([
        gripper_tf.transform.translation.x,
        gripper_tf.transform.translation.y,
        gripper_tf.transform.translation.z,
    ])

    q_port = (port_tf.transform.rotation.w, port_tf.transform.rotation.x,
              port_tf.transform.rotation.y, port_tf.transform.rotation.z)
    q_plug = (plug_tf.transform.rotation.w, plug_tf.transform.rotation.x,
              plug_tf.transform.rotation.y, plug_tf.transform.rotation.z)
    q_gripper = (gripper_tf.transform.rotation.w, gripper_tf.transform.rotation.x,
                 gripper_tf.transform.rotation.y, gripper_tf.transform.rotation.z)

    R_port = quat_to_rotmat(q_port)
    insertion_axis = R_port[:, 2]
    plug_to_gripper = gripper_pos - plug_pos

    target_plug_pos = port_pos + insertion_axis * z_offset
    target_gripper_pos = target_plug_pos + plug_to_gripper

    q_diff = quat_multiply(q_port, quat_conjugate(q_plug))
    q_gripper_target = quat_multiply(q_diff, q_gripper)

    return target_gripper_pos, q_gripper_target


# ---------------------------------------------------------------------------
# LeRobot dataset helpers
# ---------------------------------------------------------------------------

def create_or_resume_dataset(repo_id, root, fps=30, resume=False, vcodec="libsvtav1"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if resume and Path(root).exists():
        logger.info(f"Resuming dataset at {root}")
        return LeRobotDataset(repo_id, root=root, vcodec=vcodec)

    logger.info(f"Creating new dataset at {root}")

    cam_shape = [256, 288, 3]
    features = {
        "observation.state": {
            # 32-D: 26 legacy fields + 6 tared wrench fields.  This layout
            # mirrors the AICRobotAICController ObservationState (post-
            # 2026-04-24 F/T fix) so auto-collected and teleop datasets
            # have identical schemas and can be trained on together.
            "dtype": "float32",
            "shape": [32],
            "names": [
                "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
                "tcp_pose.orientation.x", "tcp_pose.orientation.y",
                "tcp_pose.orientation.z", "tcp_pose.orientation.w",
                "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
                "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
                "tcp_error.x", "tcp_error.y", "tcp_error.z",
                "tcp_error.rx", "tcp_error.ry", "tcp_error.rz",
                "joint_positions.0", "joint_positions.1", "joint_positions.2",
                "joint_positions.3", "joint_positions.4", "joint_positions.5",
                "joint_positions.6",
                "wrench.force.x", "wrench.force.y", "wrench.force.z",
                "wrench.torque.x", "wrench.torque.y", "wrench.torque.z",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [6],
            "names": ["linear.x", "linear.y", "linear.z",
                      "angular.x", "angular.y", "angular.z"],
        },
    }
    for cam_name in ["left_camera", "center_camera", "right_camera"]:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": cam_shape,
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": cam_shape[0],
                "video.width": cam_shape[1],
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return LeRobotDataset.create(
        repo_id, fps, root=root, robot_type="ur5e_aic",
        features=features, use_videos=True,
        image_writer_processes=0, image_writer_threads=4 * 3,
        vcodec=vcodec,
    )


def build_frame(obs_dict, action_dict, images, dataset_features, task_str):
    obs_state = np.array([obs_dict[n] for n in dataset_features["observation.state"]["names"]],
                         dtype=np.float32)
    action_arr = np.array([action_dict[n] for n in dataset_features["action"]["names"]],
                          dtype=np.float32)

    frame = {
        "observation.state": obs_state,
        "action": action_arr,
        "task": task_str,
    }
    for cam_name in ["left_camera", "center_camera", "right_camera"]:
        frame[f"observation.images.{cam_name}"] = images[cam_name]
    return frame


# ---------------------------------------------------------------------------
# Insertion episode logic
# ---------------------------------------------------------------------------

# Per-plug defaults.  Selected by --plug-type; override any individual
# field via --port-frame / --plug-frame / --task.
PLUG_PROFILES = {
    "sfp": {
        "task": "insert SFP",
        # NIC card hosts the SFP port; default layout is slot 0, mount 0.
        "port_frame": "task_board/nic_card_mount_0/sfp_port_0_link",
        "plug_frame": "cable_0/sfp_tip_link",
        # SFP guide rails absorb small overshoots — moderately fast is fine.
        "approach_max_v": 0.04,
        "align_max_v": 0.02,
        "insert_max_v": 0.010,
        "insert_kp": 2.0,
    },
    "sc": {
        "task": "insert SC",
        "port_frame": "task_board/sc_port_0/sc_port_base_link",
        "plug_frame": "cable_0/sc_tip_link",
        # SC has no guide rails; the exact failure mode documented in
        # docs/sc_insertion_problem.md is oscillation under too-high kp.
        # Back off the proportional gain and crawl the descent.
        "approach_max_v": 0.03,
        "align_max_v": 0.012,
        "insert_max_v": 0.004,
        "insert_kp": 1.2,
    },
}

# Backwards-compat exports (old imports referenced these by name).
TASK_STR = PLUG_PROFILES["sfp"]["task"]
DEFAULT_PORT_FRAME = PLUG_PROFILES["sfp"]["port_frame"]
DEFAULT_PLUG_FRAME = PLUG_PROFILES["sfp"]["plug_frame"]


@dataclass
class EpisodeResult:
    success: bool = False
    aborted: bool = False
    abort_reason: str = ""
    frames: list = field(default_factory=list)
    max_force: float = 0.0
    time_above_force_threshold: float = 0.0
    duration: float = 0.0


def run_episode(node, cameras, port_frame, plug_frame, fps=30, max_time=60.0,
                profile=None):
    """Execute one insertion episode with full safety monitoring."""
    node.reset_episode_monitors()
    result = EpisodeResult()

    # Use sim time exclusively for the control loop: ``node`` has
    # use_sim_time=True, so self.get_clock() advances with Gazebo.  Wall
    # time (time.monotonic) would drift relative to the 30 Hz dataset
    # frame rate whenever Gazebo runs faster or slower than real time.
    t_sim_start = node.get_clock().now()
    last_pos = None
    stuck_counter = 0
    last_action = dict(ZERO_ACTION)

    def step(action_dict):
        """Record one frame and send velocity command.

        Returns False if the frame could not be recorded (e.g. a camera
        stall); the caller drops the step so we never write a corrupted
        frame into the dataset.
        """
        try:
            images = cameras.get_images()
        except RuntimeError as e:
            logger.warning(f"    skipping frame: {e}")
            return False
        obs_dict = node.get_observation_dict()
        result.frames.append((obs_dict, action_dict, images))
        node.send_velocity(
            [action_dict["linear.x"], action_dict["linear.y"], action_dict["linear.z"]],
            [action_dict["angular.x"], action_dict["angular.y"], action_dict["angular.z"]],
        )
        return True

    def elapsed_sim() -> float:
        return (node.get_clock().now() - t_sim_start).nanoseconds / 1e9

    def check_abort():
        """Check all abort conditions. Returns reason string or None."""
        nonlocal last_pos, stuck_counter
        if elapsed_sim() > max_time:
            return "timeout"
        if node.collision_detected.is_set():
            return "off-limit collision"
        if node.is_force_abort():
            return f"force too high ({node.get_force_stats()['current']:.1f}N > {FORCE_ABORT}N)"

        # Stuck detection: if TCP hasn't moved >0.5mm across ~2s of sim
        # time (measured in control ticks since the outer loop runs at
        # ``fps``).  Resets on any motion above the threshold.
        cur_pos, _ = node.get_tcp_pose()
        if last_pos is not None:
            if np.linalg.norm(cur_pos - last_pos) < 0.0005:
                stuck_counter += 1
                if stuck_counter > fps * 2:
                    return "robot stuck (no motion for 2s)"
            else:
                stuck_counter = 0
        last_pos = cur_pos.copy()
        return None

    # Sim-time rate: ticks at ``fps`` using the Gazebo clock, not wall
    # clock.  create_rate() must be called inside the node's executor
    # context; since the node is being spun on a background thread,
    # rate.sleep() will block until the next sim tick arrives.
    rate = node.create_rate(fps)

    def run_phase(name, z_start, z_end, gains, max_steps, is_insertion=False):
        """Run a control phase. Returns True if should continue, False if aborted/done."""
        nonlocal last_action
        logger.info(f"  Phase: {name}")
        if z_start == z_end:
            z_values = [z_start] * max_steps
        else:
            z_values = np.linspace(z_start, z_end, num=max_steps)

        for i, z in enumerate(z_values):
            if node.insertion_detected.is_set():
                return True

            reason = check_abort()
            if reason:
                result.aborted = True
                result.abort_reason = reason
                return False

            try:
                target_pos, target_quat = compute_target_pose(
                    node, port_frame, plug_frame, z_offset=z
                )
                cur_pos, cur_quat = node.get_tcp_pose()
                action, dist = compute_velocity_action(
                    cur_pos, cur_quat, target_pos, target_quat, gains,
                    insertion_phase=is_insertion,
                )
                if step(action):
                    last_action = action

                # Log force periodically during insertion phase
                if is_insertion and i % (fps * 2) == 0:
                    fs = node.get_force_stats()
                    logger.info(f"    z={z:.4f} dist={dist:.4f}m force={fs['current']:.1f}N "
                                f"(max={fs['max']:.1f}N)")

                # Early convergence for approach/alignment phases
                if not is_insertion and dist < 0.003:
                    logger.info(f"    Converged (dist={dist:.4f}m)")
                    break

            except TransformException as e:
                # Do NOT record a frame with zero action on transient TF
                # failure — that pollutes the training distribution with
                # 'held still' frames in the middle of active motion.
                # Keep publishing the last valid velocity so the robot
                # holds course and wait for TF to recover.
                logger.warning(f"    TF failed, holding last action: {e}")
                node.send_velocity(
                    [last_action["linear.x"], last_action["linear.y"], last_action["linear.z"]],
                    [last_action["angular.x"], last_action["angular.y"], last_action["angular.z"]],
                )

            rate.sleep()
        return True

    # Per-plug gain schedule (see PLUG_PROFILES).  SFP uses the original
    # tuning that historically works; SC uses lower kp and slower descent
    # because the SC port has no guide rails to absorb overshoot.
    prof = profile or PLUG_PROFILES["sfp"]

    # Phase 1: Approach — move above port
    approach_gains = ControllerGains(
        kp_linear=1.5, max_linear_vel=prof["approach_max_v"],
        kp_angular=2.0, max_angular_vel=0.3,
    )
    if not run_phase("Approach", z_start=0.12, z_end=0.12, gains=approach_gains, max_steps=fps * 8):
        node.stop_robot()
        result.duration = elapsed_sim()
        return result

    # Phase 2: Fine alignment — lower and correct
    fine_gains = ControllerGains(
        kp_linear=2.0, max_linear_vel=prof["align_max_v"],
        kp_angular=2.5, max_angular_vel=0.2,
    )
    if not run_phase("Fine align", z_start=0.06, z_end=0.06, gains=fine_gains, max_steps=fps * 4):
        node.stop_robot()
        result.duration = elapsed_sim()
        return result

    # Phase 3: Insertion descent — slow and force-aware
    insert_gains = ControllerGains(
        kp_linear=prof["insert_kp"],
        max_linear_vel=prof["insert_max_v"],
        insertion_linear_vel=prof["insert_max_v"] * 0.8,
        kp_angular=2.5, max_angular_vel=0.15,
    )
    if not run_phase("Insertion", z_start=0.06, z_end=-0.015, gains=insert_gains,
                     max_steps=fps * 12, is_insertion=True):
        node.stop_robot()
        result.duration = elapsed_sim()
        return result

    # Phase 4: Hold — wait for insertion event
    logger.info("  Phase: Hold")
    for _ in range(fps * 3):
        if node.insertion_detected.is_set():
            break
        step(ZERO_ACTION)
        rate.sleep()

    node.stop_robot()

    # Collect results
    result.success = node.insertion_detected.is_set()
    fs = node.get_force_stats()
    result.max_force = fs["max"]
    result.time_above_force_threshold = fs["time_above_threshold"]
    result.duration = elapsed_sim()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Automated SFP/SC demo collection")
    ap.add_argument("--dataset-root", required=False,
                    help="Full path for the dataset (omit with --dry-run).")
    ap.add_argument("--repo-id", default=None,
                    help="Lerobot repo_id; defaults to local/aic_<plug>_auto.")
    ap.add_argument("--num-episodes", type=int, default=20, help="Target successful episodes")
    ap.add_argument("--max-attempts", type=int, default=50, help="Max total attempts")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--plug-type", choices=list(PLUG_PROFILES.keys()), default="sfp",
                    help="Which plug profile to target; selects default frames, task "
                         "string, and per-phase gains.  Override individual fields with "
                         "--port-frame / --plug-frame / --task.")
    ap.add_argument("--port-frame", default=None,
                    help="Override TF frame for target port (default: plug profile)")
    ap.add_argument("--plug-frame", default=None,
                    help="Override TF frame for cable plug tip (default: plug profile)")
    ap.add_argument("--task", default=None,
                    help="Override task string (default: plug profile)")
    ap.add_argument("--resume", action="store_true", help="Resume existing dataset")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--max-episode-time", type=float, default=60.0, help="Max seconds per attempt")
    ap.add_argument("--discard-high-force", action="store_true",
                    help="Discard episodes where force exceeded scoring threshold for >1s")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run one full episode without creating a dataset.  Useful "
                         "for verifying Gazebo + TF + cameras + F/T are all working "
                         "before committing to a long session.")
    args = ap.parse_args()

    # Resolve plug-profile-driven defaults (allowing CLI overrides).
    profile = dict(PLUG_PROFILES[args.plug_type])
    port_frame = args.port_frame or profile["port_frame"]
    plug_frame = args.plug_frame or profile["plug_frame"]
    task_str   = args.task       or profile["task"]
    repo_id    = args.repo_id    or f"local/aic_{args.plug_type}_auto"

    if not args.dry_run and not args.dataset_root:
        ap.error("--dataset-root is required unless --dry-run is set")

    rclpy.init()
    node = AutoCollectNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    cameras = CameraCapture(node, {
        "left_camera": "/left_camera/image",
        "center_camera": "/center_camera/image",
        "right_camera": "/right_camera/image",
    }, scale=0.25)

    # --- Pre-flight checks (fail fast with actionable diagnostics) --------
    logger.info(f"Plug profile: {args.plug_type}  |  Task: {task_str!r}")
    logger.info(f"Port frame: {port_frame}")
    logger.info(f"Plug frame: {plug_frame}")
    logger.info(f"Dataset: {'<DRY-RUN>' if args.dry_run else args.dataset_root} (repo_id={repo_id})")

    logger.info("Waiting for controller_state + joint_states + wrench ...")
    try:
        node.wait_for_state(timeout=15.0, require_wrench=True)
    except TimeoutError as e:
        logger.error(
            f"{e}\n"
            "Hint: is Gazebo running with ground_truth:=true and the "
            "aic_controller active?  The wrench topic is specifically "
            "/fts_broadcaster/wrench — if that's missing, the F/T "
            "sensor bringup failed."
        )
        rclpy.shutdown()
        return

    logger.info("Waiting for cameras ...")
    if not cameras.wait_for_images(timeout=15.0):
        logger.error(
            "No frames from one or more cameras after 15s.  Expected topics:\n"
            "  /left_camera/image, /center_camera/image, /right_camera/image\n"
            "Check with: ros2 topic hz /left_camera/image"
        )
        rclpy.shutdown()
        return

    logger.info("Waiting for TF frames...")
    tf_ok = False
    for _ in range(100):
        try:
            node.lookup_tf("base_link", port_frame)
            node.lookup_tf("base_link", plug_frame)
            node.lookup_tf("base_link", "gripper/tcp")
            tf_ok = True
            break
        except TransformException:
            time.sleep(0.2)
    if not tf_ok:
        logger.error(f"TF frames not found. Tried:\n"
                     f"  port: {port_frame}\n"
                     f"  plug: {plug_frame}\n"
                     f"  gripper: gripper/tcp\n"
                     f"Is ground_truth:=true set? Is the scene spawned with "
                     f"the appropriate rails for --plug-type={args.plug_type}?")
        rclpy.shutdown()
        return
    logger.info("TF ready.")

    # Sanity check: is the cable still in the gripper, or did the
    # CablePlugin's startup race drop it on the floor?  The cable's
    # plug_link is supposed to be held at roughly the gripper's TCP
    # height (~0.25-0.35 m in world frame).  If it's near the ground
    # (z < 0.1 m) the gripper lost the grasp and the whole episode
    # is doomed -- abort now with an actionable message instead of
    # letting the operator wait through a 30-second approach that
    # can never converge.
    try:
        plug_tf = node.lookup_tf("base_link", plug_frame)
        plug_z_in_base = plug_tf.transform.translation.z
        # base_link is ~0.94 m below world origin on this rig; a cable
        # grasped near world z~0.3 shows up near base-link z~-0.65.  A
        # cable on the floor (world z~0.02) shows up near base-link
        # z~-0.92.  Threshold chosen conservatively for the default
        # robot home pose.
        if plug_z_in_base < -0.85:
            logger.error(
                "Cable appears to be on the table, not in the gripper!\n"
                f"  {plug_frame} z-in-base_link = {plug_z_in_base:.3f} m\n"
                "  Expected something like -0.6 to -0.7 when grasped.\n\n"
                "This is the CablePlugin startup race in aic_gazebo/src/CablePlugin.cc:\n"
                "the plugin won't pin the cable in place until it finds\n"
                "ur5e::ati/tool_link in the ECM, and during the ~2 s it takes\n"
                "for the arm model to load, the cable free-falls under gravity.\n"
                "By the time the plugin is ready to attach, the cable is too far\n"
                "from the gripper for the fixed joint to pull it back.\n\n"
                "Kill Gazebo (Ctrl+C in terminal 1) and relaunch.  The race is\n"
                "CPU-load-dependent, so it often resolves on the next try.  If\n"
                "it happens consistently, we need a plugin-side fix."
            )
            rclpy.shutdown()
            return
        logger.info(f"Cable grasp OK: {plug_frame} z-in-base_link = {plug_z_in_base:.3f} m")
    except TransformException as e:
        logger.warning(f"Could not check cable grasp state: {e}")

    # Confirm we're actually receiving wrench updates (tared).  If tare is
    # still zeros at this point, either the controller didn't tare or we
    # grabbed the snapshot too early.
    w = node.get_tared_wrench()
    logger.info(f"Initial tared wrench: force={w[:3].round(3).tolist()}  "
                f"torque={w[3:].round(3).tolist()}")
    if np.allclose(w, 0, atol=1e-6) and np.allclose(node._tare_force, 0, atol=1e-6):
        logger.warning(
            "Tare offset is zero — the controller may not have tared yet. "
            "Consider relaunching Gazebo and waiting a few seconds before retrying."
        )

    if args.dry_run:
        logger.info("=== DRY RUN: executing one episode without saving ===")
        ep = run_episode(
            node, cameras,
            port_frame=port_frame, plug_frame=plug_frame,
            fps=args.fps, max_time=args.max_episode_time,
            profile=profile,
        )
        node.stop_robot()
        logger.info(f"\nDry-run result: success={ep.success}  aborted={ep.aborted} "
                    f"({ep.abort_reason!r})  frames={len(ep.frames)}  "
                    f"max_force={ep.max_force:.1f}N  duration={ep.duration:.1f}s")
        if ep.frames:
            obs = ep.frames[-1][0]
            w_keys = [k for k in obs if k.startswith("wrench")]
            logger.info(f"Sample final-frame wrench channels: "
                        f"{ {k: round(obs[k], 3) for k in w_keys} }")
        rclpy.shutdown()
        return

    dataset = create_or_resume_dataset(
        repo_id, args.dataset_root, args.fps, args.resume, args.vcodec
    )

    successes = 0
    attempts = 0
    discarded_force = 0
    discarded_collision = 0
    discarded_stuck = 0
    discarded_failed = 0

    logger.info(f"=== Starting collection: {args.num_episodes} episodes, max {args.max_attempts} attempts ===")

    try:
        while successes < args.num_episodes and attempts < args.max_attempts:
            attempts += 1
            logger.info(f"\n--- Attempt {attempts} | Saved: {successes}/{args.num_episodes} ---")

            ep = run_episode(
                node, cameras,
                port_frame=port_frame,
                plug_frame=plug_frame,
                fps=args.fps,
                max_time=args.max_episode_time,
                profile=profile,
            )

            # Decision: save or discard
            save = False
            if ep.aborted:
                reason = ep.abort_reason
                logger.warning(f"  ABORTED: {reason}")
                if "collision" in reason:
                    discarded_collision += 1
                elif "stuck" in reason:
                    discarded_stuck += 1

            elif not ep.success:
                logger.info(f"  FAILED: no insertion detected ({ep.duration:.1f}s)")
                discarded_failed += 1

            elif args.discard_high_force and ep.time_above_force_threshold > 1.0:
                logger.warning(f"  DISCARDED: force above {FORCE_THRESHOLD}N for "
                               f"{ep.time_above_force_threshold:.2f}s (>{1.0}s limit)")
                discarded_force += 1

            elif len(ep.frames) <= 10:
                logger.info("  DISCARDED: episode too short")

            else:
                save = True

            if save:
                logger.info(f"  SUCCESS! Saving {len(ep.frames)} frames "
                            f"(max_force={ep.max_force:.1f}N, "
                            f"force_above_threshold={ep.time_above_force_threshold:.2f}s, "
                            f"duration={ep.duration:.1f}s)")
                for obs_dict, action_dict, images in ep.frames:
                    frame = build_frame(obs_dict, action_dict, images,
                                        dataset.meta.features, task_str)
                    dataset.add_frame(frame)
                dataset.save_episode()
                successes += 1
            else:
                dataset.clear_episode_buffer()

            # Pause between episodes
            node.stop_robot()
            time.sleep(2.0)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user (Ctrl+C)")
    finally:
        node.stop_robot()
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Successful episodes: {successes}")
        logger.info(f"  Total attempts:      {attempts}")
        logger.info(f"  Success rate:        {100*successes/max(attempts,1):.0f}%")
        logger.info(f"  Discarded breakdown:")
        logger.info(f"    Insertion failed:  {discarded_failed}")
        logger.info(f"    High force:        {discarded_force}")
        logger.info(f"    Collision:         {discarded_collision}")
        logger.info(f"    Robot stuck:       {discarded_stuck}")
        logger.info(f"{'='*60}")

        dataset.finalize()
        node.destroy_node()
        rclpy.shutdown()

    logger.info(f"Dataset saved at: {args.dataset_root}  (repo_id={repo_id})")


if __name__ == "__main__":
    main()
