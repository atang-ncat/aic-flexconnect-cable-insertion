#!/usr/bin/env python3
"""
Automated demo collection using CheatCode-style ground-truth targeting
with velocity commands compatible with the LeRobot training pipeline.

Architecture:
    - Uses the same MotionUpdate/velocity interface as lerobot-record
    - Uses tf2_ros for ground-truth port/plug frame lookups
    - Proportional controller converts pose targets -> 6D velocity actions
    - Records directly into LeRobot dataset format via LeRobotDataset API
    - Filters episodes by /scoring/insertion_event success
    - Monitors force and collisions to discard bad episodes

Recommended per-launch workflow (inside distrobox, with Gazebo +
ground_truth:=true running):

    cd /run/host/scratch2/atang/ws_aic/src/aic

    # First launch:
    pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
        --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
        --repo-id atang/aic_sfp_auto \
        --num-episodes 1 --max-attempts 5 --exit-on-success

    # Ctrl-C Gazebo, relaunch, then:
    pixi run python3 /run/host/scratch2/atang/ws_aic/scripts/auto_collect.py \
        --dataset-root /run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp \
        --repo-id atang/aic_sfp_auto \
        --num-episodes 1 --max-attempts 5 --exit-on-success --resume

    # repeat until you have N episodes.

Why per-launch: the scene only spawns one cable, and once it's inserted
the cable is latched into the port. Trying another attempt in the same
launch would either pull it back out or wedge the gripper. The
--reset-scene flag (which was intended to avoid relaunches) is currently
broken — see the note in main() — so the manual-relaunch loop is the
supported path.

The dataset it produces can be merged with teleop data for training.
"""

import argparse
import math
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

# Force unbuffered stdout/stderr so log messages appear immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench, WrenchStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

try:
    from ros_gz_interfaces.msg import Contacts
    HAS_CONTACTS_MSG = True
except ImportError:
    HAS_CONTACTS_MSG = False

try:
    # Standard ROS2 simulator interface (REP-2014). Used to reset Gazebo
    # back to its initial state between attempts — avoids relaunching.
    from simulation_interfaces.srv import ResetSimulation
    HAS_RESET_SIM_SRV = True
except ImportError:
    HAS_RESET_SIM_SRV = False

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


def quat_nlerp(q1_wxyz, q2_wxyz, alpha):
    """Normalized LERP between two quaternions. Good enough for small-to-medium angles."""
    q1 = np.array(q1_wxyz, dtype=float)
    q2 = np.array(q2_wxyz, dtype=float)
    # Ensure shortest-path (dot product >= 0)
    if float(np.dot(q1, q2)) < 0.0:
        q2 = -q2
    q = (1.0 - alpha) * q1 + alpha * q2
    n = np.linalg.norm(q)
    if n < 1e-9:
        return tuple(q1_wxyz)
    q = q / n
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


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
            rclpy.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True),
        ])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_controller_state = None
        self.last_joint_states = None
        self.insertion_detected = Event()
        self.collision_detected = Event()

        # Force monitoring state
        self._force_lock = Lock()
        self._tare_force = np.zeros(3)
        self._current_force_mag = 0.0
        self._max_force_mag = 0.0
        self._time_above_threshold = 0.0
        self._last_wrench_time = None

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

        self.motion_pub = self.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10,
        )

        # Service client for Gazebo scene reset. Created lazily — only
        # matters when --reset-scene is used.
        self._reset_sim_client = None
        if HAS_RESET_SIM_SRV:
            self._reset_sim_client = self.create_client(
                ResetSimulation, "/gz_server/reset_simulation"
            )

    def _controller_state_cb(self, msg):
        self.last_controller_state = msg
        tare = msg.fts_tare_offset.wrench.force
        with self._force_lock:
            self._tare_force = np.array([tare.x, tare.y, tare.z])

    def _joint_states_cb(self, msg):
        self.last_joint_states = msg

    def _insertion_cb(self, msg):
        # Log every message (not just the first) so we can see whether the
        # `/scoring/insertion_event` bridge is actually firing. Events come
        # from CablePlugin as e.g. "/nic_card_mount_0/sfp_port_0".
        logger.info(f"  [EVENT] /scoring/insertion_event: {msg.data}")
        self.insertion_detected.set()

    def _collision_cb(self, msg):
        if len(msg.contacts) > 0:
            logger.warning("  [EVENT] Off-limit collision detected!")
            self.collision_detected.set()

    def _wrench_cb(self, msg):
        raw = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        with self._force_lock:
            tared = raw - self._tare_force
            mag = float(np.linalg.norm(tared))
            self._current_force_mag = mag
            if mag > self._max_force_mag:
                self._max_force_mag = mag
            now = time.monotonic()
            if self._last_wrench_time is not None and mag > FORCE_THRESHOLD:
                self._time_above_threshold += now - self._last_wrench_time
            self._last_wrench_time = now

    def reset_episode_monitors(self):
        """Reset per-episode monitoring state."""
        self.insertion_detected.clear()
        self.collision_detected.clear()
        with self._force_lock:
            self._current_force_mag = 0.0
            self._max_force_mag = 0.0
            self._time_above_threshold = 0.0
            self._last_wrench_time = None

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

    def wait_for_state(self, timeout=10.0):
        t0 = time.monotonic()
        while self.last_controller_state is None or self.last_joint_states is None:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("Timed out waiting for controller_state / joint_states")
            time.sleep(0.1)

    def lookup_tf(self, target, source):
        return self.tf_buffer.lookup_transform(target, source, Time())

    def plug_to_port_distance(self, port_frame, plug_frame):
        """TF-based measure of how close the plug tip is to the port.

        Returns ``(total_m, axial_m, lateral_m)`` where:
          * total_m   : Euclidean distance plug -> port (meters)
          * axial_m   : signed offset along the port's +Z (insertion) axis;
                       positive means plug is ABOVE the port face
          * lateral_m : in-plane distance from the port's insertion axis.
        Raises TransformException if TF is unavailable.
        """
        port_tf = self.lookup_tf("base_link", port_frame)
        plug_tf = self.lookup_tf("base_link", plug_frame)
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
        q_port = (
            port_tf.transform.rotation.w, port_tf.transform.rotation.x,
            port_tf.transform.rotation.y, port_tf.transform.rotation.z,
        )
        axis = quat_to_rotmat(q_port)[:, 2]  # port +Z = insertion axis
        delta = plug_pos - port_pos
        axial = float(np.dot(delta, axis))
        lateral = float(np.linalg.norm(delta - axial * axis))
        total = float(np.linalg.norm(delta))
        return total, axial, lateral

    def send_velocity(self, linear, angular, frame_id="base_link"):
        """Send a velocity (twist) target. Used only for stop_robot()."""
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

    def send_pose_target(self, position, quat_wxyz, frame_id="base_link"):
        """Send an absolute pose target (MODE_POSITION).

        Matches CheatCode's `set_pose_target` path which is the proven way to
        drive the aic_controller. Use this for all motion during episodes.
        """
        msg = MotionUpdate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose = Pose(
            position=Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2])
            ),
            orientation=Quaternion(
                w=float(quat_wxyz[0]), x=float(quat_wxyz[1]),
                y=float(quat_wxyz[2]), z=float(quat_wxyz[3]),
            ),
        )
        # Same defaults as CheatCode / aic_model.policy.set_pose_target
        msg.target_stiffness = np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten()
        msg.target_damping = np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        self.motion_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity([0, 0, 0], [0, 0, 0])

    def retreat_vertical(self, lift_m=0.15, duration_s=2.5, fps=30):
        """Smoothly lift the TCP straight up by `lift_m` in base_link Z+.

        Used between attempts so the next episode's Approach doesn't yank a
        seated cable back out, and to un-wedge the arm after a failed run.
        Does not abort on force/stuck — just interpolates and returns.
        """
        try:
            start_pos, start_quat = self.get_tcp_pose()
        except Exception:
            logger.warning("  retreat_vertical: TCP pose unavailable, skipping")
            return
        target_pos = start_pos + np.array([0.0, 0.0, lift_m])
        steps = max(2, int(duration_s * fps))
        for i in range(steps):
            alpha = (i + 1) / steps
            cmd_pos = (1.0 - alpha) * start_pos + alpha * target_pos
            self.send_pose_target(cmd_pos, start_quat)
            time.sleep(1.0 / fps)

    def reset_simulation(self, port_frame, plug_frame, timeout_s=20.0):
        """Call `/gz_server/reset_simulation` and wait for the scene to settle.

        This is the core of Level-1 automation: instead of manually killing
        and relaunching Gazebo between episodes, we ask gz_server to rewind
        the world to its initial state. After the reset we:
          1. Re-acquire TF frames for the port and plug,
          2. Wait for the controller state + joint states to start publishing
             fresh values (the tare offset will also reset),
          3. Clear per-episode monitors.

        Returns True on success, False on timeout/failure (caller should
        consider aborting the whole run).
        """
        if not HAS_RESET_SIM_SRV or self._reset_sim_client is None:
            logger.error("  reset_simulation: simulation_interfaces not available")
            return False

        if not self._reset_sim_client.wait_for_service(timeout_sec=5.0):
            logger.error("  reset_simulation: /gz_server/reset_simulation not available")
            return False

        logger.info("  [RESET] Calling /gz_server/reset_simulation ...")
        req = ResetSimulation.Request()
        # scope=0 (SCOPE_DEFAULT): let the simulator decide. For gz_server
        # this resets the world to its initial state, which is what we want.
        future = self._reset_sim_client.call_async(req)

        # Drop stale state so wait_for_state below blocks until the
        # simulator genuinely re-publishes after the reset.
        self.last_controller_state = None
        self.last_joint_states = None
        self.reset_episode_monitors()

        t_start = time.monotonic()
        while not future.done():
            if time.monotonic() - t_start > timeout_s:
                logger.error("  [RESET] service call timed out")
                return False
            time.sleep(0.05)

        try:
            resp = future.result()
        except Exception as e:
            logger.error(f"  [RESET] service raised: {e}")
            return False

        # simulation_interfaces uses a Result subfield with `result` code.
        # Accept both the REP-2014 style (result.result) and the older
        # success/bool style if present.
        ok = True
        if hasattr(resp, "result") and hasattr(resp.result, "result"):
            ok = (resp.result.result == 1)  # RESULT_OK = 1
            if not ok:
                logger.error(
                    f"  [RESET] service returned error: "
                    f"{getattr(resp.result, 'error_message', '')}"
                )
        elif hasattr(resp, "success"):
            ok = bool(resp.success)

        if not ok:
            return False

        # Wait for controller to republish state post-reset.
        try:
            self.wait_for_state(timeout=10.0)
        except TimeoutError:
            logger.error("  [RESET] controller state never came back after reset")
            return False

        # Wait for TF frames (the scoring TF relay can take a beat to re-emit).
        tf_deadline = time.monotonic() + 10.0
        while time.monotonic() < tf_deadline:
            try:
                self.lookup_tf("base_link", port_frame)
                self.lookup_tf("base_link", plug_frame)
                self.lookup_tf("base_link", "gripper/tcp")
                break
            except TransformException:
                time.sleep(0.2)
        else:
            logger.error("  [RESET] TF frames never reappeared after reset")
            return False

        # Extra settle time: the cable may still be falling / snapping into
        # its initial attached-to-gripper pose right after the reset.
        time.sleep(1.5)
        logger.info("  [RESET] Scene reset complete.")
        return True

    def get_tcp_pose(self):
        cs = self.last_controller_state
        p = cs.tcp_pose
        return (
            np.array([p.position.x, p.position.y, p.position.z]),
            (p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
        )

    def get_observation_dict(self):
        """Build observation dict matching AICRobotAICController format."""
        cs = self.last_controller_state
        js = self.last_joint_states
        p = cs.tcp_pose
        v = cs.tcp_velocity
        e = cs.tcp_error
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
        }


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

class CameraCapture:
    """Subscribes to ROS image topics and captures the latest frame."""
    def __init__(self, node, cameras, scale=0.25):
        self.frames = {}
        self.scale = scale
        for name, topic in cameras.items():
            self.frames[name] = None
            node.create_subscription(
                Image, topic,
                lambda msg, n=name: self._img_cb(n, msg),
                qos_profile_sensor_data,
            )

    def _img_cb(self, name, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if self.scale != 1.0:
            arr = cv2.resize(arr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.frames[name] = arr

    def get_images(self):
        return {k: (v.copy() if v is not None else np.zeros((256, 288, 3), dtype=np.uint8))
                for k, v in self.frames.items()}

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


@dataclass
class LateralIntegrator:
    """CheatCode-style integrator for lateral (XY-in-port-frame) plug error.

    During insertion descent, any residual lateral offset between the plug tip
    and the port mouth is accumulated and fed back as a target-pose correction.
    Mirrors the logic in CheatCode.calc_gripper_pose().
    """
    x: float = 0.0
    y: float = 0.0
    max_windup: float = 0.05  # matches CheatCode's _max_integrator_windup

    def reset(self):
        self.x = 0.0
        self.y = 0.0

    def update(self, u_error: float, v_error: float):
        self.x = float(np.clip(self.x + u_error, -self.max_windup, self.max_windup))
        self.y = float(np.clip(self.y + v_error, -self.max_windup, self.max_windup))


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

def compute_target_pose(
    node,
    port_frame,
    plug_frame,
    z_offset=0.1,
    integrator: "LateralIntegrator | None" = None,
    p_gain_lateral: float = 0.05,
    i_gain_lateral: float = 0.05,
    reset_integrator: bool = False,
):
    """
    Compute gripper target pose that aligns plug with port,
    offset along the port's insertion axis by z_offset.

    If `integrator` is provided, applies CheatCode-style P+I correction on
    the plug's lateral (port-local XY) error. This is needed for reliable
    descent — without it, steady-state lateral offset causes insertion to jam.
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
    insertion_axis = R_port[:, 2]   # port local Z = approach direction
    lateral_u = R_port[:, 0]        # port local X
    lateral_v = R_port[:, 1]        # port local Y
    plug_to_gripper = gripper_pos - plug_pos

    # Lateral P+I correction (CheatCode parity)
    lateral_correction = np.zeros(3)
    if integrator is not None:
        error_vec = port_pos - plug_pos
        u_error = float(np.dot(error_vec, lateral_u))
        v_error = float(np.dot(error_vec, lateral_v))
        if reset_integrator:
            integrator.reset()
        else:
            integrator.update(u_error, v_error)
        lateral_correction = (
            lateral_u * (p_gain_lateral * u_error + i_gain_lateral * integrator.x)
            + lateral_v * (p_gain_lateral * v_error + i_gain_lateral * integrator.y)
        )

    target_plug_pos = port_pos + insertion_axis * z_offset + lateral_correction
    target_gripper_pos = target_plug_pos + plug_to_gripper

    q_diff = quat_multiply(q_port, quat_conjugate(q_plug))
    q_gripper_target = quat_multiply(q_diff, q_gripper)

    return target_gripper_pos, q_gripper_target


def compute_pose_diagnostics(
    node, port_frame, plug_frame, cur_pos, target_pos
):
    """Snapshot of *why* an insertion attempt is or isn't progressing.

    All linear values are in meters and decomposed in the port frame:
      * +Z (`port_axis`) is the insertion direction (positive = above the
        port face, ie. not yet inserted).
      * X/Y are in-plane.

    Returned fields:
      plug_axial_m       : plug tip along port +Z. Negative means seated.
      plug_lat_m         : plug tip distance to the port insertion axis.
      plug_yaw_deg       : rotation of the plug frame about port +Z relative
                           to the port frame. Critical for SFP because the
                           connector body is keyed.
      plug_tilt_deg      : angle between plug +Z and port +Z. Non-zero means
                           the plug is not pointed straight into the port.
      grip_track_axial_m : (cur_gripper - cmd_gripper) along port +Z. Tells
                           us if the impedance controller is lagging the
                           commanded pose along the insertion axis.
      grip_track_lat_m   : same but lateral. Lag in this direction hints
                           that lateral correction is being absorbed by
                           tracking error rather than by plug motion.
      cable_lat_m        : lateral component of (plug - gripper) in port
                           frame. Big values = the cable is bending sideways
                           between the gripper jaws and the SFP plug tip.
    """
    port_tf = node.lookup_tf("base_link", port_frame)
    plug_tf = node.lookup_tf("base_link", plug_frame)
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
    q_port = (
        port_tf.transform.rotation.w, port_tf.transform.rotation.x,
        port_tf.transform.rotation.y, port_tf.transform.rotation.z,
    )
    q_plug = (
        plug_tf.transform.rotation.w, plug_tf.transform.rotation.x,
        plug_tf.transform.rotation.y, plug_tf.transform.rotation.z,
    )
    R_port = quat_to_rotmat(q_port)
    R_plug = quat_to_rotmat(q_plug)
    port_axis = R_port[:, 2]

    delta = plug_pos - port_pos
    plug_axial = float(np.dot(delta, port_axis))
    plug_lat = float(np.linalg.norm(delta - plug_axial * port_axis))

    plug_axis = R_plug[:, 2]
    cos_tilt = float(np.clip(np.dot(plug_axis, port_axis), -1.0, 1.0))
    plug_tilt_deg = math.degrees(math.acos(cos_tilt))

    # Yaw: express plug rotation in the port frame and read the rotation
    # about the port +Z axis. This is well-defined when tilt is small.
    R_in_port = R_port.T @ R_plug
    plug_yaw_deg = math.degrees(math.atan2(R_in_port[1, 0], R_in_port[0, 0]))

    grip_err = cur_pos - target_pos
    grip_track_axial = float(np.dot(grip_err, port_axis))
    grip_track_lat = float(np.linalg.norm(grip_err - grip_track_axial * port_axis))

    plug_grip_vec = plug_pos - cur_pos
    plug_grip_axial = float(np.dot(plug_grip_vec, port_axis))
    cable_lat = float(np.linalg.norm(plug_grip_vec - plug_grip_axial * port_axis))

    return {
        "plug_axial_m": plug_axial,
        "plug_lat_m": plug_lat,
        "plug_yaw_deg": plug_yaw_deg,
        "plug_tilt_deg": plug_tilt_deg,
        "grip_track_axial_m": grip_track_axial,
        "grip_track_lat_m": grip_track_lat,
        "cable_lat_m": cable_lat,
    }


def format_diag_line(d, prefix="    "):
    """Render `compute_pose_diagnostics` output as a one-line log message."""
    return (
        f"{prefix}plug ax={d['plug_axial_m']*1000:+6.1f}mm "
        f"lat={d['plug_lat_m']*1000:5.1f}mm "
        f"yaw={d['plug_yaw_deg']:+6.1f}deg "
        f"tilt={d['plug_tilt_deg']:5.1f}deg | "
        f"grip-trk ax={d['grip_track_axial_m']*1000:+6.1f}mm "
        f"lat={d['grip_track_lat_m']*1000:5.1f}mm | "
        f"cable-lat={d['cable_lat_m']*1000:5.1f}mm"
    )


# ---------------------------------------------------------------------------
# LeRobot dataset helpers
# ---------------------------------------------------------------------------

def create_or_resume_dataset(repo_id, root, fps=30, resume=False, vcodec="libsvtav1"):
    root_path = Path(root)

    # Pre-flight: LeRobotDataset.create() requires the root NOT to exist.
    # A common cause of failure is a previous aborted run leaving behind
    # `root/meta/info.json` — give a clear error pointing at the fix.
    if not resume and root_path.exists():
        raise SystemExit(
            f"\nERROR: dataset root already exists: {root_path}\n"
            f"  - If this is leftover from a previous failed run, delete it:\n"
            f"      rm -rf {root_path}\n"
            f"  - If you want to keep adding episodes to it, rerun with --resume\n"
        )

    logger.info("Importing LeRobotDataset (torch/lerobot first-time import can take ~60s on cold cache)...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    logger.info("Import done.")

    if resume and root_path.exists():
        logger.info(f"Resuming dataset at {root}")
        ds = LeRobotDataset(repo_id, root=root, vcodec=vcodec)
        _normalize_feature_shapes(ds)
        return ds

    logger.info(f"Creating new dataset at {root}")

    # NOTE: shapes must be TUPLES, not lists. lerobot.datasets.utils
    # validate_frame() compares numpy's `value.shape` (tuple) against this
    # field with strict `!=`, and `[6] != (6,)` always. See lerobot issue.
    cam_shape = (256, 288, 3)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (26,),
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
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
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

    logger.info("Calling LeRobotDataset.create() ...")
    ds = LeRobotDataset.create(
        repo_id, fps, root=root, robot_type="ur5e_aic",
        features=features, use_videos=True,
        image_writer_processes=0, image_writer_threads=4 * 3,
        vcodec=vcodec,
    )
    _normalize_feature_shapes(ds)
    logger.info("Dataset created successfully.")
    return ds


def _normalize_feature_shapes(ds):
    """Force every feature's ``shape`` to be a tuple.

    lerobot's ``validate_frame`` uses strict ``!=`` between ``np.ndarray.shape``
    (a tuple) and the ``shape`` entry from the feature dict. When features are
    round-tripped through info.json they become lists, so ``(26,) != [26]``
    raises ValueError even though the shapes are logically identical.
    """
    for feat in ds.meta.features.values():
        if "shape" in feat:
            feat["shape"] = tuple(feat["shape"])


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

TASK_STR = "Insert SFP connector into SFP port on NIC card"

# Correct TF frame names (verified against NIC Card Mount model.sdf and spawn_cable.launch.py)
DEFAULT_PORT_FRAME = "task_board/nic_card_mount_0/sfp_port_0_link"
DEFAULT_PLUG_FRAME = "cable_0/sfp_tip_link"


@dataclass
class EpisodeResult:
    success: bool = False
    aborted: bool = False
    abort_reason: str = ""
    frames: list = field(default_factory=list)
    max_force: float = 0.0
    time_above_force_threshold: float = 0.0
    duration: float = 0.0


def run_episode(node, cameras, port_frame, plug_frame, fps=30, max_time=60.0):
    """Execute one insertion episode with full safety monitoring."""
    node.reset_episode_monitors()
    result = EpisodeResult()
    t0 = time.monotonic()
    # Windowed stuck detection: record (timestamp, pos) samples and abort if
    # total displacement over STUCK_WINDOW seconds is below STUCK_THRESH.
    stuck_window_s = 3.0
    stuck_thresh_m = 0.001   # 1 mm over 3s
    stuck_samples: list = []
    # CheatCode-style lateral integrator (shared across phases for one episode)
    lateral_integrator = LateralIntegrator()

    def step(action_dict, target_pos=None, target_quat_wxyz=None):
        """Record one frame; drive robot by POSE target (MODE_POSITION).

        The action recorded in the dataset is still a velocity (matching
        lerobot-record/teleop conventions), but the robot is actually driven
        by absolute pose targets -- the same proven control path used by
        CheatCode. This avoids the "tracking error not converging" behavior
        we saw when sending velocity commands in a tight loop.
        """
        obs_dict = node.get_observation_dict()
        images = cameras.get_images()
        result.frames.append((obs_dict, action_dict, images))
        if target_pos is not None and target_quat_wxyz is not None:
            node.send_pose_target(target_pos, target_quat_wxyz)

    def check_abort():
        """Check all abort conditions. Returns reason string or None."""
        now = time.monotonic()
        if now - t0 > max_time:
            return "timeout"
        if node.collision_detected.is_set():
            return "off-limit collision"
        if node.is_force_abort():
            return f"force too high ({node.get_force_stats()['current']:.1f}N > {FORCE_ABORT}N)"

        # Stuck detection: max TCP displacement from any sample in the last
        # `stuck_window_s` seconds must exceed `stuck_thresh_m`.
        cur_pos, _ = node.get_tcp_pose()
        stuck_samples.append((now, cur_pos.copy()))
        while stuck_samples and now - stuck_samples[0][0] > stuck_window_s:
            stuck_samples.pop(0)
        if (
            len(stuck_samples) > 1
            and now - stuck_samples[0][0] >= stuck_window_s - 0.1
        ):
            max_disp = max(
                np.linalg.norm(cur_pos - p) for _, p in stuck_samples
            )
            if max_disp < stuck_thresh_m:
                return (
                    f"robot stuck (max {max_disp*1000:.2f}mm in "
                    f"{stuck_window_s:.1f}s < {stuck_thresh_m*1000:.1f}mm)"
                )
        return None

    def run_phase(name, z_start, z_end, gains, max_steps, is_insertion=False,
                  use_integrator=False, reset_integrator_first=False,
                  interpolate_from_current=False, interpolate_steps=None,
                  require_lateral_m=None, require_axial_above_m=None,
                  converge_hold_s=0.3, converge_log_every_s=1.0,
                  diag_log_every_s=2.0):
        """Run a control phase. Returns True if should continue, False if aborted/done.

        If `interpolate_from_current` is True, the FIRST `interpolate_steps`
        ticks (default: all of max_steps) send a linearly interpolated pose
        target going from the current TCP pose to the final computed target.
        This avoids the impedance controller snapping to a distant target and
        slamming into obstacles.

        Note on the force-drag during Approach: with straight-line
        interpolation, the plug typically spends ~3 s being dragged
        across the card face while it transitions from "behind the card"
        (axial ≈ -250 mm) to "in front of the port" (axial ≈ -20 mm).
        That drag produces the Tier-2 scoring penalty. Two-leg
        "lift-then-translate" paths were tried (world +z and port +z)
        and both regressed success rate to 0%; see
        docs/auto_collection_force_notes.md for the diagnostic record.
        The proper fix is a joint-space home move before Approach to
        make the starting pose deterministic — not a Cartesian path
        tweak.

        If `require_lateral_m` and/or `require_axial_above_m` are set, the
        phase holds z constant (z_start) and exits *early* when all set
        conditions have held simultaneously for `converge_hold_s` seconds:
          * plug_lat   < require_lateral_m        (plug over slot axis)
          * plug_axial > require_axial_above_m    (plug above card face)
        If `max_steps` is exhausted without convergence, the attempt is
        aborted with a "<condition> not converged (...)" reason. This
        prevents descending into the card while the plug is still off-
        axis or already dangling past the card edge \u2014 both are
        permanently-stuck failure modes.

        For any phase, a one-line pose diagnostic (axial/lateral/yaw/tilt)
        is logged every `diag_log_every_s` seconds (except during Insertion,
        which logs its own richer every-0.5s message, and during
        convergence-gated phases, which log their own every-1s message).
        """
        try:
            tp, _ = compute_target_pose(node, port_frame, plug_frame, z_offset=z_start)
            cp, _ = node.get_tcp_pose()
            logger.info(
                f"  Phase: {name}  (initial dist-to-target = "
                f"{np.linalg.norm(tp - cp)*100:.1f} cm)"
            )
        except TransformException:
            logger.info(f"  Phase: {name}")
        if z_start == z_end:
            z_values = [z_start] * max_steps
        else:
            z_values = np.linspace(z_start, z_end, num=max_steps)

        # Snapshot of the TCP pose at the START of the phase, used only for
        # the optional interpolation ramp.
        if interpolate_from_current:
            start_pos, start_quat = node.get_tcp_pose()
            ramp_steps = interpolate_steps if interpolate_steps else max_steps

        gated = (require_lateral_m is not None) or (require_axial_above_m is not None)
        converge_ticks = 0
        converge_ticks_needed = max(1, int(round(fps * converge_hold_s)))
        converge_log_every_ticks = max(1, int(round(fps * converge_log_every_s)))
        diag_log_every_ticks = max(1, int(round(fps * diag_log_every_s)))
        last_plug_lat = None     # for post-mortem in abort reason
        last_plug_axial = None

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
                    node, port_frame, plug_frame, z_offset=z,
                    integrator=lateral_integrator if use_integrator else None,
                    reset_integrator=(reset_integrator_first and i == 0),
                )
                cur_pos, cur_quat = node.get_tcp_pose()
                action, dist = compute_velocity_action(
                    cur_pos, cur_quat, target_pos, target_quat, gains, insertion_phase=is_insertion
                )

                if interpolate_from_current and i < ramp_steps:
                    alpha = (i + 1) / ramp_steps
                    cmd_pos = (1.0 - alpha) * start_pos + alpha * target_pos
                    cmd_quat = quat_nlerp(start_quat, target_quat, alpha)
                else:
                    cmd_pos, cmd_quat = target_pos, target_quat
                step(action, target_pos=cmd_pos, target_quat_wxyz=cmd_quat)

                # Log force + pose diagnostics every ~0.5s during insertion.
                # The diagnostics tell us *which dimension* the plug is stuck
                # in (axial vs lateral vs yaw vs tilt) and whether the gripper
                # itself is lagging the commanded pose.
                if is_insertion and i % max(1, fps // 2) == 0:
                    fs = node.get_force_stats()
                    try:
                        d = compute_pose_diagnostics(
                            node, port_frame, plug_frame, cur_pos, target_pos
                        )
                        logger.info(
                            f"    z={z:+.4f} dist={dist:.4f}m "
                            f"force={fs['current']:.1f}N (max={fs['max']:.1f}N)"
                        )
                        logger.info(format_diag_line(d))
                    except TransformException:
                        logger.info(
                            f"    z={z:.4f} dist={dist:.4f}m "
                            f"force={fs['current']:.1f}N (max={fs['max']:.1f}N)"
                        )

                # Two-axis gating: only unlock the next phase once the plug
                # is simultaneously over the slot (lateral) AND above the
                # card face (axial). Either alone is insufficient: we
                # learned the hard way that the plug can be laterally
                # centred while hanging 46 mm past the card edge.
                if gated:
                    try:
                        _, plug_axial, plug_lat = node.plug_to_port_distance(
                            port_frame, plug_frame
                        )
                        last_plug_lat = plug_lat
                        last_plug_axial = plug_axial
                        lat_ok = (
                            require_lateral_m is None
                            or plug_lat < require_lateral_m
                        )
                        ax_ok = (
                            require_axial_above_m is None
                            or plug_axial > require_axial_above_m
                        )
                        if lat_ok and ax_ok:
                            converge_ticks += 1
                        else:
                            converge_ticks = 0
                        if i % converge_log_every_ticks == 0:
                            want = []
                            if require_lateral_m is not None:
                                want.append(f"lat<{require_lateral_m*1000:.1f}mm")
                            if require_axial_above_m is not None:
                                want.append(f"ax>{require_axial_above_m*1000:+.1f}mm")
                            logger.info(
                                f"    lateral={plug_lat*1000:5.1f}mm "
                                f"axial={plug_axial*1000:+6.1f}mm "
                                f"(want {' & '.join(want)} for "
                                f"{converge_hold_s:.1f}s)"
                            )
                        if converge_ticks >= converge_ticks_needed:
                            logger.info(
                                f"    Converged: "
                                f"lateral={plug_lat*1000:.2f}mm "
                                f"axial={plug_axial*1000:+.2f}mm "
                                f"sustained {converge_hold_s:.1f}s"
                            )
                            return True
                    except TransformException:
                        pass
                elif not is_insertion and i % diag_log_every_ticks == 0:
                    # Generic diagnostics for ungated non-insertion phases
                    # (currently: Approach). Gives visibility into where
                    # the plug is as we move toward the port.
                    try:
                        d = compute_pose_diagnostics(
                            node, port_frame, plug_frame, cur_pos, target_pos
                        )
                        logger.info(format_diag_line(d))
                    except TransformException:
                        pass

                # Early convergence for approach/alignment phases
                if (
                    not is_insertion
                    and not gated
                    and dist < 0.003
                ):
                    logger.info(f"    Converged (dist={dist:.4f}m)")
                    break

            except TransformException as e:
                logger.warning(f"    TF failed: {e}")
                step(ZERO_ACTION)

            time.sleep(1.0 / fps)

        # Reached here => the loop ended without early-exit. For gated
        # phases that's a failure: descending past the card with a still-
        # misaligned plug would jam it in the outside-the-slot dead zone.
        if gated:
            parts = []
            if require_lateral_m is not None:
                parts.append(
                    f"lat={last_plug_lat*1000:.1f}mm>"
                    f"{require_lateral_m*1000:.1f}mm"
                    if last_plug_lat is not None else "lat=?"
                )
            if require_axial_above_m is not None:
                parts.append(
                    f"ax={last_plug_axial*1000:+.1f}mm<"
                    f"{require_axial_above_m*1000:+.1f}mm"
                    if last_plug_axial is not None else "ax=?"
                )
            result.aborted = True
            result.abort_reason = (
                f"alignment not converged ({', '.join(parts)} after "
                f"{max_steps/fps:.1f}s)"
            )
            return False

        return True

    # Phase 1: Approach — lift the plug clear of the card face.
    #
    # The cable from gripper/tcp to cable_0/sfp_tip_link is ~20 cm long,
    # so for the plug to end up above the port face the gripper itself
    # has to be held roughly 22 cm above the port (plug dangles below
    # gripper by ~cable length under gravity). An Approach that only
    # aims at z_offset=0.12 leaves the gripper ~10 cm too low and the
    # plug hangs ~45 mm *past* the card edge for the entire phase.
    #
    # We raise the target to z_offset=0.22 and gate Approach on
    # plug_axial > +10 mm (sustained 0.3 s). This way Fine align only
    # starts once the plug has already cleared the card face; before
    # that happens there is nothing useful Fine align can do anyway.
    # If the plug never climbs above the card within 15 s we abort
    # and retry from scratch rather than wedge it against the edge.
    # Straight-line Cartesian interpolation from the current TCP pose
    # to port+0.22 in port frame. See run_phase() for the lift-waypoint
    # machinery that's available but not used here: two independent
    # experiments (world +z lift, port +z lift) both regressed success
    # rate to 0% — Cartesian path shaping alone cannot avoid the cable
    # drag that accumulates ~3 s of >20 N force during Approach. See
    # docs/auto_collection_force_notes.md for the diagnostic data and
    # a sketch of the proper fix (joint-space "go-home" pose before
    # Approach, to guarantee a deterministic starting configuration).
    approach_gains = ControllerGains(kp_linear=1.5, max_linear_vel=0.04, kp_angular=2.0, max_angular_vel=0.3)
    approach_steps = fps * 15
    if not run_phase("Approach", z_start=0.22, z_end=0.22, gains=approach_gains,
                     max_steps=approach_steps, use_integrator=False,
                     interpolate_from_current=True,
                     interpolate_steps=int(approach_steps * 0.6),
                     require_axial_above_m=0.010,
                     converge_hold_s=0.3,
                     diag_log_every_s=2.0):
        node.stop_robot()
        result.duration = time.monotonic() - t0
        return result

    # Phase 2: Fine alignment — hold a safe height ABOVE the port face and
    # wait until the plug is simultaneously laterally over the slot AND
    # axially above the card face before allowing any descent.
    #
    # Held height is z_offset=0.10 (plug target at +10 cm, gripper
    # target at +30 cm given the ~20 cm cable). This keeps the plug
    # well above the card while the lateral P+I settles it over the
    # slot axis. Descending from a plug that is already past the card
    # edge is unrecoverable, so we refuse to start Insertion unless
    # both gates are satisfied.
    #
    # Thresholds:
    #   require_lateral_m=0.003 — roughly half the SFP slot width, giving
    #     clearance for the plug tip to enter the slot rather than next to it.
    #   require_axial_above_m=0.005 — plug must be at least 5 mm above the
    #     port face so that "lower to 0" actually sends the plug into the slot.
    fine_gains = ControllerGains(kp_linear=2.0, max_linear_vel=0.02, kp_angular=2.5, max_angular_vel=0.2)
    if not run_phase("Fine align", z_start=0.10, z_end=0.10, gains=fine_gains,
                     max_steps=fps * 10, use_integrator=True, reset_integrator_first=True,
                     require_lateral_m=0.003, require_axial_above_m=0.005,
                     converge_hold_s=0.3):
        node.stop_robot()
        result.duration = time.monotonic() - t0
        return result

    # Phase 3: Insertion descent — slow, force-aware, integrator still active.
    # Starts at z_offset=0.10 (matching Fine align hold) and ramps to
    # -0.015 over 15 s (~7.7 mm/s target-plug descent, which matches
    # the insertion_linear_vel cap).
    insert_gains = ControllerGains(
        kp_linear=2.0, max_linear_vel=0.01, insertion_linear_vel=0.008,
        kp_angular=2.5, max_angular_vel=0.15,
    )
    if not run_phase("Insertion", z_start=0.10, z_end=-0.015, gains=insert_gains,
                     max_steps=fps * 15, is_insertion=True, use_integrator=True):
        node.stop_robot()
        result.duration = time.monotonic() - t0
        return result

    # Phase 4: Hold — keep pressing at insertion depth and wait for either
    # the /scoring/insertion_event message (gold standard) OR a TF-based
    # check that the plug tip has seated into the port (fallback that works
    # even if the lazy gz_ros_bridge drops the event).
    #
    # Hold duration is deliberately long (~8s) because the cable has
    # noticeable settling dynamics: plug may need a moment to fully seat.
    logger.info("  Phase: Hold (keeping insertion pressure)")
    hold_seconds = 8.0
    tf_success = False
    tf_success_streak = 0  # require a few consecutive matches to avoid flicker
    hold_start = time.monotonic()
    hold_tick = 0
    while time.monotonic() - hold_start < hold_seconds:
        if node.insertion_detected.is_set():
            break

        # Keep pressing: recompute the same "slightly below port face" target
        # each tick so the controller maintains insertion force.
        try:
            tp, tq = compute_target_pose(
                node, port_frame, plug_frame, z_offset=-0.015,
                integrator=lateral_integrator, reset_integrator=False,
            )
            step(ZERO_ACTION, target_pos=tp, target_quat_wxyz=tq)
        except TransformException:
            step(ZERO_ACTION)

        # TF-based success: plug is at port (<=1cm total, <=5mm lateral) and
        # it's there for >=0.5s.
        try:
            total, axial, lateral = node.plug_to_port_distance(port_frame, plug_frame)
            if total < 0.01 and lateral < 0.005:
                tf_success_streak += 1
                if tf_success_streak >= int(fps * 0.5):
                    logger.info(
                        f"  [TF] Plug seated: total={total*1000:.1f}mm "
                        f"axial={axial*1000:+.1f}mm lateral={lateral*1000:.1f}mm"
                    )
                    tf_success = True
                    break
            else:
                tf_success_streak = 0
        except TransformException:
            pass

        # 1Hz diagnostic snapshot during Hold so we can see how the plug
        # evolves during the press (does it slowly seat? does it slip
        # sideways? does the gripper give up tracking?).
        if hold_tick % max(1, fps) == 0:
            try:
                cur_pos_h, _ = node.get_tcp_pose()
                tp_h, _ = compute_target_pose(
                    node, port_frame, plug_frame, z_offset=-0.015,
                )
                d = compute_pose_diagnostics(
                    node, port_frame, plug_frame, cur_pos_h, tp_h
                )
                fs = node.get_force_stats()
                logger.info(
                    f"    [hold {hold_tick // max(1, fps)}s] "
                    f"force={fs['current']:.1f}N (max={fs['max']:.1f}N)"
                )
                logger.info(format_diag_line(d))
            except TransformException:
                pass
        hold_tick += 1

        time.sleep(1.0 / fps)

    # DON'T call stop_robot(): keep the insertion-pressure target in place
    # until the caller explicitly retreats. Otherwise the impedance controller
    # could drift and pull the cable back out.

    # Collect results
    result.success = node.insertion_detected.is_set() or tf_success
    if tf_success and not node.insertion_detected.is_set():
        logger.info("  (insertion confirmed by TF; /scoring/insertion_event not received)")
    fs = node.get_force_stats()
    result.max_force = fs["max"]
    result.time_above_force_threshold = fs["time_above_threshold"]
    result.duration = time.monotonic() - t0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Automated SFP demo collection")
    ap.add_argument("--dataset-root", required=True, help="Full path for the dataset")
    ap.add_argument("--repo-id", default="atang/aic_sfp_auto")
    ap.add_argument("--num-episodes", type=int, default=20, help="Target successful episodes")
    ap.add_argument("--max-attempts", type=int, default=50, help="Max total attempts")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port-frame", default=DEFAULT_PORT_FRAME,
                    help="TF frame for the target port (default: nic_card_mount_0/sfp_port_0)")
    ap.add_argument("--plug-frame", default=DEFAULT_PLUG_FRAME,
                    help="TF frame for the cable plug tip (default: cable_0/sfp_tip)")
    ap.add_argument("--resume", action="store_true", help="Resume existing dataset")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--max-episode-time", type=float, default=60.0, help="Max seconds per attempt")
    ap.add_argument("--discard-high-force", action="store_true",
                    help="Discard episodes where force exceeded scoring threshold for >1s")
    ap.add_argument(
        "--reset-scene", action="store_true",
        help=(
            "[BROKEN/EXPERIMENTAL] Was intended to call "
            "/gz_server/reset_simulation between attempts. On the current "
            "build this reset causes ros2_control inside the same container "
            "to reload and segfault (container exit code -11). Flag is "
            "accepted for backward compatibility but is a no-op and logs a "
            "warning."
        ),
    )
    ap.add_argument(
        "--exit-on-success", action="store_true",
        help=(
            "Stop as soon as one episode is saved. Recommended per-launch "
            "workflow: with a fresh Gazebo launch, run with "
            "--num-episodes 1 --max-attempts 5 --exit-on-success. Once the "
            "first successful insertion is captured, the script exits "
            "cleanly. Kill Gazebo, relaunch, and rerun with --resume to "
            "append the next episode. This avoids follow-up attempts "
            "trying to insert a cable that's already seated."
        ),
    )
    args = ap.parse_args()

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

    logger.info("Waiting for robot state and cameras...")
    node.wait_for_state()
    cameras.wait_for_images(timeout=15.0)

    logger.info("Waiting for TF frames...")
    tf_ok = False
    for _ in range(100):
        try:
            node.lookup_tf("base_link", args.port_frame)
            node.lookup_tf("base_link", args.plug_frame)
            node.lookup_tf("base_link", "gripper/tcp")
            tf_ok = True
            break
        except TransformException:
            time.sleep(0.2)
    if not tf_ok:
        logger.error(f"TF frames not found. Tried:\n"
                     f"  port: {args.port_frame}\n"
                     f"  plug: {args.plug_frame}\n"
                     f"  gripper: gripper/tcp\n"
                     f"Is ground_truth:=true set? Is the scene spawned?")
        rclpy.shutdown()
        return

    logger.info(f"TF ready. Port: {args.port_frame} | Plug: {args.plug_frame}")

    dataset = create_or_resume_dataset(
        args.repo_id, args.dataset_root, args.fps, args.resume, args.vcodec
    )

    successes = 0
    attempts = 0
    discarded_force = 0
    discarded_collision = 0
    discarded_stuck = 0
    discarded_failed = 0
    discarded_lateral = 0

    logger.info(f"=== Starting collection: {args.num_episodes} episodes, max {args.max_attempts} attempts ===")

    if args.reset_scene:
        logger.warning(
            "--reset-scene is currently a no-op. On this build, calling "
            "/gz_server/reset_simulation crashes the ros_gz_container "
            "(ros2_control tries to reload inside the same process and "
            "segfaults). Use the manual-relaunch + --resume workflow "
            "instead. See docs/auto_collection_guide.md."
        )

    try:
        while successes < args.num_episodes and attempts < args.max_attempts:
            attempts += 1
            logger.info(f"\n--- Attempt {attempts} | Saved: {successes}/{args.num_episodes} ---")

            ep = run_episode(
                node, cameras,
                port_frame=args.port_frame,
                plug_frame=args.plug_frame,
                fps=args.fps,
                max_time=args.max_episode_time,
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
                elif "alignment not converged" in reason:
                    discarded_lateral += 1

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
                                        dataset.meta.features, TASK_STR)
                    dataset.add_frame(frame)
                dataset.save_episode()
                successes += 1

                # Per-launch workflow: the cable is now seated in the port.
                # Any further attempts within this Gazebo session would
                # have to rip it back out, which produces poor data and
                # can wedge the arm. Exit cleanly so the user can relaunch
                # Gazebo and come back with --resume.
                if args.exit_on_success:
                    logger.info(
                        "  --exit-on-success: stopping after first saved "
                        "episode. Relaunch Gazebo and rerun with --resume "
                        "to collect the next one."
                    )
                    break
            else:
                # Only clear if the buffer was actually populated. Early
                # aborts (e.g. "robot stuck" fires during the first Approach
                # tick, before any frame is recorded) leave
                # dataset.episode_buffer as None, and
                # LeRobotDataset.clear_episode_buffer() crashes on
                # episode_buffer["episode_index"] if called then.
                if getattr(dataset, "episode_buffer", None) is not None:
                    dataset.clear_episode_buffer()

            # Between failed attempts within the same launch: lift straight
            # up by 15 cm so that
            #   (a) a stuck/jammed arm is pulled off any contact, and
            #   (b) the next Approach interpolation starts from clear space.
            node.retreat_vertical(lift_m=0.15, duration_s=2.5, fps=args.fps)
            node.stop_robot()
            time.sleep(1.0)

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
        logger.info(f"    Insertion failed:    {discarded_failed}")
        logger.info(f"    Alignment not conv.: {discarded_lateral}")
        logger.info(f"    High force:          {discarded_force}")
        logger.info(f"    Collision:           {discarded_collision}")
        logger.info(f"    Robot stuck:         {discarded_stuck}")
        logger.info(f"{'='*60}")

        dataset.finalize()
        node.destroy_node()
        rclpy.shutdown()

    logger.info(f"Dataset saved at: {args.dataset_root}")


if __name__ == "__main__":
    main()
