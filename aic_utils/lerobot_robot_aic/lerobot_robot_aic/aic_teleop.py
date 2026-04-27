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

import math
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, cast

import pyspacemouse
import rclpy
from geometry_msgs.msg import Twist
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.teleoperators.keyboard import (
    KeyboardEndEffectorTeleop,
    KeyboardEndEffectorTeleopConfig,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot_teleoperator_devices import KeyboardJointTeleop, KeyboardJointTeleopConfig
from rclpy.executors import SingleThreadedExecutor

from .aic_robot import arm_joint_names
from .types import JointMotionUpdateActionDict, MotionUpdateActionDict

# EMA smoothing on the published action.  Each get_action() tick the
# emitted value moves a fraction ``ACTION_SMOOTHING_ALPHA`` of the way
# from its current value toward the raw key-driven target (full
# scaling on press, 0 on release).  At lerobot-record's ~30 Hz poll:
#   alpha=0.5  -> 50% catch-up per tick, ~75% within 67 ms (2 ticks)
#   alpha=0.3  -> noticeably softer
#   alpha=1.0  -> no smoothing (legacy step-function behaviour)
# A small alpha is what cable insertion needs: it converts the
# {-scale, 0, +scale} keyboard step-functions into continuous ramps so
# recorded actions are smooth out of the box (no SavGol post-pass) and
# the trained policy emits smooth velocities at inference.  Without
# this every action column has only 2-3 unique values per axis with
# ~73% all-zero frames -- exactly the failure mode of the existing
# 201-episode SFP dataset.
ACTION_SMOOTHING_ALPHA = 0.5

# Snap-to-zero deadband: once the smoothed value drops below this
# magnitude, force it to exact 0.  Stops the EMA from carrying an
# asymptotic micro-velocity for many ticks after a key release, which
# would otherwise pollute the dataset with permanent millimetre/sec
# drift.  5e-4 is 0.5% of the 0.1 m/s EE scaling and 1% of the 0.05
# rad/s joint scaling -- well below any meaningful robot motion.
ACTION_DEADBAND = 5e-4


def _ema_update(current: float, target: float) -> float:
    """One-tick EMA step with snap-to-zero deadband."""
    smoothed = ACTION_SMOOTHING_ALPHA * target + (1.0 - ACTION_SMOOTHING_ALPHA) * current
    if abs(smoothed) < ACTION_DEADBAND:
        smoothed = 0.0
    return smoothed


@TeleoperatorConfig.register_subclass("aic_keyboard_joint")
@dataclass
class AICKeyboardJointTeleopConfig(KeyboardJointTeleopConfig):
    arm_action_keys: list[str] = field(
        default_factory=lambda: [f"{x}" for x in arm_joint_names]
    )
    high_command_scaling: float = 0.05
    low_command_scaling: float = 0.02


class AICKeyboardJointTeleop(KeyboardJointTeleop):
    def __init__(self, config: AICKeyboardJointTeleopConfig):
        super().__init__(config)

        self.config = config
        self._low_scaling = config.low_command_scaling
        self._high_scaling = config.high_command_scaling
        self._current_scaling = self._high_scaling

        self.curr_joint_actions: JointMotionUpdateActionDict = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        }

        # Persistent EMA state for action smoothing -- see EE class
        # comment.  Joint mode emits the same step-function pattern
        # (val on press, 0 on release) without smoothing, so it
        # benefits from the same fix.
        self._smoothed_joint_actions: JointMotionUpdateActionDict = {
            k: 0.0 for k in self.curr_joint_actions
        }

    @property
    def action_features(self) -> dict:
        return {"names": JointMotionUpdateActionDict.__annotations__}

    def _get_action_value(self, is_pressed: bool) -> float:
        return self._current_scaling if is_pressed else 0.0

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError()

        self._drain_pressed_keys()

        for key, is_pressed in self.current_pressed.items():

            if key == "u" and is_pressed:
                is_low_scaling = self._current_scaling == self._low_scaling
                self._current_scaling = (
                    self._high_scaling if is_low_scaling else self._low_scaling
                )
                print(f"Command scaling toggled to: {self._current_scaling}")
                continue

            val = self._get_action_value(is_pressed)

            if key == "q":
                self.curr_joint_actions["shoulder_pan_joint"] = val
            elif key == "a":
                self.curr_joint_actions["shoulder_pan_joint"] = -val
            elif key == "w":
                self.curr_joint_actions["shoulder_lift_joint"] = val
            elif key == "s":
                self.curr_joint_actions["shoulder_lift_joint"] = -val
            elif key == "e":
                self.curr_joint_actions["elbow_joint"] = val
            elif key == "d":
                self.curr_joint_actions["elbow_joint"] = -val
            elif key == "r":
                self.curr_joint_actions["wrist_1_joint"] = val
            elif key == "f":
                self.curr_joint_actions["wrist_1_joint"] = -val
            elif key == "t":
                self.curr_joint_actions["wrist_2_joint"] = val
            elif key == "g":
                self.curr_joint_actions["wrist_2_joint"] = -val
            elif key == "y":
                self.curr_joint_actions["wrist_3_joint"] = val
            elif key == "h":
                self.curr_joint_actions["wrist_3_joint"] = -val
            elif is_pressed:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        for k in self._smoothed_joint_actions:
            self._smoothed_joint_actions[k] = _ema_update(
                self._smoothed_joint_actions[k], self.curr_joint_actions[k]
            )

        return cast(dict, dict(self._smoothed_joint_actions))


@TeleoperatorConfig.register_subclass("aic_keyboard_ee")
@dataclass(kw_only=True)
class AICKeyboardEETeleopConfig(KeyboardEndEffectorTeleopConfig):
    high_command_scaling: float = 0.1
    low_command_scaling: float = 0.02


class AICKeyboardEETeleop(KeyboardEndEffectorTeleop):
    def __init__(self, config: AICKeyboardEETeleopConfig):
        super().__init__(config)
        self.config = config

        self._high_scaling = config.high_command_scaling
        self._low_scaling = config.low_command_scaling
        self._current_scaling = self._high_scaling

        self._current_actions: MotionUpdateActionDict = {
            "linear.x": 0.0,
            "linear.y": 0.0,
            "linear.z": 0.0,
            "angular.x": 0.0,
            "angular.y": 0.0,
            "angular.z": 0.0,
        }

        # Persistent EMA state.  ``_current_actions`` holds the raw
        # key-driven target (snaps to ``+/- scaling`` on press, 0 on
        # release); ``_smoothed_actions`` is what we publish to the
        # robot and what lerobot-record writes to the dataset.  See
        # ACTION_SMOOTHING_ALPHA at the top of the file.
        self._smoothed_actions: MotionUpdateActionDict = {
            k: 0.0 for k in self._current_actions
        }

    @property
    def action_features(self) -> dict:
        return MotionUpdateActionDict.__annotations__

    def _get_action_value(self, is_pressed: bool) -> float:
        return self._current_scaling if is_pressed else 0.0

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError()

        self._drain_pressed_keys()

        for key, is_pressed in self.current_pressed.items():

            if key == "t" and is_pressed:
                is_low_speed = self._current_scaling == self._low_scaling
                self._current_scaling = (
                    self._high_scaling if is_low_speed else self._low_scaling
                )
                print(f"Command scaling toggled to: {self._current_scaling}")
                continue

            val = self._get_action_value(is_pressed)

            if key == "w":
                self._current_actions["linear.y"] = -val
            elif key == "s":
                self._current_actions["linear.y"] = val
            elif key == "a":
                self._current_actions["linear.x"] = -val
            elif key == "d":
                self._current_actions["linear.x"] = val
            elif key == "r":
                self._current_actions["linear.z"] = -val
            elif key == "f":
                self._current_actions["linear.z"] = val
            elif key == "W":
                self._current_actions["angular.x"] = val
            elif key == "S":
                self._current_actions["angular.x"] = -val
            elif key == "A":
                self._current_actions["angular.y"] = -val
            elif key == "D":
                self._current_actions["angular.y"] = val
            elif key == "q":
                self._current_actions["angular.z"] = -val
            elif key == "e":
                self._current_actions["angular.z"] = val
            elif is_pressed:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        for k in self._smoothed_actions:
            self._smoothed_actions[k] = _ema_update(
                self._smoothed_actions[k], self._current_actions[k]
            )

        return cast(dict, dict(self._smoothed_actions))


@TeleoperatorConfig.register_subclass("aic_spacemouse")
@dataclass(kw_only=True)
class AICSpaceMouseTeleopConfig(TeleoperatorConfig):
    operator_position_front: bool = True
    device: str | None = None  # only needed for multiple space mice
    command_scaling: float = 0.1


class AICSpaceMouseTeleop(Teleoperator):
    def __init__(self, config: AICSpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._device: pyspacemouse.SpaceMouseDevice | None = None

        self._current_actions: MotionUpdateActionDict = {
            "linear.x": 0.0,
            "linear.y": 0.0,
            "linear.z": 0.0,
            "angular.x": 0.0,
            "angular.y": 0.0,
            "angular.z": 0.0,
        }

    @property
    def name(self) -> str:
        return "aic_spacemouse"

    @property
    def action_features(self) -> dict:
        return MotionUpdateActionDict.__annotations__

    @property
    def feedback_features(self) -> dict:
        # TODO
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("spacemouse_teleop")
        if calibrate:
            self._node.get_logger().warn(
                "Calibration not supported, ensure the robot is calibrated before running teleop."
            )

        self._device_open_success = pyspacemouse.open(
            dof_callback=None,
            # button_callback_arr=[
            #     pyspacemouse.ButtonCallback([0], self._button_callback),  # Button 1
            #     pyspacemouse.ButtonCallback([1], self._button_callback),  # Button 2
            # ],
            device=self.config.device,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = Thread(target=self._executor.spin)
        self._executor_thread.start()
        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        # Calibration not supported
        return True

    def calibrate(self) -> None:
        # Calibration not supported
        pass

    def configure(self) -> None:
        pass

    def apply_deadband(self, value, threshold=0.02):
        return value if abs(value) > threshold else 0.0

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected or not self._device:
            raise DeviceNotConnectedError()

        state = self._device.read()

        clean_x = self.apply_deadband(float(state.x))
        clean_y = self.apply_deadband(float(state.y))
        clean_z = self.apply_deadband(float(state.z))
        clean_roll = self.apply_deadband(float(state.roll))
        clean_pitch = self.apply_deadband(float(state.pitch))
        clean_yaw = self.apply_deadband(float(state.yaw))

        twist_msg = Twist()
        twist_msg.linear.x = clean_x**1 * self.config.command_scaling
        twist_msg.linear.y = -(clean_y**1) * self.config.command_scaling
        twist_msg.linear.z = -(clean_z**1) * self.config.command_scaling
        twist_msg.angular.x = -(clean_pitch**1) * self.config.command_scaling
        twist_msg.angular.y = clean_roll**1 * self.config.command_scaling  #
        twist_msg.angular.z = clean_yaw**1 * self.config.command_scaling

        if not self.config.operator_position_front:
            twist_msg.linear.x *= -1
            twist_msg.linear.y *= -1
            twist_msg.angular.x *= -1
            twist_msg.angular.y *= -1

        self._current_actions = {
            "linear.x": twist_msg.linear.x,
            "linear.y": twist_msg.linear.y,
            "linear.z": twist_msg.linear.z,
            "angular.x": twist_msg.angular.x,
            "angular.y": twist_msg.angular.y,
            "angular.z": twist_msg.angular.z,
        }

        return cast(dict, self._current_actions)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._device:
            self._device.close()
        self._is_connected = False
        pass


# ---------------------------------------------------------------------------
# Gamepad (PS5 DualSense / Xbox / Logitech F710) end-effector teleop.
#
# Why this class exists at all:
# --------------------------------
# The keyboard-based AICKeyboardEETeleop (above) emits a step-function on
# every axis -- value snaps to ``±scaling`` while a key is held and back
# to 0 on release.  After 30 Hz EMA smoothing (alpha=0.5) and a 5e-4
# deadband, that produces the bimodal action distribution we measured in
# ``teleop-dataset-ft-v1``: ~75% frames at zero, ~12% pegged at the
# velocity rails, ~5-12% in the smooth-transition mid-band per axis.
# That distribution is poison for L1-regression policies like ACT --
# the policy minimizes loss by predicting near-zero, so it never
# commits to the high-magnitude motions an actual insertion needs.
# (See scripts/probe_action_discreteness.py for the diagnostic.)
#
# A dual-stick gamepad (DualSense / Xbox / F710) has 6 spring-centered
# proportional analog axes -- 4 stick axes + 2 trigger axes -- which is
# exactly the right shape for our 6-D Cartesian twist action space.
# Output is naturally continuous, so we *must not* run the keyboard EMA
# smoothing on top of it (that would low-pass an already-smooth signal
# and introduce phase lag).  We just deadband the sticks and apply an
# expo curve for fine control near center.
#
# Mapping (DualSense / Xbox SDL2 standard layout):
#
#   Translation:
#     left stick X   -> linear.x  (push right to go +X)
#     left stick Y   -> linear.y  (push forward to go +Y; Y axis is
#                                  inverted in SDL convention so we
#                                  negate it to match the teleop guide)
#     L2 / R2        -> linear.z  (R2 = up/+Z, L2 = down/-Z; held for
#                                  proportional descent during insertion)
#
#   Rotation (right stick + bumpers):
#     right stick Y  -> angular.x  (push forward = pitch forward = +X)
#     right stick X  -> angular.y  (push right = roll right = +Y)
#     L1 / R1        -> angular.z  (R1 = +Z = CCW yaw, L1 = -Z = CW yaw)
#
#   Recording flow (synthesized keyboard events via pynput, so
#   lerobot-record's existing ``init_keyboard_listener`` picks them up
#   without any patches to lerobot itself):
#     Triangle (3)   -> Right Arrow  (save current episode)
#     Square   (2)   -> Left Arrow   (discard / rerecord)
#     Cross    (0)   -> ESC          (stop recording entirely)
#     Options  (6)   -> toggle slow / fast scaling
#
# The Cross-as-ESC choice is deliberate: it matches the X52 / DualSense
# convention of "X = cancel" and avoids a one-press footgun by being
# distinct from the save/discard buttons.
#
# Why no EMA / no deadband-snap on this path:
#   - Stick analog signal already has ~8-12 bit ADC resolution per axis
#     and is mechanically spring-centered -- the rest value sits well
#     under our ``stick_deadzone`` (typical 0.05) so any drift is
#     filtered without introducing the keyboard's bang-bang artifact.
#   - The expo curve (default 1.5) gives more fine control near center
#     without sacrificing top speed at full deflection -- exactly what
#     fine alignment near the SC port needs.
# ---------------------------------------------------------------------------


# SDL2 axis indices for a DualSense / Xbox / F710 controller as exposed
# by pygame.joystick.Joystick.get_axis().  Verified live on this rig
# with a DualSense via the hid-playstation kernel driver:
#   axes 0,3 read ~0 at rest; axes 2,5 read -1 at rest (= triggers
#   released).  See ``scripts/probe_gamepad_axes.py`` if you ever need
#   to verify on a different controller -- if those rest values change
#   you've got a non-standard mapping and should adjust the indices.
_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1
_AXIS_L2 = 2
_AXIS_RIGHT_X = 3
_AXIS_RIGHT_Y = 4
_AXIS_R2 = 5

# SDL2 button indices for the DualSense (13 buttons total).  There are
# two layouts you'll see in the wild and which one you get depends on
# how the kernel + SDL ended up mapping your controller, NOT on the
# controller hardware itself:
#
#   "hid-playstation" layout (modern, Linux 5.12+ with the dedicated
#   driver):
#     0=Cross 1=Circle 2=Square 3=Triangle 4=Share 5=PS 6=Options
#     7=L3 8=R3 9=L1 10=R1 11=Touchpad 12=Mute
#
#   "DS4-style" layout (SDL2 generic HID parser fallback, also used by
#   F710/Xbox controllers in XInput mode):
#     0=Square 1=Cross 2=Circle 3=Triangle 4=L1 5=R1 6=L2 7=R2
#     8=Share 9=Options 10=PS 11=L3 12=R3
#
# We default to the DS4-style layout because that's what we observed
# live on the rig (SDL 2.28.4 + DualSense via 0bda hub on Ubuntu) and
# it also works for Logitech F710 / Xbox in XInput mode without any
# remapping -- so a single config covers all three controllers we
# realistically use.
#
# If you ever plug in a different controller and the bindings feel
# wrong, run ``scripts/probe_gamepad_buttons.py`` to live-verify the
# indices, then override the *_button fields on the config (don't
# edit these constants -- per-config overrides are how a future
# different controller stays supported).
_BTN_SQUARE = 0       # X on Xbox -- mapped to Left Arrow (discard)
_BTN_CROSS = 1        # A on Xbox, "X" on DualSense face -- mapped to ESC (stop)
_BTN_CIRCLE = 2       # B on Xbox -- unused
_BTN_TRIANGLE = 3     # Y on Xbox -- mapped to Right Arrow (save)
_BTN_L1 = 4           # LB on Xbox -- yaw negative
_BTN_R1 = 5           # RB on Xbox -- yaw positive
# Indices 6/7 are L2/R2 reported as digital buttons in DS4 layout.
# We deliberately leave these UNBOUND because we already use the
# triggers as analog axes for linear.z -- having L2 also trigger a
# discrete event would mean every descent press toggles a button.
_BTN_L2_DIGITAL = 6
_BTN_R2_DIGITAL = 7
_BTN_SHARE = 8
_BTN_OPTIONS = 9      # Start on Xbox -- mapped to slow/fast toggle
_BTN_PS = 10
_BTN_L3 = 11
_BTN_R3 = 12


def _expo(value: float, deadzone: float, expo: float) -> float:
    """Apply deadzone + expo curve to a normalized [-1, 1] stick axis.

    Below ``deadzone`` we return exact 0.  Above it we re-normalize so
    the live range is [0, 1] regardless of deadzone size, then raise to
    ``expo`` (>1 = more fine control near center, less sensitive at
    edges).  Sign is preserved.  expo=1.0 disables the curve.
    """
    if abs(value) < deadzone:
        return 0.0
    s = (abs(value) - deadzone) / (1.0 - deadzone)
    return math.copysign(s ** expo, value)


def _trigger_unipolar(raw: float) -> float:
    """Convert SDL trigger axis ([-1 released .. +1 fully pressed]) to
    a unipolar [0, 1] press strength.  Returns exact 0 when the trigger
    is at rest -- some controllers report -0.999 instead of -1.0 so we
    clamp.
    """
    s = (raw + 1.0) * 0.5
    return 0.0 if s < 1e-3 else min(s, 1.0)


@TeleoperatorConfig.register_subclass("aic_gamepad_ee")
@dataclass(kw_only=True)
class AICGamepadEETeleopConfig(TeleoperatorConfig):
    # Per-axis scaling at full stick / trigger deflection.  Match the
    # keyboard EE class so a recorded gamepad dataset has the same
    # action range as the keyboard one (and so the trained policy emits
    # the same ranges at deploy time, no retuning of action_rescale).
    high_command_scaling: float = 0.1
    low_command_scaling: float = 0.02

    # Stick / trigger response shaping.  The DualSense reports a small
    # static bias on left-stick at rest (~0.02) so a deadzone of 0.05
    # gives plenty of margin while not eating into useful range.
    stick_deadzone: float = 0.05
    stick_expo: float = 1.5

    trigger_deadzone: float = 0.02
    trigger_expo: float = 1.3

    # Index of the joystick to open (0 = first one pygame finds).  Only
    # change if you have multiple controllers plugged in.
    joystick_index: int = 0

    # Synthesize keyboard events for lerobot-record's flow control.
    # Set to False if you want to drive save/discard/stop manually
    # (e.g. for headless / SSH operation, or to avoid any global
    # keyboard simulation).  When False the class still polls the
    # buttons but does not press any keys; you'll need to use the real
    # arrow keys and ESC.
    inject_keyboard_events: bool = True

    # Button mappings.  Overridable in case you want a different
    # convention -- set these to ``-1`` to disable any individual binding.
    save_button: int = _BTN_TRIANGLE        # Right Arrow
    discard_button: int = _BTN_SQUARE       # Left Arrow
    stop_button: int = _BTN_CROSS           # ESC
    speed_toggle_button: int = _BTN_OPTIONS

    # Yaw is digital (bumpers).  At full press it commands ±1 *
    # current_scaling on angular.z.  You can disable it by setting both
    # to -1 if you'd rather control yaw from elsewhere.
    yaw_pos_button: int = _BTN_R1
    yaw_neg_button: int = _BTN_L1


class AICGamepadEETeleop(Teleoperator):
    """6-DOF Cartesian-twist teleop driven by a DualSense / Xbox / F710 gamepad.

    Drop-in replacement for ``AICKeyboardEETeleop``: same action shape
    (``MotionUpdateActionDict``), same config knobs (``*_command_scaling``),
    consumed the same way by ``AICRobotAICController``.  The only
    difference is that the action is sampled from a continuous analog
    source instead of a {-scale, 0, +scale} keyboard step function, so
    no EMA smoothing is needed and the resulting dataset is unimodal.
    """

    config_class = AICGamepadEETeleopConfig
    name = "aic_gamepad_ee"

    def __init__(self, config: AICGamepadEETeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        self._high_scaling = config.high_command_scaling
        self._low_scaling = config.low_command_scaling
        self._current_scaling = self._high_scaling

        self._joystick = None
        self._kbd_controller = None  # pynput keyboard for synthesized events
        self._key_save = None  # Key.right, populated in connect()
        self._key_discard = None  # Key.left
        self._key_stop = None  # Key.esc

        # Edge-detection state for buttons we treat as "press once" -- we
        # only act on the rising edge so a held button doesn't spam the
        # same event every tick.
        self._prev_buttons: dict[int, bool] = {}

        # Most-recent action emitted, used both as the public state and
        # as the value we publish for record / control.
        self._current_actions: MotionUpdateActionDict = {
            "linear.x": 0.0,
            "linear.y": 0.0,
            "linear.z": 0.0,
            "angular.x": 0.0,
            "angular.y": 0.0,
            "angular.z": 0.0,
        }

    @property
    def action_features(self) -> dict:
        return MotionUpdateActionDict.__annotations__

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError()

        # SDL2 requires the video subsystem to be initialized before
        # ``pygame.event.pump()`` can drain the OS event queue (which is
        # what keeps joystick state fresh on Linux).  When running with
        # a normal display this just works; in headless / SSH sessions
        # we transparently fall back to SDL's "dummy" video driver so
        # the operator doesn't have to know about either case.  We
        # never override an existing SDL_VIDEODRIVER setting -- if the
        # operator chose wayland / x11 explicitly we respect it.
        import os

        import pygame

        try:
            pygame.display.init()
        except pygame.error:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n == 0:
            raise DeviceNotConnectedError(
                "No gamepad detected.  Plug in a DualSense / Xbox / F710 "
                "controller and rerun.  ``lsusb`` should show e.g. "
                "``Sony Corp. DualSense wireless controller (PS5)``."
            )
        if self.config.joystick_index >= n:
            raise DeviceNotConnectedError(
                f"joystick_index={self.config.joystick_index} but only "
                f"{n} joystick(s) detected."
            )

        self._joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self._joystick.init()

        if self.config.inject_keyboard_events:
            # pynput pulls in an Xlib connection on Linux, which is why
            # we lazy-import it instead of doing it at module load -- a
            # headless smoke test (or running with
            # ``inject_keyboard_events=False``) shouldn't require X.
            from pynput.keyboard import Controller, Key

            self._kbd_controller = Controller()
            self._key_save = Key.right
            self._key_discard = Key.left
            self._key_stop = Key.esc

        # Wait one update to settle (some drivers report all axes at 0
        # for the first poll before the device hands over real values).
        pygame.event.pump()
        time.sleep(0.05)
        pygame.event.pump()

        self._is_connected = True
        print(
            f"[aic_gamepad_ee] connected: {self._joystick.get_name()!r}  "
            f"axes={self._joystick.get_numaxes()}  "
            f"buttons={self._joystick.get_numbuttons()}  "
            f"scaling={self._current_scaling}  "
            f"keyboard_events={'ON' if self._kbd_controller else 'OFF'}"
        )
        print(
            "  Mapping: LStick = linear.xy   RStick = pitch/roll   "
            "L2/R2 = down/up   L1/R1 = yaw -/+\n"
            "  Triangle = save (-> Right Arrow)   Square = discard (-> Left Arrow)\n"
            "  Cross = stop recording (-> ESC)   Options = toggle slow/fast"
        )

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        # Drop references to pynput resources before our caller tears
        # down the X connection so the Controller's __del__ doesn't
        # fire after the display is already gone (which raises noisy
        # AttributeErrors during interpreter shutdown).
        self._kbd_controller = None
        self._key_save = None
        self._key_discard = None
        self._key_stop = None
        self._is_connected = False

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tap_key(self, key) -> None:
        """Synthesize a single key tap so lerobot-record's pynput-based
        ``init_keyboard_listener`` picks it up.  No-op if the user opted
        out of keyboard injection (``inject_keyboard_events=False``) or
        if the cached key reference is missing."""
        if self._kbd_controller is None or key is None:
            return
        try:
            self._kbd_controller.press(key)
            self._kbd_controller.release(key)
        except Exception as e:
            print(f"[aic_gamepad_ee] failed to inject key {key}: {e}")

    def _rising_edge(self, btn_idx: int) -> bool:
        """Return True only on the transition not-pressed -> pressed.

        Held buttons must NOT keep firing -- otherwise a single
        Triangle press would save dozens of episodes.
        """
        if btn_idx is None or btn_idx < 0:
            return False
        cur = bool(self._joystick.get_button(btn_idx))
        prev = self._prev_buttons.get(btn_idx, False)
        self._prev_buttons[btn_idx] = cur
        return cur and not prev

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError()

        import pygame

        pygame.event.pump()
        j = self._joystick

        # Speed toggle has to come before we read scaling so a press
        # this tick affects this tick's emitted action -- otherwise the
        # operator presses Options expecting slow mode and the next
        # action they take is still in fast mode, which is exactly the
        # kind of footgun that breaks an in-progress fine-alignment.
        if self._rising_edge(self.config.speed_toggle_button):
            is_low = self._current_scaling == self._low_scaling
            self._current_scaling = (
                self._high_scaling if is_low else self._low_scaling
            )
            print(f"[aic_gamepad_ee] scaling toggled to: {self._current_scaling}")

        scale = self._current_scaling

        # Sticks: deadzone + expo, then scale.  Note the sign on
        # left-stick Y and right-stick Y -- SDL reports "stick pushed
        # forward (away from operator)" as a *negative* Y value, but our
        # convention from the keyboard EE class (and from
        # ``teleop_guide.md``) is "forward = +linear.y" / "pitch
        # forward = +angular.x", so we flip them.
        lx = _expo(j.get_axis(_AXIS_LEFT_X), self.config.stick_deadzone, self.config.stick_expo)
        ly = _expo(j.get_axis(_AXIS_LEFT_Y), self.config.stick_deadzone, self.config.stick_expo)
        rx = _expo(j.get_axis(_AXIS_RIGHT_X), self.config.stick_deadzone, self.config.stick_expo)
        ry = _expo(j.get_axis(_AXIS_RIGHT_Y), self.config.stick_deadzone, self.config.stick_expo)

        # Triggers: SDL reports -1 released -> +1 fully pressed; convert
        # to unipolar [0, 1] then apply expo for fine descent control.
        lt = _expo(_trigger_unipolar(j.get_axis(_AXIS_L2)), self.config.trigger_deadzone, self.config.trigger_expo)
        rt = _expo(_trigger_unipolar(j.get_axis(_AXIS_R2)), self.config.trigger_deadzone, self.config.trigger_expo)

        # Yaw is digital -- L1/R1 bumpers commit a full ±1 ramp scaled
        # by current_scaling.  If both are pressed they cancel.
        def _btn(idx: int) -> int:
            return 1 if (idx is not None and idx >= 0 and j.get_button(idx)) else 0

        yaw = _btn(self.config.yaw_pos_button) - _btn(self.config.yaw_neg_button)

        self._current_actions = {
            "linear.x": lx * scale,
            "linear.y": -ly * scale,
            "linear.z": (rt - lt) * scale,
            "angular.x": -ry * scale,
            "angular.y": rx * scale,
            "angular.z": yaw * scale,
        }

        # Recording flow: synthesize the key events that lerobot-record
        # already listens for.  We only fire on the rising edge so the
        # operator can hold a button without re-triggering.
        if self._rising_edge(self.config.save_button):
            print("[aic_gamepad_ee] SAVE pressed -> Right Arrow")
            self._tap_key(self._key_save)
        if self._rising_edge(self.config.discard_button):
            print("[aic_gamepad_ee] DISCARD pressed -> Left Arrow")
            self._tap_key(self._key_discard)
        if self._rising_edge(self.config.stop_button):
            print("[aic_gamepad_ee] STOP pressed -> ESC")
            self._tap_key(self._key_stop)

        return cast(dict, dict(self._current_actions))
