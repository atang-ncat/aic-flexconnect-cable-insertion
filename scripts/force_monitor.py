#!/usr/bin/env python3
"""
Real-time force monitor for teleoperation.

Subscribes to /fts_broadcaster/wrench and /aic_controller/controller_state,
computes the tared force magnitude (same as the Tier 2 scoring logic), and
prints a live dashboard showing whether you're approaching the penalty
threshold.

Scoring rules (from ScoringTier2.cc):
  - Tared force magnitude = sqrt(fx² + fy² + fz²) after subtracting the
    fts_tare_offset from the raw wrench.
  - If the magnitude exceeds 20 N for more than 1 second cumulative,
    a -12 point penalty is applied.
  - Torque is ignored for scoring.

Usage (in a separate terminal while teleoperating):
    source /ws_aic/install/setup.bash
    python3 /scratch2/atang/ws_aic/scripts/force_monitor.py
"""

import math
import sys
import time

import rclpy
from aic_control_interfaces.msg import ControllerState
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

FORCE_THRESHOLD = 20.0  # Newtons
DURATION_THRESHOLD = 1.0  # seconds cumulative
PENALTY = -12.0

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"


class ForceMonitor(Node):
    def __init__(self):
        super().__init__("force_monitor")

        self._tare_force_x = None
        self._tare_force_y = None
        self._tare_force_z = None

        self._last_wrench_time = None
        self._time_above_threshold = 0.0
        self._max_force = 0.0
        self._current_force = 0.0
        self._raw_force = (0.0, 0.0, 0.0)
        self._tared_force = (0.0, 0.0, 0.0)
        self._wrench_count = 0
        self._state_count = 0
        self._display_lines = 0

        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._controller_state_cb,
            10,
        )
        self.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._wrench_cb,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self._display_cb)

    def _controller_state_cb(self, msg: ControllerState):
        tare = msg.fts_tare_offset.wrench.force
        self._tare_force_x = tare.x
        self._tare_force_y = tare.y
        self._tare_force_z = tare.z
        self._state_count += 1

    def _wrench_cb(self, msg: WrenchStamped):
        self._wrench_count += 1

        raw = msg.wrench.force
        self._raw_force = (raw.x, raw.y, raw.z)

        if self._tare_force_x is None:
            return

        fx = raw.x - self._tare_force_x
        fy = raw.y - self._tare_force_y
        fz = raw.z - self._tare_force_z
        self._tared_force = (fx, fy, fz)
        mag = math.sqrt(fx * fx + fy * fy + fz * fz)
        self._current_force = mag

        if mag > self._max_force:
            self._max_force = mag

        now = time.monotonic()
        if self._last_wrench_time is not None and mag > FORCE_THRESHOLD:
            dt = now - self._last_wrench_time
            self._time_above_threshold += dt
        self._last_wrench_time = now

    def _force_bar(self, force: float, width: int = 40) -> str:
        ratio = min(force / 30.0, 1.0)
        filled = int(ratio * width)
        empty = width - filled

        if force > FORCE_THRESHOLD:
            color = RED
        elif force > FORCE_THRESHOLD * 0.7:
            color = YELLOW
        else:
            color = GREEN

        bar = color + "█" * filled + DIM + "░" * empty + RESET
        return bar

    def _display_cb(self):
        if self._display_lines > 0:
            sys.stdout.write(MOVE_UP * self._display_lines)

        lines = []

        lines.append(f"{CLEAR_LINE}{BOLD}── Force Monitor (Tier 2 Scoring) ──{RESET}")
        lines.append(f"{CLEAR_LINE}")

        if self._tare_force_x is None:
            lines.append(f"{CLEAR_LINE}{YELLOW}Waiting for controller_state (tare offset)...{RESET}")
            lines.append(f"{CLEAR_LINE}  wrench msgs: {self._wrench_count}   state msgs: {self._state_count}")
        else:
            force = self._current_force
            bar = self._force_bar(force)

            if force > FORCE_THRESHOLD:
                force_str = f"{RED}{BOLD}{force:6.1f} N{RESET}"
                status = f"{RED}{BOLD}⚠ ABOVE THRESHOLD{RESET}"
            elif force > FORCE_THRESHOLD * 0.7:
                force_str = f"{YELLOW}{force:6.1f} N{RESET}"
                status = f"{YELLOW}↑ approaching limit{RESET}"
            else:
                force_str = f"{GREEN}{force:6.1f} N{RESET}"
                status = f"{GREEN}✓ OK{RESET}"

            lines.append(f"{CLEAR_LINE}  Tared force:  {force_str}  / {FORCE_THRESHOLD:.0f} N   {status}")
            lines.append(f"{CLEAR_LINE}  {bar}  ")
            lines.append(f"{CLEAR_LINE}")

            fx, fy, fz = self._tared_force
            lines.append(f"{CLEAR_LINE}  Components:   x={fx:+7.2f}  y={fy:+7.2f}  z={fz:+7.2f} N")

            t = self._time_above_threshold
            if t > DURATION_THRESHOLD:
                time_color = RED + BOLD
                penalty_str = f"  → {RED}{BOLD}{PENALTY:+.0f} pts PENALTY{RESET}"
            elif t > 0:
                time_color = YELLOW
                penalty_str = ""
            else:
                time_color = GREEN
                penalty_str = ""

            lines.append(
                f"{CLEAR_LINE}  Time > {FORCE_THRESHOLD:.0f}N:  "
                f"{time_color}{t:5.2f}{RESET} / {DURATION_THRESHOLD:.1f} s{penalty_str}"
            )
            lines.append(f"{CLEAR_LINE}  Max force:    {self._max_force:6.1f} N")
            lines.append(f"{CLEAR_LINE}")
            lines.append(f"{CLEAR_LINE}{DIM}  Press Ctrl+C to stop{RESET}")

        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        self._display_lines = len(lines)


def main():
    print(f"{BOLD}Starting force monitor...{RESET}")
    print(f"Subscribes to /fts_broadcaster/wrench & /aic_controller/controller_state")
    print(f"Penalty threshold: {FORCE_THRESHOLD} N for > {DURATION_THRESHOLD} s cumulative = {PENALTY} pts")
    print()

    rclpy.init()
    node = ForceMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        t = node._time_above_threshold
        max_f = node._max_force
        print()
        print(f"{BOLD}── Session Summary ──{RESET}")
        print(f"  Max force:        {max_f:6.1f} N")
        print(f"  Time > {FORCE_THRESHOLD:.0f}N:      {t:5.2f} s")
        if t > DURATION_THRESHOLD:
            print(f"  {RED}{BOLD}Penalty would apply: {PENALTY:+.0f} pts{RESET}")
        else:
            print(f"  {GREEN}No penalty{RESET}")
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
