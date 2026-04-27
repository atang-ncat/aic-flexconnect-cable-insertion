#!/usr/bin/env python3
"""Live-probe gamepad axis & button indices for AICGamepadEETeleop.

Why this script exists:
  SDL2's button-index assignment for the DualSense / Xbox / F710 depends
  on which Linux kernel driver claimed the device first (hid-playstation
  vs the generic HID parser).  Two different layouts both with 13
  buttons are common, and they relabel almost everything except the
  Triangle face button.  When the gamepad teleop's bindings feel
  wrong, run this script and physically press each button.  It prints
  the live (button index, axis index) so you can verify which layout
  your rig is using and override the relevant ``*_button`` fields on
  ``AICGamepadEETeleopConfig`` if needed.

Usage (from inside the pixi env):
    cd /scratch2/atang/ws_aic/src/aic
    pixi run python /scratch2/atang/ws_aic/scripts/probe_gamepad_buttons.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import os
import time


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    try:
        pygame.display.init()
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
    pygame.joystick.init()

    n = pygame.joystick.get_count()
    if n == 0:
        print("No gamepad found.  Plug one in and re-run.")
        return 1

    j = pygame.joystick.Joystick(0)
    j.init()
    print(f"Connected: {j.get_name()!r}")
    print(f"  GUID: {j.get_guid()}")
    print(f"  axes={j.get_numaxes()}  buttons={j.get_numbuttons()}  hats={j.get_numhats()}")
    print()
    print("Press the buttons / move the sticks you want to identify.")
    print("Active inputs print live below; everything else is suppressed.")
    print("Ctrl+C to stop.\n")

    n_axes = j.get_numaxes()
    n_buttons = j.get_numbuttons()
    n_hats = j.get_numhats()

    last_buttons = [False] * n_buttons
    last_axes = [j.get_axis(i) for i in range(n_axes)]

    try:
        while True:
            pygame.event.pump()

            # Show every button rising-edge press (so the operator
            # can map "pressing this physical button" -> "this index").
            for i in range(n_buttons):
                cur = bool(j.get_button(i))
                if cur and not last_buttons[i]:
                    print(f"BUTTON DOWN  index={i}")
                if not cur and last_buttons[i]:
                    print(f"BUTTON UP    index={i}")
                last_buttons[i] = cur

            # Show axis movements >0.2 from the last "rest" reading
            # so trigger / stick presses print their index without
            # spamming idle readings.
            for i in range(n_axes):
                v = j.get_axis(i)
                if abs(v - last_axes[i]) > 0.2:
                    print(f"AXIS MOVED   index={i}  value={v:+.3f}")
                    last_axes[i] = v

            # Hat is one int (-1, 0, +1) per direction.
            for i in range(n_hats):
                h = j.get_hat(i)
                if h != (0, 0):
                    print(f"HAT {i}: {h}")

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
