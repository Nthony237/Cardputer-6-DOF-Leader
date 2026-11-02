"""Cartesian keyboard input: held keys -> TCP twist, plus single-press commands.

Two backends:

* ``terminal`` (default): reads the terminal in cbreak mode, no extra permissions. Terminals
  only report key presses, so a key counts as held for a short window after each press and
  autorepeat keeps it alive. A tap therefore moves a small fixed step.
* ``pynput``: real key-down/key-up events, smoother. On macOS it needs Accessibility /
  Input Monitoring permission for your terminal app (``pip install pynput``).
"""
from __future__ import annotations

import os
import select
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

# key -> (twist index, sign). Indices 0-2 linear x/y/z, 3-5 angular x/y/z.
MOTION_KEYS: dict[str, tuple[int, float]] = {
    "w": (0, +1), "s": (0, -1),  # forward / back
    "a": (1, +1), "d": (1, -1),  # left / right
    "r": (2, +1), "f": (2, -1),  # up / down
    "u": (3, +1), "o": (3, -1),  # roll
    "i": (4, +1), "k": (4, -1),  # pitch
    "j": (5, +1), "l": (5, -1),  # yaw
}
GRIPPER_KEYS: dict[str, float] = {"n": -1.0, "m": +1.0}  # close / open
COMMAND_KEYS = {
    " ": "toggle_hold",
    "h": "home",
    "t": "toggle_frame",
    "[": "slower",
    "]": "faster",
    "q": "quit",
    "\x1b": "quit",  # Esc
}

HELP = """\
Cartesian keyboard teleop (sim)
  w/s  +x/-x (forward/back)      u/o  roll  +/-
  a/d  +y/-y (left/right)        i/k  pitch +/-
  r/f  +z/-z (up/down)           j/l  yaw   +/-
  n/m  close/open gripper        t    rotate in base/tool frame
  space  hold/resume             h    go home
  [ / ]  slower/faster           q/Esc  quit
"""


@dataclass
class HeldKeys:
    """Tracks which keys are down. ``first_hold``/``repeat_hold`` emulate key-up for
    press-only sources; sources with real releases call :meth:`release`."""

    first_hold: float = 0.30  # s a fresh press counts as held (covers autorepeat delay)
    repeat_hold: float = 0.10  # s each autorepeat extends it
    _until: dict[str, float] = field(default_factory=dict)

    def press(self, key: str, now: float) -> None:
        held = key in self._until and self._until[key] >= now
        self._until[key] = now + (self.repeat_hold if held else self.first_hold)

    def press_forever(self, key: str) -> None:
        self._until[key] = float("inf")

    def release(self, key: str) -> None:
        self._until.pop(key, None)

    def is_held(self, key: str, now: float) -> bool:
        return self._until.get(key, -1.0) >= now

    def clear(self) -> None:
        self._until.clear()


def twist_from_keys(held: HeldKeys, now: float, linear_speed: float, angular_speed: float) -> np.ndarray:
    """6-vector TCP twist from the currently held motion keys."""
    twist = np.zeros(6)
    for key, (idx, sign) in MOTION_KEYS.items():
        if held.is_held(key, now):
            twist[idx] += sign
    twist[:3] *= linear_speed
    twist[3:] *= angular_speed
    return twist


def gripper_from_keys(held: HeldKeys, now: float) -> float:
    """-1 closing, +1 opening, 0 idle."""
    return float(sum(v for k, v in GRIPPER_KEYS.items() if held.is_held(k, now)))


class KeyboardInput:
    """Background key reader. Call :meth:`poll` once per control tick."""

    def __init__(self, backend: str = "terminal", clock=time.monotonic):
        self.held = HeldKeys()
        self._clock = clock
        self._commands: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.backend = backend
        if backend == "terminal":
            self._thread = threading.Thread(target=self._terminal_loop, daemon=True)
            self._thread.start()
        elif backend == "pynput":
            self._start_pynput()
        else:
            raise ValueError(f"Unknown keyboard backend '{backend}'")

    # -- shared --------------------------------------------------------------
    def _on_press(self, key: str) -> None:
        key = key.lower() if len(key) == 1 and key != "\x1b" else key
        with self._lock:
            if key in MOTION_KEYS or key in GRIPPER_KEYS:
                if self.backend == "pynput":
                    self.held.press_forever(key)
                else:
                    self.held.press(key, self._clock())
            elif key in COMMAND_KEYS:
                self._commands.append(COMMAND_KEYS[key])

    def poll(self) -> tuple[HeldKeys, list[str]]:
        with self._lock:
            cmds, self._commands = self._commands, []
        return self.held, cmds

    def close(self) -> None:
        self._stop.set()
        if self.backend == "pynput":
            self._listener.stop()

    # -- terminal backend ----------------------------------------------------
    def _terminal_loop(self) -> None:
        if not sys.stdin.isatty():
            print("stdin is not a TTY; keyboard input disabled.", file=sys.stderr)
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if r:
                    for ch in os.read(fd, 32).decode(errors="ignore"):
                        self._on_press(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # -- pynput backend ------------------------------------------------------
    def _start_pynput(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise SystemExit("pynput backend needs `pip install pynput`.") from e

        def name(k):
            if k == keyboard.Key.space:
                return " "
            if k == keyboard.Key.esc:
                return "\x1b"
            return getattr(k, "char", None)

        def on_press(k):
            ch = name(k)
            if ch:
                self._on_press(ch)

        def on_release(k):
            ch = name(k)
            if ch:
                with self._lock:
                    self.held.release(ch.lower())

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
