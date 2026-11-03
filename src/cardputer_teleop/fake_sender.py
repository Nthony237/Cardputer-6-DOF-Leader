"""Pretend to be the Cardputer: send realistic leader packets over UDP.

Lets you (and the tests) run the receiver, viewer and teleop with no hardware. Scenarios:

* ``rock`` (default): slowly rocks in roll, pitch and yaw, holds the clutch for 2 s out of
  every 4, and alternates the gripper keys.
* ``push``: held level with the clutch down, the hand makes quick 10 cm pushes forward, left
  and up, then back, then lets go of the clutch. Tests push-to-nudge translation.
* ``tilt``: holds the tilt key (x) and tips forward, then right, then twists. Tests
  tilt-as-joystick translation.
* ``hand``: holds the clutch for 7 s out of every 8 and turns gently. Pair it with
  ``cardputer-teleop --hand fake`` (or a real webcam) to test Phase 4 hand-position control.
* ``wrist``: the same turning but no keys at all, as if strapped to a wrist. Pair it with
  ``--hand fake --pinch``: the (fake) pinch does the grabbing.

Options simulate a bad network: --drop loses a fraction of packets and --stall-after/--stall-for
go silent for a while to show the heartbeat watchdog holding. --accel-noise adds sensor noise.

    cardputer-fake                     # to this laptop, port 9870
    cardputer-fake --scenario push
    cardputer-fake --drop 0.1          # 10% packet loss
    cardputer-fake --stall-after 5 --stall-for 2
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import time

import numpy as np

from cardputer_teleop.orientation import quat_from_rpy
from cardputer_teleop.protocol import (
    DEFAULT_PORT,
    FLAG_CALIBRATING,
    FLAG_CLUTCH,
    FLAG_GRIP_CLOSE,
    FLAG_GRIP_OPEN,
    FLAG_TILT_MOVE,
    LeaderPacket,
)
from cardputer_teleop.transforms import so3_log, wxyz_to_matrix

GRAVITY = 9.80665
SCENARIOS = ("rock", "push", "tilt", "hand", "wrist")
CYCLE = 8.0  # s, length of one push / tilt routine

# push routine: (start time in the cycle, duration, displacement in m). Each is a smooth
# "minimum-jerk" reach, the bell-shaped speed profile human reaches have.
PUSHES = [(0.5, 0.4, (0.10, 0.0, 0.0)), (2.0, 0.4, (0.0, 0.10, 0.0)), (3.5, 0.4, (0.0, 0.0, 0.08)),
          (5.0, 0.8, (-0.10, -0.10, -0.08))]
# tilt routine: (start, duration, (roll, pitch, yaw) in deg held then released)
TILTS = [(1.0, 1.5, (0.0, 20.0, 0.0)), (3.0, 1.5, (20.0, 0.0, 0.0)), (5.0, 1.5, (0.0, 0.0, 30.0))]


def _min_jerk(s: float) -> tuple[float, float]:
    """Position fraction and acceleration factor (x D/T^2) of a minimum-jerk move at s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 60 * s - 180 * s**2 + 120 * s**3


def _bump(t: float, start: float, dur: float, ramp: float = 0.3) -> float:
    """0 -> 1 -> 0 with smooth ramps: a tilt that is tipped, held, and brought back."""
    up = _min_jerk((t - start) / ramp)[0]
    down = _min_jerk((t - (start + dur - ramp)) / ramp)[0]
    return up - down


class FakeCardputer:
    """Generates the packet a real device would send at time ``t`` (seconds since boot)."""

    def __init__(self, calibrate_seconds: float = 1.0, scenario: str = "rock", accel_noise: float = 0.0,
                 seed: int | None = None):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.calibrate_seconds = calibrate_seconds
        self.scenario = scenario
        self.accel_noise = accel_noise
        self.seq = 0
        self._rng = np.random.default_rng(seed)
        self._last: tuple[float, np.ndarray] | None = None

    @staticmethod
    def orientation(t: float) -> np.ndarray:
        roll = math.radians(30) * math.sin(0.9 * t)
        pitch = math.radians(20) * math.sin(0.6 * t)
        yaw = math.radians(45) * math.sin(0.3 * t)
        return quat_from_rpy(roll, pitch, yaw)

    @staticmethod
    def hand_motion(c: float) -> tuple[np.ndarray, np.ndarray]:
        """push scenario at ``c`` s into the routine: hand offset (m) and acceleration (m/s^2), world frame."""
        pos, acc = np.zeros(3), np.zeros(3)
        for start, dur, d in PUSHES:
            f, a = _min_jerk((c - start) / dur)
            pos += f * np.asarray(d)
            if start <= c <= start + dur:
                acc += a * np.asarray(d) / dur**2
        return pos, acc

    def _state(self, t: float) -> tuple[np.ndarray, np.ndarray, int]:
        """Orientation, world-frame hand acceleration and key flags for the scenario."""
        c = (t - self.calibrate_seconds) % CYCLE  # routines start once calibration is over
        if self.scenario == "push":
            q = quat_from_rpy(0.0, 0.0, 0.0)
            flags = FLAG_CLUTCH if c < 7.0 else 0  # let go for the last second: next cycle re-grabs
            return q, self.hand_motion(c)[1], flags
        if self.scenario == "tilt":
            rpy = np.zeros(3)
            for start, dur, angles in TILTS:
                rpy += _bump(c, start, dur) * np.radians(angles)
            flags = FLAG_TILT_MOVE if 0.5 < c < 7.0 else 0
            return quat_from_rpy(*rpy), np.zeros(3), flags
        if self.scenario == "wrist":  # strapped to a wrist: turns, but no keys (pinch does the grabbing)
            q = quat_from_rpy(math.radians(10) * math.sin(0.7 * c), 0.0, math.radians(20) * math.sin(0.4 * c))
            return q, np.zeros(3), 0
        if self.scenario == "hand":
            q = quat_from_rpy(math.radians(10) * math.sin(0.7 * c), 0.0, math.radians(20) * math.sin(0.4 * c))
            return q, np.zeros(3), FLAG_CLUTCH if 0.5 < c < 7.5 else 0
        phase = t % 4.0
        flags = FLAG_CLUTCH if phase < 2.0 else 0
        flags |= FLAG_GRIP_CLOSE if 0.5 < phase < 1.5 else 0
        flags |= FLAG_GRIP_OPEN if 2.5 < phase < 3.5 else 0
        return self.orientation(t), np.zeros(3), flags

    def packet(self, t: float) -> LeaderPacket:
        calibrating = t < self.calibrate_seconds
        if calibrating:
            q, a_world, flags = np.array([1.0, 0, 0, 0]), np.zeros(3), FLAG_CALIBRATING
        else:
            q, a_world, flags = self._state(t)
        R = wxyz_to_matrix(q)
        gyro = np.zeros(3)
        if self._last is not None and t > self._last[0]:
            t0, R0 = self._last
            gyro = so3_log(R0.T @ R) / (t - t0)  # body-frame angular velocity
        self._last = (t, R)
        # An accelerometer feels "up" g at rest, plus whatever the hand is doing.
        accel = R.T @ (a_world + np.array([0.0, 0.0, GRAVITY]))
        if self.accel_noise:
            accel = accel + self._rng.normal(0.0, self.accel_noise, 3)
        pkt = LeaderPacket(seq=self.seq, t_ms=int(t * 1000), quat=q, gyro=gyro, accel=accel, flags=flags,
                           battery_pct=87)
        self.seq += 1
        return pkt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Where the receiver runs")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--scenario", choices=SCENARIOS, default="rock", help="What the fake hand does")
    p.add_argument("--accel-noise", type=float, default=0.05, help="Accelerometer noise, m/s^2 std")
    p.add_argument("--drop", type=float, default=0.0, help="Fraction of packets to lose (0-1)")
    p.add_argument("--stall-after", type=float, default=0.0, help="Go silent after N s (0 = never)")
    p.add_argument("--stall-for", type=float, default=2.0, help="How long to stay silent")
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N s (0 = run until Ctrl+C)")
    return p.parse_args(argv)


def run(host: str = "127.0.0.1", port: int = DEFAULT_PORT, hz: float = 100.0, drop: float = 0.0,
        stall_after: float = 0.0, stall_for: float = 2.0, duration: float = 0.0, stop=None,
        verbose: bool = True, scenario: str = "rock", accel_noise: float = 0.0) -> int:
    """Send packets until ``duration`` passes or ``stop`` (a threading.Event) is set. Returns # sent."""
    device = FakeCardputer(scenario=scenario, accel_noise=accel_noise)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.monotonic()
    next_tick, sent, last_print = t0, 0, t0
    try:
        while not (stop is not None and stop.is_set()):
            now = time.monotonic()
            t = now - t0
            if duration and t > duration:
                break
            pkt = device.packet(t)
            stalled = stall_after and stall_after <= t < stall_after + stall_for
            if not stalled and random.random() >= drop:
                sock.sendto(pkt.pack(), (host, port))
                sent += 1
            if verbose and now - last_print >= 1.0:
                last_print = now
                state = "STALLED (watchdog should hold)" if stalled else "sending"
                print(f"t={t:5.1f}s seq={pkt.seq:6d} sent={sent:6d} {state}", flush=True)
            next_tick += 1.0 / hz
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return sent


def main(argv=None):
    a = parse_args(argv)
    print(f"Fake Cardputer ({a.scenario}) -> {a.host}:{a.port} at {a.hz:.0f} Hz (Ctrl+C to stop)")
    run(a.host, a.port, a.hz, a.drop, a.stall_after, a.stall_for, a.duration, scenario=a.scenario,
        accel_noise=a.accel_noise)


if __name__ == "__main__":
    main()
