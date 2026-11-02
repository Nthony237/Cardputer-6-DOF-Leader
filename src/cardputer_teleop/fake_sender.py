"""Pretend to be the Cardputer: send realistic leader packets over UDP.

Lets you (and the tests) run the receiver and viewer with no hardware. The fake device slowly
rocks in roll, pitch and yaw, holds the clutch for 2 s out of every 4, and alternates the
gripper keys. Options simulate a bad network: --drop loses a fraction of packets and
--stall-after/--stall-for go silent for a while to show the heartbeat watchdog holding.

    cardputer-fake                     # to this laptop, port 9870
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
    LeaderPacket,
)
from cardputer_teleop.transforms import so3_log, wxyz_to_matrix

GRAVITY = 9.80665


class FakeCardputer:
    """Generates the packet a real device would send at time ``t`` (seconds since boot)."""

    def __init__(self, calibrate_seconds: float = 1.0):
        self.calibrate_seconds = calibrate_seconds
        self.seq = 0
        self._last: tuple[float, np.ndarray] | None = None

    @staticmethod
    def orientation(t: float) -> np.ndarray:
        roll = math.radians(30) * math.sin(0.9 * t)
        pitch = math.radians(20) * math.sin(0.6 * t)
        yaw = math.radians(45) * math.sin(0.3 * t)
        return quat_from_rpy(roll, pitch, yaw)

    def packet(self, t: float) -> LeaderPacket:
        calibrating = t < self.calibrate_seconds
        q = np.array([1.0, 0, 0, 0]) if calibrating else self.orientation(t)
        R = wxyz_to_matrix(q)
        gyro = np.zeros(3)
        if self._last is not None and t > self._last[0]:
            t0, R0 = self._last
            gyro = so3_log(R0.T @ R) / (t - t0)  # body-frame angular velocity
        self._last = (t, R)
        accel = R.T @ np.array([0.0, 0.0, GRAVITY])  # at rest an accelerometer feels "up"
        phase = t % 4.0
        flags = FLAG_CLUTCH if phase < 2.0 else 0
        flags |= FLAG_GRIP_CLOSE if 0.5 < phase < 1.5 else 0
        flags |= FLAG_GRIP_OPEN if 2.5 < phase < 3.5 else 0
        flags |= FLAG_CALIBRATING if calibrating else 0
        pkt = LeaderPacket(seq=self.seq, t_ms=int(t * 1000), quat=q, gyro=gyro, accel=accel, flags=flags,
                           battery_pct=87)
        self.seq += 1
        return pkt


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Where the receiver runs")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--drop", type=float, default=0.0, help="Fraction of packets to lose (0-1)")
    p.add_argument("--stall-after", type=float, default=0.0, help="Go silent after N s (0 = never)")
    p.add_argument("--stall-for", type=float, default=2.0, help="How long to stay silent")
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N s (0 = run until Ctrl+C)")
    return p.parse_args(argv)


def run(host: str = "127.0.0.1", port: int = DEFAULT_PORT, hz: float = 100.0, drop: float = 0.0,
        stall_after: float = 0.0, stall_for: float = 2.0, duration: float = 0.0, stop=None,
        verbose: bool = True) -> int:
    """Send packets until ``duration`` passes or ``stop`` (a threading.Event) is set. Returns # sent."""
    device = FakeCardputer()
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
    print(f"Fake Cardputer -> {a.host}:{a.port} at {a.hz:.0f} Hz (Ctrl+C to stop)")
    run(a.host, a.port, a.hz, a.drop, a.stall_after, a.stall_for, a.duration)


if __name__ == "__main__":
    main()
