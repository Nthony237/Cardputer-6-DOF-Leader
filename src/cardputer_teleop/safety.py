"""Safety primitives shared by sim and real: workspace bounds, speed limits, heartbeat watchdog.

These sit between any input device (keyboard now, Cardputer later) and the IK, so the same
limits apply no matter where commands come from.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Workspace:
    """Allowed TCP positions in the robot base frame: an axis-aligned box minus a keep-out
    cylinder around the base, intersected with a max-reach sphere."""

    lower: np.ndarray = field(default_factory=lambda: np.array([0.10, -0.40, 0.03]))
    upper: np.ndarray = field(default_factory=lambda: np.array([0.55, 0.40, 0.55]))
    min_radius_xy: float = 0.10  # keep the TCP out of the base column
    max_reach: float = 0.60  # from the base origin

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

    def contains(self, p: np.ndarray, tol: float = 1e-9) -> bool:
        p = np.asarray(p, dtype=float)
        if np.any(p < self.lower - tol) or np.any(p > self.upper + tol):
            return False
        if np.linalg.norm(p[:2]) < self.min_radius_xy - tol:
            return False
        return bool(np.linalg.norm(p) <= self.max_reach + tol)

    def clamp(self, p: np.ndarray) -> np.ndarray:
        """A point inside the workspace near ``p`` (``p`` itself if already inside).

        Alternates projections onto each constraint until all hold; a few rounds suffice.
        """
        p = np.asarray(p, dtype=float).copy()
        for _ in range(50):
            if self.contains(p):
                break
            p = np.clip(p, self.lower, self.upper)
            r_xy = np.linalg.norm(p[:2])
            if r_xy < self.min_radius_xy:
                direction = p[:2] / r_xy if r_xy > 1e-9 else np.array([1.0, 0.0])
                p[:2] = direction * self.min_radius_xy
            r = np.linalg.norm(p)
            if r > self.max_reach:
                p = p * (self.max_reach / r)
        return p


@dataclass
class SpeedLimits:
    linear: float = 0.15  # m/s at the TCP
    angular: float = 1.0  # rad/s at the TCP
    joint_scale: float = 0.5  # fraction of each URDF joint velocity limit we allow
    joint_default: float = 1.5  # rad/s used when the URDF has no velocity limit

    def clip_twist(self, twist: np.ndarray) -> np.ndarray:
        """Scale linear and angular parts independently so neither exceeds its limit."""
        twist = np.array(twist, dtype=float)
        for sl, limit in ((slice(0, 3), self.linear), (slice(3, 6), self.angular)):
            n = np.linalg.norm(twist[sl])
            if n > limit:
                twist[sl] *= limit / n
        return twist

    def joint_limits(self, urdf_velocity: np.ndarray) -> np.ndarray:
        v = np.where(np.isfinite(urdf_velocity), urdf_velocity, self.joint_default)
        return v * self.joint_scale


class Heartbeat:
    """Watchdog: ``alive()`` is False once ``timeout`` seconds pass without ``beat()``.

    The control loop must hold the arm whenever the heartbeat is not alive.
    """

    def __init__(self, timeout: float = 0.25, clock=time.monotonic):
        self.timeout = timeout
        self._clock = clock
        self._last: float | None = None

    def beat(self) -> None:
        self._last = self._clock()

    def alive(self) -> bool:
        return self._last is not None and (self._clock() - self._last) <= self.timeout

    def reset(self) -> None:
        self._last = None
