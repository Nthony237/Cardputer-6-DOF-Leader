"""Kinematic simulator: tracks commanded joint positions under per-joint speed limits."""
from __future__ import annotations

import time

import numpy as np


class SimRobot:
    """Joint-space "robot" that moves toward each command at bounded speed.

    No dynamics, but like the real arm it never teleports: each call to
    :meth:`command_joint_positions` advances at most ``qdot_max * dt`` per joint, where ``dt``
    is the wall time since the previous command (capped at ``max_dt``).
    """

    def __init__(self, joint_names, q0, lower, upper, qdot_max, gripper_range=(0.0, 1.0),
                 max_dt: float = 0.1, clock=time.monotonic):
        self.joint_names = list(joint_names)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.qdot_max = np.asarray(qdot_max, dtype=float)
        self._q = np.clip(np.asarray(q0, dtype=float), self.lower, self.upper)
        self._gripper_range = gripper_range
        self._gripper = gripper_range[0]
        self._max_dt = max_dt
        self._clock = clock
        self._last = clock()

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_gripper(self) -> float:
        return self._gripper

    def command_joint_positions(self, q: np.ndarray, gripper: float | None = None) -> None:
        now = self._clock()
        dt = min(now - self._last, self._max_dt)
        self._last = now
        target = np.clip(np.asarray(q, dtype=float), self.lower, self.upper)
        step = np.clip(target - self._q, -self.qdot_max * dt, self.qdot_max * dt)
        self._q = self._q + step
        if gripper is not None:
            self._gripper = float(np.clip(gripper, *self._gripper_range))

    def hold(self) -> None:
        self._last = self._clock()

    def close(self) -> None:
        pass
