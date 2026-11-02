"""Damped-least-squares differential IK with joint position and velocity limits.

Each call to :meth:`DifferentialIK.step` makes one bounded move of the TCP toward a target
pose. It never jumps: joint speed is capped, joints are stopped before their limits, and
damping grows near singularities so the arm slows down instead of flailing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cardputer_teleop.kinematics import SerialChain
from cardputer_teleop.safety import SpeedLimits
from cardputer_teleop.transforms import pose_error


@dataclass
class IKResult:
    q: np.ndarray  # next joint positions
    dq: np.ndarray  # joint velocity used for this step
    error: np.ndarray  # 6-vector pose error before the step [dp, dw]
    manipulability: float
    damping: float
    locked: list[int] = field(default_factory=list)  # joints held at a limit this step


@dataclass
class DifferentialIK:
    chain: SerialChain
    speed: SpeedLimits = field(default_factory=SpeedLimits)
    gain_linear: float = 8.0  # 1/s, proportional gain on position error
    gain_angular: float = 6.0  # 1/s, proportional gain on orientation error
    damping_max: float = 0.08  # lambda at a singularity
    damping_min: float = 1e-3  # always-on numerical floor
    manipulability_threshold: float = 0.004  # below this, damping ramps up
    limit_margin: float = 0.02  # rad (or m) kept clear of each URDF joint limit
    position_only: bool = False  # ignore orientation (useful for <6-DOF arms)

    def __post_init__(self):
        self.qdot_max = self.speed.joint_limits(self.chain.velocity_limits)
        self.lower = self.chain.lower + self.limit_margin
        self.upper = self.chain.upper - self.limit_margin

    def damping(self, J: np.ndarray) -> tuple[float, float]:
        """Nakamura-style adaptive damping from Yoshikawa manipulability sqrt(det(J J^T))."""
        w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        w0 = self.manipulability_threshold
        lam = self.damping_min
        if w < w0:
            lam = max(lam, self.damping_max * np.sqrt(1.0 - (w / w0) ** 2))
        return lam, w

    def _solve(self, J: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
        m = J.shape[0]
        return J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(m), v)

    def desired_twist(self, T_current: np.ndarray, T_target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        err = pose_error(T_current, T_target)
        # Gains are capped at 1/dt so a single step never overshoots the target.
        kl = min(self.gain_linear, 1.0 / dt)
        ka = min(self.gain_angular, 1.0 / dt)
        v = np.concatenate([kl * err[:3], ka * err[3:]])
        return self.speed.clip_twist(v), err

    def step(self, q: np.ndarray, T_target: np.ndarray, dt: float) -> IKResult:
        q = np.asarray(q, dtype=float)
        v, err = self.desired_twist(self.chain.fk(q), T_target, dt)
        return self.step_twist(q, v, dt, error=err)

    def step_twist(self, q: np.ndarray, twist: np.ndarray, dt: float, error: np.ndarray | None = None) -> IKResult:
        """Track a base-frame TCP twist [v; w] for ``dt`` seconds."""
        q = np.asarray(q, dtype=float)
        J_full = self.chain.jacobian(q)
        rows = slice(0, 3) if self.position_only else slice(0, 6)
        J_full, v = J_full[rows], np.asarray(twist, dtype=float)[rows]
        lam, w = self.damping(J_full)

        n = self.chain.dof
        free = np.ones(n, dtype=bool)
        dq = np.zeros(n)
        # Active-set loop: lock any joint that would cross its (margined) limit while moving
        # outward, then re-solve with the remaining joints.
        for _ in range(n + 1):
            dq = np.zeros(n)
            if free.any():
                dq[free] = self._solve(J_full[:, free], v, lam)
            # Uniformly scale so every joint respects its velocity limit (keeps direction).
            ratio = np.max(np.abs(dq) / self.qdot_max) if n else 0.0
            if ratio > 1.0:
                dq /= ratio
            q_next = q + dq * dt
            violating = free & (
                ((q_next < self.lower) & (dq < 0)) | ((q_next > self.upper) & (dq > 0))
            )
            if not violating.any():
                break
            free &= ~violating

        q_next = np.clip(q + dq * dt, self.chain.lower, self.chain.upper)
        return IKResult(
            q=q_next,
            dq=dq,
            error=error if error is not None else np.zeros(6),
            manipulability=w,
            damping=lam,
            locked=[int(i) for i in np.flatnonzero(~free)],
        )

    def solve(self, q0: np.ndarray, T_target: np.ndarray, dt: float = 0.02, max_steps: int = 500,
              tol_pos: float = 1e-4, tol_rot: float = 1e-3) -> tuple[np.ndarray, bool]:
        """Iterate :meth:`step` to convergence (for tests and choosing home poses, not teleop)."""
        q = np.asarray(q0, dtype=float)
        for _ in range(max_steps):
            res = self.step(q, T_target, dt)
            q = res.q
            if np.linalg.norm(res.error[:3]) < tol_pos and (
                self.position_only or np.linalg.norm(res.error[3:]) < tol_rot
            ):
                return q, True
        err = pose_error(self.chain.fk(q), T_target)
        ok = np.linalg.norm(err[:3]) < tol_pos and (self.position_only or np.linalg.norm(err[3:]) < tol_rot)
        return q, bool(ok)
