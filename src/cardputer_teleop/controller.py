"""Cartesian teleop controller: input twist -> safe TCP target -> IK joint command.

Device-agnostic. The keyboard drives it now; the Cardputer will drive it in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cardputer_teleop.ik import DifferentialIK, IKResult
from cardputer_teleop.safety import Heartbeat, Workspace
from cardputer_teleop.transforms import make_transform, so3_exp, so3_log


@dataclass
class CartesianController:
    ik: DifferentialIK
    workspace: Workspace = field(default_factory=Workspace)
    heartbeat: Heartbeat = field(default_factory=Heartbeat)
    # How far the target may lead the actual TCP. Stops an unreachable target from
    # "winding up" so the arm doesn't lurch when it becomes reachable again.
    max_lead_linear: float = 0.05  # m
    max_lead_angular: float = 0.5  # rad
    T_target: np.ndarray = field(default_factory=lambda: np.eye(4))
    held: bool = False

    @property
    def chain(self):
        return self.ik.chain

    def reset_target(self, q: np.ndarray) -> None:
        """Snap the target to the current TCP pose (used on start, hold and resume)."""
        self.T_target = self.chain.fk(q)

    def apply_twist(self, twist: np.ndarray, dt: float, q: np.ndarray, rotation_frame: str = "base") -> None:
        """Integrate a commanded TCP twist [v (base frame); w] into the target.

        ``rotation_frame`` is "base" (rotate about fixed world axes) or "tool" (about the
        gripper's own axes, which feels more natural for wrist motions).
        """
        twist = self.ik.speed.clip_twist(twist)
        p = self.T_target[:3, 3] + twist[:3] * dt
        R = self.T_target[:3, :3]
        dR = so3_exp(twist[3:] * dt)
        R = dR @ R if rotation_frame == "base" else R @ dR
        self.set_target(make_transform(R, p), q)

    def set_target(self, T: np.ndarray, q: np.ndarray) -> None:
        """Set an absolute target, clamped to the workspace and leashed to the current TCP."""
        T_ee = self.chain.fk(q)
        p = self.workspace.clamp(T[:3, 3])
        lead = p - T_ee[:3, 3]
        n = np.linalg.norm(lead)
        if n > self.max_lead_linear:
            p = T_ee[:3, 3] + lead * (self.max_lead_linear / n)
            p = self.workspace.clamp(p)
        R = T[:3, :3]
        w = so3_log(R @ T_ee[:3, :3].T)
        a = np.linalg.norm(w)
        if a > self.max_lead_angular:
            R = so3_exp(w * (self.max_lead_angular / a)) @ T_ee[:3, :3]
        self.T_target = make_transform(R, p)

    def update(self, q: np.ndarray, dt: float) -> IKResult | None:
        """One control tick. Returns None (and freezes the target) when holding."""
        if self.held or not self.heartbeat.alive():
            self.reset_target(q)
            return None
        return self.ik.step(q, self.T_target, dt)
