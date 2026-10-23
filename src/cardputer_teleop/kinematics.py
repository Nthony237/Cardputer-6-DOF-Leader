"""Serial-chain kinematics read straight from a URDF: forward kinematics and the geometric Jacobian.

Only the chain from the URDF root to ``ee_link`` is used, so side branches (gripper fingers,
camera mounts) are ignored. Fixed joints are folded into their neighbours' origins.
Pure numpy + stdlib XML so the IK can be tested without viser/yourdfpy installed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardputer_teleop.transforms import axis_angle_to_matrix, make_transform, rpy_to_matrix

_MOVABLE = {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class Joint:
    name: str
    type: str  # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray  # 4x4, parent link -> joint frame at q = 0
    axis: np.ndarray  # unit vector in the joint frame
    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = np.inf  # rad/s or m/s, from <limit velocity=...>

    @property
    def movable(self) -> bool:
        return self.type in _MOVABLE

    def motion(self, q: float) -> np.ndarray:
        if self.type in ("revolute", "continuous"):
            return make_transform(R=axis_angle_to_matrix(self.axis, q))
        if self.type == "prismatic":
            return make_transform(p=self.axis * q)
        return np.eye(4)


def _parse_origin(el: ET.Element | None) -> np.ndarray:
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return make_transform(R=rpy_to_matrix(rpy), p=xyz)


def parse_urdf_joints(urdf: str | Path) -> dict[str, Joint]:
    """All joints in a URDF file (or XML string), keyed by child link name."""
    text = Path(urdf).read_text() if not str(urdf).lstrip().startswith("<") else str(urdf)
    root = ET.fromstring(text)
    joints: dict[str, Joint] = {}
    for el in root.findall("joint"):
        jtype = el.get("type")
        axis_el = el.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
        axis = axis / np.linalg.norm(axis)
        limit = el.find("limit")
        lower, upper, vel = -np.inf, np.inf, np.inf
        if limit is not None and jtype in ("revolute", "prismatic"):
            lower = float(limit.get("lower", -np.inf))
            upper = float(limit.get("upper", np.inf))
        if limit is not None and limit.get("velocity") is not None:
            vel = float(limit.get("velocity"))
        child = el.find("child").get("link")
        joints[child] = Joint(
            name=el.get("name"),
            type=jtype,
            parent=el.find("parent").get("link"),
            child=child,
            origin=_parse_origin(el.find("origin")),
            axis=axis,
            lower=lower,
            upper=upper,
            velocity=vel,
        )
    return joints


class SerialChain:
    """Kinematic chain from the URDF root link to ``ee_link``.

    ``q`` vectors are ordered root -> tip and contain only movable joints in the chain.
    An optional ``tcp_offset`` (4x4) is applied after ``ee_link``, e.g. to put the
    controlled point between the gripper fingers.
    """

    def __init__(self, urdf: str | Path, ee_link: str, tcp_offset: np.ndarray | None = None):
        by_child = parse_urdf_joints(urdf)
        chain: list[Joint] = []
        link = ee_link
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.parent
        if not chain:
            raise ValueError(f"Link '{ee_link}' not found as a joint child in the URDF.")
        chain.reverse()
        self.base_link = link
        self.ee_link = ee_link
        self.tcp_offset = np.eye(4) if tcp_offset is None else np.asarray(tcp_offset, dtype=float)

        # Fold fixed joints into the next movable joint's origin (or into the tip offset).
        self.joints: list[Joint] = []
        self._origins: list[np.ndarray] = []
        pending = np.eye(4)
        for j in chain:
            if j.movable:
                self.joints.append(j)
                self._origins.append(pending @ j.origin)
                pending = np.eye(4)
            else:
                pending = pending @ j.origin
        self._tip = pending @ self.tcp_offset

        self.joint_names = [j.name for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        self.velocity_limits = np.array([j.velocity for j in self.joints])

    @property
    def dof(self) -> int:
        return len(self.joints)

    def _frames(self, q: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """World pose of each joint frame (before its motion) and of the TCP."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"Expected q of shape ({self.dof},), got {q.shape}")
        T = np.eye(4)
        frames = []
        for j, origin, qi in zip(self.joints, self._origins, q):
            T = T @ origin
            frames.append(T)
            T = T @ j.motion(qi)
        return frames, T @ self._tip

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Base -> TCP pose (4x4)."""
        return self._frames(q)[1]

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6 x dof geometric Jacobian in the base frame; rows are [v; w] of the TCP."""
        frames, T_ee = self._frames(q)
        p_ee = T_ee[:3, 3]
        J = np.zeros((6, self.dof))
        for i, (j, T) in enumerate(zip(self.joints, frames)):
            z = T[:3, :3] @ j.axis
            if j.type == "prismatic":
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_ee - T[:3, 3])
                J[3:, i] = z
        return J

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        return bool(np.all(q >= self.lower - tol) and np.all(q <= self.upper + tol))
