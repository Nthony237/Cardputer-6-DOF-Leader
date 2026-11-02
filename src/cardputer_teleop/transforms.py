"""Small SE(3)/SO(3) helpers on plain numpy arrays (4x4 homogeneous, 3x3 rotations)."""
from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula. ``axis`` must be a unit vector."""
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix."""
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    return axis_angle_to_matrix(w / theta, theta)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector, robust near 0 and pi."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        # First-order: R ~ I + [w]x
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        # Near pi: axis from the diagonal of (R + I) / 2 = a a^T.
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / np.sqrt(max(B[i, i], 1e-12))
        # Pick the sign consistent with the antisymmetric part (if any is left).
        s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(a, s) < 0:
            a = -a
        return a * theta
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def make_transform(R: np.ndarray | None = None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-vector [dp, dw] in the base frame that moves ``T_current`` toward ``T_target``."""
    dp = T_target[:3, 3] - T_current[:3, 3]
    dw = so3_log(T_target[:3, :3] @ T_current[:3, :3].T)
    return np.concatenate([dp, dw])


def matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), the order viser uses."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def wxyz_to_matrix(wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
