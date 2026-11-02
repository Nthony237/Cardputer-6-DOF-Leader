"""Quaternion helpers and the Madgwick IMU filter.

The filter here is the same math as ``firmware/cardputer_leader/include/madgwick.h``. The
firmware runs it on the Cardputer; this copy exists so the math is unit-tested and so later
phases can re-filter raw IMU data on the laptop if needed.

Quaternions are (w, x, y, z) like viser, and rotate device-frame vectors into the world frame.
"""
from __future__ import annotations

import math

import numpy as np


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), the usual aircraft / URDF convention."""
    qx = quat_from_axis_angle([1, 0, 0], roll)
    qy = quat_from_axis_angle([0, 1, 0], pitch)
    qz = quat_from_axis_angle([0, 0, 1], yaw)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def remove_yaw(q, yaw: float) -> np.ndarray:
    """Rotate the world frame so heading ``yaw`` becomes zero (what the zero-yaw key does)."""
    return quat_mul(quat_from_axis_angle([0, 0, 1], -yaw), q)


class Madgwick:
    """IMU-only Madgwick filter. Gyro fixes fast motion, gravity slowly corrects roll/pitch.

    ``beta`` is the correction gain (rad/s): bigger trusts the accelerometer more.
    """

    def __init__(self, beta: float = 0.04):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, accel, dt: float) -> np.ndarray:
        """``gyro`` in rad/s, ``accel`` in any unit (direction only), device frame."""
        w, x, y, z = self.q
        gx, gy, gz = gyro
        qdot = 0.5 * np.array([
            -x * gx - y * gy - z * gz,
            w * gx + y * gz - z * gy,
            w * gy - x * gz + z * gx,
            w * gz + x * gy - y * gx,
        ])
        a = np.asarray(accel, dtype=float)
        an = np.linalg.norm(a)
        if an > 1e-6:
            ax, ay, az = a / an
            # Predicted gravity direction in the device frame minus the measured one...
            f = np.array([2 * (x * z - w * y) - ax, 2 * (w * x + y * z) - ay, 2 * (0.5 - x * x - y * y) - az])
            # ...and its Jacobian transpose gives the direction that shrinks that gap.
            J = np.array([[-2 * y, 2 * z, -2 * w, 2 * x], [2 * x, 2 * w, 2 * z, 2 * y], [0, -4 * x, -4 * y, 0]])
            step = J.T @ f
            sn = np.linalg.norm(step)
            if sn > 1e-9:
                qdot -= self.beta * step / sn
        q = self.q + qdot * dt
        self.q = q / np.linalg.norm(q)
        return self.q
