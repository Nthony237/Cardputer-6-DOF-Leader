import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cardputer_teleop.fake_sender import FakeCardputer
from cardputer_teleop.orientation import Madgwick, quat_from_rpy, quat_mul, quat_to_rpy, remove_yaw
from cardputer_teleop.transforms import rpy_to_matrix, wxyz_to_matrix

FIRMWARE_INCLUDE = Path(__file__).parents[1] / "firmware" / "cardputer_leader" / "include"


def same_rotation(q1, q2, tol=1e-6):
    return abs(abs(float(np.dot(q1, q2))) - 1.0) < tol  # q and -q are the same rotation


def test_rpy_roundtrip_matches_urdf_convention(rng):
    for _ in range(20):
        rpy = rng.uniform([-3, -1.4, -3], [3, 1.4, 3])
        q = quat_from_rpy(*rpy)
        np.testing.assert_allclose(wxyz_to_matrix(q), rpy_to_matrix(rpy), atol=1e-9)
        np.testing.assert_allclose(quat_to_rpy(q), rpy, atol=1e-9)


def test_quat_mul_is_matrix_product(rng):
    a, b = (quat_from_rpy(*rng.uniform(-2, 2, 3)) for _ in range(2))
    np.testing.assert_allclose(wxyz_to_matrix(quat_mul(a, b)), wxyz_to_matrix(a) @ wxyz_to_matrix(b), atol=1e-9)


def test_remove_yaw_keeps_tilt():
    q = quat_from_rpy(0.3, -0.2, 1.1)
    roll, pitch, yaw = quat_to_rpy(remove_yaw(q, 1.1))
    assert abs(yaw) < 1e-9 and abs(roll - 0.3) < 1e-9 and abs(pitch + 0.2) < 1e-9


def test_madgwick_finds_tilt_from_gravity():
    """Held still and tilted, the filter should settle on the true roll and pitch.

    Same schedule as the firmware: a high gain for 2 s to snap to the tilt, then the low gain
    that smooths out the small wobble the high gain leaves."""
    q_true = quat_from_rpy(math.radians(25), math.radians(-15), 0.0)
    accel = wxyz_to_matrix(q_true).T @ np.array([0, 0, 9.81])  # gravity seen by the device
    f = Madgwick(beta=2.0)
    for _ in range(400):
        f.update([0, 0, 0], accel, 0.005)
    f.beta = 0.04
    for _ in range(400):
        f.update([0, 0, 0], accel, 0.005)
    roll, pitch, _ = quat_to_rpy(f.q)
    assert abs(math.degrees(roll) - 25) < 0.5 and abs(math.degrees(pitch) + 15) < 0.5


def test_madgwick_tracks_rotation_with_gyro():
    """Following the fake device's motion with its gyro + accel keeps the error small."""
    dev = FakeCardputer(calibrate_seconds=0.0)
    f = Madgwick(beta=0.04)
    dt = 0.005
    dev.packet(0.0)
    f.q = dev.orientation(0.0)
    worst = 0.0
    for i in range(1, 2000):  # 10 s
        pkt = dev.packet(i * dt)
        f.update(pkt.gyro, pkt.accel, dt)
        err = np.linalg.norm(quat_to_rpy(quat_mul(f.q, pkt.quat * [1, -1, -1, -1])))
        worst = max(worst, err)
    assert math.degrees(worst) < 2.0


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++ to compile the firmware filter")
def test_firmware_filter_matches_python(tmp_path):
    """Compile the firmware's madgwick.h on this computer and check it agrees with orientation.py."""
    src = tmp_path / "check.cpp"
    src.write_text(r'''
#include <cstdio>
#include "madgwick.h"
int main() {
  Madgwick f; f.beta = 0.1f;
  for (int i = 0; i < 500; i++) {
    float t = i * 0.005f;
    f.update(0.5f * sinf(t), 0.3f, -0.2f * cosf(2 * t), 0.2f, 0.4f * sinf(t), 9.6f, 0.005f);
  }
  printf("%.7f %.7f %.7f %.7f\n", f.w, f.x, f.y, f.z);
}
''')
    exe = tmp_path / "check"
    subprocess.run(["g++", "-std=c++17", f"-I{FIRMWARE_INCLUDE}", str(src), "-o", str(exe)], check=True)
    q_cpp = np.array([float(v) for v in subprocess.run([str(exe)], capture_output=True, text=True,
                                                         check=True).stdout.split()])
    f = Madgwick(beta=0.1)
    for i in range(500):
        t = i * 0.005
        f.update([0.5 * math.sin(t), 0.3, -0.2 * math.cos(2 * t)], [0.2, 0.4 * math.sin(t), 9.6], 0.005)
    assert same_rotation(q_cpp, f.q, tol=1e-4)
