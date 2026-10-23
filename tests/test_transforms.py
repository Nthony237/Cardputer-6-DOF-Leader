import numpy as np
import pytest

from cardputer_teleop.transforms import (
    make_transform, matrix_to_wxyz, pose_error, rpy_to_matrix, so3_exp, so3_log, wxyz_to_matrix,
)


def random_rotvecs(rng, n=50):
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes * rng.uniform(0, np.pi - 1e-3, size=(n, 1))


def test_exp_log_roundtrip(rng):
    for w in random_rotvecs(rng):
        R = so3_exp(w)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(so3_log(R), w, atol=1e-8)


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
def test_log_near_pi(axis):
    a = np.array(axis, float) / np.linalg.norm(axis)
    for theta in (np.pi - 1e-6, np.pi):
        w = so3_log(so3_exp(a * theta))
        np.testing.assert_allclose(so3_exp(w), so3_exp(a * theta), atol=1e-6)
        assert abs(np.linalg.norm(w) - theta) < 1e-5


def test_log_small_angle():
    w = np.array([1e-9, -2e-9, 3e-9])
    np.testing.assert_allclose(so3_log(so3_exp(w)), w, atol=1e-15)


def test_rpy_matches_composition():
    r, p, y = 0.3, -0.7, 1.9
    Rx = so3_exp([r, 0, 0])
    Ry = so3_exp([0, p, 0])
    Rz = so3_exp([0, 0, y])
    np.testing.assert_allclose(rpy_to_matrix([r, p, y]), Rz @ Ry @ Rx, atol=1e-12)


def test_quaternion_roundtrip(rng):
    for w in random_rotvecs(rng):
        R = so3_exp(w)
        np.testing.assert_allclose(wxyz_to_matrix(matrix_to_wxyz(R)), R, atol=1e-10)


def test_pose_error_zero_and_sign():
    T = make_transform(so3_exp([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose_error(T, T), 0, atol=1e-12)
    T2 = make_transform(so3_exp([0, 0, 0.1]) @ T[:3, :3], T[:3, 3] + [0.01, 0, 0])
    np.testing.assert_allclose(pose_error(T, T2), [0.01, 0, 0, 0, 0, 0.1], atol=1e-12)
