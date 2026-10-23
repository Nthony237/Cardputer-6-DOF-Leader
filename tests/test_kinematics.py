import numpy as np
import pytest

from cardputer_teleop.config import PLACEHOLDER_URDF
from cardputer_teleop.kinematics import SerialChain


def test_chain_skips_gripper_branch(chain):
    assert chain.joint_names == [f"joint{i}" for i in range(1, 7)]
    assert chain.base_link == "base_link"


def test_fk_at_zero(chain):
    # Upper arm straight up (0.09 + 0.04 + 0.26), forearm + wrist + tool along +x.
    T = chain.fk(np.zeros(6))
    np.testing.assert_allclose(T[:3, 3], [0.24 + 0.06 + 0.05 + 0.10, 0, 0.39], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_base_yaw_rotates_tcp(chain):
    q = np.zeros(6)
    q[0] = np.pi / 2
    np.testing.assert_allclose(chain.fk(q)[:3, 3], [0, 0.45, 0.39], atol=1e-12)


def test_jacobian_matches_finite_differences(chain, rng):
    from cardputer_teleop.transforms import so3_log

    for _ in range(20):
        q = rng.uniform(chain.lower, chain.upper)
        J = chain.jacobian(q)
        T0 = chain.fk(q)
        eps = 1e-6
        J_fd = np.zeros_like(J)
        for i in range(chain.dof):
            dq = np.zeros(chain.dof)
            dq[i] = eps
            T1 = chain.fk(q + dq)
            J_fd[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            J_fd[3:, i] = so3_log(T1[:3, :3] @ T0[:3, :3].T) / eps
        np.testing.assert_allclose(J, J_fd, atol=1e-5)


def test_limits_parsed(chain):
    assert chain.lower[1] == -1.0 and chain.upper[1] == 2.2
    assert np.all(np.isfinite(chain.velocity_limits))


def test_tcp_offset(chain):
    from cardputer_teleop.transforms import make_transform

    c2 = SerialChain(PLACEHOLDER_URDF, "tcp", tcp_offset=make_transform(p=[0.02, 0, 0]))
    np.testing.assert_allclose(c2.fk(np.zeros(6))[:3, 3], chain.fk(np.zeros(6))[:3, 3] + [0.02, 0, 0])


def test_unknown_link_raises():
    with pytest.raises(ValueError):
        SerialChain(PLACEHOLDER_URDF, "no_such_link")


def test_bad_q_shape(chain):
    with pytest.raises(ValueError):
        chain.fk(np.zeros(5))
