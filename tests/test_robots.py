import numpy as np
import pytest

from cardputer_teleop.robots import make_robot
from cardputer_teleop.robots.sim import SimRobot


def test_sim_rate_limits(clock):
    r = SimRobot(["a", "b"], [0, 0], [-1, -1], [1, 1], qdot_max=[1.0, 0.5], clock=clock)
    clock.advance(0.1)
    r.command_joint_positions(np.array([1.0, 1.0]))
    np.testing.assert_allclose(r.get_joint_positions(), [0.1, 0.05])


def test_sim_clamps_to_limits(clock):
    r = SimRobot(["a"], [0.95], [-1], [1], qdot_max=[10.0], clock=clock)
    clock.advance(0.1)
    r.command_joint_positions(np.array([5.0]))
    np.testing.assert_allclose(r.get_joint_positions(), [1.0])


def test_sim_caps_large_dt(clock):
    r = SimRobot(["a"], [0.0], [-10], [10], qdot_max=[1.0], max_dt=0.1, clock=clock)
    clock.advance(5.0)  # e.g. the loop stalled
    r.command_joint_positions(np.array([5.0]))
    np.testing.assert_allclose(r.get_joint_positions(), [0.1])


def test_make_robot_sim(chain, q_home):
    r = make_robot("sim", chain=chain, q0=q_home, qdot_max=np.ones(6))
    np.testing.assert_allclose(r.get_joint_positions(), q_home)


def test_make_robot_real_not_implemented(chain, q_home):
    with pytest.raises(NotImplementedError):
        make_robot("real", chain=chain, q0=q_home, qdot_max=np.ones(6))


def test_make_robot_bad_mode(chain, q_home):
    with pytest.raises(ValueError):
        make_robot("bogus", chain=chain, q0=q_home, qdot_max=np.ones(6))
