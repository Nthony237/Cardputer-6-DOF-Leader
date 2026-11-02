import numpy as np
import pytest

from cardputer_teleop.config import PLACEHOLDER_URDF, RobotConfig
from cardputer_teleop.ik import DifferentialIK
from cardputer_teleop.kinematics import SerialChain


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def chain():
    return SerialChain(PLACEHOLDER_URDF, "tcp")


@pytest.fixture
def ik(chain):
    return DifferentialIK(chain)


@pytest.fixture
def q_home():
    return np.array(RobotConfig().home)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
