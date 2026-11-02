"""Robot interface shared by the simulator and the real YAM driver."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Robot(Protocol):
    """What the teleop loop needs from an arm. Joint vectors follow ``joint_names`` order."""

    joint_names: list[str]

    def get_joint_positions(self) -> np.ndarray: ...

    def command_joint_positions(self, q: np.ndarray, gripper: float | None = None) -> None:
        """Send a position target. Implementations must rate-limit or reject large jumps."""
        ...

    def get_gripper(self) -> float: ...

    def hold(self) -> None:
        """Stop moving and hold the current position (watchdog / e-stop path)."""
        ...

    def close(self) -> None: ...
