"""Robot + teleop settings. Defaults target the placeholder URDF; override with a TOML file.

Example ``yam.toml``::

    [robot]
    urdf = "path/to/yam.urdf"
    ee_link = "tcp"
    gripper_joint = "gripper"
    home = [0.0, 0.24, -0.35, 1.68, 0.0, 0.0]

    [workspace]
    lower = [0.10, -0.40, 0.03]
    upper = [0.55, 0.40, 0.55]

    [speed]
    linear = 0.15
    angular = 1.0
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cardputer_teleop.safety import SpeedLimits, Workspace

ASSETS = Path(__file__).parent / "assets"
PLACEHOLDER_URDF = ASSETS / "yam_placeholder.urdf"


@dataclass
class RobotConfig:
    urdf: Path = PLACEHOLDER_URDF
    ee_link: str = "tcp"
    gripper_joint: str | None = "gripper"
    home: list[float] = field(default_factory=lambda: [0.0, 0.24, -0.35, 1.68, 0.0, 0.0])


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    workspace: Workspace = field(default_factory=Workspace)
    speed: SpeedLimits = field(default_factory=SpeedLimits)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    data = tomllib.loads(path.read_text())
    for key, value in data.get("robot", {}).items():
        if key == "urdf":
            value = (path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        setattr(cfg.robot, key, value)
    for key, value in data.get("workspace", {}).items():
        setattr(cfg.workspace, key, np.asarray(value, dtype=float) if isinstance(value, list) else value)
    for key, value in data.get("speed", {}).items():
        setattr(cfg.speed, key, value)
    return cfg
