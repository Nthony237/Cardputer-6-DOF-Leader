"""Sim/real switch: :func:`make_robot` returns an object implementing :class:`Robot`."""
from __future__ import annotations

from cardputer_teleop.robots.base import Robot
from cardputer_teleop.robots.sim import SimRobot


def make_robot(mode: str, *, chain, q0, qdot_max, gripper_range=(0.0, 1.0), **kwargs) -> Robot:
    if mode == "sim":
        return SimRobot(chain.joint_names, q0, chain.lower, chain.upper, qdot_max, gripper_range)
    if mode == "real":
        from cardputer_teleop.robots.real import YamRobot

        return YamRobot(**kwargs)
    raise ValueError(f"Unknown mode '{mode}', expected 'sim' or 'real'.")


__all__ = ["Robot", "SimRobot", "make_robot"]
