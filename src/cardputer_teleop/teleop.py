"""Cartesian keyboard teleop. ``cardputer-teleop --mode sim`` runs on a laptop with no arms."""
from __future__ import annotations

import argparse
import time

import numpy as np

from cardputer_teleop.config import load_config
from cardputer_teleop.controller import CartesianController
from cardputer_teleop.ik import DifferentialIK
from cardputer_teleop.keyboard import HELP, HeldKeys, KeyboardInput, gripper_from_keys, twist_from_keys
from cardputer_teleop.kinematics import SerialChain
from cardputer_teleop.robots import make_robot
from cardputer_teleop.safety import Heartbeat

GRIPPER_RATE = 1.0  # fraction of full travel per second
ANGULAR_PER_LINEAR = 8.0  # rad/s of rotation per m/s of the speed slider


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sim", "real"), default="sim")
    p.add_argument("--config", help="TOML config (robot URDF, workspace, speed limits)")
    p.add_argument("--urdf", help="Override the URDF path from the config")
    p.add_argument("--ee-link", help="Override the end-effector link from the config")
    p.add_argument("--hz", type=float, default=50.0, help="Control loop rate")
    p.add_argument("--port", type=int, default=8080, help="viser port")
    p.add_argument("--keys", choices=("terminal", "pynput", "none"), default="terminal")
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = run until quit)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.urdf:
        cfg.robot.urdf = args.urdf
    if args.ee_link:
        cfg.robot.ee_link = args.ee_link

    chain = SerialChain(cfg.robot.urdf, cfg.robot.ee_link)
    ik = DifferentialIK(chain, speed=cfg.speed)
    q_home = np.asarray(cfg.robot.home, dtype=float)
    if q_home.shape != (chain.dof,):
        raise SystemExit(f"home has {q_home.size} values but the chain has {chain.dof} joints {chain.joint_names}")

    controller = CartesianController(ik, workspace=cfg.workspace, heartbeat=Heartbeat(timeout=0.25))
    robot = make_robot(args.mode, chain=chain, q0=q_home, qdot_max=ik.qdot_max)
    controller.reset_target(robot.get_joint_positions())

    from cardputer_teleop.viz import SimViewer

    viewer = SimViewer(cfg.robot.urdf, chain.joint_names, cfg.workspace, cfg.robot.gripper_joint,
                       port=args.port, max_linear_speed=cfg.speed.linear)
    keyboard = KeyboardInput(args.keys) if args.keys != "none" else None
    print(HELP)

    dt = 1.0 / args.hz
    gripper = robot.get_gripper()
    homing = False
    t_start = time.monotonic()
    next_tick = t_start
    try:
        while True:
            held, cmds = keyboard.poll() if keyboard else (HeldKeys(), [])
            if "quit" in cmds or (args.duration and time.monotonic() - t_start > args.duration):
                break
            for cmd in cmds:
                if cmd == "toggle_hold":
                    viewer.hold.value = not viewer.hold.value
                elif cmd == "home":
                    homing = True
                elif cmd == "toggle_frame":
                    viewer.frame.value = "base" if viewer.frame.value == "tool" else "tool"
                elif cmd in ("slower", "faster"):
                    factor = 0.5 if cmd == "slower" else 2.0
                    viewer.speed.value = float(np.clip(viewer.speed.value * factor, 0.01, cfg.speed.linear))
            if viewer.home_requested:
                homing, viewer.home_requested = True, False

            # Local keyboard is always connected. The Cardputer receiver will beat() per packet
            # instead, so a dropped Wi-Fi link makes the controller hold.
            controller.heartbeat.beat()
            controller.held = viewer.hold.value

            q = robot.get_joint_positions()
            now = time.monotonic()
            lin = viewer.speed.value
            twist = twist_from_keys(held, now, lin, lin * ANGULAR_PER_LINEAR)
            drag = viewer.take_drag_target()
            if np.any(twist) or drag is not None:
                homing = False

            status = ""
            if controller.held:
                robot.hold()
                controller.reset_target(q)
                status = "**HOLD**"
            elif homing:
                # Joint-space homing: predictable path, independent of the IK.
                step = np.clip(q_home - q, -ik.qdot_max * dt, ik.qdot_max * dt)
                robot.command_joint_positions(q + step, gripper)
                controller.reset_target(robot.get_joint_positions())
                homing = bool(np.max(np.abs(q_home - q)) > 1e-3)
                status = "homing"
            else:
                if drag is not None:
                    controller.set_target(drag, q)
                if np.any(twist):
                    controller.apply_twist(twist, dt, q, rotation_frame=viewer.frame.value)
                gripper = float(np.clip(gripper + gripper_from_keys(held, now) * GRIPPER_RATE * dt, 0.0, 1.0))
                res = controller.update(q, dt)
                if res is None:
                    robot.hold()
                    status = "**HOLD (no heartbeat)**"
                else:
                    robot.command_joint_positions(res.q, gripper)
                    locked = ", ".join(chain.joint_names[i] for i in res.locked) or "none"
                    status = (
                        f"err {np.linalg.norm(res.error[:3]) * 1000:.1f} mm / "
                        f"{np.degrees(np.linalg.norm(res.error[3:])):.1f} deg  \n"
                        f"manipulability {res.manipulability:.4f}, damping {res.damping:.3f}  \n"
                        f"at joint limit: {locked}"
                    )

            q = robot.get_joint_positions()
            T_tcp = chain.fk(q)
            p = T_tcp[:3, 3]
            status += f"  \nTCP [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] m, mode {args.mode}"
            viewer.update(q, robot.get_gripper(), T_tcp, controller.T_target, status)

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        robot.hold()
        if keyboard:
            keyboard.close()
        robot.close()
        viewer.close()


if __name__ == "__main__":
    main()
