"""Cartesian teleop. ``cardputer-teleop --mode sim`` runs on a laptop with no arms.

With ``--cardputer`` the Cardputer's packet stream becomes the heartbeat (the arm only moves
while packets arrive) and the Cardputer steers the gripper (Phase 3, see ``leader.py``):

* hold space / G0 and turn the Cardputer: the gripper turns the same way;
* hold x and tip it like a joystick: the gripper moves (``--translation tilt``, the default);
* or, with ``--translation push``, quick pushes while holding space nudge the gripper;
* c / v close / open the gripper, z sets "forward", b recalibrates the gyro.

``--fake push`` (or ``tilt`` / ``rock``) runs a pretend Cardputer in the background.
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from cardputer_teleop.config import load_config
from cardputer_teleop.controller import CartesianController
from cardputer_teleop.ik import DifferentialIK
from cardputer_teleop.keyboard import HELP, HeldKeys, KeyboardInput, gripper_from_keys, twist_from_keys
from cardputer_teleop.kinematics import SerialChain
from cardputer_teleop.leader import TRANSLATION_MODES, CardputerLeader, LeaderSettings
from cardputer_teleop.protocol import DEFAULT_PORT
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
    p.add_argument("--cardputer", action="store_true",
                   help="Listen for the Cardputer: its packets are the heartbeat, its keys drive the gripper")
    p.add_argument("--udp-port", type=int, default=DEFAULT_PORT, help="UDP port for --cardputer")
    p.add_argument("--fake", choices=("rock", "push", "tilt"),
                   help="Also run a fake Cardputer doing this routine (implies --cardputer)")
    p.add_argument("--translation", choices=TRANSLATION_MODES, default="tilt",
                   help="How the Cardputer moves the gripper: tilt (hold x, tip it), push (pushes while "
                        "holding space) or off (orientation only)")
    p.add_argument("--heading-offset", type=float, default=0.0,
                   help="Degrees between the Cardputer's forward (after z) and the robot's +x. "
                        "180 if you press z while facing the robot")
    p.add_argument("--rotation-scale", type=float, default=1.0,
                   help="Gripper rotation per hand rotation (0.5 = half as much, finer control)")
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
    receiver = None
    leader = None
    stop_fake = threading.Event()
    if args.fake:
        args.cardputer = True
    if args.cardputer:
        from cardputer_teleop.receiver import CardputerReceiver
        from cardputer_teleop.viz import CardputerModel

        receiver = CardputerReceiver(port=args.udp_port, heartbeat=controller.heartbeat).start()
        leader = CardputerLeader(LeaderSettings(heading_offset_deg=args.heading_offset,
                                                rotation_scale=args.rotation_scale,
                                                translation=args.translation))
        if args.fake:
            from cardputer_teleop import fake_sender

            threading.Thread(target=fake_sender.run, daemon=True, kwargs={
                "port": receiver.port, "stop": stop_fake, "verbose": False, "scenario": args.fake,
                "accel_noise": 0.05}).start()
            print(f"Fake Cardputer running the '{args.fake}' routine.")
        device_model = CardputerModel(viewer.server, "/cardputer", position=(0.0, -0.45, 0.35), scale=2.0)
        print(f"Listening for the Cardputer on UDP {receiver.port}; the arm holds until packets arrive.")
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

            # The local keyboard is always connected, so it beats the heartbeat itself. With
            # --cardputer the receiver beats once per packet instead, so a dropped Wi-Fi link
            # makes the controller hold.
            device = receiver.latest if receiver else None
            if receiver is None:
                controller.heartbeat.beat()
            else:
                device_model.update(device.quat if device else (1.0, 0.0, 0.0, 0.0), controller.heartbeat.alive())
            controller.held = viewer.hold.value

            q = robot.get_joint_positions()
            now = time.monotonic()
            lin = viewer.speed.value
            twist = twist_from_keys(held, now, lin, lin * ANGULAR_PER_LINEAR)
            drag = viewer.take_drag_target()
            if np.any(twist) or drag is not None:
                homing = False

            # Phase 3: the Cardputer's clutch / tilt key. The leader lets go whenever the link
            # is down or hold is on, and grabs afresh (from where the arm really is) afterwards.
            lead = None
            if leader is not None:
                was_steering = leader.mode != "idle"
                lead = leader.update(device, chain.fk(q), dt, lin,
                                     enabled=controller.heartbeat.alive() and not controller.held)
                if was_steering and lead.mode == "idle":
                    controller.reset_target(q)  # let go of the clutch = stop right here
                if lead.mode != "idle":
                    homing = False
                    twist[:] = 0.0  # one input at a time: the Cardputer wins while it's engaged
                    drag = None

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
                # Both go through set_target: workspace box + leash, then IK speed/joint limits.
                if lead is not None and lead.pose is not None:
                    controller.set_target(lead.pose, q)
                elif lead is not None and lead.twist is not None:
                    controller.apply_twist(lead.twist, dt, q)
                grip_dir = gripper_from_keys(held, now)
                if device is not None and controller.heartbeat.alive():
                    grip_dir = grip_dir or device.gripper_direction
                gripper = float(np.clip(gripper + grip_dir * GRIPPER_RATE * dt, 0.0, 1.0))
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
            if receiver is not None:
                st = receiver.stats
                status += f"  \nCardputer: {st.rate_hz:.0f} Hz, {st.loss_pct:.1f}% lost"
                if lead is not None:
                    mode = {"idle": "not steering", "orient": "turning (space)", "tilt": "tilt-moving (x)"}
                    status += f"  \nLeader: {mode[lead.mode]}" + (f", {lead.info}" if lead.info else "")
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
        stop_fake.set()
        if receiver:
            receiver.close()
        if keyboard:
            keyboard.close()
        robot.close()
        viewer.close()


if __name__ == "__main__":
    main()
