"""Live view of the Cardputer's orientation in the browser (Phase 2).

    cardputer-imu-view            # listen for the real Cardputer on UDP 9870
    cardputer-imu-view --fake     # also start a fake Cardputer, no hardware needed

Open http://localhost:8080. The block turns as you turn the device and goes grey if packets
stop for longer than the heartbeat timeout (the same watchdog that makes the arm hold).
"""
from __future__ import annotations

import argparse
import math
import threading
import time

import viser

from cardputer_teleop import fake_sender
from cardputer_teleop.orientation import quat_to_rpy
from cardputer_teleop.protocol import DEFAULT_PORT
from cardputer_teleop.receiver import CardputerReceiver
from cardputer_teleop.safety import Heartbeat
from cardputer_teleop.viz import CardputerModel


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--udp-port", type=int, default=DEFAULT_PORT, help="UDP port the Cardputer sends to")
    p.add_argument("--port", type=int, default=8080, help="viser (browser) port")
    p.add_argument("--fake", action="store_true", help="Run a fake Cardputer in the background")
    p.add_argument("--timeout", type=float, default=0.25, help="Heartbeat timeout in seconds")
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N s (0 = until Ctrl+C)")
    return p.parse_args(argv)


def status_text(rx: CardputerReceiver, alive: bool) -> str:
    s = rx.stats
    if rx.latest is None:
        return f"**Waiting for the Cardputer** on UDP port {rx.port}...  \n" \
               "Check it shows this laptop's IP and that both are on the same Wi-Fi."
    pkt = rx.latest
    roll, pitch, yaw = (math.degrees(a) for a in quat_to_rpy(pkt.quat))
    link = "**LINK OK**" if alive else f"**LINK LOST** (no packet for {rx.age():.1f} s, arm would hold)"
    keys = [name for name, on in (("CLUTCH", pkt.clutch), ("CLOSE", pkt.grip_close), ("OPEN", pkt.grip_open))
            if on]
    battery = "?" if pkt.battery_pct > 100 else f"{pkt.battery_pct}%"
    lines = [
        f"{link} from {s.sender}",
        (f"{s.rate_hz:.0f} Hz, {s.received} received, {s.lost} lost ({s.loss_pct:.1f}%), "
        f"{s.out_of_order} late, {s.bad} bad, {s.restarts} reboots"),
        f"roll {roll:6.1f}°  pitch {pitch:6.1f}°  yaw {yaw:6.1f}°  (yaw zeroed {pkt.yaw_reset_count}x)",
        f"keys: {' '.join(keys) or '-'}   battery {battery}",
    ]
    if pkt.calibrating:
        lines.append("**Calibrating: hold the Cardputer still**")
    return "  \n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    heartbeat = Heartbeat(timeout=args.timeout)
    rx = CardputerReceiver(port=args.udp_port, heartbeat=heartbeat).start()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (0.35, -0.35, 0.25)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)
    server.scene.add_grid("/grid", width=0.5, height=0.5, cell_size=0.05)
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)
    model = CardputerModel(server, "/cardputer", position=(0.0, 0.0, 0.08), scale=2.0)
    status = server.gui.add_markdown("starting...")

    stop = threading.Event()
    if args.fake:
        threading.Thread(target=fake_sender.run, kwargs={"port": rx.port, "stop": stop, "verbose": False},
                         daemon=True).start()
    print(f"Listening for the Cardputer on UDP {rx.port}. Open http://localhost:{args.port}")

    t_start = time.monotonic()
    try:
        while not (args.duration and time.monotonic() - t_start > args.duration):
            alive = heartbeat.alive()
            if rx.latest is not None:
                model.update(rx.latest.quat, alive)
            status.content = status_text(rx, alive)
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        rx.close()
        server.stop()


if __name__ == "__main__":
    main()
