"""Start the viewers headless with the fake Cardputer for a moment and quit cleanly."""
import socket

import pytest

pytest.importorskip("viser")


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_imu_view_with_fake_device():
    from cardputer_teleop.imu_view import main

    main(["--fake", "--duration", "1.0", "--port", str(free_port()), "--udp-port", str(free_port(socket.SOCK_DGRAM))])


def test_teleop_with_cardputer_heartbeat():
    pytest.importorskip("yourdfpy")
    import threading

    from cardputer_teleop import fake_sender
    from cardputer_teleop.teleop import main

    udp = free_port(socket.SOCK_DGRAM)
    stop = threading.Event()
    t = threading.Thread(target=fake_sender.run, kwargs={"port": udp, "stop": stop, "verbose": False})
    t.start()
    try:
        main(["--keys", "none", "--cardputer", "--udp-port", str(udp), "--duration", "1.0",
              "--port", str(free_port())])
    finally:
        stop.set()
        t.join()
