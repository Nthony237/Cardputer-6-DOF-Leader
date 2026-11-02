import socket
import threading
import time

import numpy as np

from cardputer_teleop import fake_sender
from cardputer_teleop.protocol import LeaderPacket
from cardputer_teleop.receiver import CardputerReceiver
from cardputer_teleop.safety import Heartbeat


def pkt(seq):
    return LeaderPacket(seq=seq, t_ms=seq * 10).pack()


def test_counts_loss_and_drops_late_packets(clock):
    rx = CardputerReceiver(heartbeat=Heartbeat(timeout=0.25, clock=clock), clock=clock)
    assert rx.handle_datagram(pkt(10))
    assert rx.handle_datagram(pkt(13))  # 11 and 12 lost
    assert not rx.handle_datagram(pkt(12))  # arrives late: ignored
    assert not rx.handle_datagram(pkt(13))  # duplicate
    assert not rx.handle_datagram(b"hello")
    assert rx.latest.seq == 13
    s = rx.stats
    assert (s.received, s.lost, s.out_of_order, s.bad) == (2, 2, 1, 1)
    assert abs(s.loss_pct - 50.0) < 1e-9


def test_device_reboot_and_wraparound(clock):
    rx = CardputerReceiver(heartbeat=Heartbeat(clock=clock), clock=clock)
    rx.handle_datagram(pkt(50_000))
    assert rx.handle_datagram(pkt(0))  # seq restarted: the Cardputer rebooted
    assert rx.stats.restarts == 1 and rx.latest.seq == 0
    rx2 = CardputerReceiver(heartbeat=Heartbeat(clock=clock), clock=clock)
    rx2.handle_datagram(pkt(0xFFFFFFFF))
    assert rx2.handle_datagram(pkt(0)) and rx2.stats.lost == 0  # counter wrapped, nothing lost


def test_heartbeat_follows_packets(clock):
    hb = Heartbeat(timeout=0.25, clock=clock)
    rx = CardputerReceiver(heartbeat=hb, clock=clock)
    assert not hb.alive()  # nothing yet: the arm must hold
    rx.handle_datagram(pkt(1))
    assert hb.alive()
    clock.advance(0.3)
    assert not hb.alive()  # link went quiet
    rx.handle_datagram(pkt(1))  # a duplicate is not proof of life
    assert not hb.alive()
    rx.handle_datagram(pkt(2))
    assert hb.alive()


def test_over_udp_with_fake_sender():
    with CardputerReceiver(port=0) as rx:
        sent = fake_sender.run(port=rx.port, hz=200, duration=0.5, verbose=False)
        time.sleep(0.05)
        assert rx.stats.received >= 0.9 * sent > 50
        assert rx.heartbeat.alive()
        assert rx.stats.sender.startswith("127.0.0.1")
        np.testing.assert_allclose(np.linalg.norm(rx.latest.quat), 1.0, atol=1e-6)


def test_stall_makes_heartbeat_die():
    with CardputerReceiver(port=0, heartbeat=Heartbeat(timeout=0.15)) as rx:
        stop = threading.Event()
        t = threading.Thread(target=fake_sender.run, kwargs={"port": rx.port, "stall_after": 0.3, "stall_for": 10,
                                                                 "stop": stop, "verbose": False})
        t.start()
        try:
            time.sleep(0.25)
            assert rx.heartbeat.alive()
            time.sleep(0.4)
            assert not rx.heartbeat.alive()
        finally:
            stop.set()
            t.join()


def test_ignores_other_traffic():
    with CardputerReceiver(port=0) as rx:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"not a leader packet", ("127.0.0.1", rx.port))
        time.sleep(0.2)
        assert rx.stats.bad == 1 and rx.latest is None and not rx.heartbeat.alive()
