"""UDP receiver for the Cardputer: keeps the newest packet, counts lost ones, beats the heartbeat.

Runs a small background thread that waits for packets so the control loop never blocks on the
network. The control loop just reads ``receiver.latest`` whenever it wants.

Why sequence numbers: UDP is "fire and forget". Packets can be lost or (rarely) arrive out of
order, and nobody tells you. Each packet carries a counter that goes up by one, so a jump from
41 to 44 means 42 and 43 were lost, and a packet older than the newest one we already have is
thrown away so the arm never steps back in time.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

from cardputer_teleop.protocol import DEFAULT_PORT, LeaderPacket, PacketError
from cardputer_teleop.safety import Heartbeat

# An older seq whose device clock is this much earlier means the Cardputer rebooted
# (a merely late packet is only a few ms old).
RESTART_GAP_MS = 1000


@dataclass
class ReceiverStats:
    received: int = 0  # accepted packets
    lost: int = 0  # gaps in the sequence numbers
    out_of_order: int = 0  # late packets we dropped
    bad: int = 0  # not our packet format
    restarts: int = 0  # device reboots detected
    rate_hz: float = 0.0
    sender: str | None = None

    @property
    def loss_pct(self) -> float:
        total = self.received + self.lost
        return 100.0 * self.lost / total if total else 0.0


class CardputerReceiver:
    def __init__(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0", heartbeat: Heartbeat | None = None,
                 clock=time.monotonic):
        self.port = port
        self.host = host
        self.heartbeat = heartbeat if heartbeat is not None else Heartbeat()
        self.stats = ReceiverStats()
        self.latest: LeaderPacket | None = None
        self.latest_time: float | None = None
        self._clock = clock
        self._arrivals: deque[float] = deque()
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- packet logic (no sockets, so tests can call it directly) ----
    def handle_datagram(self, data: bytes, sender: str | None = None) -> bool:
        """Accept or reject one datagram. Returns True if it became ``latest``."""
        try:
            pkt = LeaderPacket.unpack(data)
        except PacketError:
            with self._lock:
                self.stats.bad += 1
            return False
        now = self._clock()
        with self._lock:
            if self.latest is not None:
                diff = (pkt.seq - self.latest.seq) & 0xFFFFFFFF
                if diff == 0:
                    return False  # duplicate
                if diff >= 1 << 31:  # older than what we have
                    if pkt.t_ms + RESTART_GAP_MS < self.latest.t_ms:
                        self.stats.restarts += 1  # device rebooted: start counting afresh
                    else:
                        self.stats.out_of_order += 1
                        return False
                else:
                    self.stats.lost += diff - 1
            self.latest = pkt
            self.latest_time = now
            self.stats.received += 1
            self.stats.sender = sender
            self._arrivals.append(now)
            while self._arrivals and now - self._arrivals[0] > 1.0:
                self._arrivals.popleft()
            self.stats.rate_hz = float(len(self._arrivals))
        self.heartbeat.beat()  # the link is alive: the controller may move
        return True

    def age(self) -> float:
        """Seconds since the last accepted packet (inf if none yet)."""
        return float("inf") if self.latest_time is None else self._clock() - self.latest_time

    # ---- socket thread ----
    def start(self) -> CardputerReceiver:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.1)  # wake up regularly to notice close()
        self.port = self._sock.getsockname()[1]  # in case port 0 picked a free one
        self._thread = threading.Thread(target=self._run, name="cardputer-rx", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(1024)
            except TimeoutError:
                continue
            except OSError:
                break
            self.handle_datagram(data, f"{addr[0]}:{addr[1]}")

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
