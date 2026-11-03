"""The Cardputer -> laptop UDP packet. Must match firmware/cardputer_leader/include/packet.h.

One packet is 56 bytes, little-endian, sent about 100 times a second::

    magic "CPTL" | version u8 | flags u8 | yaw_reset_count u8 | battery_pct u8
    seq u32 | t_ms u32 | quat w,x,y,z f32 | gyro x,y,z f32 (rad/s) | accel x,y,z f32 (m/s^2)

``quat`` rotates device-frame vectors into a z-up world frame whose heading is whatever the
device pointed at when it booted or when the zero-yaw key was last pressed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"CPTL"
VERSION = 1
DEFAULT_PORT = 9870

FLAG_CLUTCH = 1 << 0
FLAG_GRIP_CLOSE = 1 << 1
FLAG_GRIP_OPEN = 1 << 2
FLAG_CALIBRATING = 1 << 3
FLAG_TILT_MOVE = 1 << 4  # Phase 3: tilt-as-joystick key held (older firmware never sets it)

_FORMAT = struct.Struct("<4sBBBBII4f3f3f")
PACKET_SIZE = _FORMAT.size
assert PACKET_SIZE == 56


class PacketError(ValueError):
    """Bytes that aren't a valid leader packet (wrong size, magic or version)."""


@dataclass
class LeaderPacket:
    seq: int
    t_ms: int
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # w x y z
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 9.80665]))
    flags: int = 0
    yaw_reset_count: int = 0
    battery_pct: int = 255  # 255 = unknown

    @property
    def clutch(self) -> bool:
        return bool(self.flags & FLAG_CLUTCH)

    @property
    def grip_close(self) -> bool:
        return bool(self.flags & FLAG_GRIP_CLOSE)

    @property
    def grip_open(self) -> bool:
        return bool(self.flags & FLAG_GRIP_OPEN)

    @property
    def calibrating(self) -> bool:
        return bool(self.flags & FLAG_CALIBRATING)

    @property
    def tilt_move(self) -> bool:
        return bool(self.flags & FLAG_TILT_MOVE)

    @property
    def gripper_direction(self) -> float:
        """-1 closing, +1 opening, 0 neither (same sign convention as the keyboard)."""
        return float(self.grip_open) - float(self.grip_close)

    def pack(self) -> bytes:
        return _FORMAT.pack(
            MAGIC, VERSION, self.flags & 0xFF, self.yaw_reset_count & 0xFF, self.battery_pct & 0xFF,
            self.seq & 0xFFFFFFFF, self.t_ms & 0xFFFFFFFF, *np.asarray(self.quat, dtype=float),
            *np.asarray(self.gyro, dtype=float), *np.asarray(self.accel, dtype=float),
        )

    @classmethod
    def unpack(cls, data: bytes) -> LeaderPacket:
        if len(data) != PACKET_SIZE:
            raise PacketError(f"expected {PACKET_SIZE} bytes, got {len(data)}")
        magic, version, flags, yaw_resets, battery, seq, t_ms, *floats = _FORMAT.unpack(data)
        if magic != MAGIC:
            raise PacketError(f"bad magic {magic!r}")
        if version != VERSION:
            raise PacketError(f"packet version {version}, this receiver understands {VERSION}")
        values = np.array(floats, dtype=float)
        if not np.all(np.isfinite(values)):
            raise PacketError("non-finite number in packet")
        quat = values[0:4]
        n = np.linalg.norm(quat)
        if n < 0.5:  # a real unit quaternion has norm 1; this is garbage
            raise PacketError("quaternion is not normalized")
        return cls(seq=seq, t_ms=t_ms, quat=quat / n, gyro=values[4:7], accel=values[7:10], flags=flags,
                   yaw_reset_count=yaw_resets, battery_pct=battery)
