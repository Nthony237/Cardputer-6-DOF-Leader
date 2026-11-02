import re
from pathlib import Path

import numpy as np
import pytest

from cardputer_teleop import protocol
from cardputer_teleop.protocol import PACKET_SIZE, LeaderPacket, PacketError

PACKET_H = Path(__file__).parents[1] / "firmware" / "cardputer_leader" / "include" / "packet.h"


def test_roundtrip():
    pkt = LeaderPacket(seq=123456, t_ms=99, quat=np.array([0.5, 0.5, 0.5, 0.5]), gyro=np.array([0.1, -0.2, 0.3]),
                       accel=np.array([0.0, 1.0, 9.7]), flags=protocol.FLAG_CLUTCH | protocol.FLAG_GRIP_OPEN,
                       yaw_reset_count=3, battery_pct=55)
    data = pkt.pack()
    assert len(data) == PACKET_SIZE == 56
    out = LeaderPacket.unpack(data)
    assert out.seq == 123456 and out.t_ms == 99 and out.yaw_reset_count == 3 and out.battery_pct == 55
    np.testing.assert_allclose(out.quat, pkt.quat, atol=1e-6)
    np.testing.assert_allclose(out.gyro, pkt.gyro, atol=1e-6)
    assert out.clutch and out.grip_open and not out.grip_close and not out.calibrating
    assert out.gripper_direction == 1.0


@pytest.mark.parametrize("data", [b"", b"x" * 56, LeaderPacket(seq=1, t_ms=0).pack()[:-1]])
def test_rejects_garbage(data):
    with pytest.raises(PacketError):
        LeaderPacket.unpack(data)


def test_rejects_wrong_version_and_bad_quat():
    good = bytearray(LeaderPacket(seq=1, t_ms=0).pack())
    wrong_version = bytearray(good)
    wrong_version[4] = 99
    with pytest.raises(PacketError):
        LeaderPacket.unpack(bytes(wrong_version))
    with pytest.raises(PacketError):
        LeaderPacket.unpack(LeaderPacket(seq=1, t_ms=0, quat=np.zeros(4)).pack())
    with pytest.raises(PacketError):
        LeaderPacket.unpack(LeaderPacket(seq=1, t_ms=0, gyro=np.array([np.nan, 0, 0])).pack())


def test_matches_firmware_header():
    """protocol.py and the firmware's packet.h must describe the same bytes."""
    text = PACKET_H.read_text()
    defines = dict(re.findall(r"#define (\w+) (.+)", text))
    assert defines["PACKET_MAGIC"].strip('"').encode() == protocol.MAGIC
    assert int(defines["PACKET_VERSION"]) == protocol.VERSION
    for name in ("FLAG_CLUTCH", "FLAG_GRIP_CLOSE", "FLAG_GRIP_OPEN", "FLAG_CALIBRATING"):
        bit = int(re.search(r"1 << (\d+)", defines[name]).group(1))
        assert getattr(protocol, name) == 1 << bit, name
    assert f"sizeof(LeaderPacket) == {PACKET_SIZE}" in text
    fields = re.findall(r"^\s+(\w+) (\w+)(?:\[(\d)\])?;", text, re.MULTILINE)
    assert [f[1] for f in fields] == ["magic", "version", "flags", "yaw_reset_count", "battery_pct", "seq", "t_ms",
                                      "quat", "gyro", "accel"]
