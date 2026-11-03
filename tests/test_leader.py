"""Phase 3: Cardputer packets -> arm targets."""
import math

import numpy as np
import pytest

from cardputer_teleop.controller import CartesianController
from cardputer_teleop.fake_sender import FakeCardputer
from cardputer_teleop.leader import CardputerLeader, LeaderSettings, deadband_ramp
from cardputer_teleop.orientation import quat_from_rpy
from cardputer_teleop.protocol import FLAG_CALIBRATING, FLAG_CLUTCH, FLAG_TILT_MOVE, LeaderPacket
from cardputer_teleop.transforms import make_transform, matrix_to_wxyz, rpy_to_matrix, so3_log, wxyz_to_matrix

DT = 0.01
G = 9.80665


def pkt(rpy_deg=(0, 0, 0), flags=FLAG_CLUTCH, accel_world=(0, 0, 0), yaw_resets=0, seq=0):
    q = quat_from_rpy(*np.radians(rpy_deg))
    R = wxyz_to_matrix(q)
    accel = R.T @ (np.asarray(accel_world, float) + [0, 0, G])
    return LeaderPacket(seq=seq, t_ms=seq * 10, quat=q, accel=accel, flags=flags, yaw_reset_count=yaw_resets)


T0 = make_transform(rpy_to_matrix([math.pi, 0.0, 0.0]), [0.3, 0.0, 0.25])  # gripper pointing down


def rot_err_deg(Ra, Rb):
    return math.degrees(np.linalg.norm(so3_log(Ra @ Rb.T)))


def no_smoothing(**kw):
    return CardputerLeader(LeaderSettings(smoothing_tau=0.0, **kw))


def test_idle_without_clutch():
    L = no_smoothing()
    assert L.update(pkt(flags=0), T0, DT, 0.1).mode == "idle"
    assert L.update(None, T0, DT, 0.1).mode == "idle"


def test_grab_does_not_jump():
    """Whatever angle the Cardputer is at when you grab, the target starts at the arm's pose."""
    L = no_smoothing()
    cmd = L.update(pkt((25, -10, 70)), T0, DT, 0.1)
    assert cmd.mode == "orient"
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)


@pytest.mark.parametrize("delta", [(0, 0, 30), (20, 0, 0), (0, -25, 0)])
def test_gripper_turns_like_the_hand(delta):
    L = no_smoothing()
    R_d0 = rpy_to_matrix(np.radians([10.0, 5.0, 40.0]))
    R_d1 = R_d0 @ rpy_to_matrix(np.radians(delta))  # a turn about the hand's own axes
    L.update(LeaderPacket(seq=0, t_ms=0, quat=matrix_to_wxyz(R_d0), flags=FLAG_CLUTCH), T0, DT, 0.1)
    cmd = L.update(LeaderPacket(seq=1, t_ms=10, quat=matrix_to_wxyz(R_d1), flags=FLAG_CLUTCH), T0, DT, 0.1)
    expected = (R_d1 @ R_d0.T) @ T0[:3, :3]  # the same world-frame turn, applied to the gripper
    assert rot_err_deg(cmd.pose[:3, :3], expected) < 1e-6
    np.testing.assert_allclose(cmd.pose[:3, 3], T0[:3, 3])  # orientation only: position untouched


def test_heading_offset_mirrors_axes():
    """Facing the robot (offset 180): tipping your right side down tips the gripper toward *your* right."""
    L = no_smoothing(heading_offset_deg=180)
    L.update(pkt((0, 0, 0)), T0, DT, 0.1)
    cmd = L.update(pkt((20, 0, 0)), T0, DT, 0.1)
    w = so3_log(cmd.pose[:3, :3] @ T0[:3, :3].T)
    np.testing.assert_allclose(w, [-math.radians(20), 0, 0], atol=1e-9)


def test_rotation_scale():
    L = no_smoothing(rotation_scale=0.5)
    L.update(pkt((0, 0, 0)), T0, DT, 0.1)
    cmd = L.update(pkt((0, 0, 40)), T0, DT, 0.1)
    assert rot_err_deg(cmd.pose[:3, :3], T0[:3, :3]) == pytest.approx(20, abs=1e-6)


def test_smoothing_eases_in():
    L = CardputerLeader(LeaderSettings(smoothing_tau=0.06))
    L.update(pkt((0, 0, 0)), T0, DT, 0.1)
    first = rot_err_deg(L.update(pkt((0, 0, 30)), T0, DT, 0.1).pose[:3, :3], T0[:3, :3])
    assert 0 < first < 10
    for _ in range(100):
        cmd = L.update(pkt((0, 0, 30)), T0, DT, 0.1)
    assert rot_err_deg(cmd.pose[:3, :3], T0[:3, :3]) == pytest.approx(30, abs=0.1)


def test_release_and_regrab_is_relative():
    """Let go, turn your wrist back, grab again: the arm doesn't move (like lifting a mouse)."""
    L = no_smoothing()
    L.update(pkt((0, 0, 0)), T0, DT, 0.1)
    T1 = L.update(pkt((0, 0, 30)), T0, DT, 0.1).pose
    L.update(pkt((0, 0, 0), flags=0), T1, DT, 0.1)
    cmd = L.update(pkt((0, 0, 0)), T1, DT, 0.1)  # hand back at 0 deg, but the arm stays at 30
    np.testing.assert_allclose(cmd.pose, T1, atol=1e-9)
    assert L.grabs == 2


def test_calibrating_is_ignored():
    L = no_smoothing()
    cmd = L.update(pkt((0, 0, 50), flags=FLAG_CLUTCH | FLAG_CALIBRATING), T0, DT, 0.1)
    assert cmd.mode == "idle" and "calibrating" in cmd.info
    cmd = L.update(pkt((0, 0, 50)), T0, DT, 0.1)  # calibration done: fresh grab, no jump
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)


def test_zero_yaw_mid_grab_regrabs():
    """Pressing z makes the quaternion jump in yaw. That must not spin the gripper."""
    L = no_smoothing()
    L.update(pkt((0, 0, 40)), T0, DT, 0.1)
    cmd = L.update(pkt((0, 0, 0), yaw_resets=1), T0, DT, 0.1)
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)


def test_disabled_lets_go():
    L = no_smoothing()
    L.update(pkt((0, 0, 0)), T0, DT, 0.1)
    assert L.update(pkt((0, 0, 30)), T0, DT, 0.1, enabled=False).mode == "idle"
    cmd = L.update(pkt((0, 0, 30)), T0, DT, 0.1)
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)


# ---- tilt as joystick ----

def tilt_twist(rpy_deg, **kw):
    L = no_smoothing(**kw)
    L.update(pkt((0, 0, 0), flags=FLAG_TILT_MOVE), T0, DT, 0.1)
    return L.update(pkt(rpy_deg, flags=FLAG_TILT_MOVE), T0, DT, 0.1)


@pytest.mark.parametrize("rpy,axis,sign", [((0, 30, 0), 0, +1),  # front down -> forward
                                            ((0, -30, 0), 0, -1),
                                            ((30, 0, 0), 1, -1),  # right side down -> right (-y)
                                            ((0, 0, 30), 2, +1)])  # twist CCW -> up
def test_tilt_directions(rpy, axis, sign):
    cmd = tilt_twist(rpy)
    assert cmd.mode == "tilt" and cmd.pose is None
    expected = np.zeros(6)
    expected[axis] = sign * 0.1  # 30 deg = full speed
    np.testing.assert_allclose(cmd.twist, expected, atol=1e-9)


def test_tilt_deadband():
    np.testing.assert_allclose(tilt_twist((4, -4, 4)).twist, 0)
    assert deadband_ramp(18, 6, 30) == pytest.approx(0.5)
    assert deadband_ramp(-90, 6, 30) == -1


def test_tilt_key_needs_tilt_mode():
    cmd = tilt_twist((0, 30, 0), translation="off")
    assert cmd.mode == "idle"  # only the tilt key held, and tilt translation off


# ---- push to nudge ----

def run_push(seconds, noise=0.0, start=9.0):  # 9 s = routine start of the 2nd cycle
    """Drive the leader with the fake 'push' routine; returns the offset over time."""
    dev = FakeCardputer(scenario="push", accel_noise=noise, seed=3)
    L = CardputerLeader(LeaderSettings(translation="push"))
    out = {}
    for k in range(int(seconds / DT)):
        t = start + k * DT
        L.update(dev.packet(t), T0, DT, 0.15)
        out[round(t - start, 2)] = L.push.offset.copy()
    return out, L


def test_push_moves_about_the_hand_distance():
    out, L = run_push(7.0, noise=0.05)
    np.testing.assert_allclose(out[1.5], [0.10, 0, 0], atol=0.01)  # pushed 10 cm forward
    np.testing.assert_allclose(out[3.0], [0.10, 0.10, 0], atol=0.01)  # then 10 cm left
    assert out[4.5][2] > 0.05  # then up
    assert L.push.pushes == 4


def test_push_ignores_stillness_and_small_tilt_errors():
    """No pushes: noise and a 2 deg filter error must not make the gripper creep."""
    L = CardputerLeader(LeaderSettings(translation="push"))
    rng = np.random.default_rng(0)
    tilt = quat_from_rpy(math.radians(2), 0, 0)
    for k in range(1000):
        p = LeaderPacket(seq=k, t_ms=k * 10, quat=tilt, accel=np.array([0, 0, G]) + rng.normal(0, 0.05, 3),
                         flags=FLAG_CLUTCH)
        L.update(p, T0, DT, 0.15)
    assert np.linalg.norm(L.push.offset) < 0.005
    assert L.push.pushes == 0


def test_push_offset_is_capped():
    L = CardputerLeader(LeaderSettings(translation="push", push_max_offset=0.05, push_max_time=10))
    for k in range(200):
        L.update(pkt(accel_world=(3.0, 0, 0), seq=k), T0, DT, 0.15)
    assert np.linalg.norm(L.push.offset) == pytest.approx(0.05)


# ---- end to end through the controller and IK ----

def test_closed_loop_orientation_tracking(ik, q_home):
    """Turn the Cardputer 30 deg: the simulated arm's gripper follows within a couple of seconds,
    never faster than the IK's angular speed limit."""
    ctrl = CartesianController(ik)
    L = CardputerLeader()
    q = q_home.copy()
    ctrl.reset_target(q)
    R_start = ik.chain.fk(q)[:3, :3]
    prev = R_start
    for k in range(300):
        yaw = min(k * DT / 0.5, 1.0) * 30  # turn the hand over half a second
        ctrl.heartbeat.beat()
        cmd = L.update(pkt((0, 0, yaw), seq=k), ik.chain.fk(q), DT, 0.1)
        ctrl.set_target(cmd.pose, q)
        res = ctrl.update(q, DT)
        q = res.q
        R = ik.chain.fk(q)[:3, :3]
        assert rot_err_deg(R, prev) / DT <= math.degrees(ik.speed.angular) * 1.05
        prev = R
    expected = rpy_to_matrix([0, 0, math.radians(30)]) @ R_start
    assert rot_err_deg(ik.chain.fk(q)[:3, :3], expected) < 1.0


def test_fake_tilt_scenario_flags():
    dev = FakeCardputer(scenario="tilt")
    assert dev.packet(10.6).tilt_move and not dev.packet(10.6).clutch
    assert not dev.packet(16.5).tilt_move
    assert dev.packet(0.5).calibrating
