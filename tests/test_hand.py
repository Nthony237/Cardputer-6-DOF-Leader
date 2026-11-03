"""Phase 4: webcam hand position -> clutch-relative arm offset, fused with IMU orientation.

None of this needs a camera or mediapipe: the landmarks are synthesised from a known hand pose.
"""
import math

import numpy as np
import pytest

from cardputer_teleop.controller import CartesianController
from cardputer_teleop.fake_sender import FakeCardputer
from cardputer_teleop.hand import (
    DEFAULT_AXES,
    HandFollower,
    HandObservation,
    HandSettings,
    OneEuroFilter,
    focal_length_px,
    hand_position,
    parse_axes,
)
from cardputer_teleop.hand_tracker import FakeHandTracker
from cardputer_teleop.leader import CardputerLeader, LeaderSettings
from cardputer_teleop.orientation import quat_from_rpy
from cardputer_teleop.protocol import FLAG_CLUTCH, FLAG_TILT_MOVE, LeaderPacket
from cardputer_teleop.transforms import make_transform, rpy_to_matrix, so3_log

W, H = 1280, 720
F = focal_length_px(W, 65.0)
rng = np.random.default_rng(1)
HAND = rng.uniform(-0.05, 0.05, (21, 3))  # a hand-sized cloud of 21 "joints", metres


def project(P_world, t, R=None, f=F):
    """What MediaPipe would report for a hand of shape P_world at camera position t."""
    P = P_world if R is None else P_world @ R.T
    X = P + t
    xy = np.stack([(W / 2 + f * X[:, 0] / X[:, 2]) / W, (H / 2 + f * X[:, 1] / X[:, 2]) / H], axis=1)
    return xy, P


# ---------------------------------------------------------------- landmarks -> position
@pytest.mark.parametrize("t", [(0.0, 0.0, 0.5), (0.12, -0.08, 0.35), (-0.2, 0.1, 0.9)])
def test_hand_position_recovers_the_true_position(t):
    xy, P = project(HAND, np.array(t))
    np.testing.assert_allclose(hand_position(xy, P, W, H, F), t, atol=1e-9)


def test_turning_the_hand_does_not_look_like_moving_it():
    """Foreshortening is in the 3D shape, so a tilted hand isn't mistaken for a farther one."""
    t = np.array([0.05, 0.02, 0.5])
    xy, P = project(HAND, t, R=rpy_to_matrix(np.radians([50, 30, 0])))
    np.testing.assert_allclose(hand_position(xy, P, W, H, F), t, atol=1e-9)


def test_wrong_field_of_view_only_scales_depth():
    """A guessed focal length scales toward/away distances but leaves sideways ones right."""
    t = np.array([0.1, -0.05, 0.5])
    xy, P = project(HAND, t)
    est = hand_position(xy, P, W, H, F * 1.2)
    assert abs(est[2] - 0.6) < 0.01  # 20 % longer focal length -> 20 % farther
    np.testing.assert_allclose(est[:2], t[:2], atol=0.005)


def test_nonsense_is_rejected():
    xy, P = project(HAND, np.array([0.0, 0.0, 0.5]))
    assert hand_position(xy, -P * 1000, W, H, F) is None


def test_noisy_landmarks_give_mm_level_error():
    t = np.array([0.0, 0.0, 0.5])
    xy, P = project(HAND, t)
    errs = []
    for _ in range(50):
        noisy = xy + rng.normal(0, 1.0 / W, xy.shape)  # 1 pixel of landmark noise
        errs.append(hand_position(noisy, P, W, H, F) - t)
    assert np.std(errs, axis=0).max() < 0.004


# ---------------------------------------------------------------- axes and filter
def test_default_axes():
    M = parse_axes(DEFAULT_AXES)
    np.testing.assert_allclose(M @ [0, 0, -1], [1, 0, 0])  # toward the camera = forward
    np.testing.assert_allclose(M @ [-1, 0, 0], [0, 1, 0])  # left in the mirrored picture = left
    np.testing.assert_allclose(M @ [0, -1, 0], [0, 0, 1])  # up in the picture = up


@pytest.mark.parametrize("bad", ["x,y", "x,x,z", "x,y,w"])
def test_bad_axes(bad):
    with pytest.raises(ValueError):
        parse_axes(bad)


def test_one_euro_removes_jitter_when_still_and_keeps_up_when_moving():
    f = OneEuroFilter()
    still = [f(np.array([0, 0, 0.5]) + rng.normal(0, 0.003, 3), k / 30) for k in range(90)]
    assert np.std(still[30:], axis=0).max() < 0.0016  # 3 mm raw -> about half
    f.reset()
    out = [f(np.array([0.3 * k / 30, 0, 0]), k / 30) for k in range(30)]  # 0.3 m/s sweep
    assert 0.29 - out[-1][0] < 0.02  # under 2 cm behind


# ---------------------------------------------------------------- follower
def obs(t, p):
    return HandObservation(t=t, p_cam=np.asarray(p, float))


def raw(**kw):  # filter so light it's basically off, so the numbers are easy to check
    return HandFollower(HandSettings(min_cutoff=1e4, **kw))


def test_follower_is_relative_to_the_grab():
    f = raw()
    np.testing.assert_allclose(f.update(obs(0.0, [0.1, 0.1, 0.6]), 0.0), 0)
    off = f.update(obs(0.1, [0.1, 0.1, 0.55]), 0.1)  # 5 cm toward the camera
    np.testing.assert_allclose(off, [0.05, 0, 0], atol=1e-4)
    off = f.update(obs(0.2, [0.07, 0.08, 0.55]), 0.2)  # then 3 cm to its left, 2 cm up
    np.testing.assert_allclose(off, [0.05, 0.03, 0.02], atol=1e-4)


def test_follower_gain():
    f = raw(gain=0.5)
    f.update(obs(0.0, [0, 0, 0.6]), 0.0)
    np.testing.assert_allclose(f.update(obs(0.1, [0, 0, 0.5]), 0.1), [0.05, 0, 0], atol=1e-4)


def test_lost_hand_freezes_then_resumes_without_a_jump():
    f = raw()
    f.update(obs(0.0, [0, 0, 0.6]), 0.0)
    f.update(obs(0.1, [0, 0, 0.5]), 0.1)  # +10 cm forward
    held = f.update(obs(0.1, [0, 0, 0.5]), 0.5).copy()  # no new frame for 0.4 s
    assert not f.tracking
    np.testing.assert_allclose(held, [0.1, 0, 0], atol=1e-4)
    # it comes back somewhere else: nothing moves on that frame...
    np.testing.assert_allclose(f.update(obs(0.6, [0.3, 0.0, 0.8]), 0.6), held, atol=1e-4)
    # ...and moves from there on
    np.testing.assert_allclose(f.update(obs(0.7, [0.3, 0.0, 0.75]), 0.7), [0.15, 0, 0], atol=1e-4)
    assert f.reacquired == 1


def test_jump_is_treated_as_a_glitch():
    f = raw()
    f.update(obs(0.0, [0, 0, 0.6]), 0.0)
    f.update(obs(0.033, [0, 0, 0.58]), 0.033)
    off = f.update(obs(0.066, [0.25, 0, 0.58]), 0.066).copy()  # 25 cm in one frame: not a real hand
    np.testing.assert_allclose(off, [0.02, 0, 0], atol=1e-4)
    assert f.reacquired == 1


def test_same_frame_twice_is_ignored():
    f = raw()
    f.update(obs(0.0, [0, 0, 0.6]), 0.0)
    a = f.update(obs(0.033, [0, 0, 0.55]), 0.033).copy()
    np.testing.assert_allclose(f.update(obs(0.033, [0, 0, 0.55]), 0.05), a)


# ---------------------------------------------------------------- fused with the Cardputer
DT = 0.02
T0 = make_transform(rpy_to_matrix([math.pi, 0.0, 0.0]), [0.3, 0.0, 0.25])


def pkt(yaw_deg=0.0, flags=FLAG_CLUTCH, seq=0):
    return LeaderPacket(seq=seq, t_ms=seq * 10, quat=quat_from_rpy(0, 0, math.radians(yaw_deg)),
                        accel=np.array([0, 0, 9.80665]), flags=flags)


def hand_leader():
    return CardputerLeader(LeaderSettings(smoothing_tau=0.0, hand=HandSettings(min_cutoff=1e4)))


def test_hand_moves_and_imu_turns():
    L = hand_leader()
    cmd = L.update(pkt(0), T0, DT, 0.1, hand=obs(0.0, [0, 0, 0.6]), now=0.0)
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)  # grabbing never jumps
    cmd = L.update(pkt(20), T0, DT, 0.1, hand=obs(0.1, [-0.04, 0, 0.55]), now=0.1)
    np.testing.assert_allclose(cmd.pose[:3, 3], T0[:3, 3] + [0.05, 0.04, 0.0], atol=1e-4)
    turned = math.degrees(np.linalg.norm(so3_log(cmd.pose[:3, :3] @ T0[:3, :3].T)))
    assert abs(turned - 20) < 1e-6
    assert "moved" in cmd.info


def test_letting_go_and_regrabbing_is_relative():
    L = hand_leader()
    L.update(pkt(0), T0, DT, 0.1, hand=obs(0.0, [0, 0, 0.6]), now=0.0)
    L.update(pkt(0), T0, DT, 0.1, hand=obs(0.1, [0, 0, 0.5]), now=0.1)
    L.update(pkt(0, flags=0), T0, DT, 0.1, hand=obs(0.2, [0, 0, 0.7]), now=0.2)  # let go, move back
    T1 = make_transform(T0[:3, :3], T0[:3, 3] + [0.1, 0, 0])  # arm got there meanwhile
    cmd = L.update(pkt(0), T1, DT, 0.1, hand=obs(0.3, [0, 0, 0.7]), now=0.3)
    np.testing.assert_allclose(cmd.pose, T1, atol=1e-9)


def test_hand_out_of_view_keeps_orientation_working():
    L = hand_leader()
    L.update(pkt(0), T0, DT, 0.1, hand=None, now=0.0)
    cmd = L.update(pkt(30), T0, DT, 0.1, hand=None, now=0.1)
    np.testing.assert_allclose(cmd.pose[:3, 3], T0[:3, 3])
    assert "not in view" in cmd.info
    assert math.degrees(np.linalg.norm(so3_log(cmd.pose[:3, :3] @ T0[:3, :3].T))) > 29


def test_tilt_still_works_with_the_camera_on():
    L = hand_leader()
    L.update(pkt(0, flags=FLAG_TILT_MOVE), T0, DT, 0.1, hand=obs(0.0, [0, 0, 0.6]), now=0.0)
    cmd = L.update(LeaderPacket(seq=1, t_ms=10, quat=quat_from_rpy(0, math.radians(20), 0),
                                flags=FLAG_TILT_MOVE), T0, DT, 0.1, hand=obs(0.1, [0, 0, 0.3]), now=0.1)
    assert cmd.mode == "tilt" and cmd.twist[0] > 0 and cmd.pose is None


def test_closed_loop_follows_the_hand(ik, q_home):
    """Move the hand 10 cm toward the screen: the sim gripper goes 10 cm forward, never faster
    than the speed limit, and keeps its angle."""
    ctrl = CartesianController(ik)
    L = CardputerLeader(LeaderSettings(hand=HandSettings()))
    q = q_home.copy()
    ctrl.reset_target(q)
    T_start = ik.chain.fk(q)
    prev = T_start[:3, 3]
    for k in range(200):  # 4 s at 50 Hz, camera at 25 Hz
        t = k * DT
        cam_t = math.floor(t * 25) / 25
        z = 0.6 - 0.1 * min(cam_t / 1.0, 1.0)
        ctrl.heartbeat.beat()
        cmd = L.update(pkt(0, seq=k), ik.chain.fk(q), DT, 0.1, hand=obs(cam_t, [0, 0, z]), now=t)
        ctrl.set_target(cmd.pose, q)
        q = ctrl.update(q, DT).q
        p = ik.chain.fk(q)[:3, 3]
        assert np.linalg.norm(p - prev) / DT <= ik.speed.linear * 1.05
        prev = p
    T_end = ik.chain.fk(q)
    np.testing.assert_allclose(T_end[:3, 3], T_start[:3, 3] + [0.1, 0, 0], atol=0.005)
    assert math.degrees(np.linalg.norm(so3_log(T_end[:3, :3] @ T_start[:3, :3].T))) < 1.0


# ---------------------------------------------------------------- fakes
def test_fake_camera_loses_the_hand_and_glitches():
    cam = FakeHandTracker(noise_m=0.0)
    t0 = cam._t0
    assert cam.observation(t0 + 1.0) is not None
    assert cam.observation(t0 + 5.8) is None  # dropout
    a, b = cam.observation(t0 + 7.9), cam.observation(t0 + 8.1)
    assert np.linalg.norm(a.p_cam - b.p_cam) > 0.15  # glitch frame


def test_fake_cardputer_hand_scenario_holds_the_clutch():
    dev = FakeCardputer(scenario="hand")
    assert dev.packet(5.0).clutch and not dev.packet(1.2).clutch


# ---------------------------------------------------------------- pinch to grab
from cardputer_teleop.hand import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, PinchDetector, pinch_ratio


def hand_with_gap(gap):
    P = np.zeros((21, 3))
    P[MIDDLE_MCP] = [0, -0.09, 0]  # 9 cm palm
    P[THUMB_TIP], P[INDEX_TIP] = [0.03, -0.1, 0], [0.03 + gap, -0.1, 0]
    return P


def test_pinch_ratio_ignores_how_the_hand_is_turned():
    P = hand_with_gap(0.018)
    R = rpy_to_matrix(np.radians([40, -20, 70]))
    assert pinch_ratio(P) == pytest.approx(0.2)
    assert pinch_ratio(P @ R.T) == pytest.approx(0.2)


def pinch_obs(t, ratio):
    return HandObservation(t=t, p_cam=np.array([0, 0, 0.5]), pinch=ratio)


def test_pinch_has_hysteresis_and_stops_when_the_hand_is_lost():
    d = PinchDetector()
    assert not d.update(pinch_obs(0.0, 0.35), 0.0)  # between the thresholds, not pinched yet
    assert d.update(pinch_obs(0.1, 0.2), 0.1)  # pinch
    assert d.update(pinch_obs(0.2, 0.4), 0.2)  # opening a little: still holding
    assert not d.update(pinch_obs(0.3, 0.6), 0.3)  # open: let go
    assert d.update(pinch_obs(0.4, 0.2), 0.4)
    assert not d.update(pinch_obs(0.4, 0.2), 0.8)  # no new frame for 0.4 s: hand lost = let go


def pinch_leader():
    return CardputerLeader(LeaderSettings(smoothing_tau=0.0, hand=HandSettings(min_cutoff=1e4), pinch_clutch=True))


def test_pinch_grabs_without_any_key():
    L = pinch_leader()
    assert L.update(pkt(0, flags=0), T0, DT, 0.1, hand=pinch_obs(0.0, 0.7), now=0.0).mode == "idle"
    cmd = L.update(pkt(0, flags=0), T0, DT, 0.1, hand=pinch_obs(0.1, 0.15), now=0.1)
    assert cmd.mode == "orient" and L.pinched
    np.testing.assert_allclose(cmd.pose, T0, atol=1e-9)  # grabbing never jumps
    o = HandObservation(t=0.2, p_cam=np.array([0, 0, 0.45]), pinch=0.15)
    cmd = L.update(pkt(0, flags=0), T0, DT, 0.1, hand=o, now=0.2)
    np.testing.assert_allclose(cmd.pose[:3, 3], T0[:3, 3] + [0.05, 0, 0], atol=1e-4)
    assert L.update(pkt(0, flags=0), T0, DT, 0.1, hand=pinch_obs(0.3, 0.8), now=0.3).mode == "idle"


def test_pinch_is_off_unless_asked_for():
    L = hand_leader()
    assert L.update(pkt(0, flags=0), T0, DT, 0.1, hand=pinch_obs(0.0, 0.1), now=0.0).mode == "idle"


def test_pinch_still_needs_the_cardputer_link():
    L = pinch_leader()
    assert L.update(None, T0, DT, 0.1, hand=pinch_obs(0.0, 0.1), now=0.0).mode == "idle"
    assert L.update(pkt(0, flags=0), T0, DT, 0.1, hand=pinch_obs(0.1, 0.1), now=0.1, enabled=False).mode == "idle"
