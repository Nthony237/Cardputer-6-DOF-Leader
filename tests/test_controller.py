import numpy as np

from cardputer_teleop.controller import CartesianController
from cardputer_teleop.safety import Heartbeat, Workspace
from cardputer_teleop.transforms import make_transform, so3_log

DT = 0.02


def make(ik, clock, **kw):
    hb = Heartbeat(timeout=0.25, clock=clock)
    hb.beat()
    return CartesianController(ik, heartbeat=hb, **kw)


def test_target_moves_with_twist(ik, q_home, clock):
    c = make(ik, clock)
    c.reset_target(q_home)
    p0 = c.T_target[:3, 3].copy()
    c.apply_twist(np.array([0.1, 0, 0, 0, 0, 0]), DT, q_home)
    np.testing.assert_allclose(c.T_target[:3, 3] - p0, [0.1 * DT, 0, 0], atol=1e-12)


def test_target_is_leashed(ik, q_home, clock):
    c = make(ik, clock, max_lead_linear=0.05)
    c.reset_target(q_home)
    T_ee = ik.chain.fk(q_home)
    for _ in range(100):  # keep pushing without letting the arm move
        c.apply_twist(np.array([0.15, 0, 0, 0, 0, 0]), DT, q_home)
    assert np.linalg.norm(c.T_target[:3, 3] - T_ee[:3, 3]) <= 0.05 + 1e-9


def test_angular_leash(ik, q_home, clock):
    c = make(ik, clock, max_lead_angular=0.5)
    c.reset_target(q_home)
    for _ in range(200):
        c.apply_twist(np.array([0, 0, 0, 0, 0, 1.0]), DT, q_home)
    R_ee = ik.chain.fk(q_home)[:3, :3]
    assert np.linalg.norm(so3_log(c.T_target[:3, :3] @ R_ee.T)) <= 0.5 + 1e-9


def test_target_clamped_to_workspace(ik, q_home, clock):
    ws = Workspace()
    c = make(ik, clock, workspace=ws, max_lead_linear=10.0)
    c.set_target(make_transform(np.eye(3), [0.3, 0.0, -0.5]), q_home)
    assert ws.contains(c.T_target[:3, 3])


def test_closed_loop_tracks_moving_target(ik, q_home, clock):
    c = make(ik, clock)
    q = q_home
    c.reset_target(q)
    for _ in range(100):
        c.apply_twist(np.array([0, 0.05, 0, 0, 0, 0]), DT, q)
        q = c.update(q, DT).q
    for _ in range(100):  # settle
        q = c.update(q, DT).q
    np.testing.assert_allclose(ik.chain.fk(q)[:3, 3], c.T_target[:3, 3], atol=1e-4)
    assert ik.chain.fk(q)[1, 3] > 0.09  # moved ~0.1 m in +y


def test_holds_when_heartbeat_lost(ik, q_home, clock):
    c = make(ik, clock)
    c.reset_target(q_home)
    c.apply_twist(np.array([0.1, 0, 0, 0, 0, 0]), DT, q_home)
    clock.advance(1.0)
    assert c.update(q_home, DT) is None
    # Target snapped back to the current pose, so resuming doesn't jump.
    np.testing.assert_allclose(c.T_target, ik.chain.fk(q_home))


def test_hold_flag(ik, q_home, clock):
    c = make(ik, clock)
    c.reset_target(q_home)
    c.held = True
    assert c.update(q_home, DT) is None


def test_tool_vs_base_rotation(ik, q_home, clock):
    c = make(ik, clock)
    w = np.array([0, 0, 0, 0, 0, 0.5])
    c.reset_target(q_home)
    c.apply_twist(w, DT, q_home, rotation_frame="base")
    R_base = c.T_target[:3, :3].copy()
    c.reset_target(q_home)
    c.apply_twist(w, DT, q_home, rotation_frame="tool")
    assert not np.allclose(R_base, c.T_target[:3, :3])
