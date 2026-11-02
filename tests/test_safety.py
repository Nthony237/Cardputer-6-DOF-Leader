import numpy as np

from cardputer_teleop.safety import Heartbeat, SpeedLimits, Workspace


def test_workspace_contains_and_clamp(rng):
    ws = Workspace()
    assert ws.contains([0.3, 0.0, 0.2])
    assert not ws.contains([0.3, 0.0, -0.1])  # below the table
    assert not ws.contains([0.0, 0.0, 0.3])  # inside the base keep-out (also below x lower)
    for p in rng.uniform(-1, 1, size=(200, 3)):
        c = ws.clamp(p)
        assert ws.contains(c, tol=1e-9), (p, c)
        if ws.contains(p):
            np.testing.assert_allclose(c, p)


def test_clip_twist_independent_parts():
    sl = SpeedLimits(linear=0.1, angular=1.0)
    t = sl.clip_twist([1.0, 0, 0, 0, 0.5, 0])
    np.testing.assert_allclose(t, [0.1, 0, 0, 0, 0.5, 0])
    t = sl.clip_twist([0, 0, 0, 3.0, 4.0, 0])
    np.testing.assert_allclose(np.linalg.norm(t[3:]), 1.0)


def test_joint_limits_fallback():
    sl = SpeedLimits(joint_scale=0.5, joint_default=2.0)
    np.testing.assert_allclose(sl.joint_limits(np.array([4.0, np.inf])), [2.0, 1.0])


def test_heartbeat(clock):
    hb = Heartbeat(timeout=0.25, clock=clock)
    assert not hb.alive()  # never beaten -> not alive
    hb.beat()
    assert hb.alive()
    clock.advance(0.2)
    assert hb.alive()
    clock.advance(0.1)
    assert not hb.alive()
    hb.beat()
    assert hb.alive()
