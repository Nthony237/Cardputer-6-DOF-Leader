import numpy as np

from cardputer_teleop.transforms import make_transform, pose_error, so3_exp

DT = 0.02


def test_reaches_poses_from_random_reachable_configs(chain, ik, q_home, rng):
    successes = 0
    for _ in range(30):
        # Targets near home so they are on the same IK branch and reachable without limit hits.
        q_goal = np.clip(q_home + rng.uniform(-0.4, 0.4, 6), ik.lower, ik.upper)
        T_goal = chain.fk(q_goal)
        q, ok = ik.solve(q_home, T_goal, dt=DT, max_steps=1000)
        successes += ok
        if ok:
            err = pose_error(chain.fk(q), T_goal)
            assert np.linalg.norm(err[:3]) < 1e-4
    assert successes >= 27


def test_respects_joint_velocity_limits(chain, ik, q_home):
    T_far = make_transform(chain.fk(q_home)[:3, :3], chain.fk(q_home)[:3, 3] + [0.0, 0.3, -0.1])
    q = q_home
    for _ in range(200):
        res = ik.step(q, T_far, DT)
        assert np.all(np.abs(res.dq) <= ik.qdot_max + 1e-9)
        assert np.all(np.abs(res.q - q) <= ik.qdot_max * DT + 1e-9)
        q = res.q


def test_respects_tcp_speed_limit(chain, ik, q_home):
    T0 = chain.fk(q_home)
    T_far = make_transform(T0[:3, :3], T0[:3, 3] + [0.2, 0.0, 0.0])
    q = q_home
    for _ in range(50):
        res = ik.step(q, T_far, DT)
        moved = np.linalg.norm(chain.fk(res.q)[:3, 3] - chain.fk(q)[:3, 3])
        # Linearization error allows a small excess over v_max * dt.
        assert moved <= ik.speed.linear * DT * 1.05 + 1e-6
        q = res.q


def test_never_leaves_joint_limits_even_for_unreachable_target(chain, ik, q_home):
    T_unreachable = make_transform(np.eye(3), [2.0, 0.0, 0.0])
    q = q_home
    for _ in range(1000):
        res = ik.step(q, T_unreachable, DT)
        assert chain.within_limits(res.q)
        assert np.all(np.isfinite(res.q))
        q = res.q


def test_locks_joint_at_limit(chain, ik, q_home):
    # Start joint 1 (base yaw) at its margined upper limit and ask for motion only it can make.
    q = q_home.copy()
    q[0] = ik.upper[0]
    p = chain.fk(q)[:3, 3]
    tangent = np.cross([0, 0, 1], p)  # +yaw direction at the TCP
    twist = np.concatenate([0.1 * tangent / np.linalg.norm(tangent), [0, 0, 0]])
    res = ik.step_twist(q, twist, DT)
    assert 0 in res.locked
    assert res.q[0] <= ik.upper[0] + 1e-12


def test_damping_grows_at_singularity(chain, ik):
    # Arm fully stretched straight up: shoulder/elbow singular.
    q_sing = np.array([0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])
    q_sing = np.clip(q_sing, chain.lower, chain.upper)
    lam, w = ik.damping(chain.jacobian(q_sing))
    assert w < ik.manipulability_threshold
    assert lam > ik.damping_min
    # A twist along the singular direction gives bounded joint speeds.
    res = ik.step_twist(q_sing, np.array([0, 0, 0.15, 0, 0, 0]), DT)
    assert np.all(np.isfinite(res.dq))
    assert np.all(np.abs(res.dq) <= ik.qdot_max + 1e-9)


def test_twist_tracking_direction(chain, ik, q_home):
    for axis in range(3):
        v = np.zeros(6)
        v[axis] = 0.05
        res = ik.step_twist(q_home, v, DT)
        dp = chain.fk(res.q)[:3, 3] - chain.fk(q_home)[:3, 3]
        assert dp[axis] > 0.9 * 0.05 * DT
        assert np.linalg.norm(np.delete(dp, axis)) < 0.1 * 0.05 * DT


def test_orientation_tracking(chain, ik, q_home):
    T0 = chain.fk(q_home)
    T_goal = make_transform(so3_exp([0, 0, 0.3]) @ T0[:3, :3], T0[:3, 3])
    q, ok = ik.solve(q_home, T_goal, dt=DT)
    assert ok
    np.testing.assert_allclose(chain.fk(q)[:3, :3], T_goal[:3, :3], atol=1e-3)


def test_position_only_mode(chain, q_home):
    from cardputer_teleop.ik import DifferentialIK

    ik_p = DifferentialIK(chain, position_only=True)
    T0 = chain.fk(q_home)
    T_goal = make_transform(so3_exp([1.0, 0, 0]), T0[:3, 3] + [0.05, 0.05, 0])
    q, ok = ik_p.solve(q_home, T_goal, dt=DT)
    assert ok
    np.testing.assert_allclose(chain.fk(q)[:3, 3], T_goal[:3, 3], atol=1e-4)
