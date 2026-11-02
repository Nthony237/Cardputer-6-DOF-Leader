import numpy as np

from cardputer_teleop.keyboard import HeldKeys, gripper_from_keys, twist_from_keys


def test_tap_then_release_by_timeout():
    h = HeldKeys(first_hold=0.3, repeat_hold=0.1)
    h.press("w", 0.0)
    assert h.is_held("w", 0.29)
    assert not h.is_held("w", 0.31)


def test_autorepeat_extends_hold():
    h = HeldKeys(first_hold=0.3, repeat_hold=0.1)
    h.press("w", 0.0)
    h.press("w", 0.25)  # autorepeat while still held
    assert h.is_held("w", 0.34)
    assert not h.is_held("w", 0.36)


def test_explicit_release():
    h = HeldKeys()
    h.press_forever("a")
    assert h.is_held("a", 1e9)
    h.release("a")
    assert not h.is_held("a", 0.0)


def test_twist_mapping():
    h = HeldKeys()
    for k in ("w", "a", "r", "j"):
        h.press_forever(k)
    t = twist_from_keys(h, 0.0, linear_speed=0.1, angular_speed=0.5)
    np.testing.assert_allclose(t, [0.1, 0.1, 0.1, 0, 0, 0.5])


def test_opposing_keys_cancel():
    h = HeldKeys()
    h.press_forever("w")
    h.press_forever("s")
    assert not np.any(twist_from_keys(h, 0.0, 0.1, 0.5))


def test_gripper_keys():
    h = HeldKeys()
    assert gripper_from_keys(h, 0.0) == 0.0
    h.press_forever("m")
    assert gripper_from_keys(h, 0.0) == 1.0
