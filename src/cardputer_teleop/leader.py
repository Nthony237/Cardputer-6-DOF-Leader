"""Phase 3: turn Cardputer packets into arm targets (clutch-relative orientation + IMU translation).

Plain-language version (more in docs/phase3-orientation-control.md):

* **Clutch = "grab".** When you press the clutch (space or G0), we take two snapshots: how the
  Cardputer is turned right now, and where the gripper (TCP) is right now. While you keep
  holding, we measure how far the Cardputer has turned *since the snapshot* and turn the
  gripper by the same amount from *its* snapshot. Let go, move your hand somewhere
  comfortable, grab again: the arm stays put while you re-grip, like lifting a mouse off the
  desk. That is "clutch-relative" control. The arm never jumps to match the absolute
  angle of the Cardputer.
* **Heading.** The Cardputer can't tell north from south (no compass), so "which way is
  forward" is whatever it pointed at when you last pressed ``z``. Press ``z`` with the screen
  end pointing the way the robot faces (the robot's +x). ``heading_offset_deg`` fixes it up if
  you'd rather stand facing the robot (use 180).
* **Translation, experimental.** An IMU measures rotation well but position badly: position
  is acceleration integrated *twice*, so any small error grows fast. Two ideas to compare:

  - ``push``: while the clutch is held, quick pushes of your hand nudge the gripper.
    Gravity is subtracted from the accelerometer, we integrate only while a push is happening,
    and the speed is forced back to zero as soon as your hand stops, so errors can't pile up.
  - ``tilt``: hold ``x`` and the Cardputer becomes a joystick. Tip it forward/back/left/right
    to drive the gripper that way, twist it to go up/down. The gripper's angle stays frozen.
* **Phase 4: position from the webcam.** With ``LeaderSettings.hand`` set, holding the clutch
  also moves the gripper by however far your hand moved since the grab, as seen by the camera
  (``hand.py``). The IMU still gives the orientation, so the two sensors each do what they're
  good at. ``x`` (tilt) still works as the camera-free fallback.

Everything this module produces goes through ``CartesianController`` (workspace box, lead
leash, IK speed and joint limits, heartbeat), so it adds no new way to bypass the safety chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cardputer_teleop.hand import HandFollower, HandObservation, HandSettings
from cardputer_teleop.protocol import LeaderPacket
from cardputer_teleop.transforms import axis_angle_to_matrix, make_transform, so3_exp, so3_log, wxyz_to_matrix

GRAVITY = 9.80665
TRANSLATION_MODES = ("tilt", "push", "off")
MAX_HAND_SPEED = 2.0  # m/s


@dataclass
class LeaderSettings:
    heading_offset_deg: float = 0.0  # robot-frame heading of the Cardputer's "forward" after z
    rotation_scale: float = 1.0  # 1 = gripper turns exactly as much as your hand
    smoothing_tau: float = 0.06  # s, low-pass on the orientation target (0 = off)
    translation: str = "tilt"  # one of TRANSLATION_MODES
    # push-to-nudge (see PushIntegrator)
    push_start: float = 0.8  # m/s^2: a push starts above this (below is tremor / tilt error)
    push_stop: float = 0.4  # m/s^2: below this for push_still_time means the hand stopped
    push_still_time: float = 0.12  # s
    push_max_time: float = 1.5  # s: give up on a "push" that never ends
    push_gain: float = 1.0  # gripper metres per estimated hand metre
    push_max_offset: float = 0.25  # m: how far one clutch can move the gripper
    # tilt-as-joystick
    tilt_deadband_deg: float = 6.0  # small tilts do nothing, so a shaky hand holds still
    tilt_full_deg: float = 30.0  # tilt at which you reach full speed
    # Phase 4: webcam hand position while the clutch is held (None = off). Replaces push.
    hand: HandSettings | None = None

    def __post_init__(self):
        if self.translation not in TRANSLATION_MODES:
            raise ValueError(f"translation must be one of {TRANSLATION_MODES}, got {self.translation!r}")


@dataclass
class LeaderCommand:
    """What the teleop loop should do this tick. At most one of ``pose`` / ``twist`` is set."""

    mode: str  # "idle", "orient" or "tilt"
    pose: np.ndarray | None = None  # absolute TCP target (4x4) -> controller.set_target
    twist: np.ndarray | None = None  # [v; w] TCP velocity -> controller.apply_twist
    info: str = ""


def _scale_rotation(R: np.ndarray, s: float) -> np.ndarray:
    return R if s == 1.0 else so3_exp(s * so3_log(R))


def deadband_ramp(x: float, dead: float, full: float) -> float:
    """0 inside +-dead, rises linearly to +-1 at +-full, capped at +-1. Keeps the sign of x."""
    mag = (abs(x) - dead) / max(full - dead, 1e-9)
    return math.copysign(min(max(mag, 0.0), 1.0), x)


@dataclass
class PushIntegrator:
    """Hand acceleration (world frame, gravity removed) -> a position offset, one push at a time.

    Position is acceleration integrated twice, and integrating forever drifts without limit (a
    constant 0.1 m/s^2 error is 5 m after 10 s). So we only integrate *during a push*:

    1. Idle: ignore everything below ``push_start`` (hand tremor, filter tilt error).
    2. A reading above it starts a push. Now integrate every reading, small ones included,
       because a push is "speed up, then slow down" and both halves are needed to get the
       distance right (the slowing-down half is what brings the speed back to zero).
    3. Once the readings have stayed under ``push_stop`` for ``push_still_time``, the hand has
       stopped: force the speed to exactly zero ("zero-velocity update", the trick pedometers
       use) and go back to idle. Whatever drift built up during the push is thrown away.

    A push that never seems to end (``push_max_time``) is cut off the same way.
    """

    settings: LeaderSettings
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pushing: bool = False
    _quiet: float = 0.0
    _elapsed: float = 0.0
    pushes: int = 0

    def reset(self) -> None:
        self.v = np.zeros(3)
        self.offset = np.zeros(3)
        self.pushing = False

    def _stop(self) -> None:
        self.pushing = False
        self.v = np.zeros(3)

    def update(self, accel_world: np.ndarray, dt: float) -> np.ndarray:
        s = self.settings
        a = np.asarray(accel_world, dtype=float)
        n = float(np.linalg.norm(a))
        if not self.pushing:
            if n < s.push_start:
                return self.offset
            self.pushing, self._quiet, self._elapsed = True, 0.0, 0.0
            self.pushes += 1
        self._elapsed += dt
        self._quiet = self._quiet + dt if n < s.push_stop else 0.0
        if self._quiet >= s.push_still_time or self._elapsed >= s.push_max_time:
            self._stop()
            return self.offset
        self.v = self.v + a * dt
        sp = float(np.linalg.norm(self.v))
        if sp > MAX_HAND_SPEED:  # faster than a hand moves: sensor glitch, don't trust it
            self.v *= MAX_HAND_SPEED / sp
        # This is where the *hand* went. The arm follows at its own speed limit: the
        # controller's leash and the IK caps make it catch up smoothly instead of jumping.
        self.offset = self.offset + s.push_gain * self.v * dt
        n_off = float(np.linalg.norm(self.offset))
        if n_off > s.push_max_offset:
            self.offset *= s.push_max_offset / n_off
        return self.offset


class CardputerLeader:
    """Stateful mapper from the newest packet to a controller command. No I/O, easy to test."""

    def __init__(self, settings: LeaderSettings | None = None):
        self.settings = settings or LeaderSettings()
        self.push = PushIntegrator(self.settings)
        self.hand = HandFollower(self.settings.hand) if self.settings.hand is not None else None
        self.mode = "idle"
        self._R_dev0: np.ndarray | None = None  # device orientation at the grab (robot frame)
        self._T_tcp0: np.ndarray | None = None  # TCP pose at the grab
        self._R_smooth: np.ndarray | None = None
        self._yaw_count: int | None = None
        self.grabs = 0  # how many times we re-anchored (handy in tests and the status line)

    # The Cardputer's world frame has z up and an arbitrary heading; rotate it about z so its
    # forward lines up with the robot's.
    def _M(self) -> np.ndarray:
        return axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), math.radians(self.settings.heading_offset_deg))

    def device_rotation(self, pkt: LeaderPacket) -> np.ndarray:
        """Device orientation expressed in the robot base frame."""
        return self._M() @ wxyz_to_matrix(pkt.quat)

    def accel_world(self, pkt: LeaderPacket) -> np.ndarray:
        """Hand acceleration in the robot frame with gravity removed (m/s^2).

        An accelerometer at rest reads +g "up" (it feels the table pushing). Rotate the reading
        into the world, subtract that 9.81 up, and what's left is the hand actually speeding up.
        """
        return self._M() @ (wxyz_to_matrix(pkt.quat) @ pkt.accel - np.array([0.0, 0.0, GRAVITY]))

    def release(self) -> None:
        self.mode = "idle"
        self._R_dev0 = self._T_tcp0 = self._R_smooth = None
        self.push.reset()
        if self.hand is not None:
            self.hand.reset()

    def _grab(self, pkt: LeaderPacket, T_tcp: np.ndarray, mode: str) -> None:
        self.mode = mode
        self._R_dev0 = self.device_rotation(pkt)
        self._T_tcp0 = np.array(T_tcp, dtype=float)
        self._R_smooth = self._T_tcp0[:3, :3].copy()
        self._yaw_count = pkt.yaw_reset_count
        self.push.reset()
        if self.hand is not None:
            self.hand.reset()
        self.grabs += 1

    def update(self, pkt: LeaderPacket | None, T_tcp: np.ndarray, dt: float, linear_speed: float,
               enabled: bool = True, hand: HandObservation | None = None,
               now: float | None = None) -> LeaderCommand:
        """One control tick.

        ``T_tcp`` is the arm's *actual* TCP pose (forward kinematics), used as the snapshot on a
        grab so the arm never jumps. ``enabled`` is False whenever the loop must not follow the
        device (link lost, hold pressed, homing): we let go, and grab afresh once it's True again.
        ``hand`` is the newest webcam observation (Phase 4) and ``now`` the time.monotonic() it is
        compared with to tell whether the hand is still in view.
        """
        s = self.settings
        if not enabled or pkt is None:
            self.release()
            return LeaderCommand("idle")
        if pkt.calibrating:
            # The orientation is garbage while the gyro bias is measured / the filter settles.
            self.release()
            return LeaderCommand("idle", info="Cardputer calibrating, hold it still")

        want = "idle"
        if pkt.tilt_move and s.translation == "tilt":
            want = "tilt"
        elif pkt.clutch:
            want = "orient"
        if want == "idle":
            self.release()
            return LeaderCommand("idle")
        # Re-grab when the mode changes, or when z was pressed mid-grab (the quaternion jumps).
        if want != self.mode or pkt.yaw_reset_count != self._yaw_count:
            self._grab(pkt, T_tcp, want)

        dR = self.device_rotation(pkt) @ self._R_dev0.T  # how far the hand turned since the grab
        if self.mode == "tilt":
            return LeaderCommand("tilt", twist=self._tilt_twist(dR, linear_speed))

        R_des = _scale_rotation(dR, s.rotation_scale) @ self._T_tcp0[:3, :3]
        if s.smoothing_tau > 0:
            alpha = 1.0 - math.exp(-dt / s.smoothing_tau)
            self._R_smooth = so3_exp(alpha * so3_log(R_des @ self._R_smooth.T)) @ self._R_smooth
        else:
            self._R_smooth = R_des
        p = self._T_tcp0[:3, 3].copy()
        info = f"turned {math.degrees(np.linalg.norm(so3_log(dR))):.0f} deg since grab"
        if self.hand is not None:
            off = self.hand.update(hand, now if now is not None else (hand.t if hand else 0.0))
            p = p + off
            state = "moved" if self.hand.tracking else "hand not in view, position held at"
            info += f", {state} {np.linalg.norm(off) * 100:.1f} cm"
        elif s.translation == "push":
            off = self.push.update(self.accel_world(pkt), dt)
            p = p + off
            info += f", pushed {np.linalg.norm(off) * 100:.1f} cm"
        return LeaderCommand("orient", pose=make_transform(self._R_smooth, p), info=info)

    def _tilt_twist(self, dR: np.ndarray, linear_speed: float) -> np.ndarray:
        """Tilt since the grab -> TCP velocity. Orientation stays frozen (zero angular part).

        ``so3_log`` gives the rotation as an axis * angle vector in the robot frame (right-hand
        rule: thumb along the axis, fingers curl the positive way):

        - tipping the front (+x end) down is a +y rotation -> go +x (forward);
        - tipping the right (-y side) down is a +x rotation -> go -y (to the right);
        - twisting counter-clockwise seen from above is a +z rotation -> go up.
        """
        s = self.settings
        w = so3_log(dR)
        d, f = math.radians(s.tilt_deadband_deg), math.radians(s.tilt_full_deg)
        v = linear_speed * np.array([
            deadband_ramp(w[1], d, f),  # pitch forward -> +x
            -deadband_ramp(w[0], d, f),  # roll right -> -y
            deadband_ramp(w[2], d, f),  # twist CCW -> +z
        ])
        return np.concatenate([v, np.zeros(3)])
