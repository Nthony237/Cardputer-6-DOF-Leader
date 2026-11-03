"""Phase 4: where is the hand? Webcam landmarks -> a 3D position -> a clutch-relative arm offset.

This file is the math only (numpy, no camera), so it is easy to test. The camera and MediaPipe
live in ``hand_tracker.py``. The plain-language story is in docs/phase4-hand-tracking.md.

Three steps:

1. **Landmarks -> 3D position** (``hand_position``). MediaPipe gives us, for the 21 hand
   joints, (a) where each one is *in the picture* and (b) the 3D shape of the hand in metres,
   but centred on the hand itself, so it doesn't say how far away the hand is. A pinhole camera
   shrinks things in proportion to their distance, so comparing the real shape (b) with its
   size in the picture (a) tells us the distance. With 21 points that is a small least-squares
   problem we can solve exactly in a few lines (see the function).
2. **Smoothing** (``OneEuroFilter``). The estimate jitters by a few millimetres frame to frame.
   The "1 euro filter" smooths hard when the hand is still (kills jitter) and lightly when
   it moves fast (keeps the lag low).
3. **Clutch-relative offset** (``HandFollower``). Exactly like the orientation in Phase 3:
   when you grab, the hand's position is the zero point, and the gripper moves by however far
   the hand moved *since the grab*. If the camera loses the hand, the offset freezes; when it
   finds it again we re-zero on the new spot, so the arm never jumps because of the camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# MediaPipe's 21 hand landmarks. We only need a few by name.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
]

# Camera frame used everywhere here (the usual computer-vision one), in the *mirrored* image you
# see in the preview, like a selfie: x = right in the picture, y = down, z = away from the
# camera. Default mapping to the robot, for a person sitting at the laptop with the robot's
# forward (+x) pointing from them toward the screen:
#   hand toward the screen (closer to the camera, -z)  -> robot +x (forward)
#   hand to the left (-x in the mirrored image)         -> robot +y (left)
#   hand up (-y in the picture)                         -> robot +z (up)
DEFAULT_AXES = "-z,-x,-y"


def parse_axes(spec: str) -> np.ndarray:
    """``"-z,-x,-y"`` -> the 3x3 matrix taking camera-frame vectors to robot-frame vectors.

    Entry k says which camera axis (with sign) becomes robot axis k (x, y, z).
    """
    parts = [s.strip().lower() for s in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"axes needs 3 comma-separated entries like {DEFAULT_AXES!r}, got {spec!r}")
    M = np.zeros((3, 3))
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        name = part.lstrip("+-")
        if name not in ("x", "y", "z"):
            raise ValueError(f"bad axis {part!r} in {spec!r}")
        M[row, "xyz".index(name)] = sign
    if abs(abs(np.linalg.det(M)) - 1.0) > 1e-9:
        raise ValueError(f"axes {spec!r} use the same camera axis twice")
    return M


def focal_length_px(width: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from the image width and horizontal field of view."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def hand_position(image_xy: np.ndarray, world_xyz: np.ndarray, width: int, height: int,
                  f_px: float) -> np.ndarray | None:
    """Camera-frame position (m) of the hand's centre, from one frame of MediaPipe output.

    ``image_xy``: (21, 2) landmarks in the picture, as fractions of width/height (MediaPipe's
    ``hand_landmarks``). ``world_xyz``: (21, 3) the same joints in metres around the hand's
    centre (``hand_world_landmarks``).

    The pinhole camera says a point at camera position (X, Y, Z) lands at pixel
    ``u = cx + f X / Z``. Each joint is at ``P_i + t`` where ``P_i`` is its world landmark and
    ``t`` is the unknown hand position. Multiply out:

        u_i - cx = f (X_i + tx) / (Z_i + tz)   ->   f tx - (u_i - cx) tz = (u_i - cx) Z_i - f X_i

    and the same for v and ty. That is linear in t: 42 equations, 3 unknowns, one least-squares
    solve. Because ``P_i`` already includes how the hand is turned, tilting the hand doesn't
    look like moving it closer (it would if we just measured the hand's size in pixels).

    Returns None if the answer is nonsense (behind the camera or absurdly far).
    """
    uv = np.asarray(image_xy, dtype=float) * np.array([width, height])
    P = np.asarray(world_xyz, dtype=float)
    a = uv[:, 0] - width / 2.0
    b = uv[:, 1] - height / 2.0
    n = len(P)
    A = np.zeros((2 * n, 3))
    rhs = np.zeros(2 * n)
    A[:n, 0], A[:n, 2], rhs[:n] = f_px, -a, a * P[:, 2] - f_px * P[:, 0]
    A[n:, 1], A[n:, 2], rhs[n:] = f_px, -b, b * P[:, 2] - f_px * P[:, 1]
    t, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if not (0.05 < t[2] < 5.0):
        return None
    return t


def pinch_ratio(world_xyz: np.ndarray) -> float:
    """Thumb-tip-to-index-tip distance divided by the palm's length (wrist to middle knuckle).

    Uses the 3D landmarks in metres, so it doesn't change with distance from the camera or with
    how the hand is turned. Touching fingers read about 0.1 to 0.2, an open hand 0.6 or more.
    """
    P = np.asarray(world_xyz, dtype=float)
    palm = float(np.linalg.norm(P[MIDDLE_MCP] - P[WRIST]))
    return float(np.linalg.norm(P[THUMB_TIP] - P[INDEX_TIP])) / max(palm, 1e-6)


class PinchDetector:
    """Pinch = "grab": thumb and index together means follow, apart means stop.

    Two thresholds ("hysteresis") so a pinch that hovers near the edge doesn't flicker on and
    off: it switches on below ``on`` and only switches off again above ``off``. A hand that
    isn't in view (no fresh frame for ``lost_timeout``) is never pinching, so losing the hand
    stops the arm just like letting go.
    """

    def __init__(self, on: float = 0.30, off: float = 0.45, lost_timeout: float = 0.25):
        self.on, self.off, self.lost_timeout = on, off, lost_timeout
        self.pinched = False

    def update(self, obs: HandObservation | None, now: float) -> bool:
        if obs is None or obs.pinch is None or now - obs.t > self.lost_timeout:
            self.pinched = False
        elif self.pinched:
            self.pinched = obs.pinch < self.off
        else:
            self.pinched = obs.pinch < self.on
        return self.pinched


class OneEuroFilter:
    """The "1 euro filter" (Casiez et al., 2012): a low-pass whose cutoff rises with speed.

    Still hand -> cutoff ``min_cutoff`` Hz, lots of smoothing, no jitter. Fast hand -> cutoff
    goes up by ``beta`` Hz per m/s, so the output keeps up instead of lagging behind.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 10.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx = np.zeros(3)
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x.copy(), t
            return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = self._dx + a_d * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x.copy()


@dataclass
class HandObservation:
    """One camera frame's answer: where the hand was, and when."""

    t: float  # time.monotonic() when the frame was captured
    p_cam: np.ndarray  # hand centre in the camera frame (m), see DEFAULT_AXES
    side: str = "?"  # "Left" / "Right", as MediaPipe sees it (on the mirrored image = your hand)
    score: float = 1.0  # MediaPipe's confidence it's a hand
    image_xy: np.ndarray | None = None  # (21, 2) landmarks, for drawing the preview
    pinch: float | None = None  # pinch_ratio(): ~0.15 fingers touching, 0.6+ open


@dataclass
class HandSettings:
    axes: str = DEFAULT_AXES  # camera -> robot axis mapping
    gain: float = 1.0  # gripper metres per hand metre
    lost_timeout: float = 0.25  # s without a fresh frame = hand lost, position freezes
    max_jump: float = 0.12  # m between two frames: a glitch (or a different hand), re-zero
    min_cutoff: float = 1.5  # Hz, OneEuroFilter
    beta: float = 10.0  # Hz per m/s, OneEuroFilter


@dataclass
class HandFollower:
    """Hand observations -> robot-frame offset since the grab. No I/O; call ``reset`` on a grab.

    Rules, in order, every control tick:

    * no new frame for ``lost_timeout`` -> "lost": keep the current offset (the arm holds its
      position; orientation from the IMU keeps working).
    * a new frame after being lost, or a jump bigger than ``max_jump`` since the last frame ->
      re-zero: the offset carries on from its current value, measured from the new spot.
    * otherwise -> offset = gain * (smoothed hand position - hand position at the zero point).
    """

    settings: HandSettings = field(default_factory=HandSettings)
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tracking: bool = False  # True while fresh frames keep coming
    reacquired: int = 0  # how many times we had to re-zero (handy in tests and the status)

    def __post_init__(self):
        self._M = parse_axes(self.settings.axes)
        self._filter = OneEuroFilter(self.settings.min_cutoff, self.settings.beta)
        self._anchor: np.ndarray | None = None
        self._last_t: float | None = None
        self._last_raw: np.ndarray | None = None

    def reset(self) -> None:
        self.offset = np.zeros(3)
        self.tracking = False
        self._anchor = self._last_t = self._last_raw = None
        self._filter.reset()

    def update(self, obs: HandObservation | None, now: float) -> np.ndarray:
        s = self.settings
        if obs is None or now - obs.t > s.lost_timeout:
            self.tracking = False
            return self.offset
        if self._last_t is not None and obs.t <= self._last_t:
            return self.offset  # same frame as last tick (camera ~30 Hz, control loop 50 Hz)
        raw = np.asarray(obs.p_cam, dtype=float)
        jumped = self._last_raw is not None and float(np.linalg.norm(raw - self._last_raw)) > s.max_jump
        self._last_t, self._last_raw = obs.t, raw
        if self._anchor is None or not self.tracking or jumped:
            if self._anchor is not None:
                self.reacquired += 1
            self._filter.reset()
            h = s.gain * (self._M @ self._filter(raw, obs.t))
            self._anchor = h - self.offset  # carry on from where we are: no jump
            self.tracking = True
            return self.offset
        h = s.gain * (self._M @ self._filter(raw, obs.t))
        self.offset = h - self._anchor
        return self.offset
