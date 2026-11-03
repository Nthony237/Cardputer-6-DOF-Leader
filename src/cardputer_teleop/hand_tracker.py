"""Phase 4: the webcam + MediaPipe side. Runs in a background thread, like the UDP receiver.

The control loop only reads ``tracker.latest`` (the newest ``HandObservation``, or None) and,
for the browser preview, ``tracker.preview``. It never waits on the camera.

Needs the optional extras: ``pip install -e ".[hand]"`` (mediapipe + opencv). The first run
downloads MediaPipe's hand model (about 8 MB) into ``~/.cache/cardputer-teleop/``.

``FakeHandTracker`` pretends to be a camera watching a hand that draws slow loops, so the
tests and ``--hand fake`` work with no camera and without mediapipe installed.
"""
from __future__ import annotations

import math
import ssl
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardputer_teleop.hand import HAND_CONNECTIONS, HandObservation, focal_length_px, hand_position

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")
DEFAULT_MODEL = Path.home() / ".cache" / "cardputer-teleop" / "hand_landmarker.task"
HAND_SIDES = ("any", "left", "right")
DEFAULT_HFOV = 65.0  # deg, close to a MacBook FaceTime camera; only scales the depth (see docs)
PREVIEW_WIDTH = 320


def ensure_model(path: str | Path | None = None) -> Path:
    """Path to the hand model, downloading it the first time."""
    path = Path(path) if path else DEFAULT_MODEL
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe's hand model (~8 MB) to {path} ...", flush=True)
    tmp = path.with_suffix(".part")
    try:
        ctx = None
        try:  # python.org Pythons on macOS often lack root certificates; mediapipe ships certifi
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(MODEL_URL, context=ctx, timeout=60) as r:
            tmp.write_bytes(r.read())
    except Exception as e:  # usually missing SSL certificates on a python.org Python
        raise SystemExit(
            f"Could not download the hand model ({e}).\nDownload it yourself with:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {MODEL_URL}") from e
    tmp.rename(path)
    return path


@dataclass
class TrackerStats:
    fps: float = 0.0  # camera frames processed per second
    seen_pct: float = 0.0  # % of recent frames with a hand found
    proc_ms: float = 0.0  # MediaPipe time per frame
    error: str = ""


def draw_preview(frame_bgr: np.ndarray, image_xy: np.ndarray | None, width: int = PREVIEW_WIDTH) -> np.ndarray:
    """Small RGB copy of the frame with the hand skeleton drawn on, for the browser."""
    import cv2

    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (width, int(h * width / w)))
    if image_xy is not None:
        pts = (np.asarray(image_xy) * [small.shape[1], small.shape[0]]).astype(int)
        for i, j in HAND_CONNECTIONS:
            cv2.line(small, tuple(pts[i]), tuple(pts[j]), (80, 220, 80), 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(small, tuple(p), 2, (40, 40, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class WebcamHandTracker:
    """Reads the webcam, runs MediaPipe, keeps the newest hand position.

    The camera is opened here, in the caller's (main) thread, on purpose: macOS only shows its
    "allow camera access?" prompt for the main thread. Frames are then read in a background one.
    """

    def __init__(self, camera: int = 0, side: str = "any", hfov_deg: float = DEFAULT_HFOV,
                 model_path: str | Path | None = None, width: int = 1280, height: int = 720):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise SystemExit(f"Hand tracking needs mediapipe and opencv ({e}).\n"
                             '  pip install -e ".[hand]"') from e
        if side not in HAND_SIDES:
            raise ValueError(f"side must be one of {HAND_SIDES}")
        self._cv2, self._mp = cv2, mp
        self.side = side
        self.hfov_deg = hfov_deg
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._cap = cv2.VideoCapture(camera)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise SystemExit(
                f"Could not open camera {camera}. On a Mac: System Settings > Privacy & Security > "
                "Camera, allow your terminal app, then quit and reopen the terminal.")
        self.latest: HandObservation | None = None
        self.preview: np.ndarray | None = None
        self.stats = TrackerStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> WebcamHandTracker:
        self._thread = threading.Thread(target=self._run, daemon=True, name="hand-tracker")
        self._thread.start()
        return self

    def _pick(self, result) -> int | None:
        best, best_score = None, -1.0
        for i, cats in enumerate(result.handedness):
            label, score = cats[0].category_name, cats[0].score
            if self.side != "any" and label.lower() != self.side:
                continue
            if score > best_score:
                best, best_score = i, score
        return best

    def _run(self) -> None:
        cv2, mp = self._cv2, self._mp
        seen = deque(maxlen=60)
        times = deque(maxlen=30)
        last_ts = -1
        fails = 0
        frame_i = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            t = time.monotonic()
            if not ok or frame is None:
                fails += 1
                if fails > 60:
                    self.stats.error = ("no frames from the camera: check the terminal is allowed to use it "
                                        "(System Settings > Privacy & Security > Camera)")
                time.sleep(0.03)
                continue
            fails = 0
            self.stats.error = ""
            frame = cv2.flip(frame, 1)  # mirror it like a selfie: MediaPipe's left/right expect that
            h, w = frame.shape[:2]
            ts = max(int(t * 1000), last_ts + 1)  # MediaPipe wants strictly increasing ms
            last_ts = ts
            t0 = time.perf_counter()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._landmarker.detect_for_video(image, ts)
            self.stats.proc_ms = (time.perf_counter() - t0) * 1000
            i = self._pick(result)
            xy = None
            if i is not None:
                xy = np.array([[p.x, p.y] for p in result.hand_landmarks[i]])
                P = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
                p = hand_position(xy, P, w, h, focal_length_px(w, self.hfov_deg))
                if p is not None:
                    cats = result.handedness[i][0]
                    self.latest = HandObservation(t=t, p_cam=p, side=cats.category_name, score=cats.score,
                                                  image_xy=xy)
            seen.append(xy is not None)
            times.append(t)
            if len(times) > 1:
                self.stats.fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
            self.stats.seen_pct = 100.0 * sum(seen) / len(seen)
            frame_i += 1
            if frame_i % 3 == 0:  # ~10 Hz is plenty for a preview
                self.preview = draw_preview(frame, xy)

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._cap.release()
        self._landmarker.close()


class FakeHandTracker:
    """A pretend camera: the hand traces a slow 3D loop about 50 cm in front of the screen.

    Every 6 s it "loses" the hand for 0.4 s, and once it reappears 20 cm away (a glitch), so
    the freeze / re-zero logic gets exercised. No thread needed: ``latest`` is computed on read.
    """

    side = "any"

    def __init__(self, rate_hz: float = 30.0, noise_m: float = 0.002, seed: int | None = 0):
        self.rate_hz = rate_hz
        self.noise_m = noise_m
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()
        self.preview = None
        self.stats = TrackerStats(fps=rate_hz, seen_pct=100.0)

    @staticmethod
    def position(c: float) -> np.ndarray:
        """Camera-frame hand position ``c`` seconds into the loop (no noise)."""
        return np.array([0.10 * math.sin(0.8 * c), -0.06 * math.sin(1.6 * c), 0.50 + 0.08 * math.sin(0.5 * c)])

    def observation(self, t: float) -> HandObservation | None:
        c = t - self._t0
        k = math.floor(c * self.rate_hz) / self.rate_hz  # frames arrive at the camera rate
        if (k % 6.0) > 5.6:
            return None
        p = self.position(k)
        if 8.0 <= (k % 24.0) < 8.4:
            p = p + np.array([0.2, 0.0, 0.0])
        p = p + self._rng.normal(0.0, self.noise_m, 3)
        return HandObservation(t=self._t0 + k, p_cam=p, side="Right", score=0.95)

    @property
    def latest(self) -> HandObservation | None:
        return self.observation(time.monotonic())

    def start(self) -> FakeHandTracker:
        return self

    def close(self) -> None:
        pass
