"""``cardputer-hand-view``: see what the webcam hand tracker measures, before it drives anything.

Open http://localhost:8080. You get the camera preview with the hand skeleton drawn on, a ball
that follows your hand in 3D (in robot-style axes: x toward the screen, y left, z up, with the
camera at the origin), a short trail, and live numbers:

* **jitter**: how much the position wobbles while you hold still (standard deviation over the
  last second, in mm). This is the main "is MediaPipe good enough?" number, see
  docs/phase4-hand-tracking.md. Hold the Cardputer the way you will when teleoperating.
* **hand seen**: % of frames where a hand was found. Drops here mean the arm would pause.

    cardputer-hand-view               # built-in webcam
    cardputer-hand-view --fake        # pretend camera, no hardware
    cardputer-hand-view --camera 1    # another camera
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from cardputer_teleop.hand import DEFAULT_AXES, OneEuroFilter, parse_axes
from cardputer_teleop.hand_tracker import DEFAULT_HFOV, HAND_SIDES, FakeHandTracker, WebcamHandTracker
from cardputer_teleop.transforms import matrix_to_wxyz

# viser draws a camera looking along its own +z; point that at you (-x), picture-down = -z.
CAMERA_R = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="Pretend camera (no hardware, no mediapipe needed)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--hand-side", choices=HAND_SIDES, default="any")
    p.add_argument("--hand-axes", default=DEFAULT_AXES)
    p.add_argument("--camera-hfov", type=float, default=DEFAULT_HFOV)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N s (0 = until Ctrl+C)")
    return p.parse_args(argv)


def jitter_mm(samples) -> np.ndarray:
    """Per-axis standard deviation of a list of positions, in mm."""
    a = np.asarray(samples)
    return a.std(axis=0) * 1000 if len(a) > 2 else np.full(3, np.nan)


def main(argv=None):
    args = parse_args(argv)
    import viser

    M = parse_axes(args.hand_axes)
    tracker = (FakeHandTracker() if args.fake else
               WebcamHandTracker(camera=args.camera, side=args.hand_side, hfov_deg=args.camera_hfov)).start()
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (-0.2, -1.2, 0.5)
    server.initial_camera.look_at = (-0.4, 0.0, 0.0)
    server.scene.add_grid("/grid", width=1.5, height=1.5, cell_size=0.1, position=(-0.5, 0.0, -0.3))
    # The camera sits at the origin looking toward -x (toward you).
    server.scene.add_camera_frustum("/camera", fov=np.radians(args.camera_hfov), aspect=16 / 9, scale=0.08,
                                    wxyz=matrix_to_wxyz(CAMERA_R), color=(120, 120, 120))
    ball = server.scene.add_icosphere("/hand", radius=0.02, color=(255, 140, 40))
    trail_pts: deque = deque(maxlen=60)
    status = server.gui.add_markdown("waiting for the camera...")
    preview = server.gui.add_image(np.zeros((180, 320, 3), dtype=np.uint8), label="Webcam", format="jpeg",
                                   jpeg_quality=70)
    filt = OneEuroFilter()
    raw_hist: deque = deque()
    smooth_hist: deque = deque()
    last_t, last_preview = None, None
    t_start = time.monotonic()
    print(f"Open http://localhost:{args.port}  (Ctrl+C to quit)")
    try:
        while not (args.duration and time.monotonic() - t_start > args.duration):
            now = time.monotonic()
            obs = tracker.latest
            fresh = obs is not None and now - obs.t < 0.25
            if fresh and obs.t != last_t:
                last_t = obs.t
                p_raw = M @ obs.p_cam
                p = filt(p_raw, obs.t)
                for hist, v in ((raw_hist, p_raw), (smooth_hist, p)):
                    hist.append((obs.t, v))
                    while hist and hist[0][0] < now - 1.0:
                        hist.popleft()
                ball.position = tuple(p)
                trail_pts.append(p)
                if len(trail_pts) > 1:
                    pts = np.array(trail_pts)
                    server.scene.add_spline_catmull_rom("/trail", pts, color=(255, 190, 120))
            elif not fresh:
                filt.reset()
            ball.visible = fresh
            st = tracker.stats
            lines = [(f"**Camera** {st.fps:.0f} fps, MediaPipe {st.proc_ms:.0f} ms/frame, "
                      f"hand seen in {st.seen_pct:.0f}% of frames")]
            if st.error:
                lines.append(f"**{st.error}**")
            if fresh:
                p = smooth_hist[-1][1]
                jr = jitter_mm([v for _, v in raw_hist])
                js = jitter_mm([v for _, v in smooth_hist])
                lines += [(f"**Hand** ({obs.side}, {obs.score:.2f}) at x {p[0]:+.3f}, y {p[1]:+.3f}, "
                           f"z {p[2]:+.3f} m (x = toward screen)"),
                          (f"**Jitter, last 1 s** (hold still): raw {jr[0]:.1f} / {jr[1]:.1f} / {jr[2]:.1f} mm, "
                           f"smoothed {js[0]:.1f} / {js[1]:.1f} / {js[2]:.1f} mm (x / y / z)")]
            else:
                lines.append("**No hand in view**")
            status.content = "  \n".join(lines)
            if tracker.preview is not None and tracker.preview is not last_preview:
                last_preview = tracker.preview
                preview.image = last_preview
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        server.stop()


if __name__ == "__main__":
    main()
