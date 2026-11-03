# Phase 4: moving the gripper with your hand, seen by the webcam (how it works)

Goal: hold the Cardputer, hold **space**, and move your hand. The simulated gripper **moves**
the way your hand moves (the laptop's webcam watches it) and **turns** the way the Cardputer
turns (its IMU, exactly as in Phase 3). Sim only for now.

```
 webcam ──► MediaPipe ──► hand position (m) ──┐
 (30 fps)    21 joints     hand.py             ├─► CardputerLeader ──► pose ──► CartesianController ──► IK ──► arm
 Cardputer ─► UDP ──────► orientation + keys ──┘    clutch-relative      (workspace box, 5 cm leash, speed
 (100 Hz)                                           position + rotation   limits, heartbeat: unchanged)
```

## Why two sensors

Phase 3 showed the problem: an IMU measures **rotation** very well, but **position** badly,
because position is acceleration integrated twice and small errors snowball. A camera is the
opposite: it sees *where* your hand is without drifting, but it can't see the Cardputer's
angle well. So each sensor does the part it's good at.

## Step 1: from a picture to a 3D position

MediaPipe (Google's hand tracker) finds 21 points on your hand (wrist, knuckles, fingertips)
in every frame. It gives two things for each point:

- **where it is in the picture** (pixels);
- **the 3D shape of the hand in metres**, but centred on the hand itself. That says how the
  hand is shaped and turned, not how far away it is.

A camera makes things look smaller the farther away they are (a hand twice as far looks half
as big). So comparing the real-size shape with its size in the picture gives the distance.
With 21 points this is a small "least squares" problem (21 points, 2 equations each, 3
unknowns: x, y and distance) that numpy solves in microseconds. Details are in the comment of
`hand_position()` in `src/cardputer_teleop/hand.py`.

Because the 3D shape already includes how your hand is tilted, tilting it doesn't fool the
estimate into thinking it moved closer. I checked this on real hand photos: shrinking the
photo to half size made the estimated distance double (0.21 m → 0.45 m), as it should.

Two honest caveats:

- **Camera field of view.** The distance needs the camera's zoom (`--camera-hfov`, default
  65°, close to a MacBook webcam). A wrong value only scales the toward/away distances, not
  left/right/up/down. Since control is relative and has a gain, it hardly matters.
- **Hand size.** MediaPipe assumes an average hand. A big hand reads a bit too close. Same
  effect, same "doesn't matter much".

## Step 2: smoothing

The estimate wobbles a few millimetres from frame to frame. A plain low-pass filter would fix
that but makes the arm feel laggy. The **1 euro filter** (`OneEuroFilter`) is the standard
trick: it smooths a lot when your hand is still and very little when it moves fast, so you
get no jitter at rest and little lag in motion.

## Step 3: clutch-relative, like the orientation

Same "computer mouse" idea as Phase 3. When you press space, your hand's position becomes the
zero point. While you hold, the gripper moves by **how far your hand moved since the grab**.
Let go, put your hand somewhere comfortable, grab again. The arm never jumps to where your
hand "is"; it only copies movement.

Directions, for the default setup (you sit facing the laptop, the robot's "forward" points from
you toward the screen):

| Move your hand... | Gripper moves |
|---|---|
| toward the screen / back toward you | forward / back (+x / -x) |
| left / right | left / right (+y / -y) |
| up / down | up / down |

Press **z** on the Cardputer while its screen end points toward the laptop, so the IMU's
"forward" matches the camera's. If your setup is different, `--hand-axes` remaps the camera
axes (see `cardputer-teleop --help`).

## What happens when things go wrong

| Situation | What the arm does |
|---|---|
| You let go of space | Stops where it is (as in Phase 3). |
| The camera loses your hand (out of view, hidden behind the Cardputer) for more than 0.25 s | **Position freezes**, turning still works. The status line says "hand not in view". |
| The hand comes back somewhere else | Nothing jumps: we re-zero on the new spot and continue from there. |
| The estimate leaps more than 12 cm between two frames (a glitch, or it switched to your other hand) | Treated the same way: re-zero, no jump. |
| Cardputer Wi-Fi drops | Heartbeat stops, the arm holds (unchanged from Phase 2). |
| Anything else | The target still goes through the workspace box, the 5 cm leash, the speed limits and the joint limits. The camera adds no new way around them. |

Hold **x** instead of space and tilt-as-joystick works exactly as before. It doesn't use the
camera at all, so it's the fallback when the lighting or camera position is bad.

## Try it (on your Mac)

In Terminal:

```bash
cd ~/Cardputer-6-DOF-Leader
source .venv/bin/activate
git fetch origin
git checkout claude/project-thread-3i3au1
pip install -e ".[dev,hand]"      # adds mediapipe + opencv
pytest                            # all tests should pass
```

The first time you use the camera, macOS asks "Terminal would like to access the camera":
click **OK**. If you missed it: System Settings → Privacy & Security → Camera → turn on
Terminal, then quit and reopen Terminal. The first run also downloads MediaPipe's hand model
(about 8 MB) to `~/.cache/cardputer-teleop/`.

**1. Look at the tracking on its own** (no arm, no Cardputer needed):

```bash
cardputer-hand-view
```

Open http://localhost:8080. You'll see the webcam with your hand's skeleton drawn on, and an
orange ball that follows your hand in 3D. Hold the Cardputer the way you would when
teleoperating. Then try:

- hold still for a few seconds and read the **smoothed jitter** numbers;
- move around and watch **hand seen in ...% of frames**;
- move toward and away from the screen and check the ball goes the right way.

**2. No hardware at all** (fake Cardputer + fake camera):

```bash
cardputer-teleop --fake hand --hand fake
```

**3. The real thing** (Cardputer on rlab Wi-Fi + webcam):

```bash
cardputer-teleop --cardputer --hand webcam
```

Press **z** with the Cardputer's screen end pointing at the laptop. Hold **space** and move
and turn your hand. The webcam picture is in the browser panel. Useful options:
`--hand-gain 0.5` (finer moves), `--hand-side right` (only follow your right hand),
`--camera 1` (an external camera).

## Pinch to grab: Cardputer on the back of your hand

Holding space while gripping the Cardputer is awkward. With `--pinch`, **touching your thumb
and index finger together is the clutch**: pinch = follow, open = stop (space / G0 still work
too). That frees the Cardputer to be strapped on.

**Where to put it: the back of your hand**, with a velcro strap or a fingerless glove, screen
facing up and the screen end toward your fingers. There it turns with your hand (wrist bends
included), your fingers stay free to pinch, and you can still see the screen and reach G0, z
and b with the other hand. On the forearm it would only feel your forearm turning.

How the pinch is measured: the distance between the thumb tip and index tip, divided by the
length of your palm, from MediaPipe's 3D hand shape. Dividing by the palm makes it the same
near or far from the camera. Fingers touching read about 0.15, an open hand 0.6 or more. It
grabs below 0.30 and only lets go above 0.45 (two thresholds, so a pinch near the edge
doesn't flicker). `cardputer-hand-view` shows your live pinch number, so tell me if those
thresholds don't suit your hand.

Safety stays the same: you must keep pinching for the arm to move. Open your fingers, or
let the camera lose your hand for 0.25 s, and the arm stops where it is. The Cardputer still
has to be streaming (it is the heartbeat).

```bash
cardputer-teleop --cardputer --hand webcam --pinch      # the real thing
cardputer-teleop --fake wrist --hand fake --pinch       # no hardware
```

Press **z** once after strapping it on, with your fingers pointing at the laptop. Keep your
palm or the side of your hand toward the camera: if the back of your hand faces it, the
Cardputer hides your fingers. The gripper's open/close is on the laptop (n / m) for now.

## Is MediaPipe good enough, or do we need AprilTag?

The plan keeps AprilTag (a printed square code stuck on the Cardputer) as the fallback.
`cardputer-hand-view` gives the numbers to decide. Rough guide, holding the Cardputer about
40 to 70 cm from the camera:

| Number | Good enough | Consider AprilTag |
|---|---|---|
| smoothed jitter, holding still | 1 to 2 mm | over 3 mm, or the ball creeps |
| hand seen, while holding the Cardputer | over 95 % | drops out whenever you turn the Cardputer |
| toward/away direction | follows your hand | gets confused when you tilt your hand |

The toward/away direction (depth) is the weakest part of a single webcam, so watch that one
most. An AprilTag gives an exact size, so it measures depth better, but only while the tag
faces the camera.

## Files

| File | What's in it |
|---|---|
| `src/cardputer_teleop/hand.py` | The math: picture → 3D position, 1 euro filter, clutch-relative follower with loss/glitch handling, pinch detector. No camera, fully tested. |
| `src/cardputer_teleop/hand_tracker.py` | Webcam + MediaPipe in a background thread (like the UDP receiver), plus a fake camera for tests. |
| `src/cardputer_teleop/hand_view.py` | `cardputer-hand-view`: preview, 3D ball, jitter numbers. |
| `src/cardputer_teleop/leader.py` | Now also takes the hand position while space is held. |
| `tests/test_hand.py` | Tests, including a closed loop: move the hand 10 cm forward, the sim gripper goes 10 cm forward, never too fast, without turning. |

Still to come: the real YAM URDF (the sim still uses the placeholder), and Phase 5 on the real
arm.
