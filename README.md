# Cardputer-6-DOF-Leader

A low-cost wireless handheld controller for Cartesian teleoperation of i2rt YAM arms: move your
hand and the gripper follows, instead of jogging one joint at a time. The controller is an
M5Stack Cardputer ADV (ESP32-S3, BMI270 IMU) streaming over UDP to a laptop that fuses IMU
orientation with AprilTag position and drives the arm through differential IK.

```
Cardputer (IMU + clutch) --UDP--> receiver --> fuse with AprilTag position
    --> damped-least-squares IK (joint / speed / workspace limits) --> viser sim | real YAM
```

## Status

| Phase | What | State |
|---|---|---|
| 1 | Differential IK in sim + Cartesian keyboard teleop | **done (this package)** |
| 2 | Cardputer IMU streaming, orientation shown in viser | next |
| 3 | Orientation control of the sim gripper with a clutch | |
| 4 | AprilTag position tracking + IMU fusion | |
| 5 | Real arm, supervised, all safety limits on | |
| 6 | Evaluation (joint jog vs Cartesian keyboard vs controller) and demo collection | |

The sim currently uses a **placeholder** 6-DOF URDF with YAM-like proportions
(`src/cardputer_teleop/assets/yam_placeholder.urdf`). Swap in the real YAM URDF with a
config file (see below) before trusting any numbers.

## Quick start (MacBook, no arm needed)

Needs Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # unit tests, no hardware
cardputer-teleop            # sim mode is the default
```

Open http://localhost:8080 and keep the terminal focused for keys:

```
w/s  +x/-x (forward/back)      u/o  roll  +/-
a/d  +y/-y (left/right)        i/k  pitch +/-
r/f  +z/-z (up/down)           j/l  yaw   +/-
n/m  close/open gripper        t    rotate in base/tool frame
space  hold/resume             h    go home
[ / ]  slower/faster           q/Esc  quit
```

You can also drag the target gizmo in the browser. The status panel shows tracking error,
manipulability, IK damping and any joint sitting at its limit.

The default `terminal` key backend needs no permissions but only sees key presses, so a tap
moves a short fixed step and holding relies on key repeat. For smooth press/release, use
`pip install -e ".[keys]"` and `cardputer-teleop --keys pynput` (macOS will ask for
Input Monitoring / Accessibility permission for your terminal).

### Using the real YAM URDF

Copy `configs/example.toml`, set `urdf`, `ee_link` (the link at the point between the fingers),
`gripper_joint` and a `home` pose, then `cardputer-teleop --config configs/yam.toml`.
The IK reads the chain from the URDF root to `ee_link` and ignores side branches like fingers.

## Package layout

| Module | Role |
|---|---|
| `kinematics.py` | URDF serial chain: forward kinematics and geometric Jacobian (numpy only) |
| `ik.py` | Damped-least-squares differential IK: adaptive damping near singularities, joint velocity limits, joints locked before their position limits |
| `safety.py` | Workspace bounds, TCP/joint speed limits, heartbeat watchdog |
| `controller.py` | Device-agnostic Cartesian controller: twist or pose in, clamped/leashed target, IK step out; holds when the heartbeat stops |
| `keyboard.py` | Key map and key readers (terminal, pynput) |
| `robots/` | Sim/real switch: `SimRobot` (rate-limited kinematic sim) and the `YamRobot` stub for Phase 5 |
| `viz.py` | viser scene: URDF, TCP and target frames, workspace box, GUI |
| `teleop.py` | `cardputer-teleop` CLI and control loop |

## Safety design

Every command, whatever the input device, goes through the same chain:
workspace clamp → target leash (the target may only lead the real TCP by 5 cm / 0.5 rad, so an
unreachable target can't wind up) → TCP speed limit → IK joint speed limit → joint position
limits with margin → robot-side rate limit. The heartbeat watchdog makes the controller hold
when input stops (the Cardputer receiver will beat once per packet). `--mode real` deliberately
raises until Phase 5, which also needs a deadman key, a hardware e-stop within reach, and
supervised testing only.
