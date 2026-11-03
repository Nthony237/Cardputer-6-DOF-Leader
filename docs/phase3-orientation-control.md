# Phase 3: steering the sim arm with the Cardputer (how it works)

Goal: hold the Cardputer, press a key, and the simulated gripper turns (and, experimentally,
moves) the way your hand does. All of the new logic is in `src/cardputer_teleop/leader.py`.

```
 packet (quat, accel, keys) ──► CardputerLeader ──► pose or velocity ──► CartesianController ──► IK ──► arm
                                  │                                        (workspace box, leash,
                                  ├─ space held: turn the gripper           speed limits, heartbeat:
                                  ├─ x held: tilt = joystick                 the same safety chain as
                                  └─ calibrating / link lost / hold: idle    the keyboard)
```

## 1. Turning the gripper: "clutch-relative" control

Hold **space** (or the side button G0). At the moment you press it, the laptop takes two
snapshots:

- how the Cardputer is turned right now (`R_dev0`);
- how the gripper is turned right now (`R_tcp0`).

While you keep holding, every packet it works out **how far the Cardputer has turned since
the snapshot** and turns the gripper by that same amount, starting from *its* snapshot:

```
turn since grab  dR = R_dev · R_dev0⁻¹
gripper target   R  = dR · R_tcp0
```

Why not just copy the Cardputer's angle onto the gripper? Because then the arm would leap
the instant you pressed the key (your hand is never held exactly like the gripper), and
you'd run out of wrist range quickly. With the clutch it works like a computer mouse: let go,
move your hand somewhere comfortable, grab again, and carry on. The arm never jumps on a grab,
and **letting go stops the arm where it is**.

A few details:

- **Heading.** The Cardputer can sense tilt from gravity, but it has no compass, so it can't
  tell which way is "forward". Forward is whatever its screen end pointed at the last time
  you pressed **z**. So: lay it flat, point the **screen end the same way the robot faces**
  (the robot's +x, away from its base) and press `z`. If you'd rather stand facing the robot,
  use `--heading-offset 180` so left and right aren't swapped. Pressing `z` while holding
  space is safe: it just counts as a fresh grab.
- **Smoothing.** The target follows your hand through a tiny low-pass filter (60 ms), which
  takes the jitter out of shaky hands without feeling laggy.
- **Calibrating.** For the first couple of seconds after boot (or after `b`) the orientation
  isn't trustworthy, so the laptop ignores the keys until it's done.
- **Finer control.** `--rotation-scale 0.5` turns the gripper half as much as your hand.

## 2. Moving the gripper with only an IMU (experimental)

This part is genuinely hard, and Phase 4 (camera hand tracking) is the real fix. Here's why:
an accelerometer measures acceleration. To get position you integrate **twice**
(acceleration → speed → distance). Any small error, say the filter thinks the device is tilted
1°, leaks 0.17 m/s² of gravity into the reading, which after 10 s of double integration is
about 8 m of fake motion. So we try two ways around that, and you get to compare.

### Option A: tilt as joystick (`--translation tilt`, the default)

Hold **x** instead of space. The Cardputer now works like a joystick, measured from how you
held it when you pressed x:

| Tip it... | Gripper moves |
|---|---|
| front (screen end) down / up | forward / back |
| right side down / left side down | right / left |
| twist counter-clockwise / clockwise (seen from above) | up / down |

Under 6° does nothing (so a shaky hand holds still). Speed then ramps up to full (the speed
slider in the browser) at 30°. The gripper's angle stays frozen while x is held. This uses
only the orientation, which the IMU measures well, so it doesn't drift at all.

### Option B: push to nudge (`--translation push`)

Hold **space** and make a quick, deliberate push with your hand. The gripper moves about as
far as your hand did. How it avoids the drift problem:

1. Turn the accelerometer reading into the world frame and subtract gravity. What's left is
   your hand actually speeding up or slowing down.
2. Ignore anything under 0.8 m/s² (tremor, filter error). A bigger reading starts a "push".
3. During the push, integrate everything. A push is "speed up, then slow down", and both
   halves are needed to get the distance right.
4. When the readings have been quiet for 0.12 s, the hand has stopped. Set the speed to
   exactly zero. This "zero-velocity update" is the trick step counters use. Any error from
   that push is thrown away instead of piling up.

One grab can move at most 25 cm, and the arm follows at its normal speed limit. Expect it to
be roughly right for quick pushes and to undershoot slow ones (a slow push never crosses
the 0.8 m/s² line).

## 3. Safety

Nothing here can bypass the Phase 1 chain. The leader only produces a target, and the
controller still clamps it to the workspace box, keeps it on a 5 cm / 0.5 rad leash from the
real gripper, and the IK caps speed and joint limits. The leader lets go (and re-grabs fresh
afterwards) whenever packets stop (heartbeat), hold is on, or the device is calibrating.
While the Cardputer is steering, the laptop's motion keys are ignored, so two inputs never
fight.

## 4. Trying it

Everything runs from the repo folder with the virtual environment on:

```bash
cd ~/Cardputer-6-DOF-Leader
source .venv/bin/activate
git pull
pip install -e ".[dev]"
pytest
```

**Without the Cardputer** (fake device in the background), then open http://localhost:8080:

```bash
cardputer-teleop --fake tilt                          # x held: forward, right, then up
cardputer-teleop --fake push --translation push       # pushes forward, left, up, then back
cardputer-teleop --fake rock --translation off        # space held 2 s of every 4: gripper turns
```

**With the Cardputer.** Re-flash first, because the new `x` key lives in the firmware:

```bash
cd ~/Cardputer-6-DOF-Leader/firmware/cardputer_leader
pio run -t upload
cd ~/Cardputer-6-DOF-Leader
cardputer-teleop --cardputer                          # tilt-joystick translation
cardputer-teleop --cardputer --translation push       # push-to-nudge instead
```

Then, in order:

1. Wait for calibration to finish (keep it still). Point the screen end the way the robot
   faces and press `z`.
2. Hold space and slowly turn the Cardputer: tip it forward, roll it, twist it. The gripper in
   the browser should turn the same way. Let go, re-grip, carry on.
3. Hold x and tip it. Check the gripper goes the direction in the table above.
4. Restart with `--translation push`, hold space and make quick 10 cm pushes. Note how far
   the gripper goes, and whether it creeps when you hold still.

Things worth writing down for the comparison: which translation felt more controllable,
whether directions were ever wrong, and whether anything drifted or felt laggy.
