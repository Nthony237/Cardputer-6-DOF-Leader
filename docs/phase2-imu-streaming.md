# Phase 2: streaming the Cardputer's orientation (how it works)

Goal: the laptop knows, live and reliably, how the Cardputer is tilted and turned, and which
keys are held. Nothing moves the arm yet. Phase 3 uses this to steer the gripper.

```
 Cardputer ADV                                     MacBook
 ─────────────                                     ───────
 BMI270 (gyro + accel) ─┐
                        ├─ Madgwick filter          CardputerReceiver (background thread)
 keyboard / G0 button ──┘   200 Hz                    ├─ checks magic/version, drops garbage
                              │                       ├─ seq numbers → lost / late packets
                       LeaderPacket, 100 Hz           ├─ heartbeat.beat() per good packet
                       ── Wi-Fi, UDP :9870 ──────────►└─ keeps the newest packet
                                                            │
                                         cardputer-imu-view: block that turns with the device
                                         cardputer-teleop --cardputer: arm holds if link drops
```

## 1. Sensing orientation (firmware)

The BMI270 has two sensors:

- a **gyroscope**, which measures how fast the device is rotating (degrees per second) around
  its three axes. Adding up those small rotations 200 times a second tracks orientation
  smoothly, but every sensor has a small error ("bias"), so the estimate slowly drifts;
- an **accelerometer**, which measures acceleration. When you're not moving it much, that is
  mostly gravity, so it tells you which way is down. It doesn't drift, but it's noisy and is
  fooled whenever you move your hand quickly.

The **Madgwick filter** combines the two. Each step it rotates by what the gyro says, then
nudges the result a little toward "the orientation where gravity points the way the
accelerometer says". The size of that nudge is `beta`. It is large for the first 2 s after
boot, so the filter snaps to the true tilt, then small, so hand jitter doesn't leak in.

Two extra tricks:

- **Gyro bias calibration.** At boot (and when you press `b`) the device averages the gyro for
  1.5 s while still. That average is the bias, and it's subtracted from every later reading.
  If you move during calibration it starts over.
- **Yaw (heading) can't be corrected.** Gravity tells you tilt but not which way you're
  facing, and the Cardputer has no compass. So heading slowly drifts (a few degrees a
  minute after bias calibration). Press `z` to say "this direction is straight ahead". Phase 4's
  hand tracking will fix heading properly.

The result is a **quaternion** `(w, x, y, z)`: four numbers that describe a 3D rotation without
the "gimbal lock" problems of roll/pitch/yaw angles. It rotates vectors from the device's
frame into the world frame (z up).

## 2. The packet

Every 10 ms the Cardputer sends 56 bytes over UDP (`include/packet.h` ↔ `protocol.py`):
orientation, gyro and accelerometer readings, key flags (clutch, grip close/open,
calibrating), a counter of `z` presses, battery percent, a **sequence number** and the
device clock.

Why UDP and not TCP? TCP resends lost data, so one dropped packet holds up all the newer ones
behind it. For control you want the newest reading, so losing one is fine. Old ones are
useless. The sequence number lets the laptop count losses and throw away late packets.

## 3. The receiver and the heartbeat (laptop)

`CardputerReceiver` listens on UDP port 9870 in a background thread. For every packet it:

1. rejects anything that isn't ours (wrong size, magic bytes, version, NaNs);
2. compares the sequence number with the newest one: a gap means lost packets, an older
   number means a late packet, which is dropped. A big jump back in the device clock means
   the Cardputer rebooted, so the counting starts over;
3. stores it as `latest` and calls `heartbeat.beat()`.

The **heartbeat** is the Phase 1 safety watchdog. The controller only moves the arm while a
beat arrived within the last 0.25 s. So if Wi-Fi drops, the battery dies or you walk out of
range, the arm **holds still** on its own, and a duplicate or garbage packet can't keep it
alive.

## 4. Trying it

```bash
pip install -e ".[dev]"

cardputer-imu-view --fake           # no hardware: a fake Cardputer rocks around
cardputer-imu-view                  # the real one (after flashing)
cardputer-teleop --cardputer        # arm sim; it only moves while the Cardputer is connected

cardputer-fake --drop 0.2           # in a second terminal: 20% packet loss
cardputer-fake --stall-after 5 --stall-for 2   # goes silent: watch the link go grey / arm hold
```

Then open http://localhost:8080.

**macOS firewall:** the first time, macOS may ask whether Python may "accept incoming network
connections". Click **Allow**, or the Cardputer's packets never arrive.

## 5. Things to check with the real device

- **Axes.** Lay it flat, screen up: roll and pitch should read near 0. Tilt the right side
  down, then the front down, and check that the on-screen block turns the same way. The IMU's axes may
  not line up with the case. If they don't, note which way it's wrong; the fix is one line
  (`M5.Imu.setAxisOrder...` in the firmware).
- **Drift.** Leave it still for a minute and watch yaw. A few degrees is expected. Tens of
  degrees means the bias calibration was disturbed, so press `b` with it on the table.
- **Link quality.** The viewer shows rate and loss. Expect about 100 Hz and under 1% loss on
  a decent network.
