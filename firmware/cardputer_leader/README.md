# Cardputer ADV leader firmware

Reads the BMI270 motion sensor, works out the device's orientation, and streams it with your
key presses to the laptop over Wi-Fi (UDP, 100 packets per second).

## Keys

| Key | What it does |
|---|---|
| `space` or the side button (G0) | **Clutch**, hold it. Phase 3 will only move the arm while it's held. |
| `c` / `v` | Hold to close / open the gripper |
| `z` | Zero yaw: "the way I'm pointing now is straight ahead" |
| `b` | Re-measure the gyro bias (put the Cardputer down on a table first) |

The screen shows the Wi-Fi state, where packets are going, the send rate, roll/pitch/yaw, and
which keys are held.

## Flash it from a MacBook

1. **Install PlatformIO** (the build tool for the ESP32). Either:
   - VS Code with the "PlatformIO IDE" extension, or
   - the command line: `brew install platformio` (gives you the `pio` command).
2. **Add your Wi-Fi.** In this folder:
   ```bash
   cp include/secrets.example.h include/secrets.h
   ```
   Open `include/secrets.h` and fill in the Wi-Fi name, password, and your laptop's IP. To get
   the IP, run `ipconfig getifaddr en0` in Terminal. `secrets.h` is in `.gitignore`, so it
   never gets committed. The laptop and the Cardputer must be on the **same** 2.4 GHz network.
   Home Wi-Fi or a phone hotspot works. eduroam and "click to accept" networks don't.
3. **Plug the Cardputer in** with a USB-C data cable and switch it on.
4. **Build and flash** (the first build downloads the ESP32 toolchain, so it takes a few minutes):
   ```bash
   cd firmware/cardputer_leader
   pio run -t upload
   ```
   In VS Code: open this folder, then click the → "Upload" arrow in the blue status bar.
   If the upload can't find the device, put it in download mode: hold the **G0** button while
   you plug in the cable (or while you press reset), then try again.
5. **Watch the log** (optional): `pio device monitor`. It prints the measured gyro bias at
   boot. Ctrl+C to exit.

After flashing, the Cardputer boots, joins Wi-Fi and says "Hold still: measuring gyro...". Set it
on the table for 2 seconds. Then it starts sending.

## Files

| File | What's in it |
|---|---|
| `src/main.cpp` | Setup (Wi-Fi, IMU) and the main loop: read sensor → filter → keys → send → screen |
| `include/madgwick.h` | The orientation filter (same math as `cardputer_teleop/orientation.py`, which is tested) |
| `include/packet.h` | The 56-byte packet layout (must match `cardputer_teleop/protocol.py`; a test checks) |
| `include/config.h` | Rates, filter gain, key bindings, UDP port. Safe to commit. |
| `include/secrets.example.h` | Template for your Wi-Fi details. Copy it to `secrets.h`. |
