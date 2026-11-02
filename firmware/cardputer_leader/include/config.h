// Settings that are safe to commit. Wi-Fi details live in secrets.h (not committed).
#pragma once

// UDP port the laptop receiver listens on (cardputer-imu-view --udp-port).
#define LAPTOP_PORT 9870

// Loop rates. The filter runs faster than we send, so each packet carries a
// well-settled orientation.
#define FILTER_HZ 200
#define SEND_HZ 100
#define SCREEN_HZ 5

// Madgwick filter gain. Higher trusts the accelerometer more: corrects tilt drift
// faster but lets hand shaking leak in. BETA_STARTUP is used for the first seconds
// so the filter snaps to the true tilt quickly after boot.
#define MADGWICK_BETA 0.04f
#define MADGWICK_BETA_STARTUP 2.0f
#define STARTUP_SECONDS 2.0f

// How long the device must sit still at boot to measure the gyro's bias.
#define GYRO_CALIB_SECONDS 1.5f

// Keys. CLUTCH is held, the others are one press each.
// The side button (G0) also works as the clutch.
#define KEY_CLUTCH ' '        // space: hold to move the arm (Phase 3+)
#define KEY_GRIP_CLOSE 'c'    // hold to close the gripper
#define KEY_GRIP_OPEN 'v'     // hold to open the gripper
#define KEY_ZERO_YAW 'z'      // press: current heading becomes "straight ahead"
#define KEY_RECALIBRATE 'b'   // press with the device still: re-measure gyro bias
