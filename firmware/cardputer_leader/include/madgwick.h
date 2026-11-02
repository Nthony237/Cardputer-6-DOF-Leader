// Madgwick orientation filter, IMU-only version (gyro + accelerometer, no magnetometer).
//
// Idea in one paragraph: integrating the gyro tells us how the device rotated since the
// last sample, which is smooth and fast but slowly drifts. The accelerometer, when the
// device isn't accelerating much, points at "up", which never drifts but is noisy.
// Each step we rotate by the gyro, then nudge the result a little (gain beta) toward
// the orientation where the measured "up" matches the real "up". That nudge fixes
// roll and pitch drift. Gravity says nothing about heading, so yaw still drifts
// slowly; the zero-yaw key handles that until Phase 4 adds camera tracking.
//
// Same math as cardputer_teleop/orientation.py, which has unit tests.
#pragma once
#include <math.h>

class Madgwick {
 public:
  float beta = 0.04f;
  float w = 1, x = 0, y = 0, z = 0;  // device -> world rotation

  void reset() { w = 1; x = y = z = 0; }

  // gx..gz in rad/s, ax..az in any unit (only the direction is used), dt in s.
  void update(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
    // Rate of change of the quaternion from the gyro: qdot = 0.5 * q * (0, g)
    float qw = 0.5f * (-x * gx - y * gy - z * gz);
    float qx = 0.5f * (w * gx + y * gz - z * gy);
    float qy = 0.5f * (w * gy - x * gz + z * gx);
    float qz = 0.5f * (w * gz + x * gy - y * gx);

    float an = sqrtf(ax * ax + ay * ay + az * az);
    if (an > 1e-6f) {
      ax /= an; ay /= an; az /= an;
      // Gradient-descent step: the direction that best reduces the gap between the
      // gravity we predict from q, and the gravity we measured.
      float fx = 2 * (x * z - w * y) - ax;
      float fy = 2 * (w * x + y * z) - ay;
      float fz = 2 * (0.5f - x * x - y * y) - az;
      float sw = -2 * y * fx + 2 * x * fy;
      float sx = 2 * z * fx + 2 * w * fy - 4 * x * fz;
      float sy = -2 * w * fx + 2 * z * fy - 4 * y * fz;
      float sz = 2 * x * fx + 2 * y * fy;
      float sn = sqrtf(sw * sw + sx * sx + sy * sy + sz * sz);
      if (sn > 1e-9f) {
        qw -= beta * sw / sn; qx -= beta * sx / sn;
        qy -= beta * sy / sn; qz -= beta * sz / sn;
      }
    }
    w += qw * dt; x += qx * dt; y += qy * dt; z += qz * dt;
    float n = sqrtf(w * w + x * x + y * y + z * z);
    w /= n; x /= n; y /= n; z /= n;
  }

  // Heading (rotation about world z) in radians.
  float yaw() const { return atan2f(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); }
};
