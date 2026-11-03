// The UDP packet the Cardputer sends, about 100 times a second.
// Must match cardputer_teleop/protocol.py byte for byte (little-endian, no padding).
#pragma once
#include <stdint.h>

#define PACKET_MAGIC "CPTL"
#define PACKET_VERSION 1

// Bits of LeaderPacket.flags
#define FLAG_CLUTCH (1 << 0)       // clutch held
#define FLAG_GRIP_CLOSE (1 << 1)   // close-gripper key held
#define FLAG_GRIP_OPEN (1 << 2)    // open-gripper key held
#define FLAG_CALIBRATING (1 << 3)  // orientation not trustworthy yet (boot / recalibration)
#define FLAG_TILT_MOVE (1 << 4)    // tilt-as-joystick key held (Phase 3)

struct __attribute__((packed)) LeaderPacket {
  char magic[4];             // "CPTL", so the receiver can ignore stray packets
  uint8_t version;           // PACKET_VERSION
  uint8_t flags;             // FLAG_* bits above
  uint8_t yaw_reset_count;   // +1 every time the zero-yaw key is pressed
  uint8_t battery_pct;       // 0-100, 255 = unknown
  uint32_t seq;              // +1 every packet: the laptop uses gaps to count lost packets
  uint32_t t_ms;             // device clock (millis())
  float quat[4];             // orientation, w x y z, device frame -> world frame (z up)
  float gyro[3];             // rad/s, bias removed, device frame
  float accel[3];            // m/s^2, device frame
};

static_assert(sizeof(LeaderPacket) == 56, "packet layout changed: update protocol.py too");
