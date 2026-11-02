// Cardputer ADV leader firmware (Phase 2): IMU orientation + keys -> UDP -> laptop.
//
// What happens, in order:
//   1. Boot: join Wi-Fi, then measure the gyro's bias while you hold the device still.
//   2. Loop at FILTER_HZ: read the BMI270, feed the Madgwick filter.
//   3. Every 1/SEND_HZ s: send one LeaderPacket (orientation, keys, sequence number).
//      The laptop treats the packet stream as a heartbeat: if packets stop, the arm holds.
//   4. A few times a second: redraw the status screen.
#include <M5Cardputer.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "config.h"
#include "madgwick.h"
#include "packet.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Missing include/secrets.h: copy include/secrets.example.h to include/secrets.h and fill in your Wi-Fi."
#endif

static constexpr float DEG2RAD = 0.01745329252f;
static constexpr float G = 9.80665f;

static Madgwick filter;
static WiFiUDP udp;
static IPAddress target_ip;
static bool use_broadcast = false;

static float gyro_bias[3] = {0, 0, 0};  // rad/s
static float yaw_offset = 0;            // rad, set by the zero-yaw key
static uint8_t yaw_reset_count = 0;
static uint32_t seq = 0;
static uint32_t sent = 0, send_fail = 0;

// Calibration state: we average the gyro while the device is still.
static bool calibrating = true;
static uint32_t calib_start_ms = 0;
static uint32_t calib_n = 0;
static double calib_sum[3] = {0, 0, 0};
static float calib_last_acc[3] = {0, 0, 0};
static uint32_t filter_start_ms = 0;

static float last_gyro[3] = {0, 0, 0}, last_acc[3] = {0, 0, 1};

static void start_calibration() {
  calibrating = true;
  calib_start_ms = millis();
  calib_n = 0;
  calib_sum[0] = calib_sum[1] = calib_sum[2] = 0;
}

// Feed one gyro sample (rad/s, bias not removed). Restarts if the device moves.
static void calibration_step(const float g[3], const float a[3]) {
  float jolt = fabsf(a[0] - calib_last_acc[0]) + fabsf(a[1] - calib_last_acc[1]) +
               fabsf(a[2] - calib_last_acc[2]);
  memcpy(calib_last_acc, a, sizeof(calib_last_acc));
  float spin = fabsf(g[0]) + fabsf(g[1]) + fabsf(g[2]);
  if (calib_n > 0 && (jolt > 0.05f || spin > 0.3f)) {  // moved: start over
    start_calibration();
    return;
  }
  for (int i = 0; i < 3; i++) calib_sum[i] += g[i];
  calib_n++;
  if (millis() - calib_start_ms >= (uint32_t)(GYRO_CALIB_SECONDS * 1000) && calib_n > 50) {
    for (int i = 0; i < 3; i++) gyro_bias[i] = calib_sum[i] / calib_n;
    calibrating = false;
    filter.reset();
    filter_start_ms = millis();
    Serial.printf("gyro bias (deg/s): %.3f %.3f %.3f\n", gyro_bias[0] / DEG2RAD,
                  gyro_bias[1] / DEG2RAD, gyro_bias[2] / DEG2RAD);
  }
}

// Orientation we send = filter orientation with the heading offset removed:
// q_out = Rz(-yaw_offset) * q
static void output_quat(float out[4]) {
  float c = cosf(-yaw_offset / 2), s = sinf(-yaw_offset / 2);
  out[0] = c * filter.w - s * filter.z;
  out[1] = c * filter.x - s * filter.y;
  out[2] = c * filter.y + s * filter.x;
  out[3] = c * filter.z + s * filter.w;
}

static bool key_down(char c) { return M5Cardputer.Keyboard.isKeyPressed(c); }

static void send_packet(uint8_t flags) {
  LeaderPacket p;
  memcpy(p.magic, PACKET_MAGIC, 4);
  p.version = PACKET_VERSION;
  p.flags = flags;
  p.yaw_reset_count = yaw_reset_count;
  int bat = M5Cardputer.Power.getBatteryLevel();
  p.battery_pct = (bat >= 0 && bat <= 100) ? bat : 255;
  p.seq = seq++;
  p.t_ms = millis();
  output_quat(p.quat);
  for (int i = 0; i < 3; i++) {
    p.gyro[i] = last_gyro[i];
    p.accel[i] = last_acc[i] * G;
  }
  if (WiFi.status() != WL_CONNECTED) return;
  IPAddress dst = use_broadcast ? WiFi.broadcastIP() : target_ip;
  bool ok = udp.beginPacket(dst, LAPTOP_PORT) && udp.write((const uint8_t*)&p, sizeof(p)) == sizeof(p) &&
            udp.endPacket();
  ok ? sent++ : send_fail++;
}

static void draw_screen(uint8_t flags, float send_rate) {
  static M5Canvas canvas(&M5Cardputer.Display);
  if (canvas.width() == 0) canvas.createSprite(M5Cardputer.Display.width(), M5Cardputer.Display.height());
  canvas.fillSprite(BLACK);
  canvas.setTextSize(1);
  canvas.setTextColor(WHITE);
  canvas.setCursor(0, 0);
  canvas.setFont(&fonts::Font2);

  if (WiFi.status() == WL_CONNECTED) {
    canvas.setTextColor(GREEN);
    canvas.printf("WiFi %s  me %s\n", WIFI_SSID, WiFi.localIP().toString().c_str());
    canvas.setTextColor(WHITE);
    canvas.printf("-> %s:%d  %.0f Hz\n",
                  use_broadcast ? "broadcast" : target_ip.toString().c_str(), LAPTOP_PORT, send_rate);
  } else {
    canvas.setTextColor(RED);
    canvas.printf("WiFi: connecting to %s...\n\n", WIFI_SSID);
    canvas.setTextColor(WHITE);
  }

  if (calibrating) {
    canvas.setTextColor(YELLOW);
    canvas.println("Hold still: measuring gyro...");
    canvas.setTextColor(WHITE);
  } else {
    float q[4];
    output_quat(q);
    float roll = atan2f(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
    float pitch = asinf(fmaxf(-1.f, fminf(1.f, 2 * (q[0] * q[2] - q[3] * q[1]))));
    float yaw = atan2f(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
    canvas.printf("roll %6.1f pitch %6.1f\nyaw  %6.1f (z zeroes)\n", roll / DEG2RAD, pitch / DEG2RAD,
                  yaw / DEG2RAD);
  }
  canvas.setTextColor((flags & FLAG_CLUTCH) ? GREEN : DARKGREY);
  canvas.print("CLUTCH ");
  canvas.setTextColor((flags & FLAG_GRIP_CLOSE) ? GREEN : DARKGREY);
  canvas.print("CLOSE ");
  canvas.setTextColor((flags & FLAG_GRIP_OPEN) ? GREEN : DARKGREY);
  canvas.println("OPEN");
  canvas.setTextColor(DARKGREY);
  canvas.println("space/G0 clutch  c/v grip  b recal");
  canvas.pushSprite(0, 0);
}

void setup() {
  auto cfg = M5.config();
  M5Cardputer.begin(cfg, true);
  M5Cardputer.Display.setRotation(1);
  M5Cardputer.Display.setBrightness(80);
  Serial.begin(115200);

  if (!M5.Imu.isEnabled()) {
    M5Cardputer.Display.setTextColor(RED);
    M5Cardputer.Display.drawString("IMU not found (needs Cardputer ADV)", 4, 4);
    Serial.println("IMU not found");
    while (true) delay(1000);
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // Wi-Fi power saving adds 100+ ms of latency spikes
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  use_broadcast = strlen(LAPTOP_IP) == 0;
  if (!use_broadcast && !target_ip.fromString(LAPTOP_IP)) {
    Serial.println("LAPTOP_IP is not a valid IP address, falling back to broadcast");
    use_broadcast = true;
  }
  udp.begin(LAPTOP_PORT);
  start_calibration();
}

void loop() {
  static uint32_t last_filter_us = micros();
  static uint32_t last_send_us = 0, last_screen_ms = 0, rate_ms = 0, rate_sent = 0;
  static float send_rate = 0;
  static bool zero_was_down = false, recal_was_down = false;

  uint32_t now_us = micros();
  uint32_t dt_us = now_us - last_filter_us;
  if (dt_us < 1000000 / FILTER_HZ) return;
  last_filter_us = now_us;
  float dt = fminf(dt_us * 1e-6f, 0.05f);  // cap so a hiccup can't cause a big jump

  M5Cardputer.update();  // buttons + keyboard
  M5.Imu.update();
  float a[3], g[3];
  M5.Imu.getAccel(&a[0], &a[1], &a[2]);  // in g
  M5.Imu.getGyro(&g[0], &g[1], &g[2]);   // in deg/s
  for (int i = 0; i < 3; i++) g[i] *= DEG2RAD;

  if (calibrating) {
    calibration_step(g, a);
  } else {
    for (int i = 0; i < 3; i++) g[i] -= gyro_bias[i];
    bool startup = (millis() - filter_start_ms) < (uint32_t)(STARTUP_SECONDS * 1000);
    filter.beta = startup ? MADGWICK_BETA_STARTUP : MADGWICK_BETA;
    filter.update(g[0], g[1], g[2], a[0], a[1], a[2], dt);
  }
  memcpy(last_gyro, g, sizeof(last_gyro));
  memcpy(last_acc, a, sizeof(last_acc));

  // Keys. Clutch/gripper are "held" states sent every packet; zero-yaw and
  // recalibrate act once per press.
  bool zero_down = key_down(KEY_ZERO_YAW), recal_down = key_down(KEY_RECALIBRATE);
  if (zero_down && !zero_was_down && !calibrating) {
    yaw_offset = filter.yaw();
    yaw_reset_count++;
  }
  if (recal_down && !recal_was_down) start_calibration();
  zero_was_down = zero_down;
  recal_was_down = recal_down;

  uint8_t flags = 0;
  if (key_down(KEY_CLUTCH) || M5Cardputer.BtnA.isPressed()) flags |= FLAG_CLUTCH;
  if (key_down(KEY_GRIP_CLOSE)) flags |= FLAG_GRIP_CLOSE;
  if (key_down(KEY_GRIP_OPEN)) flags |= FLAG_GRIP_OPEN;
  bool settling = !calibrating && (millis() - filter_start_ms) < (uint32_t)(STARTUP_SECONDS * 1000);
  if (calibrating || settling) flags |= FLAG_CALIBRATING;

  if (now_us - last_send_us >= 1000000 / SEND_HZ) {
    last_send_us = now_us;
    send_packet(flags);
  }

  uint32_t now_ms = millis();
  if (now_ms - rate_ms >= 1000) {
    send_rate = (sent - rate_sent) * 1000.0f / (now_ms - rate_ms);
    rate_sent = sent;
    rate_ms = now_ms;
  }
  if (now_ms - last_screen_ms >= 1000 / SCREEN_HZ) {
    last_screen_ms = now_ms;
    draw_screen(flags, send_rate);
  }
}
