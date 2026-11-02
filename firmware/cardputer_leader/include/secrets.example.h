// Copy this file to secrets.h (same folder) and fill in your values.
// secrets.h is listed in .gitignore, so your Wi-Fi password never reaches GitHub.
#pragma once

// 2.4 GHz network with a plain password (home Wi-Fi or a phone hotspot).
// The ESP32 can't join 5 GHz-only networks, eduroam, or networks with a login page.
#define WIFI_SSID "your-network-name"
#define WIFI_PASSWORD "your-password"

// Your laptop's IP on that network. On a Mac: System Settings > Wi-Fi > Details,
// or run `ipconfig getifaddr en0` in Terminal.
// Leave it as "" to broadcast to the whole network instead (handy when the IP keeps
// changing, but some routers and hotspots drop broadcast packets).
#define LAPTOP_IP ""
