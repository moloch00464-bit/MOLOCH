/*
 * M.O.L.O.C.H. — ReSpeaker Lite WiFi-Mic Konfiguration
 * ======================================================
 * Alle konfigurierbaren Parameter an einer Stelle.
 * WiFi-Credentials NICHT in main.cpp!
 */

#ifndef CONFIG_H
#define CONFIG_H

// =====================================================================
// WiFi — Pi5 Direkt-AP (kein Router)
// =====================================================================
#define WIFI_SSID          "MOLOCH_DIRECT"
#define WIFI_PASS          "M0l0ch2026!"
#define WIFI_STATIC_IP     "10.42.0.2"
#define WIFI_GATEWAY       "10.42.0.1"
#define WIFI_SUBNET        "255.255.255.0"
#define WIFI_DNS           "10.42.0.1"
#define WIFI_TIMEOUT_MS    15000
#define WIFI_RECONNECT_MS  5000

// =====================================================================
// I2S Pins — ReSpeaker Lite (ESP32-S3 <-> XMOS XU316)
// Quelle: Seeed Wiki + ESPHome formatBCE Config (VERIFIZIERT)
// ACHTUNG: BCLK=GPIO8, LRCK=GPIO7 — NICHT vertauschen!
// =====================================================================
#define I2S_BCLK_PIN       8     // Bit Clock (SCK) — XMOS Master Output
#define I2S_LRCK_PIN       7     // Word Select (WS) — XMOS Master Output
#define I2S_DIN_PIN        44    // Serial Data In (Mic von XMOS → ESP32)
#define I2S_DOUT_PIN       43    // Serial Data Out (Speaker ESP32 → XMOS)
#define I2S_MCLK_PIN       9     // Master Clock (ESP32 → XMOS, 12.288 MHz)

// =====================================================================
// Audio
// =====================================================================
#define I2S_SAMPLE_RATE    48000
#define I2S_MCLK_MULT      256   // MCLK = 48000 * 256 = 12.288 MHz
#define I2S_BIT_DEPTH       32    // XMOS liefert 32-bit I2S
#define I2S_DMA_FRAMES     480   // 10ms bei 48kHz (sauber teilbar durch 3)
#define I2S_DMA_COUNT        8    // 8 DMA-Puffer = 80ms Latenz (weniger Underruns)

// =====================================================================
// UDP Audio-Streaming — ESP32 sendet an Pi
// =====================================================================
#define PI_IP              "10.42.0.1"
#define UDP_PORT_16K       12345  // 16kHz Mono S16_LE (fuer Whisper STT)
#define UDP_PORT_48K       12346  // 48kHz Stereo S16_LE (Stimmbiometrie)

// =====================================================================
// LED — WS2812 NeoPixel
// =====================================================================
#define LED_PIN             1     // GPIO 1 auf ReSpeaker Lite
#define LED_BRIGHTNESS     50     // Max Helligkeit (0-255)
#define LED_UDP_PORT     8888     // UDP-Port fuer LED-Kommandos vom Pi

// =====================================================================
// HTTP + OTA
// =====================================================================
#define HTTP_PORT           80
#define OTA_PASSWORD       "m0l0ch_ota"
#define OTA_HOSTNAME       "moloch-mic"

#endif // CONFIG_H
