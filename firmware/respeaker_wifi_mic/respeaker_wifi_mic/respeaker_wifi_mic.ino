/*
 * M.O.L.O.C.H. — ReSpeaker Lite WiFi-Mic Firmware v2.1
 * =====================================================
 * Arduino/ESP-IDF Firmware fuer ESP32-S3 auf ReSpeaker Lite Board.
 * Ersetzt MicroPython — loest I2S Slave + MCLK Problem.
 *
 * Architektur:
 *   ESP32 gibt MCLK (12.288 MHz) an XMOS → XMOS generiert BCLK/WS
 *   ESP32 ist I2S SLAVE → empfaengt Clocks + Audio vom XMOS
 *   Audio wird per UDP an den Pi gestreamt
 *
 * Kompiliert mit Arduino ESP32 Core 2.0.14 (ESP-IDF 4.4, Legacy I2S API)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include "driver/i2s.h"
#include "config.h"


// =====================================================================
// Globaler State
// =====================================================================

// Audio-Modus: 0=16kHz Mono, 1=48kHz Stereo
static volatile int audio_mode = 0;
static volatile bool streaming = true;

// UDP
static WiFiUDP udp_audio;
static WiFiUDP udp_led;

// HTTP
static WebServer http_server(HTTP_PORT);

// LED (neopixelWrite() ist in ESP32 Core 2.0+ eingebaut, keine Library noetig)
static uint8_t led_r = 0, led_g = 0, led_b = 0;
static String led_mode_str = "statisch";
static int led_speed = 500;
static int led_phase = 0;
static unsigned long led_last_ms = 0;

// WiFi
static unsigned long wifi_last_check = 0;

// Audio-Task Handle
static TaskHandle_t audio_task_handle = NULL;

// Sequenznummer fuer UDP-Pakete (Paketverlust-Erkennung auf Pi-Seite)
static uint32_t udp_seq_num = 0;

// DMA-Puffer: 480 Frames * 2 Kanaele * 4 Bytes (32-bit) = 3840 Bytes
static uint8_t dma_buf[I2S_DMA_FRAMES * 2 * 4];

// Ausgabe-Puffer
static int16_t out_16k[I2S_DMA_FRAMES / 3 + 1];   // 160 Samples max
static int16_t out_48k[I2S_DMA_FRAMES * 2];        // 960 Samples max


// =====================================================================
// I2S Setup — Legacy API (ESP-IDF 4.4, Slave + MCLK)
// =====================================================================

static bool i2s_init() {
    i2s_config_t i2s_config = {};
    i2s_config.mode = (i2s_mode_t)(I2S_MODE_SLAVE | I2S_MODE_RX);
    i2s_config.sample_rate = I2S_SAMPLE_RATE;
    i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
    i2s_config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
    i2s_config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
    i2s_config.dma_buf_count = I2S_DMA_COUNT;
    i2s_config.dma_buf_len = I2S_DMA_FRAMES;
    i2s_config.use_apll = false;         // ESP32-S3 hat kein APLL
    i2s_config.tx_desc_auto_clear = false;
    i2s_config.fixed_mclk = I2S_SAMPLE_RATE * I2S_MCLK_MULT; // 12.288 MHz

    esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[I2S] Driver Install fehlgeschlagen: 0x%x\n", err);
        return false;
    }

    i2s_pin_config_t pin_config = {};
    pin_config.mck_io_num = I2S_MCLK_PIN;
    pin_config.bck_io_num = I2S_BCLK_PIN;
    pin_config.ws_io_num = I2S_LRCK_PIN;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;  // Kein TX vorerst
    pin_config.data_in_num = I2S_DIN_PIN;

    err = i2s_set_pin(I2S_NUM_0, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[I2S] Pin Config fehlgeschlagen: 0x%x\n", err);
        return false;
    }

    // MCLK-Frequenz Info
    float mclk_mhz = (float)(I2S_SAMPLE_RATE * I2S_MCLK_MULT) / 1000000.0f;
    Serial.printf("[I2S] Slave RX aktiv\n");
    Serial.printf("[I2S] MCLK=%.3f MHz (Ziel) auf GPIO%d\n", mclk_mhz, I2S_MCLK_PIN);
    Serial.printf("[I2S] BCLK=GPIO%d (Input), WS=GPIO%d (Input), DIN=GPIO%d\n",
                  I2S_BCLK_PIN, I2S_LRCK_PIN, I2S_DIN_PIN);
    Serial.printf("[I2S] DMA: %d Frames x %d Puffer = %dms Latenz\n",
                  I2S_DMA_FRAMES, I2S_DMA_COUNT,
                  I2S_DMA_FRAMES * I2S_DMA_COUNT * 1000 / I2S_SAMPLE_RATE);

    return true;
}


// =====================================================================
// Audio-Verarbeitung
// =====================================================================

// 48kHz Stereo 32-bit → 16kHz Mono 16-bit
// Kanal 0 (Links = ASR/Beamformed), jeden 3. Frame, 32→16 bit
static int process_to_16k(const uint8_t* src, size_t bytes, int16_t* dst) {
    const int frame_size = 2 * 4; // 2 Kanaele * 4 Bytes
    int num_frames = bytes / frame_size;
    int out_idx = 0;

    for (int i = 0; i < num_frames; i += 3) {
        int offset = i * frame_size;
        // Kanal 0 (Links = ASR) als 32-bit signed
        int32_t sample;
        memcpy(&sample, src + offset, 4);
        dst[out_idx++] = (int16_t)(sample >> 16);
    }
    return out_idx;
}

// 48kHz Stereo 32-bit → 48kHz Stereo 16-bit
static int process_to_48k(const uint8_t* src, size_t bytes, int16_t* dst) {
    int num_samples = bytes / 4;
    for (int i = 0; i < num_samples; i++) {
        int32_t sample;
        memcpy(&sample, src + i * 4, 4);
        dst[i] = (int16_t)(sample >> 16);
    }
    return num_samples;
}


// =====================================================================
// Audio-Task (FreeRTOS, Core 1, hohe Prioritaet)
// =====================================================================

static void audio_task(void* param) {
    Serial.println("[AUDIO] Task gestartet auf Core " + String(xPortGetCoreID()));

    // Diagnose-Variablen
    bool diag_done = false;
    unsigned long diag_start = millis();
    uint32_t total_bytes_read = 0;
    int32_t max_amplitude = 0;
    bool has_nonzero = false;

    while (true) {
        if (!streaming) {
            vTaskDelay(pdMS_TO_TICKS(50));
            diag_done = false;
            diag_start = millis();
            max_amplitude = 0;
            has_nonzero = false;
            total_bytes_read = 0;
            continue;
        }

        // I2S lesen (blockiert bis DMA-Puffer voll oder Timeout)
        size_t bytes_read = 0;
        esp_err_t err = i2s_read(I2S_NUM_0, dma_buf, sizeof(dma_buf),
                                  &bytes_read, pdMS_TO_TICKS(1000));

        if (err != ESP_OK || bytes_read == 0) {
            // Timeout = kein BCLK vom XMOS
            static unsigned long last_warn = 0;
            if (millis() - last_warn > 5000) {
                Serial.println("[AUDIO] Timeout — kein I2S-Signal vom XMOS");
                Serial.println("[AUDIO] Pruefen: XMOS I2S-Firmware aktiv? MCLK auf GPIO9?");
                last_warn = millis();
            }
            continue;
        }

        total_bytes_read += bytes_read;

        // Diagnose (erste 3 Sekunden nach Start)
        if (!diag_done) {
            for (size_t i = 0; i + 3 < bytes_read; i += 4) {
                int32_t s;
                memcpy(&s, dma_buf + i, 4);
                if (s != 0) has_nonzero = true;
                int32_t a = abs(s >> 16);
                if (a > max_amplitude) max_amplitude = a;
            }
            if (millis() - diag_start > 3000) {
                Serial.printf("[AUDIO] === DIAGNOSE ===\n");
                Serial.printf("[AUDIO] Daten:     %s\n", has_nonzero ? "JA (Non-Zero)" : "NUR NULLEN!");
                Serial.printf("[AUDIO] Max-Amp:   %d %s\n", (int)max_amplitude,
                              max_amplitude > 100 ? "(OK)" : "(ZU NIEDRIG!)");
                Serial.printf("[AUDIO] Bytes:     %u in 3s\n", total_bytes_read);
                Serial.printf("[AUDIO] Bitrate:   %.1f kbps\n",
                              total_bytes_read * 8.0f / 3000.0f);
                Serial.printf("[AUDIO] =================\n");
                diag_done = true;
            }
        }

        // Nicht senden wenn WiFi weg
        if (!WiFi.isConnected()) continue;

        // Verarbeiten + per UDP senden (mit 4-Byte Sequenznummer-Header)
        if (audio_mode == 0) {
            // 16kHz Mono — ein Paket pro DMA-Puffer: [4B seq][320B Audio] = 324 Bytes
            int n = process_to_16k(dma_buf, bytes_read, out_16k);
            if (n > 0) {
                int audio_bytes = n * sizeof(int16_t);
                udp_audio.beginPacket(PI_IP, UDP_PORT_16K);
                udp_audio.write((uint8_t*)&udp_seq_num, 4);  // Sequenznummer (LE)
                udp_audio.write((uint8_t*)out_16k, audio_bytes);
                udp_audio.endPacket();
                udp_seq_num++;
            }
        } else {
            // 48kHz Stereo — in Chunks: [4B seq][960B Audio] = 964 Bytes pro Paket
            int n = process_to_48k(dma_buf, bytes_read, out_48k);
            int total_bytes = n * sizeof(int16_t);
            const int chunk = 960;
            for (int off = 0; off < total_bytes; off += chunk) {
                int send_bytes = total_bytes - off;
                if (send_bytes > chunk) send_bytes = chunk;
                udp_audio.beginPacket(PI_IP, UDP_PORT_48K);
                udp_audio.write((uint8_t*)&udp_seq_num, 4);  // Sequenznummer (LE)
                udp_audio.write((uint8_t*)out_48k + off, send_bytes);
                udp_audio.endPacket();
                udp_seq_num++;
            }
        }
    }
}


// =====================================================================
// WiFi
// =====================================================================

static void wifi_connect() {
    Serial.printf("[WIFI] Verbinde mit '%s'...\n", WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);

    IPAddress ip, gw, sn, dns;
    ip.fromString(WIFI_STATIC_IP);
    gw.fromString(WIFI_GATEWAY);
    sn.fromString(WIFI_SUBNET);
    dns.fromString(WIFI_DNS);
    WiFi.config(ip, gw, sn, dns);

    WiFi.begin(WIFI_SSID, WIFI_PASS);

    // Audio braucht konstante Verbindung — kein PowerSave
    WiFi.setSleep(false);

    unsigned long start = millis();
    while (!WiFi.isConnected() && (millis() - start) < WIFI_TIMEOUT_MS) {
        delay(250);
        Serial.print(".");
    }

    if (WiFi.isConnected()) {
        Serial.printf("\n[WIFI] Verbunden! IP: %s, RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        Serial.println("\n[WIFI] Verbindung fehlgeschlagen!");
    }
}

static void wifi_check() {
    if (millis() - wifi_last_check < WIFI_RECONNECT_MS) return;
    wifi_last_check = millis();
    if (!WiFi.isConnected()) {
        Serial.println("[WIFI] Reconnect...");
        WiFi.reconnect();
    }
}


// =====================================================================
// HTTP-Server
// =====================================================================

static void http_setup() {
    http_server.on("/audio/status", HTTP_GET, []() {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"streaming\":%s,\"mode\":\"%s\",\"rate\":%d,"
            "\"wifi_rssi\":%d,\"ip\":\"%s\",\"uptime_s\":%lu,"
            "\"free_heap\":%u,\"mclk_mhz\":%.3f,\"udp_seq\":%lu,\"fw_version\":\"2.1\"}",
            streaming ? "true" : "false",
            audio_mode == 0 ? "16k_mono" : "48k_stereo",
            audio_mode == 0 ? 16000 : 48000,
            WiFi.RSSI(),
            WiFi.localIP().toString().c_str(),
            millis() / 1000,
            ESP.getFreeHeap(),
            (float)(I2S_SAMPLE_RATE * I2S_MCLK_MULT) / 1000000.0f,
            (unsigned long)udp_seq_num
        );
        http_server.send(200, "application/json", buf);
    });

    http_server.on("/audio/mode", HTTP_POST, []() {
        if (!http_server.hasArg("rate")) {
            http_server.send(400, "text/plain", "Parameter 'rate' fehlt");
            return;
        }
        int rate = http_server.arg("rate").toInt();
        if (rate == 16000) {
            audio_mode = 0;
            http_server.send(200, "text/plain", "Modus: 16kHz Mono (Whisper)");
            Serial.println("[HTTP] Modus -> 16kHz Mono");
        } else if (rate == 48000) {
            audio_mode = 1;
            http_server.send(200, "text/plain", "Modus: 48kHz Stereo");
            Serial.println("[HTTP] Modus -> 48kHz Stereo");
        } else {
            http_server.send(400, "text/plain", "Ungueltig. rate=16000 oder rate=48000");
        }
    });

    http_server.on("/audio/start", HTTP_POST, []() {
        streaming = true;
        http_server.send(200, "text/plain", "Streaming gestartet");
        Serial.println("[HTTP] Streaming START");
    });

    http_server.on("/audio/stop", HTTP_POST, []() {
        streaming = false;
        http_server.send(200, "text/plain", "Streaming gestoppt");
        Serial.println("[HTTP] Streaming STOP");
    });

    http_server.on("/", HTTP_GET, []() {
        http_server.send(200, "text/plain",
            "M.O.L.O.C.H. ReSpeaker WiFi-Mic v2.0\n"
            "Endpoints:\n"
            "  GET  /audio/status\n"
            "  POST /audio/mode?rate=16000|48000\n"
            "  POST /audio/start\n"
            "  POST /audio/stop\n"
        );
    });

    http_server.begin();
    Serial.printf("[HTTP] Server auf Port %d\n", HTTP_PORT);
}


// =====================================================================
// OTA
// =====================================================================

static void ota_setup() {
    ArduinoOTA.setHostname(OTA_HOSTNAME);
    ArduinoOTA.setPassword(OTA_PASSWORD);

    ArduinoOTA.onStart([]() {
        streaming = false;
        Serial.println("[OTA] Update startet...");
    });
    ArduinoOTA.onEnd([]() {
        Serial.println("[OTA] Update fertig — Reboot");
    });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("[OTA] %u%%\r", progress * 100 / total);
    });
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("[OTA] Fehler: %u\n", error);
    });

    ArduinoOTA.begin();
    Serial.println("[OTA] Bereit (Hostname: " OTA_HOSTNAME ")");
}


// =====================================================================
// RGB-LED (WS2812 auf GPIO 1)
// =====================================================================

struct FarbEintrag { const char* name; uint8_t r, g, b; };
static const FarbEintrag FARBEN[] = {
    {"rot",     255, 0,   0  },
    {"gruen",   0,   255, 0  },
    {"blau",    0,   0,   255},
    {"gelb",    255, 180, 0  },
    {"cyan",    0,   255, 255},
    {"magenta", 255, 0,   255},
    {"weiss",   255, 255, 255},
    {"aus",     0,   0,   0  },
    {NULL, 0, 0, 0}
};

static void led_set(uint8_t r, uint8_t g, uint8_t b) {
    float scale = (float)LED_BRIGHTNESS / 255.0f;
    // neopixelWrite() ist in ESP32 Arduino Core eingebaut (GRB-Order intern)
    neopixelWrite(LED_PIN, (uint8_t)(r * scale), (uint8_t)(g * scale), (uint8_t)(b * scale));
}

static void led_parse(const char* cmd) {
    if (strncasecmp(cmd, "LED:", 4) != 0) return;
    cmd += 4;

    char farbe[16] = {0}, modus[16] = {0}, speed[16] = {0};
    sscanf(cmd, "%15s %15s %15s", farbe, modus, speed);

    for (int i = 0; FARBEN[i].name; i++) {
        if (strcasecmp(farbe, FARBEN[i].name) == 0) {
            led_r = FARBEN[i].r;
            led_g = FARBEN[i].g;
            led_b = FARBEN[i].b;
            break;
        }
    }

    led_mode_str = (strlen(modus) > 0) ? modus : "statisch";
    if (strcasecmp(speed, "langsam") == 0) led_speed = 1000;
    else if (strcasecmp(speed, "mittel") == 0) led_speed = 500;
    else if (strcasecmp(speed, "schnell") == 0) led_speed = 150;

    led_phase = 0;
    if (led_mode_str == "statisch") led_set(led_r, led_g, led_b);
}

static void led_update() {
    unsigned long now = millis();
    int interval = (led_mode_str == "blinkend") ? led_speed / 4 : led_speed / 10;
    if ((now - led_last_ms) < (unsigned long)interval) return;
    led_last_ms = now;

    if (led_mode_str == "pulsierend") {
        led_phase = (led_phase + 1) % 20;
        float b = (led_phase < 10) ? led_phase / 10.0f : (20 - led_phase) / 10.0f;
        b = 0.05f + b * 0.95f;
        led_set((uint8_t)(led_r * b), (uint8_t)(led_g * b), (uint8_t)(led_b * b));
    } else if (led_mode_str == "blinkend") {
        led_phase = (led_phase + 1) % 2;
        if (led_phase == 0) led_set(led_r, led_g, led_b);
        else led_set(0, 0, 0);
    }
}

static void led_udp_check() {
    int len = udp_led.parsePacket();
    if (len <= 0) return;
    char buf[128];
    int n = udp_led.read(buf, sizeof(buf) - 1);
    if (n > 0) { buf[n] = '\0'; led_parse(buf); }
}

static void led_test() {
    led_set(255, 0, 0); delay(300);
    led_set(0, 255, 0); delay(300);
    led_set(0, 0, 255); delay(300);
    led_set(0, 0, 0);
    Serial.println("[LED] Test OK");
}


// =====================================================================
// Setup
// =====================================================================

void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("==================================================");
    Serial.println("M.O.L.O.C.H. ReSpeaker WiFi-Mic v2.1 (Arduino)");
    Serial.println("ESP32-S3 + XMOS XU316 I2S Slave Bridge");
    Serial.println("Arduino Core 2.0.14 / ESP-IDF 4.4 / Legacy I2S");
    Serial.println("==================================================");
    Serial.printf("Heap: %u, PSRAM: %u\n", ESP.getFreeHeap(), ESP.getFreePsram());

    // LED (Pin als Output, neopixelWrite braucht kein begin())
    pinMode(LED_PIN, OUTPUT);
    led_test();

    // I2S (Slave + MCLK)
    led_parse("LED:gelb pulsierend schnell");
    if (!i2s_init()) {
        Serial.println("[FATAL] I2S Init fehlgeschlagen!");
        led_parse("LED:rot blinkend schnell");
        while (true) { led_update(); delay(10); }
    }

    // WiFi
    led_parse("LED:blau pulsierend mittel");
    wifi_connect();

    if (WiFi.isConnected()) {
        udp_audio.begin(0);
        udp_led.begin(LED_UDP_PORT);
        http_setup();
        ota_setup();
        led_set(0, 255, 0); delay(500);
        led_parse("LED:blau pulsierend langsam");
    } else {
        led_parse("LED:rot blinkend schnell");
    }

    // Audio-Task (Core 1, Prioritaet 5)
    xTaskCreatePinnedToCore(audio_task, "audio", 8192, NULL, 5,
                            &audio_task_handle, 1);

    Serial.printf("[SETUP] Bereit! Modus: %s, UDP -> %s:%d\n",
                  audio_mode == 0 ? "16kHz Mono" : "48kHz Stereo",
                  PI_IP, audio_mode == 0 ? UDP_PORT_16K : UDP_PORT_48K);
}


// =====================================================================
// Loop
// =====================================================================

void loop() {
    wifi_check();

    if (WiFi.isConnected()) {
        http_server.handleClient();
        ArduinoOTA.handle();
        led_udp_check();
    }

    led_update();
    delay(5);
}
