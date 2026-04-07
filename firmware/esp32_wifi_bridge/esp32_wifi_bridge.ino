/*
 * M.O.L.O.C.H. ESP32-C3 WiFi Bridge - Using SSCMA Library
 * ========================================================
 *
 * Bridges Grove Vision AI V2 (Himax) to WiFi HTTP API.
 * Uses official Seeed SSCMA library for proper I2C protocol.
 *
 * Hardware: XIAO ESP32-C3 (Port1, upper USB-C)
 * Connected to: Grove Vision AI V2 via Sandwich (I2C)
 *
 * HTTP Endpoints:
 *   GET /          - Status page
 *   GET /status    - JSON status
 *   GET /inference - Latest inference results
 *
 * WiFi: Static IP 192.168.2.200
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Seeed_Arduino_SSCMA.h>

// ============================================================================
// Configuration
// ============================================================================

const char* WIFI_SSID = "MolochWLAN2.4";
const char* WIFI_PASS = "ClaudeMoloch666";

// Static IP Configuration
IPAddress staticIP(192, 168, 2, 200);
IPAddress gateway(192, 168, 2, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress dns(192, 168, 2, 1);

// HTTP Server
WebServer server(80);

// SSCMA AI Instance
SSCMA AI;

// ============================================================================
// Global State
// ============================================================================

String lastInference = "";
unsigned long lastInferenceTime = 0;
bool himaxConnected = false;
int inferenceCount = 0;

// ============================================================================
// HTTP Handlers
// ============================================================================

void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>M.O.L.O.C.H. Eye</title>";
  html += "<meta http-equiv='refresh' content='2'>";
  html += "<style>body{background:#1a1a2e;color:#00ff88;font-family:monospace;padding:20px;}";
  html += "h1{color:#00aaff;}.ok{color:#00ff88;}.err{color:#ff4444;}</style></head>";
  html += "<body><h1>M.O.L.O.C.H. WiFi Bridge</h1>";
  html += "<p>Himax: <span class='" + String(himaxConnected ? "ok'>Connected" : "err'>Disconnected") + "</span></p>";
  html += "<p>Inferences: " + String(inferenceCount) + "</p>";
  html += "<p>Uptime: " + String(millis() / 1000) + "s</p>";
  html += "<p>WiFi RSSI: " + String(WiFi.RSSI()) + " dBm</p>";
  html += "<h2>Endpoints:</h2>";
  html += "<ul><li><a href='/status'>/status</a> - JSON status</li>";
  html += "<li><a href='/inference'>/inference</a> - Latest inference</li></ul>";
  html += "<h2>Last Inference:</h2>";
  html += "<pre>" + lastInference + "</pre>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

void handleStatus() {
  String json = "{";
  json += "\"status\":\"ok\",";
  json += "\"mode\":\"sscma_i2c\",";
  json += "\"himax_connected\":" + String(himaxConnected ? "true" : "false") + ",";
  json += "\"inference_count\":" + String(inferenceCount) + ",";
  json += "\"uptime_ms\":" + String(millis()) + ",";
  json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += "}";

  server.send(200, "application/json", json);
}

void handleInference() {
  // Run inference via SSCMA library
  if (!AI.invoke(1, false, false)) {
    himaxConnected = true;
    inferenceCount++;

    // Build JSON response
    String json = "{\"num_detections\":" + String(AI.boxes().size()) + ",\"detections\":[";

    for (int i = 0; i < AI.boxes().size(); i++) {
      if (i > 0) json += ",";
      json += "{\"x\":" + String(AI.boxes()[i].x);
      json += ",\"y\":" + String(AI.boxes()[i].y);
      json += ",\"w\":" + String(AI.boxes()[i].w);
      json += ",\"h\":" + String(AI.boxes()[i].h);
      json += ",\"score\":" + String(AI.boxes()[i].score);
      json += ",\"target\":" + String(AI.boxes()[i].target) + "}";
    }

    json += "],\"perf\":{";
    json += "\"preprocess\":" + String(AI.perf().prepocess);
    json += ",\"inference\":" + String(AI.perf().inference);
    json += ",\"postprocess\":" + String(AI.perf().postprocess) + "}}";

    lastInference = json;
    lastInferenceTime = millis();

    server.send(200, "application/json", json);
  } else {
    himaxConnected = false;
    if (lastInference.length() > 0) {
      server.send(200, "application/json", lastInference);
    } else {
      server.send(204, "application/json", "");
    }
  }
}

void handleNotFound() {
  server.send(404, "text/plain", "Not Found");
}

// ============================================================================
// WiFi Setup
// ============================================================================

void setupWiFi() {
  Serial.println("Connecting to WiFi...");
  Serial.println("SSID: " + String(WIFI_SSID));

  WiFi.mode(WIFI_STA);
  WiFi.config(staticIP, gateway, subnet, dns);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.println("IP: " + WiFi.localIP().toString());
    Serial.println("RSSI: " + String(WiFi.RSSI()) + " dBm");
  } else {
    Serial.println("\nWiFi connection failed!");
    Serial.println("Restarting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }
}

// ============================================================================
// Main
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=====================================");
  Serial.println("M.O.L.O.C.H. ESP32-C3 WiFi Bridge");
  Serial.println("Mode: SSCMA I2C Library");
  Serial.println("=====================================");

  // Connect to WiFi first
  setupWiFi();

  // Initialize SSCMA AI via I2C (default Wire, address 0x62)
  Serial.println("Initializing SSCMA...");
  if (AI.begin()) {
    himaxConnected = true;
    Serial.println("SSCMA initialized successfully!");
  } else {
    himaxConnected = false;
    Serial.println("SSCMA initialization failed!");
  }

  // Setup HTTP server
  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.on("/inference", handleInference);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started on port 80");
  Serial.println("Ready!");
}

void loop() {
  server.handleClient();
  delay(1);
}
