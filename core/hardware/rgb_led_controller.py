#!/usr/bin/env python3
"""
RGB-LED Controller — Steuert die WS2812 LED auf dem ReSpeaker Lite ESP32-S3
============================================================================

Sendet UDP-Kommandos an den ESP32 (Port 8888) zur LED-Steuerung.
Abonniert den Event-Bus fuer automatische Zustandsanzeige.

Kommando-Format: "LED:farbe [modus] [geschwindigkeit]"
  Farben:   rot, gruen, blau, gelb, cyan, magenta, weiss, aus
  Modi:     statisch, pulsierend, blinkend, regenbogen
  Speed:    langsam, mittel, schnell

Author: M.O.L.O.C.H. System
"""

import socket
import logging
import threading
from typing import Optional

logger = logging.getLogger("RGBLed")

# Zustand → LED Mapping
ZUSTAND_LED_MAP = {
    "verbinden":        "LED:blau blinkend mittel",      # Verbindet sich mit Pi / kein WiFi-Mic
    "idle":             "LED:gruen statisch",             # WiFi-Mic verbunden, Moloch bereit
    "person_erkannt":   "LED:gruen statisch",
    "markus_erkannt":   "LED:gelb statisch",
    "shadow_modus":     "LED:rot pulsierend schnell",
    "fehler":           "LED:rot blinkend schnell",
    "nachtmodus":       "LED:orange pulsierend langsam",
    "listening":        "LED:rot pulsierend schnell",    # PTT aktiv — Zuhören
    "thinking":         "LED:gelb blinkend mittel",      # Whisper verarbeitet
    "speaking":         "LED:magenta pulsierend mittel", # TTS spricht
    "tracking":         "LED:gruen pulsierend mittel",
    "enrollment":       "LED:gelb blinkend mittel",
    "boot":             "LED:regenbogen",
}


class RGBLedController:
    """UDP-Client zur Steuerung der ESP32 RGB-LED."""

    def __init__(self, esp_ip: str = "10.42.0.2", udp_port: int = 8888,
                 event_bus=None):
        self._esp_ip = esp_ip
        self._udp_port = udp_port
        self._event_bus = event_bus
        self._sock: Optional[socket.socket] = None
        self._current_state = "verbinden"
        self._current_mood = "guardian"  # guardian oder shadow
        self._wifi_mic_connected = False  # True sobald WiFi-Mic UDP-Pakete ankommen
        self._lock = threading.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Socket erstellen und Event-Bus abonnieren."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"RGB-LED Controller gestartet (ESP: {self._esp_ip}:{self._udp_port})")

        # Event-Bus abonnieren
        if self._event_bus and hasattr(self._event_bus, 'subscribe'):
            self._event_bus.subscribe("mood.changed", self._on_mood_changed)
            self._event_bus.subscribe("zone.changed", self._on_zone_changed)
            self._event_bus.subscribe("perception.person_detected", self._on_person)
            self._event_bus.subscribe("perception.face_recognized", self._on_face)
            self._event_bus.subscribe("audio.listening_start", self._on_listening)
            self._event_bus.subscribe("whisper.processing", self._on_thinking)
            self._event_bus.subscribe("audio.speaking_start", self._on_speaking)
            self._event_bus.subscribe("audio.speaking_end", self._on_idle)
            self._event_bus.subscribe("audio.mic_source_changed", self._on_mic_source_changed)
            logger.info("Event-Bus Subscriptions aktiv")

        # Initialzustand: WiFiMic-Status direkt pruefen (Event koennte schon verpasst sein)
        try:
            from core.audio.wifi_mic import get_wifi_mic
            if get_wifi_mic()._connected_16k:
                self._wifi_mic_connected = True
                self.set_state("idle")   # Gruen — schon verbunden
                logger.info("LED-Start: WiFi-Mic bereits verbunden → gruen")
            else:
                self.set_state("verbinden")  # Blau blinkend
        except Exception:
            self.set_state("verbinden")

    def stop(self):
        """Aufraeumen."""
        self.send_command("LED:aus")
        if self._sock:
            self._sock.close()
            self._sock = None

    # =========================================================================
    # Public API
    # =========================================================================

    def send_command(self, cmd: str):
        """Sende rohes LED-Kommando an ESP32."""
        if not self._sock:
            return
        with self._lock:
            try:
                self._sock.sendto(cmd.encode('utf-8'),
                                  (self._esp_ip, self._udp_port))
                logger.debug(f"LED: {cmd}")
            except OSError as e:
                logger.warning(f"LED-Kommando fehlgeschlagen: {e}")

    def set_state(self, state: str):
        """Zustand setzen — LED wird automatisch angepasst."""
        if state == self._current_state:
            return
        self._current_state = state
        cmd = ZUSTAND_LED_MAP.get(state)
        if cmd:
            self.send_command(cmd)
            logger.info(f"LED-Zustand: {state}")
        else:
            logger.debug(f"Kein LED-Mapping fuer Zustand: {state}")

    def set_color(self, farbe: str, modus: str = "statisch",
                  geschwindigkeit: str = "mittel"):
        """Farbe direkt setzen (fuer Chat-Kommandos)."""
        cmd = f"LED:{farbe} {modus} {geschwindigkeit}"
        self.send_command(cmd)

    # =========================================================================
    # Event-Bus Callbacks
    # =========================================================================

    def _on_mood_changed(self, data):
        """Mood/Zone Aenderung — Mood merken fuer SPEAKING-Farbe."""
        mood = data.get("mood", "")
        if mood in ("shadow", "guardian"):
            self._current_mood = mood
        if mood == "shadow":
            self.set_state("shadow_modus")
        elif mood == "guardian":
            self.set_state("idle")

    def _on_zone_changed(self, data):
        """Zone Aenderung."""
        zone = data.get("zone", "")
        if zone == "night":
            self.set_state("nachtmodus")

    def _on_person(self, data):
        """Person erkannt."""
        self.set_state("person_erkannt")

    def _on_face(self, data):
        """Gesicht erkannt."""
        name = data.get("name", "")
        if name.lower() == "markus":
            self.set_state("markus_erkannt")

    def _on_listening(self, data):
        """Mikrofon aktiv — PTT gedrueckt."""
        self.set_state("listening")

    def _on_thinking(self, data):
        """Whisper verarbeitet — NPU denkt."""
        self.set_state("thinking")

    def _on_speaking(self, data):
        """TTS spricht — Farbe je nach Mood (Guardian=magenta, Shadow=rot)."""
        if self._current_mood == "shadow":
            # Shadow Modus: Rot pulsierend
            self._current_state = ""  # Reset damit set_state nicht skippt
            self.send_command("LED:rot pulsierend mittel")
            self._current_state = "speaking_shadow"
        else:
            # Guardian Modus: Magenta pulsierend (Standard)
            self.set_state("speaking")

    def _on_mic_source_changed(self, data):
        """WiFi-Mic verbunden oder getrennt — LED entsprechend anpassen."""
        source = data.get("source", "none")
        if source == "wifi":
            self._wifi_mic_connected = True
            if self._current_state in ("idle", "verbinden"):
                self._current_state = ""  # Reset damit set_state nicht skippt
                self.set_state("idle")    # → Gruen statisch
        else:
            self._wifi_mic_connected = False
            if self._current_state in ("idle", "verbinden"):
                self._current_state = ""
                self.set_state("verbinden")  # → Blau blinkend

    def _on_idle(self, data):
        """Zurueck zu Idle — Gruen wenn WiFi-Mic verbunden, sonst Blau."""
        self._current_state = ""  # Reset damit set_state nicht skippt
        if self._wifi_mic_connected:
            self.set_state("idle")
        else:
            self.set_state("verbinden")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[RGBLedController] = None


def get_rgb_led(**kwargs) -> RGBLedController:
    """Singleton-Zugriff."""
    global _instance
    if _instance is None:
        _instance = RGBLedController(**kwargs)
    return _instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    esp_ip = sys.argv[1] if len(sys.argv) > 1 else "10.42.0.2"

    led = RGBLedController(esp_ip=esp_ip)
    led.start()

    print(f"RGB-LED Controller (ESP: {esp_ip})")
    print("Teste Zustaende...")

    for state in ["idle", "person_erkannt", "markus_erkannt", "shadow_modus",
                   "listening", "speaking", "fehler", "nachtmodus", "idle"]:
        print(f"  → {state}")
        led.set_state(state)
        time.sleep(2)

    print("\nManuelle Kommandos:")
    led.set_color("regenbogen")
    time.sleep(3)

    led.stop()
    print("Beendet.")
