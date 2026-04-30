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
import time
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple

logger = logging.getLogger("RGBLed")

# W18 Cross-Process State-Writer: in /dev/shm fuer Audit-Subprozesse lesbar
LED_STATE_PATH = "/dev/shm/moloch_led_state.json"
STATE_WRITER_INTERVAL_S = 5.0

# Mapping Farbname -> RGB-Tuple (best-effort fuer Audit-Lesbarkeit)
_COLOR_NAME_TO_RGB = {
    "rot":     (255, 0, 0),
    "gruen":   (0, 255, 0),
    "blau":    (0, 0, 255),
    "gelb":    (255, 255, 0),
    "cyan":    (0, 255, 255),
    "magenta": (255, 0, 255),
    "weiss":   (255, 255, 255),
    "orange":  (255, 128, 0),
    "aus":     (0, 0, 0),
    "regenbogen": (128, 128, 128),  # Symbolwert fuer animiertes Pattern
}

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

        # W18: Cross-Prozess State-Tracking — wird vom State-Writer-Thread nach /dev/shm geschrieben
        self._last_color_rgb: Tuple[int, int, int] = (0, 0, 0)
        self._last_color_name: str = "aus"
        self._last_pattern_name: Optional[str] = None
        self._last_brightness: int = 100  # Prozent — Hardware liefert keinen Readback, Default 100
        self._last_change_ts: float = time.time()
        self._available: bool = False  # True sobald Socket erfolgreich erstellt wurde
        self._state_writer_thread: Optional[threading.Thread] = None
        self._state_writer_stop = threading.Event()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Socket erstellen und Event-Bus abonnieren."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._available = True
            logger.info(f"RGB-LED Controller gestartet (ESP: {self._esp_ip}:{self._udp_port})")
        except OSError as e:
            self._available = False
            logger.warning(f"RGB-LED Socket-Init fehlgeschlagen: {e} — state-writer laeuft trotzdem (available=false)")

        # W18: State-Writer-Thread starten (best-effort, auch wenn Socket fehlt)
        self._start_state_writer()

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
        # W18: State-Writer-Thread stoppen
        self._state_writer_stop.set()
        if self._sock:
            self._sock.close()
            self._sock = None
        self._available = False
        # Letzten State noch einmal schreiben (markiert available=false)
        try:
            self._atomic_write_state(self._get_state_dict())
        except Exception:
            pass

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
            # W18: Cmd parsen und State updaten ("LED:gruen statisch mittel")
            try:
                payload = cmd[4:] if cmd.startswith("LED:") else cmd
                parts = payload.split()
                farbe = parts[0] if parts else "aus"
                self._update_tracked_state(farbe=farbe, pattern_name=None)
            except Exception:
                pass
        else:
            logger.debug(f"Kein LED-Mapping fuer Zustand: {state}")

    def set_color(self, farbe: str, modus: str = "statisch",
                  geschwindigkeit: str = "mittel"):
        """Farbe direkt setzen (fuer Chat-Kommandos)."""
        cmd = f"LED:{farbe} {modus} {geschwindigkeit}"
        self.send_command(cmd)
        # W18: Cross-Prozess State updaten + sofort schreiben
        self._update_tracked_state(farbe=farbe, pattern_name=None)

    # =========================================================================
    # W16 EXPRESSION API — set_pattern + flash_sequence
    # =========================================================================

    # Mapping: Pattern-Name → (farbe, modus, geschwindigkeit)
    _PATTERN_MAP = {
        "solid_blue":       ("blau",    "statisch",    "mittel"),
        "solid_red":        ("rot",     "statisch",    "mittel"),
        "pulsing_magenta":  ("magenta", "pulsierend",  "mittel"),
        "pulsing_red":      ("rot",     "pulsierend",  "schnell"),
        "dim_warm_white":   ("weiss",   "statisch",    "langsam"),
    }

    def set_pattern(self, name: str, params: Optional[dict] = None):
        """W16 Expression: vordefinierte Pattern via Name setzen.
        Erlaubte Namen: solid_blue, solid_red, pulsing_magenta, pulsing_red, dim_warm_white.
        params (optional): {'speed': 'langsam|mittel|schnell'} ueberschreibt Default-Speed.
        """
        mapping = self._PATTERN_MAP.get(name)
        if not mapping:
            logger.warning(f"[LED] Unbekanntes Pattern: {name}")
            return
        farbe, modus, speed = mapping
        if params and isinstance(params, dict):
            speed_override = params.get("speed")
            if speed_override in ("langsam", "mittel", "schnell"):
                speed = speed_override
        self.set_color(farbe, modus, speed)
        # W18: Pattern-Name explizit nachziehen (set_color setzt nur Farbe)
        self._update_tracked_state(farbe=farbe, pattern_name=name)
        logger.debug(f"[LED] set_pattern({name}) -> {farbe} {modus} {speed}")

    def flash_sequence(self, sequence: List[Tuple[Tuple[int, int, int], float]]):
        """W16 Expression: blockierende Sequenz aus (rgb, dauer_s)-Paaren.
        Beispiel: [((255,0,0), 0.1), ((0,0,0), 0.1), ((255,0,0), 0.1)] = 3 Blitze rot.
        RGB wird auf naechstgelegene benannte Farbe gemappt.
        """
        if not sequence:
            return
        for entry in sequence:
            try:
                rgb, dauer = entry
                farbe = self._rgb_to_color_name(rgb)
                self.set_color(farbe, "statisch", "mittel")
                # W18: Roh-RGB-Wert behalten (set_color hat nur Farbnamen)
                self._update_tracked_state(
                    farbe=farbe,
                    pattern_name="flash_sequence",
                    rgb_override=tuple(int(c) for c in rgb),
                )
                time.sleep(max(0.0, float(dauer)))
            except (TypeError, ValueError) as e:
                logger.warning(f"[LED] flash_sequence Eintrag invalid {entry}: {e}")
                continue

    @staticmethod
    def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
        """W16: RGB-Tuple auf naechste benannte Farbe mappen (euklid. Distanz)."""
        try:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        except (TypeError, IndexError, ValueError):
            return "aus"
        # Schwarz/Aus separat
        if r < 20 and g < 20 and b < 20:
            return "aus"
        palette = {
            "rot":     (255, 0, 0),
            "gruen":   (0, 255, 0),
            "blau":    (0, 0, 255),
            "gelb":    (255, 255, 0),
            "cyan":    (0, 255, 255),
            "magenta": (255, 0, 255),
            "weiss":   (255, 255, 255),
        }
        best_name = "weiss"
        best_dist = 10**9
        for name, (pr, pg, pb) in palette.items():
            d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

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


    # =========================================================================
    # W18 — Cross-Prozess State-Writer (/dev/shm/moloch_led_state.json)
    # =========================================================================

    def _update_tracked_state(self, farbe: Optional[str] = None,
                              pattern_name: Optional[str] = None,
                              rgb_override: Optional[Tuple[int, int, int]] = None,
                              brightness: Optional[int] = None) -> None:
        """W18: Aktualisiert den nachgehaltenen LED-State + schreibt sofort.
        Wird nach jeder set_color / set_pattern / flash_sequence aufgerufen.
        """
        try:
            if rgb_override is not None:
                self._last_color_rgb = (
                    int(rgb_override[0]),
                    int(rgb_override[1]),
                    int(rgb_override[2]),
                )
                # Farbnamen aus RGB ableiten falls nicht uebergeben
                if farbe is None:
                    farbe = self._rgb_to_color_name(rgb_override)
            elif farbe is not None:
                self._last_color_rgb = _COLOR_NAME_TO_RGB.get(farbe, (0, 0, 0))
            if farbe is not None:
                self._last_color_name = farbe
            if pattern_name is not None:
                self._last_pattern_name = pattern_name
            if brightness is not None:
                self._last_brightness = max(0, min(100, int(brightness)))
            self._last_change_ts = time.time()
            # Sofort schreiben — Audit-Subprozesse sehen Aenderung ohne 5s-Latenz
            self._atomic_write_state(self._get_state_dict())
        except Exception as e:
            logger.debug(f"[LED] _update_tracked_state failed: {e}")

    def _get_state_dict(self) -> dict:
        """W18: Aktuellen LED-State als Dict fuer JSON-Export."""
        ts = time.time()
        return {
            "ts": ts,
            "iso": datetime.utcfromtimestamp(ts).isoformat() + "Z",
            "available": bool(self._available),
            "color": [int(self._last_color_rgb[0]),
                      int(self._last_color_rgb[1]),
                      int(self._last_color_rgb[2])],
            "color_name": self._last_color_name,
            "pattern_name": self._last_pattern_name,
            "brightness": int(self._last_brightness),
            "current_state": self._current_state,
            "current_mood": self._current_mood,
            "last_change_ts": float(self._last_change_ts),
        }

    @staticmethod
    def _atomic_write_state(d: dict) -> None:
        """W18: Atomic JSON-Write nach LED_STATE_PATH (NEVER 6: tempfile + os.replace)."""
        try:
            target_dir = os.path.dirname(LED_STATE_PATH) or "/tmp"
            fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".led_state_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(d, f, ensure_ascii=False)
                os.replace(tmp, LED_STATE_PATH)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.debug(f"[LED] state-write failed: {e}")

    def _state_writer_loop(self) -> None:
        """W18: Periodischer Writer-Thread — alle STATE_WRITER_INTERVAL_S Sekunden."""
        logger.debug(f"[LED] state-writer-thread gestartet (interval={STATE_WRITER_INTERVAL_S}s)")
        while not self._state_writer_stop.is_set():
            try:
                self._atomic_write_state(self._get_state_dict())
            except Exception as e:
                logger.debug(f"[LED] state-writer tick failed: {e}")
            # wait() laesst sich sauber unterbrechen via stop-Event
            if self._state_writer_stop.wait(timeout=STATE_WRITER_INTERVAL_S):
                break
        logger.debug("[LED] state-writer-thread beendet")

    def _start_state_writer(self) -> None:
        """W18: Daemon-Thread starten (idempotent)."""
        if self._state_writer_thread and self._state_writer_thread.is_alive():
            return
        self._state_writer_stop.clear()
        # Initial-Write damit die Datei sofort existiert
        try:
            self._atomic_write_state(self._get_state_dict())
        except Exception:
            pass
        t = threading.Thread(
            target=self._state_writer_loop,
            name="led-state-writer",
            daemon=True,
        )
        t.start()
        self._state_writer_thread = t


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
