#!/usr/bin/env python3
"""
Audio Source Pipeline — WiFi-Mic primaer, USB-Soundkarte Fallback
=================================================================

Routing-Layer fuer Audio-Quellen. Entscheidet woher Audio kommt:
1. WiFi-Mic (ESP32-S3 TCP) — primaer, wenn verbunden
2. USB-Soundkarte (ReSpeaker Lite XMOS) — Fallback

Andere Module (push_to_talk, Whisper, Stimmbiometrie) holen Audio
ueber get_audio_chunk() statt direkt von sounddevice/pw-record.

Author: M.O.L.O.C.H. System
"""

import logging
import threading
import time
import subprocess
import numpy as np
from typing import Optional

logger = logging.getLogger("AudioSourcePipeline")

# USB ReSpeaker Lite PipeWire Node-Name
USB_SOURCE_NODE = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
USB_CARD_INDEX = 2  # arecord -l: Card 2


class AudioSourcePipeline:
    """
    Verwaltet Audio-Quellen und liefert Audio-Chunks.

    Primaer: WiFi-Mic (ESP32 TCP auf Port 12345/12346)
    Fallback: USB-Soundkarte (ReSpeaker Lite direkt via PipeWire)
    """

    def __init__(self, esp_ip: str = "10.42.0.2", event_bus=None):
        self._esp_ip = esp_ip
        self._event_bus = event_bus
        self._wifi_mic = None
        self._source = "none"  # "wifi", "usb", "none"
        self._running = False
        self._lock = threading.Lock()

        # USB-Fallback Recording State
        self._usb_recording = False
        self._usb_process: Optional[subprocess.Popen] = None
        self._usb_buf = bytearray()
        self._usb_lock = threading.Lock()
        self._usb_thread: Optional[threading.Thread] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Startet die Audio-Pipeline. Versucht WiFi-Mic, Fallback auf USB."""
        if self._running:
            return
        self._running = True

        # WiFi-Mic importieren und starten
        try:
            from core.audio.wifi_mic import WiFiMic
            self._wifi_mic = WiFiMic(
                esp_ip=self._esp_ip,
                event_bus=self._event_bus
            )
            self._wifi_mic.start()
            logger.info(f"WiFi-Mic gestartet, Ziel: {self._esp_ip}")
        except ImportError:
            logger.warning("WiFi-Mic Modul nicht gefunden, nur USB verfuegbar")
            self._wifi_mic = None

        # Source-Monitor Thread
        self._monitor_thread = threading.Thread(
            target=self._source_monitor, daemon=True,
            name="AudioSource-Monitor"
        )
        self._monitor_thread.start()

    def stop(self):
        """Stoppt alle Audio-Quellen."""
        self._running = False

        if self._wifi_mic:
            self._wifi_mic.stop()

        self._stop_usb_recording()
        logger.info("AudioSourcePipeline gestoppt")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_audio_chunk(self, rate: int = 16000, duration_ms: int = 100) -> bytes:
        """
        Gibt Audio-Chunk von der aktiven Quelle zurueck.

        Args:
            rate: 16000 oder 48000
            duration_ms: Laenge in Millisekunden

        Returns:
            bytes: PCM 16-bit Audio
        """
        # WiFi-Mic bevorzugt
        if self._wifi_mic and self._wifi_mic.connected:
            chunk = self._wifi_mic.get_audio_chunk(rate=rate, duration_ms=duration_ms)
            if len(chunk) > 0:
                if self._source != "wifi":
                    self._source = "wifi"
                    logger.info("Audio-Quelle: WiFi-Mic")
                return chunk

        # Fallback: USB
        if rate == 16000:
            return self._get_usb_chunk(duration_ms)

        # 48kHz von USB nicht unterstuetzt (nur 16kHz Fallback)
        return b''

    @property
    def source(self) -> str:
        """Aktuelle Audio-Quelle."""
        return self._source

    def get_status(self) -> dict:
        """Status fuer IPC/Panel."""
        wifi_status = self._wifi_mic.get_status() if self._wifi_mic else {}
        return {
            "source": self._source,
            "wifi_mic": wifi_status,
            "usb_recording": self._usb_recording,
            "usb_card": USB_CARD_INDEX,
        }

    # =========================================================================
    # USB Fallback
    # =========================================================================

    def _start_usb_recording(self):
        """Startet USB-Aufnahme via pw-record als Fallback."""
        if self._usb_recording:
            return

        try:
            cmd = [
                "pw-record",
                "--target", USB_SOURCE_NODE,
                "--format", "s16",
                "--rate", "16000",
                "--channels", "1",
                "-"
            ]
            self._usb_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            self._usb_recording = True

            self._usb_thread = threading.Thread(
                target=self._usb_read_loop, daemon=True,
                name="AudioSource-USB"
            )
            self._usb_thread.start()

            self._source = "usb"
            logger.info("USB-Aufnahme gestartet (ReSpeaker Lite)")
            self._fire_source_event("usb", 16000)

        except Exception as e:
            logger.error(f"USB-Aufnahme Fehler: {e}")
            self._usb_recording = False

    def _stop_usb_recording(self):
        """Stoppt USB-Aufnahme."""
        self._usb_recording = False
        if self._usb_process:
            try:
                self._usb_process.terminate()
                self._usb_process.wait(timeout=2)
            except:
                pass
            self._usb_process = None

    def _usb_read_loop(self):
        """Liest Audio-Daten vom USB pw-record Prozess."""
        while self._usb_recording and self._usb_process:
            try:
                data = self._usb_process.stdout.read(3200)  # 100ms bei 16kHz Mono 16-bit
                if not data:
                    break
                with self._usb_lock:
                    # Ringpuffer: max 2s = 64000 Bytes
                    self._usb_buf.extend(data)
                    if len(self._usb_buf) > 64000:
                        self._usb_buf = self._usb_buf[-64000:]
            except:
                break

        self._usb_recording = False

    def _get_usb_chunk(self, duration_ms: int) -> bytes:
        """Holt Audio-Chunk aus dem USB-Puffer."""
        if not self._usb_recording:
            self._start_usb_recording()
            # Kurz warten bis erste Daten da sind
            time.sleep(0.1)

        num_bytes = (16000 * 2 * duration_ms) // 1000  # 16kHz Mono 16-bit

        with self._usb_lock:
            if len(self._usb_buf) < num_bytes:
                chunk = bytes(self._usb_buf)
                self._usb_buf.clear()
                return chunk

            chunk = bytes(self._usb_buf[:num_bytes])
            self._usb_buf = self._usb_buf[num_bytes:]
            return chunk

    # =========================================================================
    # Source Monitor
    # =========================================================================

    def _source_monitor(self):
        """Ueberwacht Quellen-Status, wechselt zwischen WiFi und USB."""
        while self._running:
            time.sleep(2)

            wifi_ok = self._wifi_mic and self._wifi_mic.connected

            if wifi_ok and self._source != "wifi":
                # WiFi ist zurueck → USB stoppen
                self._stop_usb_recording()
                self._source = "wifi"
                logger.info("Wechsel: USB → WiFi-Mic")
                self._fire_source_event("wifi", 16000)

            elif not wifi_ok and self._source != "usb":
                # WiFi weg → USB starten
                if not self._usb_recording:
                    self._start_usb_recording()

    def _fire_source_event(self, source: str, rate: int):
        """Event auf Event-Bus feuern."""
        if self._event_bus and hasattr(self._event_bus, 'emit'):
            try:
                self._event_bus.emit("audio.mic_source_changed", {
                    "source": source,
                    "rate": rate,
                    "priority": 5
                })
            except Exception as e:
                logger.warning(f"Event-Bus Fehler: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[AudioSourcePipeline] = None


def get_audio_source(**kwargs) -> AudioSourcePipeline:
    """Singleton-Zugriff auf AudioSourcePipeline."""
    global _instance
    if _instance is None:
        _instance = AudioSourcePipeline(**kwargs)
    return _instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    esp_ip = sys.argv[1] if len(sys.argv) > 1 else "10.42.0.2"

    pipeline = AudioSourcePipeline(esp_ip=esp_ip)
    pipeline.start()

    print(f"AudioSourcePipeline laeuft (ESP: {esp_ip})")
    print("Druecke Ctrl+C zum Beenden")

    try:
        while True:
            time.sleep(1)
            chunk = pipeline.get_audio_chunk(rate=16000, duration_ms=100)
            status = pipeline.get_status()
            # RMS berechnen fuer Level-Anzeige
            if len(chunk) >= 2:
                samples = np.frombuffer(chunk, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                rms_db = 20 * np.log10(max(rms, 1) / 32768)
            else:
                rms_db = -100
            print(f"Source={status['source']} | "
                  f"Chunk={len(chunk)}B | "
                  f"Level={rms_db:.1f}dB")
    except KeyboardInterrupt:
        pipeline.stop()
        print("\nBeendet.")
