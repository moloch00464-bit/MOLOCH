"""
WiFiMic — TCP-Client fuer ReSpeaker ESP32-S3 WiFi-Mikrofon
============================================================

Verbindet sich per TCP zum ESP32-S3, empfaengt Audio-Streams
in 16kHz (Whisper) und 48kHz (Stimmbiometrie).

Features:
- Dual-Stream: Port 12345 (16kHz) + Port 12346 (48kHz)
- Ringpuffer 2s je Stream
- get_audio_chunk(rate) → bytes fuer Whisper/Biometrie
- Reconnect-Loop alle 5s bei Verbindungsverlust
- Event audio.mic_source_changed bei Verbindungsaufbau
- Fallback auf USB-Soundkarte nach 10s ohne TCP

Author: M.O.L.O.C.H. System
"""

import socket
import threading
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger("WiFiMic")


class WiFiMic:
    """TCP-Client fuer ESP32-S3 WiFi-Mikrofon."""

    def __init__(self, esp_ip: str = "10.42.0.2",
                 port_16k: int = 12345, port_48k: int = 12346,
                 event_bus=None):
        """
        Args:
            esp_ip: IP-Adresse des ESP32-S3
            port_16k: TCP-Port fuer 16kHz Stream
            port_48k: TCP-Port fuer 48kHz Stream
            event_bus: Optionaler Event-Bus fuer mic_source_changed
        """
        self.esp_ip = esp_ip
        self.port_16k = port_16k
        self.port_48k = port_48k
        self.event_bus = event_bus

        # Ringpuffer: 2 Sekunden Audio
        # 16kHz Mono 16-bit = 32.000 Bytes/s → 64.000 Bytes fuer 2s
        # 48kHz Stereo 16-bit = 192.000 Bytes/s → 384.000 Bytes fuer 2s
        self._buf_16k = deque(maxlen=64000)
        self._buf_48k = deque(maxlen=384000)

        # Sockets
        self._sock_16k: Optional[socket.socket] = None
        self._sock_48k: Optional[socket.socket] = None

        # Status
        self._connected_16k = False
        self._connected_48k = False
        self._running = False
        self._source = "none"  # "wifi", "usb", "none"

        # Locks
        self._lock_16k = threading.Lock()
        self._lock_48k = threading.Lock()

        # Threads
        self._thread_16k: Optional[threading.Thread] = None
        self._thread_48k: Optional[threading.Thread] = None
        self._thread_reconnect: Optional[threading.Thread] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """Startet Empfangs-Threads und Reconnect-Loop."""
        if self._running:
            return
        self._running = True

        self._thread_16k = threading.Thread(target=self._recv_loop,
                                            args=(16000,), daemon=True,
                                            name="WiFiMic-16k")
        self._thread_48k = threading.Thread(target=self._recv_loop,
                                            args=(48000,), daemon=True,
                                            name="WiFiMic-48k")
        self._thread_reconnect = threading.Thread(target=self._reconnect_loop,
                                                  daemon=True,
                                                  name="WiFiMic-Reconnect")

        self._thread_reconnect.start()
        self._thread_16k.start()
        self._thread_48k.start()

        logger.info(f"WiFiMic gestartet, ESP32 Ziel: {self.esp_ip}")

    def stop(self):
        """Stoppt alle Threads und schliesst Sockets."""
        self._running = False
        self._close_socket(16000)
        self._close_socket(48000)
        logger.info("WiFiMic gestoppt")

    def get_audio_chunk(self, rate: int = 16000, duration_ms: int = 100) -> bytes:
        """
        Gibt Audio-Chunk zurueck.

        Args:
            rate: 16000 oder 48000
            duration_ms: Laenge in Millisekunden

        Returns:
            bytes: PCM 16-bit Audio (Mono bei 16k, Stereo bei 48k)
        """
        if rate == 16000:
            # 16kHz Mono 16-bit: 32 Bytes pro ms
            num_bytes = (rate * 2 * duration_ms) // 1000
            buf = self._buf_16k
            lock = self._lock_16k
        elif rate == 48000:
            # 48kHz Stereo 16-bit: 192 Bytes pro ms
            num_bytes = (rate * 2 * 2 * duration_ms) // 1000
            buf = self._buf_48k
            lock = self._lock_48k
        else:
            logger.error(f"Ungueltge Sample-Rate: {rate}")
            return b''

        with lock:
            available = len(buf)
            if available < num_bytes:
                # Nicht genug Daten, gib was da ist (oder leer)
                chunk = bytes(buf)
                buf.clear()
                return chunk

            # Aelteste Daten aus dem Ringpuffer holen
            chunk = bytes(buf.popleft() for _ in range(num_bytes))
            return chunk

    @property
    def connected(self) -> bool:
        """True wenn mindestens der 16kHz Stream verbunden ist."""
        return self._connected_16k

    @property
    def source(self) -> str:
        """Aktuelle Audio-Quelle: 'wifi', 'usb', oder 'none'."""
        return self._source

    def get_status(self) -> dict:
        """Status-Dict fuer IPC/Panel."""
        return {
            "source": self._source,
            "connected_16k": self._connected_16k,
            "connected_48k": self._connected_48k,
            "esp_ip": self.esp_ip,
            "buf_16k_bytes": len(self._buf_16k),
            "buf_48k_bytes": len(self._buf_48k),
        }

    # =========================================================================
    # Interne Empfangs-Loops
    # =========================================================================

    def _recv_loop(self, rate: int):
        """Empfaengt Audio-Daten von einem TCP-Stream."""
        port = self.port_16k if rate == 16000 else self.port_48k
        label = f"{rate // 1000}kHz"

        while self._running:
            sock = self._sock_16k if rate == 16000 else self._sock_48k
            if sock is None:
                time.sleep(0.5)
                continue

            try:
                data = sock.recv(4096)
                if not data:
                    # Verbindung geschlossen
                    logger.warning(f"[{label}] Verbindung geschlossen")
                    self._mark_disconnected(rate)
                    continue

                # In Ringpuffer schreiben
                if rate == 16000:
                    with self._lock_16k:
                        self._buf_16k.extend(data)
                else:
                    with self._lock_48k:
                        self._buf_48k.extend(data)

            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    logger.warning(f"[{label}] Recv-Fehler: {e}")
                    self._mark_disconnected(rate)
                    time.sleep(1)

    def _reconnect_loop(self):
        """Versucht alle 5s, getrennte Verbindungen wiederherzustellen."""
        # Erster Versuch sofort
        initial_deadline = time.time() + 10  # 10s Timeout fuer Fallback

        while self._running:
            if not self._connected_16k:
                self._try_connect(16000)
            if not self._connected_48k:
                self._try_connect(48000)

            # Fallback-Check: Nach 10s ohne 16kHz-Verbindung → USB melden
            if not self._connected_16k and time.time() > initial_deadline:
                if self._source != "usb":
                    self._source = "usb"
                    logger.info("WiFi-Mic nicht erreichbar, Fallback auf USB")
                    self._fire_source_event("usb", 16000)

            time.sleep(5)

    def _try_connect(self, rate: int):
        """Versucht eine TCP-Verbindung aufzubauen."""
        port = self.port_16k if rate == 16000 else self.port_48k
        label = f"{rate // 1000}kHz"

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.esp_ip, port))
            sock.settimeout(2)  # Recv-Timeout

            if rate == 16000:
                self._sock_16k = sock
                self._connected_16k = True
            else:
                self._sock_48k = sock
                self._connected_48k = True

            logger.info(f"[{label}] TCP verbunden: {self.esp_ip}:{port}")

            # Bei 16kHz-Verbindung: Source auf WiFi setzen
            if rate == 16000:
                self._source = "wifi"
                self._fire_source_event("wifi", rate)

        except (OSError, socket.timeout) as e:
            logger.debug(f"[{label}] Connect fehlgeschlagen: {e}")

    def _mark_disconnected(self, rate: int):
        """Markiert einen Stream als getrennt."""
        if rate == 16000:
            self._connected_16k = False
            self._close_socket(16000)
        else:
            self._connected_48k = False
            self._close_socket(48000)

    def _close_socket(self, rate: int):
        """Schliesst einen Socket sauber."""
        if rate == 16000 and self._sock_16k:
            try:
                self._sock_16k.close()
            except:
                pass
            self._sock_16k = None
            self._connected_16k = False
        elif rate == 48000 and self._sock_48k:
            try:
                self._sock_48k.close()
            except:
                pass
            self._sock_48k = None
            self._connected_48k = False

    def _fire_source_event(self, source: str, rate: int):
        """Event auf Event-Bus feuern wenn vorhanden."""
        if self.event_bus and hasattr(self.event_bus, 'emit'):
            try:
                self.event_bus.emit("audio.mic_source_changed", {
                    "source": source,
                    "rate": rate,
                    "priority": 5
                })
            except Exception as e:
                logger.warning(f"Event-Bus Fehler: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[WiFiMic] = None


def get_wifi_mic(**kwargs) -> WiFiMic:
    """Singleton-Zugriff auf WiFiMic-Instanz."""
    global _instance
    if _instance is None:
        _instance = WiFiMic(**kwargs)
    return _instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # ESP-IP als Argument oder Default
    esp_ip = sys.argv[1] if len(sys.argv) > 1 else "10.42.0.2"

    mic = WiFiMic(esp_ip=esp_ip)
    mic.start()

    print(f"WiFiMic laeuft, verbinde zu {esp_ip}...")
    print("Druecke Ctrl+C zum Beenden")

    try:
        while True:
            time.sleep(1)
            status = mic.get_status()
            chunk = mic.get_audio_chunk(rate=16000, duration_ms=100)
            print(f"Source={status['source']} | "
                  f"16k={'OK' if status['connected_16k'] else '--'} "
                  f"48k={'OK' if status['connected_48k'] else '--'} | "
                  f"Buf16k={status['buf_16k_bytes']}B | "
                  f"Chunk={len(chunk)}B")
    except KeyboardInterrupt:
        mic.stop()
        print("\nBeendet.")
