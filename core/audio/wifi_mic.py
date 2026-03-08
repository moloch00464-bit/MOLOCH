"""
WiFiMic — UDP-Client fuer ReSpeaker ESP32-S3 WiFi-Mikrofon
============================================================

Empfaengt Audio-Streams per UDP vom ESP32-S3
in 16kHz (Whisper) und 48kHz (Stimmbiometrie).

Features:
- Dual-Stream: Port 12345 (16kHz, 320B/Paket) + Port 12346 (48kHz, 960B/Paket)
- Ringpuffer 2s je Stream
- get_audio_chunk(rate) → bytes fuer Whisper/Biometrie
- Health-Monitor: connected=True wenn Pakete innerhalb 2s empfangen
- Event audio.mic_source_changed bei Verbindungsaufbau
- Fallback auf USB-Soundkarte nach 10s ohne UDP-Daten

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
    """UDP-Client fuer ESP32-S3 WiFi-Mikrofon."""

    # UDP Paketgroessen (ESP32 sendet genau diese Chunks)
    CHUNK_16K = 320   # 16kHz Mono 16-bit: 10ms = 320 Bytes
    CHUNK_48K = 960   # 48kHz Stereo 16-bit: 5ms = 960 Bytes

    # Timeout: Kein Paket seit X Sekunden → disconnected
    HEALTH_TIMEOUT = 2.0

    def __init__(self, esp_ip: str = "10.42.0.2",
                 port_16k: int = 12345, port_48k: int = 12346,
                 event_bus=None):
        """
        Args:
            esp_ip: IP-Adresse des ESP32-S3 (fuer Status-Anzeige)
            port_16k: UDP-Port fuer 16kHz Stream (bind lokal)
            port_48k: UDP-Port fuer 48kHz Stream (bind lokal)
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
        self._last_recv_16k = 0.0
        self._last_recv_48k = 0.0

        # Paket-Statistiken
        self._packets_recv_16k = 0
        self._packets_recv_48k = 0
        self._recv_start_16k = 0.0  # Zeitpunkt erster Empfang

        # Software Gain (Multiplikator fuer WiFi-Audio, 0.0 - 3.0)
        self._software_gain = 1.0

        # Quellen-Erzwingung: "auto" (Default), "wifi", "usb"
        self._force_source = "auto"

        # Locks
        self._lock_16k = threading.Lock()
        self._lock_48k = threading.Lock()

        # Threads
        self._thread_16k: Optional[threading.Thread] = None
        self._thread_48k: Optional[threading.Thread] = None
        self._thread_health: Optional[threading.Thread] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """Startet UDP-Sockets und Empfangs-Threads."""
        if self._running:
            return
        self._running = True

        # UDP-Sockets erstellen und binden
        self._sock_16k = self._create_udp_socket(self.port_16k)
        self._sock_48k = self._create_udp_socket(self.port_48k)

        self._thread_16k = threading.Thread(target=self._recv_loop,
                                            args=(16000,), daemon=True,
                                            name="WiFiMic-16k")
        self._thread_48k = threading.Thread(target=self._recv_loop,
                                            args=(48000,), daemon=True,
                                            name="WiFiMic-48k")
        self._thread_health = threading.Thread(target=self._health_loop,
                                               daemon=True,
                                               name="WiFiMic-Health")

        self._thread_16k.start()
        self._thread_48k.start()
        self._thread_health.start()

        logger.info(f"WiFiMic gestartet, UDP Ports {self.port_16k}/{self.port_48k}")

    def stop(self):
        """Stoppt alle Threads und schliesst Sockets."""
        self._running = False
        self._close_sockets()
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

            # Software Gain anwenden (nur wenn != 1.0)
            if self._software_gain != 1.0 and len(chunk) >= 2:
                import struct as _struct
                n = len(chunk) // 2
                samples = _struct.unpack(f"<{n}h", chunk[:n * 2])
                gained = _struct.pack(f"<{n}h", *(
                    max(-32768, min(32767, int(s * self._software_gain)))
                    for s in samples
                ))
                return gained

            return chunk

    @property
    def connected(self) -> bool:
        """True wenn 16kHz Stream verbunden UND nicht auf USB erzwungen."""
        if self._force_source == "usb":
            return False
        return self._connected_16k

    @property
    def source(self) -> str:
        """Aktuelle Audio-Quelle: 'wifi', 'usb', oder 'none'."""
        return self._source

    def get_status(self) -> dict:
        """Status-Dict fuer IPC/Panel."""
        # Paket-Verlust schaetzen (16kHz: 100 Pakete/s erwartet)
        elapsed = time.monotonic() - self._recv_start_16k if self._recv_start_16k else 0
        expected = int(elapsed * 100) if elapsed > 1 else self._packets_recv_16k
        lost = max(0, expected - self._packets_recv_16k)

        return {
            "source": self._source,
            "connected_16k": self._connected_16k,
            "connected_48k": self._connected_48k,
            "esp_ip": self.esp_ip,
            "buf_16k_bytes": len(self._buf_16k),
            "buf_48k_bytes": len(self._buf_48k),
            "packets_recv_16k": self._packets_recv_16k,
            "packets_recv_48k": self._packets_recv_48k,
            "packets_lost_16k": lost,
            "software_gain": self._software_gain,
            "force_source": self._force_source,
        }

    def peek_rms(self, num_samples: int = 160) -> float:
        """RMS-Pegel aus den neuesten Samples OHNE Buffer zu leeren.

        Args:
            num_samples: Anzahl 16-bit Samples (160 = 10ms bei 16kHz)

        Returns:
            RMS in dB (0 = Vollaussteuerung, -80 = Stille)
        """
        import math
        with self._lock_16k:
            buf_len = len(self._buf_16k)
            if buf_len < 4:
                return -80.0
            # Letzte N Bytes lesen (2 Bytes pro Sample)
            num_bytes = min(num_samples * 2, buf_len)
            # Aus deque hinten lesen ohne zu entfernen
            start = buf_len - num_bytes
            raw = bytes(self._buf_16k[i] for i in range(start, buf_len))

        import struct as _struct
        n = len(raw) // 2
        if n < 2:
            return -80.0
        samples = _struct.unpack(f"<{n}h", raw[:n * 2])
        rms = math.sqrt(sum(s * s for s in samples) / n)
        return 20 * math.log10(max(rms, 1) / 32768.0)

    @property
    def software_gain(self) -> float:
        """Aktueller Software-Gain Multiplikator."""
        return self._software_gain

    @software_gain.setter
    def software_gain(self, value: float):
        """Software-Gain setzen (0.0 - 3.0)."""
        self._software_gain = max(0.0, min(3.0, float(value)))

    def set_force_source(self, mode: str):
        """Audio-Quelle erzwingen: 'auto', 'wifi', oder 'usb'.

        'usb' macht connected=False → VoicePipeline nutzt arecord Fallback.
        """
        if mode in ("auto", "wifi", "usb"):
            self._force_source = mode
            logger.info(f"Audio-Quelle erzwungen: {mode}")

    # =========================================================================
    # Interne Methoden
    # =========================================================================

    def _create_udp_socket(self, port: int) -> socket.socket:
        """Erstellt und bindet einen UDP-Socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind(("0.0.0.0", port))
        logger.info(f"UDP-Socket gebunden auf Port {port}")
        return sock

    def _recv_loop(self, rate: int):
        """Empfaengt Audio-Daten per UDP."""
        sock = self._sock_16k if rate == 16000 else self._sock_48k
        chunk_size = self.CHUNK_16K if rate == 16000 else self.CHUNK_48K
        label = f"{rate // 1000}kHz"

        while self._running:
            try:
                data, addr = sock.recvfrom(chunk_size * 4)  # Puffer grosszuegig
                if not data:
                    continue

                # Zeitstempel fuer Health-Monitor
                now = time.monotonic()
                if rate == 16000:
                    self._last_recv_16k = now
                    self._packets_recv_16k += 1
                    if self._recv_start_16k == 0.0:
                        self._recv_start_16k = now
                    with self._lock_16k:
                        self._buf_16k.extend(data)
                else:
                    self._last_recv_48k = now
                    self._packets_recv_48k += 1
                    with self._lock_48k:
                        self._buf_48k.extend(data)

                # Erster Empfang → connected melden
                if rate == 16000 and not self._connected_16k:
                    self._connected_16k = True
                    self._source = "wifi"
                    logger.info(f"[{label}] UDP empfange von {addr[0]}:{addr[1]}")
                    self._fire_source_event("wifi", rate)
                elif rate == 48000 and not self._connected_48k:
                    self._connected_48k = True
                    logger.info(f"[{label}] UDP empfange von {addr[0]}:{addr[1]}")

            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    logger.warning(f"[{label}] UDP Recv-Fehler: {e}")
                    time.sleep(1)

    def _health_loop(self):
        """Prueft ob UDP-Pakete noch ankommen, meldet Fallback auf USB."""
        initial_deadline = time.time() + 10  # 10s Timeout fuer Fallback

        while self._running:
            time.sleep(2)
            now = time.monotonic()

            # 16kHz Health-Check
            if self._connected_16k:
                if now - self._last_recv_16k > self.HEALTH_TIMEOUT:
                    self._connected_16k = False
                    logger.warning("[16kHz] Keine UDP-Pakete seit 2s")

            if self._connected_48k:
                if now - self._last_recv_48k > self.HEALTH_TIMEOUT:
                    self._connected_48k = False
                    logger.warning("[48kHz] Keine UDP-Pakete seit 2s")

            # Fallback auf USB wenn 16kHz weg
            if not self._connected_16k and time.time() > initial_deadline:
                if self._source != "usb":
                    self._source = "usb"
                    logger.info("WiFi-Mic keine Daten, Fallback auf USB")
                    self._fire_source_event("usb", 16000)

            # Wieder connected → zurueck auf WiFi
            if self._connected_16k and self._source != "wifi":
                self._source = "wifi"
                logger.info("WiFi-Mic wieder aktiv")
                self._fire_source_event("wifi", 16000)

    def _close_sockets(self):
        """Schliesst beide UDP-Sockets."""
        for sock in (self._sock_16k, self._sock_48k):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        self._sock_16k = None
        self._sock_48k = None
        self._connected_16k = False
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
