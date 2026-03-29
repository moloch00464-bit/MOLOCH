"""
WiFiMic — UDP-Client fuer ReSpeaker ESP32-S3 WiFi-Mikrofon
============================================================

Empfaengt Audio-Streams per UDP vom ESP32-S3
in 16kHz (Whisper) und 48kHz (Stimmbiometrie).

Features:
- Dual-Stream: Port 12345 (16kHz, 324B/Paket) + Port 12346 (48kHz, 964B/Paket)
- 4-Byte Sequenznummer-Header pro Paket (Paketverlust-Erkennung)
- Jitter-Buffer: 100ms, sortiert nach Sequenznummer, Stille bei Luecken
- Grosser UDP-Recv-Buffer (1MB) gegen Kernel-Drops
- Schneller Chunk-basierter Ringpuffer (bytearray, nicht deque)
- get_audio_chunk(rate) → bytes fuer Whisper/Biometrie
- Health-Monitor: connected=True wenn Pakete innerhalb 2s empfangen
- Fallback auf USB-Soundkarte nach 10s ohne UDP-Daten

Author: M.O.L.O.C.H. System
v3.0 — Komplett optimiert: schneller Buffer + grosser UDP-Socket + chunk I/O
"""

import socket
import struct
import threading
import time
import logging
import math
from typing import Optional

logger = logging.getLogger("WiFiMic")


class WiFiMic:
    """UDP-Client fuer ESP32-S3 WiFi-Mikrofon."""

    # UDP Paketgroessen (ESP32 sendet: 4B Header + Audio)
    SEQ_HEADER_SIZE = 4   # uint32_t Sequenznummer (Little-Endian)
    CHUNK_16K = 320       # 16kHz Mono 16-bit: 10ms = 320 Bytes Audio
    CHUNK_48K = 960       # 48kHz Stereo 16-bit: 5ms = 960 Bytes Audio
    PACKET_16K = SEQ_HEADER_SIZE + CHUNK_16K  # 324 Bytes total
    PACKET_48K = SEQ_HEADER_SIZE + CHUNK_48K  # 964 Bytes total

    # Jitter-Buffer: 150ms = 15 Pakete bei 16kHz (10ms pro Paket)
    # Erhoet von 100ms/10 Paketen wegen WiFi-Bursts (Buffer war oft fast voll)
    JITTER_BUF_SIZE = 15  # Max Pakete im Jitter-Buffer
    JITTER_TIMEOUT_MS = 150  # Max Wartezeit bevor Ausspielen

    # UDP Socket Empfangspuffer: 1MB (default 208KB reicht nicht bei CPU-Last)
    UDP_RECV_BUF = 1048576

    # Ringpuffer Groesse (Bytes)
    RING_16K_SIZE = 128000   # 4 Sekunden bei 16kHz Mono 16-bit (32KB/s)
    RING_48K_SIZE = 768000   # 4 Sekunden bei 48kHz Stereo 16-bit (192KB/s)

    # Timeout: Kein Paket seit X Sekunden → disconnected
    # Erhoet von 2s auf 5s — WiFi hat kurze Aussetzer, sofortiger Fallback verliert Audio
    HEALTH_TIMEOUT = 5.0

    def __init__(self, esp_ip: str = "10.42.0.2",
                 port_16k: int = 12345, port_48k: int = 12346,
                 event_bus=None):
        self.esp_ip = esp_ip
        self.port_16k = port_16k
        self.port_48k = port_48k
        self.event_bus = event_bus

        # Schnelle Ringpuffer: bytearray + read/write Position
        self._ring_16k = bytearray(self.RING_16K_SIZE)
        self._ring_16k_wr = 0   # Schreibposition
        self._ring_16k_rd = 0   # Leseposition
        self._ring_16k_avail = 0  # Verfuegbare Bytes

        self._ring_48k = bytearray(self.RING_48K_SIZE)
        self._ring_48k_wr = 0
        self._ring_48k_rd = 0
        self._ring_48k_avail = 0

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
        self._recv_start_16k = 0.0

        # Sequenznummer-Tracking (Paketverlust-Erkennung)
        self._last_seq_16k = -1
        self._last_seq_48k = -1
        self._packets_lost_16k = 0
        self._packets_lost_48k = 0
        self._packets_total_16k = 0
        self._packets_total_48k = 0
        self._packets_ooo_16k = 0

        # Jitter-Buffer: dict[seq_num] = (audio_data, recv_timestamp)
        self._jitter_buf_16k: dict = {}
        self._jitter_next_seq_16k = -1
        self._jitter_lock_16k = threading.Lock()

        # Software Gain (Multiplikator fuer WiFi-Audio, 0.0 - 3.0)
        self._software_gain = 1.0

        # Quellen-Erzwingung: "auto" (Default), "wifi", "usb"
        self._force_source = "auto"

        # Locks (ein Lock pro Stream, chunk-basiert = kurze Lock-Zeiten)
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

        logger.info(f"WiFiMic v3.0 gestartet, UDP {self.port_16k}/{self.port_48k} "
                    f"(RecvBuf={self.UDP_RECV_BUF // 1024}KB, "
                    f"Jitter={self.JITTER_TIMEOUT_MS}ms)")

    def stop(self):
        """Stoppt alle Threads und schliesst Sockets."""
        self._running = False
        self._close_sockets()
        logger.info("WiFiMic gestoppt")

    def get_audio_chunk(self, rate: int = 16000, duration_ms: int = 100) -> bytes:
        """Gibt Audio-Chunk zurueck — schnelles memoryview-basiertes I/O."""
        if rate == 16000:
            num_bytes = (rate * 2 * duration_ms) // 1000  # 32 Bytes/ms
            return self._ring_read_16k(num_bytes)
        elif rate == 48000:
            num_bytes = (rate * 2 * 2 * duration_ms) // 1000  # 192 Bytes/ms
            return self._ring_read_48k(num_bytes)
        else:
            return b''

    @property
    def connected(self) -> bool:
        """True wenn 16kHz Stream verbunden."""
        if self._force_source == "usb":
            return False
        if self._force_source == "wifi":
            return True
        return self._connected_16k

    @property
    def source(self) -> str:
        """Aktuelle Audio-Quelle: 'wifi', 'usb', oder 'none'."""
        return self._source

    def get_status(self) -> dict:
        """Status-Dict fuer IPC/Panel."""
        total = self._packets_total_16k
        lost = self._packets_lost_16k
        loss_pct = (lost / total * 100) if total > 10 else 0.0

        return {
            "source": self._source,
            "connected_16k": self._connected_16k,
            "connected_48k": self._connected_48k,
            "esp_ip": self.esp_ip,
            "buf_16k_bytes": self._ring_16k_avail,
            "buf_48k_bytes": self._ring_48k_avail,
            "jitter_buf_16k": len(self._jitter_buf_16k),
            "packets_recv_16k": self._packets_recv_16k,
            "packets_recv_48k": self._packets_recv_48k,
            "packets_total_16k": total,
            "packets_lost_16k": lost,
            "packets_ooo_16k": self._packets_ooo_16k,
            "loss_pct_16k": round(loss_pct, 2),
            "software_gain": self._software_gain,
            "force_source": self._force_source,
        }

    def peek_rms(self, num_samples: int = 160) -> float:
        """RMS-Pegel OHNE Buffer zu leeren (160 Samples = 10ms bei 16kHz)."""
        with self._lock_16k:
            avail = self._ring_16k_avail
            if avail < 4:
                return -80.0
            num_bytes = min(num_samples * 2, avail)
            # Letzte N Bytes lesen (Ring-Ende) ohne zu konsumieren
            ring = self._ring_16k
            size = self.RING_16K_SIZE
            end = self._ring_16k_wr
            start = (end - num_bytes) % size
            if start < end:
                raw = bytes(ring[start:end])
            else:
                raw = bytes(ring[start:]) + bytes(ring[:end])

        n = len(raw) // 2
        if n < 2:
            return -80.0
        samples = struct.unpack(f"<{n}h", raw[:n * 2])
        rms = math.sqrt(sum(s * s for s in samples) / n)
        return 20 * math.log10(max(rms, 1) / 32768.0)

    @property
    def software_gain(self) -> float:
        return self._software_gain

    @software_gain.setter
    def software_gain(self, value: float):
        self._software_gain = max(0.0, min(3.0, float(value)))

    def set_force_source(self, mode: str):
        """Audio-Quelle erzwingen: 'auto', 'wifi', oder 'usb'."""
        if mode in ("auto", "wifi", "usb"):
            self._force_source = mode
            logger.info(f"Audio-Quelle erzwungen: {mode}")

    # =========================================================================
    # Ringpuffer — schnelles chunk-basiertes I/O (kein Byte-fuer-Byte)
    # =========================================================================

    def _ring_write_16k(self, data: bytes):
        """Schreibt Audio-Daten in den 16kHz Ringpuffer (chunk-basiert)."""
        n = len(data)
        if n == 0:
            return
        size = self.RING_16K_SIZE

        with self._lock_16k:
            wr = self._ring_16k_wr
            # Pruefen ob Daten in einem Stueck passen
            end = wr + n
            if end <= size:
                self._ring_16k[wr:end] = data
            else:
                # Wrap: zwei Teile schreiben
                first = size - wr
                self._ring_16k[wr:size] = data[:first]
                self._ring_16k[0:n - first] = data[first:]
            self._ring_16k_wr = end % size
            self._ring_16k_avail = min(self._ring_16k_avail + n, size)

    def _ring_read_16k(self, num_bytes: int) -> bytes:
        """Liest Audio-Daten aus dem 16kHz Ringpuffer."""
        with self._lock_16k:
            avail = self._ring_16k_avail
            if avail == 0:
                return b''
            n = min(num_bytes, avail)
            size = self.RING_16K_SIZE
            rd = self._ring_16k_rd
            end = rd + n
            if end <= size:
                chunk = bytes(self._ring_16k[rd:end])
            else:
                first = size - rd
                chunk = bytes(self._ring_16k[rd:size]) + bytes(self._ring_16k[0:n - first])
            self._ring_16k_rd = end % size
            self._ring_16k_avail -= n

        # Software Gain (ausserhalb Lock)
        if self._software_gain != 1.0 and len(chunk) >= 2:
            ns = len(chunk) // 2
            samples = struct.unpack(f"<{ns}h", chunk[:ns * 2])
            g = self._software_gain
            chunk = struct.pack(f"<{ns}h", *(
                max(-32768, min(32767, int(s * g))) for s in samples
            ))
        return chunk

    def _ring_write_48k(self, data: bytes):
        """Schreibt Audio-Daten in den 48kHz Ringpuffer."""
        n = len(data)
        if n == 0:
            return
        size = self.RING_48K_SIZE

        with self._lock_48k:
            wr = self._ring_48k_wr
            end = wr + n
            if end <= size:
                self._ring_48k[wr:end] = data
            else:
                first = size - wr
                self._ring_48k[wr:size] = data[:first]
                self._ring_48k[0:n - first] = data[first:]
            self._ring_48k_wr = end % size
            self._ring_48k_avail = min(self._ring_48k_avail + n, size)

    def _ring_read_48k(self, num_bytes: int) -> bytes:
        """Liest Audio-Daten aus dem 48kHz Ringpuffer."""
        with self._lock_48k:
            avail = self._ring_48k_avail
            if avail == 0:
                return b''
            n = min(num_bytes, avail)
            size = self.RING_48K_SIZE
            rd = self._ring_48k_rd
            end = rd + n
            if end <= size:
                chunk = bytes(self._ring_48k[rd:end])
            else:
                first = size - rd
                chunk = bytes(self._ring_48k[rd:size]) + bytes(self._ring_48k[0:n - first])
            self._ring_48k_rd = end % size
            self._ring_48k_avail -= n
        return chunk

    def _ring_clear_16k(self):
        """16kHz Ringpuffer leeren."""
        with self._lock_16k:
            self._ring_16k_rd = 0
            self._ring_16k_wr = 0
            self._ring_16k_avail = 0

    # =========================================================================
    # UDP Empfang + Jitter-Buffer
    # =========================================================================

    def _create_udp_socket(self, port: int) -> socket.socket:
        """Erstellt UDP-Socket mit grossem Empfangspuffer."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Grosser Kernel-Puffer: 1MB statt default 208KB
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.UDP_RECV_BUF)
            actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            logger.info(f"UDP Port {port}: RecvBuf angefordert={self.UDP_RECV_BUF // 1024}KB "
                        f"tatsaechlich={actual // 1024}KB")
        except Exception as e:
            logger.warning(f"UDP RecvBuf setzen fehlgeschlagen: {e}")
        sock.settimeout(1.0)
        sock.bind(("0.0.0.0", port))
        return sock

    def _recv_loop(self, rate: int):
        """Empfaengt Audio-Daten per UDP mit Sequenznummer-Header + Jitter-Buffer."""
        sock = self._sock_16k if rate == 16000 else self._sock_48k
        label = f"{rate // 1000}kHz"

        # Groesserer recvfrom Buffer: bis zu 10 Pakete auf einmal
        recv_buf_size = (self.PACKET_16K if rate == 16000 else self.PACKET_48K) * 10

        while self._running:
            try:
                data, addr = sock.recvfrom(recv_buf_size)
                if not data or len(data) < self.SEQ_HEADER_SIZE + 2:
                    continue

                # Sequenznummer extrahieren (4 Byte Little-Endian Header)
                seq_num = struct.unpack('<I', data[:self.SEQ_HEADER_SIZE])[0]
                audio_data = data[self.SEQ_HEADER_SIZE:]

                now = time.monotonic()

                if rate == 16000:
                    self._last_recv_16k = now
                    self._packets_recv_16k += 1
                    self._packets_total_16k += 1
                    if self._recv_start_16k == 0.0:
                        self._recv_start_16k = now

                    # Paketverlust erkennen (nur Statistik)
                    if self._last_seq_16k >= 0:
                        expected = (self._last_seq_16k + 1) & 0xFFFFFFFF
                        if seq_num != expected:
                            if seq_num > expected:
                                gap = seq_num - expected
                                if gap < 1000:
                                    self._packets_lost_16k += gap
                                    # Nur alle 100 Verluste loggen (nicht spammen)
                                    if self._packets_lost_16k % 100 < gap:
                                        logger.warning(
                                            f"[16kHz] LOSS: {gap} Pakete "
                                            f"(gesamt: {self._packets_lost_16k}/"
                                            f"{self._packets_total_16k})")
                            elif self._last_seq_16k - seq_num < 0xFFFFF000:
                                self._packets_ooo_16k += 1
                    self._last_seq_16k = seq_num

                    # In Jitter-Buffer einfuegen
                    with self._jitter_lock_16k:
                        self._jitter_buf_16k[seq_num] = (audio_data, now)
                        if self._jitter_next_seq_16k < 0:
                            self._jitter_next_seq_16k = seq_num
                        self._flush_jitter_buffer_16k(now)

                else:
                    # 48kHz: direkt in Ringpuffer (kein Jitter-Buffer)
                    self._last_recv_48k = now
                    self._packets_recv_48k += 1
                    self._packets_total_48k += 1

                    if self._last_seq_48k >= 0:
                        expected = (self._last_seq_48k + 1) & 0xFFFFFFFF
                        if seq_num > expected and (seq_num - expected) < 1000:
                            self._packets_lost_48k += (seq_num - expected)
                    self._last_seq_48k = seq_num

                    self._ring_write_48k(audio_data)

                # Erster Empfang → connected
                if rate == 16000 and not self._connected_16k:
                    self._connected_16k = True
                    self._source = "wifi"
                    logger.info(f"[{label}] UDP von {addr[0]}:{addr[1]} "
                                f"(Seq-Header, Jitter={self.JITTER_TIMEOUT_MS}ms, "
                                f"RecvBuf={self.UDP_RECV_BUF // 1024}KB)")
                    self._fire_source_event("wifi", rate)
                elif rate == 48000 and not self._connected_48k:
                    self._connected_48k = True
                    logger.info(f"[{label}] UDP von {addr[0]}:{addr[1]}")

            except socket.timeout:
                if rate == 16000:
                    now = time.monotonic()
                    with self._jitter_lock_16k:
                        self._flush_jitter_buffer_16k(now)
                continue
            except OSError as e:
                if self._running:
                    logger.warning(f"[{label}] UDP Fehler: {e}")
                    time.sleep(0.5)

    def _flush_jitter_buffer_16k(self, now: float):
        """Spielt Pakete aus dem Jitter-Buffer in den Ringpuffer.

        MUSS mit _jitter_lock_16k gehalten aufgerufen werden.
        - Pakete in Sequenz-Reihenfolge
        - Luecken → Stille (kein Wort verschlucken!)
        - Buffer voll oder aeltestes Paket >100ms → sofort raus
        """
        if not self._jitter_buf_16k:
            return

        oldest_seq = min(self._jitter_buf_16k.keys())
        oldest_time = self._jitter_buf_16k[oldest_seq][1]
        age_ms = (now - oldest_time) * 1000

        buf_full = len(self._jitter_buf_16k) >= self.JITTER_BUF_SIZE
        timeout = age_ms >= self.JITTER_TIMEOUT_MS

        if not buf_full and not timeout:
            return

        next_seq = self._jitter_next_seq_16k
        if next_seq < 0:
            next_seq = oldest_seq

        max_seq = max(self._jitter_buf_16k.keys())
        silence = bytes(self.CHUNK_16K)  # 320 Bytes Stille

        # Batch: alle Chunks sammeln, dann EIN write
        chunks = []
        while next_seq <= max_seq:
            if next_seq in self._jitter_buf_16k:
                audio, _ = self._jitter_buf_16k.pop(next_seq)
                chunks.append(audio)
            else:
                chunks.append(silence)
            next_seq = (next_seq + 1) & 0xFFFFFFFF
            if not self._jitter_buf_16k:
                break

        self._jitter_next_seq_16k = next_seq

        # EIN Ringpuffer-Write fuer alle Chunks (minimale Lock-Zeit)
        if chunks:
            combined = b''.join(chunks)
            self._ring_write_16k(combined)

    # =========================================================================
    # Health Monitor
    # =========================================================================

    def _health_loop(self):
        """Prueft ob UDP-Pakete noch ankommen."""
        initial_deadline = time.time() + 10

        while self._running:
            time.sleep(2)
            now = time.monotonic()

            if self._connected_16k:
                if now - self._last_recv_16k > self.HEALTH_TIMEOUT:
                    self._connected_16k = False
                    logger.warning("[16kHz] Keine Pakete seit 2s")

            if self._connected_48k:
                if now - self._last_recv_48k > self.HEALTH_TIMEOUT:
                    self._connected_48k = False
                    logger.warning("[48kHz] Keine Pakete seit 2s")

            # Fallback auf USB
            if not self._connected_16k and time.time() > initial_deadline:
                if self._source != "usb":
                    self._source = "usb"
                    logger.info("WiFi-Mic weg, Fallback USB")
                    self._fire_source_event("usb", 16000)

            # Wieder connected → WiFi
            if self._connected_16k and self._source != "wifi":
                self._source = "wifi"
                logger.info("WiFi-Mic wieder aktiv")
                self._fire_source_event("wifi", 16000)

    def _close_sockets(self):
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

    esp_ip = sys.argv[1] if len(sys.argv) > 1 else "10.42.0.2"

    mic = WiFiMic(esp_ip=esp_ip)
    mic.start()

    print(f"WiFiMic v3.0 laeuft, ESP32 @ {esp_ip}...")
    print("Ctrl+C zum Beenden")

    try:
        while True:
            time.sleep(1)
            status = mic.get_status()
            rms = mic.peek_rms()
            chunk = mic.get_audio_chunk(rate=16000, duration_ms=100)
            print(f"Src={status['source']} | "
                  f"16k={'OK' if status['connected_16k'] else '--'} | "
                  f"Buf={status['buf_16k_bytes']}B | "
                  f"Loss={status['loss_pct_16k']:.1f}% | "
                  f"RMS={rms:.0f}dB | "
                  f"Chunk={len(chunk)}B")
    except KeyboardInterrupt:
        mic.stop()
        print("\nBeendet.")
