#!/usr/bin/env python3
"""
M.O.L.O.C.H. Music Listener — Echtzeit FFT via WiFi-Mic 48kHz
=============================================================

MOLOCH hoert Musik mit seinem eigenen Ohr (ReSpeaker 48kHz via ESP32 UDP).

Zwei-Thread Echtzeit-Architektur:
  [Receive-Thread]  UDP recv → decode → sample_queue.put_nowait()
  [Analysis-Thread] sample_queue.get() → Ring-Buffer → Beat + FFT

Beat-Detection: RMS-Spike auf jedem Chunk (~5ms Latenz, kein FFT noetig)
Band-Analyse:   FFT auf Ring-Buffer (2048 Samples, 40Hz, genaue Frequenzen)

Kein Throttle im Receive-Thread — Analyse blockiert auf Queue.
Bei Queue-Ueberlauf: aeltestes Sample droppen → immer frischeste Daten.

Events:
  music.beat             {'strength': float, 'bpm_estimate': float}  ~5ms Latenz
  music.frequency_bands  {'bass': 0-1, 'mid': 0-1, 'high': 0-1, 'overall_energy': 0-1}

Schreibt /dev/shm/moloch_music.bin (Format: =5f2B) fuer panel_avatar.py.
"""

import collections
import logging
import queue
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from core.moloch_event_bus import get_event_bus, PRIO_INFO

logger = logging.getLogger("MusicListener")

# =========================================================================
# Konfiguration
# =========================================================================

UDP_HOST = "0.0.0.0"
UDP_PORT = 12346

SAMPLE_RATE  = 48000
FFT_SIZE     = 2048   # Ring-Buffer Groesse fuer Band-FFT (42ms Fenster)
BAND_HZ      = 40     # Band-Analyse Rate (25ms Intervall)

# Frequenz-Baender (Hz)
BASS_LO, BASS_HI = 20,   250
MID_LO,  MID_HI  = 250,  4000
HIGH_LO, HIGH_HI = 4000, 16000

# Beat Detection (RMS-basiert, laeuft auf jedem Chunk ~5ms)
BEAT_THRESHOLD    = 1.35   # Chunk-RMS > Durchschnitt * Faktor = Beat
BEAT_COOLDOWN_MS  = 180    # Mindestabstand zwischen Beats (ms)
BEAT_HISTORY_LEN  = 40     # Anzahl Chunks fuer rollierenden RMS-Durchschnitt

# Normalisierung (empirisch fuer ReSpeaker 48kHz WiFi-Stream)
NORM_BASS = 0.020
NORM_MID  = 0.060
NORM_HIGH = 0.300
NORM_RMS  = 20.0

# Silence Gate (unter diesem Roh-RMS: Stille)
SILENCE_THRESHOLD = 0.002

# EMA Smoothing fuer Band-Werte (0.55 = ~125ms bis 90%)
EMA_ALPHA = 0.55

# Queue-Groesse: max N Chunks Backlog (bei Ueberlauf droppen)
QUEUE_MAXSIZE = 20

# IPC Binary (kompatibel mit music_visualizer.py + panel_avatar.py)
SHM_PATH      = "/dev/shm/moloch_music.bin"
SHM_FORMAT    = "=5f2B"   # rms, bass, mid, high, ts, active, beat
MAX_VISUAL_AMP = 0.15


class MusicListener:
    """
    Echtzeit-Musikanalyse vom ESP32 WiFi-Mic (48kHz UDP).

    Receive-Thread: maximal schnell, nur decode + queue.put
    Analysis-Thread: blockiert auf queue, Beat sofort, Bands 40Hz
    """

    def __init__(self):
        self._bus = get_event_bus()
        self._lock = threading.Lock()
        self._running = False
        self._active = False

        # Threads
        self._recv_thread:     Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Producer-Consumer Queue (Mono-Chunks als np.ndarray)
        self._sample_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

        # Ring-Buffer fuer Band-FFT (Thread-sicher: nur vom Analysis-Thread geschrieben)
        self._ring: collections.deque = collections.deque(maxlen=FFT_SIZE)

        # Vorberechnete FFT-Masken (fuer FFT_SIZE)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        self._bass_mask = (freqs >= BASS_LO) & (freqs <= BASS_HI)
        self._mid_mask  = (freqs >= MID_LO)  & (freqs <= MID_HI)
        self._high_mask = (freqs >= HIGH_LO) & (freqs <= HIGH_HI)

        # Geglaettete Band-Werte (nur Analysis-Thread schreibt)
        self._s_bass = 0.0
        self._s_mid  = 0.0
        self._s_high = 0.0
        self._s_rms  = 0.0

        # Beat-State (RMS-basiert)
        self._rms_history: collections.deque = collections.deque(maxlen=BEAT_HISTORY_LEN)
        self._beat_cooldown_until = 0.0
        self._last_beat_time      = 0.0
        self._bpm_estimate        = 0.0

        # Band-Analyse Throttle
        self._last_band_time = 0.0
        self._band_interval  = 1.0 / BAND_HZ

        # IPC Buffer
        self._shm_buf = bytearray(struct.calcsize(SHM_FORMAT))

    # =========================================================================
    # Start / Stop
    # =========================================================================

    def start(self):
        if self._running:
            return
        self._running = True

        # Socket oeffnen
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self._sock.settimeout(0.5)
            self._sock.bind((UDP_HOST, UDP_PORT))
        except Exception as e:
            logger.error(f"[MUSIC-LISTENER] Socket-Fehler: {e}")
            self._running = False
            return

        # Threads starten
        self._recv_thread = threading.Thread(
            target=self._run_receive, daemon=True, name="MusicRcv"
        )
        self._analysis_thread = threading.Thread(
            target=self._run_analysis, daemon=True, name="MusicFFT"
        )
        self._recv_thread.start()
        self._analysis_thread.start()

        self._bus.subscribe("mic.mode_changed", self._on_mic_mode_changed, priority=5)
        logger.info(f"[MUSIC-LISTENER] Gestartet (Port {UDP_PORT}, 2-Thread Echtzeit)")

    def stop(self):
        self._running = False
        self._bus.unsubscribe("mic.mode_changed", self._on_mic_mode_changed)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        # Queue aufwecken damit Analysis-Thread beendet
        try:
            self._sample_queue.put_nowait(None)
        except Exception:
            pass
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)
        self._write_idle_to_shm()
        logger.info("[MUSIC-LISTENER] Gestoppt")

    # =========================================================================
    # Event Handler
    # =========================================================================

    def _on_mic_mode_changed(self, event):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        rate = payload.get("rate", 16000)
        with self._lock:
            self._active = (rate == 48000)
        if self._active:
            logger.info("[MUSIC-LISTENER] 48kHz aktiv — Echtzeit-Analyse laeuft")
        else:
            logger.info("[MUSIC-LISTENER] 16kHz — Analyse pausiert")
            self._write_idle_to_shm()

    # =========================================================================
    # Thread 1: Receive (so schnell wie moeglich)
    # =========================================================================

    def _run_receive(self):
        """Empfaengt UDP-Pakete und schreibt dekodierte Mono-Chunks in Queue."""
        logger.info(f"[MUSIC-LISTENER] Receive-Thread aktiv (Port {UDP_PORT})")

        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"[MUSIC-LISTENER] Recv-Fehler: {e}")
                break

            with self._lock:
                active = self._active
            if not active:
                continue

            # Decode: int16 Stereo → float32 Mono
            try:
                s = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                s /= 32768.0
                if len(s) >= 2 and len(s) % 2 == 0:
                    mono = (s[::2] + s[1::2]) * 0.5
                elif len(s) >= 1:
                    mono = s[::2]
                else:
                    continue
                mono -= np.mean(mono)  # DC-Block
            except Exception:
                continue

            # Queue: bei Ueberlauf aeltestes droppen → immer frischeste Daten
            if self._sample_queue.full():
                try:
                    self._sample_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._sample_queue.put_nowait(mono)
            except queue.Full:
                pass

    # =========================================================================
    # Thread 2: Analyse (blockiert auf Queue)
    # =========================================================================

    def _run_analysis(self):
        """Holt Chunks aus Queue, fuehrt Beat + Band-Analyse durch."""
        logger.info("[MUSIC-LISTENER] Analysis-Thread aktiv")

        while self._running:
            # Auf neuen Chunk warten (blockiert, kein Polling!)
            try:
                chunk = self._sample_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if chunk is None:  # Stop-Signal
                break

            now = time.monotonic()

            # Ring-Buffer fuellen
            self._ring.extend(chunk.tolist())

            # --- BEAT: jeder Chunk, ~5ms Latenz, kein FFT ---
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
            self._rms_history.append(chunk_rms)
            avg_rms = sum(self._rms_history) / len(self._rms_history)

            beat = (
                chunk_rms > avg_rms * BEAT_THRESHOLD
                and chunk_rms > SILENCE_THRESHOLD
                and now > self._beat_cooldown_until
            )
            if beat:
                self._beat_cooldown_until = now + BEAT_COOLDOWN_MS / 1000.0
                if self._last_beat_time > 0:
                    interval = now - self._last_beat_time
                    if 0.2 < interval < 2.0:
                        bpm = 60.0 / interval
                        self._bpm_estimate = bpm * 0.8 + self._bpm_estimate * 0.2
                self._last_beat_time = now
                # Beat sofort publishen + SHM schreiben
                self._bus.publish(
                    event_type="music.beat",
                    source="music_listener",
                    priority=PRIO_INFO,
                    payload={
                        "strength":     round(chunk_rms * NORM_RMS, 3),
                        "bpm_estimate": round(self._bpm_estimate, 1),
                    },
                )
                self._write_to_shm(now, active=True, beat=True)

            # --- BAND-FFT: 40Hz throttled, genaue Frequenzen ---
            if now - self._last_band_time >= self._band_interval:
                self._last_band_time = now
                self._analyze_bands(now, beat)

    def _analyze_bands(self, now: float, beat: bool):
        """FFT-Band-Analyse auf dem Ring-Buffer (2048 Samples)."""
        if len(self._ring) < FFT_SIZE:
            return  # Noch nicht genug Daten

        samples = np.array(list(self._ring), dtype=np.float32)
        rms_raw = float(np.sqrt(np.mean(samples ** 2)))

        if rms_raw < SILENCE_THRESHOLD:
            # Stille: Werte exponentiell abklingen lassen
            self._s_bass *= 0.8
            self._s_mid  *= 0.8
            self._s_high *= 0.8
            self._s_rms  *= 0.8
            self._write_to_shm(now, active=True, beat=beat)
            return

        # Hanning + FFT
        windowed = samples * np.hanning(FFT_SIZE)
        fft_mag = np.abs(np.fft.rfft(windowed))

        bass_raw = self._band_rms(fft_mag, self._bass_mask)
        mid_raw  = self._band_rms(fft_mag, self._mid_mask)
        high_raw = self._band_rms(fft_mag, self._high_mask)

        # Normalisierung
        rms_norm  = min(1.0, rms_raw  * NORM_RMS)
        bass_norm = min(1.0, bass_raw * NORM_BASS)
        mid_norm  = min(1.0, mid_raw  * NORM_MID)
        high_norm = min(1.0, high_raw * NORM_HIGH)

        # EMA Smoothing
        a = EMA_ALPHA
        self._s_rms  += (rms_norm  - self._s_rms)  * a
        self._s_bass += (bass_norm - self._s_bass) * a
        self._s_mid  += (mid_norm  - self._s_mid)  * a
        self._s_high += (high_norm - self._s_high) * a

        # IPC + Event
        self._write_to_shm(now, active=True, beat=beat)
        self._bus.publish(
            event_type="music.frequency_bands",
            source="music_listener",
            priority=PRIO_INFO,
            payload={
                "bass":           round(self._s_bass, 3),
                "mid":            round(self._s_mid,  3),
                "high":           round(self._s_high, 3),
                "overall_energy": round(self._s_rms,  3),
            },
        )

    # =========================================================================
    # IPC Binary
    # =========================================================================

    def _write_to_shm(self, now: float, active: bool, beat: bool):
        """Schreibt Daten in /dev/shm/moloch_music.bin (panel_avatar.py Format)."""
        try:
            struct.pack_into(
                SHM_FORMAT, self._shm_buf, 0,
                self._s_rms  * MAX_VISUAL_AMP,
                self._s_bass * MAX_VISUAL_AMP,
                self._s_mid  * MAX_VISUAL_AMP,
                self._s_high * MAX_VISUAL_AMP,
                now,
                1 if active else 0,
                1 if beat else 0,
            )
            with open(SHM_PATH, "wb") as f:
                f.write(self._shm_buf)
        except Exception:
            pass

    def _write_idle_to_shm(self):
        try:
            buf = struct.pack(SHM_FORMAT, 0.0, 0.0, 0.0, 0.0, time.monotonic(), 0, 0)
            with open(SHM_PATH, "wb") as f:
                f.write(buf)
        except Exception:
            pass

    @staticmethod
    def _band_rms(fft_mag: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.sqrt(np.mean(fft_mag[mask] ** 2)))


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MusicListener] = None
_instance_lock = threading.Lock()


def get_music_listener() -> MusicListener:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MusicListener()
    return _instance
