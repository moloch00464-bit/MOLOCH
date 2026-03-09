#!/usr/bin/env python3
"""
M.O.L.O.C.H. Music Listener — FFT-Analyse via WiFi-Mic 48kHz
=============================================================

MOLOCH hoert Musik mit seinem eigenen Ohr (ReSpeaker 48kHz via ESP32 UDP).

UDP Port 12346: 48kHz Stereo, 960 Bytes/Paket = 240 Samples pro Kanal (float32).
Nur aktiv wenn mic.mode_changed → 48kHz (Musik-Modus).

FFT-Analyse:
  Bass (20-250 Hz)   — Iris-Puls
  Mid  (250-4kHz)    — Glow
  High (4-16kHz)     — Textur/Strahlen
  Beat Detection     — Pupillen-Kontraktion

Events (20x/Sek):
  music.beat             {'strength': float, 'bpm_estimate': float}
  music.frequency_bands  {'bass': 0-1, 'mid': 0-1, 'high': 0-1, 'overall_energy': 0-1}

Schreibt auch in /dev/shm/moloch_music.bin (Format: =5f2B)
→ panel_avatar.py liest diese Datei direkt (kein Umweg noetig).

Kein librosa! Nur numpy + scipy.

Singleton: get_music_listener()
"""

import logging
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from core.moloch_event_bus import get_event_bus, PRIO_INFO

logger = logging.getLogger("MusicListener")

# UDP Konfiguration
UDP_HOST = "0.0.0.0"
UDP_PORT = 12346
UDP_PACKET_SIZE = 960  # Bytes — 48kHz Stereo, 240 Samples/Kanal @ int16

# Audio Analyse
SAMPLE_RATE = 48000
FFT_SIZE = 2048        # Samples fuer FFT (21ms Fenster)
UPDATE_RATE_HZ = 40    # Event-Rate (25ms, schnellere Reaktion)

# Frequenz-Baender (Hz)
BASS_LO, BASS_HI   = 20,   250
MID_LO,  MID_HI    = 250,  4000
HIGH_LO, HIGH_HI   = 4000, 16000

# Beat Detection
BEAT_THRESHOLD = 1.3      # Bass > 1.3x Durchschnitt = Beat
BEAT_COOLDOWN_MS = 200    # Mindestabstand zwischen Beats

# Normalisierung (empirisch fuer ReSpeaker 48kHz WiFi-Stream)
NORM_BASS = 0.020
NORM_MID  = 0.060
NORM_HIGH = 0.300
NORM_RMS  = 20.0

# Silence Gate
SILENCE_RAW_THRESHOLD = 0.002

# IPC Binary Format (kompatibel mit music_visualizer.py und panel_avatar.py)
SHM_PATH = "/dev/shm/moloch_music.bin"
SHM_FORMAT = "=5f2B"  # rms, bass, mid, high, timestamp, active, beat
SHM_SIZE = struct.calcsize(SHM_FORMAT)
MAX_VISUAL_AMP = 0.15  # Maximale visuelle Amplitude (Skalierung fuer IPC)


class MusicListener:
    """
    Empfaengt UDP 48kHz Audio vom ESP32 WiFi-Mic,
    berechnet FFT-Baender, Beat-Detection,
    und publisht Events + schreibt IPC-Binary.
    """

    def __init__(self):
        self._bus = get_event_bus()
        self._lock = threading.Lock()
        self._running = False
        self._active = False  # Wird True wenn mic.mode_changed → 48kHz

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Analyse-Buffer (rolling 2048 Samples, Mono-Mix aus Stereo)
        self._sample_buffer: np.ndarray = np.zeros(FFT_SIZE, dtype=np.float32)

        # Vorberechnete FFT-Masken
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
        self._bass_mask = (freqs >= BASS_LO) & (freqs <= BASS_HI)
        self._mid_mask  = (freqs >= MID_LO)  & (freqs <= MID_HI)
        self._high_mask = (freqs >= HIGH_LO) & (freqs <= HIGH_HI)

        # Smoothing State (EMA, alpha=0.3)
        self._s_bass = 0.0
        self._s_mid  = 0.0
        self._s_high = 0.0
        self._s_rms  = 0.0
        EMA_ALPHA = 0.55  # Schnelle Reaktion auf Energie-Wechsel (~125ms bis 90%)
        self._alpha = EMA_ALPHA

        # Beat Detection State
        self._bass_history: list = []
        self._beat_cooldown_until = 0.0
        self._last_beat_time = 0.0
        self._bpm_estimate = 0.0

        # Event-Rate Throttle
        self._last_event_time = 0.0
        self._event_interval = 1.0 / UPDATE_RATE_HZ

        # IPC Binary Buffer
        self._shm_buf = bytearray(SHM_SIZE)

    # =========================================================================
    # Start / Stop
    # =========================================================================

    def start(self):
        """Startet UDP-Socket und Analyse-Thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MusicListener"
        )
        self._thread.start()
        # Event-Bus: auf mic.mode_changed hoeren
        self._bus.subscribe("mic.mode_changed", self._on_mic_mode_changed, priority=5)
        logger.info("[MUSIC-LISTENER] Gestartet (wartet auf 48kHz-Modus)")

    def stop(self):
        """Stoppt den Listener."""
        self._running = False
        self._bus.unsubscribe("mic.mode_changed", self._on_mic_mode_changed)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._write_idle_to_shm()
        logger.info("[MUSIC-LISTENER] Gestoppt")

    # =========================================================================
    # Event Handler
    # =========================================================================

    def _on_mic_mode_changed(self, event):
        """Aktiviert/deaktiviert Analyse je nach Mic-Modus."""
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        rate = payload.get("rate", 16000)
        with self._lock:
            self._active = (rate == 48000)
        if self._active:
            logger.info("[MUSIC-LISTENER] 48kHz aktiv — Musik-Analyse laeuft")
        else:
            logger.info("[MUSIC-LISTENER] 16kHz — Musik-Analyse pausiert")
            self._write_idle_to_shm()

    # =========================================================================
    # Haupt-Thread
    # =========================================================================

    def _run(self):
        """UDP empfangen und analysieren."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(0.5)
            self._sock.bind((UDP_HOST, UDP_PORT))
            logger.info(f"[MUSIC-LISTENER] UDP Socket gebunden auf Port {UDP_PORT}")
        except Exception as e:
            logger.error(f"[MUSIC-LISTENER] Socket-Fehler: {e}")
            self._running = False
            return

        while self._running:
            try:
                raw, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"[MUSIC-LISTENER] Empfangsfehler: {e}")
                break

            with self._lock:
                active = self._active
            if not active:
                continue  # Paket ignorieren wenn nicht im Musik-Modus

            self._process_packet(raw)

        try:
            self._sock.close()
        except Exception:
            pass

    def _process_packet(self, raw: bytes):
        """Ein UDP-Paket verarbeiten: Decode → Buffer → FFT → Events."""
        try:
            # int16 Stereo → float32 Mono-Mix
            # 960 Bytes = 480 int16 Samples = 240 Stereo-Paare
            samples_raw = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            samples_raw /= 32768.0

            if len(samples_raw) >= 2:
                # Stereo → Mono (L+R / 2)
                if len(samples_raw) % 2 == 0:
                    mono = (samples_raw[::2] + samples_raw[1::2]) * 0.5
                else:
                    mono = samples_raw[::2]
            else:
                return

            # DC-Block
            mono -= np.mean(mono)

            # Rolling Buffer fuellen
            n = len(mono)
            if n >= FFT_SIZE:
                self._sample_buffer = mono[-FFT_SIZE:]
            else:
                self._sample_buffer = np.roll(self._sample_buffer, -n)
                self._sample_buffer[-n:] = mono

        except Exception as e:
            logger.debug(f"[MUSIC-LISTENER] Packet-Decode Fehler: {e}")
            return

        # FFT-Analyse throttlen
        now = time.monotonic()
        if now - self._last_event_time < self._event_interval:
            return
        self._last_event_time = now

        self._analyze(now)

    def _analyze(self, now: float):
        """FFT-Analyse des aktuellen Buffers."""
        samples = self._sample_buffer.copy()

        # Silence Gate
        rms_raw = float(np.sqrt(np.mean(samples ** 2)))
        if rms_raw < SILENCE_RAW_THRESHOLD:
            # Stille — alles auf 0 fahren
            self._s_bass *= 0.7
            self._s_mid  *= 0.7
            self._s_high *= 0.7
            self._s_rms  *= 0.7
            self._write_to_shm(now, active=True, beat=False)
            return

        # Hanning Window + FFT
        windowed = samples * np.hanning(FFT_SIZE)
        fft_mag = np.abs(np.fft.rfft(windowed))

        # Band-Energien (quadratischer Mittelwert)
        bass_raw = self._band_rms(fft_mag, self._bass_mask)
        mid_raw  = self._band_rms(fft_mag, self._mid_mask)
        high_raw = self._band_rms(fft_mag, self._high_mask)

        # Normalisierung auf 0-1
        rms_norm  = min(1.0, rms_raw  * NORM_RMS)
        bass_norm = min(1.0, bass_raw * NORM_BASS)
        mid_norm  = min(1.0, mid_raw  * NORM_MID)
        high_norm = min(1.0, high_raw * NORM_HIGH)

        # EMA Smoothing
        a = self._alpha
        self._s_rms  += (rms_norm  - self._s_rms)  * a
        self._s_bass += (bass_norm - self._s_bass) * a
        self._s_mid  += (mid_norm  - self._s_mid)  * a
        self._s_high += (high_norm - self._s_high) * a

        # Beat Detection
        self._bass_history.append(bass_norm)
        if len(self._bass_history) > 30:
            self._bass_history.pop(0)
        avg_bass = sum(self._bass_history) / len(self._bass_history)

        beat = (
            bass_norm > avg_bass * BEAT_THRESHOLD
            and bass_norm > 0.25
            and now > self._beat_cooldown_until
        )
        if beat:
            self._beat_cooldown_until = now + BEAT_COOLDOWN_MS / 1000.0
            if self._last_beat_time > 0:
                interval = now - self._last_beat_time
                bpm = 60.0 / interval if 0.2 < interval < 2.0 else self._bpm_estimate
                self._bpm_estimate = bpm * 0.8 + self._bpm_estimate * 0.2  # EMA
            self._last_beat_time = now

        # IPC Binary schreiben
        self._write_to_shm(now, active=True, beat=beat)

        # Events publishen
        self._publish_events(beat)

    def _publish_events(self, beat: bool):
        """Events auf den Event Bus schreiben."""
        # Frequenz-Baender Event (20x/Sek)
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
        # Beat Event (nur wenn Beat erkannt)
        if beat:
            self._bus.publish(
                event_type="music.beat",
                source="music_listener",
                priority=PRIO_INFO,
                payload={
                    "strength":     round(self._s_bass, 3),
                    "bpm_estimate": round(self._bpm_estimate, 1),
                },
            )

    # =========================================================================
    # IPC Binary
    # =========================================================================

    def _write_to_shm(self, now: float, active: bool, beat: bool):
        """Schreibt Analyse-Daten in /dev/shm/moloch_music.bin.

        Format identisch mit music_visualizer.py: =5f2B
        (rms, bass, mid, high, timestamp, active_byte, beat_byte)
        Skaliert auf MAX_VISUAL_AMP fuer panel_avatar.py.
        """
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
        """Nullwerte in IPC schreiben (Musik-Modus inaktiv)."""
        try:
            buf = struct.pack(SHM_FORMAT, 0.0, 0.0, 0.0, 0.0, time.monotonic(), 0, 0)
            with open(SHM_PATH, "wb") as f:
                f.write(buf)
        except Exception:
            pass

    @staticmethod
    def _band_rms(fft_mag: np.ndarray, mask: np.ndarray) -> float:
        """Quadratischer Mittelwert eines Frequenz-Bands."""
        if not np.any(mask):
            return 0.0
        return float(np.sqrt(np.mean(fft_mag[mask] ** 2)))


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[MusicListener] = None
_instance_lock = threading.Lock()


def get_music_listener() -> MusicListener:
    """Singleton-Zugriff auf den MusicListener."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MusicListener()
    return _instance
