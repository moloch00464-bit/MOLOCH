#!/usr/bin/env python3
"""
M.O.L.O.C.H. Music Visualizer — Audio-Analyse fuer Avatar
===========================================================

Captured Audio vom PipeWire HDMI Monitor-Port, berechnet
Frequenz-Baender und Lautstaerke fuer Avatar-Reaktionen.

KEIN GUI-Modul! Laeuft als Backend-Thread.
Schreibt Analyse-Daten in ein Thread-safe Dict das der Avatar liest.

Capture:     pw-record → stdout pipe → numpy
Analyse:     DC-Block → Hanning → FFT → Band-Energien
Smoothing:   Adaptive EMA (Guardian=smooth, Berserker=hart)
Fallback:    Spotify API Metadaten bei fehlender Audio-Capture

NPU bleibt komplett frei. Kein ML, nur FFT/Heuristik.
"""

import os
import time
import math
import json
import subprocess
import threading
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("MusicVisualizer")


# =============================================================================
# Analyse-Datenstruktur
# =============================================================================

@dataclass
class MusicData:
    """Thread-safe Analyse-Ergebnis eines Frames."""
    rms_volume: float = 0.0       # 0.0-0.15, Gesamtlautstaerke (skaliert)
    bass_energy: float = 0.0      # 0.0-0.15, 20-120 Hz
    mid_energy: float = 0.0       # 0.0-0.15, 120-2000 Hz
    high_energy: float = 0.0      # 0.0-0.15, 4000-12000 Hz
    beat_detected: bool = False    # Bass-Spike erkannt
    is_active: bool = False        # Musik laeuft
    timestamp: float = 0.0        # monotonic


# =============================================================================
# Konstanten
# =============================================================================

# PipeWire HDMI Monitor Sink Name (Pi5, HDMI-1)
HDMI_SINK_NAME = "alsa_output.platform-107c706400.hdmi.hdmi-stereo"

# Audio-Parameter
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024          # Samples pro Chunk (~23ms bei 44.1kHz, weniger Latenz)
ANALYSIS_HZ = 60            # Ziel-Analyse-Rate (schnellere Updates)
MAX_VISUAL_AMP = 0.15       # Maximale visuelle Amplitude
IDLE_TIMEOUT_S = 5.0        # Sekunden ohne Signal → Idle
CPU_THROTTLE_TEMP = 75.0    # Ab hier 15 Hz statt 30 Hz
BEAT_COOLDOWN_S = 0.15      # 150ms Cooldown zwischen Beats
BEAT_THRESHOLD = 1.5        # Bass > 1.5x Durchschnitt = Beat

# Zone-abhaengige Smoothing-Alphas (hoeher = weniger Latenz, mehr Jitter)
# Panel uebernimmt Werte DIREKT (kein zweites Smoothing), also hier moderat halten
ZONE_ALPHAS = {
    "guardian": 0.6,     # ~27ms Lag, kein Jitter weil einziges Smoothing
    "shadow": 0.75,      # ~18ms Lag
    "berserker": 0.9,    # ~8ms, fast roh
}

# Frequenz-Baender (Hz)
BASS_LO, BASS_HI = 20, 120
MID_LO, MID_HI = 120, 2000
HIGH_LO, HIGH_HI = 4000, 12000

# Normalisierungs-Faktoren (kalibriert mit Skinny Puppy - Smothered Hope, 2026-02-24)
# FFT unnormalisiert, Monitor-Capture via stream.capture.sink=true
# Gemessene Roh-Werte: bass=1.5-30, mid=1.2-6.3, high=0.1-1.2
NORM_RMS = 5.0
NORM_BASS = 0.06
NORM_MID = 0.2
NORM_HIGH = 2.0

# Noise Floor (subtrahiert von Band-Energien, Feintuning)
NOISE_FLOOR = 0.0

# Silence Gate: Roh-RMS unter diesem Wert = kein echtes Signal
# spotifyd haelt Kanal offen → Idle-Rauschen bis RMS 0.13, echte Musik > 0.5
SILENCE_RAW_THRESHOLD = 0.20

# Silence Threshold (normalisierter RMS unter diesem Wert = keine Musik)
SILENCE_THRESHOLD = 0.05


# =============================================================================
# MusicVisualizer
# =============================================================================

class MusicVisualizer:
    """
    Audio-Analyse Thread fuer M.O.L.O.C.H. Avatar.

    Captured PipeWire Monitor-Audio, berechnet Frequenz-Baender.
    Singleton via get_music_visualizer().

    Thread-safe: get_data() kann jederzeit aufgerufen werden.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = MusicData()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Smoothing State (intern, volle Range 0-1)
        self._s_rms = 0.0
        self._s_bass = 0.0
        self._s_mid = 0.0
        self._s_high = 0.0
        self._zone = "guardian"

        # Beat Detection State
        self._bass_history: list = []
        self._beat_cooldown = 0.0

        # Capture State
        self._process: Optional[subprocess.Popen] = None
        self._capture_failed = False
        self._last_signal_time = 0.0

        # Vorberechnete FFT-Frequenz-Bins (werden beim ersten Chunk gesetzt)
        self._freqs: Optional[np.ndarray] = None
        self._bass_mask: Optional[np.ndarray] = None
        self._mid_mask: Optional[np.ndarray] = None
        self._high_mask: Optional[np.ndarray] = None

        # Shared Memory Buffer fuer direkte Panel-IPC (22 bytes: 5f+2B)
        import struct as _struct
        self._shm_buf = bytearray(_struct.calcsize("=5f2B"))

    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """Analyse-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MusicVisualizer",
        )
        self._thread.start()
        logger.info("[MUSIC] Visualizer gestartet")

    def stop(self):
        """Analyse-Thread stoppen."""
        self._running = False
        self._kill_capture()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("[MUSIC] Visualizer gestoppt")

    def get_data(self) -> MusicData:
        """Thread-safe Kopie der aktuellen Analyse-Daten."""
        with self._lock:
            return MusicData(
                rms_volume=self._data.rms_volume,
                bass_energy=self._data.bass_energy,
                mid_energy=self._data.mid_energy,
                high_energy=self._data.high_energy,
                beat_detected=self._data.beat_detected,
                is_active=self._data.is_active,
                timestamp=self._data.timestamp,
            )

    def set_zone(self, zone: str):
        """Zone-Update fuer adaptives Smoothing (thread-safe)."""
        self._zone = zone

    # =========================================================================
    # Thread-Logik
    # =========================================================================

    def _run(self):
        """Haupt-Thread: Audio capturen und analysieren."""
        # Nice-Wert erhoehen (weniger Prioritaet)
        try:
            os.nice(10)
        except OSError:
            pass

        while self._running:
            try:
                if not self._capture_failed:
                    self._run_capture_loop()
                else:
                    self._run_fallback_loop()
            except Exception as e:
                logger.error(f"[MUSIC] Visualizer Fehler: {e}")
                time.sleep(2.0)

        # Cleanup
        with self._lock:
            self._data.is_active = False

    def _run_capture_loop(self):
        """PipeWire Monitor Capture Loop."""
        # pw-record vom HDMI Sink Monitor (stream.capture.sink=true fuer Monitor-Capture)
        cmd = [
            "pw-record",
            "--target", HDMI_SINK_NAME,
            "--properties", "stream.capture.sink=true",
            "--format", "s16",
            "--rate", str(SAMPLE_RATE),
            "--channels", "1",
            "-",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                },
            )
        except FileNotFoundError:
            logger.error("[MUSIC] pw-record nicht gefunden")
            self._capture_failed = True
            return
        except Exception as e:
            logger.error(f"[MUSIC] pw-record Start fehlgeschlagen: {e}")
            self._capture_failed = True
            return

        # Kurz warten und pruefen ob pw-record sofort crasht
        time.sleep(0.3)
        if self._process.poll() is not None:
            stderr = ""
            try:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning(f"[MUSIC] pw-record sofort beendet: {stderr[:200]}")
            self._capture_failed = True
            self._process = None
            return

        logger.info("[MUSIC] PipeWire Capture gestartet")

        # Bytes pro Analyse-Chunk (s16 = 2 Bytes/Sample)
        bytes_per_chunk = BUFFER_SIZE * 2
        last_analysis = time.monotonic()

        while self._running and self._process and self._process.poll() is None:
            try:
                raw = self._process.stdout.read(bytes_per_chunk)
                if not raw or len(raw) < bytes_per_chunk:
                    continue

                # Throttling: Analyse-Interval einhalten
                now = time.monotonic()
                interval = self._get_analysis_interval()
                if now - last_analysis < interval:
                    continue
                last_analysis = now

                # PCM s16 → numpy float32 [-1.0, 1.0]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                samples /= 32768.0

                # DC-Block: Mean subtrahieren (einfach und effektiv)
                samples -= np.mean(samples)

                # FFT Analyse
                self._analyze(samples, now)

            except Exception as e:
                logger.debug(f"[MUSIC] Analyse-Fehler: {e}")
                continue

        # pw-record beendet — Fallback aktivieren
        if self._running:
            stderr = ""
            try:
                if self._process and self._process.stderr:
                    stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.warning(f"[MUSIC] pw-record beendet, Fallback. stderr: {stderr[:200]}")
            self._capture_failed = True
            self._kill_capture()

    def _analyze(self, samples: np.ndarray, now: float):
        """FFT + Bandpass Energie berechnen."""
        n = len(samples)
        if n < 64:
            return

        # Frequenz-Masken initialisieren (einmalig)
        if self._freqs is None or len(self._freqs) != n // 2 + 1:
            self._freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
            self._bass_mask = (self._freqs >= BASS_LO) & (self._freqs <= BASS_HI)
            self._mid_mask = (self._freqs >= MID_LO) & (self._freqs <= MID_HI)
            self._high_mask = (self._freqs >= HIGH_LO) & (self._freqs <= HIGH_HI)

        # Hanning Window + FFT (unnormalisiert, Silence Gate reicht)
        windowed = samples * np.hanning(n)
        fft_mag = np.abs(np.fft.rfft(windowed))

        # RMS Volume (Zeitbereich)
        rms_raw = float(np.sqrt(np.mean(samples ** 2)))

        # Silence Gate: PipeWire Rauschen unterdruecken
        if rms_raw < SILENCE_RAW_THRESHOLD:
            rms_norm = 0.0
            bass_norm = 0.0
            mid_norm = 0.0
            high_norm = 0.0
        else:
            # Band-Energien (Frequenzbereich, quadratischer Mittelwert)
            bass_raw = max(0.0, self._band_energy(fft_mag, self._bass_mask) - NOISE_FLOOR)
            mid_raw = max(0.0, self._band_energy(fft_mag, self._mid_mask) - NOISE_FLOOR)
            high_raw = max(0.0, self._band_energy(fft_mag, self._high_mask) - NOISE_FLOOR)

            # Normalisierung auf 0-1 (empirische Faktoren)
            rms_norm = min(1.0, rms_raw * NORM_RMS)
            bass_norm = min(1.0, bass_raw * NORM_BASS)
            mid_norm = min(1.0, mid_raw * NORM_MID)
            high_norm = min(1.0, high_raw * NORM_HIGH)

        # Adaptive EMA Smoothing (Zone-abhaengig)
        alpha = ZONE_ALPHAS.get(self._zone, 0.15)
        self._s_rms += (rms_norm - self._s_rms) * alpha
        self._s_bass += (bass_norm - self._s_bass) * alpha
        self._s_mid += (mid_norm - self._s_mid) * alpha
        self._s_high += (high_norm - self._s_high) * alpha

        # Beat Detection (Bass-Spike ueber Durchschnitt)
        self._bass_history.append(bass_norm)
        if len(self._bass_history) > 30:
            self._bass_history.pop(0)
        avg_bass = sum(self._bass_history) / len(self._bass_history)
        beat = (bass_norm > avg_bass * BEAT_THRESHOLD
                and bass_norm > 0.3
                and now > self._beat_cooldown)
        if beat:
            self._beat_cooldown = now + BEAT_COOLDOWN_S

        # Signal-Erkennung (Idle nach IDLE_TIMEOUT_S Stille)
        is_active = rms_norm > SILENCE_THRESHOLD
        if is_active:
            self._last_signal_time = now
        elif now - self._last_signal_time > IDLE_TIMEOUT_S:
            is_active = False
        else:
            is_active = True  # Noch im Timeout-Fenster

        # Ergebnis schreiben (auf MAX_VISUAL_AMP skaliert)
        with self._lock:
            self._data.rms_volume = self._s_rms * MAX_VISUAL_AMP
            self._data.bass_energy = self._s_bass * MAX_VISUAL_AMP
            self._data.mid_energy = self._s_mid * MAX_VISUAL_AMP
            self._data.high_energy = self._s_high * MAX_VISUAL_AMP
            self._data.beat_detected = beat
            self._data.is_active = is_active
            self._data.timestamp = now

        # Direkte IPC: Music-Daten sofort in /dev/shm/ schreiben (60 Hz)
        # Panel liest diese Datei direkt — kein Umweg ueber Service Status-JSON
        try:
            import struct as _struct
            # Binary statt JSON fuer Speed: 5 floats + 1 byte active + 1 byte beat
            _struct.pack_into(
                "=5f2B", self._shm_buf, 0,
                self._s_rms * MAX_VISUAL_AMP,
                self._s_bass * MAX_VISUAL_AMP,
                self._s_mid * MAX_VISUAL_AMP,
                self._s_high * MAX_VISUAL_AMP,
                now,
                1 if is_active else 0,
                1 if beat else 0,
            )
            with open("/dev/shm/moloch_music.bin", "wb") as f:
                f.write(self._shm_buf)
        except Exception:
            pass

    @staticmethod
    def _band_energy(fft_mag: np.ndarray, mask: np.ndarray) -> float:
        """Mittlere Energie in einem Frequenz-Band."""
        if not np.any(mask):
            return 0.0
        return float(np.sqrt(np.mean(fft_mag[mask] ** 2)))

    def _get_analysis_interval(self) -> float:
        """Analyse-Interval, throttled bei CPU > 75°C."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read().strip()) / 1000.0
            if temp > CPU_THROTTLE_TEMP:
                return 1.0 / 15.0  # 15 Hz
        except Exception:
            pass
        return 1.0 / ANALYSIS_HZ  # 30 Hz

    # =========================================================================
    # Fallback: Spotify API Metadaten
    # =========================================================================

    def _run_fallback_loop(self):
        """Fallback: Liest Status-JSON, generiert sanften Puls wenn Musik laeuft."""
        logger.info("[MUSIC] Fallback-Modus aktiv (keine PipeWire Capture)")
        status_path = "/dev/shm/moloch_status.json"

        while self._running and self._capture_failed:
            try:
                is_playing = False
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        status = json.load(f)
                    spotify = status.get("spotify", {})
                    track = spotify.get("current_track")
                    is_playing = bool(track and track.get("is_playing", False))

                now = time.monotonic()
                with self._lock:
                    if is_playing:
                        # Simulierter Musik-Puls (Sinus-basiert, subtil)
                        phase = now * 2.0
                        self._data.rms_volume = 0.05 + 0.03 * math.sin(phase)
                        self._data.bass_energy = 0.04 + 0.03 * math.sin(phase * 0.5)
                        self._data.mid_energy = 0.03 + 0.02 * math.sin(phase * 1.3)
                        self._data.high_energy = 0.02 + 0.01 * math.sin(phase * 2.1)
                        self._data.beat_detected = False
                        self._data.is_active = True
                    else:
                        self._data.rms_volume = 0.0
                        self._data.bass_energy = 0.0
                        self._data.mid_energy = 0.0
                        self._data.high_energy = 0.0
                        self._data.beat_detected = False
                        self._data.is_active = False
                    self._data.timestamp = now

            except Exception:
                pass

            # 2 Hz Polling im Fallback-Modus (kaum CPU)
            time.sleep(0.5)

    # =========================================================================
    # Hilfsfunktionen
    # =========================================================================

    def _kill_capture(self):
        """pw-record Prozess sauber beenden."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[MusicVisualizer] = None
_instance_lock = threading.Lock()


def get_music_visualizer() -> MusicVisualizer:
    """Globale MusicVisualizer Instanz."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MusicVisualizer()
    return _instance
