"""Music-Listener (Sprint-2 Fix-3 NEU 2026-05-10).

FFT + Beat-Detection auf 48kHz Mic-Stream. Publisht events:
  - music.beat: {bpm, energy, ts}
  - music.frequency_bands: {bass, mid, treble, ts}

Subscribers:
  - personality_engine.tension_modulation_from_music()
  - chat_server /api/state/current music_beat_phase

Architektur: lazy-init Singleton, opt-in via spotify_bridge bei
music.playing-Event. Stoppt bei music.stopped.

NB: Skeleton-Implementierung. Echte FFT-Berechnung in process_block()
ist minimal (sliding-window energy detection, kein BPM-tracker noch).
Erweiterung folgt iterativ.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

_BEAT_HISTORY_LEN = 8  # rolling window for BPM-Schaetzung
_MIN_BEAT_INTERVAL_S = 0.25  # >= 240 BPM unwahrscheinlich
_MAX_BEAT_INTERVAL_S = 2.0   # < 30 BPM unwahrscheinlich
_ENERGY_THRESHOLD = 0.02     # FFT-bass-energy fuer Beat-Detection


class MusicListener:
    """FFT + Beat-Detection auf Audio-Stream."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._last_beat_ts = 0.0
        self._last_beat = None  # {bpm, energy, ts}
        self._beat_intervals: deque = deque(maxlen=_BEAT_HISTORY_LEN)
        self._energy_history: deque = deque(maxlen=20)
        self._last_freq_bands = {"bass": 0.0, "mid": 0.0, "treble": 0.0, "ts": 0.0}
        # PC-Spec: mic.mode_changed subscribe -> Auto-Start bei rate==48000
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().subscribe("mic.mode_changed", self._on_mic_mode_changed)
        except Exception as e:
            logger.debug(f"[music_listener] event_bus subscribe fail: {e}")

    def _on_mic_mode_changed(self, event):
        """Subscribe-Handler fuer mic.mode_changed events (PC-Spec)."""
        try:
            data = event.get("data", {}) if isinstance(event, dict) else {}
            rate = int(data.get("rate", 0))
            if rate == 48000:
                self.start()
            elif rate == 16000:
                self.stop()
        except Exception as e:
            logger.debug(f"[music_listener] mic.mode_changed handler err: {e}")

    def start(self):
        """Setzt running-Flag. Audio-Akquise faellt unter audio_pipeline."""
        with self._lock:
            self._running = True
            logger.info("[music_listener] gestartet")

    def stop(self):
        """Stoppt Beat-Tracking."""
        with self._lock:
            self._running = False
            self._beat_intervals.clear()
            self._energy_history.clear()
            logger.info("[music_listener] gestoppt")

    def is_running(self) -> bool:
        return self._running

    def process_block(self, audio_block):
        """Empfaengt Audio-Block (numpy array, float32, 48kHz mono).

        Macht minimale FFT-Bandsplit + Energy-Spike-Detection fuer Beat.
        """
        if not self._running:
            return
        try:
            import numpy as np
            samples = np.asarray(audio_block).flatten()
            if samples.size < 256:
                return

            energy = float(np.mean(samples ** 2))

            try:
                from numpy.fft import rfft
                spec = np.abs(rfft(samples[:1024])) if samples.size >= 1024 else np.abs(rfft(samples))
                if len(spec) >= 100:
                    bass = float(np.mean(spec[2:9]))
                    mid = float(np.mean(spec[10:50]))
                    treble = float(np.mean(spec[50:200]))
                    now_ts = time.time()
                    self._last_freq_bands = {
                        "bass": bass, "mid": mid, "treble": treble, "ts": now_ts,
                    }
                    # PC-Spec: music.frequency_bands event-publish
                    # gedrosselt auf alle ~500ms (sonst Event-Spam)
                    if not hasattr(self, "_last_freq_publish_ts"):
                        self._last_freq_publish_ts = 0.0
                    if now_ts - self._last_freq_publish_ts > 0.5:
                        self._last_freq_publish_ts = now_ts
                        try:
                            from core.moloch_event_bus import get_event_bus
                            get_event_bus().publish(
                                "music.frequency_bands",
                                {"low": round(bass, 4), "mid": round(mid, 4),
                                 "high": round(treble, 4), "ts": now_ts},
                                source="music_listener", priority=3,
                            )
                        except Exception:
                            pass
            except Exception:
                pass

            self._energy_history.append(energy)
            if len(self._energy_history) >= 5:
                avg = sum(self._energy_history) / len(self._energy_history)
                if energy > avg * 2.0 and energy > _ENERGY_THRESHOLD:
                    self._on_beat_detected(time.time(), energy)
        except Exception as e:
            logger.debug(f"[music_listener] process_block err: {e}")

    def _on_beat_detected(self, ts: float, energy: float):
        """Beat erkannt — BPM-Schaetzung + EventBus-Publish."""
        with self._lock:
            if self._last_beat_ts > 0:
                interval = ts - self._last_beat_ts
                if _MIN_BEAT_INTERVAL_S < interval < _MAX_BEAT_INTERVAL_S:
                    self._beat_intervals.append(interval)

            bpm = 0
            if len(self._beat_intervals) >= 4:
                sorted_intervals = sorted(self._beat_intervals)
                median_interval = sorted_intervals[len(sorted_intervals) // 2]
                if median_interval > 0:
                    bpm = int(round(60.0 / median_interval))

            self._last_beat_ts = ts
            self._last_beat = {"bpm": bpm, "energy": energy, "ts": ts}

        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                "music.beat",
                {"bpm": bpm, "energy": round(energy, 4), "ts": ts},
                source="music_listener", priority=3,
            )
        except Exception:
            pass

    def get_last_beat(self) -> Optional[dict]:
        """Letztes erkanntes Beat-Event (oder None wenn keins)."""
        with self._lock:
            return dict(self._last_beat) if self._last_beat else None

    def get_freq_bands(self) -> dict:
        """Letzte FFT-Bandsplit-Werte (bass/mid/treble/ts)."""
        with self._lock:
            return dict(self._last_freq_bands)


_instance: Optional[MusicListener] = None
_inst_lock = threading.Lock()


def get_music_listener() -> MusicListener:
    global _instance
    with _inst_lock:
        if _instance is None:
            _instance = MusicListener()
    return _instance
