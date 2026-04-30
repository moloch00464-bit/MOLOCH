"""
TensionToTtsVolume — Tension steuert TTS-Volume-Multiplier

Subscribed: tension_changed
Mapping Tension -> Multiplier:
  0.0-0.3 calm:       0.7 (leise, nachdenklich)
  0.3-0.7:            1.0 (normal)
  0.7-0.9:            1.15 (eindringlich)
  0.9-1.0 berserker:  1.3 (laut)

Schreibt /dev/shm/moloch_tts_volume.json (atomic) — voice_pipeline liest beim TTS-Call.
Wenn voice_pipeline das nicht liest: nur Datenvorbereitung (best-effort).
"""
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("expression.tension_to_tts_volume")

VOLUME_FILE = "/dev/shm/moloch_tts_volume.json"

_TENSION_TO_MULTIPLIER = [
    (0.30, 0.7),
    (0.70, 1.0),
    (0.90, 1.15),
    (1.01, 1.3),
]


def _tension_to_multiplier(value: float) -> float:
    try:
        v = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0
    for threshold, mult in _TENSION_TO_MULTIPLIER:
        if v < threshold:
            return mult
    return 1.3


def _atomic_write(path: str, data: dict) -> bool:
    """Atomic JSON-Write — tempfile + os.replace."""
    try:
        dir_ = os.path.dirname(path) or "/dev/shm"
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug(f"_atomic_write {path}: {e}")
        return False


class TensionToTtsVolume:
    """TTS-Volume-Multiplier basierend auf Tension."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_value: float = 0.0
        self._last_multiplier: float = 1.0
        self._last_apply_ts: float = 0.0
        self._min_apply_interval: float = 1.0  # debounce
        self._subscribed = False

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            try:
                from core.moloch_event_bus import get_event_bus
                self._bus = get_event_bus()
                self._bus.subscribe("tension_changed", self._on_tension_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("TensionToTtsVolume gestartet")
                return True
            except Exception as e:
                logger.warning(f"TensionToTtsVolume start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            logger.info("TensionToTtsVolume gestoppt")

    def _on_tension_event(self, payload):
        try:
            data = payload if isinstance(payload, dict) else {}
            value = data.get("value", data.get("tension", 0.0))
            self.on_tension(float(value))
        except Exception as e:
            logger.debug(f"TensionToTtsVolume _on_tension_event Fehler: {e}")

    def on_tension(self, value: float):
        """Externe API: schreibt neuen Multiplier nach /dev/shm/."""
        new_mult = _tension_to_multiplier(value)
        with self._lock:
            self._last_value = value
            now = time.time()
            if abs(new_mult - self._last_multiplier) < 0.01 and (now - self._last_apply_ts) < self._min_apply_interval:
                return
            self._last_multiplier = new_mult
            self._last_apply_ts = now
        self._write_volume(new_mult)

    def _write_volume(self, multiplier: float):
        data = {
            "multiplier": float(multiplier),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        ok = _atomic_write(VOLUME_FILE, data)
        if ok:
            logger.debug(f"TensionToTtsVolume: multiplier={multiplier:.2f} -> {VOLUME_FILE}")
        else:
            logger.debug(f"TensionToTtsVolume: write failed")

    def get_state(self) -> dict:
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_value": self._last_value,
                "last_multiplier": self._last_multiplier,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": time.time() - self._last_apply_ts if self._last_apply_ts else None,
                "volume_file": VOLUME_FILE,
            }


_instance: Optional[TensionToTtsVolume] = None
_instance_lock = threading.Lock()


def get_tension_to_tts_volume() -> TensionToTtsVolume:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TensionToTtsVolume()
        return _instance
