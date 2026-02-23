#!/usr/bin/env python3
"""
M.O.L.O.C.H. Model Health Monitoring
=====================================
Pro Modell: FPS, Latency, Error Rate, Drop Rate tracken.
Erkennt haengende/tote Modelle und informiert den Service.

Health-Status fliesst in den Status-JSON fuer Panel-Anzeige.
"""

import time
import threading
import logging
from typing import Dict, Optional, List
from collections import deque

_logger = logging.getLogger("ModelHealth")

# Modell gilt als "stuck" wenn FPS < 1 fuer diese Zeit
_STUCK_TIMEOUT = 10.0
# Maximale Latenz bevor Warnung
_HIGH_LATENCY_MS = 200.0
# Error-Rate Schwelle (Errors/Minute)
_ERROR_RATE_THRESHOLD = 10


class ModelHealthEntry:
    """Health-Tracking fuer ein einzelnes Modell."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()

        # FPS Tracking (letzte 30 Samples)
        self._frame_times: deque = deque(maxlen=60)
        self._latencies: deque = deque(maxlen=60)

        # Error Tracking
        self._errors: deque = deque(maxlen=100)  # Timestamps
        self._drops: deque = deque(maxlen=100)

        # Zustand
        self.active = False
        self.paused = False  # Bewusst pausiert (nicht Fehler)
        self._last_inference = 0.0
        self._total_inferences = 0
        self._total_errors = 0

    def record_inference(self, latency_ms: float):
        """Erfolgreiche Inferenz aufzeichnen."""
        now = time.time()
        with self._lock:
            self._frame_times.append(now)
            self._latencies.append(latency_ms)
            self._last_inference = now
            self._total_inferences += 1
            self.active = True

    def record_error(self):
        """Fehler bei Inferenz."""
        with self._lock:
            self._errors.append(time.time())
            self._total_errors += 1

    def record_drop(self):
        """Frame-Drop (Modell zu langsam)."""
        with self._lock:
            self._drops.append(time.time())

    def set_paused(self, paused: bool):
        """Modell bewusst pausiert/resumed."""
        self.paused = paused
        if not paused:
            self.active = True

    @property
    def fps(self) -> float:
        """Aktuelle FPS (gleitender Durchschnitt ueber 1s)."""
        now = time.time()
        with self._lock:
            recent = [t for t in self._frame_times if now - t < 1.0]
            return float(len(recent))

    @property
    def avg_latency_ms(self) -> float:
        """Durchschnittliche Latenz der letzten 60 Inferenzen."""
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def max_latency_ms(self) -> float:
        """Maximale Latenz der letzten 60 Inferenzen."""
        with self._lock:
            return max(self._latencies) if self._latencies else 0.0

    @property
    def error_rate(self) -> float:
        """Fehler pro Minute (letzte 100 Fehler)."""
        now = time.time()
        with self._lock:
            recent = [t for t in self._errors if now - t < 60.0]
            return float(len(recent))

    @property
    def is_stuck(self) -> bool:
        """Modell haengt (aktiv aber kein Output seit _STUCK_TIMEOUT)."""
        if self.paused or not self.active:
            return False
        if self._last_inference == 0:
            return False
        return (time.time() - self._last_inference) > _STUCK_TIMEOUT

    @property
    def is_healthy(self) -> bool:
        """Modell ist gesund (nicht stuck, Error-Rate niedrig)."""
        if self.paused:
            return True  # Pausiert = ok
        if not self.active:
            return True  # Inaktiv = ok
        if self.is_stuck:
            return False
        if self.error_rate > _ERROR_RATE_THRESHOLD:
            return False
        return True

    def get_status(self) -> Dict:
        """Status-Dict fuer JSON-Export."""
        return {
            "name": self.name,
            "active": self.active,
            "paused": self.paused,
            "fps": round(self.fps, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "max_latency_ms": round(self.max_latency_ms, 1),
            "error_rate": round(self.error_rate, 1),
            "is_stuck": self.is_stuck,
            "is_healthy": self.is_healthy,
            "total_inferences": self._total_inferences,
            "total_errors": self._total_errors,
        }


class ModelHealthMonitor:
    """Zentrale Health-Ueberwachung fuer alle NPU-Modelle."""

    ALL_MODELS = ["scrfd", "arcface", "yolov8m", "hand_landmark", "pose", "face_attr"]

    def __init__(self):
        self._models: Dict[str, ModelHealthEntry] = {}
        for name in self.ALL_MODELS:
            self._models[name] = ModelHealthEntry(name)
        _logger.info(f"[HEALTH] Monitor initialisiert fuer {len(self._models)} Modelle")

    def record_inference(self, model: str, latency_ms: float):
        """Erfolgreiche Inferenz aufzeichnen."""
        entry = self._models.get(model)
        if entry:
            entry.record_inference(latency_ms)

    def record_error(self, model: str):
        """Fehler bei Inferenz."""
        entry = self._models.get(model)
        if entry:
            entry.record_error()

    def record_drop(self, model: str):
        """Frame-Drop."""
        entry = self._models.get(model)
        if entry:
            entry.record_drop()

    def set_paused(self, model: str, paused: bool):
        """Modell bewusst pausiert/resumed."""
        entry = self._models.get(model)
        if entry:
            entry.set_paused(paused)

    def get_stuck_models(self) -> List[str]:
        """Liste der haengenden Modelle."""
        return [name for name, entry in self._models.items() if entry.is_stuck]

    def get_unhealthy_models(self) -> List[str]:
        """Liste der ungesunden Modelle."""
        return [name for name, entry in self._models.items() if not entry.is_healthy]

    def get_fps(self, model: str) -> float:
        """FPS eines Modells."""
        entry = self._models.get(model)
        return entry.fps if entry else 0.0

    def get_status(self) -> Dict:
        """Gesamt-Status fuer JSON-Export."""
        models = {}
        for name, entry in self._models.items():
            if entry.active or entry.paused:
                models[name] = entry.get_status()
        stuck = self.get_stuck_models()
        unhealthy = self.get_unhealthy_models()
        return {
            "models": models,
            "stuck": stuck,
            "unhealthy": unhealthy,
            "all_healthy": len(unhealthy) == 0,
        }


# Singleton
_instance: Optional[ModelHealthMonitor] = None

def get_model_health() -> ModelHealthMonitor:
    """Singleton-Zugriff auf den ModelHealthMonitor."""
    global _instance
    if _instance is None:
        _instance = ModelHealthMonitor()
    return _instance
