#!/usr/bin/env python3
"""
Vision Workers — HailoRT-Direct Model Worker Framework.

Jeder Worker laeuft in eigenem Thread, nutzt das SHARED VDevice,
und ist crash-isoliert (try/except um jeden Inference-Call).

Pattern uebernommen von npu_extras.py (bewaehrt fuer CLIP/OCR/VLM).

Architektur:
  BaseWorker   — Thread + Queue + VDevice + Health
  ResultCollector — Thread-safe Ergebnis-Sammlung
"""

import time
import queue
import logging
import threading
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger("VisionWorkers")

VDEVICE_GROUP_ID = "SHARED"
INFERENCE_TIMEOUT_MS = 10000


# ============================================================
# WorkItem — Auftrag an einen Worker
# ============================================================

@dataclass
class WorkItem:
    """Ein Frame + Metadaten fuer einen Worker."""
    frame: np.ndarray              # RGB numpy array
    frame_id: int = 0              # Sequenz-Nummer fuer Sync
    timestamp: float = 0.0         # time.monotonic()
    detections: List[Dict] = field(default_factory=list)  # YOLO-Detections (fuer Crop-Worker)


# ============================================================
# WorkerResult — Ergebnis eines Workers
# ============================================================

@dataclass
class WorkerResult:
    """Ergebnis eines Worker-Durchlaufs."""
    worker_name: str
    frame_id: int = 0
    timestamp: float = 0.0
    inference_ms: float = 0.0
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# BaseWorker — Basis-Klasse fuer alle Vision Worker
# ============================================================

class BaseWorker(threading.Thread):
    """Basis-Worker: eigener Thread, eigene Queue, eigenes InferModel.

    Subklassen implementieren:
      - _load_models(vdevice) — Modelle laden
      - _process(item: WorkItem) -> WorkerResult — Inference
    """

    def __init__(self, name: str, max_queue: int = 2):
        super().__init__(name=name, daemon=True)
        self._worker_name = name
        self._queue: queue.Queue[WorkItem] = queue.Queue(maxsize=max_queue)
        self._running = False
        self._vdevice = None
        self._models_loaded = False

        # Health-Statistiken
        self._total_inferences = 0
        self._total_errors = 0
        self._last_inference_ms = 0.0
        self._last_error = ""

        # Letztes Ergebnis (thread-safe via Lock)
        self._result_lock = threading.Lock()
        self._latest_result: Optional[WorkerResult] = None

    def submit(self, item: WorkItem) -> bool:
        """Frame zur Verarbeitung einreichen. Drop bei voller Queue."""
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False  # Frame droppen statt stauen

    def get_latest_result(self) -> Optional[WorkerResult]:
        """Letztes Ergebnis holen (thread-safe)."""
        with self._result_lock:
            return self._latest_result

    def start_worker(self):
        """Worker starten."""
        self._running = True
        self.start()
        logger.info("[%s] Worker gestartet", self._worker_name)

    def stop_worker(self):
        """Worker sauber stoppen."""
        self._running = False
        # Poison-Pill in Queue damit Thread aufwacht
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        """Worker-Loop: Queue lesen, Modelle laden, Inference ausfuehren."""
        # Lazy Model-Loading im Worker-Thread (nicht im Main-Thread!)
        try:
            self._ensure_vdevice()
            self._load_models(self._vdevice)
            self._models_loaded = True
            logger.info("[%s] Modelle geladen", self._worker_name)
        except Exception as e:
            logger.error("[%s] Model-Loading fehlgeschlagen: %s", self._worker_name, e)
            self._last_error = str(e)
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # Poison-Pill
                break

            try:
                t0 = time.monotonic()
                result = self._process(item)
                dt = (time.monotonic() - t0) * 1000
                result.inference_ms = dt
                self._last_inference_ms = dt
                self._total_inferences += 1

                with self._result_lock:
                    self._latest_result = result

            except Exception as e:
                self._total_errors += 1
                self._last_error = str(e)
                logger.error("[%s] Inference-Fehler #%d: %s",
                             self._worker_name, self._total_errors, e)
                # Worker laeuft weiter — naechster Frame

        logger.info("[%s] Worker gestoppt (inferences=%d, errors=%d)",
                    self._worker_name, self._total_inferences, self._total_errors)

    def _ensure_vdevice(self):
        """Shared VDevice erstellen/joinen."""
        if self._vdevice is not None:
            return
        import hailo_platform as hp
        params = hp.VDevice.create_params()
        params.group_id = VDEVICE_GROUP_ID
        self._vdevice = hp.VDevice(params)
        logger.info("[%s] VDevice joined (group=%s)", self._worker_name, VDEVICE_GROUP_ID)

    def _load_models(self, vdevice):
        """Modelle laden — von Subklasse implementiert."""
        raise NotImplementedError

    def _process(self, item: WorkItem) -> WorkerResult:
        """Frame verarbeiten — von Subklasse implementiert."""
        raise NotImplementedError

    def get_health(self) -> Dict:
        """Health-Status fuer Monitoring."""
        return {
            "name": self._worker_name,
            "running": self._running and self.is_alive(),
            "models_loaded": self._models_loaded,
            "total_inferences": self._total_inferences,
            "total_errors": self._total_errors,
            "last_inference_ms": round(self._last_inference_ms, 1),
            "last_error": self._last_error,
            "queue_size": self._queue.qsize(),
        }


# ============================================================
# create_configured_model — Shared Helper (aus npu_extras.py)
# ============================================================

def create_configured_model(vdevice, hef_path: str, float_output: bool = True):
    """InferModel erstellen, Output auf FLOAT32, konfigurieren.

    Identisch zu npu_extras._create_configured_model().

    Returns:
        (model, configured, input_names, output_names, output_shapes)
    """
    model = vdevice.create_infer_model(hef_path)
    if float_output:
        from hailo_platform.pyhailort._pyhailort import FormatType
        for name in model.output_names:
            model.output(name).set_format_type(FormatType.FLOAT32)
    configured = model.configure()
    input_names = list(model.input_names)
    output_names = list(model.output_names)
    output_shapes = {name: list(model.output(name).shape) for name in output_names}
    return model, configured, input_names, output_names, output_shapes


# ============================================================
# ResultCollector — Thread-safe Ergebnis-Sammlung
# ============================================================

class ResultCollector:
    """Sammelt Ergebnisse von mehreren Workern, thread-safe.

    Downstream (perception_loop) liest gesammelte Ergebnisse
    und baut daraus einen PerceptionFrame.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._workers: Dict[str, BaseWorker] = {}
        self._latest: Dict[str, WorkerResult] = {}

    def register_worker(self, worker: BaseWorker):
        """Worker registrieren."""
        with self._lock:
            self._workers[worker._worker_name] = worker
            logger.info("[ResultCollector] Worker registriert: %s", worker._worker_name)

    def collect(self) -> Dict[str, WorkerResult]:
        """Neueste Ergebnisse aller Worker sammeln.

        Returns:
            Dict[worker_name -> WorkerResult]
        """
        results = {}
        with self._lock:
            for name, worker in self._workers.items():
                result = worker.get_latest_result()
                if result is not None:
                    results[name] = result
                    self._latest[name] = result
        return results

    def get_latest(self, worker_name: str) -> Optional[WorkerResult]:
        """Letztes Ergebnis eines bestimmten Workers (direkt vom Worker)."""
        with self._lock:
            worker = self._workers.get(worker_name)
            if worker:
                result = worker.get_latest_result()
                if result is not None:
                    self._latest[worker_name] = result
                return self._latest.get(worker_name)
            return None

    def start_all(self):
        """Alle registrierten Worker starten."""
        with self._lock:
            for name, worker in self._workers.items():
                if not worker.is_alive():
                    worker.start_worker()

    def stop_all(self):
        """Alle Worker stoppen."""
        with self._lock:
            for name, worker in self._workers.items():
                worker.stop_worker()
            # Auf alle Threads warten (max 5s pro Worker)
            for name, worker in self._workers.items():
                worker.join(timeout=5.0)
                if worker.is_alive():
                    logger.warning("[ResultCollector] Worker %s haengt!", name)

    def get_health(self) -> Dict[str, Dict]:
        """Health-Status aller Worker."""
        with self._lock:
            return {name: worker.get_health() for name, worker in self._workers.items()}
