#!/usr/bin/env python3
"""
ROI Dispatcher — Frame + Detection Routing an Vision Worker.

Sitzt zwischen GStreamer appsink (YOLO-Detections + Raw Frame)
und den HailoRT-Direct Workern.

Entscheidet:
  - Welche Worker einen Frame bekommen (Rate-Limiting)
  - Welche Crops (Person-BBoxes) an Crop-Worker gehen
  - Wann ein Frame uebersprungen wird (Worker voll)

Architektur:
  YOLO Pipeline → appsink → ROI Dispatcher → Worker Queues
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional

from core.perception.vision_workers import BaseWorker, WorkItem

logger = logging.getLogger("ROIDispatcher")


class ROIDispatcher:
    """Verteilt Frames und YOLO-Detections an registrierte Worker.

    Jeder Worker kann eine eigene Rate haben (nicht jeder Frame
    muss an jeden Worker gehen — z.B. Face alle 2 Frames,
    Pose alle 3 Frames).
    """

    def __init__(self):
        self._workers: Dict[str, BaseWorker] = {}
        self._rates: Dict[str, int] = {}  # Worker-Name -> jeder N-te Frame
        self._counters: Dict[str, int] = {}
        self._frame_count = 0

        # Statistiken
        self._dispatched = 0
        self._dropped = 0

    def register_worker(self, worker: BaseWorker, every_n_frames: int = 1):
        """Worker registrieren mit Rate-Limit.

        Args:
            worker: Der Vision-Worker
            every_n_frames: Worker bekommt jeden N-ten Frame (1=alle, 2=jeden zweiten, ...)
        """
        self._workers[worker._worker_name] = worker
        self._rates[worker._worker_name] = max(1, every_n_frames)
        self._counters[worker._worker_name] = 0
        logger.info("[ROIDispatcher] Worker registriert: %s (every %d frames)",
                    worker._worker_name, every_n_frames)

    def dispatch(self, frame: np.ndarray, detections: List[Dict],
                 frame_id: int = 0):
        """Frame + Detections an alle registrierten Worker verteilen.

        Args:
            frame: RGB numpy array (z.B. 1280x720 oder 640x360)
            detections: YOLO Person-Detections [{bbox, confidence, class, ...}]
            frame_id: Sequenz-Nummer fuer Frame-Sync
        """
        self._frame_count += 1
        ts = time.monotonic()

        for name, worker in self._workers.items():
            rate = self._rates[name]
            self._counters[name] += 1

            if self._counters[name] < rate:
                continue
            self._counters[name] = 0

            item = WorkItem(
                frame=frame,
                frame_id=frame_id,
                timestamp=ts,
                detections=detections,
            )

            if worker.submit(item):
                self._dispatched += 1
            else:
                self._dropped += 1

    def get_stats(self) -> Dict:
        """Dispatch-Statistiken."""
        return {
            "total_frames": self._frame_count,
            "dispatched": self._dispatched,
            "dropped": self._dropped,
            "workers": list(self._workers.keys()),
        }
