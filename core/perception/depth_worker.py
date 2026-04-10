#!/usr/bin/env python3
"""
DepthWorker — Monokulare Tiefenschaetzung via scdepthv3 auf Hailo-10H.

HEF: /mnt/moloch-data/hailo/models/zoo/depth/scdepthv3.hef
Input:  256x320x3 uint8 RGB
Output: 256x320 float32 Tiefenkarte (skalierte inverse Tiefe)

Laeuft als Hintergrund-Worker (BaseWorker-Pattern), teilt shared VDevice
mit TAPPAS und allen anderen Workern (group_id=SHARED).

Ergebnis: {"depth_m": float, "depth_center": float, "map_shape": list}
  - depth_m: Median-Tiefe im Zentrum (ca. 15% des Bildes)
  - depth_center: Rohwert (nicht metrisch kalibriert, relativ)
"""

import os
import logging
import numpy as np
import cv2

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model,
    INFERENCE_TIMEOUT_MS
)

logger = logging.getLogger("DepthWorker")

# --- Modell-Pfad ---
DEPTH_HEF = "/mnt/moloch-data/hailo/models/zoo/depth/scdepthv3.hef"

# Eingabegroesse laut HEF-Spezifikation
DEPTH_INPUT_W = 320
DEPTH_INPUT_H = 256

# Zentrum-Region fuer Median-Berechnung (Mitte 20% des Bildes)
_CX1 = int(DEPTH_INPUT_H * 0.40)
_CX2 = int(DEPTH_INPUT_H * 0.60)
_CY1 = int(DEPTH_INPUT_W * 0.40)
_CY2 = int(DEPTH_INPUT_W * 0.60)


class DepthWorker(BaseWorker):
    """Monokulare Tiefenschaetzung via scdepthv3.

    Gibt inverse Tiefe zurueck (hoehere Werte = naeher).
    Kein absoluter Meterwert moeglich (monokulare Kamera ohne Kalibrierung).
    """

    def __init__(self):
        super().__init__(name="DepthWorker", max_queue=1)
        self._depth_configured = None
        self._depth_out_name: str = ""
        self._depth_out_shape = None

    def _load_models(self, vdevice):
        if not os.path.exists(DEPTH_HEF):
            raise FileNotFoundError(f"Depth HEF fehlt: {DEPTH_HEF}")
        _, self._depth_configured, _, out_names, out_shapes = \
            create_configured_model(vdevice, DEPTH_HEF)
        self._depth_out_name = out_names[0]
        self._depth_out_shape = out_shapes[self._depth_out_name]
        logger.info("[DepthWorker] scdepthv3 geladen — Output: %s %s",
                    self._depth_out_name, self._depth_out_shape)

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame

        # Resize auf 320x256 (kein Letterbox noetig — Groesse direkt)
        img = cv2.resize(frame_rgb, (DEPTH_INPUT_W, DEPTH_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)

        # Inference
        bindings = self._depth_configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(img))
        out_buf = np.empty(self._depth_out_shape, dtype=np.float32)
        bindings.output(self._depth_out_name).set_buffer(out_buf)
        self._depth_configured.run([bindings], INFERENCE_TIMEOUT_MS)

        # Tiefenkarte: squeeze auf (256, 320) oder (1, 256, 320)
        depth_map = out_buf.squeeze()

        # Zentrum-Median als Naeherungswert
        center_region = depth_map[_CX1:_CX2, _CY1:_CY2]
        depth_center = float(np.median(center_region))

        # Gesamtbild-Statistik
        depth_mean = float(np.mean(depth_map))
        depth_min = float(np.min(depth_map))
        depth_max = float(np.max(depth_map))

        # Relativer Naeherungswert (0.0 = sehr fern, 1.0 = sehr nah)
        # scdepthv3 liefert inverse Tiefe: hoeher = naeher
        depth_rel = round(float(np.clip(depth_center / (depth_max + 1e-6), 0.0, 1.0)), 3)

        return WorkerResult(
            worker_name="DepthWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            inference_ms=0.0,  # wird von BaseWorker.run() gesetzt
            success=True,
            data={
                "depth_m": round(depth_center, 3),
                "depth_rel": depth_rel,
                "depth_mean": round(depth_mean, 3),
                "depth_min": round(depth_min, 3),
                "depth_max": round(depth_max, 3),
                "map_shape": list(depth_map.shape),
            },
        )
