#!/usr/bin/env python3
"""
PersonAttrWorker — Personen-Attribut-Erkennung via person_attr_resnet_v1_18.

Erkennt 35 binaere Attribute pro Person-Crop:
Alter, Geschlecht, Kleidungsfarbe, Zubehoer, Koerperausrichtung.

Input:  224x224x3 uint8 (Person-Crop)
Output: [35] float32 Sigmoid-Wahrscheinlichkeiten (Hailo gibt float32)
Modell: /mnt/moloch-data/hailo/models/person_attr_resnet_v1_18.hef
FPS:    ~2000+ (trivial schnell, On-Demand per Person)
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional

from core.perception.vision_workers import (
    BaseWorker, WorkItem, WorkerResult, create_configured_model,
    INFERENCE_TIMEOUT_MS
)

logger = logging.getLogger("PersonAttrWorker")

MODEL_DIR = "/mnt/moloch-data/hailo/models"
PERSON_ATTR_HEF = os.path.join(MODEL_DIR, "person_attr_resnet_v1_18.hef")

# PA-100K Attribut-Labels (35 Klassen, Index-treu)
ATTR_LABELS = [
    "weiblich",        # 0
    "alter_ueber60",   # 1
    "alter_18_60",     # 2
    "alter_unter18",   # 3
    "blickrichtung_vorne",  # 4
    "blickrichtung_seite",  # 5
    "blickrichtung_hinten", # 6
    "hut",             # 7
    "brille",          # 8
    "handtasche",      # 9
    "umhaengetasche",  # 10
    "rucksack",        # 11
    "gegenstand_vorne",# 12
    "kurzarm",         # 13
    "langarm",         # 14
    "oberteil_streifen",    # 15
    "oberteil_logo",        # 16
    "oberteil_kariert",     # 17
    "oberteil_gemustert",   # 18
    "hose_streifen",        # 19
    "hose_muster",          # 20
    "langer_mantel",        # 21
    "lange_hose",           # 22
    "kurze_hose",           # 23
    "rock_kleid",           # 24
    "stiefel",              # 25
    "oberteil_schwarz",     # 26
    "oberteil_weiss",       # 27
    "oberteil_grau",        # 28
    "oberteil_rot",         # 29
    "oberteil_gruen",       # 30
    "oberteil_blau",        # 31
    "hose_schwarz",         # 32
    "hose_weiss",           # 33
    "hose_grau",            # 34
]

# Attribute die fuer den Moloch-Kontext besonders relevant sind
RELEVANT_ATTRS = {
    "rucksack", "hut", "brille", "weiblich",
    "alter_ueber60", "alter_unter18",
    "langer_mantel", "stiefel"
}

SIGMOID_THRESHOLD = 0.5  # Ueber diesem Wert = Attribut aktiv


class PersonAttrWorker(BaseWorker):
    """Erkennt Personen-Attribute aus Person-Crops.

    Wird von roi_dispatcher mit Person-BBoxes gefuettert.
    Ergebnis: Liste aktiver Attribute pro erkannter Person.
    """

    def __init__(self):
        super().__init__(name="PersonAttrWorker", max_queue=2)
        self._model = None
        self._out_names = []
        self._out_shapes = {}

    def _load_models(self, vdevice):
        if not os.path.exists(PERSON_ATTR_HEF):
            raise FileNotFoundError(f"PersonAttr HEF fehlt: {PERSON_ATTR_HEF}")
        _, self._model, _, self._out_names, self._out_shapes = \
            create_configured_model(vdevice, PERSON_ATTR_HEF)
        logger.info("[PersonAttrWorker] Modell geladen — Outputs: %s", self._out_names)

    def _process(self, item: WorkItem) -> WorkerResult:
        frame_rgb = item.frame
        fh, fw = frame_rgb.shape[:2]
        results = []

        dets = item.detections if item.detections else []

        for det in dets:
            bbox = det.get("bbox", [0, 0, 1, 1])
            x1 = max(0, int(bbox[0] * fw))
            y1 = max(0, int(bbox[1] * fh))
            x2 = min(fw, int(bbox[2] * fw))
            y2 = min(fh, int(bbox[3] * fh))

            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
                continue

            # Auf 224x224 skalieren
            inp = cv2.resize(crop, (224, 224))
            inp = np.ascontiguousarray(inp, dtype=np.uint8)

            bindings = self._model.create_bindings()
            bindings.input().set_buffer(inp)

            bufs = {}
            for name in self._out_names:
                buf = np.empty(self._out_shapes[name], dtype=np.float32)
                bindings.output(name).set_buffer(buf)
                bufs[name] = buf

            self._model.run([bindings], INFERENCE_TIMEOUT_MS)

            scores = bufs[self._out_names[0]].flatten().copy()

            # Aktive Attribute (ueber Threshold)
            aktive = [
                ATTR_LABELS[i]
                for i in range(min(len(scores), len(ATTR_LABELS)))
                if scores[i] >= SIGMOID_THRESHOLD
            ]

            results.append({
                "bbox": bbox,
                "attribute": aktive,
                "scores_raw": scores.tolist(),
            })

        return WorkerResult(
            worker_name="PersonAttrWorker",
            frame_id=item.frame_id,
            timestamp=item.timestamp,
            success=True,
            data={
                "persons": results,
                "count": len(results),
            }
        )


_instance: Optional[PersonAttrWorker] = None


def get_person_attr_worker() -> PersonAttrWorker:
    """Singleton-Getter."""
    global _instance
    if _instance is None:
        _instance = PersonAttrWorker()
    return _instance


def decode_person_attrs(result: WorkerResult) -> List[Dict]:
    """Hilfsfunktion: Ergebnis in lesbare Attribut-Liste umwandeln."""
    if not result or not result.success:
        return []
    return result.data.get("persons", [])
