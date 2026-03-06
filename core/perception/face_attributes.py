#!/usr/bin/env python3
"""
FaceAttributes — Separater HailoRT-Thread fuer Face-Attribut-Erkennung.

Liest Face-Crops aus dem aktuellen PFrame (face_bbox) + annotated Frame,
inferiert face_attr_resnet_v1_18.hef auf dem NPU (shared VDevice),
schreibt gender + smiling zurueck ins PFrame.

Kein Eingriff in tappas_pipeline.py — laeuft vollstaendig separat.

CelebA 40-Attribut Layout (je 2 Ausgaenge: negativ/positiv):
  Index 20 = Male (idx 40/41 im 80er Output)
  Index 31 = Smiling (idx 62/63 im 80er Output)
"""

import time
import threading
import logging
from typing import Optional

import cv2
import numpy as np

from hailo_platform import (
    VDevice, HEF, ConfigureParams, InputVStreamParams,
    OutputVStreamParams, InferVStreams, FormatType,
    HailoSchedulingAlgorithm
)

logger = logging.getLogger("FaceAttributes")

FACE_ATTR_HEF = "/mnt/moloch-data/hailo/models/face_attr_resnet_v1_18.hef"
VDEVICE_GROUP_ID = "SHARED"

# HEF Input: 218x178x3 (H x W x C)
INPUT_H = 218
INPUT_W = 178

# CelebA Attribut-Indizes im 80er Output (je 2: negativ, positiv)
ATTR_MALE_POS = 41       # Index 20 * 2 + 1
ATTR_MALE_NEG = 40       # Index 20 * 2
ATTR_SMILING_POS = 63    # Index 31 * 2 + 1
ATTR_SMILING_NEG = 62    # Index 31 * 2


class FaceAttributes:
    """Separater Thread: Face-Crop → NPU → gender/smiling ins PFrame."""

    def __init__(self, pipeline, interval: float = 0.3):
        """
        Args:
            pipeline: TappasPipeline-Instanz (fuer get_annotated_frame/get_current_pframe)
            interval: Sekunden zwischen Inferenzen (Default 0.3 = ~3 Hz)
        """
        self._pipeline = pipeline
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # HailoRT
        self._vdevice = None
        self._network_group = None
        self._input_vstreams_params = None
        self._output_vstreams_params = None
        self._input_name = None
        self._output_name = None

    def start(self):
        """NPU-Modell laden und Inferenz-Thread starten."""
        if self._running:
            return

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"[FACE-ATTR] Modell-Laden fehlgeschlagen: {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True,
                                        name="FaceAttrThread")
        self._thread.start()
        logger.info(f"[FACE-ATTR] Thread gestartet (interval={self._interval}s)")

    def stop(self):
        """Thread stoppen und NPU-Ressourcen freigeben."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._vdevice:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None
        logger.info("[FACE-ATTR] Gestoppt")

    def _load_model(self):
        """HEF laden, shared VDevice oeffnen, Network Group konfigurieren."""
        hef = HEF(FACE_ATTR_HEF)

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = VDEVICE_GROUP_ID
        self._vdevice = VDevice(params=params)

        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoSchedulingAlgorithm.ROUND_ROBIN)
        self._network_group = self._vdevice.configure(hef, configure_params)[0]

        self._input_vstreams_params = InputVStreamParams.make(self._network_group,
                                                              format_type=FormatType.UINT8)
        self._output_vstreams_params = OutputVStreamParams.make(self._network_group,
                                                                format_type=FormatType.FLOAT32)

        input_infos = hef.get_input_vstream_infos()
        output_infos = hef.get_output_vstream_infos()
        self._input_name = input_infos[0].name
        self._output_name = output_infos[0].name

        logger.info(f"[FACE-ATTR] Modell geladen: {FACE_ATTR_HEF}")
        logger.info(f"[FACE-ATTR] Input: {self._input_name} ({INPUT_H}x{INPUT_W}x3)")
        logger.info(f"[FACE-ATTR] Output: {self._output_name} (80 Attribute)")

    def _crop_face(self, frame: np.ndarray, face_bbox: tuple) -> Optional[np.ndarray]:
        """Face-Region aus Frame croppen und auf 178x218 resizen.

        Args:
            frame: RGB-Frame (H, W, 3)
            face_bbox: (x1, y1, x2, y2) normalisiert [0.0-1.0]

        Returns:
            np.ndarray (218, 178, 3) uint8 oder None
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        px1 = max(0, int(x1 * w))
        py1 = max(0, int(y1 * h))
        px2 = min(w, int(x2 * w))
        py2 = min(h, int(y2 * h))

        if px2 - px1 < 10 or py2 - py1 < 10:
            return None

        crop = frame[py1:py2, px1:px2]
        resized = cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        return resized

    def _parse_output(self, output: np.ndarray) -> dict:
        """80er Output-Vektor in gender + smiling parsen.

        Jedes Attribut hat 2 Werte (negativ/positiv). Softmax ergibt Wahrscheinlichkeit.
        """
        # Softmax pro Attribut-Paar
        def softmax_pair(neg_idx, pos_idx):
            vals = np.array([output[neg_idx], output[pos_idx]], dtype=np.float32)
            e = np.exp(vals - np.max(vals))
            probs = e / e.sum()
            return probs[1]  # Wahrscheinlichkeit fuer positiv

        male_prob = softmax_pair(ATTR_MALE_NEG, ATTR_MALE_POS)
        smiling_prob = softmax_pair(ATTR_SMILING_NEG, ATTR_SMILING_POS)

        gender = "M" if male_prob > 0.5 else "F"
        smiling = smiling_prob > 0.5

        return {
            "gender": gender,
            "male_prob": round(float(male_prob), 3),
            "smiling": smiling,
            "smiling_prob": round(float(smiling_prob), 3),
        }

    def _inference_loop(self):
        """Hauptschleife: Face-Crop → NPU → PFrame Update."""
        logger.info("[FACE-ATTR] Inferenz-Loop gestartet")

        while self._running:
            try:
                time.sleep(self._interval)
                if not self._running:
                    break

                # Aktuelles PFrame + Frame holen
                pf = self._pipeline.get_current_pframe()
                if not pf.face_detected or pf.face_bbox is None:
                    continue

                frame = self._pipeline.get_annotated_frame()
                if frame is None:
                    continue

                # Face croppen
                crop = self._crop_face(frame, pf.face_bbox)
                if crop is None:
                    continue

                # NPU Inferenz (Batch=1)
                input_data = {self._input_name: np.expand_dims(crop, axis=0)}

                with InferVStreams(self._network_group,
                                   self._input_vstreams_params,
                                   self._output_vstreams_params) as infer_pipeline:
                    results = infer_pipeline.infer(input_data)

                raw_output = results[self._output_name][0]  # (80,)
                attrs = self._parse_output(raw_output)

                # PFrame aktualisieren (thread-safe via Pipeline Lock)
                with self._pipeline._lock:
                    current_pf = self._pipeline._current_pframe
                    current_pf.gender = attrs["gender"]
                    # smiling in emotion-Feld schreiben (PFrame hat kein eigenes smiling-Feld)
                    current_pf.emotion = "happy" if attrs["smiling"] else "neutral"

            except Exception as e:
                logger.error(f"[FACE-ATTR] Inferenz-Fehler: {e}")
                time.sleep(1.0)

        logger.info("[FACE-ATTR] Inferenz-Loop beendet")
