#!/usr/bin/env python3
"""
Super Resolution via Real-ESRGAN x2 auf Hailo-10H NPU.

On-Demand Prozessor (kein Thread) — fuer Snapshots und Galerie-Bilder.
Input:  beliebige Bildgröße RGB uint8
Output: 2x hochskaliert RGB uint8 (512x512 → 1024x1024 intern)

Modell: real_esrgan_x2.hef (27MB, Hailo-10H, v5.2.0)
Download: hailo-model-zoo.s3.eu-west-2.amazonaws.com
"""

import logging
import threading
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger("SuperRes")

HEF_PATH = "/mnt/moloch-data/hailo/models/real_esrgan_x2.hef"
VDEVICE_GROUP_ID = "SHARED"
MODEL_INPUT_SIZE = 512   # Real-ESRGAN erwartet 512x512
MODEL_OUTPUT_SIZE = 1024  # Real-ESRGAN gibt 1024x1024 aus
INFERENCE_TIMEOUT_MS = 5000


class SuperResProcessor:
    """Real-ESRGAN x2 — synchroner On-Demand Upscaler.

    Singleton via get_super_res(). Modell wird beim ersten Aufruf geladen.
    Nutzt SHARED VDevice — koexistiert mit GStreamer-Pipeline und anderen Workern.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vdevice = None
        self._configured = None
        self._out_names = []
        self._out_shapes = {}
        self._loaded = False
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        """Modell lazy laden beim ersten Aufruf."""
        if self._loaded:
            return True
        if self._load_error:
            return False
        try:
            import hailo_platform as hp
            from hailo_platform.pyhailort._pyhailort import FormatType

            params = hp.VDevice.create_params()
            params.group_id = VDEVICE_GROUP_ID
            self._vdevice = hp.VDevice(params)

            model = self._vdevice.create_infer_model(HEF_PATH)
            for name in model.output_names:
                model.output(name).set_format_type(FormatType.FLOAT32)
            self._configured = model.configure()
            self._out_names = list(model.output_names)
            self._out_shapes = {n: list(model.output(n).shape) for n in self._out_names}

            logger.info("[SuperRes] Real-ESRGAN geladen — Input %dx%d → Output %s",
                        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, self._out_shapes)
            self._loaded = True
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error("[SuperRes] Laden fehlgeschlagen: %s", e)
            return False

    def upscale(self, img_rgb: np.ndarray) -> np.ndarray:
        """Bild 2x hochskalieren via Real-ESRGAN.

        Args:
            img_rgb: RGB uint8 numpy array, beliebige Größe

        Returns:
            Hochskaliertes RGB uint8 — bei Fehler: Original (kein Crash)
        """
        with self._lock:
            if not self._ensure_loaded():
                logger.warning("[SuperRes] Modell nicht geladen — gebe Original zurück")
                return img_rgb

            try:
                h_orig, w_orig = img_rgb.shape[:2]

                # Auf 512x512 skalieren (Modell-Input)
                inp = cv2.resize(img_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                 interpolation=cv2.INTER_LANCZOS4)

                # Modell erwartet uint8 RGB ohne Batch-Dimension: (512,512,3)
                inp_batch = np.ascontiguousarray(inp)  # (512,512,3) uint8

                # Inference
                bindings = self._configured.create_bindings()
                bindings.input().set_buffer(inp_batch)

                out_bufs = {}
                for name in self._out_names:
                    buf = np.empty(self._out_shapes[name], dtype=np.float32)
                    bindings.output(name).set_buffer(buf)
                    out_bufs[name] = buf

                self._configured.run([bindings], INFERENCE_TIMEOUT_MS)

                # Erstes Output-Tensor nehmen (1,1024,1024,3)
                out_name = self._out_names[0]
                out = out_bufs[out_name]

                # (1,H,W,3) → (H,W,3) + float→uint8
                if out.ndim == 4:
                    out = out[0]
                out_img = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

                logger.debug("[SuperRes] %dx%d → %dx%d OK",
                             w_orig, h_orig, out_img.shape[1], out_img.shape[0])
                return out_img

            except Exception as e:
                logger.error("[SuperRes] Inference fehlgeschlagen: %s", e)
                return img_rgb  # Fallback: Original

    def is_available(self) -> bool:
        """True wenn Modell geladen und bereit."""
        return self._loaded and self._load_error is None

    def stop(self):
        """VDevice freigeben."""
        with self._lock:
            self._vdevice = None
            self._configured = None
            self._loaded = False
        logger.info("[SuperRes] gestoppt")


# Singleton
_instance: Optional[SuperResProcessor] = None
_instance_lock = threading.Lock()


def get_super_res() -> SuperResProcessor:
    """Singleton-Zugriff auf SuperResProcessor."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = SuperResProcessor()
    return _instance
