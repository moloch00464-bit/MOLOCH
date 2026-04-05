#!/usr/bin/env python3
"""
Low Light Enhancement via zero_dce auf Hailo-10H NPU.

On-Demand Prozessor — laeuft automatisch wenn Frame zu dunkel ist.
Brightness-Check per CPU (kostenlos), NPU nur bei Dunkelheit aktiviert.

Modell: zero_dce.hef (856KB, 200 FPS, Hailo-10H, v5.2.0)
Input:  400x600x3 uint8 RGB
Output: 400x600x3 float32 [0,1] aufgehellt
"""

import logging
import threading
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger("LowLight")

HEF_PATH = "/mnt/moloch-data/hailo/models/zero_dce.hef"
VDEVICE_GROUP_ID = "SHARED"
MODEL_W = 600           # zero_dce Input: Breite
MODEL_H = 400           # zero_dce Input: Hoehe
DARK_THRESHOLD = 50     # Mittlere Helligkeit [0-255] unter der Enhancement aktiv wird
                        # 50 = echte Dunkelheit (~20%) — verhindert AE-Blip-Trigger
DARK_FRAMES_MIN = 5     # Hysterese: erst nach N aufeinanderfolgenden dunklen Frames aktiv
BRIGHT_FRAMES_MIN = 3   # Hysterese: erst nach N hellen Frames wieder deaktivieren
INFERENCE_TIMEOUT_MS = 3000


class LowLightProcessor:
    """zero_dce — automatisches Low-Light Enhancement.

    Singleton via get_low_light(). Lazy-Loaded beim ersten dunklen Frame.
    Nutzt SHARED VDevice — koexistiert mit GStreamer und anderen Workern.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vdevice = None
        self._configured = None
        self._out_names = []
        self._out_shapes = {}
        self._loaded = False
        self._load_error: Optional[str] = None
        self._last_brightness: int = -1
        self._dark_streak: int = 0    # aufeinanderfolgende dunkle Frames
        self._bright_streak: int = 0  # aufeinanderfolgende helle Frames
        self._enhancement_active: bool = False

    def _ensure_loaded(self) -> bool:
        """Modell lazy laden beim ersten dunklen Frame."""
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

            logger.info("[LowLight] zero_dce geladen — Input %dx%d → Output %s",
                        MODEL_W, MODEL_H, self._out_shapes)
            self._loaded = True
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error("[LowLight] Laden fehlgeschlagen: %s", e)
            return False

    def maybe_enhance(self, img_rgb: np.ndarray) -> np.ndarray:
        """Frame aufhellen wenn zu dunkel. Bei Helligkeit: sofortiger Return ohne NPU.

        Args:
            img_rgb: RGB uint8 numpy array, beliebige Groesse

        Returns:
            Aufgehelltes RGB uint8 — oder Original wenn hell genug / Fehler
        """
        # 1. Schneller Brightness-Check per CPU — kein NPU noetig
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            brightness = int(np.mean(gray))
        except Exception:
            brightness = 128
        self._last_brightness = brightness

        # Hysterese: Streaks zaehlen, Enhancement nur bei stabiler Dunkelheit
        if brightness < DARK_THRESHOLD:
            self._dark_streak += 1
            self._bright_streak = 0
            if self._dark_streak >= DARK_FRAMES_MIN:
                self._enhancement_active = True
        else:
            self._bright_streak += 1
            self._dark_streak = 0
            if self._bright_streak >= BRIGHT_FRAMES_MIN:
                self._enhancement_active = False

        if not self._enhancement_active:
            return img_rgb  # Hell genug oder noch nicht stabil dunkel genug

        # 2. Dunkel: Modell laden + Enhancement
        with self._lock:
            if not self._ensure_loaded():
                logger.debug("[LowLight] Modell nicht geladen — gebe Original zurück")
                return img_rgb

            try:
                h_orig, w_orig = img_rgb.shape[:2]

                # Resize auf Modell-Input: (H=400, W=600)
                inp = cv2.resize(img_rgb, (MODEL_W, MODEL_H),
                                 interpolation=cv2.INTER_LINEAR)
                inp_buf = np.ascontiguousarray(inp)  # (400,600,3) uint8

                # Inference
                bindings = self._configured.create_bindings()
                bindings.input().set_buffer(inp_buf)

                out_name = self._out_names[0]
                out_buf = np.empty(self._out_shapes[out_name], dtype=np.float32)
                bindings.output(out_name).set_buffer(out_buf)

                self._configured.run([bindings], INFERENCE_TIMEOUT_MS)

                # float32 [0,1] → uint8, zurueck auf Originalgroesse
                out = out_buf
                if out.ndim == 4:
                    out = out[0]
                enhanced = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)

                if enhanced.shape[:2] != (h_orig, w_orig):
                    enhanced = cv2.resize(enhanced, (w_orig, h_orig),
                                          interpolation=cv2.INTER_LINEAR)

                logger.debug("[LowLight] brightness=%d → enhanced, shape=%s",
                             brightness, enhanced.shape)
                return enhanced

            except Exception as e:
                logger.error("[LowLight] Inference fehlgeschlagen: %s", e)
                return img_rgb  # Fallback: Original

    def get_brightness(self) -> int:
        """Letzte gemessene Helligkeit [0-255]. -1 = noch kein Frame."""
        return self._last_brightness

    def is_active(self) -> bool:
        """True wenn Enhancement gerade aktiv (Hysterese beachtet)."""
        return self._enhancement_active

    def is_available(self) -> bool:
        """True wenn Modell geladen und bereit."""
        return self._loaded and self._load_error is None

    def stop(self):
        """VDevice freigeben."""
        with self._lock:
            self._vdevice = None
            self._configured = None
            self._loaded = False
        logger.info("[LowLight] gestoppt")


# Singleton
_instance: Optional[LowLightProcessor] = None
_instance_lock = threading.Lock()


def get_low_light() -> LowLightProcessor:
    """Singleton-Zugriff auf LowLightProcessor."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = LowLightProcessor()
    return _instance
