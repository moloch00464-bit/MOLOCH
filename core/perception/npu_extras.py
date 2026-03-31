#!/usr/bin/env python3
"""
NPU-Extras — CLIP, PaddleOCR, Qwen2-VL auf Hailo-10H.

Eigenstaendiges Modul, NICHT in der GStreamer-Pipeline.
Nutzt dasselbe shared VDevice (group_id=SHARED) wie TAPPAS + hailo-ollama.

Alle Modelle werden LAZY geladen (erst beim ersten Aufruf).
Jede Inference grabbt den aktuellen Frame aus SHM.

HailoRT API:
  vdevice.create_infer_model(hef_path) → InferModel
  model.output(...).set_format_type(FormatType.FLOAT32) → dequantisierter Output
  model.configure() → ConfiguredInferModel
  configured.create_bindings() → Bindings (Input/Output Buffer)
  configured.run([bindings], timeout_ms)

Nutzung:
    extras = get_npu_extras()
    embedding = extras.clip_embed(frame)         # 640d Vektor
    texte = extras.ocr_read(frame)               # [{text, confidence, bbox}]
    beschreibung = extras.vlm_describe(frame)     # Freitext
"""

import os
import time
import struct
import logging
import threading
import numpy as np
from typing import Optional, List, Dict

logger = logging.getLogger("NpuExtras")

# --- Modell-Pfade (SSD2) ---
CLIP_HEF = "/mnt/moloch-data/hailo/models/zoo/vision/clip_resnet_50x4_image_encoder.hef"
OCR_DET_HEF = "/mnt/moloch-data/hailo/models/zoo/ocr/ocr_det.hef"
OCR_REC_HEF = "/mnt/moloch-data/hailo/models/zoo/ocr/ocr.hef"
VLM_HEF = "/mnt/moloch-data/hailo/models/zoo/vlm/Qwen2-VL-2B-Instruct.hef"

VDEVICE_GROUP_ID = "SHARED"
INFERENCE_TIMEOUT_MS = 10000

# SHM Frame-Buffer (gleich wie tappas_pipeline.py)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"
SHM_HEADER_SIZE = 24

# OCR CTC-Alphabet: PaddleOCR v5 Standard (96 druckbare + blank)
# Index 0 = blank, 1-96 = Zeichen
OCR_CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)


def _grab_shm_frame() -> Optional[np.ndarray]:
    """Aktuellen Frame aus SHM lesen (640x360 BGR)."""
    try:
        with open(SHM_FRAME_PATH, "rb") as f:
            header = f.read(SHM_HEADER_SIZE)
            if len(header) < SHM_HEADER_SIZE:
                return None
            h, w, c = struct.unpack("<III", header[:12])
            data = f.read(h * w * c)
            if len(data) != h * w * c:
                return None
            return np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
    except Exception:
        return None


def _create_configured_model(vdevice, hef_path: str, float_output: bool = True):
    """InferModel erstellen, Output auf FLOAT32 setzen, konfigurieren.

    Returns:
        (model, configured, input_shape, output_shapes)
    """
    model = vdevice.create_infer_model(hef_path)
    if float_output:
        from hailo_platform.pyhailort._pyhailort import FormatType
        for name in model.output_names:
            model.output(name).set_format_type(FormatType.FLOAT32)
    configured = model.configure()
    input_shape = list(model.input(model.input_names[0]).shape)
    output_shapes = {name: list(model.output(name).shape) for name in model.output_names}
    return model, configured, input_shape, output_shapes


class NpuExtras:
    """CLIP + OCR + VLM Inference auf Hailo-10H (shared VDevice)."""

    def __init__(self):
        self._lock = threading.Lock()

        # Lazy: werden erst beim ersten Aufruf initialisiert
        self._vdevice = None

        # CLIP
        self._clip_model = None
        self._clip_configured = None
        self._clip_out_shape = None

        # OCR
        self._ocr_det_model = None
        self._ocr_det_configured = None
        self._ocr_det_out_shape = None
        self._ocr_rec_model = None
        self._ocr_rec_configured = None
        self._ocr_rec_out_shape = None

        # VLM
        self._vlm = None

        # Statistiken
        self._clip_count = 0
        self._ocr_count = 0
        self._vlm_count = 0

    # =================================================================
    # VDevice (lazy, shared)
    # =================================================================

    def _ensure_vdevice(self):
        """Shared VDevice erstellen/joinen (gleich wie TAPPAS + hailo-ollama)."""
        if self._vdevice is not None:
            return
        import hailo_platform as hp
        params = hp.VDevice.create_params()
        params.group_id = VDEVICE_GROUP_ID
        self._vdevice = hp.VDevice(params)
        logger.info("[NPU-EXTRAS] VDevice joined (group=%s)", VDEVICE_GROUP_ID)

    # =================================================================
    # CLIP — Bild-Embedding (640d)
    # =================================================================

    def _ensure_clip(self):
        """CLIP InferModel laden."""
        if self._clip_configured is not None:
            return
        if not os.path.exists(CLIP_HEF):
            logger.error("[CLIP] HEF nicht gefunden: %s", CLIP_HEF)
            return
        self._ensure_vdevice()
        self._clip_model, self._clip_configured, inp_shape, out_shapes = \
            _create_configured_model(self._vdevice, CLIP_HEF)
        self._clip_out_shape = list(out_shapes.values())[0]
        logger.info("[CLIP] Modell geladen — Input %s, Output %s", inp_shape, self._clip_out_shape)

    def clip_embed(self, frame: np.ndarray = None) -> Optional[np.ndarray]:
        """640-dimensionales CLIP-Embedding erzeugen.

        Args:
            frame: BGR-Bild (optional, sonst aus SHM)

        Returns:
            np.ndarray (640,) L2-normalisiert, oder None bei Fehler
        """
        with self._lock:
            try:
                self._ensure_clip()
                if self._clip_configured is None:
                    return None

                if frame is None:
                    frame = _grab_shm_frame()
                if frame is None:
                    return None

                import cv2
                # Preprocessing: 288x288 RGB uint8
                img = cv2.resize(frame, (288, 288), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Bindings erstellen + Buffer setzen
                bindings = self._clip_configured.create_bindings()
                bindings.input().set_buffer(img)
                out_buf = np.empty(self._clip_out_shape, dtype=np.float32)
                bindings.output().set_buffer(out_buf)

                t0 = time.monotonic()
                self._clip_configured.run([bindings], INFERENCE_TIMEOUT_MS)
                dt = time.monotonic() - t0

                # (1, 1, 640) → (640,) + L2-Normalisierung
                embedding = out_buf.flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                self._clip_count += 1
                if self._clip_count <= 3 or self._clip_count % 50 == 0:
                    logger.info("[CLIP] #%d in %.0fms, norm=%.3f",
                                self._clip_count, dt * 1000, np.linalg.norm(embedding))
                return embedding

            except Exception as e:
                logger.error("[CLIP] Fehler: %s", e)
                return None

    # =================================================================
    # PaddleOCR — Text im Bild lesen
    # =================================================================

    def _ensure_ocr(self):
        """OCR Detection + Recognition InferModels laden."""
        if self._ocr_det_configured is not None:
            return
        if not os.path.exists(OCR_DET_HEF) or not os.path.exists(OCR_REC_HEF):
            logger.error("[OCR] HEF nicht gefunden: %s / %s", OCR_DET_HEF, OCR_REC_HEF)
            return
        self._ensure_vdevice()
        self._ocr_det_model, self._ocr_det_configured, _, det_outs = \
            _create_configured_model(self._vdevice, OCR_DET_HEF)
        self._ocr_det_out_shape = list(det_outs.values())[0]
        self._ocr_rec_model, self._ocr_rec_configured, _, rec_outs = \
            _create_configured_model(self._vdevice, OCR_REC_HEF)
        self._ocr_rec_out_shape = list(rec_outs.values())[0]
        logger.info("[OCR] Modelle geladen — Det→%s, Rec→%s",
                    self._ocr_det_out_shape, self._ocr_rec_out_shape)

    def _ocr_ctc_decode(self, logits: np.ndarray) -> str:
        """CTC Greedy Decode: (40, 97) Logits → Text."""
        indices = np.argmax(logits, axis=-1)  # (40,)
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx == 0:  # blank
                prev_idx = idx
                continue
            if idx == prev_idx:  # Duplikat
                continue
            if 1 <= idx <= len(OCR_CHARSET):
                chars.append(OCR_CHARSET[idx - 1])
            prev_idx = idx
        return "".join(chars)

    def ocr_read(self, frame: np.ndarray = None, conf_threshold: float = 0.3) -> List[Dict]:
        """Text im Bild erkennen (2-Stage: Detection + Recognition).

        Args:
            frame: BGR-Bild (optional, sonst aus SHM)
            conf_threshold: Mindest-Confidence fuer Text-Regionen

        Returns:
            Liste von {text, confidence, bbox: [x1,y1,x2,y2]}
        """
        with self._lock:
            try:
                self._ensure_ocr()
                if self._ocr_det_configured is None:
                    return []

                if frame is None:
                    frame = _grab_shm_frame()
                if frame is None:
                    return []

                import cv2
                orig_h, orig_w = frame.shape[:2]

                # --- Stage 1: Text Detection (544x960) ---
                det_input = cv2.resize(frame, (960, 544), interpolation=cv2.INTER_LINEAR)
                det_input = cv2.cvtColor(det_input, cv2.COLOR_BGR2RGB)

                det_bindings = self._ocr_det_configured.create_bindings()
                det_bindings.input().set_buffer(det_input)
                det_out_buf = np.empty(self._ocr_det_out_shape, dtype=np.float32)
                det_bindings.output().set_buffer(det_out_buf)

                t0 = time.monotonic()
                self._ocr_det_configured.run([det_bindings], INFERENCE_TIMEOUT_MS)
                dt_det = time.monotonic() - t0

                # Probability Map → Threshold + Konturen
                prob_map = det_out_buf.squeeze()  # (544, 960)
                binary = (prob_map > conf_threshold).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    return []

                # --- Stage 2: Text Recognition pro Region ---
                results = []
                scale_x = orig_w / 960.0
                scale_y = orig_h / 544.0

                for cnt in contours:
                    if cv2.contourArea(cnt) < 100:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    x1, y1 = max(0, int(x * scale_x)), max(0, int(y * scale_y))
                    x2, y2 = min(orig_w, int((x + w) * scale_x)), min(orig_h, int((y + h) * scale_y))

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    rec_input = cv2.resize(crop, (320, 48), interpolation=cv2.INTER_LINEAR)
                    rec_input = cv2.cvtColor(rec_input, cv2.COLOR_BGR2RGB)

                    rec_bindings = self._ocr_rec_configured.create_bindings()
                    rec_bindings.input().set_buffer(rec_input)
                    rec_out_buf = np.empty(self._ocr_rec_out_shape, dtype=np.float32)
                    rec_bindings.output().set_buffer(rec_out_buf)

                    self._ocr_rec_configured.run([rec_bindings], INFERENCE_TIMEOUT_MS)

                    logits = rec_out_buf.squeeze()  # (40, 97)
                    text = self._ocr_ctc_decode(logits)
                    if text.strip():
                        conf_vals = np.max(logits, axis=-1)
                        avg_conf = float(np.mean(conf_vals[conf_vals > 0]))
                        results.append({
                            "text": text,
                            "confidence": round(avg_conf, 3),
                            "bbox": [x1, y1, x2, y2]
                        })

                self._ocr_count += 1
                dt_total = time.monotonic() - t0
                if results:
                    logger.info("[OCR] #%d: %d Texte in %.0fms (det=%.0fms) — %s",
                                self._ocr_count, len(results), dt_total * 1000,
                                dt_det * 1000,
                                "; ".join(r["text"][:30] for r in results))
                return results

            except Exception as e:
                logger.error("[OCR] Fehler: %s", e)
                return []

    # =================================================================
    # Qwen2-VL — Szenen-Beschreibung
    # =================================================================

    def _ensure_vlm(self):
        """VLM laden (Qwen2-VL-2B via hailo_platform.genai)."""
        if self._vlm is not None:
            return
        if not os.path.exists(VLM_HEF):
            logger.error("[VLM] HEF nicht gefunden: %s", VLM_HEF)
            return
        self._ensure_vdevice()
        from hailo_platform.genai import VLM
        self._vlm = VLM(self._vdevice, VLM_HEF)
        logger.info("[VLM] Qwen2-VL-2B geladen — Input 336x336x3")

    def vlm_describe(self, frame: np.ndarray = None,
                     prompt: str = "Beschreibe was du siehst. Kurz und praezise, auf Deutsch.",
                     max_tokens: int = 150) -> str:
        """Szene beschreiben mit Qwen2-VL-2B.

        Args:
            frame: BGR-Bild (optional, sonst aus SHM)
            prompt: Frage an das VLM
            max_tokens: Max. Token-Laenge der Antwort

        Returns:
            Freitext-Beschreibung oder leerer String bei Fehler
        """
        with self._lock:
            try:
                self._ensure_vlm()
                if self._vlm is None:
                    return ""

                if frame is None:
                    frame = _grab_shm_frame()
                if frame is None:
                    return ""

                import cv2
                img = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "Du bist M.O.L.O.C.H., eine KI mit Kamera. Beschreibe praezise was du siehst."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                t0 = time.monotonic()
                response = self._vlm.generate_all(
                    prompt=messages,
                    frames=[img],
                    temperature=0.1,
                    seed=42,
                    max_generated_tokens=max_tokens
                )
                dt = time.monotonic() - t0

                text = response.split("<|im_end|>")[0].strip()
                text = text.split(". [{'type'")[0].strip()

                self._vlm_count += 1
                logger.info("[VLM] #%d in %.1fs: %s", self._vlm_count, dt, text[:80])
                return text

            except Exception as e:
                logger.error("[VLM] Fehler: %s", e)
                return ""

    # =================================================================
    # Status + Cleanup
    # =================================================================

    def get_status(self) -> Dict:
        """Aktueller Status aller Extra-Modelle."""
        return {
            "clip_loaded": self._clip_configured is not None,
            "clip_inferences": self._clip_count,
            "ocr_loaded": self._ocr_det_configured is not None,
            "ocr_inferences": self._ocr_count,
            "vlm_loaded": self._vlm is not None,
            "vlm_inferences": self._vlm_count,
            "vdevice_active": self._vdevice is not None,
        }

    def stop(self):
        """Alle Modelle freigeben."""
        with self._lock:
            if self._vlm:
                try:
                    self._vlm.clear_context()
                    self._vlm.release()
                except Exception:
                    pass
                self._vlm = None

            for cfg in [self._clip_configured, self._ocr_det_configured, self._ocr_rec_configured]:
                if cfg:
                    try:
                        cfg.shutdown()
                    except Exception:
                        pass
            self._clip_configured = None
            self._clip_model = None
            self._ocr_det_configured = None
            self._ocr_det_model = None
            self._ocr_rec_configured = None
            self._ocr_rec_model = None

            if self._vdevice:
                try:
                    self._vdevice.release()
                except Exception:
                    pass
                self._vdevice = None

            logger.info("[NPU-EXTRAS] Alle Modelle freigegeben")


# Singleton
_instance = None
_instance_lock = threading.Lock()


def get_npu_extras() -> NpuExtras:
    """Globale NpuExtras-Instanz (lazy Singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NpuExtras()
    return _instance
