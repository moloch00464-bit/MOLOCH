#!/usr/bin/env python3
"""
TappasPipeline — GStreamer/TAPPAS Multi-Model NPU Pipeline.

Gate 0.5 Phase 3.1: Ersetzt inference_engine.py + model_orchestrator.py
durch native TAPPAS-Pipeline mit Model Scheduler.

Pipeline:
  rtspsrc → YOLO_WRAPPER(letterbox→hailonet→yolo_postproc)
          → SCRFD_WRAPPER(letterbox→hailonet→scrfd_postproc)
          → hailotracker
          → FACE_CROPPER(face_align→hailonet_arcface→arcface_postproc)
          → identity_callback → hailooverlay → appsink

Alle Modelle teilen sich den NPU via vdevice-group-id=SHARED (Model Scheduler).

Nutzung:
    # Feature-Flag: MOLOCH_USE_TAPPAS=1
    from core.perception.tappas_pipeline import TappasPipeline
    pipeline = TappasPipeline()
    pipeline.start()
    detections = pipeline.get_detections()
    pframe = pipeline.get_current_pframe()
    pipeline.stop()
"""

import os
import time
import struct
import threading
import logging
from typing import List, Dict, Optional

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import hailo

from core.perception.perception_frame import PerceptionFrame, estimate_distance

logger = logging.getLogger("TappasPipeline")

# --- Modell-Pfade (SSD2) ---
YOLO_HEF = "/mnt/moloch-data/hailo/models/yolov8m_h10.hef"
SCRFD_HEF = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
ARCFACE_HEF = "/mnt/moloch-data/hailo/models/arcface_mobilefacenet.hef"

# --- Postprocess SOs ---
YOLO_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
YOLO_POSTPROCESS_FUNC = "filter_letterbox"
SCRFD_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libscrfd.so"
SCRFD_POSTPROCESS_FUNC = "scrfd_10g_letterbox"
SCRFD_CONFIG_JSON = "/usr/local/hailo/resources/json/scrfd.json"
ARCFACE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libface_recognition_post.so"
ARCFACE_POSTPROCESS_FUNC = "filter"
FACE_ALIGN_SO = "/usr/local/hailo/resources/so/libvms_face_align.so"
FACE_CROP_SO = "/usr/local/hailo/resources/so/libvms_croppers.so"
FACE_CROP_FUNC = "face_recognition"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

VDEVICE_GROUP_ID = "SHARED"

# IPC: Frame-Preview fuer Panel (gleicher Weg wie InferenceEngine → IPCRouter)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"
SHM_PREVIEW_W = 640
SHM_PREVIEW_H = 360


def _build_rtsp_url() -> str:
    """RTSP-URL aus Environment-Variablen zusammenbauen."""
    # Zuerst: fertige URL aus ENV
    url = os.environ.get("MOLOCH_RTSP_URL", "")
    if url:
        return url
    # Fallback: Einzelteile
    host = os.environ.get("MOLOCH_CAMERA_HOST", "192.168.178.25")
    user = os.environ.get("MOLOCH_CAMERA_USER", "")
    pw = os.environ.get("MOLOCH_CAMERA_PASS", "")
    if user and pw:
        return f"rtsp://{user}:{pw}@{host}:554/av_stream/ch0"
    return f"rtsp://{host}:554/av_stream/ch0"


class TappasPipeline:
    """GStreamer/TAPPAS Multi-Model NPU Pipeline.

    Gleiche Output-Struktur wie InferenceEngine, damit moloch_service.py
    beide nutzen kann (Feature-Flag MOLOCH_USE_TAPPAS=1).
    """

    def __init__(self, rtsp_url: str = None, face_db: dict = None,
                 width: int = 1280, height: int = 720):
        """
        Args:
            rtsp_url: RTSP Stream URL (oder None fuer ENV-Variable)
            face_db: Face-Embedding-DB {name: np.array} fuer ArcFace-Matching
            width: Pipeline-Breite (nach Source-Scale)
            height: Pipeline-Hoehe (nach Source-Scale)
        """
        self._rtsp_url = rtsp_url or _build_rtsp_url()
        self._width = width
        self._height = height

        # Face DB fuer Embedding-Matching
        self._face_db = face_db or {}
        self._face_db_lock = threading.Lock()

        # State
        self._pipeline = None
        self._loop = None
        self._loop_thread = None
        self._running = False

        # Detections (thread-safe, letzter Frame)
        self._lock = threading.Lock()
        self._detections: List[Dict] = []
        self._current_pframe = PerceptionFrame()
        self._annotated_frame: Optional[np.ndarray] = None

        # FPS Tracking
        self._fps_lock = threading.Lock()
        self._frame_count = 0
        self._fps_start = 0.0
        self._fps_last_count = 0
        self._fps_last_time = 0.0
        self._current_fps = 0.0

        # SHM IPC Sequenznummer
        self._shm_seq = 0

        # --- Model-Active-Flags (TAPPAS = immer aktiv, Panel liest diese) ---
        self.scrfd_active = True
        self.arcface_active = True
        self.yolo_active = True
        self.hand_active = False   # Hand-Modell nicht in TAPPAS Pipeline
        self.pose_active = False   # Pose-Modell nicht in TAPPAS Pipeline

        # --- Threshold-Werte (Panel setzt diese, TAPPAS managed intern) ---
        self.scrfd_conf_val = 0.30
        self.scrfd_nms_val = 0.45
        self.arcface_thresh_val = 0.50
        self.yolo_conf_val = 0.30
        self.pose_conf_val = 0.30
        self.hand_conf_val = 0.30

        # --- Feature-Flags (Panel/Settings lesen/schreiben diese) ---
        self._learner_flash = False
        self._hand_occlusion_enabled = False

        # GStreamer einmal initialisieren
        if not Gst.is_initialized():
            Gst.init(None)

    # =====================================================================
    # Public API
    # =====================================================================

    def start(self):
        """Pipeline starten. Blockiert NICHT — laeuft in eigenem Thread."""
        if self._running:
            logger.warning("Pipeline laeuft bereits")
            return

        logger.info("Starte TAPPAS Multi-Model Pipeline...")
        logger.info(f"  RTSP: {self._rtsp_url.split('@')[1] if '@' in self._rtsp_url else self._rtsp_url}")
        logger.info(f"  Modelle: YOLO + SCRFD + ArcFace (vdevice-group-id={VDEVICE_GROUP_ID})")

        pipeline_str = self._build_pipeline_string()

        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            logger.error(f"Pipeline-Erstellen fehlgeschlagen: {e}")
            raise RuntimeError(f"GStreamer Pipeline Error: {e}")

        # Identity Callback (Pad-Probe fuer Detection-Auswertung)
        identity = self._pipeline.get_by_name("identity_callback")
        if identity is None:
            raise RuntimeError("identity_callback Element nicht in Pipeline gefunden")
        pad = identity.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer, None)

        # appsink — Frames abholen damit Pipeline nicht blockiert
        appsink = self._pipeline.get_by_name("sink")
        if appsink:
            appsink.connect("new-sample", self._on_appsink_sample)

        # Bus fuer Fehler/EOS
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # FPS Reset
        self._fps_start = time.time()
        self._fps_last_time = self._fps_start
        self._fps_last_count = 0
        self._frame_count = 0

        # Pipeline starten
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Pipeline konnte nicht gestartet werden")

        self._running = True

        # GLib MainLoop in eigenem Thread
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="TappasPipeline-GLib", daemon=True
        )
        self._loop_thread.start()

        logger.info("TAPPAS Pipeline gestartet")

    def stop(self):
        """Pipeline sauber beenden."""
        if not self._running:
            return

        logger.info("Stoppe TAPPAS Pipeline...")
        self._running = False

        if self._loop and self._loop.is_running():
            self._loop.quit()

        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        self._pipeline = None
        self._loop = None
        self._loop_thread = None

        logger.info("TAPPAS Pipeline gestoppt")

    def is_running(self) -> bool:
        return self._running

    def get_detections(self) -> List[Dict]:
        """Letzte Detections als Liste von Dicts.

        Returns:
            [{"class": "person"/"face", "bbox": [x1, y1, x2, y2],
              "confidence": float, "embedding": np.array or None,
              "track_id": int or None}]

        BBox-Koordinaten sind normalisiert (0.0-1.0).
        """
        with self._lock:
            return list(self._detections)

    def get_current_pframe(self) -> PerceptionFrame:
        """Letzten aggregierten PerceptionFrame zurueckgeben.

        Kompatibel mit InferenceEngine.get_current_pframe().
        """
        with self._lock:
            return self._current_pframe

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Letzten annotierten Frame (RGB, mit Overlay) zurueckgeben."""
        with self._lock:
            return self._annotated_frame

    def get_fps(self) -> dict:
        """FPS-Snapshot zurueckgeben. Kompatibel mit InferenceEngine.get_fps()."""
        with self._fps_lock:
            return {
                "scrfd": self._current_fps,
                "arcface": self._current_fps,
                "yolov8m": self._current_fps,
                "total": self._current_fps,
                # Modelle laufen parallel in Pipeline — gleiche Frame-Rate
            }

    def reload_face_db(self, face_db: dict = None):
        """Face-DB aktualisieren.

        Wenn face_db=None (Service ruft ohne Parameter auf), wird die DB
        aus data/face_embeddings.json geladen (gleicher Weg wie InferenceEngine).
        """
        if face_db is None:
            face_db = self._load_face_db_from_disk()
        with self._face_db_lock:
            self._face_db = face_db
        logger.info(f"Face-DB aktualisiert: {len(self._face_db)} Personen")

    def sync_flags_from_npu(self):
        """No-op: TAPPAS Pipeline hat alle Modelle permanent aktiv."""
        pass

    def reset_fps(self):
        """FPS-Counter zuruecksetzen (z.B. nach Stream-Restart)."""
        with self._fps_lock:
            self._frame_count = 0
            self._fps_start = time.time()
            self._fps_last_count = 0
            self._fps_last_time = self._fps_start
            self._current_fps = 0.0

    # =====================================================================
    # Face-DB von Disk laden
    # =====================================================================

    def _load_face_db_from_disk(self) -> dict:
        """Face-Embeddings aus data/face_embeddings.json laden.

        Gleicher Pfad wie InferenceEngine nutzt.
        Returns: {name: np.array} oder {} bei Fehler.
        """
        import json
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "face_embeddings.json"
        )
        if not os.path.exists(embeddings_path):
            logger.warning(f"Face-DB nicht gefunden: {embeddings_path}")
            return {}
        try:
            with open(embeddings_path, 'r') as f:
                raw = json.load(f)
            db = {}
            for name, emb_list in raw.items():
                emb = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                db[name] = emb
            logger.info(f"Face-DB von Disk geladen: {len(db)} Personen aus {embeddings_path}")
            return db
        except Exception as e:
            logger.error(f"Face-DB laden fehlgeschlagen: {e}")
            return {}

    # =====================================================================
    # GStreamer Pipeline String
    # =====================================================================

    def _build_pipeline_string(self) -> str:
        """Komplette Multi-Model Pipeline (YOLO + SCRFD + Tracker + ArcFace)."""

        # --- Source: RTSP → H264 depay → decode → scale → RGB ---
        source = (
            f'rtspsrc location="{self._rtsp_url}" name=source latency=300 protocols=tcp ! '
            f'rtph264depay name=source_depay ! '
            f'queue name=source_queue_decode leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
            f'avdec_h264 name=source_decode ! '
            f'queue name=source_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=source_videoscale n-threads=2 ! '
            f'queue name=source_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=3 name=source_convert qos=false ! '
            f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width={self._width}, height={self._height} '
        )

        # --- Stage 1: YOLO Person Detection ---
        yolo_inner = (
            f'queue name=yolo_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=yolo_videoscale n-threads=2 qos=false ! '
            f'queue name=yolo_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=yolo_videoconvert n-threads=2 ! '
            f'queue name=yolo_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=yolo_hailonet hef-path={YOLO_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'nms-score-threshold=0.3 nms-iou-threshold=0.45 '
            f'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 '
            f'force-writable=true ! '
            f'queue name=yolo_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=yolo_hailofilter so-path={YOLO_POSTPROCESS_SO} '
            f'function-name={YOLO_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=yolo_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        yolo_wrapper = (
            f'queue name=yolo_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=yolo_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=true '
            f'hailoaggregator name=yolo_wrapper_agg '
            f'yolo_wrapper_crop. ! queue name=yolo_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! yolo_wrapper_agg.sink_0 '
            f'yolo_wrapper_crop. ! {yolo_inner} ! yolo_wrapper_agg.sink_1 '
            f'yolo_wrapper_agg. ! queue name=yolo_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Stage 2: SCRFD Face Detection ---
        scrfd_inner = (
            f'queue name=scrfd_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=scrfd_videoscale n-threads=2 qos=false ! '
            f'queue name=scrfd_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=scrfd_videoconvert n-threads=2 ! '
            f'queue name=scrfd_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=scrfd_hailonet hef-path={SCRFD_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'force-writable=true ! '
            f'queue name=scrfd_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=scrfd_hailofilter so-path={SCRFD_POSTPROCESS_SO} '
            f'function-name={SCRFD_POSTPROCESS_FUNC} config-path={SCRFD_CONFIG_JSON} qos=false ! '
            f'queue name=scrfd_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        scrfd_wrapper = (
            f'queue name=scrfd_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=scrfd_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=true '
            f'hailoaggregator name=scrfd_wrapper_agg '
            f'scrfd_wrapper_crop. ! queue name=scrfd_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! scrfd_wrapper_agg.sink_0 '
            f'scrfd_wrapper_crop. ! {scrfd_inner} ! scrfd_wrapper_agg.sink_1 '
            f'scrfd_wrapper_agg. ! queue name=scrfd_wrapper_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Stage 3: Tracker + Face Cropper (face_align + ArcFace) ---
        tracker = (
            f'hailotracker name=hailo_face_tracker class-id=-1 '
            f'kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 '
            f'keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 '
            f'keep-past-metadata=true qos=false ! '
            f'queue name=tracker_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        arcface_inner = (
            f'hailofilter so-path={FACE_ALIGN_SO} name=face_align_hailofilter use-gst-buffer=true qos=false ! '
            f'queue name=face_align_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'queue name=arcface_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=arcface_videoscale n-threads=2 qos=false ! '
            f'queue name=arcface_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=arcface_videoconvert n-threads=2 ! '
            f'queue name=arcface_hailonet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=arcface_hailonet hef-path={ARCFACE_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'force-writable=true ! '
            f'queue name=arcface_hailofilter_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=arcface_hailofilter so-path={ARCFACE_POSTPROCESS_SO} '
            f'function-name={ARCFACE_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=arcface_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        face_cropper = (
            f'queue name=face_crop_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=face_cropper so-path={FACE_CROP_SO} function-name={FACE_CROP_FUNC} '
            f'use-letterbox=true no-scaling-bbox=true internal-offset=true resize-method=bilinear '
            f'hailoaggregator name=face_crop_agg '
            f'face_cropper. ! queue name=face_crop_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! face_crop_agg.sink_0 '
            f'face_cropper. ! {arcface_inner} ! face_crop_agg.sink_1 '
            f'face_crop_agg. ! queue name=face_crop_output_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 '
        )

        # --- Callback + Overlay + appsink ---
        callback_and_sink = (
            f'queue name=cb_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback ! '
            f'queue name=overlay_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'hailooverlay name=hailo_overlay ! '
            f'queue name=sink_convert_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=2 qos=false ! '
            f'video/x-raw, format=RGB ! '
            f'appsink name=sink emit-signals=true drop=true max-buffers=2 sync=false'
        )

        return (
            f'{source} ! '
            f'{yolo_wrapper} ! '
            f'{scrfd_wrapper} ! '
            f'{tracker} ! '
            f'{face_cropper} ! '
            f'{callback_and_sink}'
        )

    # =====================================================================
    # GStreamer Callbacks
    # =====================================================================

    def _on_buffer(self, pad, info, user_data):
        """Pad-Probe auf identity element — extrahiert Detections + baut PerceptionFrame."""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        detections = []
        persons = []
        faces = []
        best_face_conf = 0.0
        best_face_bbox = None
        face_id = None
        face_similarity = 0.0

        for det in hailo_detections:
            label = det.get_label()
            conf = det.get_confidence()
            bbox = det.get_bbox()

            # Threshold-Filterung (Panel-Slider Werte anwenden)
            if label == "person" and conf < self.yolo_conf_val:
                continue
            if label == "face" and conf < self.scrfd_conf_val:
                continue

            # Normalisierte BBox [0.0-1.0]
            x1 = bbox.xmin()
            y1 = bbox.ymin()
            x2 = bbox.xmax()
            y2 = bbox.ymax()

            entry = {
                "class": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "embedding": None,
                "track_id": None,
            }

            # Track-ID aus hailotracker
            track_ids = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if track_ids:
                entry["track_id"] = track_ids[0].get_id()

            if label == "person":
                persons.append(entry)
            elif label == "face":
                # ArcFace Embedding extrahieren
                embeddings = det.get_objects_typed(hailo.HAILO_MATRIX)
                if embeddings:
                    emb_data = np.array(embeddings[0].get_data(), dtype=np.float32)
                    entry["embedding"] = emb_data

                    # Face-Matching gegen DB
                    matched_name, matched_sim = self._match_face(emb_data)
                    if matched_name:
                        entry["face_id"] = matched_name
                        entry["face_similarity"] = matched_sim

                faces.append(entry)

                if conf > best_face_conf:
                    best_face_conf = conf
                    best_face_bbox = (x1, y1, x2, y2)

            detections.append(entry)

        # Bestes Face-Match fuer PerceptionFrame
        for f in faces:
            if f.get("face_similarity", 0) > face_similarity:
                face_id = f.get("face_id")
                face_similarity = f.get("face_similarity", 0)

        # PerceptionFrame bauen
        pf = self._build_pframe(persons, faces, best_face_conf,
                                best_face_bbox, face_id, face_similarity)

        # Thread-safe update (Frame kommt aus appsink — NACH hailooverlay)
        with self._lock:
            self._detections = detections
            self._current_pframe = pf

        # FPS Tracking
        self._update_fps()

        return Gst.PadProbeReturn.OK

    def _on_appsink_sample(self, appsink):
        """appsink Callback — annotiertes Frame (MIT BBoxen von hailooverlay) extrahieren."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        if buf is None or caps is None:
            return Gst.FlowReturn.OK

        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(mapinfo)

        expected = width * height * 3
        if len(data) != expected:
            return Gst.FlowReturn.OK

        frame = data.reshape(height, width, 3)

        # Thread-safe: annotiertes Frame speichern
        with self._lock:
            self._annotated_frame = frame

        # SHM IPC: Annotiertes Frame (MIT BBoxen) fuer Panel Preview
        self._write_shm_frame(frame)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """GStreamer Bus Messages verarbeiten."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer Fehler: {err}")
            logger.debug(f"GStreamer Debug: {debug}")
            self._running = False
            if self._loop and self._loop.is_running():
                self._loop.quit()
        elif t == Gst.MessageType.EOS:
            logger.warning("End-of-Stream — Pipeline beendet")
            self._running = False
            if self._loop and self._loop.is_running():
                self._loop.quit()
        return True

    def _run_loop(self):
        """GLib MainLoop in eigenem Thread ausfuehren."""
        try:
            self._loop.run()
        except Exception as e:
            logger.error(f"GLib MainLoop Fehler: {e}")
        finally:
            self._running = False

    # =====================================================================
    # Frame-Extraktion
    # =====================================================================

    def _extract_frame(self, buffer, pad) -> Optional[np.ndarray]:
        """Extrahiert numpy Frame (RGB) aus GStreamer Buffer."""
        caps = pad.get_current_caps()
        if caps is None:
            return None
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buffer.unmap(mapinfo)

        expected = width * height * 3
        if len(data) != expected:
            return None
        return data.reshape(height, width, 3)

    # =====================================================================
    # Face Matching
    # =====================================================================

    def _match_face(self, embedding: np.ndarray) -> tuple:
        """Face-Embedding gegen DB matchen.

        Returns:
            (name, similarity) oder (None, 0.0)
        """
        with self._face_db_lock:
            if not self._face_db:
                return (None, 0.0)

            # Embedding normalisieren
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_name = None
            best_sim = 0.0
            threshold = self.arcface_thresh_val  # Panel-Slider Wert

            for name, db_emb in self._face_db.items():
                sim = float(np.dot(embedding, db_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim >= threshold:
                return (best_name, best_sim)
            return (None, 0.0)

    # =====================================================================
    # PerceptionFrame Builder
    # =====================================================================

    def _build_pframe(self, persons: list, faces: list, best_face_conf: float,
                      best_face_bbox: tuple, face_id: str, face_similarity: float) -> PerceptionFrame:
        """Baut PerceptionFrame aus Detections — kompatibel mit InferenceEngine."""
        pf = PerceptionFrame()
        pf.timestamp = time.time()

        # Person Detection
        pf.person_detected = len(persons) > 0 or len(faces) > 0
        pf.person_count = len(persons) if persons else (1 if faces else 0)

        # Distanz aus groesster Person-BBox
        if persons:
            biggest = max(persons, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
            bbox = biggest["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)
        elif best_face_bbox:
            area = (best_face_bbox[2] - best_face_bbox[0]) * (best_face_bbox[3] - best_face_bbox[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)

        # Face Detection
        pf.face_detected = len(faces) > 0
        pf.face_count = len(faces)
        if faces:
            pf.face_confidence = best_face_conf
            pf.face_bbox = best_face_bbox

        # Face Recognition
        if face_id:
            pf.face_id = face_id.lower()
            pf.face_similarity = face_similarity

        # Active Models (TAPPAS Pipeline = alle immer aktiv)
        pf.active_models = ["yolov8m", "scrfd", "arcface"]

        return pf

    # =====================================================================
    # FPS
    # =====================================================================

    def _update_fps(self):
        """FPS alle 2 Sekunden aktualisieren."""
        with self._fps_lock:
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_last_time
            if elapsed >= 2.0:
                frames = self._frame_count - self._fps_last_count
                self._current_fps = frames / elapsed
                self._fps_last_time = now
                self._fps_last_count = self._frame_count

    # =====================================================================
    # SHM IPC (Panel Preview)
    # =====================================================================

    def _write_shm_frame(self, frame: np.ndarray):
        """Frame nach /dev/shm/moloch_frame schreiben (gleicher IPC-Weg wie InferenceEngine).

        Konvertiert RGB→BGR und skaliert auf Preview-Groesse.
        """
        try:
            # RGB → BGR (Panel erwartet BGR)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if h != SHM_PREVIEW_H or w != SHM_PREVIEW_W:
                bgr = cv2.resize(bgr, (SHM_PREVIEW_W, SHM_PREVIEW_H))
                h, w = SHM_PREVIEW_H, SHM_PREVIEW_W
            c = bgr.shape[2] if len(bgr.shape) > 2 else 1
            self._shm_seq = (self._shm_seq + 1) & 0xFFFFFFFF
            header = struct.pack('<IIII', h, w, c, self._shm_seq)
            with open(SHM_FRAME_PATH + '.tmp', 'wb') as f:
                f.write(header)
                f.write(bgr.tobytes())
            os.rename(SHM_FRAME_PATH + '.tmp', SHM_FRAME_PATH)
        except Exception:
            pass
