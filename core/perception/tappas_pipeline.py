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
from core.perception.model_scheduler import ModelScheduler
from core.perception.temporal_memory import get_perception_memory
from core.perception.action_inference import get_action_inferrer
from core.moloch_event_bus import get_event_bus, PRIO_PERCEPTION
from core.perception.vision_workers import WorkItem, ResultCollector
from core.perception.roi_dispatcher import ROIDispatcher
from core.perception.face_pipeline import FaceWorker
from core.perception.pose_worker import PoseWorker, ReIDWorker, HandWorker
from core.perception.person_attr_worker import PersonAttrWorker
from core.perception.activity_worker import ActivityWorker
from core.perception.yolo_world_worker import YOLOWorldWorker

logger = logging.getLogger("TappasPipeline")

# --- Modell-Pfade (SSD2) ---
YOLO_HEF = "/mnt/moloch-data/hailo/models/yolov11m_h10.hef"
SCRFD_HEF = "/mnt/moloch-data/hailo/models/scrfd_10g.hef"
ARCFACE_HEF = "/mnt/moloch-data/hailo/models/arcface_mobilefacenet.hef"
FACE_ATTR_HEF = "/mnt/moloch-data/hailo/models/face_attr_resnet_v1_18.hef"

# --- Postprocess SOs ---
YOLO_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
YOLO_POSTPROCESS_FUNC = "filter_letterbox"
SCRFD_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libscrfd.so"
SCRFD_POSTPROCESS_FUNC = "scrfd_10g_letterbox"
SCRFD_CONFIG_JSON = "/usr/local/hailo/resources/json/scrfd.json"
ARCFACE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libface_recognition_post.so"
ARCFACE_POSTPROCESS_FUNC = "filter"
FACE_ATTR_POSTPROCESS_SO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "lib", "libface_attributes_fc2.so")
FACE_ATTR_POSTPROCESS_FUNC = "filter"
FACE_ALIGN_SO = "/usr/local/hailo/resources/so/libvms_face_align.so"
FACE_CROP_SO = "/usr/local/hailo/resources/so/libvms_croppers.so"
FACE_CROP_FUNC = "face_recognition"
WHOLE_BUFFER_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"

# --- Pose Estimation (YOLOv8s Pose) ---
POSE_HEF = "/mnt/moloch-data/hailo/models/yolov8s_pose_h10.hef"
POSE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libyolov8pose_postprocess.so"
POSE_POSTPROCESS_FUNC = "filter_letterbox"  # Letterbox-Korrektur fuer Keypoints

# --- Person ReID (RepVGG-A0, 512d Embedding) ---
REID_HEF = "/mnt/moloch-data/hailo/models/repvgg_a0_person_reid_512.hef"
REID_POSTPROCESS_SO = "/usr/local/hailo/resources/so/librepvgg_reid_postprocess.so"
REID_POSTPROCESS_FUNC = "filter"
REID_CROP_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libre_id.so"
REID_CROP_FUNC = "create_crops"

# Face-BBox: Native Hailo-Koordinaten sind KORREKT (hailocropper + scrfd_10g_letterbox).
# Keine Python-seitige Korrektur noetig — Markus bestaetigt visuell (2026-03-30).

# --- Debug-Overlay: Dicke BBoxen + Landmarks fuer Snapshot-Analyse ---
# True = dicke Linien im SHM-Frame (fuer Claude-Referenzbilder)
DEBUG_THICK_OVERLAY = False

# --- Hand Landmark (PoC, Full-Frame) ---
HAND_HEF = "/mnt/moloch-data/hailo/models/hand_landmark_lite.hef"

# --- PaddleOCR (Text im Raum lesen, 2-Stage: Detection + Recognition) ---
OCR_DET_HEF = "/mnt/moloch-data/hailo/models/zoo/ocr/ocr_det.hef"
OCR_REC_HEF = "/mnt/moloch-data/hailo/models/zoo/ocr/ocr.hef"
OCR_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libocr_postprocess.so"
OCR_DET_FUNC = "paddleocr_det"
OCR_CROP_FUNC = "crop_text_regions_filter"
OCR_REC_FUNC = "paddleocr_recognize"

VDEVICE_GROUP_ID = "SHARED"

# YOLO Klassen-Whitelist (nur diese werden verarbeitet, Rest ignoriert)
YOLO_ALLOWED_CLASSES = {"person"}

# NPU Model-Scheduler Zustände
SCHED_YOLO_ONLY = "YOLO_ONLY"    # Niemand da → nur YOLO aktiv
SCHED_YOLO_SCRFD = "YOLO_SCRFD"  # Person erkannt → YOLO + SCRFD
SCHED_ALL_ACTIVE = "ALL_ACTIVE"   # Gesicht sichtbar → alle Modelle
SCHED_COOLDOWN_DOWN = 3.0         # Sekunden ohne Daten bis Downgrade

# IPC: Frame-Preview fuer Panel (mmap-basiert, kein file-rename mehr)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"
SHM_PREVIEW_W = 640
SHM_PREVIEW_H = 360
SHM_HEADER_SIZE = 24   # h(4) + w(4) + c(4) + seq(4) + ts(8)
SHM_DATA_SIZE = SHM_PREVIEW_W * SHM_PREVIEW_H * 3
SHM_TOTAL_SIZE = SHM_HEADER_SIZE + SHM_DATA_SIZE  # 691224 Bytes


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

        # Face Attributes Cache (befuellt von _on_face_attr_buffer Probe)
        self._face_attr_cache = {}  # {"gender": "M"|"F", "smiling": True|False}
        self._face_attr_lock = threading.Lock()

        # --- Model-Active-Flags (Scheduler-basiert, Panel liest diese) ---
        # Start in YOLO_ONLY → scrfd/arcface inaktiv bis Person erkannt
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = True
        self.hand_active = False   # Hand-Modell nicht in TAPPAS Pipeline
        self.pose_active = False   # Pose-Modell nicht in TAPPAS Pipeline

        # --- Threshold-Werte (Panel setzt diese, TAPPAS managed intern) ---
        self.scrfd_conf_val = 0.30
        self.scrfd_nms_val = 0.45
        self.arcface_thresh_val = 0.70
        self.yolo_conf_val = 0.30
        self.pose_conf_val = 0.30
        self.hand_conf_val = 0.30

        # --- Feature-Flags (Panel/Settings lesen/schreiben diese) ---
        self._learner_flash = False
        self._hand_occlusion_enabled = False

        # --- Live-Enrollment ---
        self._enroll_active = False
        self._enroll_done = False
        self._enroll_name = ""
        self._enroll_target = 20
        self._enroll_candidates = []  # (score, embedding_copy)
        self._enroll_min_score = 0.50
        self._enroll_diversity = 0.85  # Cosine-Sim Schwelle
        self._enroll_lock = threading.Lock()

        # --- Passives Continuous-Learning ---
        # Wenn Owner erkannt wird UND neuer Winkel, Embedding automatisch speichern
        self._cl_enabled = True           # Feature aktiv
        self._cl_interval_sec = 30.0      # Min. Sekunden zwischen Speicherungen
        self._cl_last_save = 0.0          # Zeitstempel letzte Speicherung
        self._cl_min_sim = 0.55           # Min. Similarity (muss schon als Owner erkannt sein)
        self._cl_max_sim = 0.92           # Max. Similarity (ueber 0.92 = bekannter Winkel)
        self._cl_min_scrfd = 0.70         # Min. SCRFD Confidence (Gesicht gut sichtbar)
        self._cl_max_embeddings = 50      # Max. Embeddings pro Person
        self._cl_diversity_thresh = 0.80  # Neues Embedding muss sich unterscheiden

        # Event Bus fuer Action Bridge
        self._event_bus = get_event_bus()
        self._last_person_state = False  # Fuer target_lost Erkennung

        # --- NPU Model-Scheduler ---
        self._sched_mode = SCHED_YOLO_ONLY
        self._sched_person_last_seen = 0.0
        self._sched_face_last_seen = 0.0
        self._sched_lock = threading.Lock()
        self._sched_force_all = False  # Teach-Modus: Scheduler auf ALL_ACTIVE erzwingen

        # --- Perception Router (neuer 7-Szenario Scheduler, Phase 1: nur Logging) ---
        self._scheduler = ModelScheduler()

        # --- SCRFD Valve-Gating (echtes NPU-Gating via GStreamer valve) ---
        self._scrfd_valve = None       # valve element: drop=True = SCRFD aus
        self._scrfd_selector = None    # input-selector: sink_0=SCRFD, sink_1=Bypass

        # --- Pose Valve-Gating ---
        self._pose_valve = None
        self._pose_selector = None
        self._pose_gate_state = False  # Aktueller Valve-Zustand

        # --- ReID Valve-Gating ---
        self._reid_valve = None
        self._reid_selector = None

        # --- Hand Detection Valve-Gating ---
        self._hand_valve = None
        self._hand_selector = None

        # --- OCR Valve-Gating ---
        self._ocr_valve = None
        self._ocr_selector = None
        self._ocr_enabled = False
        self._last_ocr_texts: List[str] = []  # Letzte erkannte Texte

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
        logger.info(f"  YOLO filter: person-only active (allowed={YOLO_ALLOWED_CLASSES})")

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

        # Phase 2: Keine Valve-Elemente mehr — alle Modelle ausser YOLO laufen als Worker

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

        # --- HailoRT-Direct Worker starten (Phase 2-5) ---
        try:
            self._roi_dispatcher = ROIDispatcher()
            self._result_collector = ResultCollector()

            # FaceWorker: SCRFD + ArcFace + FaceAttr (jeden 2. Frame)
            self._face_worker = FaceWorker()
            self._result_collector.register_worker(self._face_worker)
            self._roi_dispatcher.register_worker(self._face_worker, every_n_frames=2)

            # PoseWorker: YOLOv8s Pose (jeden 3. Frame)
            self._pose_worker = PoseWorker()
            self._result_collector.register_worker(self._pose_worker)
            self._roi_dispatcher.register_worker(self._pose_worker, every_n_frames=3)

            # ReIDWorker: Person ReID (jeden 5. Frame)
            self._reid_worker = ReIDWorker()
            self._result_collector.register_worker(self._reid_worker)
            self._roi_dispatcher.register_worker(self._reid_worker, every_n_frames=5)

            # HandWorker: Hand Landmarks (jeden 4. Frame)
            self._hand_worker = HandWorker()
            self._result_collector.register_worker(self._hand_worker)
            self._roi_dispatcher.register_worker(self._hand_worker, every_n_frames=4)

            # PersonAttrWorker: Kleidung, Alter, Zubehoer (jeden 6. Frame)
            self._person_attr_worker = PersonAttrWorker()
            self._result_collector.register_worker(self._person_attr_worker)
            self._roi_dispatcher.register_worker(self._person_attr_worker, every_n_frames=6)

            # ActivityWorker: r3d_18 Aktivitaetserkennung (Frame-Puffer + Inference alle 30 Frames)
            self._activity_worker = ActivityWorker()
            self._result_collector.register_worker(self._activity_worker)
            # KEIN ROI-Dispatcher — push_frame() direkt in _on_appsink_sample()

            # YOLOWorldWorker: Zero-Shot Objektsuche (alle 60 Frames via ROI-Dispatcher)
            self._yolo_world_worker = YOLOWorldWorker()
            self._result_collector.register_worker(self._yolo_world_worker)
            self._roi_dispatcher.register_worker(self._yolo_world_worker, every_n_frames=60)

            self._result_collector.start_all()
            self._activity_frame_counter = 0
            logger.info("[WORKERS] 7 Worker gestartet: Face(2), Pose(3), Hand(4), ReID(5), PersonAttr(6), Activity(30), YOLOWorld(60)")
        except Exception as e:
            logger.error("[WORKERS] Worker-Start fehlgeschlagen: %s", e)
            self._face_worker = None
            self._roi_dispatcher = None
            self._result_collector = None

        logger.info("TAPPAS Pipeline gestartet (YOLO-Only + FaceWorker)")

    def stop(self):
        """Pipeline sauber beenden. Funktioniert auch wenn _running schon False (z.B. nach Bus-Error)."""
        was_running = self._running
        self._running = False

        if was_running:
            logger.info("Stoppe TAPPAS Pipeline...")
        else:
            logger.info("Raeume tote TAPPAS Pipeline auf...")

        if self._loop and self._loop.is_running():
            self._loop.quit()

        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

        # Worker stoppen (Phase 2)
        if getattr(self, '_result_collector', None):
            try:
                self._result_collector.stop_all()
                logger.info("[WORKERS] Alle Worker gestoppt")
            except Exception as e:
                logger.error("[WORKERS] Stop-Fehler: %s", e)

        self._pipeline = None
        self._loop = None
        self._loop_thread = None
        self._face_worker = None
        self._roi_dispatcher = None
        self._result_collector = None

        # SHM-Frame loeschen damit Panel "Kein Signal" zeigt statt Frozen Frame
        self._cleanup_shm()

        logger.info("TAPPAS Pipeline gestoppt + aufgeraeumt")

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

    def get_npu_sched_mode(self) -> str:
        """Aktueller NPU Scheduler-Modus (jetzt Szenario-Name)."""
        return self._scheduler.get_scenario()

    def get_scenario(self) -> str:
        """Aktuelles Szenario des Perception Routers."""
        return self._scheduler.get_scenario()

    def force_all_active(self, enabled: bool):
        """Scheduler auf ALL_ACTIVE erzwingen (Teach-Modus).

        enabled=True:  SCRFD + ArcFace bleiben immer aktiv
        enabled=False: Scheduler kehrt zu normalem adaptiven Betrieb zurueck
        """
        self._sched_force_all = enabled
        if enabled:
            # Sofort SCRFD Valve oeffnen
            self._apply_scrfd_gate(enabled=True)
            self.scrfd_active = True
            self.arcface_active = True
            logger.info("[NPU-SCHED] Force ALL_ACTIVE fuer Teach-Modus")
        else:
            logger.info("[NPU-SCHED] Force ALL_ACTIVE aufgehoben")

    def _init_scrfd_gate(self) -> bool:
        """GLib-Timeout-Callback: Gate-State nach Pipeline-Start setzen.

        Wird 300ms nach set_state(PLAYING) aufgerufen, damit Pads verhandelt sind.
        Start: SCRFD zu (IDLE Szenario), Scheduler bestimmt spaeter.
        """
        scrfd_needed = self._sched_force_all or self._scheduler.is_model_active("scrfd")
        self._apply_scrfd_gate(enabled=scrfd_needed)
        scenario = self._scheduler.get_scenario()
        logger.info(f"[SCRFD-GATE] Initial-State: {scenario} "
                    f"(Valve {'auf' if scrfd_needed else 'zu'})")
        return False  # Nicht wiederholen

    def _apply_scrfd_gate(self, enabled: bool):
        """SCRFD echtes NPU-Gating via GStreamer valve + input-selector.

        enabled=True  → SCRFD aktiv: Valve auf, Selector auf SCRFD-Pfad (sink_0)
        enabled=False → SCRFD aus:   Valve zu, Selector auf Bypass-Pfad (sink_1)

        Reihenfolge beim Deaktivieren: erst Valve zu, dann Selector umschalten.
        Reihenfolge beim Aktivieren:   erst Selector umschalten, dann Valve auf.
        """
        if self._scrfd_valve is None or self._scrfd_selector is None:
            return

        if enabled:
            # Erst Selector auf SCRFD-Pfad, dann Valve öffnen
            sink0 = self._scrfd_selector.get_static_pad("sink_0")
            if sink0:
                self._scrfd_selector.set_property("active-pad", sink0)
            self._scrfd_valve.set_property("drop", False)
            logger.debug("[SCRFD-GATE] SCRFD aktiviert (Valve auf, sink_0)")
        else:
            # Erst Valve schliessen, dann Selector auf Bypass
            self._scrfd_valve.set_property("drop", True)
            sink1 = self._scrfd_selector.get_static_pad("sink_1")
            if sink1:
                self._scrfd_selector.set_property("active-pad", sink1)
            logger.debug("[SCRFD-GATE] SCRFD deaktiviert (Valve zu, sink_1 Bypass)")

    def _apply_pose_gate(self, enabled: bool):
        """Pose NPU-Gating via GStreamer valve + input-selector."""
        if self._pose_valve is None or self._pose_selector is None:
            return
        if enabled == self._pose_gate_state:
            return  # Keine Aenderung
        self._pose_gate_state = enabled
        if enabled:
            sink0 = self._pose_selector.get_static_pad("sink_0")
            if sink0:
                self._pose_selector.set_property("active-pad", sink0)
            self._pose_valve.set_property("drop", False)
            logger.info("[POSE-GATE] Pose aktiviert (Valve auf, sink_0)")
        else:
            self._pose_valve.set_property("drop", True)
            sink1 = self._pose_selector.get_static_pad("sink_1")
            if sink1:
                self._pose_selector.set_property("active-pad", sink1)
            logger.info("[POSE-GATE] Pose deaktiviert (Valve zu, sink_1 Bypass)")

    def _apply_reid_gate(self, enabled: bool):
        """ReID NPU-Gating via GStreamer valve + input-selector."""
        if self._reid_valve is None or self._reid_selector is None:
            return
        if enabled:
            sink0 = self._reid_selector.get_static_pad("sink_0")
            if sink0:
                self._reid_selector.set_property("active-pad", sink0)
            self._reid_valve.set_property("drop", False)
            logger.debug("[REID-GATE] ReID aktiviert (Valve auf, sink_0)")
        else:
            self._reid_valve.set_property("drop", True)
            sink1 = self._reid_selector.get_static_pad("sink_1")
            if sink1:
                self._reid_selector.set_property("active-pad", sink1)
            logger.debug("[REID-GATE] ReID deaktiviert (Valve zu, sink_1 Bypass)")

    def _apply_hand_gate(self, enabled: bool):
        """Hand-Detection NPU-Gating via GStreamer valve + input-selector."""
        if self._hand_valve is None or self._hand_selector is None:
            return
        if enabled:
            sink0 = self._hand_selector.get_static_pad("sink_0")
            if sink0:
                self._hand_selector.set_property("active-pad", sink0)
            self._hand_valve.set_property("drop", False)
            logger.debug("[HAND-GATE] Hand aktiviert (Valve auf, sink_0)")
        else:
            self._hand_valve.set_property("drop", True)
            sink1 = self._hand_selector.get_static_pad("sink_1")
            if sink1:
                self._hand_selector.set_property("active-pad", sink1)
            logger.debug("[HAND-GATE] Hand deaktiviert (Valve zu, sink_1 Bypass)")

    def _reid_landmarks_strip_probe(self, pad, info, user_data):
        """GStreamer Pad-Probe: HAILO_LANDMARKS aus Person-Detections entfernen.

        libre_id.so::create_crops crasht mit cv2::resize wenn Pose-Landmarks
        in den Detections sind. Diese Probe entfernt sie VOR dem hailocropper.
        Da GStreamer tee den Buffer kopiert, betrifft das nur den ReID-Branch.
        """
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buf)
            if roi:
                for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
                    for lm in list(det.get_objects_typed(hailo.HAILO_LANDMARKS)):
                        det.remove_object(lm)
        except Exception:
            pass
        return Gst.PadProbeReturn.OK

    def _apply_ocr_gate(self, enabled: bool):
        """OCR NPU-Gating via GStreamer valve + input-selector."""
        if self._ocr_valve is None or self._ocr_selector is None:
            return
        if enabled:
            sink0 = self._ocr_selector.get_static_pad("sink_0")
            if sink0:
                self._ocr_selector.set_property("active-pad", sink0)
            self._ocr_valve.set_property("drop", False)
            self._ocr_enabled = True
            logger.info("[OCR-GATE] OCR aktiviert (Valve auf, sink_0)")
        else:
            self._ocr_valve.set_property("drop", True)
            sink1 = self._ocr_selector.get_static_pad("sink_1")
            if sink1:
                self._ocr_selector.set_property("active-pad", sink1)
            self._ocr_enabled = False
            logger.info("[OCR-GATE] OCR deaktiviert (Valve zu, sink_1 Bypass)")

    def set_ocr_enabled(self, enabled: bool):
        """Oeffentliche API: OCR ein-/ausschalten zur Laufzeit."""
        self._apply_ocr_gate(enabled)

    def get_ocr_texts(self) -> List[str]:
        """Letzte erkannte OCR-Texte zurueckgeben."""
        return list(self._last_ocr_texts)

    def _update_npu_scheduler(self, has_person: bool, has_face: bool):
        """Scheduler-Modus basierend auf aktuellen Detections aktualisieren."""
        now = time.time()
        if has_person:
            self._sched_person_last_seen = now
        if has_face:
            self._sched_face_last_seen = now

        person_recent = (now - self._sched_person_last_seen) < SCHED_COOLDOWN_DOWN
        face_recent = (now - self._sched_face_last_seen) < SCHED_COOLDOWN_DOWN

        # Teach-Modus erzwingt ALL_ACTIVE (SCRFD + ArcFace immer an)
        if self._sched_force_all:
            new_mode = SCHED_ALL_ACTIVE
        elif face_recent:
            new_mode = SCHED_ALL_ACTIVE
        elif person_recent:
            new_mode = SCHED_YOLO_SCRFD
        else:
            new_mode = SCHED_YOLO_ONLY

        mode_changed = False
        scrfd_needed = False
        with self._sched_lock:
            if self._sched_mode != new_mode:
                self._sched_mode = new_mode
                mode_changed = True
                scrfd_needed = (new_mode in (SCHED_YOLO_SCRFD, SCHED_ALL_ACTIVE))

                # Model-Active-Flags fuer GUI-Checkboxen + Status-JSON
                self.scrfd_active = scrfd_needed
                self.arcface_active = (new_mode == SCHED_ALL_ACTIVE)

                logger.info(f"[NPU-SCHED] → {new_mode} "
                            f"(scrfd={'AKTIV (Valve auf)' if self.scrfd_active else 'AUS (Valve zu)'}, "
                            f"arcface={'aktiv' if self.arcface_active else 'nativ gated'})")

        # Valve-Gating ausserhalb des Locks (GStreamer-Calls nicht unter Lock)
        if mode_changed:
            self._apply_scrfd_gate(enabled=scrfd_needed)

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

    # =====================================================================
    # Live-Enrollment
    # =====================================================================

    def start_enrollment(self, name: str, n: int = 20):
        """Live-Enrollment starten: sammelt Embeddings aus GStreamer-Stream + Teachen/Snapshots.

        Ablauf:
        1. Sofort aktiv setzen (Panel sieht "laeuft")
        2. Background-Thread: Teachen-JSONs + Batch-Pipeline + Live-Stream
        3. Diversity-Filter waehlt beste 20 aus allen Kandidaten

        Embeddings kommen aus dem GLEICHEN Pipeline-Pfad wie die Erkennung
        (libvms_face_align.so → hailonet arcface → postprocess).

        Args:
            name: Personenname (z.B. "Markus")
            n: Maximale Anzahl Kandidaten zum Sammeln (beste 20 werden gespeichert)
        """
        with self._enroll_lock:
            if self._enroll_active:
                logger.warning("[ENROLLMENT] Laeuft bereits!")
                return
            self._enroll_name = name.lower()
            self._enroll_target = max(n, 60)
            self._enroll_candidates = []
            self._enroll_active = True
            self._enroll_done = False

        # Batch-Verarbeitung im Background-Thread (blockiert nicht IPC)
        threading.Thread(
            target=self._enrollment_batch_worker,
            args=(name,),
            daemon=True,
            name="EnrollmentBatch",
        ).start()

        logger.info(f"[ENROLLMENT] Gestartet fuer '{name}': "
                     f"Batch-Worker + Live-Stream (Score >{self._enroll_min_score})")

    def _enrollment_batch_worker(self, name: str):
        """Background-Worker: Teachen-JSONs Embeddings laden + Status loggen.

        Laeuft parallel zur Live-Sammlung aus _on_buffer.
        Teachen-JSON Embeddings (von frueheren Enrollments) werden thread-safe
        zu _enroll_candidates hinzugefuegt.

        Batch-Bildverarbeitung via separate GStreamer-Pipeline ist NICHT moeglich
        (Hailo-10H erlaubt nur EIN VDevice — Live-Pipeline belegt es).
        """
        try:
            # 1. Offline-Embeddings aus Teachen-JSONs laden
            offline_candidates = self._load_teachen_embeddings(name)
            if offline_candidates:
                with self._enroll_lock:
                    self._enroll_candidates.extend(offline_candidates)
                logger.info(f"[ENROLLMENT] {len(offline_candidates)} Teachen-JSON Embeddings geladen")

            # 2. Status der Bilder ohne Embeddings loggen
            self._log_enrollment_image_status(name)

            # Pruefen ob Target schon erreicht (Offline + Live zusammen)
            with self._enroll_lock:
                count = len(self._enroll_candidates)
                target = self._enroll_target

            if count >= target:
                self._finish_enrollment()

        except Exception as e:
            logger.error(f"[ENROLLMENT] Batch-Worker Fehler: {e}")
            import traceback
            traceback.print_exc()

    def _load_teachen_embeddings(self, name: str) -> list:
        """Embeddings aus Teachen-JSONs laden (von frueheren Enrollments gespeichert).

        Durchsucht /mnt/moloch-data/Teachen/<YYYY-MM-DD>/*.json nach Eintraegen
        mit passender Person und gespeichertem 'embedding' Feld.

        Returns: Liste von (score, embedding) Tupeln
        """
        import json
        import glob

        teachen_dir = "/mnt/moloch-data/Teachen"
        snap_dir = os.path.expanduser("~/moloch/snapshots")
        candidates = []
        name_lower = name.lower()
        teachen_files = 0
        snap_files = 0

        # Teachen-Ordner: Alle Tage durchsuchen
        if os.path.isdir(teachen_dir):
            for day_dir in os.listdir(teachen_dir):
                day_path = os.path.join(teachen_dir, day_dir)
                if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                    continue
                for fname in os.listdir(day_path):
                    if not fname.lower().endswith(".json"):
                        continue
                    # Filename-Match: Name muss im Dateinamen vorkommen
                    if name_lower not in fname.lower():
                        continue
                    teachen_files += 1
                    json_path = os.path.join(day_path, fname)
                    try:
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                        emb_list = meta.get("embedding")
                        if emb_list and isinstance(emb_list, list):
                            emb = np.array(emb_list, dtype=np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                            score = meta.get("confidence", 0.5)
                            candidates.append((score, emb))
                    except Exception:
                        pass

        logger.info(f"[ENROLLMENT] Teachen: {teachen_files} JSONs fuer '{name}', "
                     f"davon {len(candidates)} mit Embedding. "
                     f"Snapshots: {snap_files} Bilder")
        return candidates

    def _save_embeddings_to_teachen(self, name: str, selected: list):
        """Nach Enrollment: GStreamer-Embeddings in Teachen-JSONs zurueckschreiben.

        Damit koennen kuenftige Enrollments diese Embeddings wiederverwenden.
        Schreibt NUR in JSONs die zum Namen passen UND noch kein Embedding haben.
        """
        import json

        teachen_dir = "/mnt/moloch-data/Teachen"
        name_lower = name.lower()
        written = 0

        if not os.path.isdir(teachen_dir):
            return

        # Bestes Embedding (hoechster Score) zum Zurueckschreiben
        if not selected:
            return
        best_emb = selected[0][1]  # Score-sortiert, erstes = bestes

        for day_dir in os.listdir(teachen_dir):
            day_path = os.path.join(teachen_dir, day_dir)
            if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                continue
            for fname in os.listdir(day_path):
                if not fname.lower().endswith(".json"):
                    continue
                if name_lower not in fname.lower():
                    continue
                json_path = os.path.join(day_path, fname)
                try:
                    with open(json_path, 'r') as f:
                        meta = json.load(f)
                    if meta.get("embedding"):
                        continue  # Schon vorhanden, nicht ueberschreiben
                    meta["embedding"] = best_emb.tolist()
                    meta["embedding_learned"] = True
                    with open(json_path, 'w') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    written += 1
                except Exception:
                    pass

        if written:
            logger.info(f"[ENROLLMENT] {written} Teachen-JSONs mit Embedding aktualisiert")

    def _log_enrollment_image_status(self, name: str):
        """Bilder aus Teachen + Snapshots zaehlen und Status loggen.

        HINWEIS: Bilder ohne Embeddings koennen NICHT als separate Batch-Pipeline
        verarbeitet werden, da der Hailo-10H nur EIN VDevice erlaubt (Live-Pipeline
        belegt es). Embeddings fuer neue Bilder werden beim naechsten Enrollment
        ueber den Live-Stream + _save_embeddings_to_teachen() in die JSONs geschrieben.
        """
        import json as _json
        name_lower = name.lower()
        teachen_without = 0
        snap_count = 0

        teachen_dir = "/mnt/moloch-data/Teachen"
        if os.path.isdir(teachen_dir):
            for day_dir in os.listdir(teachen_dir):
                day_path = os.path.join(teachen_dir, day_dir)
                if not os.path.isdir(day_path) or not day_dir.startswith("20"):
                    continue
                for fname in os.listdir(day_path):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    if name_lower not in fname.lower():
                        continue
                    json_path = os.path.join(day_path, fname.rsplit(".", 1)[0] + ".json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                meta = _json.load(f)
                            if meta.get("embedding"):
                                continue
                        except Exception:
                            pass
                    teachen_without += 1

        snap_dir = os.path.expanduser("~/moloch/snapshots")
        if os.path.isdir(snap_dir):
            for fname in os.listdir(snap_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    snap_count += 1

        if teachen_without > 0 or snap_count > 0:
            logger.info(f"[ENROLLMENT] {teachen_without} Teachen-Bilder ohne Embedding, "
                         f"{snap_count} Snapshots — Batch-NPU nicht moeglich "
                         f"(Single-VDevice), Embeddings werden ueber Live-Stream gesammelt")

    def get_enrollment_status(self) -> dict:
        """Enrollment-Status fuer IPC/Panel."""
        with self._enroll_lock:
            return {
                "active": self._enroll_active,
                "done": self._enroll_done,
                "name": self._enroll_name,
                "collected": len(self._enroll_candidates),
                "target": self._enroll_target,
            }

    def _collect_enrollment_embedding(self, embedding: np.ndarray, score: float):
        """Embedding-Kandidat fuer Enrollment sammeln (aus _on_buffer Callback).

        Filtert nach Score und Limit, ruft _finish_enrollment wenn genug da sind.
        """
        with self._enroll_lock:
            if not self._enroll_active:
                return
            if score < self._enroll_min_score:
                return
            # Embedding kopieren (Buffer wird wiederverwendet)
            emb = embedding.copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self._enroll_candidates.append((score, emb))
            count = len(self._enroll_candidates)
            target = self._enroll_target

        logger.info(f"[ENROLLMENT] Embedding #{count} gesammelt (score={score:.3f})")

        # Genug Kandidaten? → Enrollment abschliessen
        if count >= target:
            self._finish_enrollment()

    def _finish_enrollment(self):
        """Enrollment abschliessen: Diversitaets-Selektion + Face-DB speichern.

        Kandidaten koennen aus 2 Quellen stammen:
        1. Teachen-JSONs (Offline, von frueheren Enrollments via GStreamer)
        2. Live-Stream (GStreamer _on_buffer Callback)
        Diversity-Filter waehlt die besten 20 aus allen Kandidaten.
        """
        import json

        with self._enroll_lock:
            if not self._enroll_active:
                return
            self._enroll_active = False
            self._enroll_done = True
            name = self._enroll_name
            candidates = list(self._enroll_candidates)
            diversity_thresh = self._enroll_diversity

        if not candidates:
            logger.warning("[ENROLLMENT] Keine Embeddings gesammelt!")
            return

        n_total = len(candidates)

        # Nach Score sortieren (hoechster zuerst)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy diverse selection
        selected = []
        for score, emb in candidates:
            if len(selected) >= 20:
                break
            is_diverse = all(
                float(np.dot(emb, sel_emb)) < diversity_thresh
                for _, sel_emb in selected
            )
            if is_diverse:
                selected.append((score, emb))

        # Falls nicht genug diverse: Rest nach Score auffuellen
        if len(selected) < 20:
            selected_embs = {id(emb) for _, emb in selected}
            for score, emb in candidates:
                if len(selected) >= 20:
                    break
                if id(emb) not in selected_embs:
                    selected.append((score, emb))

        logger.info(f"[ENROLLMENT] {n_total} Kandidaten → "
                     f"{len(selected)} diverse Embeddings ausgewaehlt fuer '{name}'")

        # Face-DB laden, alte Eintraege entfernen, neue speichern
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "face_embeddings.json"
        )
        db = {}
        if os.path.exists(embeddings_path):
            try:
                with open(embeddings_path, 'r') as f:
                    db = json.load(f)
            except Exception:
                pass

        # Alte Eintraege fuer diesen Namen entfernen
        old_keys = [k for k in db if k.split('#')[0].lower() == name]
        for k in old_keys:
            del db[k]
        logger.info(f"[ENROLLMENT] {len(old_keys)} alte '{name}' Eintraege entfernt")

        # Neues Haupt-Embedding + Varianten speichern
        db[name] = selected[0][1].tolist()
        for i, (score, emb) in enumerate(selected[1:]):
            db[f"{name}#snap_{i}"] = emb.tolist()

        # Atomar speichern
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        tmp = embeddings_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(db, f, indent=1, ensure_ascii=False)
        os.replace(tmp, embeddings_path)

        logger.info(f"[ENROLLMENT] Face-DB gespeichert: {len(selected)} Embeddings fuer '{name}'")

        # Embeddings in Teachen-JSONs zurueckschreiben (fuer kuenftige Enrollments)
        self._save_embeddings_to_teachen(name, selected)

        # Face-DB neu laden (Best-Match)
        self.reload_face_db()

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

        Gruppiert nach Person (Name vor '#') und bildet Durchschnitt.
        z.B. "Markus", "Markus#snap_1", "Markus#train_42_0" → ein Embedding fuer "Markus".
        Returns: {person_name: np.array} oder {} bei Fehler.
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
            # Gruppiere nach Person (Name vor '#')
            groups = {}
            for key, emb_list in raw.items():
                person = key.split('#')[0].lower()
                emb = np.array(emb_list, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                if person not in groups:
                    groups[person] = []
                groups[person].append(emb)
            # Alle Embeddings pro Person speichern (Best-Match statt Mean)
            db = {}
            for person, embs in groups.items():
                db[person] = embs  # Liste von normalisierten Embeddings
            total_embs = sum(len(e) for e in groups.values())
            logger.info(f"Face-DB geladen: {len(db)} Personen aus {total_embs} Embeddings ({embeddings_path})")
            return db
        except Exception as e:
            logger.error(f"Face-DB laden fehlgeschlagen: {e}")
            return {}

    # =====================================================================
    # GStreamer Pipeline String
    # =====================================================================

    def _build_pipeline_string(self) -> str:
        """YOLO-Only Pipeline — alle anderen Modelle laufen als HailoRT-Direct Worker.

        Phase 2: Radikal vereinfacht. SCRFD/ArcFace/Pose/ReID/Hand/OCR/FaceAttr
        sind aus der GStreamer-Kette entfernt und laufen in vision_workers.py Threads.
        """

        # --- Source: RTSP → H264 depay → decode → scale → RGB ---
        source = (
            f'rtspsrc location="{self._rtsp_url}" name=source latency=300 protocols=tcp '
            f'retry=5 timeout=5000000 tcp-timeout=5000000 ! '
            f'rtph264depay name=source_depay ! '
            f'queue name=source_queue_decode leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 ! '
            f'avdec_h264 name=source_decode ! '
            f'queue name=source_scale_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=source_videoscale n-threads=2 ! '
            f'queue name=source_convert_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=3 name=source_convert qos=false ! '
            f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width={self._width}, height={self._height} '
        )

        # --- Stage 1: YOLO Person Detection ---
        yolo_inner = (
            f'queue name=yolo_scale_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=yolo_videoscale n-threads=2 qos=false ! '
            f'queue name=yolo_convert_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'video/x-raw, pixel-aspect-ratio=1/1 ! '
            f'videoconvert name=yolo_videoconvert n-threads=2 ! '
            f'queue name=yolo_hailonet_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'hailonet name=yolo_hailonet hef-path={YOLO_HEF} batch-size=1 '
            f'vdevice-group-id={VDEVICE_GROUP_ID} '
            f'nms-score-threshold=0.3 nms-iou-threshold=0.45 '
            f'output-format-type=HAILO_FORMAT_TYPE_FLOAT32 '
            f'force-writable=true ! '
            f'queue name=yolo_hailofilter_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'hailofilter name=yolo_hailofilter so-path={YOLO_POSTPROCESS_SO} '
            f'function-name={YOLO_POSTPROCESS_FUNC} qos=false ! '
            f'queue name=yolo_output_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 '
        )

        yolo_wrapper = (
            f'queue name=yolo_wrapper_input_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! '
            f'hailocropper name=yolo_wrapper_crop so-path={WHOLE_BUFFER_SO} function-name=create_crops '
            f'use-letterbox=true resize-method=inter-area internal-offset=false '
            f'hailoaggregator name=yolo_wrapper_agg '
            f'yolo_wrapper_crop. ! queue name=yolo_wrapper_bypass_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! yolo_wrapper_agg.sink_0 '
            f'yolo_wrapper_crop. ! {yolo_inner} ! yolo_wrapper_agg.sink_1 '
            f'yolo_wrapper_agg. ! queue name=yolo_wrapper_output_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 '
        )

        # --- Tracker (haelt Person-IDs ueber Frames stabil) ---
        tracker = (
            f'hailotracker name=hailo_face_tracker class-id=-1 '
            f'kalman-dist-thr=0.7 iou-thr=0.8 init-iou-thr=0.9 '
            f'keep-new-frames=2 keep-tracked-frames=6 keep-lost-frames=8 '
            f'keep-past-metadata=true qos=false ! '
            f'queue name=tracker_q leaky=no max-size-buffers=8 max-size-bytes=0 max-size-time=0 '
        )

        # --- Callback + appsink (kein hailooverlay — Panel zeichnet BBoxen selbst) ---
        callback_and_sink = (
            f'queue name=cb_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
            f'identity name=identity_callback ! '
            f'queue name=sink_convert_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=2 qos=false ! '
            f'video/x-raw, format=RGB ! '
            f'appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false'
        )

        # --- YOLO-Only Topologie (Phase 2: alle anderen Modelle als HailoRT-Direct Worker) ---
        # source → yolo_wrapper → tracker → identity_callback → appsink
        # SCRFD/ArcFace/Pose/ReID/Hand/OCR/FaceAttr → FaceWorker + zukuenftige Worker Threads
        return (
            f'{source} ! '
            f'{yolo_wrapper} ! '
            f'{tracker} ! '
            f'{callback_and_sink}'
        )

    # =====================================================================
    # GStreamer Callbacks
    # =====================================================================

    def _on_face_attr_buffer(self, pad, info, user_data):
        """Pad-Probe nach fattr_hailofilter — liest HAILO_CLASSIFICATION (gender/smiling)."""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            classifications = roi.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            if not classifications:
                return Gst.PadProbeReturn.OK

            gender = None
            smiling = False
            self._fattr_probe_count = getattr(self, '_fattr_probe_count', 0) + 1

            for c in classifications:
                label = c.get_label()
                if label == "Male":
                    gender = "M"
                elif label == "Female":
                    gender = "F"
                elif label == "Smiling":
                    smiling = True

            if gender is not None:
                with self._face_attr_lock:
                    self._face_attr_cache = {"gender": gender, "smiling": smiling}

            if self._fattr_probe_count % 100 == 1:
                cls_info = [(c.get_label(), round(c.get_confidence(), 2))
                            for c in classifications]
                logger.info(f"[FACE-ATTR] #{self._fattr_probe_count}: gender={gender} "
                            f"smiling={smiling} attrs={cls_info[:6]}")

        except Exception as e:
            self._fattr_err_count = getattr(self, '_fattr_err_count', 0) + 1
            if self._fattr_err_count % 100 == 1:
                logger.error(f"[FACE-ATTR] Probe-Fehler: {e}")
        return Gst.PadProbeReturn.OK

    def _on_pre_overlay(self, pad, info, user_data):
        """VOR hailooverlay: Alle Sub-Objekte strippen die SEGV verursachen.

        hailooverlay (C-Code) crasht mit SEGV wenn es auf HAILO_MATRIX,
        HAILO_LANDMARKS oder HAILO_CLASSIFICATION Daten trifft die noch
        vom NPU beschrieben werden (Race Condition).
        Loesung: Alles ausser BBox+Label+Confidence entfernen.
        _on_buffer (identity callback) hat die Daten bereits vorher gelesen.
        """
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
                # Pose-Detections komplett entfernen (BBox-Zugriff = SEGV)
                if det.get_objects_typed(hailo.HAILO_LANDMARKS):
                    roi.remove_object(det)
                    continue
                # Alle Sub-Objekte von normalen Detections strippen
                for sub in list(det.get_objects_typed(hailo.HAILO_MATRIX)):
                    det.remove_object(sub)
                for sub in list(det.get_objects_typed(hailo.HAILO_UNIQUE_ID)):
                    det.remove_object(sub)
                for sub in list(det.get_objects_typed(hailo.HAILO_CLASSIFICATION)):
                    det.remove_object(sub)
        except Exception as e:
            self._overlay_err = getattr(self, '_overlay_err', 0) + 1
            if self._overlay_err % 100 == 1:
                logger.error(f"[OVERLAY-PROBE] Fehler: {e}")
        return Gst.PadProbeReturn.OK

    def _on_buffer(self, pad, info, user_data):
        """Pad-Probe auf identity element — extrahiert YOLO-Detections + merged FaceWorker.

        Phase 2: GStreamer liefert nur YOLO Person-Detections.
        Face/Pose/ReID kommen von HailoRT-Direct Workern.
        """
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        # Frame-ID: monoton steigend
        self._frame_id = getattr(self, '_frame_id', 0) + 1
        frame_id = self._frame_id

        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        detections = []
        persons = []

        # --- YOLO-Only: nur Person-Detections extrahieren ---
        for det in hailo_detections:
            label = det.get_label()
            conf = det.get_confidence()

            if label not in YOLO_ALLOWED_CLASSES:
                continue
            if conf < self.yolo_conf_val:
                continue

            bbox = det.get_bbox()
            x1 = max(0.0, min(1.0, bbox.xmin()))
            y1 = max(0.0, min(1.0, bbox.ymin()))
            x2 = max(0.0, min(1.0, bbox.xmax()))
            y2 = max(0.0, min(1.0, bbox.ymax()))

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

            persons.append(entry)
            detections.append(entry)

        # --- FaceWorker-Ergebnisse einmergen (HailoRT-Direct) ---
        faces = []
        best_face_conf = 0.0
        best_face_bbox = None
        face_id = None
        face_similarity = 0.0

        if getattr(self, '_result_collector', None):
            face_result = self._result_collector.get_latest("FaceWorker")
            if face_result and face_result.data.get("faces"):
                for face in face_result.data["faces"]:
                    face_entry = {
                        "class": "face",
                        "bbox": face["bbox"],
                        "confidence": face["confidence"],
                        "face_id": face.get("face_id"),
                        "face_similarity": face.get("similarity", 0),
                        "embedding": face.get("embedding"),
                        "gender": face.get("gender"),
                        "smiling": face.get("emotion") == "happy" if face.get("emotion") else None,
                        "track_id": None,
                    }
                    faces.append(face_entry)
                    detections.append(face_entry)

                    if face["confidence"] > best_face_conf:
                        best_face_conf = face["confidence"]
                        best_face_bbox = tuple(face["bbox"])

                # Bestes Face-Match
                for f in faces:
                    sim = f.get("face_similarity", 0)
                    if sim > face_similarity:
                        face_id = f.get("face_id")
                        face_similarity = sim

        # --- PerceptionMemory: Temporale Wahrnehmung aktualisieren ---
        max_person_height = 0.0
        if persons:
            max_person_height = max((p["bbox"][3] - p["bbox"][1]) for p in persons)

        best_embedding = None
        for f in faces:
            if f.get("embedding") is not None:
                best_embedding = f["embedding"]
                break

        perception_mem = get_perception_memory()
        perception_mem.tick(
            detections=detections,
            face_id=face_id,
            face_similarity=face_similarity,
            face_embedding=best_embedding,
            face_bbox=best_face_bbox,
            person_count=len(persons),
            face_detected=len(faces) > 0,
            bbox_height_pct=max_person_height,
        )

        # --- Perception Router: Geglaettete Werte fuer Scheduler ---
        smoothed = perception_mem.get_smoothed_scheduler_input()
        scenario = self._scheduler.tick(
            person_count=smoothed["person_count"],
            face_detected=smoothed["face_detected"],
            bbox_height_pct=smoothed["bbox_height_pct"],
        )

        # Model-Active-Flags: alle Worker-Modelle
        self.scrfd_active = getattr(self, '_face_worker', None) is not None
        self.arcface_active = self.scrfd_active
        self.pose_active = getattr(self, '_pose_worker', None) is not None
        self.reid_active = getattr(self, '_reid_worker', None) is not None
        self.hand_active = getattr(self, '_hand_worker', None) is not None

        # PerceptionFrame bauen
        pf = self._build_pframe(persons, faces, best_face_conf,
                                best_face_bbox, face_id, face_similarity)

        # Pose/Hand-Ergebnisse in PerceptionFrame einbauen
        if getattr(self, '_result_collector', None):
            pose_result = self._result_collector.get_latest("PoseWorker")
            if pose_result and pose_result.data.get("poses"):
                pf.pose_count = pose_result.data["pose_count"]

            hand_result = self._result_collector.get_latest("HandWorker")
            if hand_result and hand_result.data.get("hand_detected"):
                pf.hand_detected = True

            attr_result = self._result_collector.get_latest("PersonAttrWorker")
            if attr_result and attr_result.data.get("persons"):
                # Attribute der ersten erkannten Person nehmen
                pf.person_attributes = attr_result.data["persons"][0].get("attribute", [])

            activity_result = self._result_collector.get_latest("ActivityWorker")
            if activity_result and activity_result.success and activity_result.data.get("activity"):
                pf.person_action = activity_result.data["activity"]

            world_result = self._result_collector.get_latest("YOLOWorldWorker")
            if world_result and world_result.success and world_result.data.get("detections"):
                pf.objects = [
                    {"class": d["label"], "confidence": round(d["score"], 3), "bbox": d["bbox"]}
                    for d in world_result.data["detections"]
                    if d.get("label") != "person"  # Personen bereits via YOLO
                ]

        # Thread-safe update
        with self._lock:
            self._detections = detections
            self._current_pframe = pf

        # --- Perception Events publishen ---
        self._publish_perception_events(persons, faces, face_id, face_similarity)

        # FPS Tracking
        self._update_fps()

        return Gst.PadProbeReturn.OK

    def _publish_perception_events(self, persons: list, faces: list,
                                       face_id: Optional[str], face_similarity: float):
        """Perception-Events auf Event Bus publishen (fuer Action Bridge FSM).

        Events:
          perception.person_detected  — Person erkannt (YOLO)
          perception.face_confirmed   — Gesicht erkannt (SCRFD)
          perception.owner_detected   — Owner erkannt (ArcFace Match)
          perception.target_lost      — Keine Person mehr im Frame
        """
        has_person = len(persons) > 0 or len(faces) > 0

        if has_person:
            # Beste Person-BBox fuer Tracking
            best_bbox = [0, 0, 0, 0]
            best_conf = 0.0
            if persons:
                best = max(persons, key=lambda p: p["confidence"])
                best_bbox = best["bbox"]
                best_conf = best["confidence"]

            self._event_bus.publish(
                event_type="perception.person_detected",
                payload={"confidence": best_conf, "bbox": best_bbox,
                         "count": len(persons)},
                source="tappas_pipeline",
                priority=PRIO_PERCEPTION,
            )

            # Face confirmed (SCRFD hat Gesicht gefunden)
            if faces:
                best_face = max(faces, key=lambda f: f["confidence"])
                self._event_bus.publish(
                    event_type="perception.face_confirmed",
                    payload={"confidence": best_face["confidence"],
                             "bbox": best_face["bbox"],
                             "similarity": face_similarity or 0.0},
                    source="tappas_pipeline",
                    priority=PRIO_PERCEPTION,
                )

            # Owner detected (ArcFace Match ueber Threshold)
            if face_id and face_similarity >= self.arcface_thresh_val:
                self._event_bus.publish(
                    event_type="perception.owner_detected",
                    payload={"name": face_id, "similarity": face_similarity},
                    source="tappas_pipeline",
                    priority=PRIO_PERCEPTION,
                )

        # Target lost: War Person da, jetzt nicht mehr
        if self._last_person_state and not has_person:
            self._event_bus.publish(
                event_type="perception.target_lost",
                payload={"reason": "no_detection"},
                source="tappas_pipeline",
                priority=PRIO_PERCEPTION,
            )

        self._last_person_state = has_person

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

        # Resolution-Wechsel erkennen (Kamera-ST oder RTSP-Reconnect)
        prev_res = getattr(self, '_appsink_last_res', None)
        cur_res = (width, height)
        if prev_res != cur_res:
            logger.warning(f"[APPSINK] Resolution-Wechsel: {prev_res} → {cur_res}")
            self._appsink_last_res = cur_res

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK
        try:
            data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        finally:
            buf.unmap(mapinfo)

        expected = width * height * 3
        if len(data) != expected:
            logger.warning(f"[APPSINK] Size-Mismatch: expected={expected} got={len(data)} res={width}x{height}")
            return Gst.FlowReturn.OK

        frame = data.reshape(height, width, 3)

        # Low Light Enhancement: wenn dunkel → NPU-Aufhellung (zero_dce)
        try:
            from core.perception.low_light_processor import get_low_light
            frame = get_low_light().maybe_enhance(frame)
        except Exception:
            pass

        # Thread-safe: Frame speichern
        with self._lock:
            self._annotated_frame = frame

        # SHM IPC: Frame fuer Panel Preview
        self._write_shm_frame(frame)

        # Phase 2: Frame an ROIDispatcher weiterleiten (fuer FaceWorker etc.)
        # Frame ist RGB (von GStreamer videoconvert format=RGB)
        if getattr(self, '_roi_dispatcher', None):
            try:
                with self._lock:
                    yolo_dets = [d for d in self._detections if d.get("class") == "person"]
                self._roi_dispatcher.dispatch(frame, yolo_dets, getattr(self, '_frame_id', 0))
            except Exception as e:
                logger.debug("[DISPATCH] Fehler: %s", e)

        # ActivityWorker: jeden Frame in Puffer, Inference alle 30 Frames
        if getattr(self, '_activity_worker', None):
            try:
                self._activity_worker.push_frame(frame)
                self._activity_frame_counter = getattr(self, '_activity_frame_counter', 0) + 1
                if self._activity_frame_counter >= 30:
                    self._activity_frame_counter = 0
                    from core.perception.vision_workers import WorkItem
                    self._activity_worker.submit(WorkItem(
                        frame=None,
                        frame_id=getattr(self, '_frame_id', 0),
                        timestamp=0.0,
                    ))
            except Exception as e:
                logger.debug("[ACTIVITY] Fehler: %s", e)

        return Gst.FlowReturn.OK

    def _draw_thick_overlay(self, frame):
        """Dicke BBoxen + Landmarks auf Frame zeichnen (Debug-Visualisierung).

        Liest self._detections (thread-safe) und zeichnet:
        - Person: gruene BBox, dick
        - Face: cyan BBox + Face-ID Text
        - SCRFD-Landmarks: rote Punkte (Augen, Nase, Mundwinkel)
        """
        try:
            h, w = frame.shape[:2]
            out = frame.copy()
            with self._lock:
                dets = list(self._detections)

            for d in dets:
                x1 = int(d["bbox"][0] * w)
                y1 = int(d["bbox"][1] * h)
                x2 = int(d["bbox"][2] * w)
                y2 = int(d["bbox"][3] * h)
                label = d["class"]
                conf = d.get("confidence", 0)

                if label == "person":
                    # Person: gruene BBox
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(out, f"person {conf:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif label == "face":
                    # Face: cyan BBox
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    fid = d.get("face_id", "?")
                    sim = d.get("face_similarity", 0)
                    cv2.putText(out, f"face {conf:.0%} {fid}({sim:.2f})", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            return out
        except Exception:
            return frame

    def _on_bus_message(self, bus, message):
        """GStreamer Bus Messages verarbeiten. Raeumt auf bei ERROR/EOS."""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[BUS] GStreamer Fehler: {err}")
            logger.error(f"[BUS] Debug: {debug}")
            self._running = False
            # Pipeline auf NULL setzen damit NPU-Resources freigegeben werden
            if self._pipeline:
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            # SHM loeschen damit Panel "Kein Signal" zeigt
            self._cleanup_shm()
            if self._loop and self._loop.is_running():
                self._loop.quit()
        elif t == Gst.MessageType.EOS:
            logger.warning("[BUS] End-of-Stream — Pipeline beendet")
            self._running = False
            if self._pipeline:
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            self._cleanup_shm()
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
    # Passives Continuous-Learning
    # =====================================================================

    def _continuous_learn(self, embedding: np.ndarray, matched_name: str,
                          matched_sim: float, scrfd_conf: float):
        """Passiv neues Embedding speichern wenn neuer Winkel erkannt.

        Bedingungen (ALLE muessen erfuellt sein):
        1. Feature aktiviert (_cl_enabled)
        2. Person erkannt (matched_name != None)
        3. Similarity im Fenster: 0.55 <= sim <= 0.92
           (unter 0.55 = unsicher, ueber 0.92 = bereits bekannter Winkel)
        4. SCRFD Confidence >= 0.70 (Gesicht gut sichtbar)
        5. Mindestens 30s seit letzter Speicherung
        6. Neues Embedding ist divers (Cosine-Sim < 0.80 zu allen bestehenden)
        7. Max 50 Embeddings pro Person (danach aelteste ersetzen)
        """
        import json as _json

        if not self._cl_enabled:
            return
        if not matched_name:
            return
        if matched_sim < self._cl_min_sim or matched_sim > self._cl_max_sim:
            return
        if scrfd_conf < self._cl_min_scrfd:
            return

        now = time.time()
        if (now - self._cl_last_save) < self._cl_interval_sec:
            return

        # Embedding normalisieren
        norm = np.linalg.norm(embedding)
        if norm > 0:
            emb_norm = embedding / norm
        else:
            return

        # Diversitaets-Check gegen bestehende DB
        with self._face_db_lock:
            existing = self._face_db.get(matched_name, [])
            for db_emb in existing:
                sim_to_existing = float(np.dot(emb_norm, db_emb))
                if sim_to_existing >= self._cl_diversity_thresh:
                    # Zu aehnlich zu bestehendem Embedding → kein neuer Winkel
                    return

        # === Neuer Winkel erkannt — speichern! ===
        self._cl_last_save = now

        try:
            embeddings_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "face_embeddings.json"
            )
            db = {}
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'r') as f:
                    db = _json.load(f)

            # Zaehle bestehende Eintraege fuer diese Person
            person_keys = [k for k in db if k.split('#')[0].lower() == matched_name]
            n_existing = len(person_keys)

            if n_existing >= self._cl_max_embeddings:
                # Aeltestes Snap-Embedding ersetzen (nicht das Haupt-Embedding)
                snap_keys = sorted([k for k in person_keys if '#' in k])
                if snap_keys:
                    del db[snap_keys[0]]
                    logger.info(f"[CL] Max {self._cl_max_embeddings} erreicht, "
                                f"aeltestes entfernt: {snap_keys[0]}")

            # Neues Embedding hinzufuegen
            cl_key = f"{matched_name}#cl_{int(now)}"
            db[cl_key] = emb_norm.tolist()

            # Atomar speichern
            tmp = embeddings_path + ".tmp"
            with open(tmp, 'w') as f:
                _json.dump(db, f, indent=1, ensure_ascii=False)
            os.replace(tmp, embeddings_path)

            # In-Memory DB aktualisieren (ohne Disk-Reload)
            with self._face_db_lock:
                if matched_name in self._face_db:
                    self._face_db[matched_name].append(emb_norm.copy())
                else:
                    self._face_db[matched_name] = [emb_norm.copy()]

            logger.info(f"[CL] Neuer Winkel gespeichert: {cl_key} "
                        f"(sim={matched_sim:.3f}, scrfd={scrfd_conf:.3f}, "
                        f"total={n_existing + 1})")

        except Exception as e:
            logger.warning(f"[CL] Speichern fehlgeschlagen: {e}")

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

            for name, db_embs in self._face_db.items():
                # db_embs ist Liste von Embeddings (Best-Match statt Mean)
                if isinstance(db_embs, np.ndarray) and db_embs.ndim == 1:
                    db_embs = [db_embs]  # Rueckwaerts-Kompatibilitaet
                for db_emb in db_embs:
                    sim = float(np.dot(embedding, db_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

            if best_sim >= threshold:
                logger.debug(f"[FACE-MATCH] {best_name} sim={best_sim:.3f} (thresh={threshold:.2f})")
                return (best_name, best_sim)
            logger.debug(f"[FACE-MATCH] KEIN Match: best={best_name} sim={best_sim:.3f} < thresh={threshold:.2f}")
            return (None, best_sim)

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

        # Distanz + BBox-Hoehe aus groesster Person-BBox
        if persons:
            biggest = max(persons, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
            bbox = biggest["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)
            pf.person_bbox_height = bbox[3] - bbox[1]
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

        # Face Attributes (bestes Face mit Attributen)
        if faces:
            best_attr_face = max(faces, key=lambda f: f.get("confidence", 0))
            if best_attr_face.get("gender"):
                pf.gender = best_attr_face["gender"]
            if best_attr_face.get("smiling") is not None:
                pf.emotion = "happy" if best_attr_face["smiling"] else "neutral"

        # Perception Router: Szenario + aktive Modelle
        pf.scenario = self._scheduler.get_scenario()
        # Tatsaechlich laufende Modelle (nicht Scheduler-Vorschlag, sondern Valve-Realitaet)
        real_active = []
        if self.scrfd_active: real_active.extend(["scrfd", "arcface", "faceattr"])
        if self.pose_active: real_active.append("pose")
        if self.reid_active: real_active.append("reid")
        if self.hand_active: real_active.append("hand")
        real_active.append("yolo")  # YOLO laeuft immer
        pf.active_models = sorted(real_active)

        # Action Inference (Temporal Pose Buffer)
        try:
            pf.person_action = get_action_inferrer()._last_action
        except Exception:
            pass

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

    def _init_shm_mmap(self):
        """SHM mmap initialisieren — einmal allozieren, danach nur noch schreiben."""
        import mmap
        try:
            fd = os.open(SHM_FRAME_PATH, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, SHM_TOTAL_SIZE)
            self._shm_mmap = mmap.mmap(fd, SHM_TOTAL_SIZE)
            self._shm_fd_raw = fd
            logger.info(f"[SHM] mmap initialisiert: {SHM_TOTAL_SIZE} Bytes")
        except Exception as e:
            logger.error(f"[SHM] mmap init fehlgeschlagen: {e}")
            self._shm_mmap = None
            self._shm_fd_raw = -1

    def _cleanup_shm(self):
        """SHM-Frame loeschen damit Panel sofort 'Kein Signal' zeigt."""
        # mmap schliessen
        if hasattr(self, '_shm_mmap') and self._shm_mmap:
            try:
                self._shm_mmap.close()
            except Exception:
                pass
            self._shm_mmap = None
        if hasattr(self, '_shm_fd_raw') and self._shm_fd_raw >= 0:
            try:
                os.close(self._shm_fd_raw)
            except Exception:
                pass
            self._shm_fd_raw = -1
        # Datei loeschen damit Panel "Kein Signal" zeigt
        for path in [SHM_FRAME_PATH, SHM_FRAME_PATH + '.tmp']:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def _write_shm_frame(self, frame: np.ndarray):
        """Frame per mmap nach /dev/shm/moloch_frame schreiben.

        Kein open/close/rename pro Frame — nur memcpy in bestehenden mmap-Buffer.
        24-Byte Header: h, w, c, seq (uint32 LE) + timestamp (float64 LE).
        """
        try:
            # Lazy init: mmap beim ersten Frame anlegen
            if not hasattr(self, '_shm_mmap') or self._shm_mmap is None:
                self._init_shm_mmap()
                if self._shm_mmap is None:
                    return

            if frame is None or frame.size == 0:
                return
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return
            if h != SHM_PREVIEW_H or w != SHM_PREVIEW_W:
                frame = cv2.resize(frame, (SHM_PREVIEW_W, SHM_PREVIEW_H))
                h, w = SHM_PREVIEW_H, SHM_PREVIEW_W
            c = frame.shape[2] if len(frame.shape) > 2 else 1
            self._shm_seq = (self._shm_seq + 1) & 0xFFFFFFFF
            ts = time.monotonic()
            header = struct.pack('<IIIId', h, w, c, self._shm_seq, ts)
            # Direkt in mmap schreiben — kein Syscall ausser memcpy
            self._shm_mmap.seek(0)
            self._shm_mmap.write(header)
            self._shm_mmap.write(frame.tobytes())
        except Exception as e:
            logger.warning(f"[SHM] Write-Fehler: {e}")
