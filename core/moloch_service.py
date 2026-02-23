#!/usr/bin/env python3
"""
M.O.L.O.C.H. Core Service - Headless Backend
==============================================

Extrahiert aus hailo_control_panel.py (Phase 2).
Enthaelt die gesamte Backend-Logik OHNE GUI-Abhaengigkeiten.

Laeuft als systemd service, Panel verbindet sich optional.

Komponenten:
  - Hailo NPU Inference Pipeline (SCRFD, ArcFace, YOLOv8m, Pose)
  - Tentakel-Modus (Smart Tracking <-> MOLOCH Autonomie)
  - Kamera-Kontrolle (ONVIF PTZ + eWeLink Cloud Bridge)
  - Autonomer Tracker (Face/Person Detection -> PTZ Moves)

GUI-Conversions:
  - tk.BooleanVar.get()  -> self.scrfd_active / self.yolo_active / ...
  - tk.BooleanVar.set()  -> self.xxx_active = True/False
  - tk.DoubleVar.get()   -> self.scrfd_conf_val / self.yolo_conf_val / ...
  - self.root.after(0,..) -> self._notify(event, data) oder direkt
  - btn.config(...)       -> self._notify("ui_update", {...})
"""

import os
import sys
import time
import json
import threading
import logging
import subprocess
import traceback

import cv2
import numpy as np

# Moloch path
sys.path.insert(0, os.path.expanduser("~/moloch"))

from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_objects, draw_hands,
    draw_poses, enforce_draw_priority,
    decode_hand_landmark, draw_hand_landmarks,
    decode_yolov8_pose,
    estimate_head_pose,
    COCO_LABELS,
)
from core.hardware.hailo_manager import get_hailo_manager
from core.vision.gesture_detector import GestureDetector, KeypointPosition
from core.vision.hand_gesture_detector import HandGestureDetector
from core.vision.face_attr_npu import analyze_face as _analyze_face
from core.led_controller import LEDController
from core.ipc_router import IPCRouter
from core.model_orchestrator import ModelOrchestrator, MODEL_PATHS
from core.camera_manager import CameraManager
from core.longterm_memory import get_memory
from core.perception.perception_frame import PerceptionFrame, estimate_distance
from core.perception.perception_buffer import get_perception_buffer
from core.perception.model_health import get_model_health

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MolochService")
logger.setLevel(logging.INFO)

FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")


def load_face_db(path: str) -> dict:
    """Lade Face-Embeddings aus JSON."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = {}
        for name, emb in data.items():
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            db[name] = arr
        return db
    except Exception as e:
        logger.error(f"Face-DB laden fehlgeschlagen: {e}")
        return {}


class MolochService:
    """
    M.O.L.O.C.H. Headless Backend Service.

    Enthaelt alle Logik die OHNE GUI laufen kann:
    - NPU Inference Pipeline
    - Tentakel-Modus (Takeover/Release)
    - Kamera-Kontrolle
    - Smart Tracking Toggle
    - Alle 4 NPU-Modelle permanent aktiv (8GB Hailo-10H)

    GUI-Aufrufe (root.after, BooleanVar) sind durch
    self._notify() Callbacks ersetzt.
    """

    def __init__(self):
        # State
        self.running = True
        self._hailo_manager = None
        self._face_db = {}

        # Core Integrator (Zentrales Zustandsmodell: Tension/Attention/Presence)
        self._core_integrator = None
        try:
            from core.core_integrator import get_core_integrator
            self._core_integrator = get_core_integrator()
            logger.info("[INIT] CoreIntegrator bereit")
        except Exception as e:
            logger.warning(f"[INIT] CoreIntegrator nicht verfuegbar: {e}")

        # CPU Detectors: Lazy-loaded beim ersten Aufruf (siehe _ensure_cpu_detectors)
        # _load_settings() setzt _cpu_detectors_enabled + _cpu_detect_interval
        self._emotion_detector = None
        self._age_gender_detector = None
        self._cpu_detectors_loaded = False

        # Gesture Detection (aus Pose-Keypoints)
        self._gesture_detector = GestureDetector()
        self._hand_gesture_detector = HandGestureDetector()
        self._current_gesture = None
        logger.info("[INIT] GestureDetector bereit")

        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None
        try:
            from core.perception_engine import PerceptionEngine
            from core.personality.personality_engine import get_personality_engine
            _pe = get_personality_engine()
            self._perception = PerceptionEngine(personality_engine=_pe)
            logger.info(f"[INIT] Perception Engine bereit (Personality: {_pe.mode.value})")
            # Gespeicherte Hand-Occlusion Params anwenden
            self._perception._hand_occlusion_enabled = getattr(self, '_hand_occlusion_enabled', False)
            if hasattr(self, '_saved_hand_timeout'):
                self._perception._HAND_TIMEOUT = self._saved_hand_timeout
                self._perception._MIN_FACE_STREAK = self._saved_hand_streak
                self._perception._FACE_RECENCY = self._saved_hand_recency
                logger.info(f"[SETTINGS] Hand-Occlusion: enabled={self._perception._hand_occlusion_enabled} "
                            f"Params aus settings.json angewendet")
            # Gespeicherte aktive Modelle als force_models setzen
            if hasattr(self, '_saved_active_models') and self._saved_active_models:
                self._perception.force_models(self._saved_active_models)
                logger.info(f"[SETTINGS] force_models({self._saved_active_models}) aus settings.json")
        except Exception as e:
            logger.warning(f"[INIT] Perception Engine nicht verfuegbar: {e}")

        # Daily Learner
        self._daily_learner = None
        self._learner_flash = False
        try:
            from core.daily_learner import get_daily_learner
            self._daily_learner = get_daily_learner()
            logger.info("[INIT] DailyLearner bereit")
        except Exception as e:
            logger.warning(f"[INIT] DailyLearner nicht verfuegbar: {e}")

        # Voice Pipeline (PTT -> Whisper -> Claude -> TTS)
        self._voice_pipeline = None
        try:
            from core.voice_pipeline import VoicePipeline
            self._voice_pipeline = VoicePipeline()
            logger.info("[INIT] Voice Pipeline bereit")
        except Exception as e:
            logger.warning(f"[INIT] Voice Pipeline nicht verfuegbar: {e}")

        self._input_640 = np.empty((640, 640, 3), dtype=np.uint8)

        # === Phase 3: Model Orchestration ===
        # Perception Buffer (Ring-Buffer fuer Trend-Analyse)
        self._perception_buffer = get_perception_buffer()
        # Model Health Monitor
        self._model_health = get_model_health()
        # Aktueller PerceptionFrame (letzter aggregierter Zustand)
        self._current_pframe = PerceptionFrame()

        # ModelOrchestrator (NPU Pipeline + Modell-Lifecycle, Phase 4)
        self._orchestrator = ModelOrchestrator(
            perception_engine=self._perception,
            core_integrator=self._core_integrator,
            daily_learner=self._daily_learner,
            model_health=self._model_health,
            notify_callback=self._notify,
        )
        # Aliased Referenzen auf Orchestrator-Objekte
        # (Inference Loop greift bis Schritt 5 noch auf self.xxx zu)
        self._active_ctx = self._orchestrator._active_ctx
        self._ctx_lock = self._orchestrator._ctx_lock
        self._configuring = self._orchestrator._configuring
        self._models = self._orchestrator._models
        self._output_names = self._orchestrator._output_names

        # Adaptive FPS (Orchestrator berechnet, Inference Loop liest)
        self._target_frame_delay = self._orchestrator.target_frame_delay
        # Pose-Energy Tracker (Keypoint-Bewegung Frame-zu-Frame)
        self._prev_keypoints = None

        # Throttling: Emotion/Age/Gender nur alle N Frames (CPU-Sparmode)
        # _cpu_detectors_enabled und _cpu_detect_interval werden in _load_settings() gesetzt
        self._cpu_detect_interval = 30  # Default ~1x/Sek bei 20 FPS
        self._cpu_detectors_enabled = False  # Default AUS (CPU zu teuer ohne NPU)
        self._hand_occlusion_enabled = False  # Default AUS (via settings.json steuerbar)
        self._frame_counter = 0
        self._cached_emotion = {}      # name -> emotion
        self._cached_gender = {}       # name -> gender
        self._cached_age_range = {}    # name -> age_range

        # TTS Announcement Cooldown
        self._last_announce = {}

        # LED Controller (extrahiert aus moloch_service.py, Phase 4)
        # Cloud wird spaeter via CameraManager.connect_cloud() gesetzt
        self._led = LEDController(core_integrator=self._core_integrator)

        # CameraManager (RTSP + Cloud + Tentakel + Autonomer Modus, Phase 4)
        self._cam = CameraManager(
            model_orchestrator=self._orchestrator,
            perception_engine=self._perception,
            led_controller=self._led,
            notify_callback=self._notify,
            sync_flags_callback=self._sync_flags_from_npu,
            set_model_flags_callback=self._set_model_flags_cb,
            fps_reset_callback=self._reset_fps,
        )
        # Aliased mutable Referenzen (Inference Loop greift bis Schritt 5 noch auf self.xxx zu)
        self._frame_lock = self._cam._frame_lock
        self._annotated_lock = self._cam._annotated_lock
        self._first_detection_event = self._cam._first_detection_event
        self._transition_lock = self._cam._transition_lock
        self._cloud_state = self._cam._cloud_state

        # Model enable flags (plain bools, NOT tk.BooleanVar)
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.hand_active = False
        self.pose_active = False
        self.face_attr_active = False

        # Watchdog: Anti-Oszillation Swap-Log (bleibt auf Service, Inference Loop schreibt hier)
        self._swap_log = []
        # Auto-Switch: Zaehlt Frames ohne Hand-Detection
        self._hand_no_detect = 0
        self._HAND_RELEASE_FRAMES = 75  # ~5s bei 15fps

        # Threshold values (plain floats, NOT tk.DoubleVar)
        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50

        # Audio-Defaults VOR _load_settings() (W4 Audit-Fix)
        self._saved_mic_gain = 1.0
        self._saved_noise_gate = -36.0
        self._saved_agc = True
        self._audio_level = 0.0

        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # FPS Tracking
        self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "hand_landmark": 0, "pose": 0, "total": 0}
        self._fps_lock = threading.Lock()

        # IPC Router (extrahiert aus moloch_service.py, Phase 4)
        self._ipc = IPCRouter()

        # Callback: GUI kann sich hier einklinken
        # Signature: _notify(event: str, data: dict)
        self._observers = []

    # =========================================================================
    # Observer Pattern
    # =========================================================================

    def add_observer(self, callback):
        """Register GUI observer: callback(event, data)"""
        self._observers.append(callback)

    def _notify(self, event: str, data: dict = None):
        """Notify all observers of state change."""
        for cb in self._observers:
            try:
                cb(event, data or {})
            except Exception:
                pass

    def _update_status(self, text):
        """Status update via observer pattern."""
        logger.info(f"[STATUS] {text}")
        self._notify("status", {"text": text})

    # =========================================================================
    # Orchestrator Proxy Properties (Uebergangsphase bis Schritt 5)
    # =========================================================================

    @property
    def _vdevice(self):
        return self._orchestrator._vdevice

    @_vdevice.setter
    def _vdevice(self, value):
        self._orchestrator._vdevice = value

    @_vdevice.deleter
    def _vdevice(self):
        self._orchestrator._vdevice = None

    @property
    def _npu_paused(self):
        return self._orchestrator._npu_paused

    @_npu_paused.setter
    def _npu_paused(self, value):
        self._orchestrator._npu_paused = value

    @property
    def _paused_models(self):
        return self._orchestrator._paused_models

    @_paused_models.setter
    def _paused_models(self, value):
        self._orchestrator._paused_models = value

    @property
    def _models_preloaded(self):
        return self._orchestrator._models_preloaded

    @_models_preloaded.setter
    def _models_preloaded(self, value):
        self._orchestrator._models_preloaded = value

    # =========================================================================
    # CameraManager Proxy Properties (Uebergangsphase bis Schritt 5)
    # =========================================================================

    @property
    def _latest_frame(self):
        return self._cam._latest_frame

    @property
    def _annotated_frame(self):
        return self._cam._annotated_frame

    @_annotated_frame.setter
    def _annotated_frame(self, value):
        self._cam._annotated_frame = value

    @property
    def _moloch_has_control(self):
        return self._cam._moloch_has_control

    @property
    def _autonomous_mode(self):
        return self._cam._autonomous_mode

    @property
    def _manual_mode(self):
        return self._cam._manual_mode

    @property
    def _tentakel_enabled(self):
        return self._cam._tentakel_enabled

    @property
    def _smart_tracking_on(self):
        return self._cam._smart_tracking_on

    @property
    def _cloud(self):
        return self._cam._cloud

    @property
    def _alarm_on(self):
        return self._cam._alarm_on

    @_alarm_on.setter
    def _alarm_on(self, value):
        self._cam._alarm_on = value

    @property
    def _waiting_for_first_detection(self):
        return self._cam._waiting_for_first_detection

    @property
    def _takeover_found_something(self):
        return self._cam._takeover_found_something

    @_takeover_found_something.setter
    def _takeover_found_something(self, value):
        self._cam._takeover_found_something = value

    @property
    def _last_interesting_time(self):
        return self._cam._last_interesting_time

    @_last_interesting_time.setter
    def _last_interesting_time(self, value):
        self._cam._last_interesting_time = value

    @property
    def _tracker(self):
        return self._cam._tracker

    @property
    def _last_frame_write(self):
        return self._cam._last_frame_write

    @property
    def _frozen_restart_count(self):
        return self._cam._frozen_restart_count

    # CameraManager Callbacks

    def _set_model_flags_cb(self, flags_dict):
        """Callback fuer CameraManager: Model-Flags auf Service setzen."""
        for attr, val in flags_dict.items():
            setattr(self, attr, val)

    def _reset_fps(self):
        """Callback fuer CameraManager: FPS Tracking zuruecksetzen."""
        with self._fps_lock:
            self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0,
                         "hand_landmark": 0, "pose": 0, "total": 0}

    # =========================================================================
    # RTSP Capture
    # =========================================================================

    def _start_rtsp(self):
        """Thin Wrapper -> CameraManager.start_rtsp()."""
        self._cam.start_rtsp()

    # =========================================================================
    # NPU Pipeline
    # =========================================================================

    def _configure_model(self, name):
        """Thin Wrapper -> ModelOrchestrator.configure()."""
        self._orchestrator.configure(name)

    def _unconfigure_model(self, name):
        """Thin Wrapper -> ModelOrchestrator.unconfigure()."""
        self._orchestrator.unconfigure(name)

    def _run_model(self, name, input_data):
        """Thin Wrapper -> ModelOrchestrator.run()."""
        return self._orchestrator.run(name, input_data)

    # =========================================================================
    # Inference Loop
    # =========================================================================

    def _inference_loop(self):
        """Inference Worker mit Auto-Restart bei Crash."""
        restart_count = 0
        while self.running:
            try:
                self._inference_loop_inner()
            except Exception as e:
                crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
                sep = "=" * 60
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                models = list(self._active_ctx.keys())
                tb = traceback.format_exc()
                crash_info = (
                    f"\n{sep}\n"
                    f"[{ts}] INFERENCE LOOP CRASH #{restart_count + 1}\n"
                    f"Aktive Modelle: {models}\n"
                    f"Exception: {type(e).__name__}: {e}\n"
                    f"Traceback:\n{tb}\n"
                    f"{sep}\n"
                )
                logger.error(crash_info)
                try:
                    with open(crash_log, "a", encoding="utf-8") as f:
                        f.write(crash_info)
                except Exception:
                    pass
                # Recovery: reset state for clean restart
                self._npu_paused = False
                restart_count += 1
                self._update_status(f"INFERENCE CRASH #{restart_count} - Neustart in 2s...")
                logger.warning(f"[INFERENCE] Crash #{restart_count} - restarting in 2s...")
                time.sleep(2)

    def _inference_loop_inner(self):
        """Eigentliche Inference Loop (GUI-frei)."""
        while self.running:
            # Cross-process NPU coordination: Voice hat Vorrang
            if self._orchestrator.check_voice_request():
                time.sleep(0.1)
                continue

            # Safety: models empty = auto-recover
            if self._orchestrator.auto_recover_models():
                continue

            # Frame holen
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue

            # Pause waehrend Modell-Konfiguration (NPU blockiert)
            if not self._configuring.wait(timeout=0.1):
                with self._annotated_lock:
                    self._annotated_frame = frame.copy()
                continue

            # === NPU WATCHDOG: Anti-Oszillation (kein Max-Limit bei 8GB) ===
            self._npu_watchdog()
            self._last_hand_detected = False  # Default: keine Hand pro Frame

            # Kein Modell konfiguriert ODER Inference pausiert -> Raw-Frame
            any_active = bool(self._active_ctx) and (
                self.scrfd_active or self.yolo_active or self.hand_active or self.pose_active)
            if not any_active:
                # Perception tick auch ohne aktive Modelle (forced/initial swap)
                if self._perception:
                    _idle_ctx = {
                        "face_detected": False, "face_bbox": None,
                        "person_detected": False, "unknown_person": False,
                        "motion_level": 0.0, "camera_moving": False,
                    }
                    _new_slots = self._perception.tick(_idle_ctx)
                    if _new_slots:
                        _want = set(_new_slots) | {"face_attr"}
                        _have = set(self._active_ctx.keys())
                        _to_remove = _have - _want
                        _to_add = _want - _have
                        if _to_remove or _to_add:
                            logger.info(f"[PERCEPTION] Swap (idle): {_have} -> {_want}")
                            for _m in _to_remove:
                                self._unconfigure_model(_m)
                                time.sleep(0.2)
                            for _m in _to_add:
                                if _m not in self._active_ctx:
                                    self._configure_model(_m)
                            # Sync perception slots + Flags aus NPU-Realitaet
                            self._perception.slots = list(self._active_ctx.keys())
                            self._sync_flags_from_npu()
                            self._swap_log.append(time.time())
                            self._notify("model_toggle", {
                                "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                                "yolov8m": self.yolo_active,
                                "hand_landmark": self.hand_active})
                            continue
                with self._annotated_lock:
                    self._annotated_frame = frame.copy()
                # SHM: Preview-Groesse fuer Panel IPC (1080p waere 6MB/Frame)
                self._ipc.write_frame(cv2.resize(frame, (IPCRouter.PREVIEW_W, IPCRouter.PREVIEW_H)))
                self._write_status_json()
                time.sleep(0.03)
                continue

            t_total = time.perf_counter()
            annotated = frame.copy()
            fh, fw = frame.shape[:2]
            self._frame_counter += 1
            _run_cpu_detectors = (self._frame_counter % self._cpu_detect_interval == 0)

            # Preprocessing: Resize auf 640x640 fuer Modelle
            input_640 = cv2.resize(frame, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            scale_x = fw / 640.0
            scale_y = fh / 640.0

            # Max-2 Draw-Priority: face > hand
            _draw_candidates = []
            if self.scrfd_active:
                _draw_candidates.append("face")
            if self.hand_active:
                _draw_candidates.append("hand")
            _allowed_draws = set(enforce_draw_priority(_draw_candidates))

            face_boxes = []
            face_detected = False
            face_fed_to_tracker = False
            _markus_recognized = False
            _persons_detected = False

            # 1. SCRFD Face Detection
            if self.scrfd_active and "scrfd" in self._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("scrfd", input_rgb)
                    boxes, scores, landmarks = decode_scrfd(
                        outputs, img_size=640,
                        conf_thresh=self.scrfd_conf_val,
                        iou_thresh=self.scrfd_nms_val
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("scrfd", dt * 1000)

                    if len(boxes) > 0:
                        if "face" in _allowed_draws:
                            draw_faces(annotated, boxes, scores, landmarks, scale_x, scale_y)
                        face_boxes = list(zip(boxes, scores, landmarks))
                        face_detected = True
                        # Head Pose fuer erstes Gesicht (CPU, ~5ms)
                        _head_pose = estimate_head_pose(landmarks[0], fw, fh)
                        # Face hat PRIORITAET fuer Tracker
                        if self._autonomous_mode and self._tracker:
                            try:
                                face_dets = []
                                for box, score, _ in face_boxes:
                                    face_dets.append({
                                        "bbox": [box[0] * 640, box[1] * 640, box[2] * 640, box[3] * 640],
                                        "confidence": float(score),
                                        "class": "face"
                                    })
                                self._tracker.update_detection(
                                    detections=face_dets,
                                    frame_width=640, frame_height=640
                                )
                                face_fed_to_tracker = True
                            except Exception as e:
                                logger.debug(f"Tracker face feed: {e}")
                        # Guardian: Face sichtbar -> Interest
                        if self._moloch_has_control:
                            self._last_interesting_time = time.time()
                            self._takeover_found_something = True
                        # Fliessender Takeover: erste Detection signalisieren
                        if self._waiting_for_first_detection:
                            self._first_detection_event.set()
                except Exception as e:
                    logger.error(f"SCRFD Fehler: {e}")
                    self._model_health.record_error("scrfd")


            # Lazy-configure face_attr (einmalig ~400ms, danach 0ms)
            if not self.face_attr_active and "face_attr" in self._models and face_boxes:
                self._configure_model("face_attr")
                self.face_attr_active = "face_attr" in self._active_ctx

            # 2. ArcFace (nur wenn SCRFD aktiv + Faces gefunden)
            if (self.arcface_active and self.scrfd_active
                    and face_boxes and "arcface" in self._active_ctx):
                try:
                    t0 = time.perf_counter()
                    for box, score, lm in face_boxes:
                        x1 = max(0, int(box[0] * fw))
                        y1 = max(0, int(box[1] * fh))
                        x2 = min(fw, int(box[2] * fw))
                        y2 = min(fh, int(box[3] * fh))

                        bw, bh = x2 - x1, y2 - y1
                        mx, my = int(bw * 0.2), int(bh * 0.2)
                        x1 = max(0, x1 - mx)
                        y1 = max(0, y1 - my)
                        x2 = min(fw, x2 + mx)
                        y2 = min(fh, y2 + my)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        crop_112 = cv2.resize(crop, (112, 112))
                        crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

                        outputs = self._run_model("arcface", crop_rgb)
                        if outputs:
                            emb_key = self._output_names["arcface"][0]
                            embedding = outputs[emb_key].flatten()
                            embedding = normalize_arcface(embedding)

                            if self._face_db:
                                name, sim = match_face(
                                    embedding, self._face_db,
                                    threshold=self.arcface_thresh_val
                                )
                            else:
                                name, sim = "Keine DB", 0.0

                            # LED Indikator: Markus erkannt?
                            if name.lower() == "markus":
                                _markus_recognized = True

                            # Face Attributes (NPU, ~2926 FPS — Gender/Age/Emotion)
                            emotion = self._cached_emotion.get(name)
                            gender = self._cached_gender.get(name)
                            age_range = self._cached_age_range.get(name)
                            if self.face_attr_active and crop is not None:
                                try:
                                    fa_crop = cv2.resize(crop, (178, 218))
                                    fa_rgb = cv2.cvtColor(fa_crop, cv2.COLOR_BGR2RGB)
                                    fa_out = self._run_model("face_attr", fa_rgb)
                                    if fa_out:
                                        fa_key = self._output_names["face_attr"][0]
                                        gender, age_range, emotion = _analyze_face(fa_out[fa_key])
                                        self._cached_gender[name] = gender
                                        self._cached_age_range[name] = age_range
                                        self._cached_emotion[name] = emotion
                                except Exception:
                                    pass

                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._ipc.write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range,
                                                   head_pose=_head_pose if '_head_pose' in dir() else None,
                                                   detected_objects=_detected_objects if '_detected_objects' in dir() else [])

                            # DailyLearner: Snapshot bei erkanntem Gesicht
                            if self._daily_learner and self._daily_learner.enabled and name != "Keine DB":
                                try:
                                    _hp = None
                                    if '_head_pose' in dir() and _head_pose is not None:
                                        _hp = {"pitch": _head_pose[0], "yaw": _head_pose[1], "roll": _head_pose[2]}
                                    # Breiterer Crop fuer Learner (50% Margin statt 20%)
                                    _lx1 = max(0, int(box[0] * fw))
                                    _ly1 = max(0, int(box[1] * fh))
                                    _lx2 = min(fw, int(box[2] * fw))
                                    _ly2 = min(fh, int(box[3] * fh))
                                    _lbw, _lbh = _lx2 - _lx1, _ly2 - _ly1
                                    _lmx, _lmy = int(_lbw * 0.5), int(_lbh * 0.5)
                                    _lx1 = max(0, _lx1 - _lmx)
                                    _ly1 = max(0, _ly1 - _lmy)
                                    _lx2 = min(fw, _lx2 + _lmx)
                                    _ly2 = min(fh, _ly2 + _lmy)
                                    learner_crop = frame[_ly1:_ly2, _lx1:_lx2]
                                    _saved = self._daily_learner.maybe_snapshot(
                                        face_crop=learner_crop,
                                        name=name,
                                        confidence=sim,
                                        bbox=(float(_lx1), float(_ly1), float(_lx2), float(_ly2)),
                                        frame_height=fh,
                                        head_pose=_hp,
                                        full_frame=frame,
                                        embedding=embedding,
                                    )
                                    # LED-Blitz bei erfolgreichem Snapshot
                                    if _saved and self._learner_flash:
                                        threading.Thread(
                                            target=self._led.flash_white,
                                            daemon=True
                                        ).start()
                                except Exception as e:
                                    logger.debug(f"DailyLearner: {e}")

                            # TTS Ansage (60s Cooldown pro Person)
                            if name != "Unbekannt" and name != "Keine DB":
                                now = time.time()
                                if now - self._last_announce.get(name, 0) > 60:
                                    self._last_announce[name] = now
                                    threading.Thread(
                                        target=self._announce_person,
                                        args=(name,), daemon=True
                                    ).start()

                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["arcface"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("arcface", dt * 1000)
                except Exception as e:
                    logger.error(f"ArcFace Fehler: {e}")
                    self._model_health.record_error("arcface")

            # 3. YOLOv8m Detection (alle COCO Klassen, uebersprungen wenn Face erkannt)
            _detected_objects = []  # Nicht-Person-Objekte fuer Status
            if self.yolo_active and "yolov8m" in self._active_ctx and not face_detected:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("yolov8m", input_rgb)
                    out_key = self._output_names["yolov8m"][0]
                    all_dets = decode_yolov8_nms(
                        outputs[out_key],
                        class_id=-1,
                        conf_thresh=self.yolo_conf_val
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["yolov8m"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("yolov8m", dt * 1000)

                    # Personen und andere Objekte trennen
                    persons = [d for d in all_dets if d.get("class_id", -1) == 0]
                    objects = [d for d in all_dets if d.get("class_id", -1) != 0]

                    # Nicht-Person-Objekte zeichnen (orange)
                    if objects:
                        draw_objects(annotated, objects, scale_x, scale_y)
                        _detected_objects = [
                            {"class": d["class"], "confidence": round(d["confidence"], 2)}
                            for d in objects
                        ]

                    if persons:
                        _persons_detected = True
                        draw_persons(annotated, persons, scale_x, scale_y)
                        if self._moloch_has_control:
                            self._last_interesting_time = time.time()
                            self._takeover_found_something = True
                        # Fliessender Takeover: erste Detection signalisieren
                        if self._waiting_for_first_detection:
                            self._first_detection_event.set()
                        if self._autonomous_mode and self._tracker and not face_fed_to_tracker:
                            try:
                                pixel_dets = []
                                for p in persons:
                                    bx = p["bbox"]
                                    pixel_dets.append({
                                        "bbox": [bx[0] * 640, bx[1] * 640, bx[2] * 640, bx[3] * 640],
                                        "confidence": p["confidence"],
                                        "class": "person"
                                    })
                                self._tracker.update_detection(
                                    detections=pixel_dets,
                                    frame_width=640, frame_height=640
                                )
                            except Exception as e:
                                logger.debug(f"Tracker YOLOv8m feed: {e}")
                except Exception as e:
                    logger.error(f"YOLOv8m Fehler: {e}")
                    self._model_health.record_error("yolov8m")

            # 4. Hand Landmark Detection (224x224 Crop aus Person-BBox oder Bildmitte)
            if self.hand_active and "hand_landmark" in self._active_ctx:
                try:
                    t0 = time.perf_counter()

                    # Crop-Region bestimmen (in 640x640 Space)
                    if _persons_detected and 'persons' in dir() and persons:
                        # Obere Haelfte der groessten Person-BBox (Haende sind oben)
                        p = max(persons, key=lambda d: d["confidence"])
                        bx = p["bbox"]  # [x1, y1, x2, y2] normalisiert 0-1
                        cx1 = int(bx[0] * 640)
                        cy1 = int(bx[1] * 640)
                        cx2 = int(bx[2] * 640)
                        cy2 = int(bx[3] * 640)
                        # Obere 60% der Person (Haende/Arme)
                        ch = cy2 - cy1
                        cy2 = cy1 + int(ch * 0.6)
                    elif face_boxes:
                        # Face-BBox erweitert (Haende sind in der Naehe)
                        fb = face_boxes[0][0]  # (x1, y1, x2, y2) normalisiert
                        cx = int((fb[0] + fb[2]) / 2 * 640)
                        cy = int((fb[1] + fb[3]) / 2 * 640)
                        cx1 = max(0, cx - 160)
                        cy1 = max(0, cy - 80)
                        cx2 = min(640, cx + 160)
                        cy2 = min(640, cy + 240)
                    else:
                        # Bildmitte als Fallback
                        cx1, cy1, cx2, cy2 = 120, 80, 520, 560

                    # Crop aus 640x640 und auf 224x224 skalieren
                    cx1 = max(0, cx1)
                    cy1 = max(0, cy1)
                    cx2 = min(640, cx2)
                    cy2 = min(640, cy2)
                    crop_w = max(cx2 - cx1, 1)
                    crop_h = max(cy2 - cy1, 1)

                    hand_crop = input_rgb[cy1:cy2, cx1:cx2]
                    hand_224 = cv2.resize(hand_crop, (224, 224))

                    outputs = self._run_model("hand_landmark", hand_224)
                    hand_result = decode_hand_landmark(outputs)

                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["hand_landmark"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("hand_landmark", dt * 1000)

                    if hand_result is not None:
                        self._last_hand_detected = True
                        if "hand" in _allowed_draws:
                            draw_hand_landmarks(
                                annotated, hand_result,
                                crop_x=cx1, crop_y=cy1,
                                crop_w=crop_w, crop_h=crop_h,
                                scale_x=scale_x, scale_y=scale_y,
                            )
                        # Hand-Gesture Detection aus 21 MediaPipe Landmarks (W1 Audit-Fix)
                        try:
                            gesture = self._hand_gesture_detector.detect(
                                hand_result["landmarks"],
                                hand_result.get("handedness", "R")
                            )
                            self._current_gesture = gesture
                        except Exception:
                            pass
                    else:
                        self._last_hand_detected = False

                except Exception as e:
                    logger.error(f"Hand Landmark Fehler: {e}")
                    self._model_health.record_error("hand_landmark")

            # 5. Pose Estimation (YOLOv8s Pose - Skeleton + Keypoints)
            _pose_data = []
            if self.pose_active and "pose" in self._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("pose", input_rgb)
                    _pose_data = decode_yolov8_pose(
                        outputs,
                        conf_thresh=self.yolo_conf_val,
                        img_h=640, img_w=640,
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["pose"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("pose", dt * 1000)

                    if _pose_data:
                        draw_poses(annotated, _pose_data, scale_x, scale_y)
                        # Tracker mit Pose-Daten fuettern (FACE > BODY Prioritaet)
                        if self._autonomous_mode and self._tracker and not face_fed_to_tracker:
                            try:
                                pose_dets = []
                                for p in _pose_data:
                                    kpts = p["keypoints"]  # (17, 3) in model pixels
                                    # Face-Center aus Nase (kpt 0) + Augen (kpt 1,2)
                                    face_kpts = [0, 1, 2, 3, 4]  # nose, l_eye, r_eye, l_ear, r_ear
                                    face_vis = [kpts[k, 2] for k in face_kpts]
                                    has_face = sum(1 for v in face_vis if v > 0.3) >= 3
                                    face_center = None
                                    if has_face:
                                        fx = np.mean([kpts[k, 0] for k in face_kpts if kpts[k, 2] > 0.3])
                                        fy = np.mean([kpts[k, 1] for k in face_kpts if kpts[k, 2] > 0.3])
                                        face_center = (fx / 640.0, fy / 640.0)
                                    # Torso: Schultern (5,6) + Hueften (11,12)
                                    torso_kpts = [5, 6, 11, 12]
                                    has_torso = sum(1 for k in torso_kpts if kpts[k, 2] > 0.3) >= 3
                                    face_conf = float(np.mean(face_vis)) if has_face else 0.0
                                    pose_dets.append({
                                        "bbox": p["bbox"],
                                        "confidence": p["score"],
                                        "has_face": has_face,
                                        "face_center": face_center,
                                        "face_confidence": face_conf,
                                        "has_torso": has_torso,
                                    })
                                self._tracker.update_pose_detection(
                                    poses=pose_dets,
                                    frame_width=640, frame_height=640
                                )
                                face_fed_to_tracker = True  # Pose hat Tracker gefuettert
                            except Exception as e:
                                logger.debug(f"Tracker pose feed: {e}")
                except Exception as e:
                    logger.error(f"Pose Fehler: {e}")
                    self._model_health.record_error("pose")

            # ===== Perception Engine: All-Slot (alle 4 permanent, nur beim Start) =====
            if self._perception:
                _perc_face_bbox = None
                if face_boxes:
                    _fb = face_boxes[0][0]
                    _perc_face_bbox = (float(_fb[0]), float(_fb[1]), float(_fb[2]), float(_fb[3]))
                _perc_camera_moving = False
                if self._tracker and hasattr(self._tracker, '_camera') and self._tracker._camera:
                    _cam_pos = getattr(self._tracker._camera, 'current_position', None)
                    if _cam_pos:
                        _perc_camera_moving = getattr(_cam_pos, 'moving', False)
                _perc_person = False
                if self.yolo_active and 'persons' in dir() and persons:
                    _perc_person = True
                elif getattr(self, '_last_person_boxes', []):
                    _perc_person = True
                _person_count = len(persons) if self.yolo_active and 'persons' in dir() and persons else 0
                _face_count = len(face_boxes)
                _perc_ctx = {
                    "face_detected": face_detected,
                    "face_bbox": _perc_face_bbox,
                    "person_detected": _perc_person,
                    "unknown_person": face_detected and 'name' in dir() and name == "Unbekannt",
                    "person_count": _person_count,
                    "face_count": _face_count,
                    "detected_objects": _detected_objects if '_detected_objects' in dir() else [],
                    "pose_count": len(_pose_data) if '_pose_data' in dir() and _pose_data else 0,
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                    "gesture": self._current_gesture.type.value if self._current_gesture else "none",
                }
                _new_slots = self._perception.tick(_perc_ctx)
                if _new_slots:
                    _want = set(_new_slots) | {"face_attr"}
                    _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")
                        for _m in _to_remove:
                            self._unconfigure_model(_m)
                            time.sleep(0.2)
                        for _m in _to_add:
                            if _m not in self._active_ctx:
                                self._configure_model(_m)
                        # Sync perception slots + Flags aus NPU-Realitaet
                        self._perception.slots = list(self._active_ctx.keys())
                        self._sync_flags_from_npu()
                        self._swap_log.append(time.time())
                        self._notify("model_toggle", {
                            "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                            "yolov8m": self.yolo_active,
                            "hand_landmark": self.hand_active})

            # === LED Erkennungs-Indikator (Hysterese im LEDController) ===
            self._led.update_hysteresis(
                markus_recognized=_markus_recognized,
                face_detected=face_detected,
                persons_detected=_persons_detected,
                moloch_has_control=self._moloch_has_control,
            )

            # === Phase 3: Perception Frame aggregieren ===
            _pf_name = name if 'name' in dir() else None
            _pf_sim = sim if 'sim' in dir() else 0.0
            _pf_head = _head_pose if '_head_pose' in dir() else None
            _pf_persons = persons if 'persons' in dir() and _persons_detected else []
            pframe = self._build_perception_frame(
                face_detected=face_detected,
                face_boxes=face_boxes,
                _markus_recognized=_markus_recognized,
                _persons_detected=_persons_detected,
                persons=_pf_persons,
                _pose_data=_pose_data,
                _detected_objects=_detected_objects if '_detected_objects' in dir() else [],
                name=_pf_name,
                sim=_pf_sim,
                fw=fw, fh=fh,
                _head_pose=_pf_head,
                t_total=t_total,
            )
            self._current_pframe = pframe
            self._perception_buffer.push(pframe)

            # === Core Integrator fuettern (via PerceptionFrame — reichere Daten) ===
            if self._core_integrator:
                try:
                    # Perception-Daten -> Integrator (erweitert mit Trends)
                    self._core_integrator.update_inputs("perception", {
                        "face_detected": 1.0 if pframe.face_detected else 0.0,
                        "face_confidence": pframe.face_confidence,
                        "person_detected": 1.0 if pframe.person_detected else 0.0,
                        "markus_recognized": 1.0 if pframe.markus_recognized else 0.0,
                        "unknown_person": 1.0 if pframe.unknown_face else 0.0,
                        "proximity": pframe.distance_ratio,
                    })
                    # Alarm-State
                    self._core_integrator.update_input("system", "alarm_active", 1.0 if self._alarm_on else 0.0)
                    # System-Last (grob: NPU aktiv = etwas Last)
                    _npu_load = len(self._active_ctx) / 6.0  # 0-6 Modelle -> 0.0-1.0
                    self._core_integrator.update_input("system", "system_load", _npu_load)
                except Exception:
                    pass  # Integrator darf NIE die Inference-Loop stoeren

            # === Phase 3: Attention-Level basierte Modell-Orchestrierung ===
            try:
                new_level = self._compute_attention_level()
                self._apply_attention_level(new_level)
            except Exception as e:
                logger.debug(f"[ORCHESTRATION] Fehler: {e}")

            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:
                if self._last_hand_detected:
                    self._hand_no_detect = 0
                else:
                    self._hand_no_detect += 1
                    if self._hand_no_detect >= self._HAND_RELEASE_FRAMES:
                        if not self._manual_mode:
                            logger.info(f"[AUTO-SWITCH] {self._HAND_RELEASE_FRAMES} Frames keine Hand -> Auto-Scoring")
                            self._perception.force_models(None)
                        self._hand_no_detect = 0

            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0

            # Hand-Occlusion Overlay auf Video (nur wenn enabled in settings.json)
            if getattr(self, '_hand_occlusion_enabled', False) and self._perception and self._perception._hand_occlusion:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (fw, 30), (0, 0, 180), -1)
                annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                cv2.putText(annotated, "HAND OCCLUSION", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self._annotated_lock:
                self._annotated_frame = annotated

            # Panel IPC: Preview-Groesse fuer SHM (1080p waere 6MB/Frame)
            self._ipc.write_frame(cv2.resize(annotated, (IPCRouter.PREVIEW_W, IPCRouter.PREVIEW_H)))
            self._write_status_json()

            # === Phase 3: Adaptive FPS — Throttle bei niedrigem Attention-Level ===
            if dt_total < self._target_frame_delay:
                _sleep = self._target_frame_delay - dt_total
                time.sleep(_sleep)

    # =========================================================================
    # CameraManager Thin Wrappers (Uebergangsphase bis Schritt 5)
    # =========================================================================

    def _moloch_takeover(self, reason: str):
        """Thin Wrapper -> CameraManager.moloch_takeover()."""
        self._cam.moloch_takeover(reason)

    def _moloch_release(self):
        """Thin Wrapper -> CameraManager.moloch_release()."""
        self._cam.moloch_release()

    def _enable_autonomous(self):
        """Thin Wrapper -> CameraManager.enable_autonomous()."""
        self._cam.enable_autonomous()

    def _disable_autonomous(self):
        """Thin Wrapper -> CameraManager.disable_autonomous()."""
        self._cam.disable_autonomous()

    def _all_models_off(self):
        """Thin Wrapper -> ModelOrchestrator.all_models_off() + FPS Reset."""
        self._orchestrator.all_models_off()
        self._sync_flags_from_npu()
        self._reset_fps()

    def _connect_cloud(self):
        """Thin Wrapper -> CameraManager.connect_cloud()."""
        self._cam.connect_cloud()

    def _toggle_smart_tracking(self):
        """Thin Wrapper -> CameraManager.toggle_smart_tracking()."""
        self._cam.toggle_smart_tracking()

    # =========================================================================
    # Face Recognition
    # =========================================================================

    def _reload_face_db(self):
        """Face-DB neu laden (nach Enrollment)."""
        self._face_db = load_face_db(FACE_DB_PATH)
        n = len(self._face_db)
        # Basis-Namen (ohne #learn Suffix) fuer Anzeige
        base_names = set(k.split('#')[0] for k in self._face_db.keys()) if self._face_db else set()
        learned = sum(1 for k in self._face_db if '#' in k)
        self._update_status(f"Face-DB: {len(base_names)} Personen, {learned} gelernt ({', '.join(base_names)})")
        # DailyLearner Referenz aktualisieren
        if self._daily_learner:
            self._daily_learner.set_face_db(self._face_db, FACE_DB_PATH)

    def _ensure_cpu_detectors(self):
        """Lazy-load CPU Detektoren (Emotion + Age/Gender) beim ersten Aufruf."""
        if self._cpu_detectors_loaded:
            return
        self._cpu_detectors_loaded = True
        try:
            from core.vision.emotion_detector import get_emotion_detector
            det = get_emotion_detector()
            if det and det.available:
                self._emotion_detector = det
                logger.info("[CPU-DET] Emotion Detection geladen (FER+ CPU)")
        except Exception as e:
            logger.warning(f"[CPU-DET] Emotion nicht verfuegbar: {e}")
        try:
            from core.vision.age_gender_detector import get_age_gender_detector
            det = get_age_gender_detector()
            if det and det.available:
                self._age_gender_detector = det
                logger.info("[CPU-DET] Age+Gender Detection geladen (Caffe CPU)")
        except Exception as e:
            logger.warning(f"[CPU-DET] Age+Gender nicht verfuegbar: {e}")

    def _announce_person(self, name):
        """Person erkannt - Log (LED wird vom Indikator gesteuert)."""
        logger.info(f"[FACE] Person erkannt: {name}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def init(self):
        """Hardware initialisieren: VDevice, Models, RTSP, Cloud."""
        logger.info("M.O.L.O.C.H. Service initialisiert...")

        # 0. Langzeitgedaechtnis initialisieren (SSD2, persistent)
        try:
            self._memory = get_memory()
            identity = self._memory.get_identity()
            logger.info(f"[INIT] Memory bereit: {identity.get('name', '?')} v{identity.get('version', '?')}")
        except Exception as e:
            self._memory = None
            logger.error(f"[INIT] Memory fehlgeschlagen: {e}")

        # 1. Hailo VDevice + Models (via ModelOrchestrator)
        self._hailo_manager = get_hailo_manager()
        self._orchestrator._hailo_manager = self._hailo_manager
        self._hailo_manager.acquire_for_vision(timeout=10.0)
        self._orchestrator.load_models()
        for name in self._models:
            logger.info(f"Modell geladen: {name} ({len(self._output_names[name])} outputs)")

        # 1b. Whisper permanent auf NPU laden (shared VDevice, 8GB reichen)
        try:
            from core.speech.hailo_whisper import get_whisper
            whisper = get_whisper()
            whisper.set_vdevice(self._orchestrator.vdevice)
        except Exception as e:
            logger.error(f"[INIT] Whisper NPU init fehlgeschlagen: {e}")

        # 2. Face DB
        self._face_db = load_face_db(FACE_DB_PATH)
        if self._face_db:
            logger.info(f"Face-DB: {len(self._face_db)} Personen")

        # 2b. DailyLearner mit Face-DB verbinden (Real-Time Learning)
        if self._daily_learner:
            self._daily_learner.set_face_db(self._face_db, FACE_DB_PATH)

        # 3. RTSP (via CameraManager)
        self._cam.start_rtsp()

        # 4. Cloud (via CameraManager, im Hintergrund)
        threading.Thread(target=self._cam.connect_cloud, daemon=True).start()

        self._update_status("M.O.L.O.C.H. Service bereit")

    def _sync_flags_from_npu(self):
        """Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten."""
        self.scrfd_active = "scrfd" in self._active_ctx
        self.arcface_active = "arcface" in self._active_ctx
        self.yolo_active = "yolov8m" in self._active_ctx
        self.hand_active = "hand_landmark" in self._active_ctx
        self.pose_active = "pose" in self._active_ctx
        self.face_attr_active = "face_attr" in self._active_ctx

    # =========================================================================
    # Phase 3: Attention-basierte Modell-Orchestrierung (Thin Wrappers)
    # =========================================================================

    def _compute_attention_level(self) -> str:
        """Thin Wrapper -> ModelOrchestrator.compute_attention_level()."""
        return self._orchestrator.compute_attention_level()

    def _apply_attention_level(self, new_level: str):
        """Thin Wrapper -> ModelOrchestrator.apply_attention_level() + Flag-Sync."""
        if self._manual_mode:
            return
        self._orchestrator.apply_attention_level(new_level)
        self._sync_flags_from_npu()
        self._target_frame_delay = self._orchestrator.target_frame_delay

    def _build_perception_frame(self, face_detected, face_boxes, _markus_recognized,
                                 _persons_detected, persons, _pose_data, _detected_objects,
                                 name, sim, fw, fh, _head_pose, t_total) -> PerceptionFrame:
        """Baut einen aggregierten PerceptionFrame aus allen Modell-Outputs.

        Wird am Ende jedes Inference-Ticks aufgerufen.
        """
        pf = PerceptionFrame()
        pf.timestamp = time.time()

        # Person Detection
        pf.person_detected = _persons_detected or face_detected
        person_list = persons if _persons_detected and persons else []
        pf.person_count = len(person_list) if person_list else (1 if face_detected else 0)

        # Distanz aus groesster Person-BBox
        if person_list:
            biggest = max(person_list, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
            bbox = biggest["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Normalisiert 0-1
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)
        elif face_boxes:
            fb = face_boxes[0][0]
            area = (fb[2] - fb[0]) * (fb[3] - fb[1])
            pf.distance_ratio = area
            pf.distance = estimate_distance(area)

        # Face Detection
        pf.face_detected = face_detected
        pf.face_count = len(face_boxes)
        if face_boxes:
            pf.face_confidence = float(face_boxes[0][1])
            fb = face_boxes[0][0]
            pf.face_bbox = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))

        # Face Recognition
        if face_detected and name and name not in ("Keine DB", ""):
            pf.face_id = name.lower() if name != "Unbekannt" else "unknown"
            pf.face_similarity = sim if sim else 0.0

        # Face Attributes
        if face_detected and name:
            pf.gender = self._cached_gender.get(name)
            pf.age_range = self._cached_age_range.get(name)
            pf.emotion = self._cached_emotion.get(name)

        # Pose
        if _pose_data:
            pf.pose_count = len(_pose_data)
            pf.pose_energy = self._compute_pose_energy(_pose_data)

        # Hand/Gesture
        pf.hand_detected = getattr(self, '_last_hand_detected', False)
        if self._current_gesture:
            pf.hand_gesture = self._current_gesture.type.value

        # Head Pose
        if _head_pose is not None:
            pf.head_pitch = float(_head_pose[0])
            pf.head_yaw = float(_head_pose[1])

        # Objects
        pf.objects = _detected_objects if _detected_objects else []

        # Meta
        pf.inference_ms = (time.perf_counter() - t_total) * 1000
        pf.active_models = list(self._active_ctx.keys())

        return pf

    def _compute_pose_energy(self, pose_data) -> float:
        """Pose-Energie aus Keypoint-Bewegung berechnen (0.0-1.0).

        Vergleicht aktuelle Keypoints mit vorherigen. Hohe Bewegung = hohe Energie.
        """
        if not pose_data:
            return 0.0

        # Nimm die Person mit hoechstem Score
        best = max(pose_data, key=lambda p: p.get("score", 0))
        kpts = best.get("keypoints")
        if kpts is None:
            return 0.0

        # Keypoints: (17, 3) Array [x, y, confidence]
        current = kpts[:, :2]  # Nur x, y

        if self._prev_keypoints is None:
            self._prev_keypoints = current.copy()
            return 0.0

        # Differenz berechnen (nur sichtbare Keypoints)
        visible = (kpts[:, 2] > 0.3)
        if visible.sum() < 3:
            return 0.0

        diffs = np.linalg.norm(current[visible] - self._prev_keypoints[visible], axis=1)
        # Normalisieren: 640px Bildgroesse, >50px Bewegung = volle Energie
        energy = min(1.0, float(np.mean(diffs)) / 50.0)

        self._prev_keypoints = current.copy()
        return energy

    def _npu_watchdog(self):
        """Anti-Oszillation. Laeuft jede Inference-Iteration.
        Hailo-10H 8GB: Alle 4 Modelle passen gleichzeitig (~43MB)."""

        # 3) Anti-Oszillation: >3 Swaps in 1s -> Pause
        _now = time.time()
        self._swap_log = [t for t in self._swap_log if _now - t < 1.0]
        if len(self._swap_log) >= 3:
            logger.warning(f"[WATCHDOG] Anti-Oscillation: {len(self._swap_log)} Swaps in 1s! Pause 2s.")
            time.sleep(2.0)
            self._swap_log.clear()

    def start(self, blocking=True):
        """Service starten: Inference Loop + Kamera-Status Polling.

        Args:
            blocking: True = headless main loop, False = GUI-Modus (return sofort)
        """
        logger.info("M.O.L.O.C.H. Service gestartet")

        # Core Integrator Thread starten (1 Hz State-Berechnung)
        if self._core_integrator:
            self._core_integrator.start()
            logger.info("[START] CoreIntegrator 1Hz-Thread gestartet")

        # Inference Loop
        threading.Thread(target=self._inference_loop, daemon=True, name="InferenceLoop").start()

        # Kamera-Status + IPC Polling (via CameraManager)
        self._cam.start_cam_status_loop(write_status_callback=self._write_status_json)

        # Panel IPC Command Polling
        threading.Thread(target=self._poll_panel_cmds, daemon=True, name="PanelCmdPoll").start()

        # Autonomous Mode + Watchdog (via CameraManager)
        self._cam.enable_autonomous()
        logger.info("[START] Autonomous Mode aktiviert (Default nach Boot)")

        # nightVision auf day setzen (verhindert IR-Modus nach Reboot)
        def _reset_night_vision():
            try:
                time.sleep(5)  # Warten bis Cloud-Bridge bereit
                cloud = self._cam.cloud
                if cloud and cloud.connected:
                    cloud.run(cloud.bridge.set_night('day'))
                    self._cloud_state["led_level"] = 0
                    logger.info("[START] nightVision auf 'day' gesetzt")
            except Exception as e:
                logger.debug(f"[START] nightVision Reset fehlgeschlagen: {e}")
        threading.Thread(target=_reset_night_vision, daemon=True, name="NightVisionReset").start()

        self._cam.start_watchdog()

        # Spontane Kommentare Monitor starten (CoreIntegrator-gesteuert)
        if self._voice_pipeline:
            self._voice_pipeline.start_spontaneous_monitor()
            logger.info("[START] Spontane-Kommentare-Monitor gestartet")

        # Tageszeit-Begruessung (verzoegert, nach Cloud-Init)
        def _startup_greeting():
            time.sleep(15)  # Warten bis alles bereit ist
            if not self._voice_pipeline or not self._voice_pipeline._voice_enabled:
                return
            try:
                from core.core_integrator import get_core_integrator
                period = get_core_integrator().get_time_period()
                from core.personality.personality_engine import get_personality_engine
                pe = get_personality_engine()

                greetings = {
                    "morgens": {
                        "guardian": "Guten Morgen Markus. Systeme online.",
                        "shadow": "Moin. Kaffee?",
                        "berserker": "Aufstehen.",
                    },
                    "mittags": {
                        "guardian": "Systeme laufen. Alles normal.",
                        "shadow": "Na, auch mal wieder da?",
                        "berserker": "Status: Online.",
                    },
                    "abends": {
                        "guardian": "Guten Abend Markus. System bereit.",
                        "shadow": "N'Abend. Feierabend-Runde?",
                        "berserker": "Bin da.",
                    },
                    "nachts": {
                        "guardian": "Nachtmodus aktiv. Ich halte Wache.",
                        "shadow": "Nachtschicht. Wie immer.",
                        "berserker": "Nacht.",
                    },
                }
                zone = pe.mode.value
                text = greetings.get(period, {}).get(zone, "Moloch online.")
                self._voice_pipeline._speak(text)
                logger.info(f"[START] Begruessung: '{text}' (period={period}, zone={zone})")
            except Exception as e:
                logger.debug(f"[START] Begruessung fehlgeschlagen: {e}")
        threading.Thread(target=_startup_greeting, daemon=True, name="StartGreeting").start()

        if not blocking:
            return

        # Headless Main Loop
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - stopping...")
            self.stop()

    # =========================================================================
    # Public API (fuer Panel-Adapter)
    # =========================================================================

    def toggle_model(self, model_key, enabled):
        """Thin Wrapper -> ModelOrchestrator.toggle_model()."""
        self._orchestrator.toggle_model(model_key, enabled)

    def toggle_autonomous_manual(self):
        """Thin Wrapper -> CameraManager.toggle_autonomous_manual()."""
        self._cam.toggle_autonomous_manual()

    def stop(self):
        """Sauberes Herunterfahren."""
        logger.info("M.O.L.O.C.H. Service wird gestoppt...")
        self.running = False
        self._cam.running = False

        # Langzeitgedaechtnis: Core State SOFORT sichern
        if self._memory and self._core_integrator:
            try:
                state = self._core_integrator.get_state()
                state["personality_zone"] = self._core_integrator.get_personality_zone()
                self._memory.save_core_state(state)
                logger.info("[STOP] Core State persistent gesichert")
            except Exception as e:
                logger.error(f"[STOP] Core State Speichern fehlgeschlagen: {e}")

        # Core Integrator stoppen
        if self._core_integrator:
            try:
                self._core_integrator.stop()
            except Exception:
                pass

        # CameraManager: Tracker stoppen
        self._cam.stop_tracker()

        # NPU: Alle Modelle freigeben + VDevice schliessen
        self._orchestrator.release_all()

        # IPC cleanup
        self._ipc.cleanup()

        logger.info("M.O.L.O.C.H. Service gestoppt")


    # =========================================================================
    # Status-JSON (baut Dict aus Service-State, schreibt via IPCRouter)
    # =========================================================================

    def _write_status_json(self):
        """Status-JSON zusammenbauen und via IPCRouter schreiben."""
        try:
            with self._fps_lock:
                fps_snapshot = dict(self._fps)
            with self._ctx_lock:
                active_models = list(self._active_ctx.keys())

            status = {
                "scrfd_active": self.scrfd_active,
                "arcface_active": self.arcface_active,
                "yolo_active": self.yolo_active,
                "hand_active": self.hand_active,
                "pose_active": self.pose_active,
                "npu_paused": self._npu_paused,
                "active_models": active_models,
                "autonomous_mode": self._autonomous_mode,
                "manual_mode": self._manual_mode,
                "moloch_has_control": self._moloch_has_control,
                "tentakel_enabled": self._tentakel_enabled,
                "daily_learner_enabled": self._daily_learner.enabled if self._daily_learner else False,
                "learner_flash": self._learner_flash,
                "frame_age": round(time.time() - self._last_frame_write, 1) if self._last_frame_write else -1,
                "frozen_restarts": self._frozen_restart_count,
                "fps": {k: round(v, 1) for k, v in fps_snapshot.items()},
                "thresholds": {
                    "scrfd_conf": self.scrfd_conf_val,
                    "scrfd_nms": self.scrfd_nms_val,
                    "arcface_thresh": self.arcface_thresh_val,
                    "yolo_conf": self.yolo_conf_val,
                },
                "led_markus_on": self._led.markus_on,
                "cloud": self._cloud_state,
                "audio": {
                    "mic_gain": self._saved_mic_gain,
                    "noise_gate_db": self._saved_noise_gate,
                    "agc_enabled": self._saved_agc,
                    "level": self._audio_level,
                },
                "voice": self._voice_pipeline.get_state() if self._voice_pipeline else {},
            }
            if self._perception:
                status["perception"] = self._perception.get_state()
            if self._core_integrator:
                status["core"] = self._core_integrator.get_status_dict()
            self._ipc.write_status(status)
        except Exception:
            pass

    def _poll_panel_cmds(self):
        """Poll for commands from Panel via IPCRouter."""
        while self.running:
            for cmd in self._ipc.poll_commands():
                try:
                    self._execute_panel_cmd(cmd)
                except Exception as e:
                    logger.error(f"[IPC] Command execution failed: {e}")
            time.sleep(0.2)

    def _execute_panel_cmd(self, cmd):
        """Execute a command from the Panel."""
        action = cmd.get('action')
        logger.info(f"[IPC] Panel command: {cmd}")
        if action == 'toggle_model':
            model = cmd.get('model')
            enabled = cmd.get('enabled', False)
            if model:
                self.toggle_model(model, enabled)
        elif action == 'force_models':
            models = cmd.get('models')  # List[str] oder None
            if self._perception:
                self._perception.force_models(models)
                logger.info(f"[IPC] force_models({models})")
        elif action == 'toggle_smart_tracking':
            self._toggle_smart_tracking()
        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
            logger.info(f"[IPC] autonomous={self._autonomous_mode} tentakel={self._tentakel_enabled}")
        elif action == 'reload_face_db':
            self._reload_face_db()
        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            if attr and value is not None and hasattr(self, attr):
                setattr(self, attr, float(value))
                logger.info(f"[IPC] Threshold: {attr} = {float(value):.3f}")
        elif action == 'set_hand_params':
            if self._perception:
                if cmd.get('disable_occlusion'):
                    self._perception._MIN_FACE_STREAK = 999999
                    self._perception._hand_occlusion = False
                    logger.info("[IPC] Hand-Occlusion DEAKTIVIERT")
                else:
                    self._perception._HAND_TIMEOUT = float(cmd.get('timeout', 5.0))
                    self._perception._MIN_FACE_STREAK = int(cmd.get('streak', 3))
                    self._perception._FACE_RECENCY = float(cmd.get('recency', 2.0))
                    logger.info(f"[IPC] Hand params: timeout={self._perception._HAND_TIMEOUT}, "
                                f"streak={self._perception._MIN_FACE_STREAK}, "
                                f"recency={self._perception._FACE_RECENCY}")
        elif action == 'set_audio':
            # Audio-Parameter sofort uebernehmen (Slider-Aenderungen)
            self._saved_mic_gain = float(cmd.get('mic_gain', self._saved_mic_gain))
            self._saved_agc = bool(cmd.get('agc_enabled', self._saved_agc))
            self._saved_noise_gate = float(cmd.get('noise_gate_db', self._saved_noise_gate))
            logger.info(f"[IPC] Audio: gain={self._saved_mic_gain:.2f}, "
                        f"gate={self._saved_noise_gate:.0f}dB, agc={self._saved_agc}")
        elif action == 'mic_test':
            logger.info("[IPC] Mic Test angefordert (noch nicht implementiert)")
        elif action == 'save_settings':
            # Audio + Camera Werte aus Panel uebernehmen
            _au = cmd.get('audio')
            if _au:
                self._saved_mic_gain = float(_au.get('mic_gain', 1.0))
                self._saved_agc = bool(_au.get('agc_enabled', False))
                self._saved_noise_gate = float(_au.get('noise_gate_db', -60.0))
            _cam = cmd.get('camera')
            if _cam:
                self._saved_ptz_speed = float(_cam.get('ptz_speed', 15.0))
                self._saved_led = bool(_cam.get('led_enabled', False))
                self._saved_ir = str(_cam.get('ir_mode', 'Aus'))
            _th = cmd.get('thresholds')
            if _th:
                self.scrfd_conf_val = float(_th.get('scrfd_conf', self.scrfd_conf_val))
                self.scrfd_nms_val = float(_th.get('scrfd_nms', self.scrfd_nms_val))
                self.arcface_thresh_val = float(_th.get('arcface_thresh', self.arcface_thresh_val))
                self.yolo_conf_val = float(_th.get('yolo_conf', self.yolo_conf_val))
            _ho = cmd.get('hand_occlusion')
            if _ho and self._perception:
                self._perception._HAND_TIMEOUT = float(_ho.get('timeout', 5.0))
                self._perception._MIN_FACE_STREAK = int(_ho.get('streak', 3))
                self._perception._FACE_RECENCY = float(_ho.get('recency', 2.0))
            self._save_settings()
        elif action == 'toggle_daily_learner':
            if self._daily_learner:
                enabled = self._daily_learner.toggle()
                logger.info(f"[IPC] DailyLearner: {'AN' if enabled else 'AUS'}")

        elif action == 'toggle_learner_flash':
            self._learner_flash = bool(cmd.get('on', not self._learner_flash))
            logger.info(f"[IPC] Learner Flash: {'AN' if self._learner_flash else 'AUS'}")

        elif action == 'ptz_move':
            direction = cmd.get('direction', '')
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                if direction == 'home':
                    cam.goto_home()
                    logger.info("[IPC] PTZ: goto home")
                elif direction in ('up', 'down', 'left', 'right'):
                    cam.move_manual(direction, speed=0.3)
                    logger.info(f"[IPC] PTZ: move {direction}")
            except Exception as e:
                logger.error(f"[IPC] PTZ move failed: {e}")

        elif action == 'ptz_goto':
            position = cmd.get('position', '')
            positions = {
                'werkstatt': (0.0, -20.0),
                'wohnzimmer': (-90.0, 0.0),
            }
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                if position in positions:
                    pan, tilt = positions[position]
                    cam.move_absolute(pan=pan, tilt=tilt)
                    logger.info(f"[IPC] PTZ: goto {position} ({pan}, {tilt})")
                else:
                    logger.warning(f"[IPC] PTZ: unknown position '{position}'")
            except Exception as e:
                logger.error(f"[IPC] PTZ goto failed: {e}")

        elif action == 'ptz_calibrate':
            try:
                from core.hardware.camera_cloud_bridge import CameraCloudBridgeSync
                bridge = CameraCloudBridgeSync()
                bridge.trigger_ptz_calibration()
                logger.info("[IPC] PTZ: calibration triggered")
            except Exception as e:
                logger.error(f"[IPC] PTZ calibrate failed: {e}")

        elif action == 'cloud_led':
            level = cmd.get('level', 0)
            try:
                if self._cloud and self._cloud.connected:
                    # PT2 weisse LEDs ueber nightVision steuern
                    # Panel: 0=aus, 2=an. Mapping auf set_night() Modi:
                    # 'day' -> IR-only (weiss AUS), 'night' -> Farb-Nacht (weiss AN)
                    night_modes = {0: 'day', 1: 'auto', 2: 'night', 3: 'night'}
                    mode = night_modes.get(int(level), 'day')
                    self._cloud.run(self._cloud.bridge.set_night(mode))
                    self._cloud_state["led_level"] = int(level)
                    logger.info(f"[IPC] LED/Night mode: {mode} (level={level})")
            except Exception as e:
                logger.error(f"[IPC] LED/Night failed: {e}")

        elif action == 'cloud_alarm':
            try:
                if self._cloud and self._cloud.connected:
                    self._alarm_on = not self._alarm_on
                    self._cloud.run(self._cloud.bridge.set_alarm(self._alarm_on))
                    self._cloud_state["alarm_active"] = self._alarm_on
                    logger.info(f"[IPC] Alarm: {'AN' if self._alarm_on else 'AUS'}")
            except Exception as e:
                logger.error(f"[IPC] Alarm failed: {e}")

        elif action == 'snapshot':
            try:
                frame = None
                with self._annotated_lock:
                    if self._annotated_frame is not None:
                        frame = self._annotated_frame.copy()
                if frame is None:
                    with self._frame_lock:
                        if self._latest_frame is not None:
                            frame = self._latest_frame.copy()
                if frame is None:
                    logger.warning("[IPC] Snapshot: Kein Frame verfuegbar")
                else:
                    snap_dir = os.path.expanduser("~/moloch/snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(snap_dir, f"moloch_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    logger.info(f"[IPC] Snapshot gespeichert: {path}")
            except Exception as e:
                logger.error(f"[IPC] Snapshot failed: {e}")

        elif action == 'cloud_status_led':
            try:
                if self._cloud and self._cloud.connected:
                    # Toggle: aktuellen Status invertieren
                    self._status_led_on = not getattr(self, '_status_led_on', False)
                    self._cloud.run(self._cloud.bridge.set_status_led(self._status_led_on))
                    self._cloud_state["status_led"] = self._status_led_on
                    logger.info(f"[IPC] Status LED: {self._status_led_on}")
            except Exception as e:
                logger.error(f"[IPC] Status LED failed: {e}")

        elif action == 'cloud_sync':
            try:
                if self._cloud and self._cloud.connected:
                    params = self._cloud.run(self._cloud.bridge.get_device_params())
                    if params and isinstance(params, dict):
                        # Sonoff CAM-PT2 nightVision: 0=auto, 1=IR(aus), 2=farb-nacht(an)
                        # Panel erwartet: led_level 0=aus, 2=an
                        nv = int(params.get("nightVision", 1))
                        # Mapping: IR(1)->0(aus), auto(0)->0(aus), farb-nacht(2)->2(an)
                        self._cloud_state["led_level"] = 2 if nv == 2 else 0
                        self._cloud_state["alarm_active"] = bool(params.get("alarmNotify", False))
                        self._cloud_state["status_led"] = bool(params.get("sledOnline", False))
                    logger.info(f"[IPC] Cloud sync: nightVision={nv} led_level={self._cloud_state.get('led_level')}")
            except Exception as e:
                logger.error(f"[IPC] Cloud sync failed: {e}")

        # ---- Voice Pipeline Commands ----

        elif action == 'ptt_start':
            if self._voice_pipeline:
                self._voice_pipeline.start_recording()
                logger.info("[IPC] PTT: Aufnahme gestartet")
            if self._core_integrator:
                self._core_integrator.update_input("voice", "voice_activity", 1.0)

        elif action == 'ptt_stop':
            if self._voice_pipeline:
                self._voice_pipeline.stop_recording()
                logger.info("[IPC] PTT: Aufnahme gestoppt, verarbeite...")
            if self._core_integrator:
                self._core_integrator.update_input("voice", "voice_activity", 0.0)

        elif action == 'chat_message':
            text = cmd.get('text', '').strip()
            if text and self._voice_pipeline:
                self._voice_pipeline.process_text_message(text)
                logger.info(f"[IPC] Chat: '{text[:50]}...'")

        elif action == 'toggle_voice_output':
            if self._voice_pipeline:
                enabled = cmd.get('enabled')
                self._voice_pipeline.toggle_voice(enabled)
                logger.info(f"[IPC] Voice Output: {enabled}")

        elif action == 'set_voice':
            voice_id = cmd.get('voice_id', '')
            if voice_id and self._voice_pipeline:
                self._voice_pipeline.set_voice(voice_id)
                logger.info(f"[IPC] Voice: {voice_id}")

        elif action == 'voice_test':
            if self._voice_pipeline:
                text = cmd.get('text', 'Moloch ist online.')
                self._voice_pipeline.test_voice(text)
                logger.info(f"[IPC] Voice Test: '{text[:50]}'")

        elif action == 'voice_reset':
            if self._voice_pipeline:
                self._voice_pipeline.reset_conversation()
                logger.info("[IPC] Voice: Konversation zurueckgesetzt")

    # ----------------------------------------------------------------
    # Settings Persistence
    # ----------------------------------------------------------------
    def _load_settings(self):
        """Lade Settings aus config/settings.json (ueberschreibt Defaults)."""
        if not os.path.exists(SETTINGS_PATH):
            logger.info("[SETTINGS] Keine settings.json vorhanden - verwende Defaults")
            return
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[SETTINGS] Lade settings.json (version={data.get('version', '?')})")
        except Exception as e:
            logger.warning(f"[SETTINGS] settings.json korrupt, verwende Defaults: {e}")
            return

        # Thresholds
        try:
            th = data.get("thresholds", {})
            if "scrfd_conf" in th:
                self.scrfd_conf_val = float(th["scrfd_conf"])
            if "scrfd_nms" in th:
                self.scrfd_nms_val = float(th["scrfd_nms"])
            if "arcface_thresh" in th:
                self.arcface_thresh_val = float(th["arcface_thresh"])
            if "yolo_conf" in th:
                self.yolo_conf_val = float(th["yolo_conf"])
            logger.info(f"[SETTINGS] Thresholds: scrfd={self.scrfd_conf_val}/{self.scrfd_nms_val} "
                        f"arc={self.arcface_thresh_val} yolo={self.yolo_conf_val}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Thresholds-Fehler: {e}")

        # Hand-Occlusion (gespeichert fuer spaeter, Perception Engine existiert noch nicht)
        try:
            ho = data.get("hand_occlusion", {})
            if ho:
                self._hand_occlusion_enabled = bool(ho.get("enabled", False))
                self._saved_hand_timeout = float(ho.get("timeout", 5.0))
                self._saved_hand_streak = int(ho.get("streak", 3))
                self._saved_hand_recency = float(ho.get("recency", 2.0))
                logger.info(f"[SETTINGS] Hand-Occlusion: enabled={self._hand_occlusion_enabled} "
                            f"timeout={self._saved_hand_timeout} "
                            f"streak={self._saved_hand_streak} recency={self._saved_hand_recency}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Hand-Occlusion-Fehler: {e}")

        # Audio
        try:
            au = data.get("audio", {})
            if au:
                self._saved_mic_gain = float(au.get("mic_gain", 1.0))
                self._saved_agc = bool(au.get("agc_enabled", False))
                self._saved_noise_gate = float(au.get("noise_gate_db", -60.0))
                logger.info(f"[SETTINGS] Audio: gain={self._saved_mic_gain} "
                            f"agc={self._saved_agc} gate={self._saved_noise_gate}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Audio-Fehler: {e}")

        # Camera
        try:
            cam = data.get("camera", {})
            if cam:
                self._saved_ptz_speed = float(cam.get("ptz_speed", 15.0))
                self._saved_led = bool(cam.get("led_enabled", False))
                self._saved_ir = str(cam.get("ir_mode", "Aus"))
                logger.info(f"[SETTINGS] Camera: speed={self._saved_ptz_speed} "
                            f"led={self._saved_led} ir={self._saved_ir}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Camera-Fehler: {e}")

        # Learner Flash
        try:
            learner = data.get("learner", {})
            if "flash_enabled" in learner:
                self._learner_flash = bool(learner["flash_enabled"])
                logger.info(f"[SETTINGS] Learner Flash: {self._learner_flash}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Learner-Fehler: {e}")

        # CPU Detectors (Emotion, Age/Gender - Default AUS wegen CPU-Last)
        try:
            cd = data.get("cpu_detectors", {})
            self._cpu_detectors_enabled = bool(cd.get("enabled", False))
            self._cpu_detect_interval = int(cd.get("interval_frames", 30))
            logger.info(f"[SETTINGS] CPU Detectors: enabled={self._cpu_detectors_enabled} "
                        f"interval={self._cpu_detect_interval}")
        except Exception as e:
            logger.warning(f"[SETTINGS] CPU-Detectors-Fehler: {e}")

        # Aktive Modelle (fuer force_models nach Perception-Init)
        try:
            am = data.get("active_models")
            if am and isinstance(am, list):
                self._saved_active_models = list(am)
                logger.info(f"[SETTINGS] Active Models: {self._saved_active_models}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Active-Models-Fehler: {e}")

    def _save_settings(self):
        """Speichere aktuelle Settings nach config/settings.json (atomic write)."""
        data = {"version": 1}

        # Thresholds
        data["thresholds"] = {
            "scrfd_conf": round(self.scrfd_conf_val, 3),
            "scrfd_nms": round(self.scrfd_nms_val, 3),
            "arcface_thresh": round(self.arcface_thresh_val, 3),
            "yolo_conf": round(self.yolo_conf_val, 3),
        }

        # Hand-Occlusion
        if self._perception:
            data["hand_occlusion"] = {
                "enabled": self._hand_occlusion_enabled,
                "timeout": round(self._perception._HAND_TIMEOUT, 1),
                "streak": self._perception._MIN_FACE_STREAK,
                "recency": round(self._perception._FACE_RECENCY, 1),
            }

        # Audio (aus gespeicherten Werten oder Defaults)
        data["audio"] = {
            "mic_gain": round(getattr(self, '_saved_mic_gain', 1.0), 2),
            "agc_enabled": getattr(self, '_saved_agc', False),
            "noise_gate_db": round(getattr(self, '_saved_noise_gate', -60.0), 1),
        }

        # Camera
        data["camera"] = {
            "ptz_speed": round(getattr(self, '_saved_ptz_speed', 15.0), 1),
            "led_enabled": getattr(self, '_saved_led', False),
            "ir_mode": getattr(self, '_saved_ir', "Aus"),
        }

        # Learner
        data["learner"] = {
            "flash_enabled": self._learner_flash,
        }

        # Aktive Modelle (fuer Wiederherstellung nach Restart)
        data["active_models"] = [name for name in self._active_ctx.keys()]

        # Atomic write
        try:
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, SETTINGS_PATH)
            logger.info(f"[SETTINGS] Gespeichert: {SETTINGS_PATH}")
        except Exception as e:
            logger.error(f"[SETTINGS] Speichern fehlgeschlagen: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)

    print("M.O.L.O.C.H. Core Service - Phase 2")
    service = MolochService()
    service.init()
    service.start(blocking=True)
