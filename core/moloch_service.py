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
import gc
import struct
import asyncio
import threading
import logging
import subprocess
import traceback

import cv2
import numpy as np

# Moloch path
sys.path.insert(0, os.path.expanduser("~/moloch"))

from hailo_platform import HEF, VDevice, FormatType
from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms, decode_yolov8_pose,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_poses, draw_hands, enforce_draw_priority,
    decode_hand_landmark, draw_hand_landmarks,
    estimate_head_pose,
)
from core.hardware.hailo_manager import get_hailo_manager
from core.vision.gesture_detector import GestureDetector, KeypointPosition
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
from core.mpo.autonomous_tracker import AutonomousTracker, TrackerState

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MolochService")
logger.setLevel(logging.INFO)

# RTSP URL
RTSP_URL = os.environ.get(
    "MOLOCH_RTSP_URL",
    "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
)

# Modell-Pfade auf SSD2
MODEL_DIR = "/mnt/moloch-data/hailo/models"
MODEL_PATHS = {
    "scrfd": f"{MODEL_DIR}/scrfd_10g.hef",
    "arcface": f"{MODEL_DIR}/arcface_mobilefacenet.hef",
    "yolov8m": f"{MODEL_DIR}/yolov8m_h10.hef",
    "pose": f"{MODEL_DIR}/yolov8s_pose_h10.hef",
    "hand_landmark": f"{MODEL_DIR}/hand_landmark_lite.hef",
}

FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")
FACE_STATE_PATH = "/tmp/moloch_face_state.json"
NPU_VOICE_REQUEST = "/tmp/moloch_npu_voice_request"
NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"
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


class CloudController:
    """Persistent async eWeLink cloud controller."""

    def __init__(self):
        self.bridge = None
        self.loop = None
        self._thread = None
        self.connected = False

    def start(self):
        config = CloudConfig(
            enabled=True,
            api_base_url="https://eu-apia.coolkit.cc",
            app_id=os.environ.get("EWELINK_APP_ID_1", ""),
            app_secret=os.environ.get("EWELINK_APP_SECRET_1", ""),
            device_id="1002817609",
            username=os.environ.get("EWELINK_USERNAME", ""),
            password=os.environ.get("EWELINK_PASSWORD", ""),
        )
        self.bridge = CameraCloudBridge(config)

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.2)
        future = asyncio.run_coroutine_threadsafe(self.bridge.connect(), self.loop)
        try:
            self.connected = future.result(timeout=10)
        except Exception as e:
            logger.error(f"Cloud connect failed: {e}")
            self.connected = False

    def run(self, coro):
        if not self.loop:
            return False
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Cloud call failed: {e}")
            return False


class MolochService:
    """
    M.O.L.O.C.H. Headless Backend Service.

    Enthaelt alle Logik die OHNE GUI laufen kann:
    - NPU Inference Pipeline
    - Tentakel-Modus (Takeover/Release)
    - Kamera-Kontrolle
    - Smart Tracking Toggle
    - Model Swap (ArcFace/YOLOv8m)

    GUI-Aufrufe (root.after, BooleanVar) sind durch
    self._notify() Callbacks ersetzt.
    """

    PREVIEW_W = 640
    PREVIEW_H = 480

    def __init__(self):
        # State
        self.running = True
        self._hailo_manager = None
        self._vdevice = None
        self._models = {}
        self._output_names = {}
        self._face_db = {}

        # Emotion Detection (CPU, kein NPU)
        self._emotion_detector = None
        try:
            from core.vision.emotion_detector import get_emotion_detector
            self._emotion_detector = get_emotion_detector()
            if self._emotion_detector and self._emotion_detector.available:
                logger.info("[INIT] Emotion Detection bereit (FER+ CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Emotion Detection nicht verfuegbar: {e}")

        # Age + Gender Detection (CPU, kein NPU)
        self._age_gender_detector = None
        try:
            from core.vision.age_gender_detector import get_age_gender_detector
            self._age_gender_detector = get_age_gender_detector()
            if self._age_gender_detector and self._age_gender_detector.available:
                logger.info("[INIT] Age+Gender Detection bereit (Caffe CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Age+Gender Detection nicht verfuegbar: {e}")

        # Gesture Detection (aus Pose-Keypoints)
        self._gesture_detector = GestureDetector()
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
            if hasattr(self, '_saved_hand_timeout'):
                self._perception._HAND_TIMEOUT = self._saved_hand_timeout
                self._perception._MIN_FACE_STREAK = self._saved_hand_streak
                self._perception._FACE_RECENCY = self._saved_hand_recency
                logger.info(f"[SETTINGS] Hand-Occlusion Params aus settings.json angewendet")
        except Exception as e:
            logger.warning(f"[INIT] Perception Engine nicht verfuegbar: {e}")

        # Daily Learner
        self._daily_learner = None
        try:
            from core.daily_learner import get_daily_learner
            self._daily_learner = get_daily_learner()
            logger.info("[INIT] DailyLearner bereit")
        except Exception as e:
            logger.warning(f"[INIT] DailyLearner nicht verfuegbar: {e}")

        # Persistent Model Contexts
        self._active_ctx = {}
        self._ctx_lock = threading.Lock()
        self._input_640 = np.empty((640, 640, 3), dtype=np.uint8)

        # TTS Announcement Cooldown
        self._last_announce = {}

        # Cross-process NPU pause
        self._paused_models = []
        self._npu_paused = False

        # Configure Event
        self._configuring = threading.Event()
        self._configuring.set()



        # Autonomous Tracking
        self._autonomous_mode = False
        self._manual_autonomous = False
        self._manual_mode = False  # MANUELL: Service beobachtet, aber keine Kamera-Kontrolle
        self._tracker = None
        self._models_preloaded = False  # Idle Pre-Load Guard

        # Guardian/Tentakel Mode
        self._guardian_mode = True
        self._tentakel_enabled = True  # Tentakel-Modus aktiv (Status-Flag)
        self._moloch_has_control = False
        self._takeover_reason = ""
        self._takeover_time = 0
        self._last_interesting_time = 0
        self._search_start_time = 0
        self.TAKEOVER_TIMEOUT = 30
        self.SEARCH_TIMEOUT = 20
        self._guardian_last_pan = None
        self._guardian_last_tilt = None
        self._guardian_move_thresh = 5.0
        self._guardian_move_count = 0
        self._guardian_move_required = 2
        self._takeover_cooldown_until = 0
        self.RELEASE_COOLDOWN = 60
        self.MAX_COOLDOWN = 180
        self.STARTUP_GRACE = 15
        self._failed_takeovers = 0
        self._takeover_found_something = False
        self._takeover_cooldown_until = time.time() + self.STARTUP_GRACE
        self._transitioning = False
        self._transition_lock = threading.Lock()
        self._waiting_for_first_detection = False
        self._first_detection_event = threading.Event()

        # Home Position (fuer Release -> Home -> ST)
        self._home_position = {"pan": 0.0, "tilt": -15.0}
        try:
            _home_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "camera_home.json")
            if os.path.exists(_home_cfg):
                with open(_home_cfg) as f:
                    self._home_position = json.load(f)
                logger.info(f"[INIT] Home Position geladen: {self._home_position}")
        except Exception as e:
            logger.debug(f"[INIT] camera_home.json nicht geladen: {e}")

        # Frame Locks
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        # Frozen Frame Watchdog
        self._last_frame_write = time.time()
        self._frozen_restart_count = 0

        # Model enable flags (plain bools, NOT tk.BooleanVar)
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.pose_active = False
        self.hand_active = False

        # Watchdog: Anti-Oszillation Swap-Log
        self._swap_log = []
        # Auto-Switch: Zaehlt Frames ohne Hand-Detection
        self._hand_no_detect = 0
        self._HAND_RELEASE_FRAMES = 75  # ~5s bei 15fps

        # Threshold values (plain floats, NOT tk.DoubleVar)
        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50
        self.pose_conf_val = 0.50
        self.pose_nms_val = 0.70

        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # FPS Tracking
        self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "pose": 0, "hand_landmark": 0, "total": 0}
        self._fps_lock = threading.Lock()

        # Smart Tracking State
        self._smart_tracking_on = False
        self._st_lock = threading.Lock()

        # Cloud Controller
        self._cloud = None
        self._has_calibrated = False

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
    # RTSP Capture
    # =========================================================================

    def _start_rtsp(self):
        """Starte RTSP Background Reader."""
        def rtsp_reader():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            self._rtsp_cap = cap  # Fuer Watchdog-Zugriff
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self._update_status(f"RTSP FEHLER: {RTSP_URL}")
                return

            self._update_status("RTSP aktiv")

            while self.running:
                grabbed = cap.grab()
                if grabbed:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))
                        with self._frame_lock:
                            self._latest_frame = frame
                else:
                    time.sleep(0.1)

            cap.release()

        threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader").start()

    # =========================================================================
    # NPU Pipeline
    # =========================================================================

    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        if name in self._active_ctx:
            logger.info(f"[CONFIGURE] {name}: bereits konfiguriert, skip")
            return
        if name not in self._models:
            logger.warning(f"[CONFIGURE] {name}: nicht in self._models")
            return

        infer_model = self._models[name]
        out_names = self._output_names[name]

        active_before = list(self._active_ctx.keys())
        logger.info(f"[CONFIGURE] {name}: aktive Modelle VORHER: {active_before}")

        # Inference pausieren - NPU kann nicht configure + run gleichzeitig
        self._configuring.clear()
        time.sleep(0.15)

        try:
            ctx_mgr = infer_model.configure()
            configured = ctx_mgr.__enter__()

            output_buffers = {
                oname: np.empty(infer_model.output(oname).shape, dtype=np.float32)
                for oname in out_names
            }
            bindings = configured.create_bindings(output_buffers=output_buffers)

            with self._ctx_lock:
                self._active_ctx[name] = {
                    "ctx_mgr": ctx_mgr,
                    "configured": configured,
                    "bindings": bindings,
                    "output_buffers": output_buffers,
                    "out_names": out_names,
                }

            active_after = list(self._active_ctx.keys())
            logger.info(f"[CONFIGURE] {name}: OK. Aktive Modelle NACHHER: {active_after}")
        except Exception as e:
            crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
            crash_info = (
                f"\n{'='*60}\n"
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CONFIGURE CRASH: {name}\n"
                f"Aktive Modelle vorher: {active_before}\n"
                f"Alle geladenen Modelle: {list(self._models.keys())}\n"
                f"Exception: {type(e).__name__}: {e}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
                f"{'='*60}\n"
            )
            logger.error(crash_info)
            try:
                with open(crash_log, "a", encoding="utf-8") as f:
                    f.write(crash_info)
            except Exception:
                pass
            self._update_status(f"CRASH: {name} ({type(e).__name__})")
        finally:
            self._configuring.set()

    def _unconfigure_model(self, name):
        """Gib Modell-Konfiguration frei."""
        self._configuring.clear()
        time.sleep(0.1)
        try:
            with self._ctx_lock:
                ctx = self._active_ctx.pop(name, None)

            if ctx:
                try:
                    ctx["ctx_mgr"].__exit__(None, None, None)
                except Exception:
                    pass
                logger.info(f"Modell freigegeben: {name}")
        finally:
            self._configuring.set()

    def _run_model(self, name, input_data):
        """Fuehre Modell aus mit persistenter Konfiguration (~21ms statt ~450ms).

        Returns: Dict mit Output-Name -> numpy array
        """
        with self._ctx_lock:
            ctx = self._active_ctx.get(name)
        if not ctx:
            return {}

        bindings = ctx["bindings"]
        bindings.input().set_buffer(np.ascontiguousarray(input_data))
        ctx["configured"].run([bindings], timeout=10000)

        return {oname: ctx["output_buffers"][oname].copy()
                for oname in ctx["out_names"]}

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
            if os.path.exists(NPU_VOICE_REQUEST):
                # Stale File Check: PID noch am Leben?
                try:
                    with open(NPU_VOICE_REQUEST, "r") as f:
                        req = json.load(f)
                    voice_pid = req.get("pid", 0)
                    if voice_pid and not os.path.exists(f"/proc/{voice_pid}"):
                        logger.warning(f"[NPU_IPC] Stale voice request von PID {voice_pid} - aufgeraeumt")
                        try:
                            os.unlink(NPU_VOICE_REQUEST)
                        except FileNotFoundError:
                            pass
                        continue
                except (json.JSONDecodeError, FileNotFoundError):
                    try:
                        os.unlink(NPU_VOICE_REQUEST)
                    except FileNotFoundError:
                        pass
                    continue
                if not self._npu_paused:
                    try:
                        self._pause_for_voice()
                    except Exception as e:
                        logger.error(f"[NPU_IPC] Pause failed: {e}")
                        self._npu_paused = True  # Force flag so resume can fire
                time.sleep(0.1)
                continue
            if self._npu_paused:
                try:
                    self._resume_after_voice()
                except Exception as e:
                    logger.error(f"[NPU_IPC] Resume crashed: {e}")
                    self._npu_paused = False
                continue

            # Safety: models empty but not paused = dead state -> auto-recover
            if not self._models and not self._npu_paused:
                if not hasattr(self, '_recovery_count'):
                    self._recovery_count = 0
                self._recovery_count += 1
                if self._recovery_count <= 3:
                    logger.warning(f"[NPU] Models empty (attempt {self._recovery_count}/3) - auto-recovery...")
                    try:
                        if self._hailo_manager and not self._hailo_manager.is_device_free():
                            logger.info("[NPU] Device busy - waiting 2s...")
                            time.sleep(2)
                        if self._hailo_manager:
                            self._hailo_manager.acquire_for_vision(timeout=10.0)
                        self._reload_models()
                        for name in (self._paused_models or []):
                            if name in self._models:
                                self._configure_model(name)
                        logger.info(f"[NPU] Auto-recovery OK: {list(self._models.keys())}")
                        self._recovery_count = 0
                    except Exception as e:
                        logger.error(f"[NPU] Auto-recovery failed: {e}")
                        if self._vdevice:
                            try:
                                del self._vdevice
                            except Exception:
                                pass
                            self._vdevice = None
                        self._models.clear()
                        gc.collect()
                        time.sleep(5)
                    continue
                elif self._recovery_count == 4:
                    logger.error("[NPU] Auto-recovery exhausted (3 attempts) - running without NPU")
                    self._update_status("NPU: Recovery fehlgeschlagen")
                # After 3 failures, just run without models (no spam)
                time.sleep(1)
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

            # === NPU WATCHDOG: Max-2 + Anti-Oszillation ===
            self._npu_watchdog()
            self._last_hand_detected = False  # Default: keine Hand pro Frame

            # Kein Modell konfiguriert ODER Inference pausiert -> Raw-Frame
            any_active = bool(self._active_ctx) and (
                self.scrfd_active or self.yolo_active or self.pose_active or self.hand_active)
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
                        _want = set(_new_slots)
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
                                "yolov8m": self.yolo_active, "pose": self.pose_active,
                                "hand_landmark": self.hand_active})
                            continue
                with self._annotated_lock:
                    self._annotated_frame = frame.copy()
                self._write_shm(frame)
                time.sleep(0.03)
                continue

            t_total = time.perf_counter()
            annotated = frame.copy()
            fh, fw = frame.shape[:2]

            # Preprocessing: Resize auf 640x640 fuer Modelle
            input_640 = cv2.resize(frame, (640, 640))
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            scale_x = fw / 640.0
            scale_y = fh / 640.0

            # Max-2 Draw-Priority: face > pose > hand
            _draw_candidates = []
            if self.scrfd_active:
                _draw_candidates.append("face")
            if self.pose_active:
                _draw_candidates.append("pose")
            if self.hand_active or self.pose_active:
                _draw_candidates.append("hand")
            _allowed_draws = set(enforce_draw_priority(_draw_candidates))

            face_boxes = []
            face_detected = False
            face_fed_to_tracker = False

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

                            # Emotion Detection (CPU)
                            emotion = None
                            if self._emotion_detector and crop is not None:
                                try:
                                    emotion, _ = self._emotion_detector.detect(crop)
                                except Exception:
                                    pass

                            # Age + Gender Detection (CPU)
                            gender, age_range = None, None
                            if self._age_gender_detector and crop is not None:
                                try:
                                    gender, age_range, _ = self._age_gender_detector.detect(crop)
                                except Exception:
                                    pass

                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range,
                                                   head_pose=_head_pose if '_head_pose' in dir() else None)

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
                except Exception as e:
                    logger.error(f"ArcFace Fehler: {e}")

            # 3. YOLOv8m Person Detection (uebersprungen wenn Face erkannt)
            if self.yolo_active and "yolov8m" in self._active_ctx and not face_detected:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("yolov8m", input_rgb)
                    out_key = self._output_names["yolov8m"][0]
                    persons = decode_yolov8_nms(
                        outputs[out_key],
                        class_id=0,
                        conf_thresh=self.yolo_conf_val
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["yolov8m"] = 1.0 / dt if dt > 0 else 0

                    if persons:
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

            # 4. YOLOv8s Pose
            if self.pose_active and "pose" in self._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("pose", input_rgb)
                    poses = decode_yolov8_pose(
                        outputs, img_h=640, img_w=640,
                        conf_thresh=self.pose_conf_val,
                        iou_thresh=self.pose_nms_val
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["pose"] = 1.0 / dt if dt > 0 else 0

                    if poses:
                        if "pose" in _allowed_draws:
                            draw_poses(annotated, poses, scale_x, scale_y)
                        if "hand" in _allowed_draws:
                            draw_hands(annotated, poses, scale_x, scale_y)

                        # === Hand Landmark: Crop um Wrists, 21 Finger-Landmarks ===
                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:
                                _kpts = _pose["keypoints"]  # (17, 3) in 640-Space
                                for _wi in (9, 10):  # left/right wrist
                                    _wx = _kpts[_wi, 0]
                                    _wy = _kpts[_wi, 1]
                                    _wvis = _kpts[_wi, 2]
                                    if _wvis < 0.3:
                                        continue

                                    # Elbow-Index: wrist 9->elbow 7, wrist 10->elbow 8
                                    _ei = _wi - 2
                                    _ex = _kpts[_ei, 0]
                                    _ey = _kpts[_ei, 1]
                                    _evis = _kpts[_ei, 2]

                                    # Crop-Groesse: kompakt um Hand (nicht zu gross -> Kopf!)
                                    _pbx = _pose["bbox"]
                                    _pw = _pbx[2] - _pbx[0]
                                    _ph = _pbx[3] - _pbx[1]
                                    _csz = max(int(max(_pw, _ph) * 0.35), 120)
                                    _csz = min(_csz, 220)

                                    # Crop-Zentrum: Wrist + 25% Offset in Fingerrichtung
                                    _ccx = _wx
                                    _ccy = _wy
                                    if _evis > 0.2:
                                        _dx = _wx - _ex
                                        _dy = _wy - _ey
                                        _dist = max((_dx**2 + _dy**2)**0.5, 1.0)
                                        _off = _csz * 0.25
                                        _ccx = _wx + (_dx / _dist) * _off
                                        _ccy = _wy + (_dy / _dist) * _off

                                    # Crop-Region (640x640 Space)
                                    _cx1 = max(0, int(_ccx - _csz // 2))
                                    _cy1 = max(0, int(_ccy - _csz // 2))
                                    _cx2 = min(640, _cx1 + _csz)
                                    _cy2 = min(640, _cy1 + _csz)
                                    _cw = _cx2 - _cx1
                                    _ch = _cy2 - _cy1
                                    if _cw < 30 or _ch < 30:
                                        continue
                                    # Crop + Resize fuer hand_landmark_lite (224x224 RGB)
                                    _hand_crop = cv2.resize(
                                        input_rgb[_cy1:_cy2, _cx1:_cx2], (224, 224))
                                    _hand_out = self._run_model("hand_landmark", _hand_crop)
                                    _hand_res = decode_hand_landmark(_hand_out)
                                    if _hand_res:
                                        self._last_hand_detected = True
                                    if _hand_res and "hand" in _allowed_draws:
                                        draw_hand_landmarks(
                                            annotated, _hand_res,
                                            _cx1, _cy1, _cw, _ch,
                                            scale_x, scale_y)


                        # Gesten-Erkennung aus Pose-Keypoints
                        if self._gesture_detector:
                            try:
                                best_pose = poses[0]
                                kpts = best_pose.get("keypoints")
                                if kpts is not None and len(kpts) >= 17:
                                    kp_list = [
                                        KeypointPosition(
                                            x=float(kpts[i][0]) / 640.0,
                                            y=float(kpts[i][1]) / 640.0,
                                            confidence=float(kpts[i][2])
                                        )
                                        for i in range(17)
                                    ]
                                    gesture = self._gesture_detector.detect(kp_list)
                                    self._current_gesture = gesture
                                    if gesture and gesture.type.value != "none":
                                        label = f"GESTE: {gesture.type.value} ({gesture.confidence:.0%})"
                                        cv2.putText(annotated, label, (10, fh - 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            except Exception as e:
                                logger.debug(f"Gesture detection: {e}")

                        if self._autonomous_mode and self._tracker:
                            try:
                                enriched = []
                                for p in poses:
                                    ep = dict(p)
                                    kpts = p.get("keypoints")
                                    if kpts is not None and len(kpts) >= 17:
                                        face_vis = (float(kpts[0][2]) + float(kpts[1][2]) + float(kpts[2][2])) / 3
                                        if face_vis > 0.5:
                                            ep["has_face"] = True
                                            ep["face_confidence"] = face_vis
                                            ep["face_center"] = (float(kpts[0][0]) / 640, float(kpts[0][1]) / 640)
                                        else:
                                            ep["has_face"] = False
                                            ep["face_confidence"] = 0
                                        torso_vis = (float(kpts[5][2]) + float(kpts[6][2]) + float(kpts[11][2]) + float(kpts[12][2])) / 4
                                        ep["has_torso"] = torso_vis > 0.3
                                    else:
                                        ep["has_face"] = False
                                        ep["face_confidence"] = 0
                                        ep["has_torso"] = True
                                    enriched.append(ep)
                                self._tracker.update_pose_detection(
                                    poses=enriched,
                                    frame_width=640, frame_height=640
                                )
                            except Exception as e:
                                logger.debug(f"Tracker pose feed: {e}")
                except Exception as e:
                    logger.error(f"Pose Fehler: {e}")

            # ===== Perception Engine: Dual-Slot Empfehlung (nach allen Detektionen) =====
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
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                    "gesture": self._current_gesture.type.value if self._current_gesture else "none",
                }
                _new_slots = self._perception.tick(_perc_ctx)
                if _new_slots:
                    _want = set(_new_slots)
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
                            "yolov8m": self.yolo_active, "pose": self.pose_active,
                            "hand_landmark": self.hand_active})

            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:
                if self._last_hand_detected:
                    self._hand_no_detect = 0
                else:
                    self._hand_no_detect += 1
                    if self._hand_no_detect >= self._HAND_RELEASE_FRAMES:
                        logger.info(f"[AUTO-SWITCH] {self._HAND_RELEASE_FRAMES} Frames keine Hand -> Auto-Scoring")
                        self._perception.force_models(None)
                        self._hand_no_detect = 0

            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0

            # Hand-Occlusion Overlay auf Video
            if self._perception and self._perception._hand_occlusion:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (fw, 30), (0, 0, 180), -1)
                annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                cv2.putText(annotated, "HAND OCCLUSION", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self._annotated_lock:
                self._annotated_frame = annotated

            # Panel IPC: Frame + Status nach /dev/shm
            self._write_shm(annotated)

    # =========================================================================
    # Cross-Process NPU Coordination
    # =========================================================================

    def _reload_models(self):
        """(Re-)load all HEF models into a fresh VDevice.

        Creates new VDevice and loads all model files from disk.
        Used by init() and _resume_after_voice() for recovery.
        Raises on failure (caller handles retry).
        """
        params = VDevice.create_params()
        self._vdevice = VDevice(params)
        self._models.clear()
        self._output_names.clear()

        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                continue
            hef = HEF(path)
            infer_model = self._vdevice.create_infer_model(path)
            infer_model.input().set_format_type(FormatType.UINT8)
            out_names = [o.name for o in hef.get_output_vstream_infos()]
            for oname in out_names:
                infer_model.output(oname).set_format_type(FormatType.FLOAT32)
            self._models[name] = infer_model
            self._output_names[name] = out_names

        logger.info(f"[NPU] Models loaded: {list(self._models.keys())}")

    def _pause_for_voice(self):
        """Pause inference - release VDevice so voice process can use NPU."""
        logger.info("[NPU_IPC] Voice requested - pausing vision...")
        self._update_status("NPU: Pausiert fuer Sprache...")

        self._models_preloaded = False
        self._paused_models = list(self._active_ctx.keys())

        for name in list(self._active_ctx.keys()):
            self._unconfigure_model(name)

        self._models.clear()
        if self._vdevice:
            try:
                del self._vdevice
            except Exception:
                pass
            self._vdevice = None

        if self._hailo_manager:
            try:
                self._hailo_manager.release_vision()
            except Exception:
                pass

        gc.collect()
        time.sleep(0.3)

        try:
            with open(NPU_VISION_PAUSED, "w") as f:
                json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)
        except Exception:
            pass

        self._npu_paused = True
        logger.info("[NPU_IPC] Vision paused, VDevice released")

    def _resume_after_voice(self):
        """Resume inference after voice process released NPU."""
        logger.info("[NPU_IPC] Voice done - resuming vision...")
        self._update_status("NPU: Wiederherstellung...")

        for path in [NPU_VISION_PAUSED]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        time.sleep(0.5)

        # Wait for Whisper VDevice to be fully released (lsof check)
        if self._hailo_manager:
            logger.info("[NPU_IPC] Waiting for device to be free...")
            for i in range(25):  # 5s max
                if self._hailo_manager.is_device_free():
                    logger.info(f"[NPU_IPC] Device free after {i * 0.2:.1f}s")
                    break
                time.sleep(0.2)
            else:
                logger.warning("[NPU_IPC] Device not free after 5s - forcing GC")
                gc.collect()
                time.sleep(1.0)

            if not self._hailo_manager.acquire_for_vision(timeout=10.0):
                self._update_status("NPU nicht verfuegbar nach Voice!")
                self._npu_paused = False
                return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"[NPU_IPC] Resume retry {attempt + 1}/{max_retries}...")
                    time.sleep(1.0 + attempt)  # 2s, 3s backoff

                self._reload_models()

                for name in self._paused_models:
                    if name in self._models:
                        self._configure_model(name)

                self._npu_paused = False
                self._update_status("RTSP + NPU aktiv")
                logger.info("[NPU_IPC] Vision resumed successfully")
                return
            except Exception as e:
                logger.error(f"[NPU_IPC] Resume attempt {attempt + 1} failed: {e}")
                # Clean up failed VDevice before retry
                if self._vdevice:
                    try:
                        del self._vdevice
                    except Exception:
                        pass
                    self._vdevice = None
                self._models.clear()
                gc.collect()

        # All retries failed
        self._update_status("NPU Resume FEHLGESCHLAGEN nach 3 Versuchen")
        logger.error("[NPU_IPC] Resume failed after all retries - models unavailable")
        self._npu_paused = False

    # =========================================================================
    # Tentakel-Logik (Smart Tracking <-> MOLOCH Takeover)
    # =========================================================================

    def _moloch_takeover(self, reason: str):
        """MOLOCH uebernimmt: NPU Modelle AN -> Warte auf Detection -> ST AUS -> Tracker AN.

        Fliessender Uebergang: Smart Tracking bleibt AN waehrend NPU Modelle laden
        und Frames analysieren. Erst bei echter Detection wird ST abgeschaltet.
        """
        with self._transition_lock:
            if self._moloch_has_control or not self._guardian_mode or self._transitioning or self._manual_mode:
                self._last_interesting_time = time.time()
                return
            self._transitioning = True
        logger.info(f"[TENTAKEL] MOLOCH uebernimmt Kamera: {reason}")
        self._moloch_has_control = True
        self._takeover_time = time.time()
        self._takeover_reason = reason
        self._takeover_found_something = False
        self._last_interesting_time = time.time()
        self._search_start_time = 0

        def do_takeover():
            try:
                # User hat Modelle manuell gewaehlt? -> NPU nicht antasten!
                if self._perception and self._perception._forced:
                    logger.info(f"[TENTAKEL] User forced_models={self._perception._forced} - NPU bleibt!")
                    # Takeover-Flags setzen fuer Kamera-Kontrolle
                    self._sync_flags_from_npu()
                    self._first_detection_event.set()  # Skip Detection-Wait
                    self._transitioning = False
                    return

                # 1. NPU Modelle aktivieren (ST bleibt AN!)
                models_cached = "scrfd" in self._active_ctx and "yolov8m" in self._active_ctx
                if models_cached:
                    logger.info("[TENTAKEL] Modelle bereits auf NPU")
                else:
                    self._update_status("Takeover: NPU Modelle laden...")
                    # Erst Modelle aufräumen die nicht gebraucht werden (max 2!)
                    for _stale in list(self._active_ctx.keys()):
                        if _stale not in ("scrfd", "yolov8m"):
                            logger.info(f"[TENTAKEL] Raeume {_stale} auf (Platz fuer Takeover)")
                            self._unconfigure_model(_stale)
                            time.sleep(0.2)
                    logger.info("[TENTAKEL] Lade NPU Modelle (ST laeuft weiter)")
                    self._configure_model("scrfd")
                    time.sleep(0.2)
                    self._configure_model("yolov8m")

                # 2. Inference starten - Flags aus NPU-Realitaet
                self._sync_flags_from_npu()
                self._notify("model_toggle", {
                    "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                    "yolov8m": self.yolo_active, "pose": self.pose_active,
                    "hand_landmark": self.hand_active})

                # 3. Warte auf erste Detection (max 10s, ST laeuft weiter)
                self._first_detection_event.clear()
                self._waiting_for_first_detection = True
                self._update_status("Takeover: Warte auf Detection...")
                logger.info("[TENTAKEL] NPU aktiv, warte auf Detection (ST laeuft weiter)...")

                got_detection = self._first_detection_event.wait(timeout=10.0)
                self._waiting_for_first_detection = False

                if not got_detection:
                    # Timeout - nichts erkannt, alles zurueck
                    logger.info("[TENTAKEL] 10s keine Detection - Takeover abgebrochen")
                    self.scrfd_active = False
                    self.yolo_active = False
                    self._notify("model_toggle", {"scrfd": False, "yolov8m": False})
                    self._moloch_has_control = False
                    self._takeover_found_something = False
                    # Cooldown: war ein Fehlversuch
                    self._failed_takeovers += 1
                    cooldown = min(self.RELEASE_COOLDOWN * (1.5 ** self._failed_takeovers), self.MAX_COOLDOWN)
                    self._takeover_cooldown_until = time.time() + cooldown
                    self._update_status("Tentakel scannt wieder")
                    logger.info(f"[TENTAKEL] Fehlversuch #{self._failed_takeovers}, Cooldown {cooldown:.0f}s")
                    # ST war NIE aus - kein Toggle noetig!
                    return

                # 4. Detection da! JETZT ST AUS (nahtloser Uebergang)
                logger.info("[TENTAKEL] Detection erkannt! ST AUS + Tracker AN")
                self._update_status("Takeover: ST AUS...")
                st_off = False
                if self._cloud and self._cloud.connected:
                    for attempt in range(3):
                        try:
                            self._cloud.run(self._cloud.bridge.set_smart_tracking(False))
                            self._set_smart_tracking_state(False)
                            st_off = True
                            break
                        except Exception as e:
                            logger.warning(f"[TENTAKEL] ST AUS Versuch {attempt+1}/3: {e}")
                            time.sleep(0.5)

                if not st_off:
                    logger.error("[TENTAKEL] ST AUS fehlgeschlagen - Takeover ABBRUCH")
                    self.scrfd_active = False
                    self.yolo_active = False
                    self._notify("model_toggle", {"scrfd": False, "yolov8m": False})
                    self._moloch_has_control = False
                    self._update_status("Takeover abgebrochen: ST nicht erreichbar")
                    return

                # 5. Tracker AN (Detection bereits vorhanden -> sofortige Uebernahme!)
                self._enable_autonomous()

                # 6. LED AN
                self._led_on()

                self._update_status(f"MOLOCH: {reason}")
                logger.info(f"[TENTAKEL] Takeover komplett (fliessend): {reason}")
            except Exception as e:
                logger.error(f"[TENTAKEL] Takeover Fehler: {e}")
                self._moloch_has_control = False
            finally:
                self._waiting_for_first_detection = False
                self._transitioning = False

        threading.Thread(target=do_takeover, daemon=True).start()

    def _moloch_release(self):
        """MOLOCH gibt zurueck: Tracker STOP -> ST AN -> Aufraumen."""
        with self._transition_lock:
            if not self._moloch_has_control or self._transitioning:
                return
            self._transitioning = True
        try:
            # Unblock fliessender Takeover falls noch wartend
            self._waiting_for_first_detection = False
            self._first_detection_event.set()

            logger.info("[TENTAKEL] MOLOCH gibt Kamera zurueck an Smart Tracking")

            # VOR dem Release: Kamera auf Home Position fahren
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if cam.is_connected:
                    home_pan = self._home_position.get("pan", 0.0)
                    home_tilt = self._home_position.get("tilt", -15.0)
                    cam.move_absolute(home_pan, home_tilt, speed=15.0)
                    logger.info(f"[RELEASE] Home Position: pan={home_pan}, tilt={home_tilt}")
            except Exception as e:
                logger.debug(f"[RELEASE] Home move failed: {e}")

            self._moloch_has_control = False
            self._takeover_reason = ""
            self._search_start_time = 0

            # 1. LED AUS (MOLOCH gibt ab)
            self._led_off()

            # 2. Tracker SOFORT stoppen
            self._autonomous_mode = False
            if self._tracker:
                self._tracker.disable()
            logger.info("[TENTAKEL] Tracker gestoppt")
            self._notify("auto_mode", {"state": "disabled"})

            # 3. Smart Tracking SOFORT AN (minimaler Gap!)
            if self._cloud and self._cloud.connected:
                try:
                    self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                    self._set_smart_tracking_state(True)
                    logger.info("[TENTAKEL] Smart Tracking wiederhergestellt")
                except Exception:
                    pass

            # 4. Inference-Flags: Wenn User Modelle forced hat, Flags aus NPU synchen!
            if self._perception and self._perception._forced:
                self._sync_flags_from_npu()
                logger.info(f"[TENTAKEL] User forced={self._perception._forced} - Flags aus NPU: {list(self._active_ctx.keys())}")
            else:
                self.scrfd_active = False
                self.arcface_active = False
                self.yolo_active = False
                self.pose_active = False
                self.hand_active = False
            self._notify("model_toggle", {
                "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                "yolov8m": self.yolo_active, "pose": self.pose_active,
                "hand_landmark": self.hand_active})
            with self._fps_lock:
                self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "pose": 0, "hand_landmark": 0, "total": 0}
            logger.info(f"[TENTAKEL] Inference gestoppt, Modelle auf NPU: {list(self._active_ctx.keys())}")

            # Position-Tracking zuruecksetzen
            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0

            # Progressive Backoff (1.5x, max 180s)
            if self._takeover_found_something:
                self._failed_takeovers = 0
                cooldown = self.RELEASE_COOLDOWN
            else:
                self._failed_takeovers += 1
                cooldown = min(self.RELEASE_COOLDOWN * (1.5 ** self._failed_takeovers), self.MAX_COOLDOWN)
            self._takeover_found_something = False
            self._takeover_cooldown_until = time.time() + cooldown

            self._update_status("Tentakel scannt wieder")
            logger.info(f"[TENTAKEL] Release komplett - Cooldown {cooldown:.0f}s")
        finally:
            self._transitioning = False

    def _check_guardian_timeout(self):
        """Pruefe ob MOLOCH die Kamera zurueckgeben soll (kein Interest mehr)."""
        if not self._guardian_mode or self._transitioning or self._manual_mode:
            return
        # Safety: verwaister autonomer Modus
        if self._autonomous_mode and not self._moloch_has_control and not self._manual_autonomous:
            logger.warning("[SAFETY] Orphaned autonomous mode detected - disabling")
            self._disable_autonomous()
            return
        if not self._moloch_has_control:
            return
        now = time.time()
        # Timeout: zu lange nichts Interessantes
        if now - self._last_interesting_time > self.TAKEOVER_TIMEOUT:
            logger.info(f"[TENTAKEL] Takeover timeout ({self.TAKEOVER_TIMEOUT}s) - zurueckgeben")
            threading.Thread(target=self._moloch_release, daemon=True).start()
            return
        # Tracker sucht zu lange ohne Ergebnis
        if self._tracker and self._autonomous_mode:
            if self._tracker.state == TrackerState.SEARCHING:
                if self._search_start_time == 0:
                    self._search_start_time = now
                elif now - self._search_start_time > self.SEARCH_TIMEOUT:
                    logger.info(f"[TENTAKEL] Search timeout ({self.SEARCH_TIMEOUT}s) - zurueckgeben")
                    threading.Thread(target=self._moloch_release, daemon=True).start()
                    return
            else:
                self._search_start_time = 0

    # =========================================================================
    # Kamera-Status Polling (ersetzt root.after(3000, ...))
    # =========================================================================

    def _cam_status_loop(self):
        """Kamera-Status polling loop (1.5s Intervall fuer schnelle Reaktion)."""
        while self.running:
            try:
                self._update_cam_status()
            except Exception as e:
                logger.error(f"Cam status error: {e}")
            time.sleep(1.5)

    def _update_cam_status(self):
        """Kamera-Status pruefen + Tentakel-Bewegungserkennung."""
        self._check_guardian_timeout()

        onvif_ok = False
        ptz_text = "--"
        try:
            from core.hardware.camera import get_camera_controller
            cam = get_camera_controller()
            if not cam.is_connected:
                cam.connect()
            if cam.is_connected:
                onvif_ok = True
                pos = cam.get_position()
                if pos:
                    pan, tilt = pos.pan, pos.tilt
                    ptz_text = f"Pan: {pan:.1f}  Tilt: {tilt:.1f}"

                    # Tentakel: Kamera-Bewegung erkennen
                    if (self._guardian_mode and self._smart_tracking_on
                            and not self._moloch_has_control
                            and not self._transitioning
                            and not self._manual_mode):
                        if self._guardian_last_pan is not None:
                            delta = abs(pan - self._guardian_last_pan) + abs(tilt - self._guardian_last_tilt)
                            if delta > self._guardian_move_thresh:
                                self._guardian_move_count += 1
                                logger.info(f"[TENTAKEL] Bewegung {self._guardian_move_count}/{self._guardian_move_required} delta={delta:.1f}")
                                if self._guardian_move_count >= self._guardian_move_required:
                                    if time.time() >= self._takeover_cooldown_until:
                                        logger.info(f"[TENTAKEL] Sustained movement ({self._guardian_move_count}x) -> MOLOCH uebernimmt")
                                        self._guardian_move_count = 0
                                        self._moloch_takeover("Kamera trackt etwas")
                                    else:
                                        remaining = self._takeover_cooldown_until - time.time()
                                        logger.info(f"[TENTAKEL] Cooldown aktiv, noch {remaining:.0f}s")
                                        self._guardian_move_count = 0
                            else:
                                self._guardian_move_count = max(0, self._guardian_move_count - 1)
                                # Idle Pre-Load: Kamera steht still -> NPU Modelle vorladen
                                if (not self._models_preloaded
                                        and not self._active_ctx
                                        and time.time() >= self._takeover_cooldown_until
                                        and self._configuring.is_set()):
                                    self._models_preloaded = True  # Guard: nur einmal
                                    def _idle_preload():
                                        try:
                                            # Erst Modelle aufräumen die nicht gebraucht werden (max 2!)
                                            for _stale in list(self._active_ctx.keys()):
                                                if _stale not in ("scrfd", "yolov8m"):
                                                    logger.info(f"[TENTAKEL] Pre-Load: raeume {_stale} auf")
                                                    self._unconfigure_model(_stale)
                                                    time.sleep(0.2)
                                            logger.info("[TENTAKEL] Idle Pre-Load: NPU Modelle vorladen...")
                                            self._configure_model("scrfd")
                                            time.sleep(0.2)
                                            self._configure_model("yolov8m")
                                            if "scrfd" in self._active_ctx and "yolov8m" in self._active_ctx:
                                                logger.info("[TENTAKEL] Idle Pre-Load: Modelle ready auf NPU")
                                            else:
                                                logger.warning("[TENTAKEL] Idle Pre-Load: Modelle NICHT konfiguriert!")
                                                self._models_preloaded = False
                                        except Exception as e:
                                            logger.error(f"[TENTAKEL] Idle Pre-Load Fehler: {e}")
                                            self._models_preloaded = False
                                    threading.Thread(target=_idle_preload, daemon=True).start()
                        self._guardian_last_pan = pan
                        self._guardian_last_tilt = tilt
        except Exception:
            pass

        # Status Notification
        smart = "AUS" if not self._smart_tracking_on else "AN"
        onvif_str = "OK" if onvif_ok else "---"

        if self._manual_mode:
            mode = "manual"
            ctrl_text = "MANUELL"
        elif self._moloch_has_control:
            mode = "moloch"
            ctrl_text = f"MOLOCH: {self._takeover_reason[:20]}"
        elif self._smart_tracking_on:
            mode = "tentakel"
            ctrl_text = "TENTAKEL SCANNT"
        elif onvif_ok:
            mode = "manual"
            ctrl_text = "MANUELL"
        else:
            mode = "offline"
            ctrl_text = "OFFLINE"

        self._notify("cam_status", {
            "mode": mode, "ctrl_text": ctrl_text,
            "smart": smart, "onvif": onvif_str, "ptz": ptz_text,
            "frame_age": round(time.time() - self._last_frame_write, 1),
        })

    # =========================================================================
    # Frozen Frame Watchdog
    # =========================================================================

    def _frozen_frame_watchdog(self):
        """Erkennt eingefrorene Frames und startet RTSP Stream neu."""
        while self.running:
            try:
                time.sleep(10)  # Alle 10 Sekunden pruefen

                frame_age = time.time() - self._last_frame_write

                if frame_age > 30:  # Frame aelter als 30 Sekunden
                    self._frozen_restart_count += 1
                    logger.warning(
                        f"[WATCHDOG] Frame eingefroren seit {frame_age:.0f}s! "
                        f"Restart #{self._frozen_restart_count}"
                    )

                    # RTSP Stream neu verbinden
                    try:
                        if hasattr(self, '_rtsp_cap') and self._rtsp_cap is not None:
                            try:
                                self._rtsp_cap.release()
                            except Exception:
                                pass
                        self._start_rtsp()
                        logger.info("[WATCHDOG] RTSP Stream neu gestartet")
                        self._last_frame_write = time.time()
                    except Exception as e:
                        logger.error(f"[WATCHDOG] RTSP Reconnect Error: {e}")

                    # Max 5 Versuche, danach loggen und warten
                    if self._frozen_restart_count >= 5:
                        logger.error("[WATCHDOG] 5 Reconnects fehlgeschlagen, warte 60s")
                        time.sleep(60)
                        self._frozen_restart_count = 0

            except Exception as e:
                logger.error(f"[WATCHDOG] Error: {e}")

    # =========================================================================
    # Autonomous Mode
    # =========================================================================

    def _enable_autonomous(self):
        """AUTONOM aktivieren (idempotent)."""
        if self._autonomous_mode:
            logger.debug("[AUTONOM] Already enabled, skip")
            return
        self._notify("auto_mode", {"state": "starting"})

        def do_start():
            try:
                from core.mpo.autonomous_tracker import get_autonomous_tracker
                from core.hardware.camera import get_camera_controller, ControlMode
                if not self._tracker:
                    self._tracker = get_autonomous_tracker()
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                if not cam.is_connected:
                    self._update_status("AUTONOM fehlgeschlagen: Kamera offline")
                    self._notify("auto_mode", {"state": "failed"})
                    return
                self._tracker.set_camera(cam)
                cam.set_mode(ControlMode.AUTONOMOUS)
                if not self._tracker._running:
                    self._tracker.start()
                self._tracker.enable()
                self._autonomous_mode = True
                self._update_status("Modus: AUTONOM - MOLOCH sucht...")
                logger.info("Switched to AUTONOMOUS mode")
                self._notify("auto_mode", {"state": "active"})
            except Exception as e:
                logger.error(f"Autonomous start failed: {e}")
                self._update_status(f"AUTONOM Fehler: {e}")
                self._notify("auto_mode", {"state": "failed"})

        threading.Thread(target=do_start, daemon=True).start()

    def _disable_autonomous(self):
        """AUTONOM deaktivieren (idempotent)."""
        if not self._autonomous_mode:
            logger.debug("[AUTONOM] Already disabled, skip")
            return
        self._autonomous_mode = False
        if self._tracker:
            self._tracker.disable()
        self._update_status("Modus: MANUELL")
        logger.info("Switched to MANUAL mode")
        self._notify("auto_mode", {"state": "disabled"})


    def _all_models_off(self):
        """Alle Modelle deaktivieren und unconfigurieren."""
        self._models_preloaded = False
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.pose_active = False
        self.hand_active = False
        self._notify("model_toggle", {"scrfd": False, "arcface": False, "yolov8m": False, "pose": False, "hand_landmark": False})
        for name in list(self._active_ctx.keys()):
            self._unconfigure_model(name)
        with self._fps_lock:
            self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "pose": 0, "hand_landmark": 0, "total": 0}

    # =========================================================================
    # Cloud / Camera
    # =========================================================================

    def _connect_cloud(self):
        """Connect to eWeLink cloud."""
        try:
            self._cloud = CloudController()
            self._cloud.start()
            if self._cloud.connected:
                logger.info("eWeLink Cloud verbunden")
                try:
                    self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                    self._set_smart_tracking_state(True)
                    logger.info("Smart Tracking aktiviert - Kamera scannt autonom (Tentakel-Modus)")
                except Exception:
                    pass
                # LED AUS beim Start (sauberer Zustand)
                try:
                    self._cloud.run(self._cloud.bridge.set_status_led(False))
                except Exception:
                    pass
                self._notify("cloud_status", {"connected": True})
            else:
                logger.warning("eWeLink Cloud nicht erreichbar")
                self._notify("cloud_status", {"connected": False, "error": "nicht erreichbar"})
        except Exception as e:
            logger.error(f"Cloud connect: {e}")
            self._notify("cloud_status", {"connected": False, "error": str(e)})

    def _set_smart_tracking_state(self, value: bool):
        """Einziger Schreibzugriff auf _smart_tracking_on (thread-safe)."""
        with self._st_lock:
            self._smart_tracking_on = value
        self._notify("smart_tracking", {"on": value})

    def _cloud_run(self, method_name, *args):
        """Run cloud bridge method in background."""
        if not self._cloud or not self._cloud.connected:
            self._update_status("Cloud nicht verbunden")
            return
        method = getattr(self._cloud.bridge, method_name, None)
        if not method:
            return
        threading.Thread(
            target=lambda: self._cloud.run(method(*args)),
            daemon=True
        ).start()

    def _toggle_smart_tracking(self):
        """Smart Tracking toggle via persistent cloud connection."""
        new_state = not self._smart_tracking_on
        if not self._cloud or not self._cloud.connected:
            self._update_status("Cloud nicht verbunden")
            return
        def do_toggle():
            try:
                self._cloud.run(self._cloud.bridge.set_smart_tracking(new_state))
                self._set_smart_tracking_state(new_state)
                self._update_status(f"Smart Tracking {'AN' if new_state else 'AUS'}")
            except Exception as e:
                self._update_status(f"Smart Tracking Fehler: {e}")
        threading.Thread(target=do_toggle, daemon=True).start()

    # =========================================================================
    # LED Signaling (Status-LED via eWeLink Cloud)
    # =========================================================================

    def _led_on(self):
        """Status-LED AN (blau, sichtbar)."""
        if not self._cloud or not self._cloud.connected:
            return
        try:
            self._cloud.run(self._cloud.bridge.set_status_led(True))
        except Exception:
            pass

    def _led_off(self):
        """Status-LED AUS."""
        if not self._cloud or not self._cloud.connected:
            return
        try:
            self._cloud.run(self._cloud.bridge.set_status_led(False))
        except Exception:
            pass

    def _led_blink(self, count=6, interval=0.3):
        """Status-LED blinken, danach AN lassen (MOLOCH hat noch Kontrolle)."""
        def do_blink():
            for _ in range(count):
                self._led_off()
                time.sleep(interval)
                self._led_on()
                time.sleep(interval)
        threading.Thread(target=do_blink, daemon=True).start()

    # =========================================================================
    # Face Recognition
    # =========================================================================

    def _reload_face_db(self):
        """Face-DB neu laden (nach Enrollment)."""
        self._face_db = load_face_db(FACE_DB_PATH)
        n = len(self._face_db)
        names = ", ".join(self._face_db.keys()) if self._face_db else "leer"
        self._update_status(f"Face-DB: {n} Personen ({names})")

    def _write_face_state(self, name, similarity, person_count, emotion=None, gender=None, age_range=None, head_pose=None):
        """Schreibe Face-Recognition-State fuer IPC mit push_to_talk."""
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "gender": gender,
                "age_range": age_range,
                "head_pose": {"pitch": head_pose[0], "yaw": head_pose[1], "roll": head_pose[2]} if head_pose else None,
                "timestamp": time.time(),
                "source": "moloch_service"
            }
            with open(FACE_STATE_PATH, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _announce_person(self, name):
        """LED-Signal bei Gesichtserkennung (6x Blink, endet AN)."""
        logger.info(f"[LED] Person erkannt: {name}")
        self._led_blink(6, 0.3)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def init(self):
        """Hardware initialisieren: VDevice, Models, RTSP, Cloud."""
        logger.info("M.O.L.O.C.H. Service initialisiert...")

        # 1. Hailo VDevice + Models
        self._hailo_manager = get_hailo_manager()
        self._hailo_manager.acquire_for_vision(timeout=10.0)
        self._reload_models()
        for name in self._models:
            logger.info(f"Modell geladen: {name} ({len(self._output_names[name])} outputs)")

        # 2. Face DB
        self._face_db = load_face_db(FACE_DB_PATH)
        if self._face_db:
            logger.info(f"Face-DB: {len(self._face_db)} Personen")

        # 3. RTSP
        self._start_rtsp()

        # 4. Cloud (im Hintergrund)
        threading.Thread(target=self._connect_cloud, daemon=True).start()

        self._update_status("M.O.L.O.C.H. Service bereit")

    def _sync_flags_from_npu(self):
        """Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten."""
        self.scrfd_active = "scrfd" in self._active_ctx
        self.arcface_active = "arcface" in self._active_ctx
        self.yolo_active = "yolov8m" in self._active_ctx
        self.pose_active = "pose" in self._active_ctx
        self.hand_active = "hand_landmark" in self._active_ctx

    def _npu_watchdog(self):
        """Max-2 Enforcement + Anti-Oszillation. Laeuft jede Inference-Iteration."""
        # 1) Max 2 Modelle erzwingen
        _count = len(self._active_ctx)
        if _count > 2:
            logger.warning(f"[WATCHDOG] VIOLATION: {_count} Modelle aktiv! {list(self._active_ctx.keys())}")
            _prio = ["hand_landmark", "pose", "yolov8m", "arcface", "scrfd"]
            _victims = sorted(self._active_ctx.keys(),
                              key=lambda m: _prio.index(m) if m in _prio else 99)
            while len(self._active_ctx) > 2:
                _v = _victims.pop(0)
                logger.warning(f"[WATCHDOG] Unloading {_v}")
                self._unconfigure_model(_v)
            self._sync_flags_from_npu()
            if self._perception:
                self._perception.slots = list(self._active_ctx.keys())

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

        # Inference Loop
        threading.Thread(target=self._inference_loop, daemon=True, name="InferenceLoop").start()

        # Kamera-Status Polling (ersetzt root.after(3000, ...))
        threading.Thread(target=self._cam_status_loop, daemon=True, name="CamStatusLoop").start()

        # Panel IPC Command Polling
        threading.Thread(target=self._poll_panel_cmds, daemon=True, name="PanelCmdPoll").start()

        # Frozen Frame Watchdog
        threading.Thread(target=self._frozen_frame_watchdog, daemon=True, name="FrozenWatchdog").start()

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
        """Toggle model on/off via Perception Engine force_models()."""
        if not self._perception:
            logger.warning(f"[TOGGLE] Perception Engine nicht verfuegbar, ignoriere {model_key}={enabled}")
            return

        active_map = {"scrfd": "scrfd_active", "arcface": "arcface_active",
                      "yolov8m": "yolo_active", "pose": "pose_active",
                      "hand_landmark": "hand_active"}
        if model_key not in active_map:
            return

        # Aktuelle gewuenschte Modelle ermitteln (max 2!)
        current = set(self._active_ctx.keys())
        if enabled:
            wanted = current | {model_key}
            # ArcFace braucht SCRFD
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.add("scrfd")
            # Hand Landmark braucht Pose
            if "hand_landmark" in wanted and "pose" not in wanted:
                wanted.add("pose")
            # Max 2 Modelle: neues Modell + Dependencies behalten, Rest weg
            keep = {model_key}
            # Dependencies in keep aufnehmen
            DEPS = {"arcface": "scrfd", "hand_landmark": "pose"}
            if model_key in DEPS:
                keep.add(DEPS[model_key])
            while len(wanted) > 2:
                removable = wanted - keep
                if removable:
                    wanted.discard(removable.pop())
                else:
                    break
            # Post-loop: Dependencies nochmal validieren (Sicherheit)
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.discard("arcface")
                logger.info(f"[TOGGLE] arcface ohne scrfd entfernt -> wanted={wanted}")
            if "hand_landmark" in wanted and "pose" not in wanted:
                wanted.discard("hand_landmark")
                logger.info(f"[TOGGLE] hand_landmark ohne pose entfernt -> wanted={wanted}")
            logger.info(f"[TOGGLE] wanted={wanted} (max 2 enforced)")
        else:
            wanted = current - {model_key}
            # SCRFD weg -> ArcFace auch weg
            if model_key == "scrfd":
                wanted.discard("arcface")
            # Pose weg -> Hand Landmark auch weg
            if model_key == "pose":
                wanted.discard("hand_landmark")

        if wanted:
            self._perception.force_models(list(wanted))
            logger.info(f"[TOGGLE] force_models({list(wanted)}) via Panel")
        else:
            # Alles aus -> zurueck zu Auto-Scoring
            self._perception.force_models(None)
            logger.info("[TOGGLE] Alle Modelle aus -> Perception Auto-Modus")

    def toggle_autonomous_manual(self):
        """Toggle AUTONOM/MANUELL von GUI-Button.

        MANUELL: Service beobachtet weiter (Inference, Detection, Logs),
                 aber KEINE Kamera-Kontrolle. Nur Panel-Buttons steuern.
        AUTONOM: Service uebernimmt Kamera (Tentakel-Modus).
        """
        if not self._manual_mode:
            # -> MANUELL: Kamera-Kontrolle sperren
            logger.info("[MODUS] Wechsel zu MANUELL - Kamera-Kontrolle gesperrt")
            self._manual_mode = True
            self._tentakel_enabled = False

            # Tracker stoppen (falls aktiv)
            if self._autonomous_mode:
                self._disable_autonomous()

            # Takeover-State zuruecksetzen
            self._moloch_has_control = False
            self._manual_autonomous = False
            self._takeover_reason = ""
            self._guardian_move_count = 0

            # Smart Tracking AUS (wuerde sonst Kamera bewegen)
            def stop_cam_control():
                if self._cloud and self._cloud.connected:
                    try:
                        self._cloud.run(self._cloud.bridge.set_smart_tracking(False))
                        self._set_smart_tracking_state(False)
                    except Exception:
                        pass
                self._led_off()
            threading.Thread(target=stop_cam_control, daemon=True).start()

            self._notify("auto_mode", {"state": "manual"})
            self._update_status("MANUELL - Service beobachtet")
        else:
            # -> AUTONOM: Kamera-Kontrolle freigeben
            logger.info("[MODUS] Wechsel zu AUTONOM - Kamera-Kontrolle freigegeben")
            self._manual_mode = False
            self._tentakel_enabled = True

            # Smart Tracking AN (Tentakel-Default)
            def start_cam_control():
                if self._cloud and self._cloud.connected:
                    try:
                        self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                        self._set_smart_tracking_state(True)
                        logger.info("[TENTAKEL] Smart Tracking aktiviert")
                    except Exception:
                        pass
            threading.Thread(target=start_cam_control, daemon=True).start()

            # Guardian-State zuruecksetzen (frischer Start)
            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0
            self._takeover_cooldown_until = time.time() + 10  # 10s Grace nach Moduswechsel

            self._notify("auto_mode", {"state": "autonomous"})
            self._update_status("AUTONOM - Tentakel scannt")

    def stop(self):
        """Sauberes Herunterfahren."""
        logger.info("M.O.L.O.C.H. Service wird gestoppt...")
        self.running = False

        # Tracker stoppen
        if self._tracker:
            try:
                self._tracker.stop()
            except Exception:
                pass

        # Alle Modelle unconfigurieren
        for name in list(self._active_ctx.keys()):
            self._unconfigure_model(name)

        # VDevice schliessen
        if self._vdevice:
            try:
                self._models.clear()
                del self._vdevice
                self._vdevice = None
            except Exception:
                pass

        # Hailo freigeben
        if self._hailo_manager:
            try:
                self._hailo_manager.release_vision()
            except Exception:
                pass

        # IPC cleanup
        for path in [NPU_VISION_PAUSED,
                     '/dev/shm/moloch_frame', '/dev/shm/moloch_frame.tmp',
                     '/dev/shm/moloch_status.json', '/dev/shm/moloch_status.tmp']:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        logger.info("M.O.L.O.C.H. Service gestoppt")


    # =========================================================================
    # Panel IPC via /dev/shm
    # =========================================================================

    _shm_seq = 0

    def _write_shm(self, frame):
        """Write frame + status to /dev/shm for Panel IPC."""
        self._last_frame_write = time.time()
        try:
            MolochService._shm_seq = (MolochService._shm_seq + 1) & 0xFFFFFFFF
            h, w = frame.shape[:2]
            c = frame.shape[2] if len(frame.shape) > 2 else 1
            header = struct.pack('<IIII', h, w, c, MolochService._shm_seq)
            with open('/dev/shm/moloch_frame.tmp', 'wb') as f:
                f.write(header)
                f.write(frame.tobytes())
            os.rename('/dev/shm/moloch_frame.tmp', '/dev/shm/moloch_frame')
        except Exception:
            pass

        try:
            status = {
                "scrfd_active": self.scrfd_active,
                "arcface_active": self.arcface_active,
                "yolo_active": self.yolo_active,
                "pose_active": self.pose_active,
                "hand_active": self.hand_active,
                "npu_paused": self._npu_paused,
                "active_models": list(self._active_ctx.keys()),
                "autonomous_mode": self._autonomous_mode,
                "manual_mode": self._manual_mode,
                "moloch_has_control": self._moloch_has_control,
                "tentakel_enabled": self._tentakel_enabled,
                "daily_learner_enabled": self._daily_learner.enabled if self._daily_learner else False,
                "frame_age": round(time.time() - self._last_frame_write, 1),
                "frozen_restarts": self._frozen_restart_count,
                "fps": {k: round(v, 1) for k, v in self._fps.items()},
                "thresholds": {
                    "scrfd_conf": self.scrfd_conf_val,
                    "scrfd_nms": self.scrfd_nms_val,
                    "arcface_thresh": self.arcface_thresh_val,
                    "yolo_conf": self.yolo_conf_val,
                    "pose_conf": self.pose_conf_val,
                    "pose_nms": self.pose_nms_val,
                },
            }
            if self._perception:
                status["perception"] = self._perception.get_state()
            with open('/dev/shm/moloch_status.tmp', 'w') as f:
                json.dump(status, f)
            os.rename('/dev/shm/moloch_status.tmp', '/dev/shm/moloch_status.json')
        except Exception:
            pass

    def _poll_panel_cmds(self):
        """Poll for commands from Panel via IPC files (nummeriert)."""
        import glob as _glob
        while self.running:
            try:
                # Alle cmd-Files lesen (sortiert = chronologisch)
                cmd_files = sorted(_glob.glob('/tmp/moloch_cmd_*.json'))
                # Legacy single-file auch noch unterstuetzen
                legacy = '/tmp/moloch_cmd.json'
                if os.path.exists(legacy):
                    cmd_files.insert(0, legacy)
                for cf in cmd_files:
                    try:
                        with open(cf) as f:
                            cmd = json.load(f)
                        os.unlink(cf)
                        self._execute_panel_cmd(cmd)
                    except Exception as e:
                        logger.debug(f"Panel cmd poll ({cf}): {e}")
                        try:
                            os.unlink(cf)
                        except FileNotFoundError:
                            pass
            except Exception:
                pass
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
                self.pose_conf_val = float(_th.get('pose_conf', self.pose_conf_val))
                self.pose_nms_val = float(_th.get('pose_nms', self.pose_nms_val))
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
            if "pose_conf" in th:
                self.pose_conf_val = float(th["pose_conf"])
            if "pose_nms" in th:
                self.pose_nms_val = float(th["pose_nms"])
            logger.info(f"[SETTINGS] Thresholds: scrfd={self.scrfd_conf_val}/{self.scrfd_nms_val} "
                        f"arc={self.arcface_thresh_val} yolo={self.yolo_conf_val} "
                        f"pose={self.pose_conf_val}/{self.pose_nms_val}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Thresholds-Fehler: {e}")

        # Hand-Occlusion (gespeichert fuer spaeter, Perception Engine existiert noch nicht)
        try:
            ho = data.get("hand_occlusion", {})
            if ho:
                self._saved_hand_timeout = float(ho.get("timeout", 5.0))
                self._saved_hand_streak = int(ho.get("streak", 3))
                self._saved_hand_recency = float(ho.get("recency", 2.0))
                logger.info(f"[SETTINGS] Hand-Occlusion: timeout={self._saved_hand_timeout} "
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

    def _save_settings(self):
        """Speichere aktuelle Settings nach config/settings.json (atomic write)."""
        data = {"version": 1}

        # Thresholds
        data["thresholds"] = {
            "scrfd_conf": round(self.scrfd_conf_val, 3),
            "scrfd_nms": round(self.scrfd_nms_val, 3),
            "arcface_thresh": round(self.arcface_thresh_val, 3),
            "yolo_conf": round(self.yolo_conf_val, 3),
            "pose_conf": round(self.pose_conf_val, 3),
            "pose_nms": round(self.pose_nms_val, 3),
        }

        # Hand-Occlusion
        if self._perception:
            data["hand_occlusion"] = {
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
