#!/usr/bin/env python3
"""
InferenceEngine - NPU Inference Pipeline.

Extrahiert aus moloch_service.py (Phase 4, Schritt 5).

Verantwortlichkeiten:
  - Inference Loop (Auto-Restart Wrapper + Inner Loop)
  - Face Detection/Recognition (SCRFD + ArcFace)
  - Object Detection (YOLOv8m)
  - Hand Landmark Detection
  - Pose Estimation (YOLOv8s Pose)
  - Perception Frame Aggregation
  - Face Attribute Analysis (NPU)
  - NPU Watchdog (Anti-Oszillation)
"""

import os
import json
import time
import threading
import logging
import traceback

import cv2
import numpy as np

from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_objects,
    draw_poses, enforce_draw_priority,
    decode_hand_landmark, draw_hand_landmarks,
    decode_yolov8_pose,
    estimate_head_pose,
)
from core.vision.gesture_detector import GestureDetector
from core.vision.hand_gesture_detector import HandGestureDetector
from core.vision.face_attr_npu import analyze_face as _analyze_face
from core.ipc_router import IPCRouter
from core.perception.perception_frame import PerceptionFrame, estimate_distance

logger = logging.getLogger("InferenceEngine")

FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")


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


def letterbox_resize(img, target_size=640):
    """Letterbox-Resize: Aspektverhaeltnis beibehalten, graues Padding.

    Gibt zurueck: (padded_img, scale, pad_x, pad_y, content_w, content_h)
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y, new_w, new_h


def _unletterbox_scrfd(boxes, landmarks, pad_x, pad_y, rw, rh, target=640):
    """Korrigiere SCRFD-Koordinaten: Letterbox-Space -> Frame-Space.

    Input: boxes (N,4) und landmarks (N,10) normalisiert [0,1] auf target x target.
    Output: korrigierte Werte normalisiert [0,1] relativ zum Original-Content.
    """
    if pad_x == 0 and pad_y == 0:
        return boxes, landmarks
    bc = boxes.copy()
    bc[:, [0, 2]] = np.clip((boxes[:, [0, 2]] * target - pad_x) / rw, 0, 1)
    bc[:, [1, 3]] = np.clip((boxes[:, [1, 3]] * target - pad_y) / rh, 0, 1)
    lc = landmarks.copy()
    lc[:, 0::2] = np.clip((landmarks[:, 0::2] * target - pad_x) / rw, 0, 1)
    lc[:, 1::2] = np.clip((landmarks[:, 1::2] * target - pad_y) / rh, 0, 1)
    return bc, lc


def _unletterbox_yolo(detections, pad_x, pad_y, rw, rh, target=640):
    """Korrigiere YOLOv8 NMS Detections: Letterbox-Space -> Frame-Space."""
    if pad_x == 0 and pad_y == 0:
        return detections
    corrected = []
    for d in detections:
        dc = dict(d)
        bx = d["bbox"]
        dc["bbox"] = [
            float(np.clip((bx[0] * target - pad_x) / rw, 0, 1)),
            float(np.clip((bx[1] * target - pad_y) / rh, 0, 1)),
            float(np.clip((bx[2] * target - pad_x) / rw, 0, 1)),
            float(np.clip((bx[3] * target - pad_y) / rh, 0, 1)),
        ]
        corrected.append(dc)
    return corrected


def _unletterbox_pose(poses, pad_x, pad_y, rw, rh, target=640):
    """Korrigiere Pose-Daten (Model-Pixel) fuer Letterbox-Offset.

    Justiert bbox und keypoints so dass * scale_x/y korrekte Frame-Pixel ergibt.
    """
    if pad_x == 0 and pad_y == 0:
        return poses
    sx = float(target) / rw
    sy = float(target) / rh
    corrected = []
    for p in poses:
        pc = dict(p)
        bx = p["bbox"]
        pc["bbox"] = [
            (bx[0] - pad_x) * sx,
            (bx[1] - pad_y) * sy,
            (bx[2] - pad_x) * sx,
            (bx[3] - pad_y) * sy,
        ]
        kpts = p["keypoints"].copy()
        kpts[:, 0] = (kpts[:, 0] - pad_x) * sx
        kpts[:, 1] = (kpts[:, 1] - pad_y) * sy
        pc["keypoints"] = kpts
        corrected.append(pc)
    return corrected


class InferenceEngine:
    """NPU Inference Pipeline mit Auto-Restart.

    Fuehrt die Hauptschleife: Frame holen -> NPU Inference -> Annotieren -> IPC.
    Alle Abhaengigkeiten werden per DI injiziert.
    """

    def __init__(self, orchestrator, camera, led, ipc,
                 perception=None, core_integrator=None,
                 daily_learner=None, perception_buffer=None,
                 model_health=None, notify_callback=None,
                 write_status_callback=None, update_status_callback=None):
        """
        Args:
            orchestrator: ModelOrchestrator (NPU Pipeline)
            camera: CameraManager (Frame Source + Tracker)
            led: LEDController (Erkennungs-Indikator)
            ipc: IPCRouter (SHM Frame/Status Write)
            perception: PerceptionEngine (Slot-Rotation)
            core_integrator: CoreIntegrator (Tension/Attention)
            daily_learner: DailyLearner (Snapshot-Logik)
            perception_buffer: PerceptionBuffer (Ring-Buffer)
            model_health: ModelHealth (Inference-Stats)
            notify_callback: callback(event, data) fuer UI
            write_status_callback: callback() fuer Status-JSON
            update_status_callback: callback(text) fuer Status-Text
        """
        # DI References
        self._orchestrator = orchestrator
        self._cam = camera
        self._led = led
        self._ipc = ipc
        self._perception = perception
        self._core_integrator = core_integrator
        self._daily_learner = daily_learner
        self._perception_buffer = perception_buffer
        self._model_health = model_health
        self._notify = notify_callback or (lambda e, d=None: None)
        self._write_status_cb = write_status_callback or (lambda: None)
        self._update_status_cb = update_status_callback or (lambda t: None)

        # Inference Thread Control
        self._running = False
        self._thread = None

        # Face DB
        self._face_db = {}

        # Gesture Detection
        self._gesture_detector = GestureDetector()
        self._hand_gesture_detector = HandGestureDetector()
        self._current_gesture = None

        # Face Attribute Caches
        self._cached_emotion = {}
        self._cached_gender = {}
        self._cached_age_range = {}

        # TTS Announcement Cooldown
        self._last_announce = {}

        # Frame Counter
        self._frame_counter = 0
        self._letterbox_debug_done = False  # Einmalig Letterbox-Parameter loggen
        self._bbox_debug_last_save = 0.0  # Cooldown fuer BBox-Debug Snapshots

        # Pose Energy Tracker
        self._prev_keypoints = None
        self._current_pframe = PerceptionFrame()

        # Hand State
        self._last_hand_detected = False
        self._hand_occlusion_enabled = False
        self._hand_no_detect = 0
        self._HAND_RELEASE_FRAMES = 75  # ~5s bei 15fps

        # FPS Tracking
        self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0,
                     "hand_landmark": 0, "pose": 0, "total": 0}
        self._fps_lock = threading.Lock()

        # NPU Watchdog: Anti-Oszillation Swap-Log
        self._swap_log = []

        # FPS (aus Orchestrator, bei always_on = 0.033 = 30 FPS)
        self._target_frame_delay = self._orchestrator.target_frame_delay

        # Model Enable Flags (werden von sync_flags_from_npu gesetzt)
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.hand_active = False
        self.pose_active = False
        self.face_attr_active = False

        # Threshold Values
        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50
        self.pose_conf_val = 0.50
        self.hand_conf_val = 0.65

        # Learner Flash
        self._learner_flash = False

        # Gate0 Phase 8: Name-Hysterese (OSD stimmt mit Panel ueberein)
        self._sticky_name = "Unbekannt"
        self._sticky_sim = 0.0
        self._sticky_frames = 0
        self._STICKY_HOLD_FRAMES = 15  # Name bleibt 15 Frames nach letzter Erkennung
        self._last_logged_name = None  # Nur bei Namenswechsel loggen

        # FPS Profiler (ein/ausschaltbar via settings.json -> profiler.enabled)
        self._profiler_enabled = False
        self._profiler_interval = 30
        self._profiler_accum = {
            "rtsp": 0.0, "preprocess": 0.0, "npu": 0.0, "parse": 0.0,
            "arcface": 0.0, "compare": 0.0, "status": 0.0, "total": 0.0,
        }
        self._profiler_count = 0
        self._profiler_last_log = 0.0
        self._profiler_log_path = "/mnt/moloch-data/logs/fps_profiler.log"
        self._load_profiler_config()

        logger.info("[INIT] InferenceEngine bereit")

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def start(self):
        """Startet Inference Thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True, name="InferenceLoop"
        )
        self._thread.start()
        logger.info("[START] Inference Thread gestartet")

    def stop(self):
        """Stoppt Inference Thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def running(self):
        return self._running

    # =====================================================================
    # Public API
    # =====================================================================

    def reload_face_db(self):
        """Face-DB neu laden (nach Enrollment)."""
        self._face_db = load_face_db(FACE_DB_PATH)
        base_names = set(k.split('#')[0] for k in self._face_db.keys()) if self._face_db else set()
        learned = sum(1 for k in self._face_db if '#' in k)
        msg = f"Face-DB: {len(base_names)} Personen, {learned} gelernt ({', '.join(base_names)})"
        logger.info(msg)
        self._update_status_cb(msg)
        if self._daily_learner:
            self._daily_learner.set_face_db(self._face_db, FACE_DB_PATH)

    def get_fps(self) -> dict:
        """FPS-Snapshot zurueckgeben."""
        with self._fps_lock:
            return dict(self._fps)

    def get_current_pframe(self):
        """Letzten aggregierten PerceptionFrame zurueckgeben."""
        return self._current_pframe

    def reset_fps(self):
        """FPS Tracking zuruecksetzen."""
        with self._fps_lock:
            self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0,
                         "hand_landmark": 0, "pose": 0, "total": 0}

    def sync_flags_from_npu(self):
        """Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten."""
        ctx = self._orchestrator._active_ctx
        self.scrfd_active = "scrfd" in ctx
        self.arcface_active = "arcface" in ctx
        self.yolo_active = "yolov8m" in ctx
        self.hand_active = "hand_landmark" in ctx
        self.pose_active = "pose" in ctx
        self.face_attr_active = "face_attr" in ctx

    def apply_attention_level(self, new_level: str):
        """Attention-Level anwenden (Modelle aktivieren/deaktivieren)."""
        if self._cam._manual_mode:
            return
        self._orchestrator.apply_attention_level(new_level)
        self.sync_flags_from_npu()
        self._target_frame_delay = self._orchestrator.target_frame_delay

    # =====================================================================
    # Inference Loop (Auto-Restart Wrapper)
    # =====================================================================

    def _inference_loop(self):
        """Inference Worker mit Auto-Restart bei Crash."""
        restart_count = 0
        while self._running:
            try:
                self._inference_loop_inner()
            except Exception as e:
                crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
                sep = "=" * 60
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                models = list(self._orchestrator._active_ctx.keys())
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
                self._orchestrator._npu_paused = False
                restart_count += 1
                self._update_status_cb(f"INFERENCE CRASH #{restart_count} - Neustart in 2s...")
                logger.warning(f"[INFERENCE] Crash #{restart_count} - restarting in 2s...")
                time.sleep(2)

    # =====================================================================
    # Inference Loop Inner (Hauptschleife)
    # =====================================================================

    def _inference_loop_inner(self):
        """Eigentliche Inference Loop (GUI-frei)."""
        while self._running:
            # Cross-process NPU coordination: Voice hat Vorrang
            if self._orchestrator.check_voice_request():
                time.sleep(0.1)
                continue

            # Safety: models empty = auto-recover
            if self._orchestrator.auto_recover_models():
                continue

            # Frame holen + Timestamp-Check (Gate 0: Frame > 200ms = veraltet)
            _prof = self._profiler_enabled
            if _prof:
                _t_rtsp = time.perf_counter()
            with self._cam._frame_lock:
                frame = self._cam._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue
            frame_age = time.time() - self._cam._last_frame_write
            if frame_age > 0.2:
                time.sleep(0.01)
                continue
            if _prof:
                _prof_rtsp = time.perf_counter() - _t_rtsp

            # Pause waehrend Modell-Konfiguration (NPU blockiert)
            if not self._orchestrator._configuring.wait(timeout=0.1):
                with self._cam._annotated_lock:
                    self._cam._annotated_frame = frame.copy()
                continue

            # === NPU WATCHDOG: Anti-Oszillation (kein Max-Limit bei 8GB) ===
            self._npu_watchdog()
            self._last_hand_detected = False  # Default: keine Hand pro Frame

            # Kein Modell konfiguriert ODER Inference pausiert -> Raw-Frame
            any_active = bool(self._orchestrator._active_ctx) and (
                self.scrfd_active or self.yolo_active or self.hand_active or self.pose_active)
            if not any_active:
                # Always-On Recovery: Modelle sofort konfigurieren statt auf Perception zu warten
                if self._orchestrator.orchestration_mode == "always_on":
                    try:
                        new_level = self._orchestrator.compute_attention_level()
                        self._orchestrator.apply_attention_level(new_level)
                        self.sync_flags_from_npu()
                        self._target_frame_delay = self._orchestrator.target_frame_delay
                        if self._orchestrator._active_ctx:
                            continue  # Sofort mit aktiven Modellen weitermachen
                    except Exception as e:
                        logger.debug(f"[ALWAYS-ON] Recovery: {e}")
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
                        _have = set(self._orchestrator._active_ctx.keys())
                        _to_remove = _have - _want
                        _to_add = _want - _have
                        if _to_remove or _to_add:
                            logger.info(f"[PERCEPTION] Swap (idle): {_have} -> {_want}")
                            for _m in _to_remove:
                                self._orchestrator.unconfigure(_m)
                                time.sleep(0.2)
                            for _m in _to_add:
                                if _m not in self._orchestrator._active_ctx:
                                    self._orchestrator.configure(_m)
                            # Sync perception slots + Flags aus NPU-Realitaet
                            self._perception.slots = list(self._orchestrator._active_ctx.keys())
                            self.sync_flags_from_npu()
                            self._swap_log.append(time.time())
                            self._notify("model_toggle", {
                                "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                                "yolov8m": self.yolo_active,
                                "hand_landmark": self.hand_active})
                            continue
                with self._cam._annotated_lock:
                    self._cam._annotated_frame = frame.copy()
                # SHM: Preview-Groesse fuer Panel IPC (1080p waere 6MB/Frame)
                self._ipc.write_frame(cv2.resize(frame, (IPCRouter.PREVIEW_W, IPCRouter.PREVIEW_H)))
                self._write_status_cb()
                time.sleep(0.03)
                continue

            t_total = time.perf_counter()
            # Tatsaechliche Loop-FPS (inkl. Throttle-Sleep)
            if hasattr(self, '_t_prev_loop'):
                _dt_loop = t_total - self._t_prev_loop
                if _dt_loop > 0:
                    with self._fps_lock:
                        self._fps["total"] = 1.0 / _dt_loop
            self._t_prev_loop = t_total
            fh, fw = frame.shape[:2]
            self._frame_counter += 1

            # Preview-Frame: Explizit auf 640x360 skalieren, dann darauf zeichnen
            # So werden BBox-Koordinaten direkt im Preview-Space berechnet
            annotated = cv2.resize(frame, (IPCRouter.PREVIEW_W, IPCRouter.PREVIEW_H))
            ah, aw = annotated.shape[:2]  # 360, 640

            # Preprocessing: Letterbox auf 640x640 (Aspektverhaeltnis beibehalten)
            if _prof:
                _t_pre = time.perf_counter()
            input_640, _lb_scale, _lb_px, _lb_py, _lb_rw, _lb_rh = letterbox_resize(frame, 640)
            input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

            # Scale-Faktoren: Modell-Space (640x640) -> Frame-Space (1920x1080)
            scale_x = fw / 640.0
            scale_y = fh / 640.0
            # Draw-Scale: Modell-Space (640x640) -> Preview-Space (640x360)
            draw_sx = aw / 640.0   # 640/640 = 1.0
            draw_sy = ah / 640.0   # 360/640 = 0.5625
            if _prof:
                _prof_pre = time.perf_counter() - _t_pre
                _prof_npu = 0.0
                _prof_parse = 0.0
                _prof_arcface = 0.0
                _prof_compare = 0.0

            # Max-2 Draw-Priority: face > hand
            _draw_candidates = []
            if self.scrfd_active:
                _draw_candidates.append("face")
            if self.hand_active:
                _draw_candidates.append("hand")
            _allowed_draws = set(enforce_draw_priority(_draw_candidates))

            face_boxes = []
            _face_raw_640 = None  # Original SCRFD-Boxes fuer Tracker + Hand-Crop
            face_detected = False
            face_fed_to_tracker = False
            _markus_recognized = False
            _persons_detected = False

            # 1. SCRFD Face Detection
            if self.scrfd_active and "scrfd" in self._orchestrator._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._orchestrator.run("scrfd", input_rgb)
                    if _prof:
                        _t_npu_end = time.perf_counter()
                        _prof_npu += _t_npu_end - t0
                    boxes, scores, landmarks = decode_scrfd(
                        outputs, img_size=640,
                        conf_thresh=self.scrfd_conf_val,
                        iou_thresh=self.scrfd_nms_val
                    )
                    if _prof:
                        _prof_parse += time.perf_counter() - _t_npu_end
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("scrfd", dt * 1000)

                    if len(boxes) > 0:
                        # Letterbox-Korrektur: Model-Space -> Frame-Space
                        #
                        # Das Kamerabild kommt in 1920x1080 rein. SCRFD braucht
                        # 640x640, also wird das Bild runterskaliert und dabei
                        # gepaddet (Letterboxing, weil 16:9 -> 1:1). Die NPU gibt
                        # Landmark-Koordinaten zurueck die auf 640x640 passen.
                        # Diese Koordinaten muessen korrekt auf 1920x1080
                        # zurueckgerechnet werden — inklusive Padding-Offset
                        # abziehen und Skalierungsfaktor anwenden. Wenn das fehlt
                        # oder falsch ist, landen BBox und Landmarks verschoben
                        # auf Brust/Hand statt im Gesicht.
                        boxes_c, lm_c = _unletterbox_scrfd(
                            boxes, landmarks, _lb_px, _lb_py, _lb_rw, _lb_rh)

                        # === BBox Debug: Koordinaten-Log + Full-HD Snapshots ===
                        _dbg_now = time.time()
                        if _dbg_now - self._bbox_debug_last_save >= 5.0:
                            self._bbox_debug_last_save = _dbg_now
                            _dbg_dir = os.path.expanduser("~/moloch/logs/bbox_debug")
                            os.makedirs(_dbg_dir, exist_ok=True)
                            _ts = time.strftime("%Y%m%d_%H%M%S")
                            for i in range(len(boxes)):
                                # RAW Koordinaten (normalisiert auf 640x640 Space)
                                raw = boxes[i]
                                logger.info(
                                    f"[BBox-Debug] Face {i} RAW (norm 640x640): "
                                    f"x1={raw[0]:.4f} y1={raw[1]:.4f} "
                                    f"x2={raw[2]:.4f} y2={raw[3]:.4f}")
                                # Unletterbox Koordinaten (normalisiert auf Content)
                                ulb = boxes_c[i]
                                logger.info(
                                    f"[BBox-Debug] Face {i} UNLETTERBOX (norm): "
                                    f"x1={ulb[0]:.4f} y1={ulb[1]:.4f} "
                                    f"x2={ulb[2]:.4f} y2={ulb[3]:.4f}")
                                # Pixel-Koordinaten auf Original-Frame
                                px1 = int(ulb[0] * fw)
                                py1 = int(ulb[1] * fh)
                                px2 = int(ulb[2] * fw)
                                py2 = int(ulb[3] * fh)
                                logger.info(
                                    f"[BBox-Debug] Face {i} PIXEL ({fw}x{fh}): "
                                    f"x1={px1} y1={py1} x2={px2} y2={py2}")
                                logger.info(
                                    f"[BBox-Debug] Letterbox: pad_x={_lb_px} "
                                    f"pad_y={_lb_py} rw={_lb_rw} rh={_lb_rh} "
                                    f"score={scores[i]:.3f}")
                                # Full-HD Frame mit roter BBox speichern
                                frame_dbg = frame.copy()
                                cv2.rectangle(frame_dbg, (px1, py1),
                                              (px2, py2), (0, 0, 255), 3)
                                cv2.putText(
                                    frame_dbg,
                                    f"Face{i} s={scores[i]:.2f}",
                                    (px1, max(py1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 0, 255), 2)
                                _fname_full = os.path.join(
                                    _dbg_dir, f"{_ts}_face{i}_fullhd.jpg")
                                cv2.imwrite(_fname_full, frame_dbg)
                                # Crop mit 20% Margin
                                bw = px2 - px1
                                bh = py2 - py1
                                margin_x = int(bw * 0.2)
                                margin_y = int(bh * 0.2)
                                cx1 = max(0, px1 - margin_x)
                                cy1 = max(0, py1 - margin_y)
                                cx2 = min(fw, px2 + margin_x)
                                cy2 = min(fh, py2 + margin_y)
                                crop = frame[cy1:cy2, cx1:cx2]
                                if crop.size > 0:
                                    _fname_crop = os.path.join(
                                        _dbg_dir,
                                        f"{_ts}_face{i}_crop.jpg")
                                    cv2.imwrite(_fname_crop, crop)
                                logger.info(
                                    f"[BBox-Debug] Gespeichert: {_fname_full}")
                        # === Ende BBox Debug ===

                        if "face" in _allowed_draws:
                            draw_faces(annotated, boxes_c, scores, lm_c, draw_sx, draw_sy)
                        face_boxes = list(zip(boxes_c, scores, lm_c))
                        _face_raw_640 = boxes  # Original fuer Tracker + Hand-Crop
                        face_detected = True
                        # Head Pose fuer erstes Gesicht (CPU, ~5ms)
                        _head_pose = estimate_head_pose(lm_c[0], fw, fh)
                        # Face hat PRIORITAET fuer Tracker (Original Model-Space)
                        if self._cam._autonomous_mode and self._cam._tracker:
                            try:
                                face_dets = []
                                for i in range(len(boxes)):
                                    face_dets.append({
                                        "bbox": [float(boxes[i, 0] * 640), float(boxes[i, 1] * 640),
                                                 float(boxes[i, 2] * 640), float(boxes[i, 3] * 640)],
                                        "confidence": float(scores[i]),
                                        "class": "face"
                                    })
                                self._cam._tracker.update_detection(
                                    detections=face_dets,
                                    frame_width=640, frame_height=640
                                )
                                face_fed_to_tracker = True
                            except Exception as e:
                                logger.debug(f"Tracker face feed: {e}")
                        # Guardian: Face sichtbar -> Interest
                        if self._cam._moloch_has_control:
                            self._cam._last_interesting_time = time.time()
                            self._cam._takeover_found_something = True
                        # Fliessender Takeover: erste Detection signalisieren
                        if self._cam._waiting_for_first_detection:
                            self._cam._first_detection_event.set()
                except Exception as e:
                    logger.error(f"SCRFD Fehler: {e}")
                    self._model_health.record_error("scrfd")


            # face_attr wird jetzt von PerceptionEngine STAGE_MODELS gesteuert
            # (Kein lazy-configure mehr — war Ursache des Load/Unload-Loop!)

            # 2. ArcFace (nur wenn SCRFD aktiv + Faces gefunden)
            if (self.arcface_active and self.scrfd_active
                    and face_boxes and "arcface" in self._orchestrator._active_ctx):
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

                        if _prof:
                            _t_af = time.perf_counter()
                        outputs = self._orchestrator.run("arcface", crop_rgb)
                        if outputs:
                            emb_key = self._orchestrator._output_names["arcface"][0]
                            embedding = outputs[emb_key].flatten()
                            embedding = normalize_arcface(embedding)
                            if _prof:
                                _prof_arcface += time.perf_counter() - _t_af

                            if _prof:
                                _t_cmp = time.perf_counter()
                            if self._face_db:
                                name, sim = match_face(
                                    embedding, self._face_db,
                                    threshold=self.arcface_thresh_val
                                )
                            else:
                                name, sim = "Keine DB", 0.0
                            if _prof:
                                _prof_compare += time.perf_counter() - _t_cmp

                            # Gate0 Phase 8: Name-Hysterese (EINE Wahrheit fuer OSD + Panel)
                            # Wenn match_face "Markus" liefert: sticky setzen
                            # Wenn "Unbekannt" aber sticky noch aktiv: Name beibehalten
                            if name != "Unbekannt":
                                self._sticky_name = name
                                self._sticky_sim = sim
                                self._sticky_frames = self._STICKY_HOLD_FRAMES
                            elif self._sticky_frames > 0:
                                # Name halten solange Hysterese laeuft
                                name = self._sticky_name
                                self._sticky_frames -= 1
                            else:
                                self._sticky_name = "Unbekannt"
                                self._sticky_sim = 0.0

                            # Gate0 Phase 8: Face-Log nur bei Namenswechsel
                            if name != self._last_logged_name:
                                self._last_logged_name = name
                                if name != "Unbekannt":
                                    logger.info(f"[SEHE] Gesicht name={name} confidence={sim:.2f} threshold={self.arcface_thresh_val:.2f}")
                                else:
                                    logger.info(f"[SEHE] Gesicht name=unbekannt best_match={sim:.2f} threshold={self.arcface_thresh_val:.2f}")

                            # LED Indikator: Markus erkannt?
                            if name.lower() == "markus":
                                _markus_recognized = True
                                # Owner-Override loeschen: Vision hat Markus bestaetigt
                                if self._core_integrator and self._core_integrator.is_owner_confirmed():
                                    self._core_integrator.clear_owner_override()
                                # ArbitrationEngine: Identity Confirmed (Shadow gecappt)
                                # NUR wenn CoreIntegrator NICHT im Shadow-Modus ist
                                # (hohe Tension = Bedrohungslage, einzelner Markus-Frame
                                #  soll Override nicht staendig neu setzen)
                                try:
                                    from core.arbitration import get_arbitration
                                    ci_zone = self._core_integrator.get_personality_zone() if self._core_integrator else "guardian"
                                    if ci_zone != "shadow":
                                        get_arbitration().identity_confirmed()
                                except Exception:
                                    pass

                            # Face Attributes (NPU, ~2926 FPS — Gender/Age/Emotion)
                            emotion = self._cached_emotion.get(name)
                            gender = self._cached_gender.get(name)
                            age_range = self._cached_age_range.get(name)
                            if self.face_attr_active and crop is not None:
                                try:
                                    fa_crop = cv2.resize(crop, (178, 218))
                                    fa_rgb = cv2.cvtColor(fa_crop, cv2.COLOR_BGR2RGB)
                                    fa_out = self._orchestrator.run("face_attr", fa_rgb)
                                    if fa_out:
                                        fa_key = self._orchestrator._output_names["face_attr"][0]
                                        gender, age_range, emotion = _analyze_face(fa_out[fa_key])
                                        self._cached_gender[name] = gender
                                        self._cached_age_range[name] = age_range
                                        self._cached_emotion[name] = emotion
                                except Exception:
                                    pass

                            draw_name(annotated, box, name, sim, ah, aw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._ipc.write_face_state(name, sim, len(face_boxes),
                                                       emotion=emotion, gender=gender, age_range=age_range,
                                                       head_pose=_head_pose if '_head_pose' in dir() else None,
                                                       detected_objects=_detected_objects if '_detected_objects' in dir() else [])

                            # DailyLearner: Snapshot NUR bei echtem Gesicht (hoher SCRFD Score)
                            # score >= 0.65 filtert Falsch-Positive (Moebel, Kissen etc.)
                            if (self._daily_learner and self._daily_learner.enabled
                                    and name != "Keine DB" and float(score) >= 0.65):
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
            if self.yolo_active and "yolov8m" in self._orchestrator._active_ctx and not face_detected:
                try:
                    t0 = time.perf_counter()
                    outputs = self._orchestrator.run("yolov8m", input_rgb)
                    if _prof:
                        _t_yolo_npu = time.perf_counter()
                        _prof_npu += _t_yolo_npu - t0
                    out_key = self._orchestrator._output_names["yolov8m"][0]
                    all_dets = decode_yolov8_nms(
                        outputs[out_key],
                        class_id=-1,
                        conf_thresh=self.yolo_conf_val
                    )
                    if _prof:
                        _prof_parse += time.perf_counter() - _t_yolo_npu
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["yolov8m"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("yolov8m", dt * 1000)

                    # Personen und andere Objekte trennen
                    persons = [d for d in all_dets if d.get("class_id", -1) == 0]
                    objects = [d for d in all_dets if d.get("class_id", -1) != 0]

                    # Nicht-Person-Objekte zeichnen (Letterbox-korrigiert)
                    if objects:
                        objects_c = _unletterbox_yolo(
                            objects, _lb_px, _lb_py, _lb_rw, _lb_rh)
                        draw_objects(annotated, objects_c, scale_x, scale_y)
                        _detected_objects = [
                            {"class": d["class"], "confidence": round(d["confidence"], 2)}
                            for d in objects
                        ]

                    if persons:
                        _persons_detected = True
                        persons_c = _unletterbox_yolo(
                            persons, _lb_px, _lb_py, _lb_rw, _lb_rh)
                        draw_persons(annotated, persons_c, scale_x, scale_y)
                        if self._cam._moloch_has_control:
                            self._cam._last_interesting_time = time.time()
                            self._cam._takeover_found_something = True
                        # Fliessender Takeover: erste Detection signalisieren
                        if self._cam._waiting_for_first_detection:
                            self._cam._first_detection_event.set()
                        if self._cam._autonomous_mode and self._cam._tracker and not face_fed_to_tracker:
                            try:
                                pixel_dets = []
                                for p in persons:
                                    bx = p["bbox"]
                                    pixel_dets.append({
                                        "bbox": [bx[0] * 640, bx[1] * 640, bx[2] * 640, bx[3] * 640],
                                        "confidence": p["confidence"],
                                        "class": "person"
                                    })
                                self._cam._tracker.update_detection(
                                    detections=pixel_dets,
                                    frame_width=640, frame_height=640
                                )
                            except Exception as e:
                                logger.debug(f"Tracker YOLOv8m feed: {e}")
                except Exception as e:
                    logger.error(f"YOLOv8m Fehler: {e}")
                    self._model_health.record_error("yolov8m")

            # 4. Hand Landmark Detection (224x224 Crop aus Person-BBox oder Bildmitte)
            if self.hand_active and "hand_landmark" in self._orchestrator._active_ctx:
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
                    elif _face_raw_640 is not None:
                        # Face-BBox erweitert (Original Model-Space fuer 640x640 Crop)
                        fb = _face_raw_640[0]  # (x1, y1, x2, y2) normalisiert auf 640x640
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

                    outputs = self._orchestrator.run("hand_landmark", hand_224)
                    if _prof:
                        _t_hand_npu = time.perf_counter()
                        _prof_npu += _t_hand_npu - t0
                    hand_result = decode_hand_landmark(outputs, presence_thresh=self.hand_conf_val)
                    if _prof:
                        _prof_parse += time.perf_counter() - _t_hand_npu

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
                                scale_x=draw_sx, scale_y=draw_sy,
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
            if self.pose_active and "pose" in self._orchestrator._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._orchestrator.run("pose", input_rgb)
                    if _prof:
                        _t_pose_npu = time.perf_counter()
                        _prof_npu += _t_pose_npu - t0
                    _pose_data = decode_yolov8_pose(
                        outputs,
                        conf_thresh=self.pose_conf_val,
                        img_h=640, img_w=640,
                    )
                    if _prof:
                        _prof_parse += time.perf_counter() - _t_pose_npu
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["pose"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("pose", dt * 1000)

                    if _pose_data:
                        _pose_draw = _unletterbox_pose(
                            _pose_data, _lb_px, _lb_py, _lb_rw, _lb_rh)
                        draw_poses(annotated, _pose_draw, draw_sx, draw_sy)
                        # Tracker mit Pose-Daten fuettern (FACE > BODY Prioritaet)
                        if self._cam._autonomous_mode and self._cam._tracker and not face_fed_to_tracker:
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
                                    # Nose-Keypoint separat (Tracking-Prioritaet 1)
                                    nose_center = None
                                    if kpts[0, 2] > 0.3:
                                        nose_center = (float(kpts[0, 0] / 640.0),
                                                       float(kpts[0, 1] / 640.0))
                                    pose_dets.append({
                                        "bbox": p["bbox"],
                                        "confidence": p["score"],
                                        "has_face": has_face,
                                        "face_center": face_center,
                                        "face_confidence": face_conf,
                                        "has_torso": has_torso,
                                        "nose_center": nose_center,
                                    })
                                self._cam._tracker.update_pose_detection(
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
            if _prof:
                _t_status = time.perf_counter()
            if self._perception:
                _perc_face_bbox = None
                if face_boxes:
                    _fb = face_boxes[0][0]
                    _perc_face_bbox = (float(_fb[0]), float(_fb[1]), float(_fb[2]), float(_fb[3]))
                _perc_camera_moving = False
                if self._cam._tracker and hasattr(self._cam._tracker, '_camera') and self._cam._tracker._camera:
                    _cam_pos = getattr(self._cam._tracker._camera, 'current_position', None)
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
                # Always-On: KEINE Modell-Rotation, alle 6 bleiben permanent aktiv
                if _new_slots and self._orchestrator.orchestration_mode != "always_on":
                    _want = set(_new_slots)
                    _have = set(self._orchestrator._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")
                        for _m in _to_remove:
                            self._orchestrator.unconfigure(_m)
                            time.sleep(0.2)
                        for _m in _to_add:
                            if _m not in self._orchestrator._active_ctx:
                                self._orchestrator.configure(_m)
                        # Sync perception slots + Flags aus NPU-Realitaet
                        self._perception.slots = list(self._orchestrator._active_ctx.keys())
                        self.sync_flags_from_npu()
                        self._swap_log.append(time.time())
                        self._notify("model_toggle", {
                            "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                            "yolov8m": self.yolo_active,
                            "hand_landmark": self.hand_active})

            # === LED Erkennungs-Indikator (Gate0 Phase 6: EINE Wahrheit mit Iris) ===
            _led_mode = "guardian"
            if self._core_integrator:
                try:
                    _led_mode = self._core_integrator.get_personality_zone()
                except Exception:
                    pass
            self._led.update_hysteresis(
                markus_recognized=_markus_recognized,
                face_detected=face_detected,
                persons_detected=_persons_detected,
                moloch_has_control=self._cam._moloch_has_control,
                personality_mode=_led_mode,
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
                    self._core_integrator.update_input("system", "alarm_active", 1.0 if self._cam._alarm_on else 0.0)
                    # System-Last (grob: NPU aktiv = etwas Last)
                    _npu_load = len(self._orchestrator._active_ctx) / 6.0  # 0-6 Modelle -> 0.0-1.0
                    self._core_integrator.update_input("system", "system_load", _npu_load)
                except Exception:
                    pass  # Integrator darf NIE die Inference-Loop stoeren

            # === Phase 3: Attention-Level basierte Modell-Orchestrierung ===
            try:
                new_level = self._orchestrator.compute_attention_level()
                self.apply_attention_level(new_level)
            except Exception as e:
                logger.debug(f"[ORCHESTRATION] Fehler: {e}")

            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:
                if self._last_hand_detected:
                    self._hand_no_detect = 0
                else:
                    self._hand_no_detect += 1
                    if self._hand_no_detect >= self._HAND_RELEASE_FRAMES:
                        if not self._cam._manual_mode:
                            logger.info(f"[AUTO-SWITCH] {self._HAND_RELEASE_FRAMES} Frames keine Hand -> Auto-Scoring")
                            self._perception.force_models(None)
                        self._hand_no_detect = 0

            # dt_total fuer Throttle (Verarbeitungszeit OHNE Sleep)
            dt_total = time.perf_counter() - t_total

            # Hand-Occlusion Overlay auf Video (nur wenn enabled in settings.json)
            if self._hand_occlusion_enabled and self._perception and self._perception._hand_occlusion:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (aw, 30), (0, 0, 180), -1)
                annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                cv2.putText(annotated, "HAND OCCLUSION", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with self._cam._annotated_lock:
                self._cam._annotated_frame = annotated

            # Panel IPC: annotated ist bereits Preview-Groesse (640x360)
            self._ipc.write_frame(annotated)
            self._write_status_cb()

            # FPS Profiler: Zeiten akkumulieren und alle N Sekunden loggen
            if _prof:
                _prof_status = time.perf_counter() - _t_status
                _prof_total = time.perf_counter() - t_total
                self._profiler_tick({
                    "rtsp": _prof_rtsp, "preprocess": _prof_pre,
                    "npu": _prof_npu, "parse": _prof_parse,
                    "arcface": _prof_arcface, "compare": _prof_compare,
                    "status": _prof_status, "total": _prof_total,
                })

            # === Phase 3: Adaptive FPS — Throttle bei niedrigem Attention-Level ===
            # FPS-Boost: Kein Throttle wenn manuelles PTZ aktiv (letzten 3s)
            _ptz_boost = (time.time() - self._cam._last_manual_ptz) < 3.0
            if not _ptz_boost and dt_total < self._target_frame_delay:
                _sleep = self._target_frame_delay - dt_total
                time.sleep(_sleep)

    # =====================================================================
    # Perception Frame Builder
    # =====================================================================

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
        pf.hand_detected = self._last_hand_detected
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
        pf.active_models = list(self._orchestrator._active_ctx.keys())

        return pf

    # =====================================================================
    # Pose Energy
    # =====================================================================

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

    # =====================================================================
    # NPU Watchdog
    # =====================================================================

    def _npu_watchdog(self):
        """Anti-Oszillation. Laeuft jede Inference-Iteration.
        Hailo-10H 8GB: Alle 4 Modelle passen gleichzeitig (~43MB)."""

        # Anti-Oszillation: >3 Swaps in 1s -> Pause
        _now = time.time()
        self._swap_log = [t for t in self._swap_log if _now - t < 1.0]
        if len(self._swap_log) >= 3:
            logger.warning(f"[WATCHDOG] Anti-Oscillation: {len(self._swap_log)} Swaps in 1s! Pause 2s.")
            time.sleep(2.0)
            self._swap_log.clear()

    # =====================================================================
    # Helpers
    # =====================================================================

    def _announce_person(self, name):
        """Person erkannt - Log (LED wird vom Indikator gesteuert)."""
        logger.info(f"[FACE] Person erkannt: {name}")

    # =====================================================================
    # FPS Profiler
    # =====================================================================

    def _load_profiler_config(self):
        """Profiler-Config aus settings.json laden."""
        try:
            settings_path = os.path.expanduser("~/moloch/config/settings.json")
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            prof = settings.get("profiler", {})
            self._profiler_enabled = prof.get("enabled", False)
            self._profiler_interval = prof.get("log_interval_sec", 30)
            if self._profiler_enabled:
                logger.info(f"[PROFILER] Aktiv, Intervall {self._profiler_interval}s, Log: {self._profiler_log_path}")
        except Exception as e:
            logger.debug(f"[PROFILER] Config laden: {e}")

    def _profiler_tick(self, timings: dict):
        """Einzelne Frame-Zeiten akkumulieren und alle N Sekunden loggen."""
        for key in self._profiler_accum:
            self._profiler_accum[key] += timings.get(key, 0.0)
        self._profiler_count += 1

        now = time.time()
        if self._profiler_last_log == 0.0:
            self._profiler_last_log = now
            return
        elapsed = now - self._profiler_last_log
        if elapsed < self._profiler_interval:
            return

        # Durchschnitt berechnen und loggen
        n = max(self._profiler_count, 1)
        avg = {k: (v / n) * 1000 for k, v in self._profiler_accum.items()}
        fps = 1000.0 / avg["total"] if avg["total"] > 0 else 0.0

        line = (
            f"[PROFILER] RTSP: {avg['rtsp']:.0f}ms | "
            f"Preprocess: {avg['preprocess']:.0f}ms | "
            f"NPU: {avg['npu']:.0f}ms | "
            f"Parse: {avg['parse']:.0f}ms | "
            f"ArcFace: {avg['arcface']:.0f}ms | "
            f"Compare: {avg['compare']:.0f}ms | "
            f"Status: {avg['status']:.0f}ms | "
            f"TOTAL: {avg['total']:.0f}ms ({fps:.1f} FPS) [{n} frames/{elapsed:.0f}s]"
        )
        logger.info(line)

        # In Profiler-Logfile schreiben
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._profiler_log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts} {line}\n")
        except Exception:
            pass

        # Reset
        self._profiler_accum = {k: 0.0 for k in self._profiler_accum}
        self._profiler_count = 0
        self._profiler_last_log = now
