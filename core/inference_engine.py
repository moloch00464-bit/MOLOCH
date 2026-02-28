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

            # Frame holen
            with self._cam._frame_lock:
                frame = self._cam._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue

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
            annotated = frame.copy()
            fh, fw = frame.shape[:2]
            self._frame_counter += 1

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
            if self.scrfd_active and "scrfd" in self._orchestrator._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._orchestrator.run("scrfd", input_rgb)
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
                        if self._cam._autonomous_mode and self._cam._tracker:
                            try:
                                face_dets = []
                                for box, score, _ in face_boxes:
                                    face_dets.append({
                                        "bbox": [box[0] * 640, box[1] * 640, box[2] * 640, box[3] * 640],
                                        "confidence": float(score),
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

                        outputs = self._orchestrator.run("arcface", crop_rgb)
                        if outputs:
                            emb_key = self._orchestrator._output_names["arcface"][0]
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
                                # Owner-Override loeschen: Vision hat Markus bestaetigt
                                if self._core_integrator and self._core_integrator.is_owner_confirmed():
                                    self._core_integrator.clear_owner_override()
                                # ArbitrationEngine: Identity Confirmed (Shadow gecappt)
                                try:
                                    from core.arbitration import get_arbitration
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

                            draw_name(annotated, box, name, sim, fh, fw,
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
                    out_key = self._orchestrator._output_names["yolov8m"][0]
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

                    outputs = self._orchestrator.run("hand_landmark", hand_224)
                    hand_result = decode_hand_landmark(outputs, presence_thresh=self.hand_conf_val)

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
            if self.pose_active and "pose" in self._orchestrator._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._orchestrator.run("pose", input_rgb)
                    _pose_data = decode_yolov8_pose(
                        outputs,
                        conf_thresh=self.pose_conf_val,
                        img_h=640, img_w=640,
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["pose"] = 1.0 / dt if dt > 0 else 0
                    self._model_health.record_inference("pose", dt * 1000)

                    if _pose_data:
                        draw_poses(annotated, _pose_data, scale_x, scale_y)
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

            # === LED Erkennungs-Indikator (Hysterese im LEDController) ===
            self._led.update_hysteresis(
                markus_recognized=_markus_recognized,
                face_detected=face_detected,
                persons_detected=_persons_detected,
                moloch_has_control=self._cam._moloch_has_control,
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
                cv2.rectangle(overlay, (0, 0), (fw, 30), (0, 0, 180), -1)
                annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
                cv2.putText(annotated, "HAND OCCLUSION", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self._cam._annotated_lock:
                self._cam._annotated_frame = annotated

            # Panel IPC: Preview-Groesse fuer SHM (1080p waere 6MB/Frame)
            self._ipc.write_frame(cv2.resize(annotated, (IPCRouter.PREVIEW_W, IPCRouter.PREVIEW_H)))
            self._write_status_cb()

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
