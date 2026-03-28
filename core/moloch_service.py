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
# Moloch path
sys.path.insert(0, os.path.expanduser("~/moloch"))

# Feature-Flag: MOLOCH_USE_TAPPAS=1 → GStreamer/TAPPAS Pipeline statt InferenceEngine
USE_TAPPAS = os.environ.get("MOLOCH_USE_TAPPAS", "0") == "1"

from core.hardware.hailo_manager import get_hailo_manager
from core.led_controller import LEDController
from core.hardware.rgb_led_controller import get_rgb_led
from core.ipc_router import IPCRouter
from core.model_orchestrator import ModelOrchestrator, MODEL_PATHS
from core.camera_manager import CameraManager
if USE_TAPPAS:
    from core.perception.tappas_pipeline import TappasPipeline
else:
    from core.inference_engine import InferenceEngine
from core.longterm_memory import get_memory
from core.perception.perception_buffer import get_perception_buffer
from core.perception.model_health import get_model_health
from core.ptz_tracker import get_ptz_tracker
from core.memory.episodic_memory import get_episodic_memory
from core.music.music_memory import get_music_memory
from core.awareness.room_map import get_room_map
from core.awareness.motion_analyzer import get_motion_analyzer
from core.awareness.activity_analyzer import get_activity_analyzer
from core.awareness.context_evaluator import get_context_evaluator
from core.personality.tension_integrator import get_tension_integrator
from core.personality.mood_engine import get_mood_engine
from core.personality.behavior_rules import get_behavior_rules
from core.debug.event_logger import log_event
from core.autonomy.decision_engine import get_decision_engine
from core.autonomy.atmosphere_controller import get_atmosphere_controller
from core.autonomy.homeostasis import get_homeostasis
from core.autonomy.night_cycle import get_night_cycle

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MolochService")
logger.setLevel(logging.INFO)

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")


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

        # Core Integrator (Zentrales Zustandsmodell: Tension/Attention/Presence)
        self._core_integrator = None
        try:
            from core.core_integrator import get_core_integrator
            self._core_integrator = get_core_integrator()
            logger.info("[INIT] CoreIntegrator bereit")
        except Exception as e:
            logger.warning(f"[INIT] CoreIntegrator nicht verfuegbar: {e}")

        # MolochSprache (Semantisches Protokoll)
        self._sprache = None
        try:
            from core.moloch_sprache import get_sprache
            self._sprache = get_sprache()
            logger.info("[INIT] MolochSprache bereit")
        except Exception as e:
            logger.warning(f"[INIT] MolochSprache nicht verfuegbar: {e}")

        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None
        try:
            from core.perception_engine import PerceptionEngine
            from core.personality.personality_engine import get_personality_engine
            _pe = get_personality_engine()
            self._perception = PerceptionEngine(personality_engine=_pe)
            logger.info(f"[INIT] Perception Engine bereit (Personality: {_pe.mode.value})")
            # Hand-Occlusion Params werden spaeter in _load_settings() angewendet
            # (nach InferenceEngine-Erstellung)
            # Gate 0: force_models erfolgt in _apply_settings() NACH init
        except Exception as e:
            logger.warning(f"[INIT] Perception Engine nicht verfuegbar: {e}")

        # Teachen (ehemals Daily Learner)
        self._teachen = None
        try:
            from core.teachen import get_teachen
            self._teachen = get_teachen()
            logger.info("[INIT] Teachen bereit")
        except Exception as e:
            logger.warning(f"[INIT] Teachen nicht verfuegbar: {e}")

        # Einpraegen (Batch Face+Pose Enrollment)
        self._einpraegen = None
        try:
            from core.einpraegen import get_einpraegen
            self._einpraegen = get_einpraegen()
            logger.info("[INIT] Einpraegen bereit")
        except Exception as e:
            logger.warning(f"[INIT] Einpraegen nicht verfuegbar: {e}")

        # Voice Pipeline (PTT -> Whisper -> Claude -> TTS)
        self._voice_pipeline = None
        try:
            from core.voice_pipeline import VoicePipeline
            self._voice_pipeline = VoicePipeline()
            logger.info("[INIT] Voice Pipeline bereit")
        except Exception as e:
            logger.warning(f"[INIT] Voice Pipeline nicht verfuegbar: {e}")

        # === Phase 3: Model Orchestration ===
        # Perception Buffer (Ring-Buffer fuer Trend-Analyse)
        self._perception_buffer = get_perception_buffer()
        # Model Health Monitor
        self._model_health = get_model_health()

        # ModelOrchestrator (NPU Pipeline + Modell-Lifecycle, Phase 4)
        self._orchestrator = ModelOrchestrator(
            perception_engine=self._perception,
            core_integrator=self._core_integrator,
            teachen=self._teachen,
            model_health=self._model_health,
            notify_callback=self._notify,
        )
        # Aliased Referenzen auf Orchestrator-Objekte (fuer _write_status_json)
        self._active_ctx = self._orchestrator._active_ctx
        self._ctx_lock = self._orchestrator._ctx_lock

        # LED Controller (extrahiert aus moloch_service.py, Phase 4)
        # Cloud wird spaeter via CameraManager.connect_cloud() gesetzt
        self._led = LEDController(core_integrator=self._core_integrator)

        # RGB-LED Controller (ESP32 WS2812 via UDP)
        self._rgb_led = None

        # Teach-Modus: Automatische Qualitaetspruefung bei Gesichtserkennung
        self._teach_mode_enabled = False      # Toggle via IPC, persistent in settings.json
        self._teach_result = {}               # Aktueller Teach-Vorgang Status (GUI-Polling)
        self._teach_busy = False              # Verhindert Doppel-Trigger
        self._teach_last_trigger = 0.0        # Cooldown: min 30s zwischen Auto-Teaches
        self._teach_cooldown = 30.0           # Sekunden zwischen automatischen Teach-Versuchen

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
        # Aliased mutable Referenzen (fuer _write_status_json)
        self._cloud_state = self._cam._cloud_state

        # IPC Router (extrahiert aus moloch_service.py, Phase 4)
        self._ipc = IPCRouter()

        # NPU Pipeline (Phase 4 Schritt 5)
        if USE_TAPPAS:
            logger.info("[INIT] TAPPAS Pipeline (GStreamer + Model Scheduler)")
            self._inference = TappasPipeline()
        else:
            self._inference = InferenceEngine(
                orchestrator=self._orchestrator,
                camera=self._cam,
                led=self._led,
                ipc=self._ipc,
                perception=self._perception,
                core_integrator=self._core_integrator,
                teachen=self._teachen,
                perception_buffer=self._perception_buffer,
                model_health=self._model_health,
                notify_callback=self._notify,
                write_status_callback=self._write_status_json,
                update_status_callback=self._update_status,
            )

        # Audio-Defaults VOR _load_settings() (W4 Audit-Fix)
        self._saved_mic_gain = 1.0
        self._saved_noise_gate = -36.0
        self._saved_agc = True
        self._audio_level = 0.0

        # PTZ-Defaults
        self._ptz_home_pan = 0.0
        self._ptz_home_tilt = 0.0
        self._ptz_tracking_speed = 0.7
        self._ptz_search_speed = 0.15
        self._ptz_pan_limit_min = -168.4
        self._ptz_pan_limit_max = 170.0
        self._ptz_tilt_limit_min = -78.0
        self._ptz_tilt_limit_max = 78.8

        # Settings aus config/settings.json laden (schreibt auf self._inference)
        self._load_settings()

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

    # CameraManager Callbacks

    def _set_model_flags_cb(self, flags_dict):
        """Callback fuer CameraManager: Model-Flags auf InferenceEngine setzen."""
        for attr, val in flags_dict.items():
            setattr(self._inference, attr, val)

    def _reset_fps(self):
        """Callback fuer CameraManager: FPS Tracking zuruecksetzen."""
        self._inference.reset_fps()

    # =========================================================================
    # Inference (delegiert an InferenceEngine)
    # =========================================================================

    def _sync_flags_from_npu(self):
        """Delegiert an InferenceEngine.sync_flags_from_npu()."""
        self._inference.sync_flags_from_npu()

    def toggle_model(self, model_key, enabled):
        """Thin Wrapper -> ModelOrchestrator.toggle_model()."""
        self._orchestrator.toggle_model(model_key, enabled)

    def toggle_autonomous_manual(self):
        """Thin Wrapper -> CameraManager.toggle_autonomous_manual()."""
        self._cam.toggle_autonomous_manual()

    def _led_flash(self, farbe: str = "weiss", duration: float = 1.0):
        """ReSpeaker LED kurz aufblitzen, dann vorherigen Zustand restaurieren."""
        if not self._rgb_led:
            return
        prev_state = self._rgb_led._current_state
        self._rgb_led.send_command(f"LED:{farbe} statisch")

        def _restore():
            time.sleep(duration)
            # Vorherigen Zustand wiederherstellen
            from core.hardware.rgb_led_controller import ZUSTAND_LED_MAP
            self._rgb_led._current_state = ""   # Reset → set_state skipped nicht
            cmd = ZUSTAND_LED_MAP.get(prev_state, "LED:blau pulsierend langsam")
            self._rgb_led.send_command(cmd)
            self._rgb_led._current_state = prev_state
        threading.Thread(target=_restore, daemon=True, name="LedFlash").start()

    def _led_flash_white(self, duration: float = 1.0):
        """Rueckwaerts-Kompatibilitaet."""
        self._led_flash("weiss", duration)

    # =========================================================================
    # Teach Smart Snapshot — Qualitaetspruefung via NPU
    # =========================================================================

    def _teach_smart_snapshot(self, pre_frame=None, pre_detections=None):
        """Teach-Foto mit NPU-Qualitaetspruefung (3 Versuche).

        Args:
            pre_frame: Vorbereiteter Frame aus Perception Loop (optional)
            pre_detections: Vorbereitete Detections aus Perception Loop (optional)

        Prueft: SCRFD-Confidence, Face-Groesse, Helligkeit, Embedding.
        LED-Feedback: weiss=gut, rot=schlecht, gelb=aufgegeben.
        """
        import cv2
        import numpy as np
        import json
        import os

        # Teach-Qualitaet: Lieber ein mittleres Bild als gar keins
        MAX_RETRIES = 3
        MIN_CONF = 0.50       # SCRFD Confidence (war 0.80, dann 0.65 — SCRFD@50% ist valide)
        MIN_FACE_PX = 60      # Gesicht Mindestgroesse in Pixel (war 80)
        MIN_BRIGHTNESS = 25
        MAX_BRIGHTNESS = 230
        MIN_EMB_NORM = 0.5    # ArcFace MobileFaceNet liefert L2-norm ≈ 1.0 (war 10.0!)

        self._teach_busy = True
        self._teach_result = {"status": "running", "attempt": 0, "detail": ""}
        try:
            self._teach_inner(
                cv2, np, json, os,
                MAX_RETRIES, MIN_CONF, MIN_FACE_PX,
                MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_EMB_NORM,
                pre_frame, pre_detections
            )
        except Exception as e:
            logger.error(f"[TEACH] Unerwarteter Fehler: {e}")
            self._teach_result = {"status": "failed", "detail": str(e), "reason": str(e)}
        finally:
            self._teach_busy = False
            # Scheduler zuruecksetzen wenn Teach-Modus AUS (manueller Trigger)
            if not self._teach_mode_enabled and hasattr(self._inference, 'force_all_active'):
                self._inference.force_all_active(False)

    def _teach_inner(self, cv2, np, json, os,
                     MAX_RETRIES, MIN_CONF, MIN_FACE_PX,
                     MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_EMB_NORM,
                     pre_frame, pre_detections):
        """Innere Teach-Logik (try/finally-gesichert durch Wrapper)."""

        for attempt in range(1, MAX_RETRIES + 1):
            self._teach_result["attempt"] = attempt
            self._teach_result["status"] = "running"
            self._teach_result["detail"] = f"Versuch {attempt}/{MAX_RETRIES}..."
            logger.info(f"[TEACH] Versuch {attempt}/{MAX_RETRIES}")

            # --- FIX 2: Frame + Detections holen ---
            # Versuch 1: vorbereitete Daten aus Perception Loop nutzen
            # Versuch 2+: frisch aus Pipeline holen
            if attempt == 1 and pre_frame is not None and pre_detections is not None:
                frame = pre_frame
                detections = pre_detections
                logger.info("[TEACH] Nutze vorbereiteten Frame aus Perception Loop")
            else:
                detections = []
                frame = None
                try:
                    detections = self._inference.get_detections() if self._inference else []
                    frame = self._inference.get_annotated_frame() if self._inference else None
                except Exception as e:
                    logger.warning(f"[TEACH] Frame/Detection Fehler: {e}")

            if frame is None:
                self._teach_result["detail"] = "Kein Kamerabild verfuegbar"
                self._led_flash("rot", 0.5)
                logger.warning("[TEACH] Kein Frame")
                if attempt < MAX_RETRIES:
                    time.sleep(1.0)
                continue

            # --- Beste Face-Detection finden ---
            best_face = None
            for det in detections:
                if det.get("class") == "face":
                    if best_face is None or det.get("confidence", 0) > best_face.get("confidence", 0):
                        best_face = det

            if best_face is None:
                self._teach_result["detail"] = "Kein Gesicht erkannt"
                self._teach_result["status"] = "retry"
                self._led_flash("rot", 0.5)
                logger.info(f"[TEACH] Kein Gesicht — {len(detections)} Detections vorhanden")
                if attempt < MAX_RETRIES:
                    time.sleep(1.0)
                continue

            # --- Qualitaetspruefungen ---
            face_conf = best_face.get("confidence", 0.0)
            face_bbox = best_face.get("bbox", [0, 0, 0, 0])
            embedding = best_face.get("embedding", None)

            h, w = frame.shape[:2]
            x1_px = int(face_bbox[0] * w)
            y1_px = int(face_bbox[1] * h)
            x2_px = int(face_bbox[2] * w)
            y2_px = int(face_bbox[3] * h)
            face_w = x2_px - x1_px
            face_h = y2_px - y1_px

            # Helligkeit im Face-Bereich
            brightness = 128
            try:
                face_crop = frame[max(0, y1_px):max(1, y2_px), max(0, x1_px):max(1, x2_px)]
                if face_crop.size > 0:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
                    brightness = int(np.mean(gray))
            except Exception:
                pass

            # Embedding-Norm
            emb_norm = 0.0
            if embedding is not None and len(embedding) > 0:
                emb_norm = float(np.linalg.norm(embedding))

            # Qualitaets-Flags
            conf_ok = face_conf >= MIN_CONF
            size_ok = face_w >= MIN_FACE_PX and face_h >= MIN_FACE_PX
            bright_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
            emb_ok = embedding is not None and emb_norm >= MIN_EMB_NORM

            detail_parts = [
                f"Conf: {int(face_conf * 100)}%" + (" \u2713" if conf_ok else " \u2717"),
                f"Gr: {face_w}x{face_h}" + (" \u2713" if size_ok else " \u2717"),
                f"Hell: {brightness}" + (" \u2713" if bright_ok else " \u2717"),
                f"Emb: {'OK' if emb_ok else 'FEHLT'} (n={emb_norm:.2f})",
            ]
            detail_str = " | ".join(detail_parts)
            self._teach_result["detail"] = detail_str
            logger.info(f"[TEACH] Qualitaet: {detail_str}")

            # --- Fehlgrund bestimmen ---
            fail_reason = None
            if not conf_ok:
                fail_reason = f"Gesicht unscharf ({int(face_conf*100)}%)"
            elif not size_ok:
                fail_reason = f"Gesicht zu klein ({face_w}x{face_h}px)"
            elif not bright_ok:
                fail_reason = "Zu dunkel" if brightness < MIN_BRIGHTNESS else "Zu hell"
            elif not emb_ok:
                fail_reason = f"Kein Embedding (norm={emb_norm:.1f})"

            if fail_reason:
                self._teach_result["status"] = "retry"
                self._teach_result["reason"] = fail_reason
                self._led_flash("rot", 0.5)
                logger.info(f"[TEACH] FAIL: {fail_reason}")
                if attempt < MAX_RETRIES:
                    time.sleep(1.0)
                continue

            # --- ALLE CHECKS BESTANDEN — Bild speichern ---
            teach_dir = os.path.expanduser("~/moloch/media/teach")
            os.makedirs(teach_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(teach_dir, f"teach_{ts}.jpg")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"[TEACH] Foto gespeichert: {path}")

            # --- Embedding in face_embeddings.json speichern ---
            sim_score = 0.0
            try:
                emb_normalized = embedding / np.linalg.norm(embedding)

                # Similarity gegen vorhandene Embeddings testen
                matched_name, matched_sim = self._inference._match_face(embedding)
                sim_score = matched_sim

                # Embedding zu DB hinzufuegen
                emb_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "face_embeddings.json"
                )
                emb_key = f"Markus#teach_{ts}"
                if os.path.exists(emb_path):
                    with open(emb_path, 'r') as f:
                        db = json.load(f)
                else:
                    db = {}
                db[emb_key] = emb_normalized.tolist()
                with open(emb_path, 'w') as f:
                    json.dump(db, f, indent=2)
                logger.info(f"[TEACH] Embedding gespeichert: {emb_key} (sim={sim_score:.3f})")

                # Face-DB im RAM neu laden
                if hasattr(self._inference, '_load_face_db_from_disk'):
                    self._inference._face_db = self._inference._load_face_db_from_disk()
                    logger.info("[TEACH] Face-DB im RAM aktualisiert")

            except Exception as e:
                logger.warning(f"[TEACH] Embedding-Speicherung fehlgeschlagen: {e}")

            # --- LED weiss + Erfolg melden ---
            self._led_flash("weiss", 1.0)
            # Kamera-Flutlicht (weisse LED) kurz aufblitzen — sichtbarer Feedback
            if self._led:
                threading.Thread(target=self._led.flash_white, daemon=True).start()
            self._teach_result = {
                "status": "success",
                "attempt": attempt,
                "detail": detail_str,
                "similarity": round(sim_score, 3),
                "path": path,
                "reason": "",
            }
            return  # Fertig!

        # --- Alle 3 Versuche fehlgeschlagen ---
        logger.warning("[TEACH] Alle 3 Versuche fehlgeschlagen")
        self._led_flash("gelb", 1.0)
        self._teach_result["status"] = "failed"
        self._teach_result["reason"] = self._teach_result.get("reason", "unbekannt")
        self._teach_result["detail"] = f"3x fehlgeschlagen: {self._teach_result['reason']}"

        # TTS-Meldung
        if self._voice_pipeline and hasattr(self._voice_pipeline, '_speak'):
            try:
                self._voice_pipeline._speak(
                    "Ich konnte kein gutes Foto machen. Bitte vor die Kamera stellen."
                )
            except Exception as e:
                logger.debug(f"[TEACH] TTS fehlgeschlagen: {e}")

    def _apply_ptz_to_tracker(self):
        """PTZ-Settings live auf den AutonomousTracker anwenden."""
        tracker = self._cam._tracker
        if not tracker:
            return
        cfg = tracker.config
        cfg.tracking_speed = getattr(self, '_ptz_tracking_speed', cfg.tracking_speed)
        cfg.search_speed = getattr(self, '_ptz_search_speed', cfg.search_speed)
        cfg.pan_limit_min = getattr(self, '_ptz_pan_limit_min', cfg.pan_limit_min)
        cfg.pan_limit_max = getattr(self, '_ptz_pan_limit_max', cfg.pan_limit_max)
        cfg.tilt_limit_min = getattr(self, '_ptz_tilt_limit_min', cfg.tilt_limit_min)
        cfg.tilt_limit_max = getattr(self, '_ptz_tilt_limit_max', cfg.tilt_limit_max)
        # Home-Position fuer Park-Verhalten (statt hardcoded 0,0)
        cfg.home_pan = getattr(self, '_ptz_home_pan', 0.0)
        cfg.home_tilt = getattr(self, '_ptz_home_tilt', 0.0)
        # Basis-Werte fuer dynamische Anpassung aktualisieren
        tracker._base_tracking_speed = cfg.tracking_speed
        logger.info(f"[PTZ] Tracker updated: speed={cfg.tracking_speed:.2f} "
                    f"home=({cfg.home_pan:+.1f},{cfg.home_tilt:+.1f}) "
                    f"limits=[{cfg.pan_limit_min:.1f},{cfg.pan_limit_max:.1f}]")

    # =========================================================================
    # TAPPAS → Perception Loop (PFrame → PerceptionEngine/CoreIntegrator/LED/Teachen)
    # =========================================================================

    def _tappas_perception_loop(self):
        """Pollt TAPPAS PFrames und fuettert den Rest des Systems.

        Ersetzt die Integrations-Logik die bei InferenceEngine INTERN laeuft.
        Hier extern, weil TappasPipeline nur Daten liefert (Separation of Concerns).
        Laeuft mit ~5 Hz (200ms) — schnell genug fuer LED/Perception, langsam genug fuer CPU.

        WICHTIG: Loop ueberlebt Pipeline-Restart! Prueft is_running() intern
        und wartet bis Watchdog die Pipeline neu startet.
        """
        POLL_INTERVAL = 0.2  # 5 Hz
        OFFLINE_POLL = 1.0   # 1 Hz wenn Pipeline offline
        _last_pframe_id = None
        _decision_counter = 0  # DecisionEngine nur jeden 5. Frame (= 1 Hz)

        while self.running:
            # Pipeline offline → langsam pollen, warten auf Watchdog-Restart
            if not self._inference.is_running():
                time.sleep(OFFLINE_POLL)
                _last_pframe_id = None  # Reset nach Reconnect
                continue

            try:
                pframe = self._inference.get_current_pframe()
                if pframe is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                # Duplikat-Check (gleiches Frame nicht doppelt verarbeiten)
                pf_id = id(pframe)
                if pf_id == _last_pframe_id:
                    time.sleep(POLL_INTERVAL)
                    continue
                _last_pframe_id = pf_id
                self._perception_buffer.push(pframe)

                # --- PerceptionEngine: Stage-Tracking (Option C: Modelle immer aktiv) ---
                if self._perception:
                    try:
                        ctx = {
                            "face_detected": getattr(pframe, 'face_detected', False),
                            "face_bbox": getattr(pframe, 'face_bbox', None),
                            "person_detected": getattr(pframe, 'person_detected', False),
                            "unknown_person": getattr(pframe, 'face_id', None) in (None, "unknown"),
                            "motion_level": 0.0,
                            "camera_moving": False,
                        }
                        self._perception.tick(ctx)
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] PerceptionEngine tick: {e}")

                # --- CoreIntegrator: Tension/Dominance Updates ---
                if self._core_integrator:
                    try:
                        face_id = getattr(pframe, 'face_id', None)
                        face_detected = getattr(pframe, 'face_detected', False)

                        if face_id and face_id != "unknown":
                            # Owner erkannt: Tension runter + Guardian-Boost
                            self._core_integrator.feed_event("markus_recognized", 0.5)
                            self._core_integrator.feed_event("unknown_person", 0.0)
                            self._core_integrator.feed_event("person_detected", 0.0)
                            # Owner-Override triggern
                            if not self._core_integrator._owner_confirmed:
                                self._core_integrator.owner_override()
                                logger.info(f"[TENSION] Owner-Override: {face_id}")
                        elif face_detected and (not face_id or face_id == "unknown"):
                            # Gesicht aber kein Match — NICHT markus resetten!
                            # Nur unknown_person setzen, markus bleibt stehen
                            self._core_integrator.feed_event("unknown_person", 0.3)
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] CoreIntegrator feed: {e}")

                # --- ReID: Person ohne face_id gegen Embedding-DB matchen ---
                if self._reid:
                    try:
                        person_det = getattr(pframe, 'person_detected', False)
                        f_id = getattr(pframe, 'face_id', None)
                        if person_det and not f_id:
                            # Face-Embedding aus TAPPAS-Detections holen
                            dets = self._inference.get_detections()
                            for det in dets:
                                if det.get("class") == "face" and det.get("embedding") is not None:
                                    reid_name, reid_sim = self._reid.match(det["embedding"])
                                    if reid_name:
                                        from core.moloch_event_bus import get_event_bus
                                        get_event_bus().publish(
                                            event_type="perception.reid",
                                            payload={
                                                "name": reid_name,
                                                "similarity": reid_sim,
                                                "source": "arcface_embedding",
                                            },
                                            source="moloch_service",
                                            priority=1,
                                        )
                                    break  # Nur bestes Face-Embedding verwenden
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] ReID: {e}")

                # --- Episodisches Gedaechtnis: Erkannte Personen speichern ---
                try:
                    _ep_face_id = getattr(pframe, 'face_id', None)
                    _ep_conf = getattr(pframe, 'face_confidence', 0.0)
                    if _ep_face_id and _ep_conf > 0.6:
                        # ArcFace-Embedding aus TAPPAS-Detections holen
                        _ep_dets = self._inference.get_detections()
                        for _ep_det in _ep_dets:
                            if _ep_det.get("class") == "face" and _ep_det.get("embedding") is not None:
                                _ep_meta = {
                                    "confidence": round(_ep_conf, 3),
                                }
                                if self._core_integrator:
                                    _ep_meta["mood"] = getattr(self._core_integrator, 'current_mood', None)
                                    _ep_meta["tension"] = getattr(self._core_integrator, 'tension', None)
                                get_episodic_memory().store_episode(
                                    person=_ep_face_id,
                                    event_type="face_recognized",
                                    embedding=_ep_det["embedding"],
                                    metadata=_ep_meta,
                                )
                                break
                except Exception as e:
                    logger.debug(f"[TAPPAS-PERC] EpisodicMemory: {e}")

                # --- LED: Markus-Erkennung Hysterese ---
                if self._led:
                    try:
                        face_id = getattr(pframe, 'face_id', None)
                        is_markus = face_id == "markus" if face_id else False
                        face_det = getattr(pframe, 'face_detected', False)
                        person_det = getattr(pframe, 'person_detected', False)
                        has_ctrl = self._cam._moloch_has_control if self._cam else True
                        p_mode = self._led.personality_mode or "guardian"
                        self._led.update_hysteresis(is_markus, face_det, person_det, has_ctrl, p_mode)
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] LED update: {e}")

                # --- Teachen: Snapshot-Trigger bei Gesichtserkennung ---
                if self._teachen and self._teachen.enabled:
                    try:
                        _tc_face = getattr(pframe, 'face_detected', False)
                        _tc_bbox = getattr(pframe, 'face_bbox', None)
                        _tc_name = getattr(pframe, 'face_id', None)
                        _tc_conf = getattr(pframe, 'face_confidence', 0.0)
                        if _tc_face and _tc_bbox and _tc_conf > 0.3:
                            _tc_frame = self._inference.get_annotated_frame()
                            if _tc_frame is not None:
                                _saved = self._teachen.maybe_snapshot_tappas(
                                    frame=_tc_frame,
                                    face_bbox_px=_tc_bbox,
                                    name=_tc_name,
                                    confidence=_tc_conf,
                                )
                                # BLITZ: weisse LED immer aufblitzen wenn Teachen Snapshot gespeichert
                                if _saved and self._led:
                                    import threading as _t
                                    _t.Thread(target=self._led.flash_white, daemon=True).start()
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] Teachen: {e}")

                # --- Teach-Modus: Auto-Trigger bei Gesichtserkennung ---
                if self._teach_mode_enabled and not self._teach_busy:
                    try:
                        _t_face = getattr(pframe, 'face_detected', False)
                        _t_conf = getattr(pframe, 'face_confidence', 0.0)
                        _t_now = time.time()
                        _t_cooldown_ok = (_t_now - self._teach_last_trigger) >= self._teach_cooldown
                        if _t_face and _t_conf >= 0.50 and _t_cooldown_ok:
                            # Frame + Detections JETZT holen (gleicher Moment wie PFrame)
                            _t_frame = self._inference.get_annotated_frame() if self._inference else None
                            _t_dets = self._inference.get_detections() if self._inference else []
                            if _t_frame is not None:
                                self._teach_busy = True
                                self._teach_last_trigger = _t_now
                                self._teach_result = {"status": "starting", "attempt": 0, "detail": ""}
                                threading.Thread(
                                    target=self._teach_smart_snapshot,
                                    args=(_t_frame, _t_dets),
                                    daemon=True, name="TeachAuto"
                                ).start()
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] Teach auto: {e}")

                # --- Awareness: RoomMap + Motion + Activity + Context ---
                if self._context_evaluator:
                    try:
                        _aw_face_id = getattr(pframe, 'face_id', None)
                        _aw_face_conf = getattr(pframe, 'face_confidence', 0.0)
                        _aw_person_det = getattr(pframe, 'person_detected', False)
                        _aw_person_count = getattr(pframe, 'person_count', 0)
                        _aw_face_bbox = getattr(pframe, 'face_bbox', None)
                        _aw_dist_ratio = getattr(pframe, 'distance_ratio', 0.0)

                        # RoomMap: PTZ-Position → Zone
                        _aw_zone = None
                        if self._room_map and self._cam:
                            try:
                                pos = self._cam.get_position()
                                if pos and hasattr(pos, 'pan'):
                                    self._room_map.update(pos.pan)
                                _aw_zone = self._room_map.current_zone
                            except Exception:
                                pass

                        # MotionAnalyzer: BBox-Deltas
                        _aw_motion = "stationary"
                        if self._motion_analyzer:
                            self._motion_analyzer.update(
                                person_detected=_aw_person_det,
                                face_bbox=_aw_face_bbox,
                                distance_ratio=_aw_dist_ratio,
                            )
                            _aw_motion = self._motion_analyzer.current_state

                        # ActivityAnalyzer: Kombinierte Signale
                        _aw_activity = "away"
                        if self._activity_analyzer:
                            self._activity_analyzer.update_signals(
                                person_count=_aw_person_count,
                                motion_state=_aw_motion,
                                music_energy=getattr(self, '_last_awareness_music_energy', 0.0),
                                zone=_aw_zone,
                                voice_active=getattr(self, '_last_awareness_voice_active', False),
                                face_id=_aw_face_id,
                            )
                            self._activity_analyzer.evaluate()
                            _aw_activity = self._activity_analyzer.current_activity

                        # ContextEvaluator: Gesamtbewertung
                        self._context_evaluator.evaluate(
                            room_zone=_aw_zone,
                            motion_state=_aw_motion,
                            activity=_aw_activity,
                            face_id=_aw_face_id,
                            face_confidence=_aw_face_conf,
                            person_count=_aw_person_count,
                            voice_active=getattr(self, '_last_awareness_voice_active', False),
                        )
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] Awareness: {e}")

                # --- MoodEngine: Signale updaten und evaluieren ---
                if self._mood_engine:
                    try:
                        _me_tension = 0.0
                        _me_dominance = 0.5
                        _me_zone = "guardian"
                        if self._core_integrator:
                            _me_tension = self._core_integrator.get_tension()
                            _me_dominance = self._core_integrator.get_dominance()
                            _me_zone = self._core_integrator.get_personality_zone()
                        self._mood_engine.update_signals(
                            tension=_me_tension,
                            dominance=_me_dominance,
                            personality_zone=_me_zone,
                            music_mood=getattr(self, '_last_mood_music_mood', None),
                            activity=self._activity_analyzer.current_activity if self._activity_analyzer else "away",
                            face_id=getattr(pframe, 'face_id', None),
                            music_energy=getattr(self, '_last_awareness_music_energy', 0.0),
                        )
                        self._mood_engine.evaluate()
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] MoodEngine: {e}")

                # --- Decision Engine + Atmosphere: Signale updaten ---
                if self._decision_engine:
                    try:
                        _de_tension = 0.0
                        _de_dominance = 0.0
                        _de_alertness = 0.0
                        _de_engagement = 0.0
                        _de_context_score = 0.0
                        if self._core_integrator:
                            _de_tension = self._core_integrator.get_tension()
                            _de_dominance = self._core_integrator.get_dominance()
                        if self._context_evaluator:
                            _ce_state = self._context_evaluator.get_state()
                            _de_alertness = _ce_state.get("alertness", 0.0)
                            _de_engagement = _ce_state.get("engagement", 0.0)
                            _de_context_score = _ce_state.get("score", 0.0)
                        self._decision_engine.update_signals(
                            mood=self._mood_engine.current_mood if self._mood_engine else "calm",
                            tension=_de_tension,
                            dominance=_de_dominance,
                            activity=self._activity_analyzer.current_activity if self._activity_analyzer else "away",
                            zone=self._room_map.current_zone if self._room_map else None,
                            face_id=getattr(pframe, 'face_id', None),
                            music_energy=getattr(self, '_last_awareness_music_energy', 0.0),
                            context_score=_de_context_score,
                            alertness=_de_alertness,
                            engagement=_de_engagement,
                            music_playing=getattr(self, '_last_awareness_music_energy', 0.0) > 0.01,
                        )
                        _decision_counter += 1
                        if _decision_counter >= 5:  # 1 Hz bei 5 Hz Loop
                            self._decision_engine.decide()
                            _decision_counter = 0
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] DecisionEngine: {e}")

                # --- Atmosphere Controller: Tageszeit + Face updaten ---
                if self._atmosphere:
                    try:
                        self._atmosphere.update_signals(
                            hour=time.localtime().tm_hour,
                            face_id=getattr(pframe, 'face_id', None),
                        )
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] Atmosphere: {e}")

                # --- Homeostasis: FPS-Wert updaten ---
                if self._homeostasis:
                    try:
                        fps = self._inference.get_fps() if hasattr(self._inference, 'get_fps') else 20.0
                        if isinstance(fps, dict):
                            fps = fps.get("current", 20.0)
                        self._homeostasis.set_fps(fps)
                    except Exception as e:
                        logger.debug(f"[TAPPAS-PERC] Homeostasis FPS: {e}")

            except Exception as e:
                logger.debug(f"[TAPPAS-PERC] Loop error: {e}")

            time.sleep(POLL_INTERVAL)

        logger.info("[TAPPAS] Perception-Loop beendet (Service gestoppt)")

    # =========================================================================
    # TAPPAS → Tracker Feed (ersetzt InferenceEngine-interne Tracker-Aufrufe)
    # =========================================================================

    def _tappas_tracker_feed_loop(self):
        """Pollt TAPPAS-Detections und fuettert den AutonomousTracker.

        Gleiche Logik wie in InferenceEngine._inference_loop():
        - Face hat IMMER Prioritaet (face_fed_to_tracker)
        - BBoxen sind normalisiert (0-1) → skaliert auf 640x640 Pixel
        - Laeuft mit ~15 Hz (alle 66ms) um Tracker nicht zu ueberlasten

        WICHTIG: Loop ueberlebt Pipeline-Restart! Wartet wenn Pipeline offline.
        """
        FEED_INTERVAL = 0.066  # ~15 Hz
        OFFLINE_POLL = 1.0     # 1 Hz wenn Pipeline offline
        FRAME_DIM = 640

        while self.running:
            # Pipeline offline → langsam pollen
            if not self._inference.is_running():
                time.sleep(OFFLINE_POLL)
                continue

            try:
                detections = self._inference.get_detections()
                if not detections:
                    time.sleep(FEED_INTERVAL)
                    continue

                tracker = self._cam._tracker
                if not tracker or not self._cam._autonomous_mode:
                    # G1-T01: Person erkannt aber Tracker noch nicht aktiv
                    # → Automatisch Takeover triggern
                    has_person = any(d.get("class") in ("person", "face")
                                    for d in detections)
                    if has_person and not self._cam._manual_mode:
                        if not getattr(self._cam, '_transitioning', False):
                            logger.info("[ACTION-BRIDGE] Person erkannt, Tracker inaktiv → Takeover!")
                            self._cam.moloch_takeover("Person erkannt (Auto-Bridge)")
                    time.sleep(FEED_INTERVAL)
                    continue

                # Face/Person trennen — Face hat Prioritaet
                face_dets = []
                person_dets = []
                for d in detections:
                    cls = d.get("class", "")
                    bbox = d.get("bbox", [0, 0, 0, 0])
                    conf = d.get("confidence", 0)
                    pixel_bbox = [bbox[0] * FRAME_DIM, bbox[1] * FRAME_DIM,
                                  bbox[2] * FRAME_DIM, bbox[3] * FRAME_DIM]
                    entry = {"bbox": pixel_bbox, "confidence": conf, "class": cls}
                    if cls == "face":
                        face_dets.append(entry)
                    elif cls == "person":
                        person_dets.append(entry)

                if face_dets:
                    tracker.update_detection(
                        detections=face_dets,
                        frame_width=FRAME_DIM, frame_height=FRAME_DIM
                    )
                elif person_dets:
                    tracker.update_detection(
                        detections=person_dets,
                        frame_width=FRAME_DIM, frame_height=FRAME_DIM
                    )

                if self._cam._waiting_for_first_detection:
                    self._cam._first_detection_event.set()
                if self._cam._moloch_has_control:
                    self._cam._last_interesting_time = time.time()
                    self._cam._takeover_found_something = True

            except Exception as e:
                logger.debug(f"[TAPPAS] Tracker feed error: {e}")

            time.sleep(FEED_INTERVAL)

        logger.info("[TAPPAS] Tracker-Feed Loop beendet (Service gestoppt)")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def init(self):
        """Hardware initialisieren: VDevice, Models, RTSP, Cloud."""
        logger.info("M.O.L.O.C.H. Service initialisiert...")

        # System Capabilities generieren (config/system_capabilities.json)
        try:
            import subprocess
            cap_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "scripts", "generate_capabilities.py")
            if os.path.exists(cap_script):
                subprocess.run(["python3", cap_script], timeout=30)
                logger.info("[INIT] system_capabilities.json aktualisiert")
        except Exception as e:
            logger.warning(f"[INIT] Capabilities-Generator fehlgeschlagen: {e}")

        # 0. Langzeitgedaechtnis initialisieren (SSD2, persistent)
        try:
            self._memory = get_memory()
            identity = self._memory.get_identity()
            logger.info(f"[INIT] Memory bereit: {identity.get('name', '?')} v{identity.get('version', '?')}")
        except Exception as e:
            self._memory = None
            logger.error(f"[INIT] Memory fehlgeschlagen: {e}")

        # 0b. MolochSprache Writer-Thread starten
        if self._sprache:
            self._sprache.start()
            self._sprache.log(self._sprache.build("STARTE", "Service",
                                                   version="4.0"))

        # 1. Hailo VDevice + Models (via ModelOrchestrator)
        # Bei TAPPAS: NPU wird von GStreamer Model Scheduler verwaltet — NICHT hier oeffnen!
        if not USE_TAPPAS:
            self._hailo_manager = get_hailo_manager()
            self._orchestrator._hailo_manager = self._hailo_manager
            self._hailo_manager.acquire_for_vision(timeout=10.0)
            self._orchestrator.load_models()
            for name in self._orchestrator._models:
                logger.info(f"Modell geladen: {name} ({len(self._orchestrator._output_names[name])} outputs)")

            # 1b. Whisper VDevice uebergeben (On-Demand: wird erst bei PTT geladen)
            try:
                from core.speech.hailo_whisper import get_whisper
                whisper = get_whisper()
                whisper.set_vdevice(self._orchestrator.vdevice)
            except Exception as e:
                logger.error(f"[INIT] Whisper VDevice-Uebergabe fehlgeschlagen: {e}")

            # 1c. NPU Idle-Modus: NUR yolov8m beim Start konfigurieren
            # Weitere Modelle werden von PerceptionEngine stufenweise dazugeschaltet:
            #   IDLE (nur yolov8m) → PERSON (+scrfd) → FACE (+arcface)
            idle_models = ["yolov8m"]
            if self._perception:
                idle_models = self._perception.get_stage_models()  # Stage "idle" → ["yolov8m"]
            for name in idle_models:
                if name in self._orchestrator._models:
                    self._orchestrator.configure(name)
            self._orchestrator.sync_flags()
            self._inference.sync_flags_from_npu()
            logger.info(f"[INIT] Boot → Idle-Modus — nur Person-Waechter aktiv: {self._orchestrator.active_models}")
        else:
            logger.info("[INIT] TAPPAS aktiv — NPU wird von GStreamer Model Scheduler verwaltet")

        # 2. Face DB (via InferenceEngine)
        self._inference.reload_face_db()

        # 3. RTSP (via CameraManager) — NICHT bei TAPPAS (GStreamer oeffnet eigenen rtspsrc)
        if not USE_TAPPAS:
            self._cam.start_rtsp()
        else:
            logger.info("[INIT] TAPPAS aktiv — ueberspringe CameraManager RTSP (GStreamer hat eigenen rtspsrc)")

        # 4. Cloud (via CameraManager, im Hintergrund)
        threading.Thread(target=self._cam.connect_cloud, daemon=True).start()

        # 5. Music Visualizer (Audio-Analyse fuer Avatar, PipeWire Capture)
        self._music_vis = None
        try:
            from core.audio.music_visualizer import get_music_visualizer
            self._music_vis = get_music_visualizer()
            self._music_vis.start()
            logger.info("[INIT] MusicVisualizer gestartet")
        except Exception as e:
            self._music_vis = None
            logger.warning(f"[INIT] MusicVisualizer nicht verfuegbar: {e}")

        # 6. Action Bridge FSM (Perception → Entscheidung → Action)
        self._action_bridge = None
        try:
            from core.action_bridge import get_action_bridge
            self._action_bridge = get_action_bridge()
            logger.info("[INIT] ActionBridge initialisiert")
        except Exception as e:
            self._action_bridge = None
            logger.warning(f"[INIT] ActionBridge nicht verfuegbar: {e}")

        # 7. Person ReID (Embedding-basierte Wiedererkennung ohne Gesicht)
        self._reid = None
        try:
            from core.memory.person_reid import get_reid
            self._reid = get_reid()
            logger.info(f"[INIT] PersonReID bereit: {len(self._reid.get_identities())} Identitaeten")
        except Exception as e:
            self._reid = None
            logger.warning(f"[INIT] PersonReID nicht verfuegbar: {e}")

        # 8. Spotify Bridge (Track-Info + Audio Features → Event Bus)
        self._spotify_bridge = None
        try:
            from core.music.spotify_bridge import get_spotify_bridge
            self._spotify_bridge = get_spotify_bridge()
            self._spotify_bridge.start()
            logger.info("[INIT] SpotifyBridge gestartet")
        except Exception as e:
            self._spotify_bridge = None
            logger.warning(f"[INIT] SpotifyBridge nicht verfuegbar: {e}")

        # 8b. MicModeController + MusicListener (Music Soul System)
        self._mic_mode_ctrl = None
        self._music_listener = None
        try:
            from core.audio.mic_mode_controller import get_mic_mode_controller
            from core.audio.music_listener import get_music_listener
            self._mic_mode_ctrl = get_mic_mode_controller()
            self._mic_mode_ctrl.start()
            self._music_listener = get_music_listener()
            self._music_listener.start()
            logger.info("[INIT] MicModeController + MusicListener gestartet")
        except Exception as e:
            logger.warning(f"[INIT] Music Soul nicht verfuegbar: {e}")

        # 9. Awareness Module (Gate 3: Situational Awareness)
        self._room_map = None
        self._motion_analyzer = None
        self._activity_analyzer = None
        self._context_evaluator = None
        try:
            self._room_map = get_room_map()
            self._motion_analyzer = get_motion_analyzer()
            self._activity_analyzer = get_activity_analyzer()
            self._context_evaluator = get_context_evaluator()
            self._last_awareness_music_energy = 0.0
            self._last_awareness_voice_active = False
            logger.info("[INIT] Awareness Module bereit (RoomMap/Motion/Activity/Context)")
        except Exception as e:
            logger.warning(f"[INIT] Awareness Module nicht verfuegbar: {e}")

        # 10. Emergent Personality (Gate 4: Mood Engine + Behavior Rules)
        self._tension_integrator = None
        self._mood_engine = None
        self._behavior_rules = None
        try:
            self._tension_integrator = get_tension_integrator()
            if self._core_integrator:
                self._tension_integrator.set_core_integrator(self._core_integrator)
            self._mood_engine = get_mood_engine()
            self._behavior_rules = get_behavior_rules()
            self._last_mood_music_mood = None
            logger.info("[INIT] Emergent Personality bereit (Tension/Mood/Behavior)")
        except Exception as e:
            logger.warning(f"[INIT] Emergent Personality nicht verfuegbar: {e}")

        # 11. Autonomy (Gate 5: Decision Engine + Atmosphere + Homeostasis + Night Cycle)
        self._decision_engine = None
        self._atmosphere = None
        self._homeostasis = None
        self._night_cycle = None
        try:
            self._decision_engine = get_decision_engine()
            self._atmosphere = get_atmosphere_controller()
            self._homeostasis = get_homeostasis()
            self._night_cycle = get_night_cycle()
            logger.info("[INIT] Autonomy Module bereit (Decision/Atmosphere/Homeostasis/NightCycle)")
        except Exception as e:
            logger.warning(f"[INIT] Autonomy Module nicht verfuegbar: {e}")

        # RGB-LED (ESP32 WS2812) starten
        try:
            from core.moloch_event_bus import get_event_bus
            bus = get_event_bus()
            self._rgb_led = get_rgb_led(event_bus=bus)
            self._rgb_led.start()
            logger.info("[INIT] RGB-LED Controller gestartet (ESP32)")
        except Exception as e:
            self._rgb_led = None
            logger.warning(f"[INIT] RGB-LED nicht verfuegbar: {e}")

        self._update_status("M.O.L.O.C.H. Service bereit")

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

        # Action Bridge FSM starten (1 Hz Tick-Loop)
        if self._action_bridge:
            self._action_bridge.start()
            logger.info("[START] ActionBridge gestartet")

        # Spotify Bridge Events → CoreIntegrator (Musik beeinflusst Tension)
        if self._spotify_bridge and self._core_integrator:
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()

                def _on_music_mood(event):
                    mood = event.get("payload", {}).get("mood", "neutral")
                    if mood in ("aggressive", "dark"):
                        self._core_integrator.feed_event("environmental_stress", 0.05)
                    elif mood in ("euphoric", "neutral"):
                        self._core_integrator.feed_event("respect_score", 0.05)

                def _on_music_features(event):
                    energy = event.get("payload", {}).get("features", {}).get("energy", 0.0)
                    if energy > 0.8:
                        self._core_integrator.feed_event("environmental_stress", 0.03)

                bus.subscribe("music_mood_changed", _on_music_mood)
                bus.subscribe("music_features_received", _on_music_features)
                logger.info("[START] Spotify Bridge Events → CoreIntegrator registriert")
            except Exception as e:
                logger.warning(f"[START] Spotify Bridge Event-Subscriber fehlgeschlagen: {e}")

        # Music Memory: Track-Person-Mood Assoziationen speichern
        if self._spotify_bridge:
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()
                self._last_music_mood = None

                def _on_mood_for_memory(event):
                    self._last_music_mood = event.get("payload", {}).get("mood")

                def _on_track_for_memory(event):
                    payload = event.get("payload", {})
                    track_id = payload.get("uri", "")
                    track_name = f"{payload.get('artist', '?')} - {payload.get('track', '?')}"
                    # Aktuelle Person aus letztem PFrame
                    face_id = None
                    tension = None
                    try:
                        pframe = self._inference.get_current_pframe() if self._inference else None
                        if pframe:
                            face_id = getattr(pframe, 'face_id', None)
                    except Exception:
                        pass
                    if self._core_integrator:
                        tension = getattr(self._core_integrator, 'tension', None)
                    if face_id and face_id != "unknown":
                        get_music_memory().store_association(
                            track_id=track_id,
                            track_name=track_name,
                            person=face_id,
                            mood=self._last_music_mood,
                            tension=tension,
                        )

                bus.subscribe("music_mood_changed", _on_mood_for_memory)
                bus.subscribe("music_track_started", _on_track_for_memory)
                logger.info("[START] Music Memory Events registriert")
            except Exception as e:
                logger.warning(f"[START] Music Memory Event-Subscriber fehlgeschlagen: {e}")

        # Awareness: context_update → CoreIntegrator + Music Energy fuer ActivityAnalyzer
        if self._context_evaluator and self._core_integrator:
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()

                def _on_context_update(event):
                    p = event.get("payload", {})
                    alertness = p.get("alertness", 0.0)
                    # Hohe Alertness → Tension steigt, niedrige → Tension sinkt
                    if alertness > 0.6:
                        self._core_integrator.feed_event("environmental_stress", 0.02)
                    elif alertness < 0.3:
                        self._core_integrator.feed_event("respect_score", 0.02)

                def _on_music_energy_for_awareness(event):
                    energy = event.get("payload", {}).get("features", {}).get("energy", 0.0)
                    self._last_awareness_music_energy = energy

                bus.subscribe("context_update", _on_context_update)
                bus.subscribe("music_features_received", _on_music_energy_for_awareness)
                logger.info("[START] Awareness → CoreIntegrator Events registriert")
            except Exception as e:
                logger.warning(f"[START] Awareness Event-Subscriber fehlgeschlagen: {e}")

        # Emergent Personality: TensionIntegrator + MoodEngine + BehaviorRules
        if self._mood_engine:
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()

                # TensionIntegrator: Awareness Events → CoreIntegrator
                if self._tension_integrator:
                    bus.subscribe("context_update", self._tension_integrator.on_context_update)
                    bus.subscribe("activity_changed", self._tension_integrator.on_activity_changed)
                    bus.subscribe("motion_state_changed", self._tension_integrator.on_motion_state_changed)

                # Music Mood fuer MoodEngine cachen
                def _on_mood_for_personality(event):
                    self._last_mood_music_mood = event.get("payload", {}).get("mood")
                bus.subscribe("music_mood_changed", _on_mood_for_personality)

                # MoodEngine: mood_changed → BehaviorRules evaluieren
                def _on_mood_changed(event):
                    if not self._behavior_rules or not self._core_integrator:
                        return
                    payload = event.get("payload", {})
                    self._behavior_rules.evaluate(
                        mood=payload.get("mood", "calm"),
                        tension=payload.get("tension", 0.0),
                        dominance=payload.get("dominance", 0.5),
                        personality_zone=self._core_integrator.get_personality_zone(),
                        face_id=payload.get("face_id"),
                        music_energy=getattr(self, '_last_awareness_music_energy', 0.0),
                    )
                bus.subscribe("mood_changed", _on_mood_changed)

                # BehaviorTrigger → LED/Sirene/Personality-Zone ausfuehren
                def _on_behavior_trigger(event):
                    payload = event.get("payload", {})
                    action = payload.get("action", "")
                    led_cmd = payload.get("led")
                    sirene = payload.get("sirene", False)

                    # LED steuern
                    if self._led and led_cmd:
                        try:
                            if led_cmd == "on":
                                self._led.on()
                            elif led_cmd == "off":
                                self._led.off()
                            elif led_cmd == "blink":
                                self._led.blink(count=4, interval=0.4)
                            elif led_cmd == "blink_fast":
                                self._led.blink(count=8, interval=0.15)
                            elif led_cmd == "blink_slow":
                                self._led.blink(count=3, interval=0.6)
                        except Exception:
                            pass

                    # Sirene (CoreIntegrator Impulse-Flag)
                    if sirene and self._core_integrator:
                        try:
                            self._core_integrator.set_impulse_flag()
                        except Exception:
                            pass

                bus.subscribe("behavior_trigger", _on_behavior_trigger)
                logger.info("[START] Emergent Personality Events registriert")
            except Exception as e:
                logger.warning(f"[START] Emergent Personality Event-Subscriber fehlgeschlagen: {e}")

        # Autonomy: Atmosphere Controller Events + Homeostasis + Night Cycle + Decision Engine
        if self._atmosphere:
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()
                bus.subscribe("activity_changed", self._atmosphere.on_activity_changed)
                bus.subscribe("mood_changed", self._atmosphere.on_mood_changed)

                # Atmosphere → Decision Engine: atmosphere_changed → LED/Music ausfuehren
                if self._decision_engine:
                    def _on_atmosphere_changed(event):
                        profile = event.get("payload", {}).get("profile", {})
                        # LED-Kommando aus Atmosphaere-Profil ausfuehren
                        led_cmd = profile.get("led")
                        if self._led and led_cmd:
                            try:
                                if led_cmd == "on":
                                    self._led.on()
                                elif led_cmd == "off":
                                    self._led.off()
                                elif led_cmd == "blink":
                                    self._led.blink(count=4, interval=0.4)
                                elif led_cmd == "blink_slow":
                                    self._led.blink(count=3, interval=0.6)
                            except Exception:
                                pass
                    bus.subscribe("atmosphere_changed", _on_atmosphere_changed)

                logger.info("[START] Atmosphere Controller Events registriert")
            except Exception as e:
                logger.warning(f"[START] Atmosphere Event-Subscriber fehlgeschlagen: {e}")

        # Homeostasis: Background-Monitoring starten
        if self._homeostasis:
            self._homeostasis.start()
            logger.info("[START] Homeostasis Monitoring gestartet")

        # Diagnostics HTTP-Server starten (Port 5000)
        try:
            from core.diagnostics import start_diagnostics_server
            start_diagnostics_server(port=5000)
            logger.info("[START] Diagnostics HTTP-Server auf Port 5000")
        except Exception as e:
            logger.warning(f"[START] Diagnostics Server fehlgeschlagen: {e}")

        # Night Cycle: Background-Thread starten
        if self._night_cycle:
            self._night_cycle.start()
            logger.info("[START] Night Cycle Thread gestartet")

        # Event Trace Logger (Gate 3/4/5 Debug)
        try:
            from core.moloch_event_bus import get_event_bus
            _trace_bus = get_event_bus()
            for _evt in ("zone_entered", "motion_state_changed", "activity_changed",
                         "context_update", "mood_changed", "behavior_trigger",
                         "atmosphere_changed", "decision_made", "health_alert",
                         "night_cycle_complete"):
                _trace_bus.subscribe(_evt, lambda e, et=_evt: log_event(et, e.get("payload", {})))
            logger.info("[START] Event Trace Logger registriert")
        except Exception as e:
            logger.debug(f"[START] Event Trace Logger: {e}")

        # Inference Loop — bei TAPPAS mit 3s Delay (ONVIF muss zuerst verbinden fuer PTZ)
        if USE_TAPPAS:
            def _start_tappas_delayed():
                logger.info("[START] TAPPAS: Warte 3s auf ONVIF-Verbindung...")
                time.sleep(3)
                # Retry-Logik: Pipeline kann beim Boot durch RTSP-Timing crashen
                max_retries = 5
                for attempt in range(1, max_retries + 1):
                    try:
                        self._inference.start()
                        logger.info("[START] TAPPAS Pipeline gestartet")
                        # Teach-Modus: Scheduler nochmal erzwingen NACH Pipeline-Start
                        if self._teach_mode_enabled:
                            self._inference.force_all_active(True)
                            logger.info("[START] Teach-Modus AN → SCRFD/ArcFace erzwungen")
                        break
                    except Exception as e:
                        logger.error(f"[START] TAPPAS Pipeline Versuch {attempt}/{max_retries} fehlgeschlagen: {e}")
                        if attempt < max_retries:
                            wait = 3 * attempt
                            logger.info(f"[START] TAPPAS Retry in {wait}s...")
                            time.sleep(wait)
                        else:
                            logger.error("[START] TAPPAS Pipeline konnte nach 5 Versuchen nicht gestartet werden!")
                            return
                # Watchdog: Prueft ob Pipeline nach Start ueberlebt, startet bei Crash neu
                # Exponentielles Backoff + RTSP-Probe vor Neustart
                def _rtsp_probe() -> bool:
                    """Prueft ob RTSP-Stream erreichbar ist bevor Pipeline gestartet wird."""
                    import subprocess
                    rtsp_url = self._inference._rtsp_url
                    try:
                        result = subprocess.run(
                            ["ffprobe", "-rtsp_transport", "tcp", "-v", "error",
                             "-timeout", "5000000", rtsp_url],
                            capture_output=True, timeout=8
                        )
                        return result.returncode == 0
                    except Exception:
                        return False

                def _tappas_watchdog():
                    time.sleep(5)  # Pipeline 5s laufen lassen
                    consecutive_fails = 0
                    while self.running:
                        if not self._inference._running:
                            consecutive_fails += 1
                            # Backoff: 5s, 10s, 15s, 20s, max 30s
                            wait = min(5 * consecutive_fails, 30)
                            logger.warning(f"[WATCHDOG] TAPPAS Pipeline ist tot — Neustart in {wait}s (Versuch {consecutive_fails})...")
                            try:
                                self._inference.stop()
                            except Exception:
                                pass
                            time.sleep(wait)
                            # RTSP-Probe: Stream erreichbar?
                            if not _rtsp_probe():
                                logger.warning(f"[WATCHDOG] RTSP-Stream nicht erreichbar — warte...")
                                continue
                            try:
                                self._inference.start()
                                logger.info(f"[WATCHDOG] TAPPAS Pipeline neu gestartet (nach {consecutive_fails} Versuchen)")
                                consecutive_fails = 0
                            except Exception as e:
                                logger.error(f"[WATCHDOG] TAPPAS Neustart fehlgeschlagen: {e}")
                                continue
                        else:
                            consecutive_fails = 0
                        time.sleep(5)
                threading.Thread(target=_tappas_watchdog, daemon=True, name="TappasWatchdog").start()
                # Tracker-Feed Thread starten (liest Detections aus Pipeline → Tracker)
                threading.Thread(target=self._tappas_tracker_feed_loop, daemon=True,
                                 name="TappasTrackerFeed").start()
                logger.info("[START] TAPPAS Tracker-Feed Loop gestartet")
                # Perception-Loop: PFrame → PerceptionEngine/CoreIntegrator/LED/Teachen
                threading.Thread(target=self._tappas_perception_loop, daemon=True,
                                 name="TappasPerceptionLoop").start()
                logger.info("[START] TAPPAS Perception-Loop gestartet (5 Hz)")
            threading.Thread(target=_start_tappas_delayed, daemon=True, name="TappasDelayedStart").start()
        else:
            self._inference.start()

        # Kamera-Status + IPC Polling (via CameraManager)
        self._cam.start_cam_status_loop(write_status_callback=self._write_status_json)

        # Panel IPC Command Polling
        threading.Thread(target=self._poll_panel_cmds, daemon=True, name="PanelCmdPoll").start()

        # Autonomous Mode + Watchdog (via CameraManager)
        # Tentakel-Modus: Smart Tracking AN, Guardian wartet auf Bewegung
        # NICHT enable_autonomous() beim Boot — Tracker startet erst bei Takeover
        logger.info("[START] Tentakel-Modus: Smart Tracking aktiv, Guardian wartet")

        # PTZ-Settings auf Tracker anwenden (nach enable_autonomous)
        def _apply_ptz_delayed():
            import time as _t
            _t.sleep(3)  # Tracker braucht kurz zum Starten
            self._apply_ptz_to_tracker()
        threading.Thread(target=_apply_ptz_delayed, daemon=True, name="PTZSettingsApply").start()

        # PTZ-Tracker (Bewegungs-Analyse, restless_score -> CoreIntegrator)
        self._ptz_tracker = get_ptz_tracker()
        self._ptz_tracker.start()
        logger.info("[START] PTZ-Tracker gestartet (Bewegungs-Analyse)")

        # PiPower5 Power Monitor (Akku-Bewusstsein, Tentakel)
        try:
            from core.hardware.power_monitor import get_power_monitor
            self._power_monitor = get_power_monitor()
            self._power_monitor.set_core_integrator(self._core_integrator)
            self._power_monitor.start()
        except Exception as e:
            self._power_monitor = None
            logger.warning(f"[START] PowerMonitor fehlgeschlagen: {e}")

        # G1-T03: Auto-Resume Callback — TTS Spruch bei Manuell→Autonom
        try:
            from core.ptz_arbiter import get_ptz_arbiter
            arbiter = get_ptz_arbiter()
            def _on_auto_resume():
                logger.info("[ARBITER] Auto-Resume: Manuell -> Autonom, TTS Spruch")
                try:
                    from core.personality import get_personality_engine, MolochEvent
                    engine = get_personality_engine()
                    engine.speak_event(MolochEvent.TRACKING_RESUMED)
                except Exception as e:
                    logger.debug(f"[ARBITER] TTS Auto-Resume Fehler: {e}")
            arbiter.on_auto_resume = _on_auto_resume
        except Exception as e:
            logger.warning(f"[START] Arbiter Auto-Resume Callback Fehler: {e}")

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

        # Watchdog (Frozen Frame Detection) — NICHT bei TAPPAS (GStreamer managed Stream)
        if not USE_TAPPAS:
            self._cam.start_watchdog()
        else:
            logger.info("[START] TAPPAS aktiv — ueberspringe RTSP Watchdog")

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

    def stop(self):
        """Sauberes Herunterfahren."""
        logger.info("M.O.L.O.C.H. Service wird gestoppt...")
        self.running = False
        self._cam.running = False
        self._inference.stop()

        # Langzeitgedaechtnis: Core State SOFORT sichern
        if self._memory and self._core_integrator:
            try:
                state = self._core_integrator.get_state()
                state["personality_zone"] = self._core_integrator.get_personality_zone()
                self._memory.save_core_state(state)
                logger.info("[STOP] Core State persistent gesichert")
            except Exception as e:
                logger.error(f"[STOP] Core State Speichern fehlgeschlagen: {e}")

        # MolochSprache: Letzter Satz + Queue leeren
        if self._sprache:
            try:
                self._sprache.log(self._sprache.build("STOPPE", "Service"))
                self._sprache.stop()
                logger.info("[STOP] MolochSprache gestoppt")
            except Exception:
                pass

        # Core Integrator stoppen
        if self._core_integrator:
            try:
                self._core_integrator.stop()
            except Exception:
                pass

        # MusicVisualizer stoppen
        if hasattr(self, '_music_vis') and self._music_vis:
            try:
                self._music_vis.stop()
                logger.info("[STOP] MusicVisualizer gestoppt")
            except Exception:
                pass

        # RGB-LED stoppen
        if self._rgb_led:
            try:
                self._rgb_led.stop()
                logger.info("[STOP] RGB-LED gestoppt")
            except Exception:
                pass

        # ActionBridge stoppen
        if hasattr(self, '_action_bridge') and self._action_bridge:
            try:
                self._action_bridge.stop()
                logger.info("[STOP] ActionBridge gestoppt")
            except Exception:
                pass

        # MusicListener + MicModeController stoppen
        if hasattr(self, '_music_listener') and self._music_listener:
            try:
                self._music_listener.stop()
                logger.info("[STOP] MusicListener gestoppt")
            except Exception:
                pass
        if hasattr(self, '_mic_mode_ctrl') and self._mic_mode_ctrl:
            try:
                self._mic_mode_ctrl.stop()
                logger.info("[STOP] MicModeController gestoppt")
            except Exception:
                pass

        # SpotifyBridge stoppen
        if hasattr(self, '_spotify_bridge') and self._spotify_bridge:
            try:
                self._spotify_bridge.stop()
                logger.info("[STOP] SpotifyBridge gestoppt")
            except Exception:
                pass

        # Diagnostics HTTP-Server stoppen
        try:
            from core.diagnostics import stop_diagnostics_server
            stop_diagnostics_server()
        except Exception:
            pass

        # Autonomy Module stoppen (Gate 5)
        if hasattr(self, '_homeostasis') and self._homeostasis:
            try:
                self._homeostasis.stop()
                logger.info("[STOP] Homeostasis gestoppt")
            except Exception:
                pass
        if hasattr(self, '_night_cycle') and self._night_cycle:
            try:
                self._night_cycle.stop()
                logger.info("[STOP] NightCycle gestoppt")
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
            fps_snapshot = self._inference.get_fps()
            with self._ctx_lock:
                active_models = list(self._active_ctx.keys())

            _inf = self._inference
            # TAPPAS: Modelle laufen immer alle parallel — getattr fuer Kompatibilitaet
            status = {
                "scrfd_active": getattr(_inf, 'scrfd_active', True),
                "arcface_active": getattr(_inf, 'arcface_active', True),
                "yolo_active": getattr(_inf, 'yolo_active', True),
                "hand_active": getattr(_inf, 'hand_active', False),
                "pose_active": getattr(_inf, 'pose_active', False),
                "npu_paused": self._orchestrator._npu_paused,
                "active_models": (
                    # TAPPAS: Scheduler-basiert aus pframe, nicht aus _active_ctx
                    getattr(_inf.get_current_pframe(), 'active_models', ["yolov8m"])
                    if USE_TAPPAS and hasattr(_inf, 'get_current_pframe')
                    else (active_models if active_models else [])),
                "autonomous_mode": self._cam._autonomous_mode,
                "manual_mode": self._cam._manual_mode,
                "moloch_has_control": self._cam._moloch_has_control,
                "tentakel_enabled": self._cam._tentakel_enabled,
                "teachen_enabled": self._teachen.enabled if self._teachen else False,
                "power": self._power_monitor.get_status() if getattr(self, '_power_monitor', None) else {},
                "frame_age": round(time.time() - self._cam._last_frame_write, 1) if self._cam._last_frame_write else -1,
                "frozen_restarts": self._cam._frozen_restart_count,
                "fps": {k: round(v, 1) for k, v in fps_snapshot.items()},
                "thresholds": {
                    "scrfd_conf": getattr(_inf, 'scrfd_conf_val', 0.6),
                    "scrfd_nms": getattr(_inf, 'scrfd_nms_val', 0.5),
                    "arcface_thresh": getattr(_inf, 'arcface_thresh_val', 0.45),
                    "yolo_conf": getattr(_inf, 'yolo_conf_val', 0.5),
                },
                "led_markus_on": self._led.markus_on,
                "led_personality_mode": self._led.personality_mode,
                "cloud": self._cloud_state,
                "audio": {
                    "mic_gain": self._saved_mic_gain,
                    "noise_gate_db": self._saved_noise_gate,
                    "agc_enabled": self._saved_agc,
                    "level": self._audio_level,
                },
                "voice": self._voice_pipeline.get_state() if self._voice_pipeline else {},
            }
            # Pipeline-Status fuer Panel (Kamera offline/online Anzeige)
            status["pipeline_alive"] = _inf.is_running()

            # TAPPAS: PFrame-Daten in Status einpflegen (Panel braucht person/face/mode)
            if USE_TAPPAS:
                pframe = _inf.get_current_pframe()
                if pframe:
                    status["person_detected"] = getattr(pframe, 'person_detected', False)
                    status["face_detected"] = getattr(pframe, 'face_detected', False)
                    status["face_id"] = getattr(pframe, 'face_id', None)
                    status["face_confidence"] = round(getattr(pframe, 'face_confidence', 0.0), 3)
                    status["face_similarity"] = round(getattr(pframe, 'face_similarity', 0.0), 3)
                    status["mode"] = "tappas"
                if hasattr(self._inference, 'get_npu_sched_mode'):
                    status["npu_sched_mode"] = self._inference.get_npu_sched_mode()

            # Einpraegen Status (TAPPAS Enrollment oder Legacy)
            if hasattr(self._inference, 'get_enrollment_status'):
                enr = self._inference.get_enrollment_status()
                status["einpraegen_running"] = enr["active"]
                if enr["active"]:
                    status["einpraegen_progress"] = f"Face {enr['collected']}/{enr['target']}"
                else:
                    status["einpraegen_progress"] = ""
                status["einpraegen_done"] = enr.get("done", False)
            elif self._einpraegen:
                status["einpraegen_running"] = self._einpraegen.is_running
                status["einpraegen_progress"] = self._einpraegen.progress
                status["einpraegen_done"] = self._einpraegen.is_done

            # Teach-Modus + Smart Snapshot Status
            status["teach_mode_enabled"] = getattr(self, '_teach_mode_enabled', False)
            if hasattr(self, '_teach_result') and self._teach_result:
                status["teach_result"] = self._teach_result

            if self._perception:
                status["perception"] = self._perception.get_state()
                status["npu_stage"] = self._perception.npu_stage
                status["npu_stage_since"] = self._perception.npu_stage_since
            # PTZ-Settings + Tracker-State + restless_score fuer Panel
            ptz_status = {
                "home_pan": round(getattr(self, '_ptz_home_pan', 0.0), 1),
                "home_tilt": round(getattr(self, '_ptz_home_tilt', 0.0), 1),
                "tracking_speed": round(getattr(self, '_ptz_tracking_speed', 0.7), 2),
                "search_speed": round(getattr(self, '_ptz_search_speed', 0.15), 2),
            }
            # Aktuelle Kamera-Position (Pan/Tilt in Grad)
            try:
                from core.hardware.camera import get_camera_controller
                cam_ctrl = get_camera_controller()
                if cam_ctrl and hasattr(cam_ctrl, 'current_position'):
                    pos = cam_ctrl.current_position
                    if pos:
                        ptz_status["current_pan"] = round(pos.pan, 1)
                        ptz_status["current_tilt"] = round(pos.tilt, 1)
            except Exception:
                pass
            # Tracker-State (fuer Panel-Anzeige)
            tracker = self._cam._tracker
            if tracker:
                ptz_status["tracker_state"] = tracker.state.value
                ptz_status["tracking_moves"] = tracker.stats.get("tracking_moves", 0)
                ptz_status["search_moves"] = tracker.stats.get("search_moves", 0)
                ptz_status["camera_smart_tracking"] = getattr(tracker, '_camera_smart_tracking_on', False)
            # PTZ-Tracker restless_score
            if hasattr(self, '_ptz_tracker') and self._ptz_tracker:
                ptz_state = self._ptz_tracker.get_state()
                ptz_status.update(ptz_state)
            status["ptz"] = ptz_status
            # PTZ Arbiter Status
            try:
                from core.ptz_arbiter import get_ptz_arbiter
                arbiter = get_ptz_arbiter()
                arbiter.check_timeout()
                status.update(arbiter.get_status())
            except Exception:
                pass
            if self._core_integrator:
                raw_core = self._core_integrator.get_status_dict()
                # ArbitrationEngine: Override-Logik anwenden
                try:
                    from core.arbitration import get_arbitration
                    status["core"] = get_arbitration().apply(raw_core)
                except Exception:
                    status["core"] = raw_core
                # EINE Wahrheit: Top-Level personality_mode + tension (Gate0 Phase 5)
                core_data = status.get("core", {})
                status["personality_mode"] = core_data.get("zone", "guardian")
                status["tension"] = core_data.get("tension", 0.0)
            # Spotify Status (lazy — nur wenn bereits initialisiert)
            try:
                from core.spotify_controller import get_spotify
                sp = get_spotify()
                if sp._initialized:
                    status["spotify"] = sp.get_status()
            except Exception:
                pass
            # Music Visualizer Status
            if hasattr(self, '_music_vis') and self._music_vis:
                try:
                    md = self._music_vis.get_data()
                    status["music"] = {
                        "active": md.is_active,
                        "rms": round(md.rms_volume, 4),
                        "bass": round(md.bass_energy, 4),
                        "mid": round(md.mid_energy, 4),
                        "high": round(md.high_energy, 4),
                        "beat": md.beat_detected,
                    }
                except Exception:
                    pass
            # Action Bridge Status + Event Bus Stats (Dashboard)
            try:
                from core.action_bridge import get_action_bridge
                bridge = get_action_bridge()
                status["bridge"] = bridge.get_status()
                status["bridge_decisions"] = bridge.get_decisions(5)
            except Exception:
                pass
            try:
                from core.moloch_event_bus import get_event_bus
                bus = get_event_bus()
                status["bus_stats"] = bus.get_stats()
                status["silence_level"] = bus.silence_level
            except Exception:
                pass

            self._ipc.write_status(status)

            # MolochSprache Retention-Tick (1x/Stunde Cleanup)
            if self._sprache:
                self._sprache.tick_retention()
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
            self._cam.toggle_smart_tracking()
        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
            logger.info(f"[IPC] autonomous={self._cam._autonomous_mode} tentakel={self._cam._tentakel_enabled}")
        elif action == 'reload_face_db':
            self._inference.reload_face_db()
        elif action == 'enrollment_start':
            enroll_name = cmd.get('name', 'markus')
            enroll_n = cmd.get('n', 20)
            if hasattr(self._inference, 'start_enrollment'):
                self._inference.start_enrollment(enroll_name, enroll_n)
                logger.info(f"[IPC] Enrollment gestartet: name={enroll_name} n={enroll_n}")
            else:
                logger.warning("[IPC] Enrollment nur mit TAPPAS Pipeline verfuegbar")
        elif action == 'enrollment_status':
            if hasattr(self._inference, 'get_enrollment_status'):
                status = self._inference.get_enrollment_status()
                logger.info(f"[IPC] Enrollment-Status: {status}")
        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            # Popup sendet "scrfd_conf", InferenceEngine hat "scrfd_conf_val"
            attr_val = f"{attr}_val" if attr else None
            if attr_val and value is not None and hasattr(self._inference, attr_val):
                setattr(self._inference, attr_val, float(value))
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
            # WiFi-Mic force_source
            force_src = cmd.get('force_source')
            if force_src in ('auto', 'wifi', 'usb'):
                try:
                    from core.audio.wifi_mic import get_wifi_mic
                    get_wifi_mic().set_force_source(force_src)
                    logger.info(f"[IPC] Audio force_source={force_src}")
                except Exception as e:
                    logger.warning(f"[IPC] force_source Fehler: {e}")
            # WiFi Software-Gain
            wifi_sw_gain = cmd.get('wifi_software_gain')
            if wifi_sw_gain is not None:
                try:
                    from core.audio.wifi_mic import get_wifi_mic
                    get_wifi_mic().software_gain = float(wifi_sw_gain)
                    logger.info(f"[IPC] WiFi software_gain={wifi_sw_gain}")
                except Exception as e:
                    logger.warning(f"[IPC] wifi_software_gain Fehler: {e}")
            logger.info(f"[IPC] Audio: gain={self._saved_mic_gain:.2f}, "
                        f"gate={self._saved_noise_gate:.0f}dB, agc={self._saved_agc}")
        elif action == 'mic_test':
            # WiFi-Mic: 3s Audio aus Ringpuffer aufnehmen, als WAV speichern
            duration_s = float(cmd.get('duration', 3.0))
            rec_rate = int(cmd.get('rate', 16000))
            test_path = "/tmp/moloch_mic_test.wav"
            # 16kHz=Mono(1ch), 48kHz=Stereo(2ch)
            channels = 1 if rec_rate == 16000 else 2
            def _do_mic_test_wifi():
                try:
                    from core.audio.wifi_mic import get_wifi_mic
                    wm = get_wifi_mic()
                    # Ringpuffer leeren (alte Daten vor Test)
                    wm.get_audio_chunk(rate=rec_rate, duration_ms=2000)
                    # Warten bis neue Pakete fliessen (ESP32 braucht
                    # nach Rate-Switch bis zu 1s fuer ersten 48kHz-Chunk)
                    wait_deadline = time.time() + 2.0
                    while time.time() < wait_deadline:
                        probe = wm.get_audio_chunk(
                            rate=rec_rate, duration_ms=50)
                        if probe and len(probe) > 10:
                            logger.info(f"[IPC] Mic-Test: Erste Daten "
                                        f"bei {rec_rate}Hz empfangen")
                            break
                        time.sleep(0.1)
                    else:
                        logger.warning(f"[IPC] Mic-Test: Timeout, keine "
                                       f"Daten bei {rec_rate}Hz nach 2s")
                    # Audio sammeln
                    buf = bytearray()
                    end_time = time.time() + duration_s
                    while time.time() < end_time:
                        chunk = wm.get_audio_chunk(
                            rate=rec_rate, duration_ms=50)
                        if chunk:
                            buf.extend(chunk)
                        time.sleep(0.04)
                    if len(buf) < 100:
                        logger.warning(f"[IPC] Mic-Test: Keine Daten "
                                       f"bei {rec_rate}Hz")
                        return
                    # WAV schreiben (Rate + Channels dynamisch)
                    import struct as _st
                    bps = rec_rate * channels * 2  # bytes/sec
                    block_align = channels * 2
                    data_size = len(buf)
                    header = bytearray()
                    header.extend(b'RIFF')
                    header.extend(_st.pack('<I', 36 + data_size))
                    header.extend(b'WAVEfmt ')
                    header.extend(_st.pack('<IHHIIHH', 16, 1, channels,
                                           rec_rate, bps, block_align, 16))
                    header.extend(b'data')
                    header.extend(_st.pack('<I', data_size))
                    with open(test_path, 'wb') as f:
                        f.write(header)
                        f.write(buf)
                    logger.info(f"[IPC] Mic-Test: {len(buf)}B, "
                                f"{rec_rate}Hz {channels}ch, "
                                f"{data_size / bps:.1f}s")
                except Exception as e:
                    logger.error(f"[IPC] Mic-Test Fehler: {e}")
            threading.Thread(target=_do_mic_test_wifi, daemon=True,
                             name="MicTest-WiFi").start()
            logger.info(f"[IPC] Mic-Test WiFi gestartet "
                        f"({duration_s}s, {rec_rate}Hz)")
        elif action == 'whisper_vad':
            # VAD an/aus toggle vom Whisper-Popup
            enabled = bool(cmd.get('enabled', True))
            if self._voice_pipeline:
                self._voice_pipeline._vad_enabled = enabled
                logger.info(f"[IPC] Whisper VAD {'an' if enabled else 'aus'}")
        elif action == 'whisper_test':
            # Manueller Whisper-Test: NUR Aufnahme + Transkription (kein Claude/TTS)
            duration_s = float(cmd.get('duration_s', 8.0))
            if self._voice_pipeline:
                self._voice_pipeline.test_whisper(duration_s=duration_s)
                logger.info(f"[IPC] Whisper-Test gestartet ({duration_s}s)")
            else:
                logger.error("[IPC] Whisper-Test: Voice Pipeline nicht verfuegbar")
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
                self._inference.scrfd_conf_val = float(_th.get('scrfd_conf', self._inference.scrfd_conf_val))
                self._inference.scrfd_nms_val = float(_th.get('scrfd_nms', self._inference.scrfd_nms_val))
                self._inference.arcface_thresh_val = float(_th.get('arcface_thresh', self._inference.arcface_thresh_val))
                self._inference.yolo_conf_val = float(_th.get('yolo_conf', self._inference.yolo_conf_val))
                self._inference.pose_conf_val = float(_th.get('pose_conf', self._inference.pose_conf_val))
                self._inference.hand_conf_val = float(_th.get('hand_conf', self._inference.hand_conf_val))
            _ho = cmd.get('hand_occlusion')
            if _ho and self._perception:
                self._perception._HAND_TIMEOUT = float(_ho.get('timeout', 5.0))
                self._perception._MIN_FACE_STREAK = int(_ho.get('streak', 3))
                self._perception._FACE_RECENCY = float(_ho.get('recency', 2.0))
            self._save_settings()
        elif action == 'set_silence_level':
            level = int(cmd.get('level', 0))
            from core.moloch_event_bus import get_event_bus
            get_event_bus().set_silence_level(level)
            logger.info(f"[IPC] Silence-Level: {level}")

        elif action == 'toggle_teachen':
            if self._teachen:
                enabled = self._teachen.toggle()
                logger.info(f"[IPC] Teachen: {'AN' if enabled else 'AUS'}")

        elif action == 'set_mpo_param':
            # MPO Slider live anwenden (Popup sendet param + value)
            try:
                import math as _math
                from core.core_integrator import get_core_integrator
                ci = get_core_integrator()
                param = cmd.get('param')
                value = float(cmd.get('value', 0))
                if ci and param:
                    if param == 'tension_tau':
                        ci.TENSION_TAU = value
                        ci._TENSION_DECAY_FACTOR = _math.exp(-1.0 / value)
                    elif param == 'dominance_drift':
                        ci.DOMINANCE_DRIFT_RATE = value / 60.0
                    elif param == 'zone_hysteresis':
                        ci.ZONE_HYSTERESIS = value
                    elif param == 'berserker_threshold':
                        ci.BERSERKER_TENSION_THRESHOLD = value
                    elif param == 'thermal_damping_start':
                        ci.THERMAL_DAMPING_START = value
                    logger.info(f"[IPC] MPO: {param} = {value}")
            except Exception as e:
                logger.warning(f"[IPC] MPO-Fehler: {e}")

        elif action == 'set_gesture_params':
            # Gesten-Checkboxen live anwenden
            self._saved_gestures = {
                k: v for k, v in cmd.items() if k != 'action'
            }
            logger.info(f"[IPC] Gestures: {self._saved_gestures}")

        elif action == 'set_gesture_param':
            # Gesten-Sensitivity Slider
            param = cmd.get('param')
            value = cmd.get('value')
            if param and value is not None:
                if not hasattr(self, '_saved_gestures'):
                    self._saved_gestures = {}
                self._saved_gestures[param] = float(value)
                logger.info(f"[IPC] Gesture: {param} = {value}")

        elif action == 'set_tracker_param':
            # Tracker-Parameter live auf AutonomousTracker anwenden (Popup-Slider)
            param = cmd.get('param')
            value = cmd.get('value')
            if param is not None and value is not None:
                value = float(value)
                tracker = self._cam._tracker
                if tracker and hasattr(tracker, 'config'):
                    cfg = tracker.config
                    if hasattr(cfg, param):
                        setattr(cfg, param, value)
                        # Basis-Werte aktualisieren (fuer dynamische Anpassung)
                        if param == 'pan_gain':
                            tracker._base_pan_gain = value
                        elif param == 'tilt_gain':
                            tracker._base_tilt_gain = value
                        elif param == 'max_step_pan':
                            tracker._base_max_step_pan = value
                        elif param == 'tracking_speed':
                            tracker._base_tracking_speed = value
                            self._ptz_tracking_speed = value
                        elif param == 'search_speed':
                            self._ptz_search_speed = value
                        logger.info(f"[TRACKER] {param} = {value}")
                    else:
                        logger.warning(f"[TRACKER] Unbekannter Param: {param}")

        elif action == 'set_ptz_settings':
            # PTZ-Settings vom Panel: Home, Speed, Limits
            self._ptz_home_pan = float(cmd.get('home_pan', getattr(self, '_ptz_home_pan', 0.0)))
            self._ptz_home_tilt = float(cmd.get('home_tilt', getattr(self, '_ptz_home_tilt', 0.0)))
            self._ptz_tracking_speed = float(cmd.get('tracking_speed', getattr(self, '_ptz_tracking_speed', 0.7)))
            self._ptz_search_speed = float(cmd.get('search_speed', getattr(self, '_ptz_search_speed', 0.15)))
            self._ptz_pan_limit_min = float(cmd.get('pan_limit_min', getattr(self, '_ptz_pan_limit_min', -168.4)))
            self._ptz_pan_limit_max = float(cmd.get('pan_limit_max', getattr(self, '_ptz_pan_limit_max', 170.0)))
            self._ptz_tilt_limit_min = float(cmd.get('tilt_limit_min', getattr(self, '_ptz_tilt_limit_min', -78.0)))
            self._ptz_tilt_limit_max = float(cmd.get('tilt_limit_max', getattr(self, '_ptz_tilt_limit_max', 78.8)))
            # Home-Position in CameraManager + Camera-Controller setzen
            self._cam._home_position = {
                "pan": self._ptz_home_pan,
                "tilt": self._ptz_home_tilt
            }
            try:
                from core.hardware.camera import get_camera_controller
                cam_ctrl = get_camera_controller()
                cam_ctrl.set_home_position(self._ptz_home_pan, self._ptz_home_tilt)
            except Exception:
                pass
            # Tracker-Config live updaten
            self._apply_ptz_to_tracker()
            logger.info(f"[PTZ] Settings: Home=({self._ptz_home_pan:.1f},{self._ptz_home_tilt:.1f}) "
                        f"Speed={self._ptz_tracking_speed:.2f} Search={self._ptz_search_speed:.2f}")
        elif action == 'detach_tracker':
            # Manueller Detach vom Track → SEARCHING + Bild in galerie/detach/
            if hasattr(self, '_action_bridge') and self._action_bridge:
                self._action_bridge.force_detach()
                logger.info("[IPC] Detach: Bridge zurueck zu SEARCHING")
            try:
                self._cam.take_detach_snapshot()
            except Exception as e:
                logger.warning(f"[IPC] Detach-Snapshot fehlgeschlagen: {e}")
        elif action == 'set_ptz_home':
            # Pan/Tilt aus IPC-Message lesen (GUI sendet Werte mit)
            home_pan = float(cmd.get('pan', self._ptz_home_pan))
            home_tilt = float(cmd.get('tilt', self._ptz_home_tilt))
            self._ptz_home_pan = home_pan
            self._ptz_home_tilt = home_tilt
            self._cam._home_position = {"pan": home_pan, "tilt": home_tilt}
            # SonoffCameraController updaten
            try:
                from core.hardware.camera import get_camera_controller
                cam_ctrl = get_camera_controller()
                cam_ctrl.set_home_position(home_pan, home_tilt)
            except Exception:
                pass
            # Tracker-Config live updaten
            if hasattr(self, '_autonomous_tracker') and self._autonomous_tracker:
                self._autonomous_tracker.config.home_pan = home_pan
                self._autonomous_tracker.config.home_tilt = home_tilt
            # Persistieren in settings.json
            self._save_settings()
            logger.info(f"[PTZ] Home gesetzt: Pan={home_pan:.1f}, Tilt={home_tilt:.1f}")
        elif action == 'ptz_move':
            self._cam.ptz_move(cmd.get('direction', ''), cmd.get('speed', 0.3))
        elif action == 'ptz_goto':
            self._cam.ptz_goto(cmd.get('position', ''))
        elif action == 'ptz_calibrate':
            self._cam.ptz_calibrate()
        elif action == 'cloud_led':
            self._cam.cloud_set_night_mode(cmd.get('level', 0))
        elif action == 'cloud_alarm':
            self._cam.cloud_toggle_alarm()
        elif action == 'snapshot':
            self._cam.take_snapshot()
        elif action == 'teach_mode_toggle':
            # Teach-Modus AN/AUS umschalten (persistent)
            self._teach_mode_enabled = not self._teach_mode_enabled
            logger.info(f"[TEACH] Modus: {'AN' if self._teach_mode_enabled else 'AUS'}")
            # Scheduler erzwingen (FIX 1)
            if hasattr(self._inference, 'force_all_active'):
                self._inference.force_all_active(self._teach_mode_enabled)
            self._save_settings()
            if not self._teach_mode_enabled:
                # Modus ausgeschaltet — Result zuruecksetzen
                self._teach_result = {}
        elif action == 'teach_trigger':
            # Manueller Teach-Trigger (FIX 4: sofort, kein Cooldown)
            if not self._teach_busy:
                logger.info("[TEACH] Manueller Trigger via Panel")
                # Scheduler temporaer erzwingen
                if hasattr(self._inference, 'force_all_active'):
                    self._inference.force_all_active(True)
                self._teach_result = {"status": "starting", "attempt": 0, "detail": ""}
                threading.Thread(
                    target=self._teach_smart_snapshot,
                    daemon=True, name="TeachManual"
                ).start()
        elif action == 'cloud_status_led':
            self._cam.cloud_toggle_status_led()
        elif action == 'cloud_sync':
            self._cam.cloud_sync()

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

        # ---- Einpraegen (TAPPAS Enrollment mit Teachen/Snapshot-Bildern) ----
        elif action == 'einpraegen':
            # TAPPAS: start_enrollment() mit Batch-GStreamer-Pipeline (kompatible Embeddings)
            if hasattr(self._inference, 'start_enrollment'):
                enr = self._inference.get_enrollment_status()
                if not enr["active"]:
                    enroll_name = cmd.get('name', 'markus')
                    self._inference.start_enrollment(enroll_name)
                    logger.info(f"[IPC] TAPPAS-Enrollment gestartet fuer '{enroll_name}' "
                                 f"(Teachen + Snapshots + Live-Stream)")
                else:
                    logger.warning("[IPC] Enrollment laeuft bereits")
            elif self._einpraegen:
                # Fallback: Legacy Einpraegen (nur wenn kein TAPPAS)
                if not self._einpraegen.is_running:
                    self._einpraegen.start(orchestrator=self._orchestrator)
                    logger.info("[IPC] Legacy-Einpraegen gestartet (kein TAPPAS)")
                else:
                    logger.warning("[IPC] Einpraegen laeuft bereits")
            else:
                logger.warning("[IPC] Einpraegen nicht verfuegbar")

        # ---- Spotify Commands ----
        elif action == 'spotify_play':
            from core.spotify_controller import get_spotify
            uri = cmd.get('uri')
            get_spotify().play(uri=uri)

        elif action == 'spotify_pause':
            from core.spotify_controller import get_spotify
            get_spotify().pause()

        elif action == 'spotify_toggle':
            from core.spotify_controller import get_spotify
            get_spotify().toggle()

        elif action == 'spotify_skip':
            from core.spotify_controller import get_spotify
            get_spotify().next_track()

        elif action == 'spotify_previous':
            from core.spotify_controller import get_spotify
            get_spotify().previous_track()

        elif action == 'spotify_volume':
            from core.spotify_controller import get_spotify
            vol = cmd.get('volume', 50)
            get_spotify().set_volume(int(vol))

        elif action == 'spotify_search':
            from core.spotify_controller import get_spotify
            query = cmd.get('query', '')
            if query:
                get_spotify().search_and_play(query)

        elif action == 'spotify_mood':
            from core.spotify_controller import get_spotify
            zone = cmd.get('zone', 'shadow')
            get_spotify().play_by_mood(zone)

        elif action == 'spotify_artist':
            from core.spotify_controller import get_spotify
            artist = cmd.get('artist', '')
            if artist:
                get_spotify().play_artist(artist)

        elif action == 'spotify_auto_dj':
            from core.spotify_controller import get_spotify
            state = cmd.get('state')
            sp = get_spotify()
            if state == 'on':
                sp.auto_dj_start()
            elif state == 'off':
                sp.auto_dj_stop()
            else:
                sp.auto_dj_toggle()

        elif action == 'spotify_shuffle':
            from core.spotify_controller import get_spotify
            state = cmd.get('state', True)
            get_spotify().shuffle(bool(state))

        elif action == 'spotify_similar':
            from core.spotify_controller import get_spotify
            get_spotify().play_similar()

        elif action == 'spotify_top_tracks':
            from core.spotify_controller import get_spotify
            get_spotify().play_top_tracks()

        elif action == 'spotify_new_music':
            from core.spotify_controller import get_spotify
            get_spotify().play_new_music()

        elif action == 'spotify_from_year':
            from core.spotify_controller import get_spotify
            year = cmd.get('year', 2020)
            get_spotify().play_from_year(int(year))

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

        # Thresholds (auf InferenceEngine)
        try:
            _inf = self._inference
            th = data.get("thresholds", {})
            if "scrfd_conf" in th:
                _inf.scrfd_conf_val = float(th["scrfd_conf"])
            if "scrfd_nms" in th:
                _inf.scrfd_nms_val = float(th["scrfd_nms"])
            if "arcface_thresh" in th:
                _inf.arcface_thresh_val = float(th["arcface_thresh"])
            if "yolo_conf" in th:
                _inf.yolo_conf_val = float(th["yolo_conf"])
            if "pose_conf" in th:
                _inf.pose_conf_val = float(th["pose_conf"])
            if "hand_conf" in th:
                _inf.hand_conf_val = float(th["hand_conf"])
            logger.info(f"[SETTINGS] Thresholds: scrfd={_inf.scrfd_conf_val}/{_inf.scrfd_nms_val} "
                        f"arc={_inf.arcface_thresh_val} yolo={_inf.yolo_conf_val} "
                        f"pose={_inf.pose_conf_val} hand={_inf.hand_conf_val}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Thresholds-Fehler: {e}")

        # Hand-Occlusion (auf InferenceEngine + Perception Engine)
        try:
            ho = data.get("hand_occlusion", {})
            if ho:
                self._inference._hand_occlusion_enabled = bool(ho.get("enabled", False))
                _ho_timeout = float(ho.get("timeout", 5.0))
                _ho_streak = int(ho.get("streak", 3))
                _ho_recency = float(ho.get("recency", 2.0))
                if self._perception:
                    self._perception._hand_occlusion_enabled = self._inference._hand_occlusion_enabled
                    self._perception._HAND_TIMEOUT = _ho_timeout
                    self._perception._MIN_FACE_STREAK = _ho_streak
                    self._perception._FACE_RECENCY = _ho_recency
                logger.info(f"[SETTINGS] Hand-Occlusion: enabled={self._inference._hand_occlusion_enabled} "
                            f"timeout={_ho_timeout} streak={_ho_streak} recency={_ho_recency}")
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

        # PTZ-Settings (Home, Limits, Speed -> direkt auf Tracker anwenden)
        try:
            ptz = data.get("ptz", {})
            if ptz:
                self._ptz_home_pan = float(ptz.get("home_pan", 0.0))
                self._ptz_home_tilt = float(ptz.get("home_tilt", 0.0))
                self._ptz_tracking_speed = float(ptz.get("tracking_speed", 0.7))
                self._ptz_search_speed = float(ptz.get("search_speed", 0.15))
                self._ptz_pan_limit_min = float(ptz.get("pan_limit_min", -168.4))
                self._ptz_pan_limit_max = float(ptz.get("pan_limit_max", 170.0))
                self._ptz_tilt_limit_min = float(ptz.get("tilt_limit_min", -78.0))
                self._ptz_tilt_limit_max = float(ptz.get("tilt_limit_max", 78.8))
                # Home-Position auch in CameraManager setzen
                self._cam._home_position = {
                    "pan": self._ptz_home_pan,
                    "tilt": self._ptz_home_tilt
                }
                # Home-Position auch in SonoffCameraController setzen
                try:
                    from core.hardware.camera import get_camera_controller
                    cam_ctrl = get_camera_controller()
                    cam_ctrl.set_home_position(self._ptz_home_pan, self._ptz_home_tilt)
                except Exception:
                    pass
                logger.info(f"[PTZ] Home: Pan={self._ptz_home_pan:.1f}, Tilt={self._ptz_home_tilt:.1f}, "
                            f"Speed={self._ptz_tracking_speed:.2f}")
        except Exception as e:
            logger.warning(f"[SETTINGS] PTZ-Fehler: {e}")

        # Teach-Modus (persistent)
        try:
            teach = data.get("teach", {})
            if "mode_enabled" in teach:
                self._teach_mode_enabled = bool(teach["mode_enabled"])
                logger.info(f"[SETTINGS] Teach-Modus: {'AN' if self._teach_mode_enabled else 'AUS'}")
                # Scheduler erzwingen wenn Teach-Modus persistent AN war
                if self._teach_mode_enabled and hasattr(self._inference, 'force_all_active'):
                    self._inference.force_all_active(True)
            if "cooldown" in teach:
                self._teach_cooldown = float(teach["cooldown"])
        except Exception as e:
            logger.warning(f"[SETTINGS] Teach-Fehler: {e}")

        # Orchestration Mode — NPU Idle-Modus ersetzt always_on
        # PerceptionEngine steuert jetzt die Modelle stufenweise (idle/person/face)
        try:
            om = data.get("orchestration_mode", "adaptive")
            # "always_on" wird nicht mehr unterstuetzt — Idle-Modus ist aktiv
            if om == "always_on":
                om = "adaptive"
                logger.info("[SETTINGS] orchestration_mode: always_on → adaptive (NPU Idle-Modus)")
            if self._orchestrator:
                self._orchestrator.orchestration_mode = om
                logger.info(f"[SETTINGS] Orchestration Mode: {om}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Orchestration-Mode-Fehler: {e}")

        # Aktive Modelle — Gate 0: KEINE force_models mehr!
        # Perception Engine steuert Modelle automatisch (idle/person/face Stufen).
        # Alle Modelle gleichzeitig = NPU saturiert = 6 FPS statt 25+.
        try:
            am = data.get("active_models")
            if am and isinstance(am, list):
                self._saved_active_models = list(am)
                logger.info(f"[SETTINGS] Active Models (gespeichert, nicht erzwungen): {am}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Active-Models-Fehler: {e}")

        # MPO (auf CoreIntegrator Klassen-Konstanten)
        try:
            mpo = data.get("mpo", {})
            if mpo:
                import math as _math
                from core.core_integrator import get_core_integrator
                ci = get_core_integrator()
                if ci and "tension_tau" in mpo:
                    ci.TENSION_TAU = float(mpo["tension_tau"])
                    ci._TENSION_DECAY_FACTOR = _math.exp(-1.0 / ci.TENSION_TAU)
                if ci and "dominance_drift" in mpo:
                    ci.DOMINANCE_DRIFT_RATE = float(mpo["dominance_drift"]) / 60.0
                if ci and "zone_hysteresis" in mpo:
                    ci.ZONE_HYSTERESIS = float(mpo["zone_hysteresis"])
                if ci and "berserker_threshold" in mpo:
                    ci.BERSERKER_TENSION_THRESHOLD = float(mpo["berserker_threshold"])
                if ci and "thermal_damping_start" in mpo:
                    ci.THERMAL_DAMPING_START = float(mpo["thermal_damping_start"])
                logger.info(f"[SETTINGS] MPO: tau={mpo.get('tension_tau')} "
                            f"drift={mpo.get('dominance_drift')} hyst={mpo.get('zone_hysteresis')} "
                            f"thermal={mpo.get('thermal_damping_start')}")
        except Exception as e:
            logger.warning(f"[SETTINGS] MPO-Fehler: {e}")

        # Gesten (Flags speichern, Anwendung spaeter wenn GestureDetector aktiv)
        try:
            gest = data.get("gestures", {})
            if gest:
                self._saved_gestures = dict(gest)
                logger.info(f"[SETTINGS] Gestures: {list(gest.keys())}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Gestures-Fehler: {e}")

    def _save_settings(self):
        """Speichere aktuelle Settings nach config/settings.json (atomic merge-write).

        Liest bestehende JSON zuerst, merged unsere Keys drueber.
        So bleiben Popup-eigene Keys (mpo, gestures, pose_conf, hand_conf) erhalten.
        """
        # Bestehende Datei lesen (Merge statt Overwrite)
        data = {"version": 1}
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["version"] = 1
        except Exception:
            data = {"version": 1}

        _inf = self._inference

        # Thresholds (von InferenceEngine) — bestehende Keys (pose_conf etc.) erhalten
        th = data.get("thresholds", {})
        th["scrfd_conf"] = round(_inf.scrfd_conf_val, 3)
        th["scrfd_nms"] = round(_inf.scrfd_nms_val, 3)
        th["arcface_thresh"] = round(_inf.arcface_thresh_val, 3)
        th["yolo_conf"] = round(_inf.yolo_conf_val, 3)
        th["pose_conf"] = round(_inf.pose_conf_val, 3)
        th["hand_conf"] = round(_inf.hand_conf_val, 3)
        data["thresholds"] = th

        # Hand-Occlusion (von InferenceEngine + Perception)
        if self._perception:
            data["hand_occlusion"] = {
                "enabled": _inf._hand_occlusion_enabled,
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

        # PTZ-Settings (Home, Limits, Speed)
        data["ptz"] = {
            "home_pan": round(getattr(self, '_ptz_home_pan', 0.0), 1),
            "home_tilt": round(getattr(self, '_ptz_home_tilt', 0.0), 1),
            "tracking_speed": round(getattr(self, '_ptz_tracking_speed', 0.7), 2),
            "search_speed": round(getattr(self, '_ptz_search_speed', 0.15), 2),
            "pan_limit_min": round(getattr(self, '_ptz_pan_limit_min', -168.4), 1),
            "pan_limit_max": round(getattr(self, '_ptz_pan_limit_max', 170.0), 1),
            "tilt_limit_min": round(getattr(self, '_ptz_tilt_limit_min', -78.0), 1),
            "tilt_limit_max": round(getattr(self, '_ptz_tilt_limit_max', 78.8), 1),
        }

        # Teach-Modus (persistent)
        data["teach"] = {
            "mode_enabled": self._teach_mode_enabled,
            "cooldown": self._teach_cooldown,
        }

        # Orchestration Mode
        if self._orchestrator:
            data["orchestration_mode"] = self._orchestrator.orchestration_mode

        # Aktive Modelle (von ModelOrchestrator)
        with self._ctx_lock:
            data["active_models"] = list(self._active_ctx.keys())

        # MPO (von CoreIntegrator — aktuelle Werte zurueckschreiben)
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            if ci:
                mpo = data.get("mpo", {})
                mpo["tension_tau"] = round(ci.TENSION_TAU, 1)
                mpo["dominance_drift"] = round(ci.DOMINANCE_DRIFT_RATE * 60.0, 3)
                mpo["zone_hysteresis"] = round(ci.ZONE_HYSTERESIS, 2)
                mpo["berserker_threshold"] = round(ci.BERSERKER_TENSION_THRESHOLD, 2)
                mpo["thermal_damping_start"] = round(ci.THERMAL_DAMPING_START, 1)
                data["mpo"] = mpo
        except Exception:
            pass  # CoreIntegrator noch nicht verfuegbar

        # Gestures (gespeicherte Flags zurueckschreiben)
        if hasattr(self, '_saved_gestures') and self._saved_gestures:
            data["gestures"] = self._saved_gestures

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
