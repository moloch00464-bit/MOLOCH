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

from core.hardware.hailo_manager import get_hailo_manager
from core.led_controller import LEDController
from core.ipc_router import IPCRouter
from core.model_orchestrator import ModelOrchestrator, MODEL_PATHS
from core.camera_manager import CameraManager
from core.inference_engine import InferenceEngine
from core.longterm_memory import get_memory
from core.perception.perception_buffer import get_perception_buffer
from core.perception.model_health import get_model_health

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
            # Gespeicherte aktive Modelle als force_models setzen
            if hasattr(self, '_saved_active_models') and self._saved_active_models:
                self._perception.force_models(self._saved_active_models)
                logger.info(f"[SETTINGS] force_models({self._saved_active_models}) aus settings.json")
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
            daily_learner=self._daily_learner,
            model_health=self._model_health,
            notify_callback=self._notify,
        )
        # Aliased Referenzen auf Orchestrator-Objekte (fuer _write_status_json)
        self._active_ctx = self._orchestrator._active_ctx
        self._ctx_lock = self._orchestrator._ctx_lock

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
        # Aliased mutable Referenzen (fuer _write_status_json)
        self._cloud_state = self._cam._cloud_state

        # IPC Router (extrahiert aus moloch_service.py, Phase 4)
        self._ipc = IPCRouter()

        # InferenceEngine (NPU Pipeline + Inference Loop, Phase 4 Schritt 5)
        self._inference = InferenceEngine(
            orchestrator=self._orchestrator,
            camera=self._cam,
            led=self._led,
            ipc=self._ipc,
            perception=self._perception,
            core_integrator=self._core_integrator,
            daily_learner=self._daily_learner,
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
        for name in self._orchestrator._models:
            logger.info(f"Modell geladen: {name} ({len(self._orchestrator._output_names[name])} outputs)")

        # 1b. Whisper permanent auf NPU laden (shared VDevice, 8GB reichen)
        try:
            from core.speech.hailo_whisper import get_whisper
            whisper = get_whisper()
            whisper.set_vdevice(self._orchestrator.vdevice)
        except Exception as e:
            logger.error(f"[INIT] Whisper NPU init fehlgeschlagen: {e}")

        # 1c. Alle Modelle sofort auf NPU konfigurieren (always_on: 320MB von 8GB)
        if self._orchestrator.orchestration_mode == "always_on":
            for name in list(self._orchestrator._models.keys()):
                self._orchestrator.configure(name)
            self._orchestrator.sync_flags()
            self._inference.sync_flags_from_npu()
            logger.info(f"[INIT] Always-On: {len(self._orchestrator.active_models)} Modelle konfiguriert: {self._orchestrator.active_models}")

        # 2. Face DB (via InferenceEngine)
        self._inference.reload_face_db()

        # 3. RTSP (via CameraManager)
        self._cam.start_rtsp()

        # 4. Cloud (via CameraManager, im Hintergrund)
        threading.Thread(target=self._cam.connect_cloud, daemon=True).start()

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

        # Inference Loop (via InferenceEngine)
        self._inference.start()

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
            fps_snapshot = self._inference.get_fps()
            with self._ctx_lock:
                active_models = list(self._active_ctx.keys())

            _inf = self._inference
            status = {
                "scrfd_active": _inf.scrfd_active,
                "arcface_active": _inf.arcface_active,
                "yolo_active": _inf.yolo_active,
                "hand_active": _inf.hand_active,
                "pose_active": _inf.pose_active,
                "npu_paused": self._orchestrator._npu_paused,
                "active_models": active_models,
                "autonomous_mode": self._cam._autonomous_mode,
                "manual_mode": self._cam._manual_mode,
                "moloch_has_control": self._cam._moloch_has_control,
                "tentakel_enabled": self._cam._tentakel_enabled,
                "daily_learner_enabled": self._daily_learner.enabled if self._daily_learner else False,
                "learner_flash": _inf._learner_flash,
                "frame_age": round(time.time() - self._cam._last_frame_write, 1) if self._cam._last_frame_write else -1,
                "frozen_restarts": self._cam._frozen_restart_count,
                "fps": {k: round(v, 1) for k, v in fps_snapshot.items()},
                "thresholds": {
                    "scrfd_conf": _inf.scrfd_conf_val,
                    "scrfd_nms": _inf.scrfd_nms_val,
                    "arcface_thresh": _inf.arcface_thresh_val,
                    "yolo_conf": _inf.yolo_conf_val,
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
            self._cam.toggle_smart_tracking()
        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
            logger.info(f"[IPC] autonomous={self._cam._autonomous_mode} tentakel={self._cam._tentakel_enabled}")
        elif action == 'reload_face_db':
            self._inference.reload_face_db()
        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            if attr and value is not None and hasattr(self._inference, attr):
                setattr(self._inference, attr, float(value))
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
                self._inference.scrfd_conf_val = float(_th.get('scrfd_conf', self._inference.scrfd_conf_val))
                self._inference.scrfd_nms_val = float(_th.get('scrfd_nms', self._inference.scrfd_nms_val))
                self._inference.arcface_thresh_val = float(_th.get('arcface_thresh', self._inference.arcface_thresh_val))
                self._inference.yolo_conf_val = float(_th.get('yolo_conf', self._inference.yolo_conf_val))
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
            self._inference._learner_flash = bool(cmd.get('on', not self._inference._learner_flash))
            logger.info(f"[IPC] Learner Flash: {'AN' if self._inference._learner_flash else 'AUS'}")

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
            logger.info(f"[SETTINGS] Thresholds: scrfd={_inf.scrfd_conf_val}/{_inf.scrfd_nms_val} "
                        f"arc={_inf.arcface_thresh_val} yolo={_inf.yolo_conf_val}")
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

        # Learner Flash (auf InferenceEngine)
        try:
            learner = data.get("learner", {})
            if "flash_enabled" in learner:
                self._inference._learner_flash = bool(learner["flash_enabled"])
                logger.info(f"[SETTINGS] Learner Flash: {self._inference._learner_flash}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Learner-Fehler: {e}")

        # Orchestration Mode (always_on = alle Modelle immer aktiv, Default)
        try:
            om = data.get("orchestration_mode", "always_on")
            if self._orchestrator:
                self._orchestrator.orchestration_mode = om
                logger.info(f"[SETTINGS] Orchestration Mode: {om}")
        except Exception as e:
            logger.warning(f"[SETTINGS] Orchestration-Mode-Fehler: {e}")

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
        _inf = self._inference

        # Thresholds (von InferenceEngine)
        data["thresholds"] = {
            "scrfd_conf": round(_inf.scrfd_conf_val, 3),
            "scrfd_nms": round(_inf.scrfd_nms_val, 3),
            "arcface_thresh": round(_inf.arcface_thresh_val, 3),
            "yolo_conf": round(_inf.yolo_conf_val, 3),
        }

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

        # Learner (von InferenceEngine)
        data["learner"] = {
            "flash_enabled": _inf._learner_flash,
        }

        # Orchestration Mode
        if self._orchestrator:
            data["orchestration_mode"] = self._orchestrator.orchestration_mode

        # Aktive Modelle (von ModelOrchestrator)
        with self._ctx_lock:
            data["active_models"] = list(self._active_ctx.keys())

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
