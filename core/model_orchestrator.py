#!/usr/bin/env python3
"""
ModelOrchestrator - NPU Pipeline + Modell-Lifecycle + Attention-Level.

Extrahiert aus moloch_service.py (Phase 4, Regel 10).

Verantwortlichkeiten:
  - VDevice erstellen, HEF-Modelle laden
  - Modelle konfigurieren/dekonfigurieren auf NPU
  - Inference ausfuehren (Hot Path ~21ms pro Modell)
  - Voice/NPU Coordination (pause/resume)
  - Attention-Level basierte Modell-Orchestrierung
  - NPU Watchdog (Anti-Oszillation)
"""

import os
import gc
import time
import json
import threading
import traceback
import logging

import numpy as np
from hailo_platform import HEF, VDevice, FormatType

logger = logging.getLogger("ModelOrchestrator")

# Modell-Pfade auf SSD2
MODEL_DIR = "/mnt/moloch-data/hailo/models"
MODEL_PATHS = {
    "scrfd": f"{MODEL_DIR}/scrfd_10g.hef",
    "arcface": f"{MODEL_DIR}/arcface_mobilefacenet.hef",
    "yolov8m": f"{MODEL_DIR}/yolov8m_h10.hef",
    "hand_landmark": f"{MODEL_DIR}/hand_landmark_lite.hef",
    "pose": f"{MODEL_DIR}/yolov8s_pose_h10.hef",
    "face_attr": f"{MODEL_DIR}/face_attr_resnet_v1_18.hef",
}

# Cross-process NPU IPC files
NPU_VOICE_REQUEST = "/tmp/moloch_npu_voice_request"
NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"


class ModelOrchestrator:
    """NPU Pipeline + Modell-Lifecycle + Attention-Level Orchestrierung."""

    def __init__(self, hailo_manager=None, perception_engine=None,
                 core_integrator=None, teachen=None,
                 model_health=None, notify_callback=None):
        """
        Args:
            hailo_manager: HailoManager fuer Device-Zugriff
            perception_engine: PerceptionEngine fuer Slot-Rotation
            core_integrator: CoreIntegrator fuer Attention-Level
            teachen: Teachen fuer Teach-Mode Detection
            model_health: ModelHealth fuer Inference-Stats
            notify_callback: Callback(event, data) fuer UI-Notifications
        """
        self._hailo_manager = hailo_manager
        self._perception = perception_engine
        self._core_integrator = core_integrator
        self._teachen = teachen
        self._model_health = model_health
        self._notify = notify_callback or (lambda e, d: None)

        # VDevice + geladene Modelle
        self._vdevice = None
        self._models = {}
        self._output_names = {}

        # Persistente Model-Kontexte (konfigurierte Modelle auf NPU)
        self._active_ctx = {}
        self._ctx_lock = threading.Lock()

        # Configuring Event (verhindert run() waehrend configure())
        self._configuring = threading.Event()
        self._configuring.set()

        # Cross-process NPU pause
        self._paused_models = []
        self._npu_paused = False

        # Model enable flags
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.hand_active = False
        self.pose_active = False
        self.face_attr_active = False

        # Threshold values
        self.scrfd_conf_val = 0.40
        self.scrfd_nms_val = 0.40
        self.arcface_thresh_val = 0.60
        self.yolo_conf_val = 0.50

        # Attention-Level basierte Modell-Aktivierung
        self._attention_level = "high"
        self._attention_level_lock = threading.Lock()
        self._target_frame_delay = 0.033  # 30 FPS Default

        # Orchestration Mode: "adaptive" = NPU Idle-Modus (PerceptionEngine steuert)
        self._orchestration_mode = "adaptive"

        # NPU Watchdog: Anti-Oszillation
        self._swap_log = []
        self._models_preloaded = False

        # Recovery Counter
        self._recovery_count = 0

    # =================================================================
    # Properties
    # =================================================================

    @property
    def vdevice(self):
        """VDevice Referenz (fuer Whisper shared)."""
        return self._vdevice

    @property
    def active_models(self) -> list:
        """Liste der aktuell konfigurierten Modelle."""
        with self._ctx_lock:
            return list(self._active_ctx.keys())

    @property
    def active_ctx(self) -> dict:
        """Direkter Zugriff auf active context (fuer Inference Loop)."""
        return self._active_ctx

    @property
    def is_paused(self) -> bool:
        return self._npu_paused

    @property
    def configuring(self) -> threading.Event:
        return self._configuring

    @property
    def models(self) -> dict:
        """Geladene Modelle."""
        return self._models

    @property
    def target_frame_delay(self) -> float:
        return self._target_frame_delay

    @property
    def orchestration_mode(self) -> str:
        return self._orchestration_mode

    @orchestration_mode.setter
    def orchestration_mode(self, mode: str):
        """Orchestration Mode setzen: 'always_on' oder 'adaptive'."""
        if mode not in ("always_on", "adaptive"):
            logger.warning(f"[ORCHESTRATION] Unbekannter Modus: {mode}, bleibe bei {self._orchestration_mode}")
            return
        old = self._orchestration_mode
        self._orchestration_mode = mode
        logger.info(f"[ORCHESTRATION] Modus: {old} -> {mode}")
        if mode == "always_on":
            # Sofort alle Modelle aktivieren, 30 FPS
            self._attention_level = "high"
            self._target_frame_delay = 0.033

    # =================================================================
    # Model Loading
    # =================================================================

    def load_models(self):
        """VDevice erstellen und alle HEF-Modelle laden."""
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

    # =================================================================
    # Model Configure/Unconfigure/Run (Hot Path)
    # =================================================================

    def configure(self, name, _retry=0):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms).

        Gate 0: Retry mit Backoff bei DRIVER_OPERATION_FAILED (max 3 Versuche).
        """
        MAX_RETRIES = 3
        with self._ctx_lock:
            already_active = name in self._active_ctx
        if already_active:
            logger.info(f"[CONFIGURE] {name}: bereits konfiguriert, skip")
            return
        if name not in self._models:
            logger.warning(f"[CONFIGURE] {name}: nicht in self._models")
            return

        infer_model = self._models[name]
        out_names = self._output_names[name]
        active_before = list(self._active_ctx.keys())
        logger.info(f"[CONFIGURE] {name}: aktive Modelle VORHER: {active_before}")

        # Inference pausieren
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
            err_str = str(e)
            # Hailo DRIVER_OPERATION_FAILED(36) oder CONNECTION_REFUSED(89): Retry mit Backoff
            if _retry < MAX_RETRIES and ("36" in err_str or "89" in err_str
                                          or "DRIVER_OPERATION" in err_str
                                          or "CONNECTION_REFUSED" in err_str):
                backoff = 1.0 * (2 ** _retry)
                logger.warning(f"[CONFIGURE] {name}: Hailo-Error, retry {_retry+1}/{MAX_RETRIES} "
                               f"in {backoff:.0f}s — {type(e).__name__}: {e}")
                self._configuring.set()
                time.sleep(backoff)
                self.configure(name, _retry=_retry + 1)
                return

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
        finally:
            self._configuring.set()

    def unconfigure(self, name):
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

    def run(self, name, input_data):
        """Fuehre Modell aus mit persistenter Konfiguration (~21ms).

        Gate 0: Bei Hailo-Error leeres Dict statt Crash (Inference Loop laeuft weiter).

        Returns: Dict mit Output-Name -> numpy array
        """
        with self._ctx_lock:
            ctx = self._active_ctx.get(name)
            if not ctx:
                return {}
            try:
                bindings = ctx["bindings"]
                bindings.input().set_buffer(np.ascontiguousarray(input_data))
                ctx["configured"].run([bindings], timeout=10000)
                return {oname: ctx["output_buffers"][oname].copy()
                        for oname in ctx["out_names"]}
            except Exception as e:
                logger.warning(f"[NPU] run({name}) fehlgeschlagen: {type(e).__name__}: {e}")
                return {}

    # =================================================================
    # Flags + Sync
    # =================================================================

    def sync_flags(self):
        """Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten."""
        self.scrfd_active = "scrfd" in self._active_ctx
        self.arcface_active = "arcface" in self._active_ctx
        self.yolo_active = "yolov8m" in self._active_ctx
        self.hand_active = "hand_landmark" in self._active_ctx
        self.pose_active = "pose" in self._active_ctx
        self.face_attr_active = "face_attr" in self._active_ctx

    def all_models_off(self):
        """Alle Modelle deaktivieren und unconfigurieren."""
        self._models_preloaded = False
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.hand_active = False
        self._notify("model_toggle", {"scrfd": False, "arcface": False,
                                       "yolov8m": False, "hand_landmark": False})
        for name in list(self._active_ctx.keys()):
            self.unconfigure(name)

    # =================================================================
    # Voice Coordination (NPU Pause/Resume)
    # =================================================================

    def pause_for_voice(self):
        """Pause inference - release VDevice so voice process can use NPU."""
        logger.info("[NPU_IPC] Voice requested - pausing vision...")
        self._models_preloaded = False
        self._paused_models = list(self._active_ctx.keys())

        for name in list(self._active_ctx.keys()):
            self.unconfigure(name)

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

    def resume_after_voice(self):
        """Resume inference after voice process released NPU."""
        logger.info("[NPU_IPC] Voice done - resuming vision...")

        for path in [NPU_VISION_PAUSED]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        time.sleep(0.5)

        # Wait for Whisper VDevice to be fully released
        if self._hailo_manager:
            logger.info("[NPU_IPC] Waiting for device to be free...")
            for i in range(25):
                if self._hailo_manager.is_device_free():
                    logger.info(f"[NPU_IPC] Device free after {i * 0.2:.1f}s")
                    break
                time.sleep(0.2)
            else:
                logger.warning("[NPU_IPC] Device not free after 5s - forcing GC")
                gc.collect()
                time.sleep(1.0)

            if not self._hailo_manager.acquire_for_vision(timeout=10.0):
                self._npu_paused = False
                return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"[NPU_IPC] Resume retry {attempt + 1}/{max_retries}...")
                    time.sleep(1.0 + attempt)

                self.load_models()
                for name in self._paused_models:
                    if name in self._models:
                        self.configure(name)

                self._npu_paused = False
                logger.info("[NPU_IPC] Vision resumed successfully")
                return
            except Exception as e:
                logger.error(f"[NPU_IPC] Resume attempt {attempt + 1} failed: {e}")
                if self._vdevice:
                    try:
                        del self._vdevice
                    except Exception:
                        pass
                    self._vdevice = None
                self._models.clear()
                gc.collect()

        logger.error("[NPU_IPC] Resume failed after all retries")
        self._npu_paused = False

    def check_voice_request(self) -> bool:
        """Prueft ob Voice NPU anfordert und handelt entsprechend.

        Returns: True wenn NPU pausiert/pausiert wird (Caller soll warten)
        """
        if os.path.exists(NPU_VOICE_REQUEST):
            # Stale File Check
            try:
                with open(NPU_VOICE_REQUEST, "r") as f:
                    req = json.load(f)
                voice_pid = req.get("pid", 0)
                if voice_pid and not os.path.exists(f"/proc/{voice_pid}"):
                    logger.warning(f"[NPU_IPC] Stale voice request von PID {voice_pid}")
                    try:
                        os.unlink(NPU_VOICE_REQUEST)
                    except FileNotFoundError:
                        pass
                    return False
            except (json.JSONDecodeError, FileNotFoundError):
                try:
                    os.unlink(NPU_VOICE_REQUEST)
                except FileNotFoundError:
                    pass
                return False

            if not self._npu_paused:
                try:
                    self.pause_for_voice()
                except Exception as e:
                    logger.error(f"[NPU_IPC] Pause failed: {e}")
                    self._npu_paused = True
            return True

        if self._npu_paused:
            try:
                self.resume_after_voice()
            except Exception as e:
                logger.error(f"[NPU_IPC] Resume crashed: {e}")
                self._npu_paused = False
            return True

        return False

    def auto_recover_models(self) -> bool:
        """Auto-Recovery wenn Models leer sind.

        Returns: True wenn Recovery laeuft (Caller soll warten)
        """
        if self._models or self._npu_paused:
            return False

        self._recovery_count += 1
        if self._recovery_count <= 3:
            logger.warning(f"[NPU] Models empty (attempt {self._recovery_count}/3)")
            try:
                if self._hailo_manager and not self._hailo_manager.is_device_free():
                    time.sleep(2)
                if self._hailo_manager:
                    self._hailo_manager.acquire_for_vision(timeout=10.0)
                self.load_models()
                for name in (self._paused_models or []):
                    if name in self._models:
                        self.configure(name)
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
            return True
        elif self._recovery_count == 4:
            logger.error("[NPU] Auto-recovery exhausted (3 attempts)")
        time.sleep(1)
        return True

    # =================================================================
    # NPU Watchdog (Anti-Oszillation)
    # =================================================================

    def watchdog_tick(self):
        """Anti-Oszillation. Laeuft jede Inference-Iteration."""
        _now = time.time()
        self._swap_log = [t for t in self._swap_log if _now - t < 1.0]
        if len(self._swap_log) >= 3:
            logger.warning(f"[WATCHDOG] Anti-Oscillation: {len(self._swap_log)} Swaps in 1s!")
            time.sleep(2.0)
            self._swap_log.clear()

    def record_swap(self):
        """Swap-Event fuer Watchdog registrieren."""
        self._swap_log.append(time.time())

    # =================================================================
    # Attention-Level basierte Orchestrierung
    # =================================================================

    def compute_attention_level(self) -> str:
        """Attention-Level aus PerceptionEngine NPU-Stage ableiten.

        NPU Idle-Modus: Stage wird von PerceptionEngine gesteuert.
        idle → nur yolov8m, person → +scrfd, face → scrfd+arcface.

        Gate 0 Fix: always_on Modus = IMMER "high" (30 FPS).
        Vorher: npu_stage blieb auf "idle" bei forced models → 10 FPS Throttle!
        """
        # always_on: Kein Throttle, immer 30 FPS
        if self._orchestration_mode == "always_on":
            return "high"

        if self._teachen and self._teachen.enabled:
            return "teach"

        # Forced models: Stage-Machine laeuft nicht → immer "high"
        if self._perception and self._perception._forced:
            return "high"

        # NPU-Stage aus PerceptionEngine als Attention-Level
        if self._perception and hasattr(self._perception, 'npu_stage'):
            stage = self._perception.npu_stage
            if stage == "idle":
                return "idle"
            elif stage == "person":
                return "normal"
            elif stage == "face":
                return "high"

        # Fallback auf CoreIntegrator
        if not self._core_integrator:
            return "normal"
        attention = self._core_integrator.get_attention()
        if attention < 0.2:
            return "idle"
        elif attention < 0.6:
            return "normal"
        else:
            return "high"

    def get_target_models(self, level: str) -> set:
        """Welche Modelle sollen bei diesem Attention-Level aktiv sein?

        face_attr ENTFERNT — verursachte Load/Unload Loop (Gate 0 Phase 1).
        face_attr laeuft nur wenn manuell konfiguriert.
        """
        if level == "idle":
            return {"yolov8m"}
        elif level == "normal":
            return {"yolov8m", "scrfd", "arcface"}
        elif level == "high":
            return {"yolov8m", "scrfd", "arcface", "pose", "hand_landmark"}
        elif level == "teach":
            return {"scrfd", "arcface"}
        return {"yolov8m", "scrfd", "arcface"}

    def get_target_fps_delay(self, level: str) -> float:
        """Target-Delay zwischen Frames fuer Adaptive FPS.

        Gate 0: idle von 10 auf 20 FPS erhoeht (Akzeptanz: FPS > 15).
        """
        delays = {"idle": 0.05, "normal": 0.04, "high": 0.033, "teach": 0.05}
        return delays.get(level, 0.04)

    def apply_attention_level(self, new_level: str):
        """FPS-Throttle setzen basierend auf Attention-Level.

        NPU Idle-Modus: PerceptionEngine steuert Stufen (idle/person/face).
        Modell-Swaps passieren NUR im InferenceEngine via perception.tick().
        Orchestrator setzt hier NUR den FPS-Throttle — KEINE Modell-Aenderungen!
        """
        # PerceptionEngine aktiv: NUR FPS-Throttle setzen, Modelle NICHT anfassen
        if self._perception and hasattr(self._perception, 'slots') and self._perception.slots:
            # FPS-Throttle anpassen (idle=10fps, person=15fps, face=30fps)
            self._target_frame_delay = self.get_target_fps_delay(new_level)

            with self._attention_level_lock:
                if self._attention_level != new_level:
                    self._attention_level = new_level
                    fps = round(1.0 / self._target_frame_delay) if self._target_frame_delay > 0 else 30
                    self._notify("attention_level", {"level": new_level, "fps_target": fps})
            return

        # Adaptive Modus (Legacy)
        with self._attention_level_lock:
            old_level = self._attention_level
            if old_level == new_level:
                return
            self._attention_level = new_level

        # Manueller Override? -> nicht eingreifen
        if self._perception and self._perception._forced:
            return

        wanted = self.get_target_models(new_level)
        have = set(self._active_ctx.keys())
        to_pause = have - wanted
        to_resume = wanted - have

        if not to_pause and not to_resume:
            return

        logger.info(f"[ORCHESTRATION] Level {old_level}->{new_level}: "
                     f"pause={to_pause or 'none'} resume={to_resume or 'none'}")

        for m in to_pause:
            self.unconfigure(m)
            if self._model_health:
                self._model_health.set_paused(m, True)
            time.sleep(0.1)

        for m in to_resume:
            if m in self._models and m not in self._active_ctx:
                self.configure(m)
                if self._model_health:
                    self._model_health.set_paused(m, False)

        self.sync_flags()
        self._target_frame_delay = self.get_target_fps_delay(new_level)
        self._notify("attention_level", {"level": new_level,
                     "fps_target": round(1.0 / self._target_frame_delay)})

    # =================================================================
    # Model Toggle (Panel-API)
    # =================================================================

    def toggle_model(self, model_key: str, enabled: bool):
        """Toggle model on/off via Perception Engine force_models()."""
        if not self._perception:
            logger.warning(f"[TOGGLE] Perception Engine nicht verfuegbar")
            return

        active_map = {"scrfd": "scrfd_active", "arcface": "arcface_active",
                      "yolov8m": "yolo_active", "hand_landmark": "hand_active",
                      "pose": "pose_active", "person_reid": "person_reid_active"}
        if model_key not in active_map:
            logger.warning(f"[TOGGLE] Unbekannter model_key: {model_key}")
            return

        # Panel nutzt "person_reid", Scheduler kennt "reid"
        sched_key = "reid" if model_key == "person_reid" else model_key
        current = set(self._active_ctx.keys())
        if enabled:
            wanted = current | {sched_key}
            if "arcface" in wanted and "scrfd" not in wanted:
                wanted.add("scrfd")
        else:
            wanted = current - {sched_key}
            if sched_key == "scrfd":
                wanted.discard("arcface")

        if wanted:
            self._perception.force_models(list(wanted))
            logger.info(f"[TOGGLE] force_models({list(wanted)}) via Panel")
        else:
            self._perception.force_models(None)
            logger.info("[TOGGLE] Alle Modelle aus -> Perception Auto-Modus")

    # =================================================================
    # Shutdown
    # =================================================================

    def release_all(self):
        """Alle Modelle unconfigurieren und VDevice freigeben."""
        for name in list(self._active_ctx.keys()):
            self.unconfigure(name)
        if self._vdevice:
            try:
                self._models.clear()
                del self._vdevice
                self._vdevice = None
            except Exception:
                pass
        if self._hailo_manager:
            try:
                self._hailo_manager.release_vision()
            except Exception:
                pass
