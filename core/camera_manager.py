#!/usr/bin/env python3
"""
CameraManager - RTSP + Cloud + Tentakel + Autonomer Modus.

Extrahiert aus moloch_service.py (Phase 4, Regel 10).

Verantwortlichkeiten:
  - RTSP Background Reader (mit Frozen-Frame-Detection + Auto-Reconnect)
  - eWeLink Cloud Verbindung (LED, Smart Tracking)
  - Tentakel-Modus (Smart Tracking <-> MOLOCH Takeover)
  - Autonomer Tracker (Face/Person Detection -> PTZ Moves)
  - Frozen Frame Watchdog
"""

import os
import time
import json
import threading
import logging

import cv2
import numpy as np

from core.cloud_controller import CloudController
from core.model_orchestrator import MODEL_PATHS
from core.mpo.autonomous_tracker import TrackerState

logger = logging.getLogger("CameraManager")

# RTSP URL
RTSP_URL = os.environ.get("MOLOCH_RTSP_URL", "")


class CameraManager:
    """RTSP + Cloud + Tentakel + Autonomer Modus."""

    def __init__(self, rtsp_url=None, model_orchestrator=None,
                 perception_engine=None, led_controller=None,
                 notify_callback=None, sync_flags_callback=None,
                 set_model_flags_callback=None, fps_reset_callback=None):
        """
        Args:
            rtsp_url: RTSP Stream URL (oder None fuer ENV-Variable)
            model_orchestrator: ModelOrchestrator fuer configure/unconfigure
            perception_engine: PerceptionEngine fuer force_models
            led_controller: LEDController fuer LED Steuerung
            notify_callback: callback(event, data) fuer UI Notifications
            sync_flags_callback: callback() um Model-Flags auf Service zu synchen
            set_model_flags_callback: callback(dict) um Model-Flags auf Service zu setzen
            fps_reset_callback: callback() um FPS Tracking zurueckzusetzen
        """
        self._rtsp_url = rtsp_url or RTSP_URL
        self._orchestrator = model_orchestrator
        self._perception = perception_engine
        self._led = led_controller
        self._notify = notify_callback or (lambda e, d: None)
        self._sync_flags = sync_flags_callback or (lambda: None)
        self._set_model_flags = set_model_flags_callback or (lambda d: None)
        self._fps_reset = fps_reset_callback or (lambda: None)

        # Running Flag (vom Service gesetzt)
        self.running = True

        # Frame Locks
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        # RTSP State
        self._last_frame_write = time.time()
        self._frozen_restart_count = 0
        self._rtsp_frame_hash = None
        self._rtsp_identical_count = 0
        self._rtsp_stop_reader = threading.Event()
        self._rtsp_thread = None
        self._rtsp_cap = None

        # Cloud Controller
        self._cloud = None
        self._has_calibrated = False

        # Cloud State (LED/Night/Alarm - fuer Panel Sync)
        self._cloud_state = {"led_level": 0, "alarm_active": False, "status_led": False}

        # Alarm State
        self._alarm_on = False

        # Smart Tracking State
        self._smart_tracking_on = False
        self._st_lock = threading.Lock()

        # Autonomous Tracking
        self._autonomous_mode = False
        self._manual_autonomous = False
        self._manual_mode = False
        self._tracker = None

        # Guardian/Tentakel Mode — IMMER aktiv (auch bei TAPPAS)
        # Guardian wartet auf Bewegung, Tentakel steuert Smart Tracking ↔ MOLOCH Handoff
        self._guardian_mode = True
        self._tentakel_enabled = True
        self._moloch_has_control = False
        self._takeover_reason = ""
        self._takeover_time = 0
        self._last_interesting_time = 0
        self._search_start_time = 0
        self.TAKEOVER_TIMEOUT = 30
        self.SEARCH_TIMEOUT = 120
        self._guardian_last_pan = None
        self._guardian_last_tilt = None
        self._guardian_move_thresh = 5.0
        self._guardian_move_count = 0
        self._guardian_move_required = 2
        self._takeover_cooldown_until = 0
        self.RELEASE_COOLDOWN = 60
        self.MAX_COOLDOWN = 180
        self.STARTUP_GRACE = 60
        self._failed_takeovers = 0
        self._takeover_found_something = False
        self._takeover_cooldown_until = time.time() + self.STARTUP_GRACE
        self._transitioning = False
        self._transition_lock = threading.Lock()
        self._waiting_for_first_detection = False
        self._first_detection_event = threading.Event()

        # Letzter manueller PTZ-Befehl (fuer FPS-Boost in Inference)
        self._last_manual_ptz = 0

        # Home Position (fuer Release -> Home -> ST)
        self._home_position = {"pan": 50.0, "tilt": -20.0}
        try:
            _home_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "config", "camera_home.json")
            if os.path.exists(_home_cfg):
                with open(_home_cfg) as f:
                    self._home_position = json.load(f)
                logger.info(f"[INIT] Home Position geladen: {self._home_position}")
        except Exception as e:
            logger.debug(f"[INIT] camera_home.json nicht geladen: {e}")

    def _update_status(self, text):
        """Status-Update via Notification Callback."""
        logger.info(f"[STATUS] {text}")
        self._notify("status", {"text": text})

    # =================================================================
    # Properties
    # =================================================================

    @property
    def moloch_has_control(self) -> bool:
        with self._transition_lock:
            return self._moloch_has_control

    @property
    def autonomous_mode(self) -> bool:
        return self._autonomous_mode

    @property
    def manual_mode(self) -> bool:
        return self._manual_mode

    @property
    def tentakel_enabled(self) -> bool:
        return self._tentakel_enabled

    @property
    def cloud(self):
        return self._cloud

    @property
    def cloud_state(self) -> dict:
        return self._cloud_state

    @property
    def alarm_on(self) -> bool:
        return self._alarm_on

    @property
    def smart_tracking_on(self) -> bool:
        return self._smart_tracking_on

    # =================================================================
    # RTSP Capture
    # =================================================================

    def start_rtsp(self):
        """Starte RTSP Background Reader (mit Frozen-Frame-Detection + Auto-Reconnect)."""
        # Alten Reader-Thread SAUBER beenden
        if hasattr(self, '_rtsp_stop_reader'):
            self._rtsp_stop_reader.set()

        if hasattr(self, '_rtsp_thread') and self._rtsp_thread is not None:
            self._rtsp_thread.join(timeout=5)
            if self._rtsp_thread.is_alive():
                logger.warning("[RTSP] Alter Reader-Thread lebt noch nach 5s join")
            self._rtsp_thread = None

        if hasattr(self, '_rtsp_cap') and self._rtsp_cap is not None:
            try:
                self._rtsp_cap.release()
            except Exception:
                pass
            self._rtsp_cap = None

        self._rtsp_stop_reader = threading.Event()
        self._rtsp_frame_hash = None
        self._rtsp_identical_count = 0
        stop_event = self._rtsp_stop_reader

        rtsp_url = self._rtsp_url

        def _rtsp_connect():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        def rtsp_reader():
            cap = _rtsp_connect()
            self._rtsp_cap = cap

            if not cap.isOpened():
                self._update_status(f"RTSP FEHLER: {rtsp_url}")
                return

            self._update_status("RTSP aktiv")
            self._last_frame_write = time.time()
            identical_count = 0
            prev_hash = None

            while self.running and not stop_event.is_set():
                grabbed = cap.grab()
                if grabbed:
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        frame_hash = hash(frame[::20, ::20].tobytes())
                        if frame_hash == prev_hash:
                            identical_count += 1
                            if identical_count >= 10:
                                self._frozen_restart_count += 1
                                logger.warning(
                                    f"[WATCHDOG] RTSP reconnect — "
                                    f"{identical_count} identical frames detected"
                                )
                                cap.release()
                                if stop_event.wait(2):
                                    break
                                cap = _rtsp_connect()
                                self._rtsp_cap = cap
                                identical_count = 0
                                prev_hash = None
                                if cap.isOpened():
                                    logger.info("[RTSP] Stream wiederhergestellt (frozen-detect)")
                                    self._last_frame_write = time.time()
                                else:
                                    logger.warning("[RTSP] Reconnect fehlgeschlagen, retry...")
                                continue
                        else:
                            prev_hash = frame_hash
                            identical_count = 0

                        with self._frame_lock:
                            self._latest_frame = frame
                        self._last_frame_write = time.time()
                    else:
                        if stop_event.wait(0.05):
                            break
                else:
                    self._frozen_restart_count += 1
                    logger.warning(
                        f"[RTSP] grab() fehlgeschlagen - Reconnect "
                        f"#{self._frozen_restart_count} in 2s..."
                    )
                    cap.release()
                    if stop_event.wait(2):
                        break
                    cap = _rtsp_connect()
                    self._rtsp_cap = cap
                    identical_count = 0
                    prev_hash = None
                    if cap.isOpened():
                        logger.info("[RTSP] Stream wiederhergestellt (grab-fail)")
                        self._last_frame_write = time.time()
                    else:
                        logger.warning("[RTSP] Reconnect fehlgeschlagen, retry in 5s...")
                        if stop_event.wait(5):
                            break

            cap.release()
            logger.info("[RTSP] Reader-Thread beendet")

        t = threading.Thread(target=rtsp_reader, daemon=True, name="RTSPReader")
        t.start()
        self._rtsp_thread = t

    def get_frame(self):
        """Thread-safe aktuellen Frame holen."""
        with self._frame_lock:
            return self._latest_frame

    def get_frame_age(self) -> float:
        """Alter des letzten Frames in Sekunden."""
        return time.time() - self._last_frame_write

    # =================================================================
    # Frozen Frame Watchdog
    # =================================================================

    def start_watchdog(self):
        """Frozen Frame Watchdog Thread starten."""
        threading.Thread(target=self._frozen_frame_watchdog,
                         daemon=True, name="FrozenWatchdog").start()

    def _frozen_frame_watchdog(self):
        """Backup-Watchdog: Erkennt wenn Reader-Thread kein Frame mehr liefert."""
        while self.running:
            try:
                time.sleep(10)
                frame_age = time.time() - self._last_frame_write
                if frame_age > 30:
                    logger.warning(
                        f"[WATCHDOG] RTSP reconnect — "
                        f"kein Frame seit {frame_age:.0f}s (Reader haengt)"
                    )
                    try:
                        self.start_rtsp()
                        logger.info("[WATCHDOG] RTSP Stream neu gestartet")
                    except Exception as e:
                        logger.error(f"[WATCHDOG] RTSP Reconnect Error: {e}")
                    if self._frozen_restart_count >= 5:
                        logger.error("[WATCHDOG] 5 Reconnects fehlgeschlagen, warte 60s")
                        time.sleep(60)
                        self._frozen_restart_count = 0
            except Exception as e:
                logger.error(f"[WATCHDOG] Error: {e}")

    # =================================================================
    # Cloud / Camera
    # =================================================================

    def connect_cloud(self):
        """Connect to eWeLink cloud."""
        try:
            self._cloud = CloudController()
            self._cloud.start()
            if self._led:
                self._led.set_cloud(self._cloud)
            if self._cloud.connected:
                logger.info("eWeLink Cloud verbunden")
                # Gate 0: Smart Tracking PERMANENT AUS — Moloch steuert ALLES
                if self._cloud.set_smart_tracking(False):
                    self._set_smart_tracking_state(False)
                    logger.info("[STARTE] ptz_modus=moloch_allein smart_tracking=deaktiviert")
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
        # PTZ Arbiter synchronisieren
        try:
            from core.ptz_arbiter import get_ptz_arbiter
            get_ptz_arbiter().sync_smart_tracking(value)
        except Exception:
            pass

    def toggle_smart_tracking(self):
        """Gate 0: Smart Tracking bleibt PERMANENT AUS. Toggle deaktiviert."""
        logger.info("[GATE0] Smart Tracking Toggle ignoriert — permanent AUS")
        self._update_status("ST permanent AUS (Gate 0)")

    # =================================================================
    # Tentakel-Logik (Smart Tracking <-> MOLOCH Takeover)
    # =================================================================

    def moloch_takeover(self, reason: str):
        """MOLOCH uebernimmt: NPU Modelle AN -> Warte auf Detection -> ST AUS -> Tracker AN."""
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
                if self._perception and self._perception._forced:
                    logger.info(f"[TENTAKEL] User forced_models={self._perception._forced} - NPU bleibt!")
                    self._sync_flags()
                    self._first_detection_event.set()
                    self._transitioning = False
                    return

                active_ctx = self._orchestrator.active_ctx if self._orchestrator else {}
                all_loaded = all(m in active_ctx for m in MODEL_PATHS)
                if all_loaded:
                    logger.info("[TENTAKEL] Alle Modelle bereits auf NPU")
                else:
                    self._update_status("Takeover: NPU Modelle laden...")
                    logger.info("[TENTAKEL] Lade alle NPU Modelle (ST laeuft weiter)")
                    for _m in MODEL_PATHS:
                        if _m not in active_ctx:
                            if self._orchestrator:
                                self._orchestrator.configure(_m)
                            time.sleep(0.2)

                self._sync_flags()
                orch = self._orchestrator
                self._notify("model_toggle", {
                    "scrfd": orch.scrfd_active if orch else False,
                    "arcface": orch.arcface_active if orch else False,
                    "yolov8m": orch.yolo_active if orch else False,
                    "hand_landmark": orch.hand_active if orch else False})

                self._first_detection_event.clear()
                self._waiting_for_first_detection = True
                self._update_status("Takeover: Warte auf Detection...")
                logger.info("[TENTAKEL] NPU aktiv, warte auf Detection (ST laeuft weiter)...")

                got_detection = self._first_detection_event.wait(timeout=10.0)
                self._waiting_for_first_detection = False

                if not got_detection:
                    logger.info("[TENTAKEL] 10s keine Detection - Takeover abgebrochen")
                    self._set_model_flags({
                        "scrfd_active": False, "yolo_active": False})
                    self._notify("model_toggle", {"scrfd": False, "yolov8m": False})
                    self._moloch_has_control = False
                    self._takeover_found_something = False
                    self._failed_takeovers += 1
                    cooldown = min(self.RELEASE_COOLDOWN * (1.5 ** self._failed_takeovers),
                                   self.MAX_COOLDOWN)
                    self._takeover_cooldown_until = time.time() + cooldown
                    self._update_status("Tentakel scannt wieder")
                    logger.info(f"[TENTAKEL] Fehlversuch #{self._failed_takeovers}, Cooldown {cooldown:.0f}s")
                    return

                logger.info("[TENTAKEL] Detection erkannt! ST AUS + Tracker AN")
                self._update_status("Takeover: ST AUS...")
                st_off = False
                if self._cloud and self._cloud.connected:
                    for attempt in range(3):
                        if self._cloud.set_smart_tracking(False):
                            self._set_smart_tracking_state(False)
                            st_off = True
                            break
                        logger.warning(f"[TENTAKEL] ST AUS Versuch {attempt+1}/3 fehlgeschlagen")
                        time.sleep(0.5)

                if not st_off:
                    logger.error("[TENTAKEL] ST AUS fehlgeschlagen - Takeover ABBRUCH")
                    self._set_model_flags({
                        "scrfd_active": False, "yolo_active": False})
                    self._notify("model_toggle", {"scrfd": False, "yolov8m": False})
                    self._moloch_has_control = False
                    self._update_status("Takeover abgebrochen: ST nicht erreichbar")
                    return

                self.enable_autonomous()
                if self._led:
                    self._led.on()
                # PTZ Arbiter: MOLOCH uebernimmt
                try:
                    from core.ptz_arbiter import get_ptz_arbiter
                    get_ptz_arbiter().set_moloch_uebernimmt(reason)
                except Exception:
                    pass
                self._update_status(f"MOLOCH: {reason}")
                logger.info(f"[TENTAKEL] Takeover komplett (fliessend): {reason}")
            except Exception as e:
                logger.error(f"[TENTAKEL] Takeover Fehler: {e}")
                self._moloch_has_control = False
            finally:
                self._waiting_for_first_detection = False
                self._transitioning = False

        threading.Thread(target=do_takeover, daemon=True).start()

    def moloch_release(self):
        """MOLOCH gibt zurueck: Tracker STOP -> ST AN -> Aufraumen."""
        with self._transition_lock:
            if not self._moloch_has_control or self._transitioning:
                return
            self._transitioning = True
        try:
            self._waiting_for_first_detection = False
            self._first_detection_event.set()
            logger.info("[TENTAKEL] MOLOCH gibt Kamera zurueck an Smart Tracking")

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

            if self._led:
                self._led.off()

            self._autonomous_mode = False
            if self._tracker:
                self._tracker.disable()
            logger.info("[TENTAKEL] Tracker gestoppt")
            self._notify("auto_mode", {"state": "disabled"})

            # Gate 0: Smart Tracking bleibt AUS — Moloch steuert ALLES
            # PTZ Arbiter: zurueck zu AUTONOM
            try:
                from core.ptz_arbiter import get_ptz_arbiter
                get_ptz_arbiter().set_moloch_autonom("release")
            except Exception:
                pass

            if self._perception and self._perception._forced:
                self._sync_flags()
                active_ctx = self._orchestrator.active_ctx if self._orchestrator else {}
                logger.info(f"[TENTAKEL] User forced={self._perception._forced} "
                            f"- Flags aus NPU: {list(active_ctx.keys())}")
            else:
                self._set_model_flags({
                    "scrfd_active": False, "arcface_active": False,
                    "yolo_active": False, "hand_active": False})

            orch = self._orchestrator
            self._notify("model_toggle", {
                "scrfd": orch.scrfd_active if orch else False,
                "arcface": orch.arcface_active if orch else False,
                "yolov8m": orch.yolo_active if orch else False,
                "hand_landmark": orch.hand_active if orch else False})
            self._fps_reset()
            active_ctx = self._orchestrator.active_ctx if self._orchestrator else {}
            logger.info(f"[TENTAKEL] Inference gestoppt, Modelle auf NPU: {list(active_ctx.keys())}")

            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0

            if self._takeover_found_something:
                self._failed_takeovers = 0
                cooldown = self.RELEASE_COOLDOWN
            else:
                self._failed_takeovers += 1
                cooldown = min(self.RELEASE_COOLDOWN * (1.5 ** self._failed_takeovers),
                               self.MAX_COOLDOWN)
            self._takeover_found_something = False
            self._takeover_cooldown_until = time.time() + cooldown

            self._update_status("Tentakel scannt wieder")
            logger.info(f"[TENTAKEL] Release komplett - Cooldown {cooldown:.0f}s")
        finally:
            self._transitioning = False

    def _check_guardian_timeout(self):
        """Pruefe ob MOLOCH die Kamera zurueckgeben soll."""
        if not self._guardian_mode or self._transitioning or self._manual_mode:
            return
        if (self._autonomous_mode and not self._moloch_has_control
                and not self._manual_autonomous
                and (time.time() - getattr(self, "_autonomous_enabled_at", 0)) > self.STARTUP_GRACE):
            # Bei TAPPAS: Tracker ist NIE orphaned — TAPPAS ist das Detektionssystem
            _use_tappas = os.environ.get("MOLOCH_USE_TAPPAS", "0") == "1"
            if _use_tappas:
                return
            # Ohne TAPPAS: Tracker nicht killen wenn er aktiv arbeitet
            if self._tracker and self._tracker.state not in (TrackerState.IDLE,):
                return
            logger.warning("[SAFETY] Orphaned autonomous mode detected - disabling")
            self.disable_autonomous()
            # Cooldown: verhindert sofortiges Re-Enable durch Retry-Logik (Z. 670)
            # Ohne Cooldown: Retry nach 10s -> Orphan nach 60s -> endloser Warn-Cycle
            self._auto_retry_time = time.time() + 120
            return
        if not self._moloch_has_control:
            return
        now = time.time()

        # Waehrend aktiver Suche: NUR SEARCH_TIMEOUT entscheidet
        # TAKEOVER_TIMEOUT darf aktive Suche nicht abbrechen
        is_searching = (self._tracker and self._autonomous_mode
                        and self._tracker.state == TrackerState.SEARCHING)

        if is_searching:
            # SEARCH_TIMEOUT: Suche laeuft zu lange -> zurueckgeben
            if self._search_start_time == 0:
                self._search_start_time = now
            elif now - self._search_start_time > self.SEARCH_TIMEOUT:
                logger.info(f"[TENTAKEL] Search timeout ({self.SEARCH_TIMEOUT}s) - zurueckgeben")
                threading.Thread(target=self.moloch_release, daemon=True).start()
                return
        else:
            self._search_start_time = 0
            # TAKEOVER_TIMEOUT: Nichts Interessantes mehr -> zurueckgeben
            if now - self._last_interesting_time > self.TAKEOVER_TIMEOUT:
                logger.info(f"[TENTAKEL] Takeover timeout ({self.TAKEOVER_TIMEOUT}s) - zurueckgeben")
                threading.Thread(target=self.moloch_release, daemon=True).start()
                return

    def signal_detection(self):
        """Vom Inference Loop aufgerufen wenn Detection erkannt (fuer fliessenden Takeover)."""
        if self._waiting_for_first_detection:
            self._first_detection_event.set()

    def signal_interesting(self):
        """Vom Inference Loop aufgerufen wenn etwas Interessantes erkannt (verlaengert Takeover)."""
        self._last_interesting_time = time.time()
        self._takeover_found_something = True

    # =================================================================
    # Kamera-Status Polling
    # =================================================================

    def start_cam_status_loop(self, write_status_callback=None):
        """Starte Kamera-Status + IPC Polling Thread.

        Args:
            write_status_callback: callback() um Status-JSON zu schreiben
        """
        self._write_status_json = write_status_callback or (lambda: None)
        threading.Thread(target=self._cam_status_loop,
                         daemon=True, name="CamStatusLoop").start()

    def _cam_status_loop(self):
        """Kamera-Status + IPC Status-JSON polling loop (1.5s Intervall)."""
        while self.running:
            try:
                self._update_cam_status()
            except Exception as e:
                logger.error(f"Cam status error: {e}")
            try:
                self._write_status_json()
            except Exception:
                pass
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

                # Phase 3 Fix: Retry autonomous wenn Kamera beim Boot zu spaet kam
                # enable_autonomous() wird beim Boot aufgerufen, schlaegt aber fehl
                # wenn die Kamera noch nicht erreichbar ist (Race Condition).
                if not self._autonomous_mode and not self._manual_mode:
                    _last_retry = getattr(self, '_auto_retry_time', 0)
                    if time.time() - _last_retry > 10.0:
                        self._auto_retry_time = time.time()
                        logger.info("[AUTONOM] Kamera online aber autonomous_mode=False - retry")
                        self.enable_autonomous()

                pos = cam.get_position()
                if pos:
                    pan, tilt = pos.pan, pos.tilt
                    ptz_text = f"Pan: {pan:.1f}  Tilt: {tilt:.1f}"

                    # PTZ-Tracker: Position aufzeichnen (fuer restless_score)
                    try:
                        from core.ptz_tracker import get_ptz_tracker
                        ptz_tracker = get_ptz_tracker()
                        ptz_tracker.record_position(pan, tilt)
                        # Tracker-Stage aktualisieren
                        if self._autonomous_mode and self._tracker:
                            from core.mpo.autonomous_tracker import TrackerState
                            ts = self._tracker.state
                            if ts in (TrackerState.LOCKED, TrackerState.FROZEN, TrackerState.COAST):
                                ptz_tracker.set_stage("locked")
                            elif ts == TrackerState.SEARCHING:
                                ptz_tracker.set_stage("searching")
                            elif ts == TrackerState.TRACKING:
                                ptz_tracker.set_stage("tracking")
                            else:
                                ptz_tracker.set_stage("idle")
                        elif not self._autonomous_mode:
                            ptz_tracker.set_stage("idle")
                    except Exception:
                        pass

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
                                        self.moloch_takeover("Kamera trackt etwas")
                                    else:
                                        remaining = self._takeover_cooldown_until - time.time()
                                        logger.info(f"[TENTAKEL] Cooldown aktiv, noch {remaining:.0f}s")
                                        self._guardian_move_count = 0
                            else:
                                self._guardian_move_count = max(0, self._guardian_move_count - 1)
                                # Idle Pre-Load
                                orch = self._orchestrator
                                if orch:
                                    active_ctx = orch.active_ctx
                                    configuring = orch.configuring
                                    if (not orch._models_preloaded
                                            and not active_ctx
                                            and time.time() >= self._takeover_cooldown_until
                                            and configuring.is_set()):
                                        orch._models_preloaded = True
                                        def _idle_preload():
                                            try:
                                                logger.info("[TENTAKEL] Idle Pre-Load: Alle NPU Modelle vorladen...")
                                                for _m in MODEL_PATHS:
                                                    if _m not in active_ctx:
                                                        orch.configure(_m)
                                                        time.sleep(0.2)
                                                if all(m in active_ctx for m in MODEL_PATHS):
                                                    logger.info("[TENTAKEL] Idle Pre-Load: Alle Modelle ready auf NPU")
                                                else:
                                                    logger.warning(f"[TENTAKEL] Idle Pre-Load: Nur {list(active_ctx.keys())} konfiguriert!")
                                                    orch._models_preloaded = False
                                            except Exception as e:
                                                logger.error(f"[TENTAKEL] Idle Pre-Load Fehler: {e}")
                                                orch._models_preloaded = False
                                        threading.Thread(target=_idle_preload, daemon=True).start()
                        self._guardian_last_pan = pan
                        self._guardian_last_tilt = tilt
        except Exception:
            pass

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

    # =================================================================
    # Autonomous Mode
    # =================================================================

    def enable_autonomous(self):
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
                self._autonomous_enabled_at = time.time()
                self._update_status("Modus: AUTONOM - MOLOCH sucht...")
                logger.info("Switched to AUTONOMOUS mode")
                self._notify("auto_mode", {"state": "active"})
            except Exception as e:
                logger.error(f"Autonomous start failed: {e}")
                self._update_status(f"AUTONOM Fehler: {e}")
                self._notify("auto_mode", {"state": "failed"})

        threading.Thread(target=do_start, daemon=True).start()

    def disable_autonomous(self):
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

    def toggle_autonomous_manual(self):
        """Toggle AUTONOM/MANUELL von GUI-Button."""
        if not self._manual_mode:
            logger.info("[MODUS] Wechsel zu MANUELL - Kamera-Kontrolle gesperrt")
            self._manual_mode = True
            self._tentakel_enabled = False

            if self._autonomous_mode:
                self.disable_autonomous()

            self._moloch_has_control = False
            self._manual_autonomous = False
            self._takeover_reason = ""
            self._guardian_move_count = 0

            if self._perception:
                active_ctx = self._orchestrator.active_ctx if self._orchestrator else {}
                frozen = list(active_ctx.keys())
                self._perception.force_models(frozen)
                logger.info(f"[MODUS] MANUELL -> force_models({frozen}) - Modell-Swap gesperrt")

            def stop_cam_control():
                if self._cloud and self._cloud.connected:
                    if self._cloud.set_smart_tracking(False):
                        self._set_smart_tracking_state(False)
                if self._led:
                    self._led.off()
            threading.Thread(target=stop_cam_control, daemon=True).start()

            self._notify("auto_mode", {"state": "manual"})
            self._update_status("MANUELL - Service beobachtet")
        else:
            logger.info("[MODUS] Wechsel zu AUTONOM - Kamera-Kontrolle freigegeben")
            self._manual_mode = False
            self._tentakel_enabled = True

            if self._perception:
                self._perception.force_models(None)
                logger.info("[MODUS] AUTONOM -> force_models(None) - Perception Auto-Modus")

            # Gate 0: Smart Tracking bleibt AUS, Arbiter auf AUTONOM
            try:
                from core.ptz_arbiter import get_ptz_arbiter
                get_ptz_arbiter().set_moloch_autonom("autonom_toggle")
            except Exception:
                pass

            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0
            self._takeover_cooldown_until = time.time() + 10

            self._notify("auto_mode", {"state": "autonomous"})
            self._update_status("AUTONOM - Moloch steuert")

    # =================================================================
    # Tracker Stop
    # =================================================================

    # =================================================================
    # Panel Command Handler (PTZ / Cloud / Snapshot)
    # =================================================================

    def ptz_move(self, direction, speed=0.3):
        """PTZ Bewegung in eine Richtung oder Home."""
        self._last_manual_ptz = time.time()
        from core.hardware.camera import get_camera_controller
        cam = get_camera_controller()
        if not cam.is_connected:
            cam.connect()
        if direction == 'home':
            cam.goto_home()
        elif direction in ('up', 'down', 'left', 'right'):
            cam.move_manual(direction, speed=speed)
        logger.info(f"[PTZ] move {direction}")

    def ptz_goto(self, position):
        """PTZ zu vordefinierter Position fahren."""
        positions = {
            'werkstatt': (0.0, -20.0),
            'wohnzimmer': (-90.0, 0.0),
        }
        if position not in positions:
            logger.warning(f"[PTZ] Unbekannte Position: '{position}'")
            return
        from core.hardware.camera import get_camera_controller
        cam = get_camera_controller()
        if not cam.is_connected:
            cam.connect()
        pan, tilt = positions[position]
        cam.move_absolute(pan=pan, tilt=tilt)
        logger.info(f"[PTZ] goto {position} ({pan}, {tilt})")

    def ptz_calibrate(self):
        """PTZ Kalibrierung triggern."""
        from core.hardware.camera_cloud_bridge import CameraCloudBridgeSync
        bridge = CameraCloudBridgeSync()
        bridge.trigger_ptz_calibration()
        logger.info("[PTZ] Kalibrierung getriggert")

    def cloud_set_night_mode(self, level):
        """Weisse LEDs / Night Vision Mode setzen."""
        if not self._cloud or not self._cloud.connected:
            return
        night_modes = {0: 'day', 1: 'auto', 2: 'night', 3: 'night'}
        mode = night_modes.get(int(level), 'day')
        self._cloud.run(self._cloud.bridge.set_night(mode))
        self._cloud_state["led_level"] = int(level)
        logger.info(f"[CLOUD] Night mode: {mode} (level={level})")

    def cloud_toggle_alarm(self):
        """Alarm ein/ausschalten."""
        if not self._cloud or not self._cloud.connected:
            return
        self._alarm_on = not self._alarm_on
        self._cloud.run(self._cloud.bridge.set_alarm(self._alarm_on))
        self._cloud_state["alarm_active"] = self._alarm_on
        logger.info(f"[CLOUD] Alarm: {'AN' if self._alarm_on else 'AUS'}")

    def cloud_toggle_status_led(self):
        """Status LED toglen."""
        if not self._cloud or not self._cloud.connected:
            return
        self._status_led_on = not getattr(self, '_status_led_on', False)
        self._cloud.run(self._cloud.bridge.set_status_led(self._status_led_on))
        self._cloud_state["status_led"] = self._status_led_on
        logger.info(f"[CLOUD] Status LED: {self._status_led_on}")

    def cloud_sync(self):
        """Cloud-Status synchronisieren."""
        if not self._cloud or not self._cloud.connected:
            return
        params = self._cloud.run(self._cloud.bridge.get_device_params())
        if params and isinstance(params, dict):
            nv = int(params.get("nightVision", 1))
            self._cloud_state["led_level"] = 2 if nv == 2 else 0
            self._cloud_state["alarm_active"] = bool(params.get("alarmNotify", False))
            self._cloud_state["status_led"] = bool(params.get("sledOnline", False))
            logger.info(f"[CLOUD] Sync: nightVision={nv} led_level={self._cloud_state.get('led_level')}")

    def take_snapshot(self):
        """Snapshot vom aktuellen Frame → media/snapshots/ (Galerie Captures-Tab)."""
        import cv2
        frame = None
        with self._annotated_lock:
            if self._annotated_frame is not None:
                frame = self._annotated_frame.copy()
        if frame is None:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
        if frame is None:
            logger.warning("[SNAPSHOT] Kein Frame verfuegbar")
            return None
        snap_dir = os.path.expanduser("~/moloch/media/snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(snap_dir, f"snap_{ts}.jpg")
        cv2.imwrite(path, frame)
        logger.info(f"[SNAPSHOT] Gespeichert: {path}")
        return path

    def take_detach_snapshot(self):
        """Frame beim Detach speichern → galerie/detach/."""
        import cv2
        frame = None
        with self._annotated_lock:
            if self._annotated_frame is not None:
                frame = self._annotated_frame.copy()
        if frame is None:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
        if frame is None:
            return None
        detach_dir = os.path.expanduser("~/moloch/galerie/detach")
        os.makedirs(detach_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(detach_dir, f"detach_{ts}.jpg")
        cv2.imwrite(path, frame)
        logger.info(f"[DETACH] Bild gespeichert: {path}")
        return path

    def take_teach_snapshot(self):
        """Teach-Foto speichern → media/teach/ (Galerie Teach-Tab)."""
        import cv2
        frame = None
        with self._annotated_lock:
            if self._annotated_frame is not None:
                frame = self._annotated_frame.copy()
        if frame is None:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
        if frame is None:
            logger.warning("[TEACH] Kein Frame verfuegbar")
            return None
        teach_dir = os.path.expanduser("~/moloch/media/teach")
        os.makedirs(teach_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(teach_dir, f"teach_{ts}.jpg")
        cv2.imwrite(path, frame)
        logger.info(f"[TEACH] Foto gespeichert: {path}")
        return path

    def stop_tracker(self):
        """Tracker sauber stoppen."""
        if self._tracker:
            try:
                self._tracker.stop()
            except Exception:
                pass
