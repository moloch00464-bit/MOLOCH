#!/usr/bin/env python3
"""Patch: Panel-Service Bridge via /dev/shm.

Panel verbindet sich zu LAUFENDEM systemd MolochService statt eigenen zu erstellen.
Service schreibt Frames + Status nach /dev/shm, Panel liest davon.

Aenderungen:
  1. moloch_service.py: /dev/shm Frame+Status Output + Panel-Command Polling
  2. moloch_unified_panel.py: ServiceProxy + Remote-Mode Detection

IPC-Kanaele:
  /dev/shm/moloch_frame       - Annotierter Frame (Header + Raw BGR Bytes)
  /dev/shm/moloch_status.json - Service Status (Modelle, FPS, Autonomie, Perception)
  /tmp/moloch_cmd.json         - Panel -> Service Kommandos
"""
import sys

# =====================================================
# STEP 1: Patch moloch_service.py - /dev/shm Output
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    svc_code = f.read()

svc_changes = 0

# 1a: Add struct import (idempotent)
if "import struct" not in svc_code:
    old_import = """import json
import gc
import asyncio"""
    new_import = """import json
import gc
import struct
import asyncio"""
    if old_import in svc_code:
        svc_code = svc_code.replace(old_import, new_import, 1)
        svc_changes += 1
        print("SVC 1a: struct import hinzugefuegt")
    else:
        print("ERROR: Import-Block nicht gefunden")
        sys.exit(1)
else:
    print("SVC 1a: struct bereits importiert")
    svc_changes += 1

# 1b: Add _write_shm(), _poll_panel_cmds(), _execute_panel_cmd() before __main__
old_main = """if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)

    print("M.O.L.O.C.H. Core Service - Phase 2")
    service = MolochService()
    service.init()
    service.start(blocking=True)"""

new_shm_plus_main = """    # =========================================================================
    # Panel IPC via /dev/shm
    # =========================================================================

    _shm_seq = 0

    def _write_shm(self, frame):
        \"\"\"Write frame + status to /dev/shm for Panel IPC.\"\"\"
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
                "npu_paused": self._npu_paused,
                "active_models": list(self._active_ctx.keys()),
                "autonomous_mode": self._autonomous_mode,
                "moloch_has_control": self._moloch_has_control,
                "tentakel_enabled": getattr(self, '_tentakel_enabled', False),
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
        \"\"\"Poll for commands from Panel via IPC file.\"\"\"
        cmd_file = '/tmp/moloch_cmd.json'
        while self.running:
            try:
                if os.path.exists(cmd_file):
                    with open(cmd_file) as f:
                        cmd = json.load(f)
                    os.unlink(cmd_file)
                    self._execute_panel_cmd(cmd)
            except Exception as e:
                logger.debug(f"Panel cmd poll: {e}")
                try:
                    os.unlink(cmd_file)
                except FileNotFoundError:
                    pass
            time.sleep(1.0)

    def _execute_panel_cmd(self, cmd):
        \"\"\"Execute a command from the Panel.\"\"\"
        action = cmd.get('action')
        logger.info(f"[IPC] Panel command: {cmd}")
        if action == 'toggle_model':
            model = cmd.get('model')
            enabled = cmd.get('enabled', False)
            if model:
                self.toggle_model(model, enabled)
        elif action == 'toggle_smart_tracking':
            self._toggle_smart_tracking()
        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
        elif action == 'reload_face_db':
            self._reload_face_db()
        elif action == 'set_threshold':
            attr = cmd.get('attr')
            value = cmd.get('value')
            if attr and value is not None and hasattr(self, attr):
                setattr(self, attr, float(value))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)

    print("M.O.L.O.C.H. Core Service - Phase 2")
    service = MolochService()
    service.init()
    service.start(blocking=True)"""

if old_main in svc_code:
    svc_code = svc_code.replace(old_main, new_shm_plus_main, 1)
    svc_changes += 1
    print("SVC 1b: _write_shm() + _poll_panel_cmds() eingefuegt")
else:
    print("ERROR: __main__ Block nicht gefunden")
    sys.exit(1)

# 1c: Call _write_shm() after annotated frame is set (main detection path)
old_annotated_main = """            with self._annotated_lock:
                self._annotated_frame = annotated

    # =========================================================================
    # Cross-Process NPU Coordination"""

new_annotated_main = """            with self._annotated_lock:
                self._annotated_frame = annotated

            # Panel IPC: Frame + Status nach /dev/shm
            self._write_shm(annotated)

    # =========================================================================
    # Cross-Process NPU Coordination"""

if old_annotated_main in svc_code:
    svc_code = svc_code.replace(old_annotated_main, new_annotated_main, 1)
    svc_changes += 1
    print("SVC 1c: _write_shm() nach Haupt-Annotation")
else:
    print("ERROR: Annotated frame -> Cross-Process Block nicht gefunden")
    sys.exit(1)

# 1d: Start panel command polling thread in start()
old_start_threads = """        # Inference Loop
        threading.Thread(target=self._inference_loop, daemon=True, name="InferenceLoop").start()

        # Kamera-Status Polling (ersetzt root.after(3000, ...))
        threading.Thread(target=self._cam_status_loop, daemon=True, name="CamStatusLoop").start()"""

new_start_threads = """        # Inference Loop
        threading.Thread(target=self._inference_loop, daemon=True, name="InferenceLoop").start()

        # Kamera-Status Polling (ersetzt root.after(3000, ...))
        threading.Thread(target=self._cam_status_loop, daemon=True, name="CamStatusLoop").start()

        # Panel IPC Command Polling
        threading.Thread(target=self._poll_panel_cmds, daemon=True, name="PanelCmdPoll").start()"""

if old_start_threads in svc_code:
    svc_code = svc_code.replace(old_start_threads, new_start_threads, 1)
    svc_changes += 1
    print("SVC 1d: Panel command polling thread gestartet")
else:
    print("ERROR: start() Thread-Block nicht gefunden")
    sys.exit(1)

# 1e: Add shm cleanup to stop()
old_stop_end = """        # IPC cleanup
        for path in [NPU_VISION_PAUSED]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        logger.info("M.O.L.O.C.H. Service gestoppt")"""

new_stop_end = """        # IPC cleanup
        for path in [NPU_VISION_PAUSED,
                     '/dev/shm/moloch_frame', '/dev/shm/moloch_frame.tmp',
                     '/dev/shm/moloch_status.json', '/dev/shm/moloch_status.tmp']:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        logger.info("M.O.L.O.C.H. Service gestoppt")"""

if old_stop_end in svc_code:
    svc_code = svc_code.replace(old_stop_end, new_stop_end, 1)
    svc_changes += 1
    print("SVC 1e: shm cleanup in stop()")
else:
    print("WARN: stop() cleanup block nicht gefunden (unkritisch)")

# Write service file
with open(svc_path, "w") as f:
    f.write(svc_code)
print(f"SVC: {svc_changes} patches angewendet")

# =====================================================
# STEP 2: Patch moloch_unified_panel.py - ServiceProxy
# =====================================================
panel_path = "/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py"
with open(panel_path, "r") as f:
    panel_code = f.read()

panel_changes = 0

# 2a: Add struct import if missing
if "import struct" not in panel_code:
    old_panel_import = "import threading"
    new_panel_import = "import struct\nimport threading"
    if old_panel_import in panel_code:
        panel_code = panel_code.replace(old_panel_import, new_panel_import, 1)
        panel_changes += 1
        print("PANEL 2a: struct import hinzugefuegt")
    else:
        print("WARN: threading import nicht gefunden fuer struct-injection")
else:
    print("PANEL 2a: struct bereits importiert")

# 2b: Add ServiceProxy class before MolochUnifiedPanel
old_class_def = """class MolochUnifiedPanel:"""

service_proxy_class = '''

# =========================================================================
# ServiceProxy: IPC Bridge zu laufendem systemd MolochService
# =========================================================================

class _PerceptionProxy:
    """Proxy fuer Perception Engine Status-Anzeige."""
    def __init__(self):
        self._hand_occlusion = False
        self._HAND_TIMEOUT = 5.0
        self._MIN_FACE_STREAK = 3
        self._FACE_RECENCY = 2.0
        self._state = {}

    def get_state(self):
        return self._state


class ServiceProxy:
    """Liest Frames+Status vom laufenden MolochService via /dev/shm.

    Bietet das gleiche Attribut-Interface wie MolochService,
    damit das Panel ohne Code-Aenderungen funktioniert.
    """

    SHM_FRAME = '/dev/shm/moloch_frame'
    SHM_STATUS = '/dev/shm/moloch_status.json'
    CMD_FILE = '/tmp/moloch_cmd.json'
    CMD_TMP = '/tmp/moloch_cmd.tmp'

    def __init__(self):
        # Frame access (same interface as MolochService)
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Model states
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.pose_active = False

        # Thresholds (read from status)
        self.scrfd_conf_val = 0.5
        self.scrfd_nms_val = 0.4
        self.arcface_thresh_val = 0.6
        self.yolo_conf_val = 0.5
        self.pose_conf_val = 0.5
        self.pose_nms_val = 0.5

        # FPS
        self._fps = {}
        self._fps_lock = threading.Lock()

        # NPU state
        self._npu_paused = False
        self._active_ctx = {}

        # Perception proxy
        self._perception = _PerceptionProxy()

        # Autonomy
        self._autonomous_mode = False
        self._moloch_has_control = False
        self._tentakel_enabled = False

        # Cloud bridge (eigene Instanz fuer LED/IR/Alarm)
        self._cloud = None

        # Not available in remote mode
        self._tracker = None
        self._output_names = {}

        # Observers
        self._observers = []

        # Reader
        self._running = True
        self._last_seq = 0
        self._remote_mode = True

    def init(self):
        """No-op - Service laeuft bereits."""
        pass

    def start(self, blocking=False):
        """Start reader thread + cloud bridge."""
        threading.Thread(target=self._read_loop, daemon=True,
                         name="ShmReader").start()
        # Cloud bridge fuer LED/IR/Alarm (eigene Instanz)
        threading.Thread(target=self._init_cloud, daemon=True,
                         name="CloudInit").start()

    def stop(self):
        """Reader stoppen. Service NICHT beenden!"""
        self._running = False
        if self._cloud:
            try:
                if hasattr(self._cloud, 'close'):
                    self._cloud.close()
            except Exception:
                pass

    def add_observer(self, callback):
        self._observers.append(callback)

    def toggle_model(self, model_key, enabled):
        """Model toggle via IPC an Service senden."""
        self._send_cmd({
            "action": "toggle_model",
            "model": model_key,
            "enabled": enabled
        })

    def _toggle_smart_tracking(self):
        """Smart Tracking toggle via IPC."""
        self._send_cmd({"action": "toggle_smart_tracking"})

    def toggle_autonomous_manual(self):
        """Autonomie toggle via IPC."""
        self._send_cmd({"action": "toggle_autonomous"})

    def _run_model(self, name, input_data):
        """NPU nicht verfuegbar im Remote-Modus."""
        return None

    def _reload_face_db(self):
        """Face DB reload via IPC."""
        self._send_cmd({"action": "reload_face_db"})

    def _send_cmd(self, cmd):
        """Kommando fuer Service schreiben."""
        try:
            with open(self.CMD_TMP, 'w') as f:
                json.dump(cmd, f)
            os.rename(self.CMD_TMP, self.CMD_FILE)
        except Exception as e:
            logger.error(f"IPC cmd failed: {e}")

    def _init_cloud(self):
        """Eigene Cloud-Bridge Instanz fuer eWeLink Controls."""
        try:
            from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
            import asyncio as _aio

            config = CloudConfig(
                enabled=True,
                api_base_url="https://eu-apia.coolkit.cc",
                app_id=os.environ.get("EWELINK_APP_ID_1", ""),
                app_secret=os.environ.get("EWELINK_APP_SECRET_1", ""),
                device_id="1002817609",
                username=os.environ.get("EWELINK_USERNAME", ""),
                password=os.environ.get("EWELINK_PASSWORD", ""),
            )
            bridge = CameraCloudBridge(config)

            class _CloudCtrl:
                def __init__(self, br):
                    self.bridge = br
                    self.loop = None
                    self.connected = False

                def start(self):
                    self.loop = _aio.new_event_loop()
                    _aio.set_event_loop(self.loop)
                    def run_loop():
                        self.loop.run_forever()
                    threading.Thread(target=run_loop, daemon=True).start()
                    time.sleep(0.2)
                    future = _aio.run_coroutine_threadsafe(
                        self.bridge.connect(), self.loop)
                    try:
                        self.connected = future.result(timeout=10)
                    except Exception:
                        self.connected = False

                def run(self, coro):
                    if not self.loop:
                        return False
                    future = _aio.run_coroutine_threadsafe(coro, self.loop)
                    try:
                        return future.result(timeout=5)
                    except Exception:
                        return False

                def close(self):
                    if self.loop:
                        self.loop.call_soon_threadsafe(self.loop.stop)

            ctrl = _CloudCtrl(bridge)
            ctrl.start()
            self._cloud = ctrl
            if ctrl.connected:
                logger.info("[PROXY] Cloud bridge connected")
            else:
                logger.warning("[PROXY] Cloud bridge connection failed")
        except Exception as e:
            logger.warning(f"[PROXY] Cloud init failed: {e}")
            self._cloud = None

    def _read_loop(self):
        """Frames + Status von /dev/shm lesen."""
        while self._running:
            try:
                # Frame lesen
                if os.path.exists(self.SHM_FRAME):
                    with open(self.SHM_FRAME, 'rb') as f:
                        header = f.read(16)
                        if len(header) == 16:
                            h, w, c, seq = struct.unpack('<IIII', header)
                            if seq != self._last_seq and h > 0 and w > 0:
                                data = f.read(h * w * c)
                                if len(data) == h * w * c:
                                    frame = np.frombuffer(
                                        data, dtype=np.uint8
                                    ).reshape(h, w, c).copy()
                                    with self._annotated_lock:
                                        self._annotated_frame = frame
                                    with self._frame_lock:
                                        self._latest_frame = frame
                                    self._last_seq = seq

                # Status lesen
                if os.path.exists(self.SHM_STATUS):
                    with open(self.SHM_STATUS, 'r') as f:
                        status = json.load(f)
                    self._apply_status(status)

            except Exception:
                pass

            time.sleep(0.033)  # ~30fps

    def _apply_status(self, s):
        """Proxy-Attribute aus Status-JSON aktualisieren."""
        self.scrfd_active = s.get('scrfd_active', False)
        self.arcface_active = s.get('arcface_active', False)
        self.yolo_active = s.get('yolo_active', False)
        self.pose_active = s.get('pose_active', False)
        self._npu_paused = s.get('npu_paused', False)
        self._autonomous_mode = s.get('autonomous_mode', False)
        self._moloch_has_control = s.get('moloch_has_control', False)
        self._tentakel_enabled = s.get('tentakel_enabled', False)

        self._active_ctx = {m: True for m in s.get('active_models', [])}

        with self._fps_lock:
            self._fps = s.get('fps', {})

        thresholds = s.get('thresholds', {})
        if thresholds:
            self.scrfd_conf_val = thresholds.get('scrfd_conf', self.scrfd_conf_val)
            self.scrfd_nms_val = thresholds.get('scrfd_nms', self.scrfd_nms_val)
            self.arcface_thresh_val = thresholds.get('arcface_thresh', self.arcface_thresh_val)
            self.yolo_conf_val = thresholds.get('yolo_conf', self.yolo_conf_val)
            self.pose_conf_val = thresholds.get('pose_conf', self.pose_conf_val)
            self.pose_nms_val = thresholds.get('pose_nms', self.pose_nms_val)

        pe = s.get('perception', {})
        if pe and self._perception:
            self._perception._state = pe
            self._perception._hand_occlusion = pe.get('hand_occlusion', False)


class MolochUnifiedPanel:'''

if old_class_def in panel_code:
    panel_code = panel_code.replace(old_class_def, service_proxy_class, 1)
    panel_changes += 1
    print("PANEL 2b: ServiceProxy Klasse eingefuegt")
else:
    print("ERROR: class MolochUnifiedPanel nicht gefunden")
    sys.exit(1)

# 2c: Modify _init_service() to check for running systemd service
old_init_service = """    def _init_service(self):
        \"\"\"Initialize MolochService in background thread.\"\"\"
        def do_init():
            try:
                self.root.after(0, lambda: self.status_label.config(
                    text="Service wird gestartet..."))

                from core.moloch_service import MolochService
                self.service = MolochService()
                self.service.add_observer(self._on_service_event)
                self.service.init()
                self.service.start(blocking=False)

                self.root.after(0, self._on_service_ready)
            except Exception as e:
                logger.error(f"Service init failed: {e}\\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Service FEHLER: {e}", fg="#ff4444"))

        threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()"""

new_init_service = """    def _init_service(self):
        \"\"\"Initialize service connection.

        Prueft ob systemd moloch.service laeuft:
        - JA: ServiceProxy (liest von /dev/shm, sendet Kommandos via IPC)
        - NEIN: Eigener MolochService (Standalone-Modus)
        \"\"\"
        self._remote_mode = False

        def do_init():
            try:
                # Pruefen ob systemd Service laeuft
                service_running = False
                try:
                    result = subprocess.run(
                        ["systemctl", "is-active", "moloch.service"],
                        capture_output=True, text=True, timeout=3)
                    service_running = result.stdout.strip() == "active"
                except Exception:
                    pass

                if service_running:
                    # REMOTE MODE: Verbinde zu laufendem Service
                    self._remote_mode = True
                    self.root.after(0, lambda: self.status_label.config(
                        text="Verbinde zu Service..."))
                    logger.info("[PANEL] systemd Service erkannt -> Remote-Modus")

                    self.service = ServiceProxy()
                    self.service.add_observer(self._on_service_event)
                    self.service.init()
                    self.service.start(blocking=False)

                    # Kurz warten fuer ersten Frame
                    time.sleep(0.5)
                    self.root.after(0, self._on_service_ready)
                    self.root.after(0, lambda: self.status_label.config(
                        text="Remote: systemd Service", fg="#00ccff"))
                else:
                    # STANDALONE MODE: Eigenen Service erstellen
                    self.root.after(0, lambda: self.status_label.config(
                        text="Service wird gestartet..."))
                    logger.info("[PANEL] Kein systemd Service -> Standalone-Modus")

                    from core.moloch_service import MolochService
                    self.service = MolochService()
                    self.service.add_observer(self._on_service_event)
                    self.service.init()
                    self.service.start(blocking=False)

                    self.root.after(0, self._on_service_ready)

            except Exception as e:
                logger.error(f"Service init failed: {e}\\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Service FEHLER: {e}", fg="#ff4444"))

        threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()"""

if old_init_service in panel_code:
    panel_code = panel_code.replace(old_init_service, new_init_service, 1)
    panel_changes += 1
    print("PANEL 2c: _init_service() mit Remote-Detection")
else:
    print("ERROR: _init_service() Block nicht gefunden")
    sys.exit(1)

# 2d: Modify _on_close() to not stop systemd service
old_on_close = """        # Stop service in background
        if self.service:
            threading.Thread(target=self.service.stop, daemon=True).start()"""

new_on_close = """        # Service beenden (Proxy: nur Reader stoppen, NICHT systemd Service!)
        if self.service:
            threading.Thread(target=self.service.stop, daemon=True).start()"""

if old_on_close in panel_code:
    panel_code = panel_code.replace(old_on_close, new_on_close, 1)
    panel_changes += 1
    print("PANEL 2d: _on_close() Kommentar aktualisiert")
else:
    print("WARN: _on_close() service stop nicht gefunden (unkritisch)")

# Write panel file
with open(panel_path, "w") as f:
    f.write(panel_code)
print(f"PANEL: {panel_changes} patches angewendet")

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Panel-Service Bridge:")
print(f"  Service (moloch_service.py):")
print(f"    - _write_shm(): Frame + Status nach /dev/shm")
print(f"    - _poll_panel_cmds(): Kommandos von Panel lesen")
print(f"    - PanelCmdPoll Thread gestartet in start()")
print(f"    - shm cleanup in stop()")
print(f"  Panel (moloch_unified_panel.py):")
print(f"    - ServiceProxy: Liest von /dev/shm, selbes Interface")
print(f"    - _init_service(): Prueft systemctl is-active moloch.service")
print(f"    - Remote-Mode: Kein eigener MolochService, kein NPU-Zugriff")
print(f"    - Standalone-Mode: Wie bisher (kein systemd Service)")
print(f"\nIPC-Kanaele:")
print(f"  /dev/shm/moloch_frame        - BGR Frame (16B Header + Raw)")
print(f"  /dev/shm/moloch_status.json  - JSON Status (atomar via rename)")
print(f"  /tmp/moloch_cmd.json         - Panel->Service Kommandos")
print(f"\nTest:")
print(f"  1. sudo systemctl restart moloch")
print(f"  2. ls -la /dev/shm/moloch_*  (Dateien muessen erscheinen)")
print(f"  3. Panel starten -> 'Remote: systemd Service' in Statusbar")
print(f"  4. Video muss sichtbar sein")
print(f"  5. Model-Checkboxen senden IPC-Kommandos")
