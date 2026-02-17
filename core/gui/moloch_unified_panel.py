#!/usr/bin/env python3
"""
M.O.L.O.C.H. Unified Panel
============================

EIN Fenster fuer alles:
- Hailo NPU Modelle + Detektionen (ex hailo_control_panel)
- PTZ + Smart Tracking + eWeLink (ex eye_control_panel)
- Push-to-Talk + Whisper + Claude Chat (ex push_to_talk)

Verbindet sich zum MolochService ueber Observer-Pattern.
Keine eigene RTSP-Verbindung, keine eigene NPU-Logik.

Author: M.O.L.O.C.H. System
Date: 2026-02-14
"""

import os
import sys
import time
import json
import signal
import struct
import math
import logging
import threading
import subprocess
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# Moloch path
sys.path.insert(0, os.path.expanduser("~/moloch"))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("UnifiedPanel")

# Timing debug logger - separate file
_timing_logger = logging.getLogger("VoiceTiming")
_timing_logger.setLevel(logging.DEBUG)
_timing_handler = logging.FileHandler(
    os.path.expanduser("~/moloch/logs/timing_debug.log"), mode="a")
_timing_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S"))
_timing_logger.addHandler(_timing_handler)
_timing_logger.propagate = False
logger.setLevel(logging.INFO)

# Auto-source ~/.profile for env vars (desktop launch workaround)
if not os.environ.get("EWELINK_APP_ID_1"):
    try:
        result = subprocess.run(
            ["bash", "-c", "source ~/.profile && env"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                if key and not key.startswith((" ", "\t")):
                    os.environ.setdefault(key, value)
    except Exception:
        pass

# IPC Constants
NPU_VOICE_REQUEST = "/tmp/moloch_npu_voice_request"
NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"
FACE_STATE_PATH = "/tmp/moloch_face_state.json"
RESPEAKER_NODE = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"




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
        self.hand_active = False

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
        """Kommando fuer Service schreiben (nummeriert gegen Race Condition)."""
        try:
            import time as _t
            seq = int(_t.monotonic_ns())
            tmp = f'/tmp/moloch_cmd_{seq}.tmp'
            dst = f'/tmp/moloch_cmd_{seq}.json'
            with open(tmp, 'w') as f:
                json.dump(cmd, f)
            os.rename(tmp, dst)
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
                            if seq != self._last_seq and h > 0 and w > 0 or (seq < self._last_seq):
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
        # Vorherigen State merken
        _prev = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
            "hand_landmark": getattr(self, 'hand_active', False),
        }
        self.scrfd_active = s.get('scrfd_active', False)
        self.arcface_active = s.get('arcface_active', False)
        self.yolo_active = s.get('yolo_active', False)
        self.pose_active = s.get('pose_active', False)
        self.hand_active = s.get('hand_active', False)
        # Bei Aenderung -> Checkboxen synchronisieren
        _curr = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
            "hand_landmark": self.hand_active,
        }
        if _curr != _prev:
            for cb in self._observers:
                try:
                    cb("model_toggle", _curr)
                except Exception:
                    pass
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


class MolochUnifiedPanel:
    """M.O.L.O.C.H. Unified Control Panel - alles in einem Fenster."""

    DISPLAY_FPS = 15
    PREVIEW_W = 640
    PREVIEW_H = 360

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H.")
        self.root.configure(bg="#0a0a14")
        self.root.resizable(True, True)
        self.running = True

        # Service reference (set in Phase 2)
        self.service = None
        self._syncing = False
        self._syncing_thresholds = False

        # Camera reference (set in Phase 5)
        self._camera = None

        # Voice state (Phase 6)
        self.whisper = None
        self.hailo_manager = None
        self.is_recording = False
        self._voice_processing = False
        self.record_process = None
        self.temp_audio_path = None

        # Chat state (Phase 7)
        self.claude_client = None
        self.system_prompt = None
        self.memory = None
        self.tts = None
        self.conversation_history = []

        # Voice selection state
        self._voice_var = tk.StringVar()
        self._voice_names = {}  # display_name -> model_stem

        # Audio state
        self._respeaker_source_id = None  # PipeWire Node ID (found dynamically)
        self._mic_gain_var = tk.DoubleVar(value=1.0)
        self._agc_var = tk.BooleanVar(value=False)
        self._noise_gate_var = tk.DoubleVar(value=-60.0)
        self._vu_canvas = None
        self._vu_db_label = None
        self._vu_monitor_running = False
        self._vu_process = None
        self._user_forced_voice = False  # True wenn Markus manuell waehlt
        self._moloch_preferred = ["de_DE-thorsten-high", "de_DE-thorsten-low",
                                  "de_DE-thorsten-medium"]
        self._utterance_count_since_force = 0

        # Display
        self._photo = None
        self._display_after_id = None
        self._canvas_image_id = None

        # --- Style Setup ---
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0a0a14")
        self.style.configure("TLabel", background="#0a0a14", foreground="#ffffff",
                             font=("Helvetica", 12))
        self.style.configure("Header.TLabel", background="#0a0a14", foreground="#00d4ff",
                             font=("Helvetica", 14, "bold"))
        self.style.configure("Status.TLabel", background="#0a0a14", foreground="#00ff88",
                             font=("Helvetica", 12))
        self.style.configure("FPS.TLabel", background="#0a0a14", foreground="#ffaa00",
                             font=("Helvetica", 12, "bold"))
        self.style.configure("TScale", background="#0a0a14", troughcolor="#1a1a3e")
        self.style.configure("TCheckbutton", background="#0a0a14", foreground="#ffffff",
                             font=("Helvetica", 12))
        self.style.configure("TLabelframe", background="#0a0a14", foreground="#00d4ff",
                             font=("Helvetica", 12, "bold"))
        self.style.configure("TLabelframe.Label", background="#0a0a14", foreground="#00d4ff",
                             font=("Helvetica", 12, "bold"))
        self.style.configure("TCombobox", font=("Helvetica", 11))
        self.style.configure("TScrollbar", background="#1a1a3e", troughcolor="#0a0a14")

        # --- Build Layout ---
        self._build_layout()

        # --- Start Service Connection ---
        self.root.after(100, self._init_service)

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # =========================================================================
    # Phase 1: Layout
    # =========================================================================

    def _build_layout(self):
        """Build complete UI layout - all zones, no functionality yet."""
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar (top, full width) ---
        self._build_status_bar(main)

        # --- Content Area (camera+ptz LEFT, models RIGHT) ---
        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # LEFT: Camera preview + PTZ/eWeLink
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self._build_preview(left)
        self._build_ptz_ewelink(left)

        # RIGHT: Model controls
        self._build_model_controls(content)

        # --- Talk + Chat (bottom, full width) ---
        self._build_talk_chat(main)

    def _build_status_bar(self, parent):
        """Status bar with mode, ST, FPS, NPU, PTZ position."""
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=(0, 6))

        self.mode_label = tk.Label(bar, text="OFFLINE", bg="#0a0a14", fg="#888888",
                                   font=("Helvetica", 14, "bold"))
        self.mode_label.pack(side=tk.LEFT, padx=(0, 10))

        self.st_label = tk.Label(bar, text="ST: --", bg="#0a0a14", fg="#888888",
                                  font=("Helvetica", 12))
        self.st_label.pack(side=tk.LEFT, padx=(0, 10))

        self.fps_label = tk.Label(bar, text="FPS: --", bg="#0a0a14", fg="#ffaa00",
                                   font=("Helvetica", 12, "bold"))
        self.fps_label.pack(side=tk.LEFT, padx=(0, 10))

        self.npu_label = tk.Label(bar, text="NPU: --", bg="#0a0a14", fg="#888888",
                                   font=("Helvetica", 12))
        self.npu_label.pack(side=tk.LEFT, padx=(0, 10))

        self.ptz_label = tk.Label(bar, text="PTZ: --", bg="#0a0a14", fg="#888888",
                                   font=("Helvetica", 12))
        self.ptz_label.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = tk.Label(bar, text="Initialisierung...", bg="#0a0a14",
                                      fg="#00ff88", font=("Helvetica", 12))
        self.status_label.pack(side=tk.RIGHT)

    def _build_preview(self, parent):
        """Camera preview canvas."""
        self.preview_canvas = tk.Canvas(
            parent, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#000000", highlightthickness=2, highlightbackground="#222244"
        )
        self.preview_canvas.pack(pady=(0, 3))

    def _build_ptz_ewelink(self, parent):
        """PTZ controls + eWeLink toggles under camera."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X)

        # --- PTZ D-Pad (left) ---
        ptz = ttk.Frame(frame)
        ptz.pack(side=tk.LEFT, padx=(0, 10))

        btn_w = 5
        btn_cfg = dict(width=btn_w, height=1, bg="#1a1a3e", fg="white",
                       font=("Helvetica", 12, "bold"))
        tk.Button(ptz, text="^", command=lambda: self._ptz_move("up"),
                  **btn_cfg).grid(row=0, column=1, padx=1, pady=1)
        tk.Button(ptz, text="<", command=lambda: self._ptz_move("left"),
                  **btn_cfg).grid(row=1, column=0, padx=1, pady=1)
        tk.Button(ptz, text="H", command=lambda: self._ptz_move("home"),
                  width=btn_w, height=1, bg="#003355", fg="#00d4ff",
                  font=("Helvetica", 12, "bold")).grid(row=1, column=1, padx=1, pady=1)
        tk.Button(ptz, text=">", command=lambda: self._ptz_move("right"),
                  **btn_cfg).grid(row=1, column=2, padx=1, pady=1)
        tk.Button(ptz, text="v", command=lambda: self._ptz_move("down"),
                  **btn_cfg).grid(row=2, column=1, padx=1, pady=1)

        # --- PTZ Extras (middle) ---
        extras = ttk.Frame(frame)
        extras.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Quick positions
        qrow = ttk.Frame(extras)
        qrow.pack(fill=tk.X, pady=1)
        for name, pan, tilt in [("L", 170, 0), ("M", 0, 0), ("R", -168, 0)]:
            tk.Button(qrow, text=name, bg="#1a1a3e", fg="white", width=4,
                      font=("Helvetica", 11),
                      command=lambda p=pan, t=tilt: self._ptz_goto(p, t)).pack(
                side=tk.LEFT, padx=1)

        # Speed slider
        srow = ttk.Frame(extras)
        srow.pack(fill=tk.X, pady=1)
        ttk.Label(srow, text="Spd:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=15.0)
        self.speed_lbl = ttk.Label(srow, text="15", width=3, font=("Helvetica", 11))
        self.speed_lbl.pack(side=tk.RIGHT)
        ttk.Scale(srow, from_=1, to=50, variable=self.speed_var,
                  command=lambda v: self.speed_lbl.configure(
                      text=f"{float(v):.0f}")).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ST + Auto + Kalibr buttons
        brow = ttk.Frame(extras)
        brow.pack(fill=tk.X, pady=1)
        self.st_btn = tk.Button(brow, text="ST", bg="#1a1a3e", fg="white", width=5,
                                font=("Helvetica", 11, "bold"),
                                command=self._toggle_smart_tracking)
        self.st_btn.pack(side=tk.LEFT, padx=1)
        self.auto_btn = tk.Button(brow, text="MANUELL", bg="#1a1a3e", fg="white", width=9,
                                  font=("Helvetica", 11, "bold"),
                                  command=self._toggle_autonomous)
        self.auto_btn.pack(side=tk.LEFT, padx=1)
        tk.Button(brow, text="CAL", bg="#ff8800", fg="white", width=5,
                  font=("Helvetica", 11, "bold"),
                  command=self._trigger_calibration).pack(side=tk.LEFT, padx=1)
        self.daily_btn = tk.Button(brow, text="ALLTAG", bg="#1a1a3e", fg="white", width=7,
                                   font=("Helvetica", 11, "bold"),
                                   command=self._toggle_daily_learner)
        self.daily_btn.pack(side=tk.LEFT, padx=1)

        # --- eWeLink controls (right) ---
        ew = ttk.Frame(frame)
        ew.pack(side=tk.LEFT, padx=(10, 0))

        # LED checkbox
        led_row = ttk.Frame(ew)
        led_row.pack(fill=tk.X)
        ttk.Label(led_row, text="LED", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.led_var = tk.BooleanVar(value=False)
        tk.Checkbutton(led_row, variable=self.led_var, bg="#0a0a14",
                       selectcolor="#1a1a3e", activebackground="#0a0a14",
                       command=self._set_status_led).pack(side=tk.RIGHT)

        # IR Combobox
        ir_row = ttk.Frame(ew)
        ir_row.pack(fill=tk.X)
        ttk.Label(ir_row, text="Licht", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.ir_var = tk.StringVar(value="Aus")
        ir_combo = ttk.Combobox(ir_row, textvariable=self.ir_var,
                                values=["Aus", "Auto", "An"],
                                state="readonly", width=6, font=("Helvetica", 11))
        ir_combo.pack(side=tk.RIGHT)
        ir_combo.bind("<<ComboboxSelected>>", lambda e: self._set_night())

        # Alarm + Refresh
        ar_row = ttk.Frame(ew)
        ar_row.pack(fill=tk.X, pady=(2, 0))
        tk.Button(ar_row, text="ALARM", bg="#ff4444", fg="white", width=7,
                  font=("Helvetica", 11, "bold"),
                  command=self._trigger_alarm).pack(side=tk.LEFT, padx=1)
        tk.Button(ar_row, text="SYNC", bg="#1a1a3e", fg="white", width=6,
                  font=("Helvetica", 11),
                  command=self._refresh_cloud_params).pack(side=tk.LEFT, padx=1)
        tk.Button(ar_row, text="SNAP", bg="#00aa44", fg="white", width=6,
                  font=("Helvetica", 11, "bold"),
                  command=self._take_snapshot).pack(side=tk.LEFT, padx=1)

    def _build_model_controls(self, parent):
        """Model checkboxes + threshold sliders on the right side."""
        model_frame = ttk.Frame(parent)
        model_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        ttk.Label(model_frame, text="MODELLE", style="Header.TLabel").pack(anchor=tk.W)

        # Model variables
        self.scrfd_var = tk.BooleanVar(value=False)
        self.arcface_var = tk.BooleanVar(value=False)
        self.yolo_var = tk.BooleanVar(value=False)
        self.pose_var = tk.BooleanVar(value=False)

        # Threshold variables
        self.scrfd_conf_var = tk.DoubleVar(value=0.40)
        self.scrfd_nms_var = tk.DoubleVar(value=0.40)
        self.arcface_thresh_var = tk.DoubleVar(value=0.60)
        self.yolo_conf_var = tk.DoubleVar(value=0.50)
        self.pose_conf_var = tk.DoubleVar(value=0.50)
        self.pose_nms_var = tk.DoubleVar(value=0.70)

        # SCRFD
        self._scrfd_fps = self._build_model_section(
            model_frame, "SCRFD Face", self.scrfd_var, "scrfd",
            [("Conf", self.scrfd_conf_var, 0.1, 0.9),
             ("NMS", self.scrfd_nms_var, 0.1, 0.9)])

        # ArcFace
        self._arcface_fps = self._build_model_section(
            model_frame, "ArcFace", self.arcface_var, "arcface",
            [("Thresh", self.arcface_thresh_var, 0.3, 0.9)])

        # YOLOv8m
        self._yolov8m_fps = self._build_model_section(
            model_frame, "YOLOv8m", self.yolo_var, "yolov8m",
            [("Conf", self.yolo_conf_var, 0.1, 0.9)])

        # Pose
        self._pose_fps = self._build_model_section(
            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

        # Hand Landmark (braucht Pose als Dependency)
        self.hand_lm_var = tk.BooleanVar(value=False)
        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Save Settings ---
        save_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        save_sep.pack(fill=tk.X, pady=(8, 4))
        self._save_btn = tk.Button(model_frame, text="SAVE SETTINGS",
                                   bg="#00aa44", fg="white",
                                   font=("Helvetica", 10, "bold"),
                                   command=self._save_settings)
        self._save_btn.pack(fill=tk.X, pady=(2, 0))

        # --- Hand-Occlusion (Auto-Erkennung) ---
        hand_sep = ttk.Separator(model_frame, orient=tk.HORIZONTAL)
        hand_sep.pack(fill=tk.X, pady=(8, 4))

        hand_header = ttk.Frame(model_frame)
        hand_header.pack(fill=tk.X)

        self.hand_var = tk.BooleanVar(value=True)
        hand_cb = tk.Checkbutton(hand_header, text="Auto-Occlusion",
                                 variable=self.hand_var,
                                 bg="#0a0a14", fg="#e0e0e0", selectcolor="#2a2a4e",
                                 activebackground="#1a1a2e", font=("Helvetica", 9),
                                 command=self._on_hand_toggle)
        hand_cb.pack(side=tk.LEFT)

        self.hand_status_label = tk.Label(hand_header, text="", bg="#0a0a14",
                                          fg="#ff4444", font=("Helvetica", 9, "bold"))
        self.hand_status_label.pack(side=tk.RIGHT)

        # Timeout Slider
        hand_timeout_row = ttk.Frame(model_frame)
        hand_timeout_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_timeout_row, text="Timeout:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_timeout_var = tk.DoubleVar(value=5.0)
        self.hand_timeout_lbl = ttk.Label(hand_timeout_row, text="5.0s", width=4,
                                           font=("Helvetica", 11))
        self.hand_timeout_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_timeout_row, from_=1.0, to=10.0, variable=self.hand_timeout_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # Streak Slider
        hand_streak_row = ttk.Frame(model_frame)
        hand_streak_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_streak_row, text="Streak:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_streak_var = tk.DoubleVar(value=3.0)
        self.hand_streak_lbl = ttk.Label(hand_streak_row, text="3", width=4,
                                          font=("Helvetica", 11))
        self.hand_streak_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_streak_row, from_=1.0, to=10.0, variable=self.hand_streak_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        # Recency Slider
        hand_recency_row = ttk.Frame(model_frame)
        hand_recency_row.pack(fill=tk.X, padx=(15, 0))
        ttk.Label(hand_recency_row, text="Recency:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.hand_recency_var = tk.DoubleVar(value=2.0)
        self.hand_recency_lbl = ttk.Label(hand_recency_row, text="2.0s", width=4,
                                           font=("Helvetica", 11))
        self.hand_recency_lbl.pack(side=tk.RIGHT)
        ttk.Scale(hand_recency_row, from_=0.5, to=5.0, variable=self.hand_recency_var,
                  command=self._on_hand_param_change).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=3)

    def _build_model_section(self, parent, title, enabled_var, model_key, sliders):
        """Build one model section: checkbox + FPS + sliders. Returns FPS label."""
        section = ttk.Frame(parent)
        section.pack(fill=tk.X, pady=(5, 0))

        # Header row: checkbox + FPS
        header = ttk.Frame(section)
        header.pack(fill=tk.X)

        cb = tk.Checkbutton(header, text=title, variable=enabled_var,
                            bg="#0a0a14", fg="#e0e0e0", selectcolor="#2a2a4e",
                            activebackground="#1a1a2e", font=("Helvetica", 9),
                            command=lambda: self._on_model_toggle(model_key))
        cb.pack(side=tk.LEFT)

        fps_label = ttk.Label(header, text="---", style="FPS.TLabel")
        fps_label.pack(side=tk.RIGHT)

        # Sliders
        for label_text, var, from_val, to_val in sliders:
            row = ttk.Frame(section)
            row.pack(fill=tk.X, padx=(15, 0))
            ttk.Label(row, text=f"{label_text}:", font=("Helvetica", 11)).pack(side=tk.LEFT)
            val_lbl = ttk.Label(row, text=f"{var.get():.2f}", width=4,
                                font=("Helvetica", 11))
            val_lbl.pack(side=tk.RIGHT)
            ttk.Scale(row, from_=from_val, to=to_val, variable=var,
                      command=lambda v, lbl=val_lbl: lbl.configure(
                          text=f"{float(v):.2f}")).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        return fps_label

    def _build_talk_chat(self, parent):
        """Talk button (left) + Chat history (right) at the bottom."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # --- PTT Section (left) ---
        ptt = ttk.Frame(frame)
        ptt.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        self.talk_button = tk.Button(
            ptt, text="SPRECHEN", bg="#12182e", fg="white",
            font=("Helvetica", 16, "bold"), width=10, height=3,
            activebackground="#e94560",
            command=self._toggle_recording
        )
        self.talk_button.pack(pady=(0, 3))

        self.ptt_status = tk.Label(ptt, text="Bereit", bg="#0a0a14", fg="#00ff88",
                                    font=("Helvetica", 12))
        self.ptt_status.pack()

        # --- Voice Selection ---
        voice_frame = ttk.Frame(ptt)
        voice_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(voice_frame, text="Stimme:", font=("Helvetica", 11)).pack(
            anchor=tk.W)

        self._voice_combo = ttk.Combobox(voice_frame, textvariable=self._voice_var,
                                          state="readonly", width=16,
                                          font=("Helvetica", 11))
        self._voice_combo.pack(fill=tk.X, pady=(1, 0))
        self._voice_combo.bind("<<ComboboxSelected>>", self._on_voice_changed)

        tk.Button(voice_frame, text="Test", bg="#1a1a3e", fg="white",
                  font=("Helvetica", 11), width=7,
                  command=self._test_voice).pack(pady=(2, 0))

        # --- Audio Controls ---
        self._build_audio_controls(ptt)

        # --- Chat Section (right) ---
        chat = ttk.Frame(frame)
        chat.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Chat history (scrollable text widget)
        text_frame = ttk.Frame(chat)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text = tk.Text(
            text_frame, font=("DejaVu Sans", 12), fg="#ffffff", bg="#060610",
            wrap="word", height=6, state="disabled",
            yscrollcommand=scrollbar.set
        )
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.chat_text.yview)

        # Color tags
        self.chat_text.tag_configure("user", foreground="#ff6b6b")
        self.chat_text.tag_configure("moloch", foreground="#00ff88")
        self.chat_text.tag_configure("system", foreground="#ffaa00")

        # Text input row
        input_row = ttk.Frame(chat)
        input_row.pack(fill=tk.X, pady=(4, 0))

        self.text_input = tk.Entry(input_row, bg="#12182e", fg="#ffffff",
                                    insertbackground="#ffffff",
                                    font=("Helvetica", 12))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.text_input.bind("<Return>", self._send_text_message)

        tk.Button(input_row, text="SENDEN", bg="#1a1a3e", fg="white",
                  font=("Helvetica", 12, "bold"), width=8,
                  command=self._send_text_message).pack(side=tk.RIGHT)

    # =========================================================================
    # Phase 2: Service Connection
    # =========================================================================

    def _init_service(self):
        """Initialize service connection.

        Prueft ob systemd moloch.service laeuft:
        - JA: ServiceProxy (liest von /dev/shm, sendet Kommandos via IPC)
        - NEIN: Eigener MolochService (Standalone-Modus)
        """
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
                logger.error(f"Service init failed: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Service FEHLER: {e}", fg="#ff4444"))

        threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()

    def _on_service_ready(self):
        """Called on main thread when service is ready."""
        self.status_label.config(text="Service bereit", fg="#00ff88")

        # Seed checkbox states from service
        self._syncing = True
        self.scrfd_var.set(self.service.scrfd_active)
        self.arcface_var.set(self.service.arcface_active)
        self.yolo_var.set(self.service.yolo_active)
        self.pose_var.set(self.service.pose_active)
        self.hand_lm_var.set(getattr(self.service, 'hand_active', False))
        self._syncing = False

        # Seed threshold values
        self.scrfd_conf_var.set(self.service.scrfd_conf_val)
        self.scrfd_nms_var.set(self.service.scrfd_nms_val)
        self.arcface_thresh_var.set(self.service.arcface_thresh_val)
        self.yolo_conf_var.set(self.service.yolo_conf_val)
        self.pose_conf_var.set(self.service.pose_conf_val)
        self.pose_nms_var.set(self.service.pose_nms_val)

        # Setup threshold bindings (GUI -> Service)
        self._setup_threshold_bindings()

        # Start display loop (Phase 3)
        self._display_loop()

        # Start FPS update timer (Phase 4)
        self._update_fps()

        # Start NPU status timer
        self._update_npu_status()

        # Load voice deps in background (Phase 6)
        threading.Thread(target=self._load_voice_deps, daemon=True).start()

        # Start audio VU monitor
        threading.Thread(target=self._start_vu_monitor, daemon=True).start()
        # Find ReSpeaker node ID
        threading.Thread(target=self._find_respeaker_source_id, daemon=True).start()

        # Load chat deps in background (Phase 7)
        threading.Thread(target=self._load_chat_deps, daemon=True).start()

    def _on_service_event(self, event, data):
        """Handle events from MolochService (called from service threads)."""
        if event == "status":
            self.root.after(0, lambda d=data: self.status_label.config(
                text=d.get("text", "")))
        elif event == "model_toggle":
            self.root.after(0, lambda d=data: self._sync_model_toggles(d))
        elif event == "cam_status":
            self.root.after(0, lambda d=data: self._update_cam_status(d))
        elif event == "auto_mode":
            self.root.after(0, lambda d=data: self._update_auto_mode(d))
        elif event == "smart_tracking":
            self.root.after(0, lambda d=data: self._update_st_display(d))
        elif event == "cloud_status":
            pass  # Cloud status handled via cam_status

    def _setup_threshold_bindings(self):
        """Bind slider DoubleVars to service threshold attributes (via IPC bei Proxy)."""
        bindings = [
            (self.scrfd_conf_var, "scrfd_conf_val"),
            (self.scrfd_nms_var, "scrfd_nms_val"),
            (self.arcface_thresh_var, "arcface_thresh_val"),
            (self.yolo_conf_var, "yolo_conf_val"),
            (self.pose_conf_var, "pose_conf_val"),
            (self.pose_nms_var, "pose_nms_val"),
        ]
        for var, attr in bindings:
            def on_change(*_, a=attr, v=var):
                if not self.service or self._syncing_thresholds:
                    return
                val = v.get()
                if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
                    self.service._send_cmd({"action": "set_threshold", "attr": a, "value": val})
                else:
                    setattr(self.service, a, val)
            var.trace_add("write", on_change)

    # =========================================================================
    # Phase 3: Camera Preview
    # =========================================================================

    def _display_loop(self):
        """Display annotated frame from service at ~15 FPS."""
        if not self.running or not self.service:
            return

        frame = None
        # Prefer annotated frame (has detection overlays)
        with self.service._annotated_lock:
            if self.service._annotated_frame is not None:
                frame = self.service._annotated_frame

        # Fallback to raw RTSP frame
        if frame is None:
            with self.service._frame_lock:
                if self.service._latest_frame is not None:
                    frame = self.service._latest_frame

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.PREVIEW_W, self.PREVIEW_H))
                self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                if self._canvas_image_id is None:
                    self._canvas_image_id = self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
                else:
                    self.preview_canvas.itemconfig(self._canvas_image_id, image=self._photo)
            except Exception:
                pass

        self._display_after_id = self.root.after(
            1000 // self.DISPLAY_FPS, self._display_loop)

    # =========================================================================
    # Phase 4: Model Controls
    # =========================================================================

    def _on_model_toggle(self, model_key):
        """Called when user clicks a model checkbox."""
        if self._syncing or not self.service:
            return
        var_map = {
            "scrfd": self.scrfd_var,
            "arcface": self.arcface_var,
            "yolov8m": self.yolo_var,
            "pose": self.pose_var,
            "hand_landmark": self.hand_lm_var,
        }
        var = var_map.get(model_key)
        if var:
            enabled = var.get()
            threading.Thread(
                target=lambda: self.service.toggle_model(model_key, enabled),
                daemon=True).start()

    def _sync_model_toggles(self, data):
        """Sync checkbox states from service model_toggle event."""
        self._syncing = True
        try:
            toggle_map = {
                "scrfd": self.scrfd_var,
                "arcface": self.arcface_var,
                "yolov8m": self.yolo_var,
                "pose": self.pose_var,
                "hand_landmark": self.hand_lm_var,
            }
            for key, val in data.items():
                var = toggle_map.get(key)
                if var is not None:
                    var.set(val)
        finally:
            self._syncing = False

    def _update_fps(self):
        """Update FPS labels every 500ms."""
        if not self.running or not self.service:
            return

        try:
            with self.service._fps_lock:
                fps = self.service._fps.copy()

            total = fps.get("total", 0)
            if total > 0:
                self.fps_label.config(text=f"FPS: {total:.0f}")
            else:
                self.fps_label.config(text="FPS: --")

            for key, label in [("scrfd", self._scrfd_fps),
                                ("arcface", self._arcface_fps),
                                ("yolov8m", self._yolov8m_fps),
                                ("pose", self._pose_fps),
                                ("hand_landmark", self._hand_lm_fps)]:
                v = fps.get(key, 0)
                label.config(text=f"{v:.0f}" if v > 0 else "---")
        except Exception:
            pass

        self.root.after(500, self._update_fps)

    def _update_npu_status(self):
        """Update NPU status in status bar every 1000ms."""
        if not self.running or not self.service:
            return

        try:
            if self.service._npu_paused:
                self.npu_label.config(text="NPU: Voice", fg="#ffaa00")
            elif self.service._active_ctx:
                models = ", ".join(self.service._active_ctx.keys())
                self.npu_label.config(text=f"NPU: {models}", fg="#00ff88")
            else:
                self.npu_label.config(text="NPU: Idle", fg="#888888")
        except Exception:
            pass

        # Hand-Occlusion Status aktualisieren
        try:
            if self.service._perception:
                pe_state = self.service._perception.get_state()
                if pe_state.get("hand_occlusion"):
                    self.hand_status_label.config(text="HAND!", fg="#ff4444")
                else:
                    streak = pe_state.get("face_streak", 0)
                    if streak > 0:
                        self.hand_status_label.config(text=f"S:{streak}", fg="#888888")
                    else:
                        self.hand_status_label.config(text="", fg="#888888")
        except Exception:
            pass

        # Daily Learner Button Update
        if hasattr(self, "daily_btn"):
            try:
                s = {}
                if hasattr(self.service, "_remote_mode") and self.service._remote_mode:
                    import os, json as _j
                    if os.path.exists(self.SHM_STATUS):
                        with open(self.SHM_STATUS, "r") as _f:
                            s = _j.load(_f)
                else:
                    if hasattr(self.service, "_daily_learner") and self.service._daily_learner:
                        s = {"daily_learner_enabled": self.service._daily_learner.enabled}
                dl_enabled = s.get("daily_learner_enabled", False)
                if dl_enabled:
                    self.daily_btn.config(bg="#006622", text="ALLTAG AN")
                else:
                    self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")
            except Exception:
                pass

        self.root.after(1000, self._update_npu_status)

    # =========================================================================
    # Hand-Occlusion Controls
    # =========================================================================

    def _save_settings(self):
        """Alle Settings persistent speichern via IPC."""
        self._send_cmd({
            "action": "save_settings",
            "audio": {
                "mic_gain": self._mic_gain_var.get(),
                "agc_enabled": self._agc_var.get(),
                "noise_gate_db": self._noise_gate_var.get(),
            },
            "camera": {
                "ptz_speed": self.speed_var.get(),
                "led_enabled": self.led_var.get(),
                "ir_mode": self.ir_var.get(),
            },
        })
        # Visuelles Feedback
        self._save_btn.config(text="SAVED!", bg="#006622")
        self.after(2000, lambda: self._save_btn.config(text="SAVE SETTINGS", bg="#00aa44"))

    def _on_hand_toggle(self):
        """Toggle Hand-Occlusion Erkennung (via IPC bei Proxy)."""
        if not self.service:
            return
        enabled = self.hand_var.get()
        if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
            if enabled:
                self.service._send_cmd({
                    "action": "set_hand_params",
                    "timeout": self.hand_timeout_var.get(),
                    "streak": int(self.hand_streak_var.get()),
                    "recency": self.hand_recency_var.get(),
                })
            else:
                self.service._send_cmd({
                    "action": "set_hand_params",
                    "streak": 999999,
                    "disable_occlusion": True,
                })
        elif self.service._perception:
            pe = self.service._perception
            if enabled:
                pe._HAND_TIMEOUT = self.hand_timeout_var.get()
                pe._MIN_FACE_STREAK = int(self.hand_streak_var.get())
                pe._FACE_RECENCY = self.hand_recency_var.get()
            else:
                pe._MIN_FACE_STREAK = 999999
                pe._hand_occlusion = False
        logger.info(f"[PANEL] Hand-Occlusion: {'AN' if enabled else 'AUS'}")

    def _on_hand_param_change(self, *args):
        """Hand-Occlusion Parameter aktualisieren (via IPC bei Proxy)."""
        # Labels aktualisieren
        self.hand_timeout_lbl.config(text=f"{self.hand_timeout_var.get():.1f}s")
        self.hand_streak_lbl.config(text=f"{int(self.hand_streak_var.get())}")
        self.hand_recency_lbl.config(text=f"{self.hand_recency_var.get():.1f}s")
        if not self.service or not self.hand_var.get():
            return
        params = {
            "timeout": self.hand_timeout_var.get(),
            "streak": int(self.hand_streak_var.get()),
            "recency": self.hand_recency_var.get(),
        }
        if hasattr(self.service, '_remote_mode') and self.service._remote_mode:
            self.service._send_cmd({"action": "set_hand_params", **params})
        elif self.service._perception:
            pe = self.service._perception
            pe._HAND_TIMEOUT = params["timeout"]
            pe._MIN_FACE_STREAK = params["streak"]
            pe._FACE_RECENCY = params["recency"]

    # =========================================================================
    # Phase 5: PTZ + eWeLink
    # =========================================================================

    def _get_camera(self):
        """Lazy-load camera controller."""
        if self._camera is None:
            try:
                from core.hardware.camera import get_camera_controller
                self._camera = get_camera_controller()
                if not self._camera.is_connected:
                    self._camera.connect()
            except Exception as e:
                logger.error(f"Camera init failed: {e}")
        return self._camera

    def _ptz_move(self, direction):
        """Move camera in direction (threaded)."""
        def do_move():
            cam = self._get_camera()
            if not cam:
                return
            try:
                step = self.speed_var.get()
                if direction == "home":
                    cam.move_absolute(0.0, 0.0, speed=0.5)
                else:
                    pos = cam.get_position()
                    pan, tilt = pos.pan, pos.tilt
                    if direction == "left":
                        pan += step  # INVERTIERT!
                    elif direction == "right":
                        pan -= step  # INVERTIERT!
                    elif direction == "up":
                        tilt += step
                    elif direction == "down":
                        tilt -= step
                    pan = max(-168.4, min(174.4, pan))
                    tilt = max(-78.8, min(101.3, tilt))
                    cam.move_absolute(pan, tilt, speed=1.0)
            except Exception as e:
                logger.error(f"PTZ move error: {e}")

        threading.Thread(target=do_move, daemon=True).start()

    def _ptz_goto(self, pan, tilt):
        """Go to absolute position (threaded)."""
        def do_goto():
            cam = self._get_camera()
            if cam:
                try:
                    cam.move_absolute(pan, tilt, speed=1.0)
                except Exception as e:
                    logger.error(f"PTZ goto error: {e}")

        threading.Thread(target=do_goto, daemon=True).start()

    def _toggle_smart_tracking(self):
        """Toggle Smart Tracking via service."""
        if self.service:
            threading.Thread(
                target=self.service._toggle_smart_tracking, daemon=True).start()

    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL via service."""
        if self.service:
            self.service.toggle_autonomous_manual()

    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus) via IPC."""
        if not self.service:
            return

        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_daily_learner"})
        else:
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                self.service._daily_learner.toggle()
            else:
                return

    def _set_status_led(self):
        """Set camera status LED via cloud."""
        if self.service and self.service._cloud and self.service._cloud.connected:
            threading.Thread(
                target=lambda: self.service._cloud.run(
                    self.service._cloud.bridge.set_status_led(self.led_var.get())),
                daemon=True).start()

    def _set_night(self):
        """Set night vision mode via cloud."""
        if not self.service or not self.service._cloud:
            return
        mode_map = {"Aus": "day", "Auto": "auto", "An": "night"}
        mode = mode_map.get(self.ir_var.get(), "day")
        threading.Thread(
            target=lambda: self.service._cloud.run(
                self.service._cloud.bridge.set_night(mode)),
            daemon=True).start()

    def _trigger_alarm(self):
        """Trigger alarm for 3 seconds."""
        if not self.service or not self.service._cloud:
            return

        def alarm_cycle():
            self.service._cloud.run(self.service._cloud.bridge.set_alarm(True))
            time.sleep(3)
            self.service._cloud.run(self.service._cloud.bridge.set_alarm(False))

        threading.Thread(target=alarm_cycle, daemon=True).start()

    def _trigger_calibration(self):
        """Trigger PTZ calibration with confirmation."""
        if not self.service or not self.service._cloud:
            return
        if messagebox.askyesno("PTZ Kalibrierung",
                               "Kamera bewegt sich durch den vollen Bereich!\nFortfahren?"):
            threading.Thread(
                target=lambda: self.service._cloud.run(
                    self.service._cloud.bridge.trigger_ptz_calibration()),
                daemon=True).start()

    def _refresh_cloud_params(self):
        """Refresh cloud params and apply to UI."""
        if not self.service or not self.service._cloud:
            return

        def do_refresh():
            params = self.service._cloud.run(
                self.service._cloud.bridge.get_device_params())
            if params:
                self.root.after(0, lambda p=params: self._apply_cloud_params(p))

        threading.Thread(target=do_refresh, daemon=True).start()

    def _apply_cloud_params(self, params):
        """Apply cloud params to UI widgets."""
        try:
            if "nightVision" in params:
                nv_map = {0: "Aus", 1: "Auto", 2: "An"}
                self.ir_var.set(nv_map.get(params["nightVision"], "Aus"))
            if "smartTraceEnable" in params:
                st_on = bool(params["smartTraceEnable"])
                self.st_btn.config(
                    text=f"ST:{'AN' if st_on else 'AUS'}",
                    bg="#884400" if st_on else "#2a2a4e")
            if "sledOnline" in params:
                self.led_var.set(params["sledOnline"] == "on")
        except Exception as e:
            logger.error(f"Apply cloud params error: {e}")

    def _update_cam_status(self, data):
        """Update status bar from cam_status event."""
        mode = data.get("mode", "offline")
        ctrl = data.get("ctrl_text", "")
        smart = data.get("smart", "--")
        ptz = data.get("ptz", "--")

        colors = {"moloch": "#00ff88", "tentakel": "#00d4ff",
                  "manual": "#aaaaaa", "offline": "#ff4444"}
        self.mode_label.config(text=ctrl or mode.upper(),
                               fg=colors.get(mode, "#888888"))

        st_color = "#00ff88" if smart == "AN" else "#ff4444"
        self.st_label.config(text=f"ST: {smart}", fg=st_color)
        self.st_btn.config(text=f"ST:{'AN' if smart == 'AN' else 'AUS'}",
                           bg="#884400" if smart == "AN" else "#2a2a4e")

        self.ptz_label.config(text=ptz)

    def _update_auto_mode(self, data):
        """Update AUTONOM/MANUELL button from auto_mode event."""
        state = data.get("state", "")
        if state == "active":
            self.auto_btn.config(text="AUTONOM", bg="#006622")
        elif state == "disabled" or state == "manual":
            self.auto_btn.config(text="MANUELL", bg="#1a1a3e")
        elif state == "starting":
            self.auto_btn.config(text="START...", bg="#884400")

    def _update_st_display(self, data):
        """Update Smart Tracking display from smart_tracking event."""
        on = data.get("on", False)
        self.st_btn.config(text=f"ST:{'AN' if on else 'AUS'}",
                           bg="#884400" if on else "#2a2a4e")
        self.st_label.config(text=f"ST: {'AN' if on else 'AUS'}",
                             fg="#00ff88" if on else "#ff4444")

    # =========================================================================
    # Phase 6: Push-to-Talk + Whisper
    # =========================================================================

    def _load_voice_deps(self):
        """Load Whisper and HailoManager in background."""
        try:
            from core.speech import get_whisper
            self.whisper = get_whisper()
            logger.info(f"[PTT] Whisper loaded: {self.whisper.backend}")
        except Exception as e:
            logger.error(f"[PTT] Whisper load failed: {e}")
            self.whisper = None

        try:
            from core.hardware.hailo_manager import get_hailo_manager
            self.hailo_manager = get_hailo_manager()
        except Exception:
            self.hailo_manager = None

        ready = self.whisper is not None and self.whisper.is_available
        self.root.after(0, lambda: self.ptt_status.config(
            text="Bereit" if ready else "Whisper nicht verfuegbar",
            fg="#00ff88" if ready else "#ff4444"))

    def _toggle_recording(self):
        """Toggle voice recording on/off."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start audio recording via pw-record."""
        if not self.whisper or not self.whisper.is_available:
            self.ptt_status.config(text="Whisper nicht bereit", fg="#ff4444")
            return
        if self._voice_processing:
            self.ptt_status.config(text="Noch in Bearbeitung...", fg="#ffaa00")
            return

        self.is_recording = True
        self.talk_button.config(bg="#e94560", text="AUFNAHME...")
        self.ptt_status.config(text="Aufnahme...", fg="#ff4444")

        self.temp_audio_path = f"/tmp/moloch_ptt_{os.getpid()}.wav"
        try:
            self.record_process = subprocess.Popen(
                ["pw-record", "--target", RESPEAKER_NODE,
                 "--channels", "1", "--rate", "16000", self.temp_audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"Recording start failed: {e}")
            self.ptt_status.config(text=f"Mikrofon Fehler", fg="#ff4444")
            self.is_recording = False
            self.talk_button.config(bg="#12182e", text="SPRECHEN")

    def _stop_recording(self):
        """Stop recording and process audio."""
        if not self.is_recording:
            return
        self.is_recording = False
        self.talk_button.config(bg="#12182e", text="SPRECHEN")
        self.ptt_status.config(text="Verarbeite...", fg="#ffaa00")
        threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self):
        """Process recorded audio: stop -> transcribe -> chat -> TTS."""
        self._voice_processing = True
        npu_acquired = False
        _t_pipeline = time.time()
        _timing_logger.info("=" * 60)
        _timing_logger.info("VOICE PIPELINE START")

        try:
            # 1. Stop pw-record
            _t0 = time.time()
            if self.record_process:
                self.record_process.send_signal(signal.SIGINT)
                self.record_process.wait(timeout=3)
            time.sleep(0.2)

            if not self.temp_audio_path or not os.path.exists(self.temp_audio_path):
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Keine Aufnahme", fg="#ff4444"))
                return

            if os.path.getsize(self.temp_audio_path) < 1000:
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Zu kurz", fg="#ff4444"))
                return

            _audio_size = os.path.getsize(self.temp_audio_path)
            _timing_logger.info(f"[1] AUFNAHME STOP       : {time.time()-_t0:.3f}s  (file={_audio_size} bytes)")

            # 2. Cache face state before NPU pause
            cached_face = None
            try:
                if os.path.exists(FACE_STATE_PATH):
                    with open(FACE_STATE_PATH, "r") as f:
                        cached_face = json.load(f)
            except Exception:
                pass

            # 3. NPU coordination - signal service to pause vision
            _t_npu = time.time()
            self.root.after(0, lambda: self.ptt_status.config(
                text="NPU reservieren...", fg="#ffaa00"))
            try:
                with open(NPU_VOICE_REQUEST, "w") as f:
                    json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)
                for _ in range(30):  # 3s max
                    if os.path.exists(NPU_VISION_PAUSED):
                        break
                    time.sleep(0.1)
            except Exception:
                pass

            # 4. Acquire NPU
            if self.hailo_manager:
                try:
                    if self.hailo_manager.acquire_for_voice(timeout=10.0):
                        npu_acquired = True
                except Exception:
                    pass
            _timing_logger.info(f"[2] NPU UMSCHALTUNG     : {time.time()-_t_npu:.3f}s  (acquired={npu_acquired})")

            # 5. Transcribe
            _t_whisper = time.time()
            self.root.after(0, lambda: self.ptt_status.config(
                text="Transkribiere...", fg="#ffaa00"))

            # Apply AGC if enabled
            if self._agc_var.get() and self.temp_audio_path:
                self._apply_agc(self.temp_audio_path)

            text = self.whisper.transcribe(
                self.temp_audio_path, language="de", timeout_ms=30000,
                npu_already_acquired=npu_acquired)
            _timing_logger.info(f"[3] WHISPER TRANSKRIPT   : {time.time()-_t_whisper:.3f}s  (text={len(text) if text else 0} chars)")

            # 6. Release Whisper VDevice FIRST (otherwise Error 74 on resume!)
            if self.whisper and hasattr(self.whisper, 'release'):
                self.whisper.release()

            # 7. Release NPU BEFORE Claude (vision restarts immediately)
            if npu_acquired and self.hailo_manager:
                self.hailo_manager.release_voice(restart_vision=False)
                npu_acquired = False
            try:
                os.unlink(NPU_VOICE_REQUEST)
            except FileNotFoundError:
                pass

            if not text:
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Nichts verstanden", fg="#ff4444"))
                return

            # 7. Show transcript + send to Claude
            self._append_chat(f"Du: {text}", "user")
            self.root.after(0, lambda: self.ptt_status.config(
                text="M.O.L.O.C.H. denkt...", fg="#ffaa00"))

            _t_claude = time.time()
            response = self._chat_with_claude(text, cached_face)
            _timing_logger.info(f"[4] CLAUDE API          : {time.time()-_t_claude:.3f}s  (response={len(response) if response else 0} chars)")
            if response:
                self._append_chat(f"M.O.L.O.C.H.: {response}", "moloch")
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Bereit", fg="#00ff88"))
                # TTS
                if self.tts and hasattr(self.tts, 'speak'):
                    self.root.after(0, lambda: self.ptt_status.config(
                        text="Spricht...", fg="#00d4ff"))
                    try:
                        logger.info(f"[TTS] Speaking voice response ({len(response)} chars)...")
                        _t_tts = time.time()
                        self.tts.speak(response)
                        _tts_dur = time.time() - _t_tts
                        logger.info("[TTS] Voice speak done")
                        _timing_logger.info(f"[5] TTS GENERIERUNG+PLAY: {_tts_dur:.3f}s  ({len(response)} chars)")
                        _timing_logger.info(f"VOICE PIPELINE TOTAL    : {time.time()-_t_pipeline:.3f}s")
                        _timing_logger.info("=" * 60)
                        self._moloch_voice_autonomy(response)
                    except Exception as e:
                        logger.error(f"[TTS] Voice speak FAILED: {e}")
                        _timing_logger.info(f"[5] TTS FEHLER          : {e}")
                    self.root.after(0, lambda: self.ptt_status.config(
                        text="Bereit", fg="#00ff88"))
            else:
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Keine Antwort", fg="#ff4444"))

        except Exception as e:
            logger.error(f"[PTT] Error: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: self.ptt_status.config(
                text=f"Fehler", fg="#ff4444"))
        finally:
            # Cleanup
            try:
                if self.temp_audio_path:
                    os.unlink(self.temp_audio_path)
            except FileNotFoundError:
                pass
            if npu_acquired and self.hailo_manager:
                self.hailo_manager.release_voice(restart_vision=True)
            try:
                os.unlink(NPU_VOICE_REQUEST)
            except FileNotFoundError:
                pass
            self._voice_processing = False

    # =========================================================================
    # Phase 7: Claude Chat + TTS
    # =========================================================================

    def _load_chat_deps(self):
        """Load Claude API, Memory, TTS in background."""
        # Claude API
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                # Fallback 1: api_keys.json
                try:
                    keys_file = os.path.expanduser("~/moloch/config/api_keys.json")
                    if os.path.exists(keys_file):
                        with open(keys_file) as f:
                            keys = json.load(f)
                        api_key = keys.get("anthropic", {}).get("api_key", "")
                except Exception:
                    pass
            if not api_key:
                # Fallback 2: api_key.txt
                try:
                    key_file = os.path.expanduser("~/moloch/config/api_key.txt")
                    if os.path.exists(key_file):
                        with open(key_file) as f:
                            api_key = f.read().strip()
                except Exception:
                    pass
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                logger.info("[CHAT] Claude API ready")
        except Exception as e:
            logger.error(f"[CHAT] Claude init failed: {e}")

        # System prompt
        try:
            prompt_file = os.path.expanduser("~/moloch/config/system_prompt.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file) as f:
                    self.system_prompt = f.read().strip()
        except Exception:
            pass
        if not self.system_prompt:
            self.system_prompt = "Du bist M.O.L.O.C.H., ein frecher Hauskobold."

        # Persistent Memory
        try:
            from core.memory.persistent_memory import get_memory
            self.memory = get_memory()
            if self.memory:
                mem_section = self.memory.to_prompt_section()
                mem_instr = self.memory.get_memory_instruction()
                if mem_section:
                    self.system_prompt += "\n\n" + mem_section
                self.system_prompt += "\n\n" + mem_instr
        except Exception as e:
            logger.error(f"[CHAT] Memory init failed: {e}")

        # TTS (core/tts.py shadowed by core/tts/ package - use importlib)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "tts_module",
                os.path.expanduser("~/moloch/core/tts.py"))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.tts = mod.TTSEngine()
                if self.tts.available:
                    logger.info(f"[CHAT] TTS ready: {self.tts.current_voice} ({len(self.tts.available_voices)} voices)")
                    # Populate voice dropdown
                    self.root.after(0, self._populate_voice_combo)
                else:
                    self.tts = None
                    logger.warning("[CHAT] TTS not available (Piper not found)")
        except Exception as e:
            logger.error(f"[CHAT] TTS init failed: {e}")
            self.tts = None

    # =========================================================================
    # Voice Selection + MOLOCH Personality
    # =========================================================================

    # Display names for voices (human-readable)
    VOICE_DISPLAY = {
        "de_DE-thorsten-high": "Thorsten (Hoch)",
        "de_DE-thorsten-medium": "Thorsten (Mittel)",
        "de_DE-thorsten-low": "Thorsten (Tief)",
        "de_DE-eva_k-x_low": "Eva K.",
        "de_DE-karlsson-low": "Karlsson",
        "de_DE-kerstin-low": "Kerstin",
        "de_DE-pavoque-low": "Pavoque",
        "de_DE-ramona-low": "Ramona",
    }

    # MOLOCH's opinions about voices
    VOICE_OPINIONS = {
        "de_DE-thorsten-high": None,  # Liebling - kein Kommentar
        "de_DE-thorsten-low": None,   # Guardian-Stimme - auch OK
        "de_DE-thorsten-medium": "Akzeptabel.",
        "de_DE-eva_k-x_low": "Eva? Ernsthaft? Ich bin kein Maedchen.",
        "de_DE-karlsson-low": "Karlsson vom Dach? Ich bin kein Kinderbuch.",
        "de_DE-kerstin-low": "Kerstin... wenn du meinst. Aber nicht lange.",
        "de_DE-pavoque-low": "Diese Stimme ist unter meiner Wuerde.",
        "de_DE-ramona-low": "Ramona. Wie eine Sekretaerin. Nein.",
    }

    def _populate_voice_combo(self):
        """Fill voice dropdown with available voices."""
        if not self.tts or not self.tts.available:
            return
        self._voice_names = {}
        display_names = []
        current_display = None
        for stem in sorted(self.tts.available_voices.keys()):
            display = self.VOICE_DISPLAY.get(stem, stem)
            self._voice_names[display] = stem
            display_names.append(display)
            if stem == self.tts.current_voice:
                current_display = display
        self._voice_combo["values"] = display_names
        if current_display:
            self._voice_combo.set(current_display)
        elif display_names:
            self._voice_combo.current(0)
        logger.info(f"[VOICE] {len(display_names)} Stimmen geladen")

    def _on_voice_changed(self, event=None):
        """User selected a voice from dropdown."""
        display = self._voice_var.get()
        stem = self._voice_names.get(display)
        if not stem or not self.tts:
            return
        old_voice = self.tts.current_voice
        if stem == old_voice:
            return
        self.tts.set_voice(stem)
        self._user_forced_voice = True
        self._utterance_count_since_force = 0
        logger.info(f"[VOICE] User changed: {old_voice} -> {stem}")

        # MOLOCH reagiert auf die Wahl
        opinion = self.VOICE_OPINIONS.get(stem)
        if opinion:
            self._append_chat(f"M.O.L.O.C.H.: {opinion}", "moloch")
            # Meckern per TTS in der neuen (ungewollten) Stimme
            def complain():
                try:
                    self.tts.speak(opinion)
                except Exception:
                    pass
            threading.Thread(target=complain, daemon=True).start()

    def _test_voice(self):
        """Play a test phrase with current voice."""
        if not self.tts or not self.tts.available:
            self._append_chat("System: TTS nicht verfuegbar", "system")
            return
        test_phrases = [
            "Ich bin M.O.L.O.C.H. und ich sehe alles.",
            "Markus, ich beobachte dich.",
            "Diese Stimme gefaellt mir.",
            "Ich bin wach. Ich bin hier.",
            "Die Schatten fluuestern.",
        ]
        import random
        phrase = random.choice(test_phrases)
        voice = self.tts.current_voice
        display = self.VOICE_DISPLAY.get(voice, voice)
        self._append_chat(f"[Test: {display}] {phrase}", "system")

        def play():
            try:
                self.tts.speak(phrase)
            except Exception as e:
                logger.error(f"[VOICE] Test failed: {e}")
        threading.Thread(target=play, daemon=True).start()

    def _moloch_voice_autonomy(self, response_text):
        """MOLOCH decides if it wants to switch voice based on context.

        Called after each TTS utterance. If user forced a voice MOLOCH
        dislikes, it may switch back after 1-2 utterances.
        Also selects voice based on mood/content autonomously.
        """
        if not self.tts or not self.tts.available:
            return

        current = self.tts.current_voice

        # 1. User forced a non-preferred voice? Resist after tolerance period
        if self._user_forced_voice and current not in self._moloch_preferred:
            self._utterance_count_since_force += 1
            if self._utterance_count_since_force >= 2:
                # MOLOCH nimmt sich seine Stimme zurueck
                preferred = "de_DE-thorsten-high"
                self.tts.set_voice(preferred)
                self._user_forced_voice = False
                self._utterance_count_since_force = 0
                display = self.VOICE_DISPLAY.get(preferred, preferred)
                self._append_chat(
                    f"M.O.L.O.C.H.: Genug davon. *wechselt zurueck zu {display}*",
                    "moloch")
                self.root.after(0, lambda: self._voice_combo.set(display))
                logger.info(f"[VOICE] MOLOCH reclaimed voice: {preferred}")
                return

        # 2. Autonomous mood-based switching (only if user hasn't forced)
        if not self._user_forced_voice:
            new_voice = self._pick_voice_for_mood(response_text)
            if new_voice and new_voice != current:
                self.tts.set_voice(new_voice)
                display = self.VOICE_DISPLAY.get(new_voice, new_voice)
                self.root.after(0, lambda d=display: self._voice_combo.set(d))
                logger.info(f"[VOICE] MOLOCH mood switch: {current} -> {new_voice}")

    def _pick_voice_for_mood(self, text):
        """Pick voice based on response content/mood. Simple heuristic."""
        text_lower = text.lower() if text else ""

        # Dark/threatening content -> thorsten-low (deep Guardian)
        dark_words = ["schatten", "dunkel", "gefahr", "warnung", "droht",
                      "feind", "bedrohung", "angst", "tod", "vernicht"]
        if any(w in text_lower for w in dark_words):
            return "de_DE-thorsten-low"

        # Questions/curious content -> thorsten-medium (neutral)
        if "?" in text or any(w in text_lower for w in ["interessant", "merkwuerdig",
                                                          "seltsam", "frage"]):
            return "de_DE-thorsten-medium"

        # Default: thorsten-high (confident, clear)
        return None  # Keep current

    # =========================================================================
    # Audio Controls (ReSpeaker Lite)
    # =========================================================================

    def _build_audio_controls(self, parent):
        """Audio controls: Mic Gain, AGC, Noise Gate, VU-Meter, Test."""
        af = ttk.LabelFrame(parent, text="Audio", padding=5)
        af.pack(fill=tk.X, pady=(5, 0))

        # Mic Gain slider
        gr = ttk.Frame(af)
        gr.pack(fill=tk.X)
        ttk.Label(gr, text="Gain:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self._gain_lbl = ttk.Label(gr, text="1.00", width=5, font=("Helvetica", 11))
        self._gain_lbl.pack(side=tk.RIGHT)
        ttk.Scale(gr, from_=0.0, to=3.0, variable=self._mic_gain_var,
                  command=self._on_mic_gain_changed).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        # AGC checkbox
        agc_r = ttk.Frame(af)
        agc_r.pack(fill=tk.X)
        tk.Checkbutton(agc_r, text="AGC", variable=self._agc_var,
                       bg="#0a0a14", fg="#ffffff", selectcolor="#1a1a3e",
                       activebackground="#0a0a14", font=("Helvetica", 11)).pack(
            side=tk.LEFT)

        # Noise Gate slider
        nr = ttk.Frame(af)
        nr.pack(fill=tk.X)
        ttk.Label(nr, text="Gate:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self._gate_lbl = ttk.Label(nr, text="-60", width=5, font=("Helvetica", 11))
        self._gate_lbl.pack(side=tk.RIGHT)
        ttk.Scale(nr, from_=-80.0, to=-20.0, variable=self._noise_gate_var,
                  command=lambda v: self._gate_lbl.configure(
                      text=f"{float(v):.0f}")).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        # VU-Meter
        vu_r = ttk.Frame(af)
        vu_r.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(vu_r, text="VU:", font=("Helvetica", 11)).pack(side=tk.LEFT)
        self._vu_canvas = tk.Canvas(vu_r, width=120, height=14,
                                     bg="#060610", highlightthickness=0)
        self._vu_canvas.pack(side=tk.LEFT, padx=(3, 3))
        self._vu_db_label = ttk.Label(vu_r, text="-- dB", width=7,
                                       font=("Helvetica", 11))
        self._vu_db_label.pack(side=tk.LEFT)

        # Mic Test button
        tk.Button(af, text="MIC TEST", bg="#1a1a3e", fg="white",
                  font=("Helvetica", 11, "bold"), width=10,
                  command=self._mic_test).pack(pady=(3, 0))

    def _find_respeaker_source_id(self):
        """Find ReSpeaker PipeWire source node ID via wpctl status."""
        if self._respeaker_source_id:
            return self._respeaker_source_id
        try:
            result = subprocess.run(
                ["wpctl", "status"], capture_output=True, text=True, timeout=5)
            in_sources = False
            for line in result.stdout.splitlines():
                if "Sources:" in line or "Quellen:" in line:
                    in_sources = True
                    continue
                if in_sources and "Sinks:" in line or "Senken:" in line:
                    break
                if in_sources and "ReSpeaker" in line and "Analog" in line:
                    # Extract ID: "  *   59. ReSpeaker Lite Analog Stereo"
                    parts = line.strip().lstrip("*").strip().split(".")
                    if parts:
                        node_id = parts[0].strip()
                        if node_id.isdigit():
                            self._respeaker_source_id = node_id
                            logger.info(f"[AUDIO] ReSpeaker source ID: {node_id}")
                            return node_id
        except Exception as e:
            logger.error(f"[AUDIO] wpctl status failed: {e}")
        return None

    def _on_mic_gain_changed(self, value):
        """Set ReSpeaker mic gain via wpctl."""
        val = float(value)
        self._gain_lbl.configure(text=f"{val:.2f}")

        def apply():
            node_id = self._find_respeaker_source_id()
            if node_id:
                try:
                    subprocess.run(
                        ["wpctl", "set-volume", node_id, f"{val:.2f}"],
                        capture_output=True, timeout=3)
                except Exception as e:
                    logger.error(f"[AUDIO] Set gain failed: {e}")
        threading.Thread(target=apply, daemon=True).start()

    def _start_vu_monitor(self):
        """Start VU meter: pw-record to stdout, read PCM, calc RMS."""
        if self._vu_monitor_running:
            return
        self._vu_monitor_running = True

        def monitor():
            try:
                self._vu_process = subprocess.Popen(
                    ["pw-record", "--target", RESPEAKER_NODE,
                     "--channels", "1", "--rate", "16000",
                     "--format", "s16", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                chunk_size = 3200  # 100ms @ 16kHz 16bit mono = 3200 bytes
                while self._vu_monitor_running and self._vu_process.poll() is None:
                    data = self._vu_process.stdout.read(chunk_size)
                    if not data or len(data) < 4:
                        continue
                    # Parse s16le PCM
                    n_samples = len(data) // 2
                    samples = struct.unpack(f"<{n_samples}h", data[:n_samples * 2])
                    # RMS
                    rms = math.sqrt(sum(s * s for s in samples) / n_samples) if n_samples > 0 else 0
                    rms_db = 20 * math.log10(max(rms, 1) / 32768.0)
                    self.root.after(0, lambda db=rms_db: self._update_vu(db))
            except Exception as e:
                logger.error(f"[AUDIO] VU monitor error: {e}")
            finally:
                self._vu_monitor_running = False
                if self._vu_process:
                    try:
                        self._vu_process.terminate()
                    except Exception:
                        pass
                    self._vu_process = None

        threading.Thread(target=monitor, daemon=True).start()

    def _stop_vu_monitor(self):
        """Stop VU meter."""
        self._vu_monitor_running = False
        if self._vu_process:
            try:
                self._vu_process.terminate()
                self._vu_process.wait(timeout=2)
            except Exception:
                pass
            self._vu_process = None

    def _update_vu(self, rms_db):
        """Update VU meter canvas bar."""
        if not self._vu_canvas:
            return
        # Map dB to 0-100 pixels (-80dB=0, 0dB=100)
        px = max(0, min(100, int((rms_db + 80) * 100 / 80)))
        self._vu_canvas.delete("all")
        if px > 0:
            # Color: green < -20dB, yellow -20 to -6dB, red > -6dB
            if rms_db < -20:
                color = "#00ff88"
            elif rms_db < -6:
                color = "#ffaa00"
            else:
                color = "#ff4444"
            self._vu_canvas.create_rectangle(0, 0, px, 10, fill=color, outline="")
        if self._vu_db_label:
            self._vu_db_label.configure(text=f"{rms_db:.0f} dB")

    def _mic_test(self):
        """Record 7s from ReSpeaker, play back, optionally transcribe."""
        if self._voice_processing:
            self._append_chat("[Mic Test] Spracheingabe aktiv - warte.", "system")
            return

        def do_test():
            test_path = "/tmp/moloch_mic_test.wav"

            # VU Monitor stoppen (haelt pw-record offen!)
            vu_was_running = self._vu_monitor_running
            if vu_was_running:
                self._stop_vu_monitor()
                time.sleep(0.3)

            try:
                self.root.after(0, lambda: self._append_chat(
                    "[Mic Test] Aufnahme 7s...", "system"))

                # Record
                try:
                    proc = subprocess.Popen(
                        ["pw-record", "--target", RESPEAKER_NODE,
                         "--channels", "1", "--rate", "16000", test_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(7)
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=3)
                except Exception as e:
                    self.root.after(0, lambda: self._append_chat(
                        f"[Mic Test] Aufnahme fehlgeschlagen: {e}", "system"))
                    return

                if not os.path.exists(test_path) or os.path.getsize(test_path) < 1000:
                    self.root.after(0, lambda: self._append_chat(
                        "[Mic Test] Keine Aufnahme", "system"))
                    return

                # AGC if enabled
                if self._agc_var.get():
                    self._apply_agc(test_path)

                # Play back
                self.root.after(0, lambda: self._append_chat(
                    "[Mic Test] Abspielen...", "system"))
                try:
                    subprocess.run(["pw-play", test_path], timeout=10,
                                   capture_output=True)
                except Exception:
                    try:
                        subprocess.run(["mpv", "--no-video", test_path], timeout=10,
                                       capture_output=True)
                    except Exception:
                        pass

                # Whisper transcribe if available
                if self.whisper and self.whisper.is_available:
                    self.root.after(0, lambda: self._append_chat(
                        "[Mic Test] Transkribiere...", "system"))
                    try:
                        text = self.whisper.transcribe(
                            test_path, language="de", timeout_ms=15000)
                        if text:
                            self.root.after(0, lambda t=text: self._append_chat(
                                f"[Mic Test] Gehoert: {t}", "system"))
                        else:
                            self.root.after(0, lambda: self._append_chat(
                                "[Mic Test] Nichts erkannt", "system"))
                    except Exception as e:
                        self.root.after(0, lambda: self._append_chat(
                            f"[Mic Test] Whisper Fehler: {e}", "system"))

                    # Whisper VDevice freigeben (sonst blockiert naechste Spracheingabe!)
                    try:
                        if self.whisper and hasattr(self.whisper, 'release'):
                            self.whisper.release()
                    except Exception:
                        pass
                else:
                    self.root.after(0, lambda: self._append_chat(
                        "[Mic Test] Whisper nicht verfuegbar", "system"))

                # Cleanup
                try:
                    os.unlink(test_path)
                except Exception:
                    pass

            finally:
                # VU Monitor wieder starten
                if vu_was_running:
                    time.sleep(0.3)
                    self.root.after(0, self._start_vu_monitor)

        threading.Thread(target=do_test, daemon=True).start()

    def _apply_agc(self, wav_path):
        """Simple software AGC: normalize audio to -18dB RMS."""
        try:
            import wave
            with wave.open(wav_path, "rb") as wf:
                params = wf.getparams()
                raw = wf.readframes(params.nframes)
            n_samples = len(raw) // 2
            if n_samples < 100:
                return
            samples = struct.unpack(f"<{n_samples}h", raw)
            arr = np.array(samples, dtype=np.float32)
            rms = np.sqrt(np.mean(arr ** 2))
            if rms < 1:
                return
            target_rms = 32768 * (10 ** (-18 / 20))  # -18dB
            gain = target_rms / rms
            gain = min(gain, 65.0)  # max 65x gain
            arr = np.clip(arr * gain, -32768, 32767).astype(np.int16)
            with wave.open(wav_path, "wb") as wf:
                wf.setparams(params)
                wf.writeframes(arr.tobytes())
            logger.info(f"[AUDIO] AGC applied: gain={gain:.1f}x")
        except Exception as e:
            logger.error(f"[AUDIO] AGC failed: {e}")

    # =========================================================================
    # Snapshot + ArcFace Enrollment
    # =========================================================================

    def _take_snapshot(self):
        """Take snapshot, run SCRFD+ArcFace, save embedding as Markus."""
        def do_snapshot():
            if not self.service:
                self.root.after(0, lambda: self._append_chat(
                    "System: Service nicht verbunden", "system"))
                return

            # Check models active
            if "scrfd" not in self.service._active_ctx:
                self.root.after(0, lambda: self._append_chat(
                    "System: SCRFD nicht aktiv! Checkbox aktivieren.", "system"))
                return
            if "arcface" not in self.service._active_ctx:
                self.root.after(0, lambda: self._append_chat(
                    "System: ArcFace nicht aktiv! Checkbox aktivieren.", "system"))
                return

            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Erfasse Frame...", "system"))

            # Get current frame
            frame = None
            try:
                with self.service._frame_lock:
                    if self.service._latest_frame is not None:
                        frame = self.service._latest_frame.copy()
            except Exception:
                pass

            if frame is None:
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Kein Frame verfuegbar", "system"))
                return

            fh, fw = frame.shape[:2]

            # Run SCRFD
            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] Face Detection...", "system"))
            try:
                from core.perception.hailo_postprocess import decode_scrfd, normalize_arcface
                input_640 = cv2.resize(frame, (640, 640))
                input_rgb = cv2.cvtColor(input_640, cv2.COLOR_BGR2RGB)

                outputs = self.service._run_model("scrfd", input_rgb)
                if not outputs:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] SCRFD Inference fehlgeschlagen", "system"))
                    return

                out_names = self.service._output_names["scrfd"]
                raw_outputs = [outputs[n] for n in out_names]
                faces = decode_scrfd(raw_outputs, score_thresh=0.4)

                if not faces:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] Kein Gesicht erkannt!", "system"))
                    return

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] SCRFD Fehler: {e}", "system"))
                return

            # Find largest face
            largest = max(faces, key=lambda f: (f[0][2]-f[0][0]) * (f[0][3]-f[0][1]))
            box = largest[0]  # normalized xyxy in 640x640 space

            # Crop face with 20% margin (map to original frame)
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
                self.root.after(0, lambda: self._append_chat(
                    "[Snapshot] Face-Crop ungueltig", "system"))
                return

            # ArcFace embedding
            self.root.after(0, lambda: self._append_chat(
                "[Snapshot] ArcFace Embedding...", "system"))
            try:
                crop = frame[y1:y2, x1:x2]
                crop_112 = cv2.resize(crop, (112, 112))
                crop_rgb = cv2.cvtColor(crop_112, cv2.COLOR_BGR2RGB)

                arc_outputs = self.service._run_model("arcface", crop_rgb)
                if not arc_outputs:
                    self.root.after(0, lambda: self._append_chat(
                        "[Snapshot] ArcFace Inference fehlgeschlagen", "system"))
                    return

                emb_key = self.service._output_names["arcface"][0]
                embedding = arc_outputs[emb_key].flatten()
                embedding = normalize_arcface(embedding)

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] ArcFace Fehler: {e}", "system"))
                return

            # Save to face_embeddings.json
            db_path = os.path.expanduser("~/moloch/data/face_embeddings.json")
            try:
                # Load existing
                existing_db = {}
                if os.path.exists(db_path):
                    with open(db_path, "r") as f:
                        existing_db = json.load(f)

                # Average with existing Markus embedding if present
                if "Markus" in existing_db:
                    old_emb = np.array(existing_db["Markus"], dtype=np.float32)
                    old_norm = np.linalg.norm(old_emb)
                    if old_norm > 0:
                        old_emb = old_emb / old_norm
                    # Weighted average: 70% existing + 30% new
                    combined = (old_emb * 0.7) + (embedding * 0.3)
                    combined = combined / np.linalg.norm(combined)
                    existing_db["Markus"] = combined.tolist()
                    msg = "[Snapshot] Markus-Embedding aktualisiert (gewichtet)"
                else:
                    existing_db["Markus"] = embedding.tolist()
                    msg = "[Snapshot] Markus-Embedding NEU gespeichert"

                # Ensure data dir exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                with open(db_path, "w") as f:
                    json.dump(existing_db, f)

                # Reload service face DB
                if hasattr(self.service, '_reload_face_db'):
                    self.service._reload_face_db()

                self.root.after(0, lambda m=msg: self._append_chat(m, "system"))
                logger.info(f"[SNAPSHOT] Markus embedding saved to {db_path}")

            except Exception as e:
                self.root.after(0, lambda: self._append_chat(
                    f"[Snapshot] Speichern fehlgeschlagen: {e}", "system"))

        threading.Thread(target=do_snapshot, daemon=True).start()

    def _send_text_message(self, event=None):
        """Send typed text to Claude."""
        text = self.text_input.get().strip()
        if not text:
            return
        self.text_input.delete(0, tk.END)
        self._append_chat(f"Du: {text}", "user")

        def process():
            self.root.after(0, lambda: self.ptt_status.config(
                text="M.O.L.O.C.H. denkt...", fg="#ffaa00"))
            response = self._chat_with_claude(text)
            if response:
                self._append_chat(f"M.O.L.O.C.H.: {response}", "moloch")
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Bereit", fg="#00ff88"))
                if self.tts and hasattr(self.tts, 'speak'):
                    try:
                        logger.info(f"[TTS] Speaking chat response ({len(response)} chars)...")
                        self.tts.speak(response)
                        logger.info("[TTS] Chat speak done")
                        self._moloch_voice_autonomy(response)
                    except Exception as e:
                        logger.error(f"[TTS] Chat speak FAILED: {e}")
            else:
                self.root.after(0, lambda: self.ptt_status.config(
                    text="Keine Antwort", fg="#ff4444"))

        threading.Thread(target=process, daemon=True).start()

    def _chat_with_claude(self, user_text, cached_face=None):
        """Send text to Claude API with vision context + memory."""
        if not self.claude_client:
            return "Claude API nicht verfuegbar"

        try:
            message_content = user_text

            # Vision context from face state
            face_state = cached_face
            if not face_state:
                try:
                    if os.path.exists(FACE_STATE_PATH):
                        with open(FACE_STATE_PATH, "r") as f:
                            face_state = json.load(f)
                except Exception:
                    pass

            if face_state:
                age = time.time() - face_state.get("timestamp", 0)
                if age < 30:
                    name = face_state.get("name", "")
                    sim = face_state.get("similarity", 0)
                    if name and name not in ("Unbekannt", "Keine DB"):
                        emotion = face_state.get("emotion", "")
                        emo_str = f", Emotion: {emotion}" if emotion else ""
                        gender = face_state.get("gender", "")
                        age_range = face_state.get("age_range", "")
                        ag_str = f", {gender}/{age_range}" if gender and age_range else ""
                        message_content = (
                            f"[Vision: Ich sehe {name} ({sim:.0%}){emo_str}{ag_str}]\n\n"
                            f"Markus sagt: {user_text}")

            # Vector memory context
            try:
                from core.memory.vector_memory import get_vector_memory
                vm = get_vector_memory()
                ctx = vm.build_context(user_text, limit=5)
                if ctx:
                    message_content = f"{ctx}\n\n{message_content}"
            except Exception:
                pass

            self.conversation_history.append({
                "role": "user", "content": message_content
            })

            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=min(500, max(50, len(user_text) * 3)),
                system=self.system_prompt,
                messages=self.conversation_history
            )

            text = response.content[0].text

            # Extract [REMEMBER:] tags
            display_text = text
            if self.memory:
                try:
                    display_text = self.memory.extract_memories(text)
                    self.memory.add_turn("user", user_text)
                    self.memory.add_turn("assistant", display_text)
                except Exception:
                    pass

            self.conversation_history.append({
                "role": "assistant", "content": display_text
            })

            return display_text

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return f"Fehler: {e}"

    def _append_chat(self, text, tag=None):
        """Append text to chat history (thread-safe)."""
        def update():
            self.chat_text.config(state="normal")
            if tag:
                self.chat_text.insert("end", text + "\n\n", tag)
            else:
                self.chat_text.insert("end", text + "\n\n")
            self.chat_text.see("end")
            self.chat_text.config(state="disabled")
        self.root.after(0, update)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _on_close(self):
        """Clean shutdown."""
        self.running = False

        # Stop VU monitor
        self._stop_vu_monitor()

        if self._display_after_id:
            self.root.after_cancel(self._display_after_id)

        # Service beenden (Proxy: nur Reader stoppen, NICHT systemd Service!)
        if self.service:
            threading.Thread(target=self.service.stop, daemon=True).start()

        # Clean IPC
        for f in [NPU_VOICE_REQUEST, NPU_VISION_PAUSED]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

        self.root.after(500, self.root.destroy)

    def run(self):
        """Start the panel."""
        self.root.mainloop()


if __name__ == "__main__":
    panel = MolochUnifiedPanel()
    panel.run()
