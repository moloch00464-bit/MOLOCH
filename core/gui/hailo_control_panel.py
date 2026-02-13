#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hailo-10H Control Panel
======================================

Live RTSP Feed mit NPU-Detection Overlays.
Alle 4 Hailo-10H Modelle einzeln ein/ausschaltbar,
Confidence/NMS Threshold Slider, FPS-Anzeige.

3-Thread Architektur:
  1. RTSP Reader (daemon) - grab/retrieve im Hintergrund
  2. Inference Worker (daemon) - NPU Inferenz + Overlay zeichnen
  3. Tkinter Main - Display + Controls

Modelle:
  - SCRFD Face Detection (scrfd_10g.hef)
  - ArcFace Face Recognition (arcface_mobilefacenet.hef)
  - YOLOv8m Person Detection (yolov8m_h10.hef)
  - YOLOv8s Pose Estimation (yolov8s_pose_h10.hef)

Author: M.O.L.O.C.H. System
"""

import os
import sys
import time
import json
import asyncio
import threading
import logging
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Auto-source ~/.profile wenn env vars fehlen (Desktop-Launch)
if not os.environ.get("MOLOCH_CAMERA_HOST"):
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

import cv2
import numpy as np
from PIL import Image, ImageTk

# Moloch path
sys.path.insert(0, os.path.expanduser("~/moloch"))

from hailo_platform import HEF, VDevice, FormatType
from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms, decode_yolov8_pose,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_poses,
)
from core.hardware.hailo_manager import get_hailo_manager
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("HailoPanel")

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
}

# Face-DB Pfad
FACE_DB_PATH = os.path.expanduser("~/moloch/data/face_embeddings.json")

# Shared Face State (IPC mit push_to_talk)
FACE_STATE_PATH = "/tmp/moloch_face_state.json"

# Hailo-10H Hardware-Limit: max 2 Modelle gleichzeitig stabil
MAX_CONCURRENT_MODELS = 2

# Cross-process NPU coordination
NPU_VOICE_REQUEST = "/tmp/moloch_npu_voice_request"
NPU_VISION_PAUSED = "/tmp/moloch_npu_vision_paused"


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
        """Start async event loop and connect to eWeLink cloud."""
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
        """Run async coroutine and return result."""
        if not self.loop:
            return False
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Cloud call failed: {e}")
            return False


class HailoControlPanel:
    """M.O.L.O.C.H. Hailo-10H Control Panel GUI."""

    PREVIEW_W = 640
    PREVIEW_H = 480
    DISPLAY_FPS = 15

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H. Hailo-10H Control Panel")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        # State
        self.running = True
        self._hailo_manager = None
        self._vdevice = None
        self._models = {}          # name -> infer_model
        self._output_names = {}    # name -> [output_layer_names]
        self._face_db = {}

        # Persistent Model Contexts (configure einmal, wiederverwenden)
        self._active_ctx = {}      # name -> _ModelContext
        self._model_order = []     # FIFO: Reihenfolge der Aktivierung
        self._ctx_lock = threading.Lock()
        self._input_640 = np.empty((640, 640, 3), dtype=np.uint8)  # Pre-allocated

        # TTS Announcement Cooldown
        self._last_announce = {}   # name -> timestamp

        # Cross-process NPU pause state
        self._paused_models = []   # Models active before voice pause
        self._npu_paused = False

        # Pause inference during model configure (NPU kann nicht beides gleichzeitig)
        self._configuring = threading.Event()
        self._configuring.set()  # Startzustand: nicht konfigurierend

        # Dynamischer Modell-Swap: Face erkannt -> ArcFace statt YOLOv8m
        self._face_mode_active = False   # ArcFace aktiv statt YOLOv8m
        self._face_seen_count = 0        # Konsekutive Frames mit Face
        self._face_lost_time = 0         # Timestamp: letzte Face verloren
        self._swapping_models = False    # Swap laeuft gerade
        self._swap_lock = threading.Lock()  # Atomisches Check+Set fuer _swapping_models
        self._FACE_MODE_FRAMES = 3      # Frames bevor Swap zu ArcFace
        self._FACE_MODE_TIMEOUT = 5.0   # Sekunden ohne Face -> zurueck zu YOLOv8m
        self._FACE_MODE_STARTUP_DELAY = 5.0  # Sekunden nach Takeover bevor Swap erlaubt

        # Autonomes Tracking
        self._autonomous_mode = False
        self._manual_autonomous = False  # True = User hat manuell AUTONOM gedrueckt
        self._tracker = None

        # Guardian/Tentakel Mode:
        # Kamera Smart Tracking scannt -> Kamera bewegt sich -> MOLOCH uebernimmt -> NPU an
        self._guardian_mode = True          # Automatischer Wechsel aktiv
        self._moloch_has_control = False    # MOLOCH hat gerade Kontrolle
        self._takeover_reason = ""          # Warum uebernommen
        self._last_interesting_time = 0     # Letzte interessante Erkennung
        self._search_start_time = 0         # Seit wann sucht der Tracker
        self.TAKEOVER_TIMEOUT = 30          # Sekunden ohne Interest -> zurueckgeben
        self.SEARCH_TIMEOUT = 20            # Sekunden SEARCHING -> zurueckgeben
        # Kamera-Bewegungserkennung (Smart Tracking hat was gesehen)
        self._guardian_last_pan = None      # Letzte bekannte Pan-Position
        self._guardian_last_tilt = None     # Letzte bekannte Tilt-Position
        self._guardian_move_thresh = 5.0    # Grad Aenderung = Kamera hat sich bewegt
        self._guardian_move_count = 0       # Aufeinanderfolgende Bewegungen
        self._guardian_move_required = 2    # 2 Polls noetig (2x3s = 6s, Counter dekrementiert statt Reset)
        self._takeover_cooldown_until = 0   # Timestamp: kein Takeover bis dahin
        self.RELEASE_COOLDOWN = 60          # Basis-Sekunden nach Release kein neuer Takeover
        self.MAX_COOLDOWN = 180             # Max 3 Minuten Cooldown (war 600)
        self.STARTUP_GRACE = 30             # Sekunden nach Start kein Takeover
        self._failed_takeovers = 0          # Aufeinanderfolgende leere Takeovers (fuer Backoff)
        self._takeover_found_something = False  # Wurde im aktuellen Takeover was gefunden?
        # Kalibrierung nur noch manuell via GUI-Button (nicht bei Takeover)
        self._takeover_cooldown_until = time.time() + self.STARTUP_GRACE  # Grace Period
        self._transitioning = False  # Verhindert ueberlappende Takeover/Release
        self._transition_lock = threading.Lock()  # Atomisches Check+Set fuer _transitioning

        # Frame Locks
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()
        self._display_after_id = None

        # Modell-Toggles (BooleanVar)
        self.scrfd_enabled = tk.BooleanVar(value=False)
        self.arcface_enabled = tk.BooleanVar(value=False)
        self.yolo_enabled = tk.BooleanVar(value=False)
        self.pose_enabled = tk.BooleanVar(value=False)

        # Threshold Vars
        self.scrfd_conf = tk.DoubleVar(value=0.40)
        self.scrfd_nms = tk.DoubleVar(value=0.40)
        self.arcface_thresh = tk.DoubleVar(value=0.60)
        self.yolo_conf = tk.DoubleVar(value=0.50)
        self.pose_conf = tk.DoubleVar(value=0.50)
        self.pose_nms = tk.DoubleVar(value=0.70)

        # FPS Tracking
        self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "pose": 0, "total": 0}
        self._fps_lock = threading.Lock()

        # Styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#1a1a2e")
        self.style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0",
                             font=("Helvetica", 10))
        self.style.configure("Header.TLabel", background="#1a1a2e", foreground="#00d4ff",
                             font=("Helvetica", 12, "bold"))
        self.style.configure("Status.TLabel", background="#1a1a2e", foreground="#66ff66",
                             font=("Helvetica", 9))
        self.style.configure("FPS.TLabel", background="#1a1a2e", foreground="#ffaa00",
                             font=("Helvetica", 9, "bold"))
        self.style.configure("TScale", background="#1a1a2e")
        self.style.configure("TCheckbutton", background="#1a1a2e", foreground="#e0e0e0")

        self._build_ui()
        self._start_init_thread()

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        """Baue komplettes UI - 3-Bereich Layout: Preview+Kamera links, Modelle rechts, Cloud unten."""
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        # === TOP: Status Bar ===
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 3))

        self.status_label = ttk.Label(top, text="Initialisierung...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)

        self.total_fps_label = ttk.Label(top, text="FPS: --", style="FPS.TLabel")
        self.total_fps_label.pack(side=tk.RIGHT)

        # === MIDDLE: Preview links + Controls rechts ===
        middle = ttk.Frame(main)
        middle.pack(fill=tk.BOTH, expand=True)

        # ---- LINKS: Preview + Kamera-Kontrolle darunter ----
        left_frame = ttk.Frame(middle)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Live Preview
        self.preview_canvas = tk.Canvas(
            left_frame, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#000000", highlightthickness=1, highlightbackground="#333"
        )
        self.preview_canvas.pack(pady=(0, 3))

        # --- KAMERA + PTZ unter dem Preview (nebeneinander) ---
        cam_bottom = tk.Frame(left_frame, bg="#1a1a2e")
        cam_bottom.pack(fill=tk.X)

        # Links: Status + Autonomie
        cam_left = tk.Frame(cam_bottom, bg="#1a1a2e")
        cam_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Status-Frame
        self._cam_status_frame = tk.Frame(cam_left, bg="#1a3a1a", padx=6, pady=4,
                                          highlightbackground="#00ff88",
                                          highlightthickness=1)
        self._cam_status_frame.pack(fill=tk.X)

        self._cam_control_label = tk.Label(
            self._cam_status_frame, text="MOLOCH KONTROLLE",
            font=("Helvetica", 10, "bold"), fg="#00ff88", bg="#1a3a1a")
        self._cam_control_label.pack()

        self._cam_detail_label = tk.Label(
            self._cam_status_frame, text="Smart Tracking: AUS | ONVIF: ...",
            font=("Helvetica", 7), fg="#aaaaaa", bg="#1a3a1a")
        self._cam_detail_label.pack()

        self._cam_ptz_label = tk.Label(
            self._cam_status_frame, text="PTZ: --",
            font=("Courier", 7), fg="#888888", bg="#1a3a1a")
        self._cam_ptz_label.pack()

        # AUTONOM/MANUELL + Tracker State
        self._auto_btn = tk.Button(
            cam_left, text="MANUELL", bg="#2a2a4e", fg="white",
            font=("Helvetica", 10, "bold"), command=self._toggle_autonomous)
        self._auto_btn.pack(fill=tk.X, pady=(3, 0))

        self._tracker_state_label = tk.Label(
            cam_left, text="", font=("Courier", 8), fg="#888888", bg="#1a1a2e")
        self._tracker_state_label.pack()

        # Smart Tracking Button
        self._smart_tracking_on = False
        self._st_lock = threading.Lock()
        self._smart_tracking_btn = tk.Button(
            cam_left, text="Smart Tracking: AUS", bg="#2a2a4e", fg="white",
            font=("Helvetica", 8), command=self._toggle_smart_tracking)
        self._smart_tracking_btn.pack(fill=tk.X, pady=(2, 0))

        # Cloud Controller starten
        self._cloud = None
        threading.Thread(target=self._connect_cloud, daemon=True).start()

        # Mitte: PTZ Steuerkreuz + Speed
        cam_mid = tk.Frame(cam_bottom, bg="#1a1a2e")
        cam_mid.pack(side=tk.LEFT, padx=5)

        btn_cfg = dict(width=3, font=("Helvetica", 11, "bold"),
                       bg="#2a2a4e", fg="white", activebackground="#4a4a6e")

        # PTZ Kreuz kompakt
        ptz_grid = tk.Frame(cam_mid, bg="#1a1a2e")
        ptz_grid.pack()
        tk.Button(ptz_grid, text="\u25B2", command=lambda: self._ptz_move("up"),
                  **btn_cfg).grid(row=0, column=1)
        tk.Button(ptz_grid, text="\u25C0", command=lambda: self._ptz_move("left"),
                  **btn_cfg).grid(row=1, column=0)
        tk.Button(ptz_grid, text="\u25CF", command=lambda: self._ptz_move("home"),
                  width=3, font=("Helvetica", 9), bg="#444466", fg="white",
                  activebackground="#666688").grid(row=1, column=1)
        tk.Button(ptz_grid, text="\u25B6", command=lambda: self._ptz_move("right"),
                  **btn_cfg).grid(row=1, column=2)
        tk.Button(ptz_grid, text="\u25BC", command=lambda: self._ptz_move("down"),
                  **btn_cfg).grid(row=2, column=1)

        # Speed Slider
        self._ptz_speed_var = tk.DoubleVar(value=15.0)
        speed_row = tk.Frame(cam_mid, bg="#1a1a2e")
        speed_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(speed_row, text="Spd:", font=("Helvetica", 7),
                 fg="#888888", bg="#1a1a2e").pack(side=tk.LEFT)
        self._ptz_speed_label = tk.Label(speed_row, text="15",
                                         font=("Courier", 7), fg="#aaaaaa", bg="#1a1a2e")
        self._ptz_speed_label.pack(side=tk.RIGHT)
        tk.Scale(speed_row, from_=1, to=50, orient=tk.HORIZONTAL,
                 variable=self._ptz_speed_var, length=80,
                 bg="#1a1a2e", fg="#aaaaaa", troughcolor="#2a2a4e",
                 highlightthickness=0, showvalue=0,
                 command=lambda v: self._ptz_speed_label.config(
                     text=f"{float(v):.0f}")).pack(side=tk.LEFT, padx=2)

        # Quick Positions
        quick_row = tk.Frame(cam_mid, bg="#1a1a2e")
        quick_row.pack(pady=(2, 0))
        for name, pan, tilt in [("L", 170, 0), ("M", 0, 0), ("R", -168, 0)]:
            tk.Button(quick_row, text=name, bg="#2a2a4e", fg="white", width=3,
                      font=("Helvetica", 8),
                      command=lambda p=pan, t=tilt: self._ptz_goto(p, t)).pack(
                side=tk.LEFT, padx=1)

        # Rechts: eWeLink Cloud Controls
        cam_right = tk.Frame(cam_bottom, bg="#1a1a2e")
        cam_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(cam_right, text="eWeLink", font=("Helvetica", 9, "bold"),
                 fg="#00d4ff", bg="#1a1a2e").pack(anchor=tk.W)

        self._cloud_status_label = tk.Label(
            cam_right, text="Cloud: ...",
            font=("Courier", 7), fg="#888888", bg="#1a1a2e")
        self._cloud_status_label.pack(anchor=tk.W)

        # LED + Flip in einer Zeile
        toggle_row1 = tk.Frame(cam_right, bg="#1a1a2e")
        toggle_row1.pack(fill=tk.X, pady=1)
        self._led_var = tk.BooleanVar(value=False)
        tk.Checkbutton(toggle_row1, text="LED", variable=self._led_var,
                       bg="#1a1a2e", fg="#cccccc", selectcolor="#2a2a4e",
                       activebackground="#1a1a2e", font=("Helvetica", 8),
                       command=self._set_cloud_led).pack(side=tk.LEFT)
        self._flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(toggle_row1, text="Flip", variable=self._flip_var,
                       bg="#1a1a2e", fg="#cccccc", selectcolor="#2a2a4e",
                       activebackground="#1a1a2e", font=("Helvetica", 8),
                       command=self._set_cloud_flip).pack(side=tk.LEFT, padx=(8, 0))

        # IR/Nachtsicht
        ir_row = tk.Frame(cam_right, bg="#1a1a2e")
        ir_row.pack(fill=tk.X, pady=1)
        tk.Label(ir_row, text="IR:", font=("Helvetica", 8),
                 fg="#cccccc", bg="#1a1a2e").pack(side=tk.LEFT)
        self._night_var = tk.StringVar(value="Aus")
        night_combo = ttk.Combobox(ir_row, textvariable=self._night_var,
                                   values=["Aus", "Auto", "An"], state="readonly", width=5)
        night_combo.pack(side=tk.LEFT, padx=3)
        night_combo.bind("<<ComboboxSelected>>", lambda e: self._set_cloud_night())

        # Alarm + Kalibrierung
        cloud_btns = tk.Frame(cam_right, bg="#1a1a2e")
        cloud_btns.pack(fill=tk.X, pady=(2, 0))
        tk.Button(cloud_btns, text="ALARM", bg="#ff4444", fg="white", width=6,
                  font=("Helvetica", 8, "bold"),
                  command=self._trigger_alarm).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(cloud_btns, text="Kalib.", bg="#ff8800", fg="white", width=5,
                  font=("Helvetica", 8),
                  command=self._trigger_calibration).pack(side=tk.LEFT)

        # ---- RECHTS: Modelle + Aktionen ----
        ctrl_frame = ttk.Frame(middle)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 3))

        ttk.Label(ctrl_frame, text="MODELLE", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 3))

        # -- SCRFD --
        self._build_model_section(
            ctrl_frame, "SCRFD Face", self.scrfd_enabled,
            "scrfd", [
                ("Conf", self.scrfd_conf, 0.1, 0.9),
                ("NMS", self.scrfd_nms, 0.1, 0.9),
            ]
        )

        # -- ArcFace --
        self._build_model_section(
            ctrl_frame, "ArcFace", self.arcface_enabled,
            "arcface", [
                ("Thresh", self.arcface_thresh, 0.3, 0.9),
            ]
        )

        # -- YOLOv8m --
        self._build_model_section(
            ctrl_frame, "YOLOv8m Person", self.yolo_enabled,
            "yolov8m", [
                ("Conf", self.yolo_conf, 0.1, 0.9),
            ]
        )

        # -- Pose --
        self._build_model_section(
            ctrl_frame, "YOLOv8s Pose", self.pose_enabled,
            "pose", [
                ("Conf", self.pose_conf, 0.1, 0.9),
                ("NMS", self.pose_nms, 0.1, 0.9),
            ]
        )

        # Separator
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Aktionen als kompakte Button-Reihen
        act_row1 = tk.Frame(ctrl_frame, bg="#1a1a2e")
        act_row1.pack(fill=tk.X, pady=1)
        self.kill_btn = tk.Button(
            act_row1, text="PTT killen", bg="#ff4444", fg="white",
            font=("Helvetica", 8, "bold"), command=self._kill_push_to_talk)
        self.kill_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        tk.Button(act_row1, text="Snapshot", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=self._save_snapshot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        act_row2 = tk.Frame(ctrl_frame, bg="#1a1a2e")
        act_row2.pack(fill=tk.X, pady=1)
        tk.Button(act_row2, text="Alle AUS", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=self._all_models_off).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        tk.Button(act_row2, text="Face-DB", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=self._reload_face_db).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        # Kamera-Status Update starten
        self.root.after(1000, self._update_cam_status)

    def _build_model_section(self, parent, title, enabled_var, model_key, sliders):
        """Baue eine Modell-Section mit Toggle, FPS, Slidern."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        # Header-Zeile: Checkbox + FPS
        header = ttk.Frame(frame)
        header.pack(fill=tk.X)

        cb = tk.Checkbutton(
            header, text=title, variable=enabled_var,
            bg="#1a1a2e", fg="#e0e0e0", selectcolor="#2a2a4e",
            activebackground="#1a1a2e", font=("Helvetica", 10),
            command=lambda: self._on_model_toggle(model_key)
        )
        cb.pack(side=tk.LEFT)

        fps_label = ttk.Label(header, text="--- FPS", style="FPS.TLabel")
        fps_label.pack(side=tk.RIGHT)
        setattr(self, f"_{model_key}_fps_label", fps_label)

        # Slider pro Parameter
        for label_text, var, from_val, to_val in sliders:
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, padx=(20, 0))

            ttk.Label(row, text=f"  {label_text}:").pack(side=tk.LEFT)
            val_label = ttk.Label(row, text=f"{var.get():.2f}", width=5)
            val_label.pack(side=tk.RIGHT)

            scale = ttk.Scale(
                row, from_=from_val, to=to_val, variable=var,
                command=lambda v, lbl=val_label: lbl.configure(text=f"{float(v):.2f}")
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # =========================================================================
    # Initialisierung (Background Thread)
    # =========================================================================

    def _start_init_thread(self):
        """Starte Hailo + RTSP Init im Hintergrund."""
        def do_init():
            # 1. Hailo Manager
            try:
                self._hailo_manager = get_hailo_manager()
                # Warte auf NPU (NICHT andere Prozesse killen!)
                if not self._hailo_manager.is_device_free():
                    self._update_status("NPU belegt - warte auf Freigabe...")
                    for i in range(20):  # Max 10s warten
                        time.sleep(0.5)
                        if self._hailo_manager.is_device_free():
                            break
                    else:
                        self._update_status("NPU belegt - starte ohne Inference")
                        # RTSP trotzdem starten fuer Preview
                        self.root.after(100, self._start_rtsp)
                        return

                if not self._hailo_manager.acquire_for_vision(timeout=10.0):
                    self._update_status("FEHLER: NPU nicht verfuegbar!")
                    return
                self._update_status("NPU acquired")
            except Exception as e:
                self._update_status(f"Hailo Manager Fehler: {e}")
                return

            # 2. VDevice + Modelle laden
            try:
                params = VDevice.create_params()
                self._vdevice = VDevice(params)
                self._update_status("VDevice OK - Lade Modelle...")

                for name, path in MODEL_PATHS.items():
                    if not os.path.exists(path):
                        logger.warning(f"Modell nicht gefunden: {path}")
                        continue
                    hef = HEF(path)
                    infer_model = self._vdevice.create_infer_model(path)

                    # Format Types setzen (vor configure!)
                    infer_model.input().set_format_type(FormatType.UINT8)
                    out_names = [o.name for o in hef.get_output_vstream_infos()]
                    for oname in out_names:
                        infer_model.output(oname).set_format_type(FormatType.FLOAT32)

                    self._models[name] = infer_model
                    self._output_names[name] = out_names
                    logger.info(f"Modell geladen: {name} ({len(out_names)} outputs)")

                loaded = ", ".join(self._models.keys())
                self._update_status(f"NPU OK | Modelle: {loaded}")
            except Exception as e:
                self._update_status(f"Modell-Ladefehler: {e}")
                logger.exception("Modell-Ladefehler")
                return

            # 3. Face-DB laden
            self._face_db = load_face_db(FACE_DB_PATH)
            if self._face_db:
                logger.info(f"Face-DB: {len(self._face_db)} Personen geladen")

            # 4. RTSP starten
            self.root.after(100, self._start_rtsp)

            # 5. Inference Thread starten
            threading.Thread(target=self._inference_loop, daemon=True,
                             name="HailoInference").start()

            # 6. FPS Update Timer
            self.root.after(500, self._update_fps_display)

        threading.Thread(target=do_init, daemon=True, name="HailoInit").start()

    def _update_status(self, text):
        """Thread-safe Status-Update."""
        self.root.after(0, lambda: self.status_label.configure(text=text))

    # =========================================================================
    # RTSP Reader Thread
    # =========================================================================

    def _start_rtsp(self):
        """Starte RTSP Background Reader."""
        def rtsp_reader():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self._update_status(f"RTSP FEHLER: {RTSP_URL}")
                return

            self._update_status("RTSP + NPU aktiv")

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
        # Display Loop starten
        self.root.after(100, self._display_loop)

    # =========================================================================
    # Inference Worker Thread
    # =========================================================================

    def _configure_model(self, name):
        """Konfiguriere Modell persistent (einmalig ~400ms, danach 0ms)."""
        if name in self._active_ctx or name not in self._models:
            return

        infer_model = self._models[name]
        out_names = self._output_names[name]

        # Log state BEFORE configure attempt
        active_before = list(self._active_ctx.keys())
        logger.info(f"[CONFIGURE] {name}: aktive Modelle VORHER: {active_before}")

        # Inference pausieren - NPU kann nicht configure + run gleichzeitig
        self._configuring.clear()
        time.sleep(0.15)  # Warten bis laufende Inference fertig

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
                if name not in self._model_order:
                    self._model_order.append(name)
            active_after = list(self._active_ctx.keys())
            logger.info(f"[CONFIGURE] {name}: OK. Aktive Modelle NACHHER: {active_after}")
        except Exception as e:
            import traceback
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
        time.sleep(0.1)  # Warten bis laufende Inference fertig
        try:
            with self._ctx_lock:
                ctx = self._active_ctx.pop(name, None)
                if name in self._model_order:
                    self._model_order.remove(name)
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

    def _inference_loop(self):
        """Inference Worker: nimmt Frames, fuehrt aktive Modelle aus, zeichnet Overlays."""
        try:
            self._inference_loop_inner()
        except Exception as e:
            import traceback
            crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
            crash_info = (
                f"\n{'='*60}\n"
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFERENCE LOOP CRASH\n"
                f"Aktive Modelle: {list(self._active_ctx.keys())}\n"
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
            self._update_status(f"INFERENCE CRASH: {type(e).__name__}")

    def _inference_loop_inner(self):
        """Eigentliche Inference Loop."""
        while self.running:
            # Cross-process NPU coordination: Voice hat Vorrang
            if os.path.exists(NPU_VOICE_REQUEST):
                if not self._npu_paused:
                    self._pause_for_voice()
                time.sleep(0.1)
                continue
            if self._npu_paused:
                self._resume_after_voice()
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

            # Kein Modell konfiguriert -> nur Raw-Frame anzeigen
            any_active = bool(self._active_ctx) and (
                self.scrfd_enabled.get() or self.yolo_enabled.get() or
                self.pose_enabled.get())
            if not any_active:
                with self._annotated_lock:
                    self._annotated_frame = frame.copy()
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
            face_boxes = []
            face_detected = False        # SCRFD hat Gesicht erkannt (YOLOv8m ueberspringen)
            face_fed_to_tracker = False  # Face an Tracker gefuettert

            # 1. SCRFD Face Detection
            if self.scrfd_enabled.get() and "scrfd" in self._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("scrfd", input_rgb)
                    boxes, scores, landmarks = decode_scrfd(
                        outputs, img_size=640,
                        conf_thresh=self.scrfd_conf.get(),
                        iou_thresh=self.scrfd_nms.get()
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:
                        draw_faces(annotated, boxes, scores, landmarks, scale_x, scale_y)
                        face_boxes = list(zip(boxes, scores, landmarks))
                        face_detected = True  # YOLOv8m wird komplett uebersprungen
                        # Face hat PRIORITAET fuer Tracker (besser als Person-BBox)
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
                except Exception as e:
                    logger.error(f"SCRFD Fehler: {e}")

            # === Dynamischer Modell-Swap: Face -> ArcFace, kein Face -> YOLOv8m ===
            # Startup-Delay: Tracker braucht Zeit um sich zu stabilisieren
            _swap_allowed = (self._moloch_has_control and not self._swapping_models
                             and hasattr(self, '_takeover_time')
                             and time.time() - self._takeover_time > self._FACE_MODE_STARTUP_DELAY)
            if _swap_allowed:
                if face_detected:
                    self._face_seen_count += 1
                    self._face_lost_time = 0
                    if not self._face_mode_active and self._face_seen_count >= self._FACE_MODE_FRAMES:
                        threading.Thread(target=self._swap_to_arcface, daemon=True).start()
                else:
                    self._face_seen_count = 0
                    if self._face_mode_active:
                        if self._face_lost_time == 0:
                            self._face_lost_time = time.time()
                        elif time.time() - self._face_lost_time > self._FACE_MODE_TIMEOUT:
                            threading.Thread(target=self._swap_to_yolov8m, daemon=True).start()

            # 2. ArcFace (nur wenn SCRFD aktiv + Faces gefunden)
            if (self.arcface_enabled.get() and self.scrfd_enabled.get()
                    and face_boxes and "arcface" in self._active_ctx):
                try:
                    t0 = time.perf_counter()
                    for box, score, lm in face_boxes:
                        # Face crop aus Original-Frame
                        x1 = max(0, int(box[0] * fw))
                        y1 = max(0, int(box[1] * fh))
                        x2 = min(fw, int(box[2] * fw))
                        y2 = min(fh, int(box[3] * fh))

                        # 20% Margin
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
                            # Erstes (einziges) Output = Embedding
                            emb_key = self._output_names["arcface"][0]
                            embedding = outputs[emb_key].flatten()
                            embedding = normalize_arcface(embedding)

                            if self._face_db:
                                name, sim = match_face(
                                    embedding, self._face_db,
                                    threshold=self.arcface_thresh.get()
                                )
                            else:
                                name, sim = "Keine DB", 0.0

                            draw_name(annotated, box, name, sim, fh, fw)

                            # Shared State fuer push_to_talk
                            self._write_face_state(name, sim, len(face_boxes))

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

            # 3. YOLOv8m Person Detection (komplett uebersprungen wenn Face erkannt)
            if self.yolo_enabled.get() and "yolov8m" in self._active_ctx and not face_detected:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("yolov8m", input_rgb)
                    # On-Chip NMS: ein Output
                    out_key = self._output_names["yolov8m"][0]
                    persons = decode_yolov8_nms(
                        outputs[out_key],
                        class_id=0,
                        conf_thresh=self.yolo_conf.get()
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["yolov8m"] = 1.0 / dt if dt > 0 else 0

                    if persons:
                        draw_persons(annotated, persons, scale_x, scale_y)
                        # Guardian: Person sichtbar -> Interest erneuern + als Fund markieren
                        if self._moloch_has_control:
                            self._last_interesting_time = time.time()
                            self._takeover_found_something = True
                        # Feed to autonomous tracker (NUR wenn kein Face erkannt!)
                        # Face hat Prioritaet - Person nur als Fallback
                        if self._autonomous_mode and self._tracker and not face_fed_to_tracker:
                            try:
                                pixel_dets = []
                                for p in persons:
                                    bx = p["bbox"]  # normalized [0,1]
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
            if self.pose_enabled.get() and "pose" in self._active_ctx:
                try:
                    t0 = time.perf_counter()
                    outputs = self._run_model("pose", input_rgb)
                    poses = decode_yolov8_pose(
                        outputs, img_h=640, img_w=640,
                        conf_thresh=self.pose_conf.get(),
                        iou_thresh=self.pose_nms.get()
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["pose"] = 1.0 / dt if dt > 0 else 0

                    if poses:
                        draw_poses(annotated, poses, scale_x, scale_y)
                        # Feed to autonomous tracker (pose method)
                        # Enrich poses with has_face/has_torso from keypoints
                        if self._autonomous_mode and self._tracker:
                            try:
                                enriched = []
                                for p in poses:
                                    ep = dict(p)
                                    kpts = p.get("keypoints")
                                    if kpts is not None and len(kpts) >= 17:
                                        # Face: nose(0), eyes(1,2)
                                        face_vis = (float(kpts[0][2]) + float(kpts[1][2]) + float(kpts[2][2])) / 3
                                        if face_vis > 0.5:
                                            ep["has_face"] = True
                                            ep["face_confidence"] = face_vis
                                            # Face center from nose, normalized to [0,1]
                                            ep["face_center"] = (float(kpts[0][0]) / 640, float(kpts[0][1]) / 640)
                                        else:
                                            ep["has_face"] = False
                                            ep["face_confidence"] = 0
                                        # Torso: shoulders(5,6) + hips(11,12)
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

            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0

            with self._annotated_lock:
                self._annotated_frame = annotated

    # =========================================================================
    # Cross-Process NPU Coordination
    # =========================================================================

    def _pause_for_voice(self):
        """Pause inference - release VDevice so voice process can use NPU."""
        logger.info("[NPU_IPC] Voice requested - pausing vision...")
        self._update_status("NPU: Pausiert fuer Sprache...")

        # Remember active models
        self._paused_models = list(self._active_ctx.keys())

        # Unconfigure all models (close context managers)
        for name in list(self._active_ctx.keys()):
            self._unconfigure_model(name)

        # Release models and VDevice
        self._models.clear()
        if self._vdevice:
            try:
                del self._vdevice
            except Exception:
                pass
            self._vdevice = None

        # Release via HailoManager
        if self._hailo_manager:
            try:
                self._hailo_manager.release_vision()
            except Exception:
                pass

        import gc
        gc.collect()
        time.sleep(0.3)

        # Signal paused
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

        # Remove paused signal
        for path in [NPU_VISION_PAUSED]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        # Wait for device to be free
        time.sleep(0.5)

        # Re-acquire via HailoManager
        if self._hailo_manager:
            if not self._hailo_manager.acquire_for_vision(timeout=10.0):
                self._update_status("NPU nicht verfuegbar nach Voice!")
                self._npu_paused = False
                return

        # Recreate VDevice and models
        try:
            params = VDevice.create_params()
            self._vdevice = VDevice(params)

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

            # Reconfigure previously active models
            for name in self._paused_models:
                if name in self._models:
                    self._configure_model(name)

            self._npu_paused = False
            self._update_status("RTSP + NPU aktiv")
            logger.info("[NPU_IPC] Vision resumed successfully")
        except Exception as e:
            self._update_status(f"NPU Resume Fehler: {e}")
            logger.error(f"[NPU_IPC] Resume failed: {e}")
            self._npu_paused = False

    # =========================================================================
    # Display Loop (Tkinter Main Thread)
    # =========================================================================

    def _display_loop(self):
        """Zeige annotiertes Frame im Canvas (~15 FPS)."""
        if not self.running:
            return

        frame = None
        with self._annotated_lock:
            if self._annotated_frame is not None:
                frame = self._annotated_frame

        if frame is None:
            # Fallback: Raw Frame ohne Annotationen
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame

        if frame is not None:
            try:
                # OpenCV BGR -> RGB fuer Tkinter
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self._photo = ImageTk.PhotoImage(image=img)
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            except Exception:
                pass

        self._display_after_id = self.root.after(
            1000 // self.DISPLAY_FPS, self._display_loop)

    def _update_fps_display(self):
        """FPS Labels updaten (alle 500ms)."""
        if not self.running:
            return

        with self._fps_lock:
            fps = self._fps.copy()

        # Total
        if fps["total"] > 0:
            self.total_fps_label.configure(text=f"Pipeline: {fps['total']:.0f} FPS")

        # Pro Modell
        for key, label_attr in [("scrfd", "_scrfd_fps_label"),
                                ("arcface", "_arcface_fps_label"),
                                ("yolov8m", "_yolov8m_fps_label"),
                                ("pose", "_pose_fps_label")]:
            label = getattr(self, label_attr, None)
            if label:
                if fps[key] > 0:
                    label.configure(text=f"{fps[key]:.0f} FPS")
                else:
                    label.configure(text="--- FPS")

        self.root.after(500, self._update_fps_display)

    # =========================================================================
    # Model Toggle Callback
    # =========================================================================

    def _on_model_toggle(self, model_key):
        """Wird aufgerufen wenn ein Modell-Toggle geaendert wird."""
        # ArcFace braucht SCRFD
        if model_key == "arcface" and self.arcface_enabled.get():
            if not self.scrfd_enabled.get():
                self.scrfd_enabled.set(True)
                self._on_model_toggle("scrfd")

        # SCRFD aus -> ArcFace auch aus
        if model_key == "scrfd" and not self.scrfd_enabled.get():
            self.arcface_enabled.set(False)

        # Mapping model_key -> enabled var
        toggle_map = {
            "scrfd": self.scrfd_enabled, "arcface": self.arcface_enabled,
            "yolov8m": self.yolo_enabled, "pose": self.pose_enabled,
        }

        enabled = toggle_map.get(model_key, None)
        if enabled is None:
            return

        if enabled.get():
            # Hardware-Limit: FIFO-Rotation bei max 2 Modellen
            evicted = False
            if len(self._active_ctx) >= MAX_CONCURRENT_MODELS:
                # Aeltestes Modell rauswerfen (FIFO)
                oldest = self._model_order[0] if self._model_order else None
                if oldest:
                    logger.info(
                        f"NPU-Limit: {oldest} wird deaktiviert fuer {model_key} (FIFO)"
                    )
                    self._update_status(f"Wechsel: {oldest} -> {model_key}...")
                    # Checkbox des rausgeworfenen Modells deaktivieren
                    evict_map = {
                        "scrfd": self.scrfd_enabled, "arcface": self.arcface_enabled,
                        "yolov8m": self.yolo_enabled, "pose": self.pose_enabled,
                    }
                    evict_var = evict_map.get(oldest)
                    if evict_var:
                        evict_var.set(False)
                    # SCRFD raus -> ArcFace auch raus
                    if oldest == "scrfd" and self.arcface_enabled.get():
                        self.arcface_enabled.set(False)
                        self._unconfigure_model("arcface")
                    self._unconfigure_model(oldest)
                    evicted = True

            # Configure im Hintergrund (erster Aufruf ~400ms)
            if not evicted:
                self._update_status(f"Konfiguriere {model_key}...")
            def do_cfg(mk=model_key, need_delay=evicted):
                try:
                    if need_delay:
                        time.sleep(0.2)  # 200ms NPU-Freigabe nach Eviction
                    self._configure_model(mk)
                    self._update_status("RTSP + NPU aktiv")
                except Exception as e:
                    import traceback
                    crash_log = os.path.expanduser("~/moloch/logs/panel_crash.log")
                    crash_info = (
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"TOGGLE THREAD CRASH: {mk}\n{traceback.format_exc()}\n"
                    )
                    logger.error(crash_info)
                    try:
                        with open(crash_log, "a", encoding="utf-8") as f:
                            f.write(crash_info)
                    except Exception:
                        pass
                    self._update_status(f"CRASH: {mk}")
            threading.Thread(target=do_cfg, daemon=True).start()
        else:
            # Unconfigure + FPS Reset
            self._unconfigure_model(model_key)
            with self._fps_lock:
                fps_key = model_key if model_key != "yolov8m" else "yolov8m"
                self._fps[fps_key] = 0

    # =========================================================================
    # Aktionen
    # =========================================================================

    def _kill_push_to_talk(self):
        """push_to_talk.py killen (blockiert /dev/hailo0)."""
        def do_kill():
            try:
                result = subprocess.run(
                    ["pkill", "-f", "push_to_talk"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self._update_status("push_to_talk gekillt")
                else:
                    self._update_status("push_to_talk nicht gefunden")
            except Exception as e:
                self._update_status(f"Kill Fehler: {e}")

        threading.Thread(target=do_kill, daemon=True).start()

    def _save_snapshot(self):
        """Aktuelles annotiertes Frame als Snapshot speichern."""
        frame = None
        with self._annotated_lock:
            if self._annotated_frame is not None:
                frame = self._annotated_frame.copy()

        if frame is None:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

        if frame is None:
            self._update_status("Kein Frame fuer Snapshot")
            return

        snap_dir = os.path.expanduser("~/moloch/snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(snap_dir, f"hailo_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._update_status(f"Snapshot: {path}")

    # =========================================================================
    # eWeLink Cloud Controls
    # =========================================================================

    def _connect_cloud(self):
        """Connect to eWeLink cloud in background."""
        try:
            self._cloud = CloudController()
            self._cloud.start()
            if self._cloud.connected:
                logger.info("eWeLink Cloud verbunden")
                # Smart Tracking beim Start AN - Kamera scannt selbststaendig
                try:
                    self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                    self._set_smart_tracking_state(True)
                    logger.info("Smart Tracking aktiviert - Kamera scannt autonom (Tentakel-Modus)")
                except Exception:
                    pass
                self.root.after(0, lambda: self._cloud_status_label.config(
                    text="Cloud: verbunden", fg="#00ff88"))
                self.root.after(500, self._refresh_cloud_params)
            else:
                logger.warning("eWeLink Cloud nicht erreichbar")
                self.root.after(0, lambda: self._cloud_status_label.config(
                    text="Cloud: FEHLER", fg="#ff4444"))
        except Exception as e:
            logger.error(f"Cloud connect: {e}")
            self.root.after(0, lambda: self._cloud_status_label.config(
                text=f"Cloud: {e}", fg="#ff4444"))

    def _set_smart_tracking_state(self, value: bool):
        """Einziger Schreibzugriff auf _smart_tracking_on (thread-safe)."""
        with self._st_lock:
            self._smart_tracking_on = value
        btn_text = "Smart Tracking: AN" if value else "Smart Tracking: AUS"
        btn_bg = "#884400" if value else "#2a2a4e"
        self.root.after(0, lambda: self._smart_tracking_btn.config(text=btn_text, bg=btn_bg))

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

    def _set_cloud_led(self):
        self._cloud_run("set_status_led", self._led_var.get())

    def _set_cloud_night(self):
        mode_map = {"Aus": "day", "Auto": "auto", "An": "night"}
        self._cloud_run("set_night", mode_map.get(self._night_var.get(), "day"))

    def _set_cloud_flip(self):
        self._cloud_run("set_screen_flip", self._flip_var.get())

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

    def _trigger_alarm(self):
        """3-Sekunden Alarm ausloesen."""
        if not self._cloud or not self._cloud.connected:
            self._update_status("Cloud nicht verbunden")
            return
        def alarm_cycle():
            self._cloud.run(self._cloud.bridge.set_alarm(True))
            time.sleep(3)
            self._cloud.run(self._cloud.bridge.set_alarm(False))
            self._update_status("Alarm beendet")
        self._update_status("ALARM aktiv (3s)")
        threading.Thread(target=alarm_cycle, daemon=True).start()

    def _trigger_calibration(self):
        """PTZ Kalibrierung mit Bestaetigung."""
        if messagebox.askyesno("PTZ Kalibrierung",
                               "Kamera bewegt sich durch den vollen Bereich!\n\nFortfahren?"):
            self._cloud_run("trigger_ptz_calibration")
            self._update_status("Kalibrierung gestartet")

    def _refresh_cloud_params(self):
        """Cloud-Parameter laden und UI synchronisieren."""
        if not self._cloud or not self._cloud.connected:
            return
        def do_refresh():
            params = self._cloud.run(self._cloud.bridge.get_device_params())
            if params:
                self.root.after(0, lambda: self._apply_cloud_params(params))
        threading.Thread(target=do_refresh, daemon=True).start()

    def _apply_cloud_params(self, params):
        """Cloud-Parameter auf UI-Widgets anwenden."""
        try:
            if "nightVision" in params:
                nv_map = {0: "Aus", 1: "Auto", 2: "An"}
                self._night_var.set(nv_map.get(params["nightVision"], "Aus"))
            if "smartTraceEnable" in params:
                self._set_smart_tracking_state(bool(params["smartTraceEnable"]))
            if "screenFlip" in params:
                self._flip_var.set(bool(params["screenFlip"]))
            if "sledOnline" in params:
                self._led_var.set(params["sledOnline"] == "on")
            self._cloud_status_label.config(text="Cloud: verbunden", fg="#00ff88")
        except Exception as e:
            logger.error(f"Apply cloud params: {e}")

    def _ptz_goto(self, pan, tilt):
        """PTZ zu bestimmter Position fahren."""
        def do_move():
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                cam.move_absolute(pan, tilt, speed=1.0)
                self._update_status(f"PTZ -> Pan={pan} Tilt={tilt}")
            except Exception as e:
                self._update_status(f"PTZ Fehler: {e}")
        threading.Thread(target=do_move, daemon=True).start()

    def _reload_face_db(self):
        """Face-DB neu laden (nach Enrollment)."""
        self._face_db = load_face_db(FACE_DB_PATH)
        n = len(self._face_db)
        names = ", ".join(self._face_db.keys()) if self._face_db else "leer"
        self._update_status(f"Face-DB: {n} Personen ({names})")

    def _write_face_state(self, name, similarity, person_count):
        """Schreibe Face-Recognition-State fuer IPC mit push_to_talk."""
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "timestamp": time.time(),
                "source": "hailo_panel"
            }
            with open(FACE_STATE_PATH, "w") as f:
                json.dump(state, f)
        except Exception:
            pass

    def _announce_person(self, name):
        """TTS-Ansage bei Gesichtserkennung (laeuft in eigenem Thread)."""
        try:
            from core.personality import get_personality_engine, MolochEvent
            engine = get_personality_engine()
            engine.speak_event(MolochEvent.PERSON_KNOWN, context={"name": name})
        except Exception as e:
            logger.error(f"TTS Ansage Fehler: {e}")

    # =========================================================================
    # Guardian Mode: Tentakel-Logik (Smart Tracking <-> MOLOCH Takeover)
    # =========================================================================

    def _moloch_takeover(self, reason: str):
        """MOLOCH uebernimmt: ST AUS -> NPU Modelle AN -> AUTONOM Tracker."""
        with self._transition_lock:
            if self._moloch_has_control or not self._guardian_mode or self._transitioning:
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
                # 1. ST AUS (muss zuerst - kaempft sonst mit Tracker)
                # Ohne ST AUS kaempfen Tracker und Smart Tracking um die Kamera!
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
                            logger.warning(f"[TENTAKEL] ST AUS fehlgeschlagen (Versuch {attempt+1}/3): {e}")
                            time.sleep(0.5)
                if not st_off:
                    logger.error("[TENTAKEL] ST AUS nach 3 Versuchen fehlgeschlagen - Takeover ABBRUCH")
                    self._moloch_has_control = False
                    self._update_status("Takeover abgebrochen: ST nicht erreichbar")
                    return

                # 2. NPU Modelle direkt konfigurieren (kein _on_model_toggle Umweg)
                self._update_status("Takeover: NPU Modelle...")
                logger.info("[TENTAKEL] Aktiviere NPU Modelle (SCRFD + YOLOv8m)")
                self.root.after(0, lambda: self.scrfd_enabled.set(True))
                self._configure_model("scrfd")
                time.sleep(0.2)  # Hailo Hardware-Minimum zwischen Konfigurationen
                self.root.after(0, lambda: self.yolo_enabled.set(True))
                self._configure_model("yolov8m")

                # 3. Tracker AN
                self.root.after(0, self._enable_autonomous)
                self._update_status(f"MOLOCH: {reason}")
                logger.info(f"[TENTAKEL] Takeover komplett: {reason}")
            except Exception as e:
                logger.error(f"[TENTAKEL] Takeover Fehler: {e}")
                self._moloch_has_control = False
            finally:
                self._transitioning = False

        threading.Thread(target=do_takeover, daemon=True).start()

    def _moloch_release(self):
        """MOLOCH gibt zurueck: Tracker STOP -> ST AN -> Aufraumen."""
        with self._transition_lock:
            if not self._moloch_has_control or self._transitioning:
                return
            self._transitioning = True
        try:
            logger.info("[TENTAKEL] MOLOCH gibt Kamera zurueck an Smart Tracking")
            self._moloch_has_control = False
            self._takeover_reason = ""
            self._search_start_time = 0

            # 1. Tracker SOFORT stoppen (thread-safe, kein root.after noetig)
            self._autonomous_mode = False
            if self._tracker:
                self._tracker.disable()
            logger.info("[TENTAKEL] Tracker gestoppt")

            # 2. Smart Tracking SOFORT AN (minimaler Gap!)
            if self._cloud and self._cloud.connected:
                try:
                    self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                    self._set_smart_tracking_state(True)
                    logger.info("[TENTAKEL] Smart Tracking wiederhergestellt")
                except Exception:
                    pass

            # 3. UI + Models OFF im Main Thread (nicht zeitkritisch)
            def cleanup():
                try:
                    self._auto_btn.config(text="MANUELL", bg="#2a2a4e")
                    self._tracker_state_label.config(text="", fg="#888888")
                    self._all_models_off()
                except Exception as e:
                    logger.error(f"[TENTAKEL] Cleanup: {e}")
            self.root.after(0, cleanup)

            # Face mode reset
            self._face_mode_active = False
            self._face_seen_count = 0
            self._face_lost_time = 0

            # Position-Tracking zuruecksetzen
            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0

            # Progressive Backoff (sanfter: 1.5x statt 2x, max 180s statt 600s)
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
        if not self._guardian_mode or self._transitioning:
            return
        # Safety: verwaister autonomer Modus (Tracker laeuft ohne Moloch-Kontrolle)
        # NICHT bei manueller Aktivierung (User hat absichtlich AUTONOM gedrueckt)
        if self._autonomous_mode and not self._moloch_has_control and not self._manual_autonomous:
            logger.warning("[SAFETY] Orphaned autonomous mode detected - disabling")
            self.root.after(0, self._disable_autonomous)
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
            from core.mpo.autonomous_tracker import TrackerState
            if self._tracker.state == TrackerState.SEARCHING:
                if self._search_start_time == 0:
                    self._search_start_time = now
                elif now - self._search_start_time > self.SEARCH_TIMEOUT:
                    logger.info(f"[TENTAKEL] Search timeout ({self.SEARCH_TIMEOUT}s) - zurueckgeben")
                    threading.Thread(target=self._moloch_release, daemon=True).start()
                    return
            else:
                self._search_start_time = 0

    def _update_cam_status(self):
        """Kamera-Kontrolle Status updaten (alle 3s) + Tentakel-Bewegungserkennung."""
        if not self.running:
            return

        def do_check():
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

                        # Tentakel: Kamera-Bewegung erkennen (Smart Tracking hat was gesehen)
                        if (self._guardian_mode and self._smart_tracking_on
                                and not self._moloch_has_control
                                and not self._transitioning):
                            if self._guardian_last_pan is not None:
                                delta = abs(pan - self._guardian_last_pan) + abs(tilt - self._guardian_last_tilt)
                                if delta > self._guardian_move_thresh:
                                    self._guardian_move_count += 1
                                    logger.info(f"[TENTAKEL] Bewegung {self._guardian_move_count}/{self._guardian_move_required} delta={delta:.1f}")
                                    # Pre-Load: Bei erster Bewegung Modelle im Hintergrund laden
                                    if self._guardian_move_count == 1 and time.time() >= self._takeover_cooldown_until:
                                        def _preload():
                                            try:
                                                logger.info("[TENTAKEL] Pre-Load: SCRFD + YOLOv8m laden...")
                                                self._configure_model("scrfd")
                                                time.sleep(0.2)
                                                self._configure_model("yolov8m")
                                                logger.info("[TENTAKEL] Pre-Load: Modelle ready!")
                                            except Exception as e:
                                                logger.error(f"[TENTAKEL] Pre-Load Fehler: {e}")
                                        threading.Thread(target=_preload, daemon=True).start()
                                    if self._guardian_move_count >= self._guardian_move_required:
                                        # Cooldown pruefen
                                        if time.time() >= self._takeover_cooldown_until:
                                            logger.info(f"[TENTAKEL] Sustained movement ({self._guardian_move_count}x) -> MOLOCH uebernimmt")
                                            self._guardian_move_count = 0
                                            self._moloch_takeover("Kamera trackt etwas")
                                        else:
                                            remaining = self._takeover_cooldown_until - time.time()
                                            logger.info(f"[TENTAKEL] Cooldown aktiv, noch {remaining:.0f}s")
                                            self._guardian_move_count = 0
                                else:
                                    # Kamera steht kurz still -> Counter dekrementieren (nicht sofort reset)
                                    self._guardian_move_count = max(0, self._guardian_move_count - 1)
                            self._guardian_last_pan = pan
                            self._guardian_last_tilt = tilt
            except Exception:
                pass

            # UI Update im Main Thread
            smart = "AUS" if not self._smart_tracking_on else "AN"
            onvif_str = "OK" if onvif_ok else "---"

            if self._moloch_has_control:
                ctrl_text = f"MOLOCH: {self._takeover_reason[:20]}"
                ctrl_color = "#ff4444"
                bg = "#3a1a1a"
                border = "#ff4444"
            elif self._smart_tracking_on:
                ctrl_text = "TENTAKEL SCANNT"
                ctrl_color = "#00d4ff"
                bg = "#1a2a3a"
                border = "#00d4ff"
            elif onvif_ok:
                ctrl_text = "MANUELL"
                ctrl_color = "#00ff88"
                bg = "#1a3a1a"
                border = "#00ff88"
            else:
                ctrl_text = "OFFLINE"
                ctrl_color = "#ffaa00"
                bg = "#3a3a1a"
                border = "#ffaa00"

            self.root.after(0, lambda: self._apply_cam_status(
                ctrl_text, ctrl_color, bg, border, smart, onvif_str, ptz_text))

        # Guardian timeout check
        self._check_guardian_timeout()

        threading.Thread(target=do_check, daemon=True).start()
        self.root.after(3000, self._update_cam_status)

    def _apply_cam_status(self, ctrl_text, ctrl_color, bg, border, smart, onvif, ptz):
        """Kamera-Status Labels aktualisieren (Main Thread)."""
        try:
            self._cam_status_frame.config(bg=bg, highlightbackground=border)
            self._cam_control_label.config(text=ctrl_text, fg=ctrl_color, bg=bg)
            self._cam_detail_label.config(
                text=f"Smart Tracking: {smart} | ONVIF: {onvif}", bg=bg)
            self._cam_ptz_label.config(text=f"PTZ: {ptz}", bg=bg)
        except Exception:
            pass

    def _enable_autonomous(self):
        """Explizit AUTONOM aktivieren (idempotent - kein Toggle!)."""
        if self._autonomous_mode:
            logger.debug("[AUTONOM] Already enabled, skip")
            return
        self._auto_btn.config(state=tk.DISABLED, text="Starte...")
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
                    self.root.after(0, lambda: self._auto_btn.config(
                        state=tk.NORMAL, text="MANUELL", bg="#2a2a4e"))
                    return
                self._tracker.set_camera(cam)
                cam.set_mode(ControlMode.AUTONOMOUS)
                if not self._tracker._running:
                    self._tracker.start()
                self._tracker.enable()
                self._autonomous_mode = True
                self._update_status("Modus: AUTONOM - MOLOCH sucht...")
                logger.info("Switched to AUTONOMOUS mode")
                self.root.after(0, lambda: self._auto_btn.config(
                    state=tk.NORMAL, text="AUTONOM", bg="#006622"))
                # Tracker State Updates starten
                self.root.after(500, self._update_tracker_state)
            except Exception as e:
                logger.error(f"Autonomous start failed: {e}")
                self._update_status(f"AUTONOM Fehler: {e}")
                self.root.after(0, lambda: self._auto_btn.config(
                    state=tk.NORMAL, text="MANUELL", bg="#2a2a4e"))
        threading.Thread(target=do_start, daemon=True).start()

    def _disable_autonomous(self):
        """Explizit AUTONOM deaktivieren (idempotent - kein Toggle!)."""
        if not self._autonomous_mode:
            logger.debug("[AUTONOM] Already disabled, skip")
            return
        self._autonomous_mode = False
        if self._tracker:
            self._tracker.disable()
        self._auto_btn.config(text="MANUELL", bg="#2a2a4e")
        self._tracker_state_label.config(text="", fg="#888888")
        self._update_status("Modus: MANUELL")
        logger.info("Switched to MANUAL mode")

    def _toggle_autonomous(self):
        """Zwischen MANUELL und AUTONOM umschalten (NUR fuer GUI Button)."""
        if self._autonomous_mode:
            self._disable_autonomous()
            self._moloch_has_control = False
            self._manual_autonomous = False
            self._takeover_reason = ""
            # Guardian: Smart Tracking wieder AN
            if self._guardian_mode and self._cloud and self._cloud.connected:
                def restore_st():
                    try:
                        self._cloud.run(self._cloud.bridge.set_smart_tracking(True))
                        self._set_smart_tracking_state(True)
                        logger.info("[TENTAKEL] Smart Tracking wiederhergestellt")
                    except Exception:
                        pass
                threading.Thread(target=restore_st, daemon=True).start()
        else:
            self._manual_autonomous = True  # User hat manuell aktiviert
            self._enable_autonomous()
            # Smart Tracking AUS wenn manuell AUTONOM aktiviert
            if self._cloud and self._cloud.connected:
                def disable_st():
                    try:
                        self._cloud.run(self._cloud.bridge.set_smart_tracking(False))
                        self._set_smart_tracking_state(False)
                    except Exception:
                        pass
                threading.Thread(target=disable_st, daemon=True).start()

    def _update_tracker_state(self):
        """Tracker-State im GUI anzeigen."""
        if not self.running or not self._autonomous_mode:
            return
        if self._tracker:
            state = self._tracker.state.value.upper()
            colors = {
                "TRACKING": "#00ff88",
                "SEARCHING": "#ffaa00",
                "LOCKED": "#00ff88",
                "IDLE": "#888888",
                "DWELL": "#aaaaff",
                "FROZEN": "#ff4444",
            }
            color = colors.get(state, "#888888")
            self._tracker_state_label.config(text=f"Tracker: {state}", fg=color)
        self.root.after(500, self._update_tracker_state)

    def _ptz_move(self, direction):
        """Kamera in eine Richtung bewegen (ONVIF AbsoluteMove)."""
        if self._autonomous_mode and self._tracker and self._tracker.is_running:
            return  # Tracker hat Vorrang!
        # Step-Groesse aus Speed Slider
        step = self._ptz_speed_var.get() if hasattr(self, '_ptz_speed_var') else 15.0
        PAN_STEP = step
        TILT_STEP = step * 0.67  # Tilt etwas langsamer

        def do_move():
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                if not cam.is_connected:
                    self._update_status("Kamera nicht verbunden!")
                    return

                pos = cam.get_position()
                if not pos:
                    self._update_status("PTZ Position nicht lesbar!")
                    return

                pan, tilt = pos.pan, pos.tilt

                if direction == "left":
                    # Pan invertiert! +Pan = links
                    pan += PAN_STEP
                elif direction == "right":
                    pan -= PAN_STEP
                elif direction == "up":
                    tilt += TILT_STEP
                elif direction == "down":
                    tilt -= TILT_STEP
                elif direction == "home":
                    pan, tilt = 0.0, 0.0

                # Limits
                pan = max(-168.4, min(174.4, pan))
                tilt = max(-78.8, min(101.3, tilt))

                result = cam.move_absolute(pan, tilt)
                if result:
                    self._update_status(f"PTZ: {pan:.1f}, {tilt:.1f}")
                else:
                    self._update_status("PTZ Bewegung fehlgeschlagen")
            except Exception as e:
                self._update_status(f"PTZ Fehler: {e}")

        threading.Thread(target=do_move, daemon=True).start()

    def _swap_to_arcface(self):
        """YOLOv8m -> ArcFace swap (MOLOCH will wissen WER da ist)."""
        with self._swap_lock:
            if self._face_mode_active or self._swapping_models:
                return
            self._swapping_models = True
        try:
            logger.info("[TENTAKEL] Face erkannt -> Swap YOLOv8m -> ArcFace")
            self._unconfigure_model("yolov8m")
            self.root.after(0, lambda: self.yolo_enabled.set(False))
            time.sleep(0.2)
            self._configure_model("arcface")
            self.root.after(0, lambda: self.arcface_enabled.set(True))
            self._face_mode_active = True
            logger.info("[TENTAKEL] ArcFace aktiv - Gesichtserkennung laeuft")
        except Exception as e:
            logger.error(f"Swap to ArcFace failed: {e}")
        finally:
            with self._swap_lock:
                self._swapping_models = False

    def _swap_to_yolov8m(self):
        """ArcFace -> YOLOv8m swap (kein Face mehr, braucht Person-Detection)."""
        with self._swap_lock:
            if not self._face_mode_active or self._swapping_models:
                return
            self._swapping_models = True
        try:
            logger.info("[TENTAKEL] Face verloren -> Swap ArcFace -> YOLOv8m")
            self._unconfigure_model("arcface")
            self.root.after(0, lambda: self.arcface_enabled.set(False))
            time.sleep(0.2)
            self._configure_model("yolov8m")
            self.root.after(0, lambda: self.yolo_enabled.set(True))
            self._face_mode_active = False
            logger.info("[TENTAKEL] YOLOv8m aktiv - Person-Detection laeuft")
        except Exception as e:
            logger.error(f"Swap to YOLOv8m failed: {e}")
        finally:
            with self._swap_lock:
                self._swapping_models = False

    def _all_models_off(self):
        """Alle Modelle deaktivieren und unconfigurieren."""
        self._face_mode_active = False
        self._face_seen_count = 0
        self._face_lost_time = 0
        self.scrfd_enabled.set(False)
        self.arcface_enabled.set(False)
        self.yolo_enabled.set(False)
        self.pose_enabled.set(False)
        for name in list(self._active_ctx.keys()):
            self._unconfigure_model(name)
        with self._fps_lock:
            self._fps = {"scrfd": 0, "arcface": 0, "yolov8m": 0, "pose": 0, "total": 0}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def run(self):
        """GUI starten."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Sauberes Herunterfahren."""
        self.running = False

        # Tracker stoppen
        if self._tracker:
            try:
                self._tracker.stop()
            except Exception:
                pass

        if self._display_after_id:
            self.root.after_cancel(self._display_after_id)

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
        for path in [NPU_VISION_PAUSED]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        self.root.destroy()


if __name__ == "__main__":
    app = HailoControlPanel()
    app.run()
