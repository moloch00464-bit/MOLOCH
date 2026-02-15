#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hailo-10H Control Panel - GUI Adapter
=====================================================

Duennes GUI-Frontend fuer MolochService (Backend).
Keine eigene NPU-Logik - alles via Service.

3-Thread Architektur:
  1. Service: RTSP + Inference + CamStatus (Hintergrund-Threads)
  2. Tkinter Main: Display + Controls
  3. Observer: Service -> GUI Updates

Author: M.O.L.O.C.H. System
"""

import os
import sys
import time
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

sys.path.insert(0, os.path.expanduser("~/moloch"))

from core.moloch_service import MolochService

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("HailoPanel")


class HailoControlPanel:
    """M.O.L.O.C.H. Hailo-10H Control Panel - GUI Adapter."""

    DISPLAY_FPS = 15

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H. Hailo-10H Control Panel")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        # GUI state
        self.running = True
        self._display_after_id = None
        self._photo = None

        # Service (Backend)
        self.service = MolochService()
        self.service.add_observer(self._on_service_event)

        # Modell-Toggles (BooleanVar fuer GUI-Binding)
        self.scrfd_enabled = tk.BooleanVar(value=False)
        self.arcface_enabled = tk.BooleanVar(value=False)
        self.yolo_enabled = tk.BooleanVar(value=False)
        self.pose_enabled = tk.BooleanVar(value=False)

        # Threshold Vars (GUI-Binding) -> sync zu Service
        self.scrfd_conf = tk.DoubleVar(value=0.40)
        self.scrfd_nms = tk.DoubleVar(value=0.40)
        self.arcface_thresh = tk.DoubleVar(value=0.60)
        self.yolo_conf = tk.DoubleVar(value=0.50)
        self.pose_conf = tk.DoubleVar(value=0.50)
        self.pose_nms = tk.DoubleVar(value=0.70)

        # Threshold sync: GUI Slider -> Service plain float
        threshold_map = [
            (self.scrfd_conf, 'scrfd_conf_val'),
            (self.scrfd_nms, 'scrfd_nms_val'),
            (self.arcface_thresh, 'arcface_thresh_val'),
            (self.yolo_conf, 'yolo_conf_val'),
            (self.pose_conf, 'pose_conf_val'),
            (self.pose_nms, 'pose_nms_val'),
        ]
        for var, attr in threshold_map:
            var.trace_add("write", lambda *_, a=attr, v=var: setattr(self.service, a, v.get()))

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
    # UI Layout
    # =========================================================================

    def _build_ui(self):
        """Baue komplettes UI - 3-Bereich Layout: Preview+Kamera links, Modelle rechts."""
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

        # ---- LINKS: Preview + Kamera-Kontrolle ----
        left_frame = ttk.Frame(middle)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Live Preview
        self.preview_canvas = tk.Canvas(
            left_frame, width=self.service.PREVIEW_W, height=self.service.PREVIEW_H,
            bg="#000000", highlightthickness=1, highlightbackground="#333"
        )
        self.preview_canvas.pack(pady=(0, 3))

        # --- KAMERA + PTZ unter dem Preview ---
        cam_bottom = tk.Frame(left_frame, bg="#1a1a2e")
        cam_bottom.pack(fill=tk.X)

        # Links: Status + Autonomie
        cam_left = tk.Frame(cam_bottom, bg="#1a1a2e")
        cam_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

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

        # AUTONOM/MANUELL Button
        self._auto_btn = tk.Button(
            cam_left, text="MANUELL", bg="#2a2a4e", fg="white",
            font=("Helvetica", 10, "bold"),
            command=lambda: self.service.toggle_autonomous_manual())
        self._auto_btn.pack(fill=tk.X, pady=(3, 0))

        self._tracker_state_label = tk.Label(
            cam_left, text="", font=("Courier", 8), fg="#888888", bg="#1a1a2e")
        self._tracker_state_label.pack()

        # Smart Tracking Button
        self._smart_tracking_btn = tk.Button(
            cam_left, text="Smart Tracking: AUS", bg="#2a2a4e", fg="white",
            font=("Helvetica", 8),
            command=lambda: self.service._toggle_smart_tracking())
        self._smart_tracking_btn.pack(fill=tk.X, pady=(2, 0))

        # Mitte: PTZ Steuerkreuz + Speed
        cam_mid = tk.Frame(cam_bottom, bg="#1a1a2e")
        cam_mid.pack(side=tk.LEFT, padx=5)

        btn_cfg = dict(width=3, font=("Helvetica", 11, "bold"),
                       bg="#2a2a4e", fg="white", activebackground="#4a4a6e")

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

        # LED + Flip
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

        self._build_model_section(
            ctrl_frame, "SCRFD Face", self.scrfd_enabled,
            "scrfd", [("Conf", self.scrfd_conf, 0.1, 0.9),
                      ("NMS", self.scrfd_nms, 0.1, 0.9)])

        self._build_model_section(
            ctrl_frame, "ArcFace", self.arcface_enabled,
            "arcface", [("Thresh", self.arcface_thresh, 0.3, 0.9)])

        self._build_model_section(
            ctrl_frame, "YOLOv8m Person", self.yolo_enabled,
            "yolov8m", [("Conf", self.yolo_conf, 0.1, 0.9)])

        self._build_model_section(
            ctrl_frame, "YOLOv8s Pose", self.pose_enabled,
            "pose", [("Conf", self.pose_conf, 0.1, 0.9),
                     ("NMS", self.pose_nms, 0.1, 0.9)])

        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Aktionen
        act_row1 = tk.Frame(ctrl_frame, bg="#1a1a2e")
        act_row1.pack(fill=tk.X, pady=1)
        tk.Button(act_row1, text="PTT killen", bg="#ff4444", fg="white",
                  font=("Helvetica", 8, "bold"),
                  command=self._kill_push_to_talk).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        tk.Button(act_row1, text="Snapshot", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=self._save_snapshot).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        act_row2 = tk.Frame(ctrl_frame, bg="#1a1a2e")
        act_row2.pack(fill=tk.X, pady=1)
        tk.Button(act_row2, text="Alle AUS", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=lambda: self.service._all_models_off()).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        tk.Button(act_row2, text="Face-DB", bg="#2a2a4e", fg="white",
                  font=("Helvetica", 8),
                  command=lambda: self.service._reload_face_db()).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

    def _build_model_section(self, parent, title, enabled_var, model_key, sliders):
        """Baue eine Modell-Section mit Toggle, FPS, Slidern."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

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
    # Service Observer
    # =========================================================================

    def _on_service_event(self, event, data):
        """Service -> GUI Updates (thread-safe via root.after)."""
        if event == "status":
            self.root.after(0, lambda: self.status_label.configure(text=data.get("text", "")))

        elif event == "cam_status":
            mode = data.get("mode", "offline")
            color_map = {
                "moloch":   ("#ff4444", "#3a1a1a", "#ff4444"),
                "tentakel": ("#00d4ff", "#1a2a3a", "#00d4ff"),
                "manual":   ("#00ff88", "#1a3a1a", "#00ff88"),
                "offline":  ("#ffaa00", "#3a3a1a", "#ffaa00"),
            }
            color, bg, border = color_map.get(mode, ("#ffaa00", "#3a3a1a", "#ffaa00"))
            self.root.after(0, lambda: self._apply_cam_status(
                data.get("ctrl_text", ""), color, bg, border,
                data.get("smart", ""), data.get("onvif", ""), data.get("ptz", "")))

        elif event == "model_toggle":
            var_map = {"scrfd": self.scrfd_enabled, "arcface": self.arcface_enabled,
                       "yolov8m": self.yolo_enabled, "pose": self.pose_enabled}
            for key, val in data.items():
                var = var_map.get(key)
                if var:
                    self.root.after(0, lambda v=var, s=val: v.set(s))

        elif event == "auto_mode":
            state = data.get("state", "")
            if state == "active":
                self.root.after(0, lambda: self._auto_btn.config(
                    state=tk.NORMAL, text="AUTONOM", bg="#006622"))
                self.root.after(500, self._update_tracker_state)
            elif state == "disabled":
                self.root.after(0, lambda: [
                    self._auto_btn.config(text="MANUELL", bg="#2a2a4e"),
                    self._tracker_state_label.config(text="", fg="#888888")])
            elif state == "starting":
                self.root.after(0, lambda: self._auto_btn.config(
                    state=tk.DISABLED, text="Starte..."))
            elif state == "failed":
                self.root.after(0, lambda: self._auto_btn.config(
                    state=tk.NORMAL, text="MANUELL", bg="#2a2a4e"))

        elif event == "smart_tracking":
            on = data.get("on", False)
            text = "Smart Tracking: AN" if on else "Smart Tracking: AUS"
            bg = "#884400" if on else "#2a2a4e"
            self.root.after(0, lambda: self._smart_tracking_btn.config(text=text, bg=bg))

        elif event == "cloud_status":
            if data.get("connected"):
                self.root.after(0, lambda: self._cloud_status_label.config(
                    text="Cloud: verbunden", fg="#00ff88"))
                self.root.after(500, self._refresh_cloud_params)
            else:
                err = data.get("error", "FEHLER")
                self.root.after(0, lambda: self._cloud_status_label.config(
                    text=f"Cloud: {err}", fg="#ff4444"))

    # =========================================================================
    # Initialization
    # =========================================================================

    def _start_init_thread(self):
        """Service im Hintergrund initialisieren."""
        # Headless Service stoppen (kaempft sonst um Hailo Device)
        subprocess.run(
            ["sudo", "systemctl", "stop", "moloch.service"],
            timeout=5, capture_output=True
        )
        # Stale IPC-Files aufraumen (tote Voice-Prozesse)
        for ipc_path in ["/tmp/moloch_npu_voice_request", "/tmp/moloch_npu_vision_paused"]:
            try:
                os.unlink(ipc_path)
            except FileNotFoundError:
                pass

        def do_init():
            try:
                self.service.init()
                self.service.start(blocking=False)

                # Display + FPS Loops starten (GUI-only)
                self.root.after(100, self._display_loop)
                self.root.after(500, self._update_fps_display)

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Init failed: {err_msg}")
                self.root.after(0, lambda msg=err_msg: self.status_label.configure(
                    text=f"Init Fehler: {msg}"))

        threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()

    # =========================================================================
    # Display Loop
    # =========================================================================

    def _display_loop(self):
        """Zeige annotiertes Frame im Canvas (~15 FPS)."""
        if not self.running:
            return

        frame = None
        with self.service._annotated_lock:
            if self.service._annotated_frame is not None:
                frame = self.service._annotated_frame

        if frame is None:
            with self.service._frame_lock:
                if self.service._latest_frame is not None:
                    frame = self.service._latest_frame

        if frame is not None:
            try:
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

        with self.service._fps_lock:
            fps = self.service._fps.copy()

        if fps["total"] > 0:
            self.total_fps_label.configure(text=f"Pipeline: {fps['total']:.0f} FPS")

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
    # Model Toggle (GUI -> Service)
    # =========================================================================

    def _on_model_toggle(self, model_key):
        """Checkbox geaendert -> Service toggle."""
        toggle_map = {"scrfd": self.scrfd_enabled, "arcface": self.arcface_enabled,
                      "yolov8m": self.yolo_enabled, "pose": self.pose_enabled}
        enabled_var = toggle_map.get(model_key)
        if enabled_var:
            self.service.toggle_model(model_key, enabled_var.get())

    # =========================================================================
    # Camera Status Display
    # =========================================================================

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

    def _update_tracker_state(self):
        """Tracker-State im GUI anzeigen."""
        if not self.running or not self.service._autonomous_mode:
            return
        if self.service._tracker:
            state = self.service._tracker.state.value.upper()
            colors = {
                "TRACKING": "#00ff88", "SEARCHING": "#ffaa00",
                "LOCKED": "#00ff88", "IDLE": "#888888",
                "DWELL": "#aaaaff", "FROZEN": "#ff4444",
            }
            color = colors.get(state, "#888888")
            self._tracker_state_label.config(text=f"Tracker: {state}", fg=color)
        self.root.after(500, self._update_tracker_state)

    # =========================================================================
    # PTZ Controls (GUI-only)
    # =========================================================================

    def _ptz_move(self, direction):
        """Kamera in eine Richtung bewegen (ONVIF AbsoluteMove)."""
        if (self.service._autonomous_mode and self.service._tracker
                and self.service._tracker.is_running):
            return
        step = self._ptz_speed_var.get()
        PAN_STEP = step
        TILT_STEP = step * 0.67

        def do_move():
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                if not cam.is_connected:
                    self._gui_status("Kamera nicht verbunden!")
                    return
                pos = cam.get_position()
                if not pos:
                    self._gui_status("PTZ Position nicht lesbar!")
                    return
                pan, tilt = pos.pan, pos.tilt
                if direction == "left":
                    pan += PAN_STEP
                elif direction == "right":
                    pan -= PAN_STEP
                elif direction == "up":
                    tilt += TILT_STEP
                elif direction == "down":
                    tilt -= TILT_STEP
                elif direction == "home":
                    pan, tilt = 0.0, 0.0
                pan = max(-168.4, min(174.4, pan))
                tilt = max(-78.8, min(101.3, tilt))
                result = cam.move_absolute(pan, tilt)
                if result:
                    self._gui_status(f"PTZ: {pan:.1f}, {tilt:.1f}")
                else:
                    self._gui_status("PTZ Bewegung fehlgeschlagen")
            except Exception as e:
                self._gui_status(f"PTZ Fehler: {e}")
        threading.Thread(target=do_move, daemon=True).start()

    def _ptz_goto(self, pan, tilt):
        """PTZ zu bestimmter Position fahren."""
        def do_move():
            try:
                from core.hardware.camera import get_camera_controller
                cam = get_camera_controller()
                if not cam.is_connected:
                    cam.connect()
                cam.move_absolute(pan, tilt, speed=1.0)
                self._gui_status(f"PTZ -> Pan={pan} Tilt={tilt}")
            except Exception as e:
                self._gui_status(f"PTZ Fehler: {e}")
        threading.Thread(target=do_move, daemon=True).start()

    def _gui_status(self, text):
        """GUI-only Status-Update (nicht ueber Service)."""
        self.root.after(0, lambda: self.status_label.configure(text=text))

    # =========================================================================
    # Cloud Controls (GUI -> Service)
    # =========================================================================

    def _set_cloud_led(self):
        self.service._cloud_run("set_status_led", self._led_var.get())

    def _set_cloud_night(self):
        mode_map = {"Aus": "day", "Auto": "auto", "An": "night"}
        self.service._cloud_run("set_night", mode_map.get(self._night_var.get(), "day"))

    def _set_cloud_flip(self):
        self.service._cloud_run("set_screen_flip", self._flip_var.get())

    def _trigger_alarm(self):
        """3-Sekunden Alarm ausloesen."""
        if not self.service._cloud or not self.service._cloud.connected:
            self._gui_status("Cloud nicht verbunden")
            return
        def alarm_cycle():
            self.service._cloud.run(self.service._cloud.bridge.set_alarm(True))
            time.sleep(3)
            self.service._cloud.run(self.service._cloud.bridge.set_alarm(False))
            self._gui_status("Alarm beendet")
        self._gui_status("ALARM aktiv (3s)")
        threading.Thread(target=alarm_cycle, daemon=True).start()

    def _trigger_calibration(self):
        """PTZ Kalibrierung mit Bestaetigung."""
        if messagebox.askyesno("PTZ Kalibrierung",
                               "Kamera bewegt sich durch den vollen Bereich!\n\nFortfahren?"):
            self.service._cloud_run("trigger_ptz_calibration")
            self._gui_status("Kalibrierung gestartet")

    def _refresh_cloud_params(self):
        """Cloud-Parameter laden und UI synchronisieren."""
        if not self.service._cloud or not self.service._cloud.connected:
            return
        def do_refresh():
            params = self.service._cloud.run(self.service._cloud.bridge.get_device_params())
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
                on = bool(params["smartTraceEnable"])
                text = "Smart Tracking: AN" if on else "Smart Tracking: AUS"
                bg = "#884400" if on else "#2a2a4e"
                self._smart_tracking_btn.config(text=text, bg=bg)
            if "screenFlip" in params:
                self._flip_var.set(bool(params["screenFlip"]))
            if "sledOnline" in params:
                self._led_var.set(params["sledOnline"] == "on")
            self._cloud_status_label.config(text="Cloud: verbunden", fg="#00ff88")
        except Exception as e:
            logger.error(f"Apply cloud params: {e}")

    # =========================================================================
    # Aktionen (GUI-only)
    # =========================================================================

    def _kill_push_to_talk(self):
        """push_to_talk.py killen."""
        def do_kill():
            try:
                result = subprocess.run(
                    ["pkill", "-f", "push_to_talk"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self._gui_status("push_to_talk gekillt")
                else:
                    self._gui_status("push_to_talk nicht gefunden")
            except Exception as e:
                self._gui_status(f"Kill Fehler: {e}")
        threading.Thread(target=do_kill, daemon=True).start()

    def _save_snapshot(self):
        """Aktuelles annotiertes Frame als Snapshot speichern."""
        frame = None
        with self.service._annotated_lock:
            if self.service._annotated_frame is not None:
                frame = self.service._annotated_frame.copy()
        if frame is None:
            with self.service._frame_lock:
                if self.service._latest_frame is not None:
                    frame = self.service._latest_frame.copy()
        if frame is None:
            self._gui_status("Kein Frame fuer Snapshot")
            return
        snap_dir = os.path.expanduser("~/moloch/snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(snap_dir, f"hailo_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._gui_status(f"Snapshot: {path}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def run(self):
        """GUI starten."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Sauberes Herunterfahren + Headless Service wieder starten."""
        self.running = False
        self.service.stop()
        if self._display_after_id:
            self.root.after_cancel(self._display_after_id)
        self.root.destroy()
        # Headless Service wieder starten
        subprocess.Popen(
            ["sudo", "systemctl", "start", "moloch.service"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


if __name__ == "__main__":
    app = HailoControlPanel()
    app.run()
