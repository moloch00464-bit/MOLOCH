#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye Control Panel
================================

Unified GUI for ALL camera controls - ONVIF PTZ + eWeLink Cloud.
Live RTSP preview with full hybrid feature access.

Features:
- ONVIF: PTZ directional controls, speed slider, home button
- eWeLink: Status LED, IR/Nachtsicht, Smart Tracking,
           Bild spiegeln, Alarm, Kalibrierung
- Live RTSP stream preview
- M.O.L.O.C.H. Tracker pause/resume

Author: M.O.L.O.C.H. System
Date: 2026-02-08
"""

import os
import sys
import time
import threading
import logging
import asyncio
import tkinter as tk
from tkinter import ttk, messagebox

# Auto-source ~/.profile if env vars are missing (desktop launch workaround)
if not os.environ.get("MOLOCH_CAMERA_HOST"):
    import subprocess
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
        print("Auto-sourced ~/.profile for env vars")
    except Exception as e:
        print(f"WARNING: Could not auto-source ~/.profile: {e}")

import cv2
from PIL import Image, ImageTk

# Add moloch to path
sys.path.insert(0, os.path.expanduser("~/moloch"))

from core.hardware.sonoff_camera_controller import SonoffCameraController
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("EyeControl")

# RTSP URL fallback
RTSP_URL = os.environ.get(
    "MOLOCH_RTSP_URL",
    "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
)


class CloudController:
    """Async cloud controller wrapper."""

    def __init__(self):
        self.bridge = None
        self.loop = None
        self._thread = None
        self.connected = False

    def start(self):
        """Start async event loop in background thread."""
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

        # Connect
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


class EyeControlPanel:
    """M.O.L.O.C.H. Eye Control Panel GUI."""

    PREVIEW_W = 640
    PREVIEW_H = 360
    PREVIEW_FPS = 10
    POSITION_UPDATE_INTERVAL = 2.0  # seconds between position polls

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H. Eye Control")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # State
        self.camera = None
        self.cloud = None
        self.cap = None
        self.running = True
        self.tracker_paused = False
        self.preview_running = False
        self._preview_after_id = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._last_pos_update = 0

        # Styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#1a1a2e")
        self.style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", background="#1a1a2e", foreground="#00d4ff", font=("Helvetica", 12, "bold"))
        self.style.configure("Status.TLabel", background="#1a1a2e", foreground="#66ff66", font=("Helvetica", 9))
        self.style.configure("TButton", font=("Helvetica", 10))
        self.style.configure("Danger.TButton", font=("Helvetica", 10, "bold"))
        self.style.configure("TScale", background="#1a1a2e")
        self.style.configure("TCheckbutton", background="#1a1a2e", foreground="#e0e0e0")

        self._build_ui()
        self._connect()

    def _build_ui(self):
        """Build the complete UI."""
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === TOP: Status bar + Tracker Pause ===
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 5))

        self.status_label = ttk.Label(top, text="Connecting...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)

        self.pause_btn = tk.Button(
            top, text="M.O.L.O.C.H. PAUSE", bg="#ff4444", fg="white",
            font=("Helvetica", 10, "bold"), command=self._toggle_tracker,
            relief=tk.RAISED, padx=10
        )
        self.pause_btn.pack(side=tk.RIGHT)

        # === MIDDLE: Preview + Controls side by side ===
        middle = ttk.Frame(main)
        middle.pack(fill=tk.BOTH, expand=True)

        # LEFT: Live Preview
        preview_frame = ttk.Frame(middle)
        preview_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(preview_frame, text="LIVE PREVIEW", style="Header.TLabel").pack()
        self.preview_canvas = tk.Canvas(
            preview_frame, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#000000", highlightthickness=1, highlightbackground="#333"
        )
        self.preview_canvas.pack(pady=5)

        # Position display under preview
        self.pos_label = ttk.Label(preview_frame, text="Pan: -- | Tilt: --", style="Status.TLabel")
        self.pos_label.pack()

        # RIGHT: Controls
        controls = ttk.Frame(middle)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        # Scrollable controls
        ctrl_canvas = tk.Canvas(controls, bg="#1a1a2e", highlightthickness=0, width=320)
        scrollbar = ttk.Scrollbar(controls, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_inner = ttk.Frame(ctrl_canvas)

        ctrl_inner.bind("<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)

        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- ONVIF Section ---
        self._build_onvif_section(ctrl_inner)

        # Separator
        ttk.Separator(ctrl_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # --- eWeLink Section ---
        self._build_ewelink_section(ctrl_inner)

    def _build_onvif_section(self, parent):
        """Build ONVIF PTZ controls."""
        ttk.Label(parent, text="ONVIF PTZ", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))

        # PTZ Directional pad
        ptz_frame = ttk.Frame(parent)
        ptz_frame.pack(pady=5)

        btn_w = 6
        tk.Button(ptz_frame, text="^", width=btn_w, height=1, bg="#2a2a4e", fg="white",
                  command=lambda: self._ptz_move("up")).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(ptz_frame, text="<", width=btn_w, height=1, bg="#2a2a4e", fg="white",
                  command=lambda: self._ptz_move("left")).grid(row=1, column=0, padx=2, pady=2)
        tk.Button(ptz_frame, text="H", width=btn_w, height=1, bg="#004466", fg="#00d4ff",
                  font=("Helvetica", 10, "bold"),
                  command=lambda: self._ptz_move("home")).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(ptz_frame, text=">", width=btn_w, height=1, bg="#2a2a4e", fg="white",
                  command=lambda: self._ptz_move("right")).grid(row=1, column=2, padx=2, pady=2)
        tk.Button(ptz_frame, text="v", width=btn_w, height=1, bg="#2a2a4e", fg="white",
                  command=lambda: self._ptz_move("down")).grid(row=2, column=1, padx=2, pady=2)

        # Speed slider
        speed_frame = ttk.Frame(parent)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="PTZ Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=15.0)
        self.speed_label = ttk.Label(speed_frame, text="15.0", width=5)
        self.speed_label.pack(side=tk.RIGHT)
        speed_scale = ttk.Scale(speed_frame, from_=1, to=50, variable=self.speed_var,
                                command=lambda v: self.speed_label.configure(text=f"{float(v):.0f}"))
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Quick positions
        pos_frame = ttk.Frame(parent)
        pos_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pos_frame, text="Quick:").pack(side=tk.LEFT)
        for name, pan, tilt in [("Links", 170, 0), ("Mitte", 0, 0), ("Rechts", -168, 0)]:
            tk.Button(pos_frame, text=name, bg="#2a2a4e", fg="white", width=7,
                      command=lambda p=pan, t=tilt: self._ptz_goto(p, t)).pack(side=tk.LEFT, padx=2)

    def _build_ewelink_section(self, parent):
        """Build eWeLink cloud controls."""
        ttk.Label(parent, text="eWeLink Cloud", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 5))

        self.cloud_status = ttk.Label(parent, text="Cloud: connecting...", style="Status.TLabel")
        self.cloud_status.pack(anchor=tk.W)

        # Status LED toggle
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Status LED:").pack(side=tk.LEFT)
        self.status_led_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, variable=self.status_led_var, bg="#1a1a2e", fg="#e0e0e0",
                       selectcolor="#2a2a4e", activebackground="#1a1a2e",
                       command=self._set_status_led).pack(side=tk.RIGHT)

        # IR/Night Mode
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="IR/Nachtsicht:").pack(side=tk.LEFT)
        self.night_var = tk.StringVar(value="Aus")
        night_combo = ttk.Combobox(row, textvariable=self.night_var, values=["Aus", "Auto", "An"],
                                   state="readonly", width=8)
        night_combo.pack(side=tk.RIGHT)
        night_combo.bind("<<ComboboxSelected>>", lambda e: self._set_night())

        # Smart Tracking toggle
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Smart Tracking:").pack(side=tk.LEFT)
        self.smart_track_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, variable=self.smart_track_var, bg="#1a1a2e", fg="#e0e0e0",
                       selectcolor="#2a2a4e", activebackground="#1a1a2e",
                       command=self._set_smart_tracking).pack(side=tk.RIGHT)

        # Screen Flip toggle
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Bild spiegeln:").pack(side=tk.LEFT)
        self.flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, variable=self.flip_var, bg="#1a1a2e", fg="#e0e0e0",
                       selectcolor="#2a2a4e", activebackground="#1a1a2e",
                       command=self._set_screen_flip).pack(side=tk.RIGHT)

        # Buttons row
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)

        tk.Button(btn_frame, text="ALARM", bg="#ff4444", fg="white", width=10,
                  font=("Helvetica", 10, "bold"),
                  command=self._trigger_alarm).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Kalibrierung", bg="#ff8800", fg="white", width=12,
                  command=self._trigger_calibration).pack(side=tk.LEFT, padx=3)

        # Refresh params button
        tk.Button(parent, text="Params aktualisieren", bg="#2a2a4e", fg="white",
                  command=self._refresh_params).pack(fill=tk.X, pady=5)

    def _connect(self):
        """Connect to camera and cloud in background."""
        def do_connect():
            # ONVIF
            try:
                self.camera = SonoffCameraController()
                self.camera.connect()
                pos = self.camera.get_position()
                self.root.after(0, lambda: self.status_label.configure(
                    text=f"ONVIF OK | Pan={pos.pan:.0f} Tilt={pos.tilt:.0f}"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(
                    text=f"ONVIF FEHLER: {e}"))

            # Cloud
            try:
                self.cloud = CloudController()
                self.cloud.start()
                if self.cloud.connected:
                    self.root.after(0, lambda: self.cloud_status.configure(
                        text="Cloud: verbunden"))
                    self.root.after(100, self._refresh_params)
                else:
                    self.root.after(0, lambda: self.cloud_status.configure(
                        text="Cloud: FEHLER"))
            except Exception as e:
                self.root.after(0, lambda: self.cloud_status.configure(
                    text=f"Cloud: {e}"))

            # Start preview
            self.root.after(500, self._start_preview)

        threading.Thread(target=do_connect, daemon=True).start()

    # =========================================================================
    # RTSP Live Preview (background thread reader)
    # =========================================================================

    def _start_preview(self):
        """Start RTSP live preview with background frame reader."""
        def open_and_read():
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )
            self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                logger.error(f"RTSP open failed: {RTSP_URL}")
                return

            self.preview_running = True
            self.root.after(0, self._display_frame)

            while self.running and self.preview_running:
                grabbed = self.cap.grab()
                if grabbed:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with self._frame_lock:
                            self._latest_frame = frame
                else:
                    time.sleep(0.5)

        threading.Thread(target=open_and_read, daemon=True).start()

    def _display_frame(self):
        """Display latest frame in tkinter canvas (called from main thread)."""
        if not self.running or not self.preview_running:
            return

        frame = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame

        if frame is not None:
            try:
                img = Image.fromarray(frame)
                self._photo = ImageTk.PhotoImage(image=img)
                self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            except Exception:
                pass

        # Update position (throttled)
        now = time.time()
        if self.camera and now - self._last_pos_update > self.POSITION_UPDATE_INTERVAL:
            self._last_pos_update = now
            try:
                pos = self.camera.get_position()
                self.pos_label.configure(text=f"Pan: {pos.pan:.1f} | Tilt: {pos.tilt:.1f}")
            except Exception:
                pass

        self._preview_after_id = self.root.after(1000 // self.PREVIEW_FPS, self._display_frame)

    # =========================================================================
    # ONVIF Controls
    # =========================================================================

    def _ptz_move(self, direction):
        """Move camera in direction."""
        if not self.camera:
            return
        step = self.speed_var.get()

        def do_move():
            try:
                if direction == "home":
                    self.camera.move_absolute(0.0, 0.0, speed=0.5)
                else:
                    pos = self.camera.get_position()
                    if direction == "left":
                        self.camera.move_absolute(pos.pan + step, pos.tilt, speed=1.0)
                    elif direction == "right":
                        self.camera.move_absolute(pos.pan - step, pos.tilt, speed=1.0)
                    elif direction == "up":
                        self.camera.move_absolute(pos.pan, pos.tilt + step, speed=1.0)
                    elif direction == "down":
                        self.camera.move_absolute(pos.pan, pos.tilt - step, speed=1.0)
            except Exception as e:
                logger.error(f"PTZ move error: {e}")

        threading.Thread(target=do_move, daemon=True).start()

    def _ptz_goto(self, pan, tilt):
        """Go to absolute position."""
        if not self.camera:
            return
        threading.Thread(
            target=lambda: self.camera.move_absolute(pan, tilt, speed=1.0),
            daemon=True
        ).start()

    # =========================================================================
    # eWeLink Controls
    # =========================================================================

    def _cloud_run(self, method_name, *args):
        """Run a cloud bridge method safely in background."""
        if not self.cloud or not self.cloud.connected or not self.cloud.bridge:
            logger.warning(f"Cloud not ready for {method_name}")
            return
        method = getattr(self.cloud.bridge, method_name, None)
        if not method:
            logger.error(f"Cloud bridge has no method {method_name}")
            return
        threading.Thread(
            target=lambda: self.cloud.run(method(*args)),
            daemon=True
        ).start()

    def _set_status_led(self):
        self._cloud_run("set_status_led", self.status_led_var.get())

    def _set_night(self):
        mode_map = {"Aus": "day", "Auto": "auto", "An": "night"}
        mode = mode_map.get(self.night_var.get(), "day")
        self._cloud_run("set_night", mode)

    def _set_smart_tracking(self):
        self._cloud_run("set_smart_tracking", self.smart_track_var.get())

    def _set_screen_flip(self):
        self._cloud_run("set_screen_flip", self.flip_var.get())

    def _trigger_alarm(self):
        """Trigger alarm for 3 seconds."""
        if not self.cloud or not self.cloud.connected:
            return

        def alarm_cycle():
            self.cloud.run(self.cloud.bridge.set_alarm(True))
            time.sleep(3)
            self.cloud.run(self.cloud.bridge.set_alarm(False))

        threading.Thread(target=alarm_cycle, daemon=True).start()

    def _trigger_calibration(self):
        """Trigger PTZ calibration with confirmation."""
        if messagebox.askyesno("PTZ Kalibrierung",
                               "Kamera wird sich durch den vollen Bereich bewegen!\n\nFortfahren?"):
            self._cloud_run("trigger_ptz_calibration")

    def _refresh_params(self):
        """Refresh all params from cloud."""
        if not self.cloud or not self.cloud.connected:
            return

        def do_refresh():
            params = self.cloud.run(self.cloud.bridge.get_device_params())
            if params:
                self.root.after(0, lambda: self._apply_params(params))

        threading.Thread(target=do_refresh, daemon=True).start()

    def _apply_params(self, params):
        """Apply cloud params to UI widgets."""
        try:
            if "nightVision" in params:
                nv_map = {0: "Aus", 1: "Auto", 2: "An"}
                self.night_var.set(nv_map.get(params["nightVision"], "Aus"))
            if "smartTraceEnable" in params:
                self.smart_track_var.set(bool(params["smartTraceEnable"]))
            if "screenFlip" in params:
                self.flip_var.set(bool(params["screenFlip"]))
            if "sledOnline" in params:
                self.status_led_var.set(params["sledOnline"] == "on")
        except Exception as e:
            logger.error(f"Apply params error: {e}")

    # =========================================================================
    # Tracker Control
    # =========================================================================

    def _toggle_tracker(self):
        """Pause/resume M.O.L.O.C.H. tracker."""
        try:
            from core.mpo.autonomous_tracker import get_autonomous_tracker
            tracker = get_autonomous_tracker()
            if self.tracker_paused:
                tracker.start()
                self.tracker_paused = False
                self.pause_btn.configure(text="M.O.L.O.C.H. PAUSE", bg="#ff4444")
            else:
                tracker.stop()
                self.tracker_paused = True
                self.pause_btn.configure(text="M.O.L.O.C.H. RESUME", bg="#44ff44")
        except ImportError:
            self.tracker_paused = not self.tracker_paused
            if self.tracker_paused:
                self.pause_btn.configure(text="M.O.L.O.C.H. RESUME", bg="#44ff44")
            else:
                self.pause_btn.configure(text="M.O.L.O.C.H. PAUSE", bg="#ff4444")

    def run(self):
        """Run the GUI."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Clean up on close."""
        self.running = False
        self.preview_running = False

        if self._preview_after_id:
            self.root.after_cancel(self._preview_after_id)

        if self.cap:
            self.cap.release()

        if self.cloud and self.cloud.loop:
            self.cloud.loop.call_soon_threadsafe(self.cloud.loop.stop)

        self.root.destroy()


if __name__ == "__main__":
    app = EyeControlPanel()
    app.run()
