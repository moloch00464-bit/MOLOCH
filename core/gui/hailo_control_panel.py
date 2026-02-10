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
import threading
import logging
import subprocess
import tkinter as tk
from tkinter import ttk

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


def load_face_db(path: str) -> dict:
    """Lade Face-Embeddings aus JSON."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
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
        self._ctx_lock = threading.Lock()
        self._input_640 = np.empty((640, 640, 3), dtype=np.uint8)  # Pre-allocated

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
        """Baue komplettes UI."""
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === TOP: Status Bar ===
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 5))

        self.status_label = ttk.Label(top, text="Initialisierung...", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)

        self.total_fps_label = ttk.Label(top, text="FPS: --", style="FPS.TLabel")
        self.total_fps_label.pack(side=tk.RIGHT)

        # === MIDDLE: Preview + Controls ===
        middle = ttk.Frame(main)
        middle.pack(fill=tk.BOTH, expand=True)

        # LINKS: Live Preview
        preview_frame = ttk.Frame(middle)
        preview_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(preview_frame, text="LIVE PREVIEW + DETECTIONS",
                  style="Header.TLabel").pack()
        self.preview_canvas = tk.Canvas(
            preview_frame, width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#000000", highlightthickness=1, highlightbackground="#333"
        )
        self.preview_canvas.pack(pady=5)

        # RECHTS: Controls
        ctrl_frame = ttk.Frame(middle)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ttk.Label(ctrl_frame, text="MODELLE", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 5))

        # -- SCRFD --
        self._build_model_section(
            ctrl_frame, "SCRFD Face Detection", self.scrfd_enabled,
            "scrfd", [
                ("Confidence", self.scrfd_conf, 0.1, 0.9),
                ("NMS IoU", self.scrfd_nms, 0.1, 0.9),
            ]
        )

        # -- ArcFace --
        self._build_model_section(
            ctrl_frame, "ArcFace Recognition", self.arcface_enabled,
            "arcface", [
                ("Threshold", self.arcface_thresh, 0.3, 0.9),
            ]
        )

        # -- YOLOv8m --
        self._build_model_section(
            ctrl_frame, "YOLOv8m Person", self.yolo_enabled,
            "yolov8m", [
                ("Confidence", self.yolo_conf, 0.1, 0.9),
            ]
        )

        # -- Pose --
        self._build_model_section(
            ctrl_frame, "YOLOv8s Pose", self.pose_enabled,
            "pose", [
                ("Confidence", self.pose_conf, 0.1, 0.9),
                ("NMS IoU", self.pose_nms, 0.1, 0.9),
            ]
        )

        # Separator
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Aktionen
        ttk.Label(ctrl_frame, text="AKTIONEN", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 5))

        self.kill_btn = tk.Button(
            ctrl_frame, text="push_to_talk killen", bg="#ff4444", fg="white",
            font=("Helvetica", 10, "bold"), command=self._kill_push_to_talk
        )
        self.kill_btn.pack(fill=tk.X, pady=3)

        tk.Button(
            ctrl_frame, text="Snapshot speichern", bg="#2a2a4e", fg="white",
            command=self._save_snapshot
        ).pack(fill=tk.X, pady=3)

        tk.Button(
            ctrl_frame, text="Alle Modelle AUS", bg="#2a2a4e", fg="white",
            command=self._all_models_off
        ).pack(fill=tk.X, pady=3)

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
                # Device frei machen
                if not self._hailo_manager.is_device_free():
                    self._update_status("NPU belegt - versuche freizugeben...")
                    self._hailo_manager.ensure_device_free(timeout=5.0)

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
            logger.info(f"Modell konfiguriert: {name}")
        except Exception as e:
            logger.error(f"Configure {name} fehlgeschlagen: {e}")

    def _unconfigure_model(self, name):
        """Gib Modell-Konfiguration frei."""
        with self._ctx_lock:
            ctx = self._active_ctx.pop(name, None)
        if ctx:
            try:
                ctx["ctx_mgr"].__exit__(None, None, None)
            except Exception:
                pass
            logger.info(f"Modell freigegeben: {name}")

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
        while self.running:
            # Frame holen
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.02)
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
                except Exception as e:
                    logger.error(f"SCRFD Fehler: {e}")

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

                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["arcface"] = 1.0 / dt if dt > 0 else 0
                except Exception as e:
                    logger.error(f"ArcFace Fehler: {e}")

            # 3. YOLOv8m Person Detection
            if self.yolo_enabled.get() and "yolov8m" in self._active_ctx:
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
                except Exception as e:
                    logger.error(f"Pose Fehler: {e}")

            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0

            with self._annotated_lock:
                self._annotated_frame = annotated

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
            # Configure im Hintergrund (erster Aufruf ~400ms)
            self._update_status(f"Konfiguriere {model_key}...")
            def do_cfg():
                self._configure_model(model_key)
                self._update_status("RTSP + NPU aktiv")
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

    def _all_models_off(self):
        """Alle Modelle deaktivieren und unconfigurieren."""
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

        self.root.destroy()


if __name__ == "__main__":
    app = HailoControlPanel()
    app.run()
