#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Push-to-Talk GUI
=============================

Simple voice interface for M.O.L.O.C.H.
Hold button to speak, release to send to M.O.L.O.C.H.

Usage:
    python3 -m core.gui.push_to_talk
"""

import sys
import os

# Fix UTF-8 encoding for proper Umlaut/Emoji display
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LC_ALL'] = 'de_DE.UTF-8'
os.environ['LANG'] = 'de_DE.UTF-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import subprocess
import threading
import tempfile
import logging
import signal
import time
import queue
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.timeline import get_timeline
from core.vision import get_vision, get_hybrid_vision, PersonEvent
from core.speech.audio_pipeline import get_pipeline as get_audio_pipeline, AudioDiagnostics
from core.hardware import CameraID
from core.hardware.camera_controller import CameraController, get_camera_controller
from core.hardware.sonoff_camera_controller import SonoffCameraController, ControlMode, get_camera_controller as get_ptz_controller
from core.mpo.ptz_orchestrator import get_ptz_orchestrator, VisionEvent, TrackingMode
from core.mpo.autonomous_tracker import get_autonomous_tracker, AutonomousTracker, TrackerState
from core.perception.perception_manager import get_perception_manager, PerceptionMode
from core.vision.gst_hailo_detector import get_gst_detector, GstHailoDetector, list_available_models
from core.vision.gst_hailo_pose_detector import get_gst_pose_detector, GstHailoPoseDetector, PoseDetectionResult
from core.hardware.ptz_calibration import get_ptz_calibration, PTZCalibration, CalibrationStep
from core.hardware.hailo_manager import get_hailo_manager, HailoManager
from context.perception_state import get_perception_state, PerceptionState, PerceptionEvent
from context.system_autonomy import get_system_autonomy, SystemAutonomy, request_npu_for_speech, release_npu_from_speech
from core.tts.config.voices import load_voice_config
from core.tts.tts_manager import ContextSignals
from core.tts.selection.voice_selector import VoiceSelector

# Detection mode: "detection" (standard) or "pose" (with face/keypoint validation)
DETECTION_MODE = "pose"  # Pose model with keypoint validation (hailo-apps postprocess)
USE_SIMPLE_OPENCV = False  # Use Hailo pipeline
try:
    from context.vision_context import get_vision_context
except ImportError:
    get_vision_context = None

import tkinter as tk
from tkinter import ttk
import base64
from io import BytesIO
from typing import Optional
from PIL import Image, ImageTk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio settings for Pipewire recording
RATE = 16000  # Whisper expects 16kHz


class PushToTalkGUI:
    """
    M.O.L.O.C.H. Push-to-Talk Interface.

    Hold the button to record, release to transcribe and chat.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("M.O.L.O.C.H. - Push to Talk")
        self.window.geometry("540x720")  # Larger window to show all controls
        self.window.minsize(540, 720)    # Prevent shrinking below usable size
        self.window.configure(bg="#1a1a2e")

        # State
        self.is_recording = False
        self.record_process = None
        self.temp_audio_path = None
        self._npu_was_running_pose = False  # Track if pose detector was running before recording

        # Load dependencies lazily
        self.whisper = None  # HailoWhisper instance (NPU-accelerated)
        self.audio_pipeline = None  # Audio preprocessing pipeline
        self.claude_client = None
        self.tts = None
        self.vision = None
        self.hybrid_vision = None  # HybridVision pipeline
        self.conversation_history = []
        self.system_prompt = None

        # Person tracking
        self.last_recognized_person = None
        self.last_recognition_time = 0

        # Vision worker (background Hailo detection)
        self.vision_worker = None

        # GStreamer Hailo Detector for person detection
        self.hailo_detector: Optional[GstHailoDetector] = None
        self.hailo_pose_detector: Optional[GstHailoPoseDetector] = None
        self.detection_mode = DETECTION_MODE  # "detection" or "pose"
        self.last_detection_result = None
        self.last_pose_result: Optional[PoseDetectionResult] = None
        self.detection_lock = threading.Lock()

        # Camera preview state (beide Kameras)
        self.camera_running = False
        self.camera_controller = None  # For frame capture
        self.ptz_controller = None     # For PTZ control
        self.ptz_orchestrator = None
        self.autonomous_tracker: Optional[AutonomousTracker] = None
        self.auto_tracking = False

        # PTZ Calibration
        self.ptz_calibration: Optional[PTZCalibration] = None
        self.calibration_dialog = None

        # Sonoff Kamera (links)
        self.sonoff_photo = None
        self.sonoff_image_id = None
        self.sonoff_box_ids = []

        # === DEBUG: GUI frame counters ===
        self._gui_frame_count = 0
        self._gui_last_frame_time = 0
        self._gui_null_frame_count = 0

        # Processing guard (prevent multiple _process_audio threads)
        self._processing = False
        self._npu_acquired_for_voice = False

        # Voice selection
        self._voice_map = {}  # display_name -> model_stem
        self._voice_id_to_stem = {}  # voice_id (e.g. "kobold_karlsson") -> model_stem (e.g. "de_DE-karlsson-low")
        self._stem_to_display = {}  # model_stem -> display_name
        self._voice_combo = None
        self._voice_selector = VoiceSelector()
        self._last_voice_id = None  # last voice_id used (for diversity)
        self._user_voice_override_until = 0.0  # timestamp until which manual override active

        # Thread-safe queue for frame updates from Hailo callback
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)

        # === FALLBACK: OpenCV RTSP capture if Hailo fails ===
        self._sonoff_cap = None
        self._sonoff_rtsp = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
        self._current_photo = None  # Keep reference to prevent GC
        self._fallback_active = False
        self._fallback_started = False
        self._hailo_first_frame_time = 0  # When we expect first Hailo frame
        self._hailo_watchdog_timeout = 5.0  # Seconds to wait for Hailo before fallback

        self._setup_ui()
        self._load_dependencies()

    def _setup_ui(self):
        """Create the UI elements with single camera view."""
        # Ensure window size is set (larger to show all controls)
        self.window.geometry("540x720")

        # Main container
        main_frame = tk.Frame(self.window, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # HEADER: Title + Status
        header_frame = tk.Frame(main_frame, bg="#1a1a2e")
        header_frame.pack(fill="x", pady=(0, 5))

        title = tk.Label(
            header_frame,
            text="M.O.L.O.C.H.",
            font=("Helvetica", 18, "bold"),
            fg="#e94560",
            bg="#1a1a2e"
        )
        title.pack(side="left", padx=5)

        self.status_label = tk.Label(
            header_frame,
            text="Bereit",
            font=("Helvetica", 10),
            fg="#00ff88",
            bg="#1a1a2e"
        )
        self.status_label.pack(side="right", padx=5)

        # CAMERA: Sonoff CAM-PT2 (centered, larger)
        camera_container = tk.Frame(main_frame, bg="#1a1a2e")
        camera_container.pack(fill="x", pady=5)

        sonoff_frame = tk.Frame(camera_container, bg="#0f0f23", bd=2, relief="sunken")
        sonoff_frame.pack(anchor="center")

        sonoff_title = tk.Label(
            sonoff_frame,
            text="Sonoff CAM-PT2 (Hailo NPU)",
            font=("Courier", 9, "bold"),
            fg="#00ff88",
            bg="#0f0f23"
        )
        sonoff_title.pack(pady=2)

        self.sonoff_canvas = tk.Canvas(
            sonoff_frame,
            width=480,
            height=360,
            bg="#0a0a15",
            highlightthickness=0,
            takefocus=False
        )
        self.sonoff_canvas.pack(padx=3, pady=3)
        # Klick auf Canvas → Fokus zurück zum Textfeld
        self.sonoff_canvas.bind("<ButtonPress-1>", lambda e: self.text_input.focus_set())

        self.sonoff_label = tk.Label(
            sonoff_frame,
            text="Verbinde...",
            font=("Courier", 9),
            fg="#888888",
            bg="#0f0f23"
        )
        self.sonoff_label.pack(pady=2)

        # CONTROLS: PTT Button + Response Text
        controls_frame = tk.Frame(main_frame, bg="#1a1a2e")
        controls_frame.pack(fill="both", expand=True, pady=5)

        # Left: Push-to-talk button
        self.talk_button = tk.Button(
            controls_frame,
            text="SPRECHEN",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="#16213e",
            activebackground="#e94560",
            activeforeground="white",
            width=10,
            height=2,
            relief="raised",
            borderwidth=3
        )
        self.talk_button.pack(side="left", padx=5)

        # Toggle mode: Click to start recording, click again to stop
        self.talk_button.bind("<ButtonPress-1>", self._toggle_recording)

        # Right: Response text area
        response_frame = tk.Frame(controls_frame, bg="#1a1a2e")
        response_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.response_text = tk.Text(
            response_frame,
            font=("DejaVu Sans", 9),
            fg="#ffffff",
            bg="#0f0f23",
            wrap="word",
            height=8,
            state="disabled"
        )
        self.response_text.pack(fill="both", expand=True, side="left")

        # Color tags for conversation
        self.response_text.tag_configure("user", foreground="#ff6b6b")      # ROT für User
        self.response_text.tag_configure("moloch", foreground="#00ff88")    # GRÜN für MOLOCH
        self.response_text.tag_configure("system", foreground="#ffaa00")    # ORANGE für System

        scrollbar = ttk.Scrollbar(response_frame, command=self.response_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.response_text.config(yscrollcommand=scrollbar.set)

        # TEXT INPUT FIELD
        input_frame = tk.Frame(main_frame, bg="#1a1a2e")
        input_frame.pack(fill="x", pady=3)

        self.text_input = tk.Entry(
            input_frame,
            font=("DejaVu Sans", 11),
            fg="#ffffff",
            bg="#0f0f23",
            insertbackground="white"
        )
        self.text_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.text_input.bind("<Return>", self._send_text_message)

        self.send_button = tk.Button(
            input_frame,
            text="SENDEN",
            font=("Helvetica", 10, "bold"),
            fg="white",
            bg="#16213e",
            activebackground="#0f3460",
            activeforeground="white",
            width=10,
            command=lambda: self._send_text_message(None)
        )
        self.send_button.pack(side="right")

        # PTZ Control Frame - Dual Mode
        ptz_frame = tk.Frame(main_frame, bg="#1a1a2e")
        ptz_frame.pack(fill="x", pady=(5, 0))
        
        # Mode Toggle
        ptz_label = tk.Label(ptz_frame, text="PTZ:", font=("Courier", 9), fg="#888888", bg="#1a1a2e")
        ptz_label.pack(side="left", padx=(0, 5))
        
        self.auto_track_var = tk.BooleanVar(value=False)
        self.btn_track = tk.Checkbutton(ptz_frame, text="Auto", 
            variable=self.auto_track_var, font=("Courier", 9),
            fg="#00ff88", bg="#1a1a2e", selectcolor="#0f3460",
            activebackground="#1a1a2e", activeforeground="#00ff88",
            command=self._toggle_tracking)
        self.btn_track.pack(side="left", padx=5)
        
        # Manual PTZ Buttons
        btn_style = {"font": ("Courier", 10, "bold"), "width": 2, "height": 1,
                     "bg": "#16213e", "fg": "white", "activebackground": "#0f3460"}
        
        self.btn_left = tk.Button(ptz_frame, text="<", command=lambda: self._ptz_move("left"), **btn_style)
        self.btn_left.pack(side="left", padx=2)
        
        self.btn_up = tk.Button(ptz_frame, text="^", command=lambda: self._ptz_move("up"), **btn_style)
        self.btn_up.pack(side="left", padx=2)
        
        self.btn_down = tk.Button(ptz_frame, text="v", command=lambda: self._ptz_move("down"), **btn_style)
        self.btn_down.pack(side="left", padx=2)
        
        self.btn_right = tk.Button(ptz_frame, text=">", command=lambda: self._ptz_move("right"), **btn_style)
        self.btn_right.pack(side="left", padx=2)
        
        self.btn_center = tk.Button(ptz_frame, text="O", command=self._ptz_center,
            font=("Courier", 10, "bold"), width=2, height=1,
            bg="#0f3460", fg="#00ff88", activebackground="#16213e")
        self.btn_center.pack(side="left", padx=5)

        # Test button for ContinuousMove debugging (orange "T")
        self.btn_test_cm = tk.Button(ptz_frame, text="T", command=self._test_continuous_move,
            font=("Courier", 10, "bold"), width=2, height=1,
            bg="#460f0f", fg="#ff8800", activebackground="#2e1616")
        self.btn_test_cm.pack(side="left", padx=5)

        # Calibration button (blue "C")
        self.btn_calibrate = tk.Button(ptz_frame, text="C", command=self._start_calibration,
            font=("Courier", 10, "bold"), width=2, height=1,
            bg="#0f3460", fg="#00ffff", activebackground="#16213e")
        self.btn_calibrate.pack(side="left", padx=2)

        self.ptz_status = tk.Label(ptz_frame, text="PTZ: MANUAL", font=("Courier", 8),
            fg="#00ff88", bg="#1a1a2e")
        self.ptz_status.pack(side="right", padx=5)

        # VOICE SELECTION
        voice_frame = tk.Frame(main_frame, bg="#1a1a2e")
        voice_frame.pack(fill="x", pady=(3, 0))

        voice_label = tk.Label(voice_frame, text="Stimme:", font=("Courier", 9),
            fg="#888888", bg="#1a1a2e")
        voice_label.pack(side="left", padx=(0, 5))

        self._voice_combo = ttk.Combobox(voice_frame, state="readonly",
            font=("DejaVu Sans", 9), width=25)
        self._voice_combo.pack(side="left", padx=2)
        self._voice_combo.bind("<<ComboboxSelected>>", self._on_voice_changed)

        self.btn_voice_test = tk.Button(voice_frame, text="Test",
            command=self._preview_voice,
            font=("Courier", 9, "bold"), width=4, height=1,
            bg="#16213e", fg="#ffaa00", activebackground="#0f3460")
        self.btn_voice_test.pack(side="left", padx=5)

        # Show placeholder on canvas
        self._draw_no_signal(self.sonoff_canvas, "Sonoff")

    # ── Voice Selection ──────────────────────────────────────

    def _load_voice_choices(self):
        """Load voice metadata from voices.json and populate combobox."""
        try:
            voices = load_voice_config()
            if not voices:
                return
            self._voice_map = {}
            self._voice_id_to_stem = {}
            self._stem_to_display = {}
            for v in voices:
                display = v["display_name"]
                model_stem = Path(v["model_path"]).stem if "model_path" in v else v["voice_id"]
                voice_id = v["voice_id"]
                self._voice_map[display] = model_stem
                self._voice_id_to_stem[voice_id] = model_stem
                self._stem_to_display[model_stem] = display

            names = list(self._voice_map.keys())
            self._voice_combo["values"] = names

            # Select current voice
            current = self.tts.current_voice if self.tts else None
            for display, stem in self._voice_map.items():
                if stem == current:
                    self._voice_combo.set(display)
                    break
            else:
                if names:
                    self._voice_combo.current(0)

            logger.info(f"Voice choices loaded: {len(names)} Stimmen")
        except Exception as e:
            logger.error(f"Failed to load voice choices: {e}")

    def _on_voice_changed(self, event=None):
        """Handle voice combobox selection change - manual override for 5 min."""
        display = self._voice_combo.get()
        voice_id = self._voice_map.get(display)
        if not voice_id or not self.tts:
            return
        if self.tts.set_voice(voice_id):
            self._user_voice_override_until = time.time() + 300  # 5 min override
            self._set_status(f"Stimme: {display} (manuell)", "#ffaa00")
            logger.info(f"Voice manually set: {display} ({voice_id}) - auto-select paused 5min")

    def _preview_voice(self):
        """Speak a short test sentence with the current voice."""
        if not self.tts or not self.tts.enabled:
            self._set_status("TTS nicht verfuegbar", "#ff4444")
            return
        self.tts.speak("Hallo, ich bin Moloch.", blocking=False)

    def _get_time_of_day(self) -> str:
        """Determine time of day category."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _auto_select_voice(self):
        """Auto-select voice based on context before speaking."""
        if not self.tts or not self._voice_id_to_stem:
            return
        # Skip if user manually overrode recently
        if time.time() < self._user_voice_override_until:
            return
        try:
            context = ContextSignals(
                time_of_day=self._get_time_of_day(),
                system_load="low",
                recent_interaction_tone=None,
                explicit_user_request=None,
                session_duration=None,
                last_voice_used=self._last_voice_id,
            )
            selection = self._voice_selector.select(context)
            voice_id = selection.voice_id
            model_stem = self._voice_id_to_stem.get(voice_id)
            if model_stem and model_stem != self.tts.current_voice:
                self.tts.set_voice(model_stem)
                self._last_voice_id = voice_id
                display = self._stem_to_display.get(model_stem, voice_id)
                self._voice_combo.set(display)
                logger.info(f"Auto-Voice: {display} | {selection.reason}")
        except Exception as e:
            logger.error(f"Auto voice selection failed: {e}")

    def _speak(self, text: str, blocking: bool = False):
        """Speak text with auto-selected voice."""
        if not self.tts or not self.tts.enabled:
            return
        self._auto_select_voice()
        self.tts.speak(text, blocking=blocking)

    def _draw_no_signal(self, canvas, name: str):
        """Draw 'Kein Signal' placeholder on a camera canvas."""
        canvas.delete("all")
        # Dark grid pattern (480x360 canvas)
        for i in range(0, 480, 32):
            canvas.create_line(i, 0, i, 360, fill="#1a1a2e", width=1)
        for i in range(0, 360, 32):
            canvas.create_line(0, i, 480, i, fill="#1a1a2e", width=1)
        # Text (centered at 240, 180)
        canvas.create_text(
            240, 180,
            text=f"KEIN SIGNAL\n\n{name}",
            font=("Courier", 14, "bold"),
            fill="#ff4444",
            justify="center"
        )

    def _poll_vision_context(self):
        """Poll VisionContext for detections."""
        if not self.auto_tracking or not get_vision_context:
            return
        try:
            ctx = get_vision_context()
            state = ctx.get_state()
            if state.person_detected:
                logger.info(f"VCTX: person=True x={state.target_center_x} w={state.frame_width} conf={state.confidence:.2f}")
                self._do_tracking(state.target_center_x, state.frame_width, state.confidence)
        except Exception as e:
            logger.debug(f"VisionContext poll error: {e}")

    def _on_hailo_detection(self, result):
        """Callback from GstHailoDetector when detection results are available."""
        try:
            # Store result for display
            with self.detection_lock:
                self.last_detection_result = result

            # Queue frame for display (thread-safe)
            if result.frame is not None:
                try:
                    self._frame_queue.put_nowait((result.frame, None))
                except queue.Full:
                    pass

            # === STRICT DETECTION FILTERING (before tracker) ===
            # Filter out hands, partial bodies, objects near lens
            filtered_detections = []
            raw_count = len(result.detections or [])
            frame_w = result.frame_width
            frame_h = result.frame_height
            frame_area = frame_w * frame_h

            for det in (result.detections or []):
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Filter 1: Minimum height (35% of frame - reject hands/partial)
                if height < 0.35 * frame_h:
                    continue

                # Filter 2: Minimum area (10% of frame)
                if area < 0.10 * frame_area:
                    continue

                # Filter 3: Aspect ratio (width/height > 0.4 - reject narrow objects)
                aspect = width / height if height > 0 else 0
                if aspect < 0.4:
                    continue

                # Filter 4: Reject if touching bottom edge (hand close to lens)
                if y2 >= frame_h - 5:
                    continue

                filtered_detections.append(det)

            # Log filtering result (only when changed or periodically)
            if raw_count > 0 and len(filtered_detections) != raw_count:
                logger.debug(f"Detection filter: {raw_count} -> {len(filtered_detections)} (rejected {raw_count - len(filtered_detections)})")

            # Feed FILTERED detections to AutonomousTracker (if active)
            if self.autonomous_tracker and self.auto_tracking:
                self.autonomous_tracker.update_detection(
                    detections=filtered_detections,
                    frame_width=frame_w,
                    frame_height=frame_h
                )

            # Update VisionContext for other systems (use filtered detections)
            if get_vision_context:
                ctx = get_vision_context()
                if filtered_detections:
                    # Get best detection (highest confidence)
                    best = max(filtered_detections, key=lambda d: d.get("confidence", 0))

                    # Calculate pixel center (bbox is already in pixels)
                    bbox = best.get("bbox", [0, 0, 0, 0])
                    if len(bbox) == 4:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                    else:
                        center_x = frame_w // 2

                    ctx.update_detection(
                        person_detected=True,
                        person_count=len(filtered_detections),
                        confidence=best.get("confidence", 0),
                        source="hailo_gst",
                        target_center_x=center_x,
                        frame_width=frame_w,
                        npu_active=True,
                        camera_connected=True
                    )
                else:
                    ctx.update_detection(
                        person_detected=False,
                        person_count=0,
                        confidence=0,
                        source="hailo_gst",
                        npu_active=True,
                        camera_connected=True
                    )
        except Exception as e:
            logger.error(f"Hailo detection callback error: {e}")

    def _on_hailo_pose_detection(self, result: PoseDetectionResult):
        """Callback from GstHailoPoseDetector - uses keypoints for person validation."""
        try:
            # === CRITICAL: Track every frame ===
            self._gui_frame_count += 1
            self._gui_last_frame_time = time.time()

            # Log FIRST frame explicitly
            if self._gui_frame_count == 1:
                logger.info("=" * 60)
                logger.info("[GUI] *** FIRST FRAME RECEIVED IN CALLBACK ***")
                logger.info(f"[GUI] frame shape: {result.frame.shape if result.frame is not None else 'None'}")
                logger.info(f"[GUI] fps: {result.fps:.1f}")
                logger.info("=" * 60)

            # Log EVERY 10 frames to confirm callback chain works
            if self._gui_frame_count % 10 == 0:
                logger.info(f"[GUI] FRAME RECEIVED #{self._gui_frame_count}")

            if result.frame is None:
                self._gui_null_frame_count += 1
                if self._gui_null_frame_count <= 5:
                    logger.warning(f"[GUI_CB] NULL frame #{self._gui_null_frame_count}!")
            else:
                self._gui_null_frame_count = 0

            # Log every 30 frames
            if self._gui_frame_count % 30 == 0:
                frame_status = f"{result.frame.shape}" if result.frame is not None else "NONE"
                logger.info(f"[GUI_CB] Frame #{self._gui_frame_count}: "
                           f"{result.total_detections} det, FPS={result.fps:.1f}, "
                           f"frame={frame_status}")

            # Store result for display
            with self.detection_lock:
                self.last_pose_result = result

            # Queue frame for display (thread-safe)
            # CRITICAL: Copy frame to avoid buffer reuse issues
            if result.frame is not None:
                try:
                    frame_copy = result.frame.copy()  # MUST copy!
                    self._frame_queue.put_nowait((frame_copy, result))
                except queue.Full:
                    pass  # Drop old frames

            # === POSE-BASED FILTERING ===
            # Only valid persons (has_face AND has_torso) are tracked
            valid_detections = []
            for pose in result.valid_persons:
                # Convert PersonPose to detection dict for tracker
                valid_detections.append(pose.to_dict())

            # Log pose validation stats periodically
            if result.total_detections > 0:
                rejected = result.total_detections - len(result.valid_persons)
                if rejected > 0:
                    logger.debug(f"[POSE] {result.total_detections} det, {result.faces_detected} faces, "
                               f"{len(result.valid_persons)} valid, {result.hands_rejected} hands rejected")

            # Feed VALID persons to AutonomousTracker using POSE method (head tracking)
            if self.autonomous_tracker and self.auto_tracking:
                # Use pose-specific update that tracks HEAD center, not bbox center
                self.autonomous_tracker.update_pose_detection(
                    poses=valid_detections,
                    frame_width=result.frame_width,
                    frame_height=result.frame_height
                )

            # Update VisionContext for other systems
            if get_vision_context:
                ctx = get_vision_context()
                if result.valid_persons:
                    # Get best valid person (highest face confidence)
                    best = max(result.valid_persons, key=lambda p: p.face_confidence)
                    # Use HEAD center for tracking, not bbox center!
                    head_center = best.get_head_center() or best.get_face_center()
                    if head_center:
                        center_x = int(head_center[0] * result.frame_width)
                    else:
                        bbox = best.bbox
                        center_x = int((bbox[0] + bbox[2]) / 2)

                    ctx.update_detection(
                        person_detected=True,
                        person_count=len(result.valid_persons),
                        confidence=best.confidence,
                        source="hailo_pose",
                        target_center_x=center_x,
                        frame_width=result.frame_width,
                        npu_active=True,
                        camera_connected=True
                    )
                else:
                    ctx.update_detection(
                        person_detected=False,
                        person_count=0,
                        confidence=0,
                        source="hailo_pose",
                        npu_active=True,
                        camera_connected=True
                    )

            # === UPDATE PERCEPTION STATE ===
            # Central state object for autonomy/dialogue integration
            perception = get_perception_state()
            if result.valid_persons:
                best = max(result.valid_persons, key=lambda p: p.confidence)
                kp_counts = best.get_keypoint_counts()

                # Check for gesture
                gesture_detected = result.gesture is not None
                gesture_type = result.gesture.type.value if result.gesture else "none"
                gesture_conf = result.gesture.confidence if result.gesture else 0.0

                perception.update(
                    user_detected=True,
                    face_detected=best.has_face,
                    gesture_detected=gesture_detected,
                    gesture_type=gesture_type,
                    person_count=len(result.valid_persons),
                    confidence=best.confidence,
                    face_confidence=best.face_confidence,
                    gesture_confidence=gesture_conf,
                    face_keypoints=kp_counts.get("face", 0),
                    torso_keypoints=kp_counts.get("torso", 0),
                    wrist_keypoints=kp_counts.get("wrist", 0),
                    source="hailo_pose"
                )
            else:
                # No valid person - update with empty to let timeout handle state
                perception.update(
                    user_detected=False,
                    face_detected=False,
                    gesture_detected=False,
                    source="hailo_pose"
                )

        except Exception as e:
            logger.error(f"Hailo pose detection callback error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _update_sonoff_display_pose(self, img_bgr, result: PoseDetectionResult):
        """Update Sonoff display with pose overlay (keypoints + validation status)."""
        try:
            import cv2

            # === VALIDATE FRAME ===
            if img_bgr is None:
                logger.warning("[DISPLAY] Received None frame!")
                return
            if not hasattr(img_bgr, 'shape') or len(img_bgr.shape) != 3:
                logger.warning(f"[DISPLAY] Invalid frame: {type(img_bgr)}")
                return

            # Debug: log display updates
            if not hasattr(self, '_display_update_count'):
                self._display_update_count = 0
            self._display_update_count += 1
            if self._display_update_count % 30 == 1:
                logger.info(f"[DISPLAY] update #{self._display_update_count}, frame shape={img_bgr.shape}")

            # CRITICAL: Clear "KEIN SIGNAL" items on first frame
            if self._display_update_count == 1:
                logger.info("[DISPLAY] First frame - clearing canvas placeholder")
                self.sonoff_canvas.delete("all")
                self.sonoff_image_id = None

            # Resize for display
            img = cv2.resize(img_bgr, (480, 360), interpolation=cv2.INTER_NEAREST)

            # Scale factors from detection frame (640x640) to display (480x360)
            scale_x = 480 / result.frame_width
            scale_y = 360 / result.frame_height

            # Draw all poses with validation color coding
            for pose in result.poses:
                bbox = pose.bbox
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)

                # Color based on validation (green=valid, red=hand, orange=partial)
                if pose.is_valid_person:
                    color = (0, 255, 0)  # Green
                elif pose.validation_reason == "hand_only":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 165, 255)  # Orange

                # Draw bounding box only (no labels)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Clean display - no text overlays (user request)

            # Convert to RGB for tkinter
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            pil_img = Image.fromarray(img)
            self.sonoff_photo = ImageTk.PhotoImage(pil_img)

            if self.sonoff_image_id:
                self.sonoff_canvas.itemconfig(self.sonoff_image_id, image=self.sonoff_photo)
            else:
                self.sonoff_image_id = self.sonoff_canvas.create_image(
                    0, 0, anchor="nw", image=self.sonoff_photo)

            # Update label
            if result.valid_persons:
                self.sonoff_label.config(text=f"{len(result.valid_persons)} Person(en)", fg="#00ff88")
            else:
                self.sonoff_label.config(text="Niemand sichtbar", fg="#888888")

        except Exception as e:
            logger.error(f"Sonoff pose display error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _start_camera_preview(self):
        """Start background thread for dual camera preview updates."""
        if self.camera_running:
            return
        self.camera_running = True

        # === ROLLBACK MODE: Simple OpenCV for stability ===
        if USE_SIMPLE_OPENCV:
            logger.info("=" * 60)
            logger.info("=== ROLLBACK MODE: Simple OpenCV ===")
            logger.info("Using direct OpenCV VideoCapture for both cameras")
            logger.info("GstHailo integration DISABLED")
            logger.info("=" * 60)

            # Initialize simple OpenCV capture for Sonoff with LOW LATENCY settings
            self._sonoff_cap = None
            self._sonoff_rtsp = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
            try:
                import cv2
                import os

                # Set FFMPEG options via environment for low latency
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

                self._sonoff_cap = cv2.VideoCapture(self._sonoff_rtsp, cv2.CAP_FFMPEG)

                # Set minimal buffer size (key for low latency!)
                self._sonoff_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if self._sonoff_cap.isOpened():
                    logger.info("Sonoff RTSP connected via OpenCV (LOW LATENCY MODE)")
                    self.window.after(0, lambda: self.sonoff_label.config(
                        text="OpenCV RTSP", fg="#00ff88"))
                else:
                    logger.warning("Sonoff RTSP failed to open")
                    self.window.after(0, lambda: self.sonoff_label.config(
                        text="RTSP Fehler", fg="#ff4444"))
            except Exception as e:
                logger.error(f"Sonoff OpenCV error: {e}")

            # Start simple update loop
            threading.Thread(target=self._simple_camera_loop, daemon=True).start()
            logger.info("Simple OpenCV camera loop started")
            return

        # === ORIGINAL ARCHITECTURE (disabled in rollback) ===
        logger.info("=== Single RTSP Source Architecture ===")
        logger.info("Sonoff: GstHailoDetector (RTSP + NPU detection + frames)")

        # Step 1: Initialize GStreamer Hailo Detector FIRST
        # CRITICAL: Use SINGLETON to prevent multiple Hailo instances (causes status=62)
        from core.vision.gst_hailo_pose_detector import get_gst_pose_detector
        try:
            if self.detection_mode == "pose":
                logger.info("=" * 60)
                logger.info("[GUI] === POSE MODE: Using SINGLETON detector ===")
                self.hailo_pose_detector = get_gst_pose_detector()
                logger.info(f"[GUI] Got detector singleton: {id(self.hailo_pose_detector)}")

                # Check if already running
                if self.hailo_pose_detector._running:
                    logger.warning("[GUI] DETECTOR ALREADY RUNNING - will use existing instance")
                else:
                    logger.info("[GUI] Detector not running - will start fresh")

                # Set callback BEFORE start
                self.hailo_pose_detector.set_detection_callback(self._on_hailo_pose_detection)
                logger.info(f"[GUI] Callback registered: {self.hailo_pose_detector._on_detection is not None}")

                # Start detector
                logger.info("[GUI] DETECTOR START CALLED")
                start_result = self.hailo_pose_detector.start()
                logger.info(f"[GUI] start() returned: {start_result}")

                if start_result:
                    logger.info("[GUI] GstHailoPoseDetector: Started successfully!")
                    self.window.after(0, lambda: self.sonoff_label.config(
                        text="HAILO POSE", fg="#00ffff"))

                    # Verify callback and pipeline state
                    logger.info(f"[GUI] Callback after start: {self.hailo_pose_detector._on_detection is not None}")
                    logger.info(f"[GUI] _running flag: {self.hailo_pose_detector._running}")
                    if self.hailo_pose_detector._pipeline:
                        state = self.hailo_pose_detector._pipeline.get_state(0)
                        logger.info(f"[GUI] Pipeline state after start: {state}")
                    else:
                        logger.error("[GUI] Pipeline is None after start!")
                    logger.info("=" * 60)
                else:
                    logger.warning("[GUI] GstHailoPoseDetector failed to start, falling back")
                    self.hailo_pose_detector = None
                    self.detection_mode = "detection"

            if self.detection_mode == "detection" or (self.detection_mode == "pose" and not self.hailo_pose_detector):
                self.hailo_detector = get_gst_detector()
                self.hailo_detector.set_detection_callback(self._on_hailo_detection)
                if self.hailo_detector.start():
                    logger.info("GstHailoDetector: Started")
                    self.window.after(0, lambda: self.sonoff_label.config(
                        text="640x640 NPU @ 15fps", fg="#00ff88"))
                else:
                    self.hailo_detector = None
                    self.window.after(0, lambda: self.sonoff_label.config(
                        text="NPU nicht verfügbar", fg="#ff4444"))

        except Exception as e:
            logger.error(f"Hailo init failed: {e}")
            self.hailo_detector = None
            self.hailo_pose_detector = None

        # === HAILO MANAGER: Register pipelines for managed lifecycle ===
        try:
            hailo_mgr = get_hailo_manager()
            if self.hailo_pose_detector:
                # Register vision callbacks
                hailo_mgr.register_vision_pipeline(
                    pipeline=self.hailo_pose_detector,
                    on_stop=lambda: self.hailo_pose_detector.stop(release_hailo=False),
                    on_start=lambda: self.hailo_pose_detector.start(skip_npu_check=True)
                )
                logger.info("[HAILO_MGR] Vision pipeline registered with manager")

            if self.whisper:
                # VDevice MUSS freigegeben werden vor Vision-Restart!
                # HAILO_OUT_OF_PHYSICAL_DEVICES wenn Whisper + GStreamer
                # gleichzeitig VDevices halten
                hailo_mgr.register_voice_pipeline(
                    pipeline=self.whisper,
                    on_stop=lambda: self.whisper.release(),
                    on_start=None   # Whisper startet sich selbst in transcribe()
                )
                logger.info("[HAILO_MGR] Voice pipeline registered with manager")
        except Exception as e:
            logger.warning(f"[HAILO_MGR] Failed to register pipelines: {e}")

        threading.Thread(target=self._camera_update_loop, daemon=True).start()
        logger.info("Camera preview + Hailo detection started")

        # === Initialize System Autonomy ===
        try:
            autonomy = get_system_autonomy()

            # Register components for autonomy management
            if self.hailo_pose_detector:
                autonomy.register_pose_detector(self.hailo_pose_detector)
                autonomy._state.vision_active = True
            if self.whisper:
                autonomy.register_whisper(self.whisper)
            if hasattr(self, 'autonomous_tracker') and self.autonomous_tracker:
                autonomy.register_tracker(self.autonomous_tracker)

            # Register perception state
            perception = get_perception_state()
            autonomy.register_perception_state(perception)

            # Start autonomy monitoring
            autonomy.start()
            logger.info("[AUTONOMY] System autonomy initialized and monitoring")

        except Exception as e:
            logger.error(f"[AUTONOMY] Init failed: {e}")

    def _simple_camera_loop(self):
        """Simple OpenCV-based camera loop for rollback mode."""
        import cv2

        logger.info("[SIMPLE_CAM] Starting OpenCV camera loop...")
        frame_count = 0
        last_log = time.time()

        while self.camera_running:
            try:
                # === SONOFF RTSP via OpenCV ===
                if self._sonoff_cap and self._sonoff_cap.isOpened():
                    ret, frame = self._sonoff_cap.read()
                    if ret and frame is not None:
                        frame_count += 1

                        # Log every second
                        now = time.time()
                        if now - last_log >= 1.0:
                            logger.info(f"[SIMPLE_CAM] FRAME UPDATE OK - {frame_count} frames")
                            last_log = now

                        # Resize and add overlay
                        display = cv2.resize(frame, (480, 360))
                        cv2.putText(display, f"LIVE #{frame_count}", (5, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Convert BGR to RGB
                        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                        # Create PhotoImage in main thread via after()
                        # Pass raw RGB data to avoid GC issues
                        rgb_copy = rgb.copy()

                        def update_canvas(data=rgb_copy, fc=frame_count):
                            try:
                                pil_img = Image.fromarray(data)
                                # Store in instance to prevent GC
                                self._current_photo = ImageTk.PhotoImage(pil_img)

                                if fc == 1:
                                    self.sonoff_canvas.delete("all")
                                    self.sonoff_image_id = None

                                if self.sonoff_image_id:
                                    self.sonoff_canvas.itemconfig(
                                        self.sonoff_image_id,
                                        image=self._current_photo)
                                else:
                                    self.sonoff_image_id = self.sonoff_canvas.create_image(
                                        0, 0, anchor="nw", image=self._current_photo)
                            except Exception as e:
                                logger.error(f"[SIMPLE_CAM] Canvas update error: {e}")

                        self.window.after(0, update_canvas)
                    else:
                        logger.warning("[SIMPLE_CAM] Frame read failed, reconnecting...")
                        self._sonoff_cap.release()
                        time.sleep(0.5)
                        self._sonoff_cap = cv2.VideoCapture(self._sonoff_rtsp, cv2.CAP_FFMPEG)
                        self._sonoff_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # ~20 FPS - CRITICAL: must have sleep!
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"[SIMPLE_CAM] Loop error: {e}")
                time.sleep(0.1)

        # Cleanup
        if self._sonoff_cap:
            self._sonoff_cap.release()
            logger.info("[SIMPLE_CAM] Capture released")

    def _camera_update_loop(self):
        """Background loop for camera updates.

        NOTE: Sonoff frames come from GstHailoDetector callback (_on_hailo_detection)
        This ensures SINGLE RTSP source architecture.
        """
        fps_start = time.time()
        frame_count = 0

        while self.camera_running:
            try:
                # FPS Logging alle 10 Sekunden
                now = time.time()
                if now - fps_start >= 10.0:
                    fps = frame_count / (now - fps_start)
                    logger.info(f"GUI Sonoff: {fps:.1f} FPS")
                    frame_count = 0
                    fps_start = now

            except Exception as e:
                logger.error(f"Camera update error: {e}")

            # Poll VisionContext for MPO tracking
            if self.auto_tracking:
                self._poll_vision_context()

            # 25 FPS polling (40ms)
            time.sleep(0.04)

    # _detection_loop removed - GstHailoDetector handles detection via callback

    def _update_sonoff_display_fast(self, img_bgr):
        """Fast Sonoff update with detection overlay from GstHailoDetector."""
        try:
            import cv2

            # Get detection result (thread-safe)
            detections = []
            det_fps = 0
            det_w, det_h = 640, 640  # Default detection frame size
            with self.detection_lock:
                if self.last_detection_result:
                    detections = self.last_detection_result.detections or []
                    det_fps = getattr(self.last_detection_result, 'fps', 0)
                    det_w = getattr(self.last_detection_result, 'frame_width', 640)
                    det_h = getattr(self.last_detection_result, 'frame_height', 640)

            # Resize for display
            img = cv2.resize(img_bgr, (480, 360), interpolation=cv2.INTER_NEAREST)

            # Draw bounding boxes - scale from detection frame (640x640) to display (480x360)
            scale_x = 480 / det_w
            scale_y = 360 / det_h

            for det in detections:
                bbox = det.get("bbox", [0, 0, 0, 0])
                conf = det.get("confidence", 0)
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Scale to display size
                    dx1 = int(x1 * scale_x)
                    dy1 = int(y1 * scale_y)
                    dx2 = int(x2 * scale_x)
                    dy2 = int(y2 * scale_y)

                    # Draw box (green for person)
                    cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

                    # Draw label background
                    label = f"{int(conf*100)}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(img, (dx1, dy1-th-4), (dx1+tw+4, dy1), (0, 255, 0), -1)
                    cv2.putText(img, label, (dx1+2, dy1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    # Draw tracking crosshair at center
                    cx, cy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
                    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

            # Draw info overlay (top-right, adjusted for 480x360)
            info_text = f"NPU: {det_fps:.0f}fps"
            cv2.putText(img, info_text, (380, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw tracker state (top-left)
            if self.autonomous_tracker and self.auto_tracking:
                state = self.autonomous_tracker.state.value.upper()
                color = (0, 255, 0) if state == "TRACKING" else (255, 255, 0) if state == "SEARCHING" else (255, 165, 0)
                cv2.putText(img, state, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Convert to RGB for tkinter
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            pil_img = Image.fromarray(img)
            self.sonoff_photo = ImageTk.PhotoImage(pil_img)

            if self.sonoff_image_id:
                self.sonoff_canvas.itemconfig(self.sonoff_image_id, image=self.sonoff_photo)
            else:
                self.sonoff_image_id = self.sonoff_canvas.create_image(
                    0, 0, anchor="nw", image=self.sonoff_photo)

            # Update Sonoff label with detection count
            if detections:
                self.sonoff_label.config(text=f"{len(detections)} Person(en)", fg="#00ff88")
            else:
                self.sonoff_label.config(text="Niemand sichtbar", fg="#888888")

        except Exception as e:
            logger.error(f"Sonoff display error: {e}")

    def _update_sonoff_display(self, frame):
        """Update Sonoff camera canvas with new frame."""
        try:
            import cv2
            # Resize zu Canvas-Groesse
            img = cv2.resize(frame.image, (480, 360), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            self.sonoff_photo = ImageTk.PhotoImage(pil_img)

            if self.sonoff_image_id:
                self.sonoff_canvas.itemconfig(self.sonoff_image_id, image=self.sonoff_photo)
            else:
                self.sonoff_image_id = self.sonoff_canvas.create_image(
                    0, 0, anchor="nw", image=self.sonoff_photo)
        except Exception as e:
            logger.error(f"Sonoff display error: {e}")

    def _draw_detection_box(self, canvas, det, src_w, src_h, box_ids_list):
        """Draw detection bounding box on canvas."""
        x = det.get('x', 0)
        y = det.get('y', 0)
        w = det.get('w', 50)
        h = det.get('h', 50)
        score = det.get('score', 0)

        # Scale to 480x360
        scale_x = 480 / src_w
        scale_y = 360 / src_h
        cx = int(x * scale_x)
        cy = int(y * scale_y)
        bw = int(w * scale_x)
        bh = int(h * scale_y)

        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        box_id = canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff88", width=2)
        box_ids_list.append(box_id)
        label_bg = canvas.create_rectangle(x1, y1 - 16, x1 + 50, y1, fill="#00ff88", outline="")
        box_ids_list.append(label_bg)
        label_txt = canvas.create_text(x1 + 25, y1 - 8, text=f"{score}%",
            font=("Courier", 8, "bold"), fill="#000000")
        box_ids_list.append(label_txt)

    def _draw_camera_grid(self):
        """Draw grid when no image available (480x360 canvas)."""
        self.camera_canvas.create_rectangle(0, 0, 480, 360, fill="#0a0a15", outline="")
        for i in range(0, 480, 32):
            self.camera_canvas.create_line(i, 0, i, 360, fill="#1a1a2e", width=1)
        for i in range(0, 360, 32):
            self.camera_canvas.create_line(0, i, 480, i, fill="#1a1a2e", width=1)

    def _load_dependencies(self):
        """Load Whisper, Claude, and TTS in background."""
        def load():
            self._set_status("Lade Whisper (Hailo-10H)...", "#ffff00")
            try:
                from core.speech import get_whisper
                self.whisper = get_whisper()
                if self.whisper.is_available:
                    logger.info(f"Whisper loaded: {self.whisper}")
                else:
                    raise RuntimeError("No Whisper backend available")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                self._set_status(f"Whisper Fehler: {e}", "#ff0000")
                return

            # Load audio preprocessing pipeline
            self._set_status("Lade Audio-Pipeline...", "#ffff00")
            try:
                self.audio_pipeline = get_audio_pipeline()
                logger.info("Audio pipeline loaded (Noise Gate + Normalization + VAD)")
            except Exception as e:
                logger.warning(f"Audio pipeline not available: {e}")
                self.audio_pipeline = None

            self._set_status("Lade Claude API...", "#ffff00")
            try:
                import anthropic
                from core.console.moloch_console import load_api_key, build_system_prompt

                api_key = load_api_key()
                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    self.system_prompt = build_system_prompt()
                    logger.info("Claude API loaded")
                else:
                    logger.warning("No Claude API key found")
            except Exception as e:
                logger.error(f"Failed to load Claude: {e}")

            self._set_status("Lade TTS...", "#ffff00")
            try:
                from core.console.moloch_console import get_tts
                self.tts = get_tts()
                logger.info(f"TTS loaded: {self.tts.enabled}")
                self.window.after(0, self._load_voice_choices)
            except Exception as e:
                logger.error(f"Failed to load TTS: {e}")

            self._set_status("Lade Vision (Dual Kamera)...", "#ffff00")
            try:
                self.vision = get_vision()
                if self.vision:
                    logger.info(f"Vision loaded: connected={self.vision.connected}")
                else:
                    logger.warning("Vision returned None - PTZ control may be unavailable")

                # Extend system prompt with vision capability
                if self.system_prompt:
                    self.system_prompt += """

=== VISION (PTZ-Kamera System) ===
Du hast ein AUGE mit PTZ!
- Sonoff CAM-PT2: HD Kamera mit PTZ (Schwenk/Neigung) + Hailo NPU Inferenz
- Du kannst die PTZ-Kamera steuern um Personen zu folgen
- Bei jeder Nachricht bekommst du Info was du siehst

WICHTIGE VISION-REGEL:
- Du darfst NUR sagen "Ich sehe..." wenn ein [Vision: ...] Tag in der Nachricht ist
- Wenn kein Vision-Tag da ist, sage NICHT dass du etwas siehst
- Keine erfundenen/halluzinierten Vision-Aussagen!
- Beispiele fuer korrekte Antworten:
  - Mit [Vision: Ich sehe jemanden!] -> "Ich sehe dich!"
  - Mit [Vision: Ich sehe niemanden] -> "Ich sehe gerade niemanden."
  - Ohne Vision-Tag -> "Ich kann gerade nicht sehen" oder gar nichts ueber Vision sagen
"""
            except Exception as e:
                logger.error(f"Failed to load Vision: {e}")

            # CRITICAL: Start camera preview OUTSIDE the try block - it must run
            # even if Vision loading fails, because pose detector is independent
            self._start_camera_preview()

            # Load HybridVision pipeline (face recognition)
            self._set_status("Lade Face Recognition...", "#ffff00")
            try:
                self.hybrid_vision = get_hybrid_vision()
                if self.vision and self.vision.connected:
                    self.hybrid_vision.start(
                        on_person_identified=self._on_person_identified,
                        on_unknown_person=self._on_unknown_person,
                        on_person_lost=self._on_person_lost
                    )
                    logger.info("HybridVision pipeline started")
            except Exception as e:
                logger.error(f"Failed to load HybridVision: {e}")

            # Start VisionWorker only if NO Hailo detector is running
            # (They all use Hailo NPU and would conflict)
            if self.hailo_detector is None and self.hailo_pose_detector is None:
                self._set_status("Starte Vision Worker (Hailo-10H)...", "#ffff00")
                try:
                    from core.vision.vision_worker import get_vision_worker
                    self.vision_worker = get_vision_worker()
                    # Start in background thread (non-blocking)
                    threading.Thread(target=self.vision_worker.start, daemon=True).start()
                    logger.info("VisionWorker starting in background")
                except Exception as e:
                    logger.error(f"Failed to start VisionWorker: {e}")
                    self.vision_worker = None
            else:
                logger.info("Skipping VisionWorker - Hailo detector already using NPU")

            # Status zusammenfassen
            cam_status = []
            if self.hailo_pose_detector or self.hailo_detector:
                cam_status.append("Sonoff+NPU")
            elif self.camera_controller and self.camera_controller.sonoff_connected:
                cam_status.append("Sonoff")

            vision_status = "+".join(cam_status) if cam_status else "blind"
            whisper_backend = self.whisper.backend if self.whisper else "none"
            self._set_status(f"Bereit ({vision_status}, STT: {whisper_backend})", "#00ff88")

        threading.Thread(target=load, daemon=True).start()

    def _set_status(self, text: str, color: str = "#00ff88"):
        """Update status label (thread-safe)."""
        self.window.after(0, lambda: self.status_label.config(text=text, fg=color))

    def _append_response(self, text: str, tag: str = None):
        """Append text to response area with optional color tag (thread-safe)."""
        def update():
            self.response_text.config(state="normal")
            if tag:
                self.response_text.insert("end", text + "\n\n", tag)
            else:
                self.response_text.insert("end", text + "\n\n")
            self.response_text.see("end")
            self.response_text.config(state="disabled")
        self.window.after(0, update)


    def _init_ptz(self):
        """Initialize PTZ controller - starts in MANUAL mode."""
        try:
            self.ptz_controller = get_ptz_controller()
            if self.ptz_controller.connect():
                logger.info("PTZ Controller verbunden")
                self.ptz_controller.set_mode(ControlMode.MANUAL_OVERRIDE)  # Start in manual mode
                if not self.ptz_orchestrator:
                    self.ptz_orchestrator = get_ptz_orchestrator()
                self.ptz_orchestrator.set_camera(self.ptz_controller)
                self._update_ptz_buttons(manual=True)
                self.window.after(0, lambda: self.ptz_status.config(text="PTZ: MANUAL", fg="#00ff88"))
                return True
            else:
                logger.warning("PTZ Verbindung fehlgeschlagen")
                self.window.after(0, lambda: self.ptz_status.config(text="PTZ: FAIL", fg="#ff4444"))
        except Exception as e:
            logger.error(f"PTZ init error: {e}")
            self.window.after(0, lambda: self.ptz_status.config(text="PTZ: ERR", fg="#ff4444"))
        return False
    
    def _update_ptz_buttons(self, manual: bool):
        """Enable/disable manual PTZ buttons based on mode."""
        state = "normal" if manual else "disabled"
        color = "#16213e" if manual else "#333333"
        try:
            self.btn_left.config(state=state, bg=color)
            self.btn_up.config(state=state, bg=color)
            self.btn_down.config(state=state, bg=color)
            self.btn_right.config(state=state, bg=color)
            self.btn_center.config(state=state)
        except:
            pass
    
    def _ptz_move(self, direction: str):
        """Manual PTZ movement - only works in MANUAL mode."""
        if not self.ptz_controller:
            self._init_ptz()
        if not self.ptz_controller:
            logger.warning(f"PTZ move {direction} - no controller")
            return
        # === Debug: Verify this is same controller as tracker uses ===
        logger.info(f"[GUI MANUAL] direction={direction}")
        logger.info(f"[GUI MANUAL] ptz_controller id={id(self.ptz_controller)}")
        if self.autonomous_tracker:
            logger.info(f"[GUI MANUAL] tracker.camera id={id(self.autonomous_tracker.camera) if self.autonomous_tracker.camera else 'None'}")
        logger.info(f"[GUI MANUAL] ptz_service id={id(self.ptz_controller.ptz_service) if self.ptz_controller.ptz_service else 'None'}")
        logger.info(f"[GUI MANUAL] Calling move_manual('{direction}')...")
        if self.ptz_controller.move_manual(direction):
            logger.info(f"[GUI MANUAL] move_manual returned True - camera should move!")
            self.window.after(0, lambda d=direction: self.ptz_status.config(
                text=f"PTZ: {d.upper()}", fg="#00ff88"))
        else:
            logger.warning(f"[GUI MANUAL] move_manual returned False!")
    
    def _ptz_center(self):
        """Center the PTZ camera - works in any mode."""
        if not self.ptz_controller:
            self._init_ptz()
        if not self.ptz_controller:
            logger.warning("PTZ center - no controller")
            return
        self.ptz_controller.center()
        self.window.after(0, lambda: self.ptz_status.config(
            text="PTZ: CENTER", fg="#00ffff"))

    def _test_continuous_move(self):
        """Test ContinuousMove directly - for debugging PTZ issues."""
        if not self.ptz_controller:
            self._init_ptz()
        if not self.ptz_controller:
            logger.warning("PTZ test - no controller")
            return

        logger.info("=" * 60)
        logger.info("=== MANUAL ContinuousMove TEST (left, 0.3 speed, 2 sec) ===")
        logger.info(f"Controller ID: {id(self.ptz_controller)}")
        logger.info(f"ptz_service ID: {id(self.ptz_controller.ptz_service) if self.ptz_controller.ptz_service else 'None'}")
        logger.info(f"profile_token: {self.ptz_controller.profile_token}")
        logger.info("=" * 60)

        # Call test_continuous_move which logs full request details
        result = self.ptz_controller.test_continuous_move("left", speed=0.3, duration=2.0)

        logger.info(f"=== TEST RESULT: {result} ===")
        self.window.after(0, lambda: self.ptz_status.config(
            text=f"PTZ: TEST {'OK' if result else 'FAIL'}", fg="#ff8800"))

    def _start_calibration(self):
        """Start PTZ calibration mode with dialog."""
        if not self.ptz_controller:
            self._init_ptz()
        if not self.ptz_controller:
            logger.warning("Cannot start calibration - no PTZ controller")
            return

        # Disable auto-tracking during calibration
        if self.auto_tracking:
            self.auto_track_var.set(False)
            self._toggle_tracking()

        # Initialize calibration system
        if not self.ptz_calibration:
            self.ptz_calibration = get_ptz_calibration()

        self.ptz_calibration.set_camera(self.ptz_controller)

        # Create calibration dialog
        self._show_calibration_dialog()

    def _show_calibration_dialog(self):
        """Show calibration dialog window."""
        if self.calibration_dialog and self.calibration_dialog.winfo_exists():
            self.calibration_dialog.lift()
            return

        # Create dialog window
        self.calibration_dialog = tk.Toplevel(self.window)
        self.calibration_dialog.title("PTZ Kalibrierung")
        self.calibration_dialog.geometry("400x350")
        self.calibration_dialog.configure(bg="#1a1a2e")
        self.calibration_dialog.transient(self.window)

        # Title
        title = tk.Label(
            self.calibration_dialog,
            text="PTZ Limit Kalibrierung",
            font=("Helvetica", 14, "bold"),
            fg="#00ffff",
            bg="#1a1a2e"
        )
        title.pack(pady=10)

        # Instructions label
        self.cal_instruction = tk.Label(
            self.calibration_dialog,
            text="Kalibrierung ermittelt die echten\nmechanischen Grenzen der Kamera.",
            font=("Helvetica", 10),
            fg="#ffffff",
            bg="#1a1a2e",
            wraplength=350,
            justify="center"
        )
        self.cal_instruction.pack(pady=10)

        # Current position display
        pos_frame = tk.Frame(self.calibration_dialog, bg="#0f0f23", bd=1, relief="sunken")
        pos_frame.pack(pady=10, padx=20, fill="x")

        self.cal_position_label = tk.Label(
            pos_frame,
            text="Position: Pan=0.00, Tilt=0.00",
            font=("Courier", 10),
            fg="#00ff88",
            bg="#0f0f23"
        )
        self.cal_position_label.pack(pady=5)

        # Step indicator
        self.cal_step_label = tk.Label(
            self.calibration_dialog,
            text="Schritt: --",
            font=("Courier", 10, "bold"),
            fg="#ffff00",
            bg="#1a1a2e"
        )
        self.cal_step_label.pack(pady=5)

        # PTZ control buttons for calibration (manual movement)
        btn_frame = tk.Frame(self.calibration_dialog, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        btn_style = {"font": ("Courier", 12, "bold"), "width": 3, "height": 1,
                     "bg": "#16213e", "fg": "white", "activebackground": "#0f3460"}

        tk.Button(btn_frame, text="<", command=lambda: self._cal_move("left"), **btn_style).grid(row=1, column=0, padx=2)
        tk.Button(btn_frame, text="^", command=lambda: self._cal_move("up"), **btn_style).grid(row=0, column=1, padx=2)
        tk.Button(btn_frame, text="v", command=lambda: self._cal_move("down"), **btn_style).grid(row=2, column=1, padx=2)
        tk.Button(btn_frame, text=">", command=lambda: self._cal_move("right"), **btn_style).grid(row=1, column=2, padx=2)

        # Action buttons
        action_frame = tk.Frame(self.calibration_dialog, bg="#1a1a2e")
        action_frame.pack(pady=15)

        self.cal_start_btn = tk.Button(
            action_frame,
            text="START",
            font=("Helvetica", 11, "bold"),
            fg="white",
            bg="#0f3460",
            activebackground="#16213e",
            width=10,
            command=self._cal_start
        )
        self.cal_start_btn.pack(side="left", padx=5)

        self.cal_confirm_btn = tk.Button(
            action_frame,
            text="BESTAETIGEN",
            font=("Helvetica", 11, "bold"),
            fg="white",
            bg="#006600",
            activebackground="#004400",
            width=12,
            state="disabled",
            command=self._cal_confirm
        )
        self.cal_confirm_btn.pack(side="left", padx=5)

        tk.Button(
            action_frame,
            text="ABBRECHEN",
            font=("Helvetica", 11, "bold"),
            fg="white",
            bg="#660000",
            activebackground="#440000",
            width=10,
            command=self._cal_cancel
        ).pack(side="left", padx=5)

        # Status/result area
        self.cal_result_label = tk.Label(
            self.calibration_dialog,
            text="",
            font=("Courier", 9),
            fg="#888888",
            bg="#1a1a2e",
            wraplength=350
        )
        self.cal_result_label.pack(pady=5)

        # Start position update loop
        self._cal_update_position()

        # Set callbacks
        self.ptz_calibration.on_step_change = self._on_cal_step_change
        self.ptz_calibration.on_complete = self._on_cal_complete

    def _cal_update_position(self):
        """Update position display in calibration dialog."""
        if not self.calibration_dialog or not self.calibration_dialog.winfo_exists():
            return

        if self.ptz_calibration and self.ptz_controller:
            pos = self.ptz_calibration.get_current_position()
            if pos:
                self.cal_position_label.config(
                    text=f"Position: Pan={pos['pan']:.2f}, Tilt={pos['tilt']:.2f}"
                )

        # Schedule next update
        self.calibration_dialog.after(500, self._cal_update_position)

    def _cal_move(self, direction: str):
        """Move camera during calibration."""
        if self.ptz_controller:
            self.ptz_controller.move_manual(direction)

    def _cal_start(self):
        """Start calibration process."""
        if self.ptz_calibration:
            self.ptz_calibration.start_calibration()
            self.cal_start_btn.config(state="disabled")
            self.cal_confirm_btn.config(state="normal")

    def _cal_confirm(self):
        """Confirm current position for calibration step."""
        if self.ptz_calibration:
            self.ptz_calibration.confirm_position()

    def _cal_cancel(self):
        """Cancel calibration and close dialog."""
        if self.ptz_calibration:
            self.ptz_calibration.cancel_calibration()
        if self.calibration_dialog:
            self.calibration_dialog.destroy()
            self.calibration_dialog = None

    def _on_cal_step_change(self, step: CalibrationStep, instruction: str):
        """Callback when calibration step changes."""
        if not self.calibration_dialog or not self.calibration_dialog.winfo_exists():
            return

        step_names = {
            CalibrationStep.PAN_LEFT: "1/4: Links",
            CalibrationStep.PAN_RIGHT: "2/4: Rechts",
            CalibrationStep.TILT_UP: "3/4: Oben",
            CalibrationStep.TILT_DOWN: "4/4: Unten",
            CalibrationStep.COMPLETE: "Fertig!",
        }

        self.cal_step_label.config(text=f"Schritt: {step_names.get(step, step.value)}")
        self.cal_instruction.config(text=instruction if instruction else "Kalibrierung abgeschlossen!")

        if step == CalibrationStep.COMPLETE:
            self.cal_confirm_btn.config(state="disabled")
            self.cal_start_btn.config(state="normal", text="NEU")

    def _on_cal_complete(self, limits):
        """Callback when calibration is complete."""
        if not self.calibration_dialog or not self.calibration_dialog.winfo_exists():
            return

        result_text = (
            f"Pan: {limits.pan_min:.1f} bis {limits.pan_max:.1f} ({limits.pan_range:.1f} Grad)\n"
            f"Tilt: {limits.tilt_min:.1f} bis {limits.tilt_max:.1f} ({limits.tilt_range:.1f} Grad)"
        )
        self.cal_result_label.config(text=result_text, fg="#00ff88")

        # Update controller with new limits
        if self.ptz_controller:
            self.ptz_controller.calibration.pan_min = limits.pan_min
            self.ptz_controller.calibration.pan_max = limits.pan_max
            self.ptz_controller.calibration.tilt_min = limits.tilt_min
            self.ptz_controller.calibration.tilt_max = limits.tilt_max
            self.ptz_controller.calibration.verified = True
            logger.info("Updated controller with calibrated limits")

        # Update tracker config if exists
        if self.autonomous_tracker:
            # Tracker will use controller's limits through the camera reference
            logger.info("Tracker will use updated controller limits")

    def _toggle_tracking(self):
        """Toggle between MANUAL and AUTONOMOUS PTZ control modes."""
        self.auto_tracking = self.auto_track_var.get()

        if not self.ptz_controller:
            self._init_ptz()

        # Initialize AutonomousTracker if needed
        if not self.autonomous_tracker:
            self.autonomous_tracker = get_autonomous_tracker()
            if self.ptz_controller:
                self.autonomous_tracker.set_camera(self.ptz_controller)

        if self.auto_tracking:
            # AUTONOMOUS mode - AutonomousTracker controls PTZ with ContinuousMove
            if self.ptz_controller:
                self.ptz_controller.set_mode(ControlMode.AUTONOMOUS)

            # === CRITICAL: Verify SAME controller instance ===
            logger.info("=" * 60)
            logger.info("=== CONTROLLER INSTANCE VERIFICATION ===")
            gui_id = id(self.ptz_controller) if self.ptz_controller else None
            tracker_id = id(self.autonomous_tracker.camera) if self.autonomous_tracker and self.autonomous_tracker.camera else None
            logger.info(f"  GUI ptz_controller id:      {gui_id}")
            logger.info(f"  Tracker.camera id:          {tracker_id}")
            logger.info(f"  SAME INSTANCE:              {gui_id == tracker_id}")
            if self.ptz_controller:
                logger.info(f"  ptz_service id:             {id(self.ptz_controller.ptz_service) if self.ptz_controller.ptz_service else 'None'}")
                logger.info(f"  profile_token:              {self.ptz_controller.profile_token}")
                logger.info(f"  is_connected:               {self.ptz_controller.is_connected}")
            logger.info("=" * 60)

            # Start the 15 Hz tracking thread
            if self.autonomous_tracker:
                self.autonomous_tracker.set_camera(self.ptz_controller)
                self.autonomous_tracker.start()

            self._update_ptz_buttons(manual=False)
            logger.info("[GUI] Mode: AUTONOMOUS - 15Hz ContinuousMove tracking")
            self.ptz_status.config(text="PTZ: AUTO", fg="#ffff00")
        else:
            # MANUAL mode - Stop tracking, buttons control PTZ
            if self.autonomous_tracker:
                self.autonomous_tracker.stop()

            if self.ptz_controller:
                self.ptz_controller.set_mode(ControlMode.MANUAL_OVERRIDE)
                self.ptz_controller.stop()  # Stop any movement

            self._update_ptz_buttons(manual=True)
            logger.info("[GUI] Mode: MANUAL - Buttons control PTZ")
            self.ptz_status.config(text="PTZ: MANUAL", fg="#00ff88")
    def _do_tracking(self, person_x: int, frame_width: int = 1920, confidence: float = 0.0):
        """Send vision event to MPO for tracking decision."""
        if not self.ptz_orchestrator:
            logger.warning("_do_tracking: No ptz_orchestrator!")
            return
        try:
            event = VisionEvent(
                detection_found=True,
                target_center_x=person_x,
                frame_center_x=frame_width // 2,
                frame_width=frame_width,
                confidence=confidence
            )
            decision = self.ptz_orchestrator.process_vision_event(event)
            logger.info(f"MPO_DECISION: action={decision.action} move={decision.should_move} err={decision.error_x} vel={decision.velocity:.2f} reason={decision.reason}")
            if decision.should_move:
                self.window.after(0, lambda a=decision.action: self.ptz_status.config(text=f"PTZ: {a.upper()}", fg="#ffff00"))
            else:
                self.window.after(0, lambda r=decision.reason: self.ptz_status.config(text=f"PTZ: {r}", fg="#00ff88"))
        except Exception as e:
            logger.error(f"MPO tracking error: {e}")

    def _toggle_recording(self, event):
        """Toggle recording on/off with single click."""
        if self.is_recording:
            self._stop_recording(event)
        else:
            self._start_recording(event)

    def _start_recording(self, event):
        """Start recording when button is pressed using Pipewire."""
        if self.whisper is None or not self.whisper.is_available:
            self._set_status("Whisper noch nicht geladen!", "#ff0000")
            return

        # Guard: Don't start new recording while processing previous one
        if self._processing:
            self._set_status("Noch in Bearbeitung...", "#ffff00")
            return

        self.is_recording = True

        # Update UI sofort (kein Blocking!)
        self.talk_button.config(bg="#e94560", text="🔴 AUFNAHME...")
        self._set_status("Aufnahme läuft...", "#ff0000")

        # Create temp file for recording
        self.temp_audio_path = f"/tmp/moloch_ptt_{os.getpid()}.wav"

        # Start pw-record mit Default Audio-Quelle
        # (USB Composite Device oder SmartMic - je nach wpctl set-default)
        try:
            self.record_process = subprocess.Popen(
                [
                    "pw-record",
                    "--channels", "1",
                    "--rate", str(RATE),
                    self.temp_audio_path
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Recording started (default source): {self.temp_audio_path}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._set_status(f"Mikrofon Fehler: {e}", "#ff0000")
            self.is_recording = False

    def _stop_recording(self, event):
        """Stop recording and process audio."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Update UI sofort (kein Blocking auf dem Hauptthread!)
        self.talk_button.config(bg="#16213e", text="🎤 SPRECHEN")
        self._set_status("Verarbeite...", "#ffff00")

        # Alles im Background-Thread: Stop Recording → NPU Acquire → Transkription
        threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self):
        """Process recorded audio: transcribe and chat.

        Runs entirely in background thread to keep GUI responsive.
        Flow: Stop recording → Acquire NPU → Transcribe → Chat → Release NPU

        CRITICAL: Everything in ONE try/finally to guarantee NPU release!
        """
        import traceback

        self._processing = True
        self._npu_acquired_for_voice = False
        processed_path = None

        try:
            # 1. Stop pw-record
            if self.record_process:
                try:
                    self.record_process.send_signal(signal.SIGINT)
                    self.record_process.wait(timeout=3)
                    logger.info("Recording stopped")
                except Exception as e:
                    logger.error(f"Error stopping recording: {e}")
                    try:
                        self.record_process.kill()
                    except:
                        pass

            # Kurz warten bis WAV komplett geschrieben
            time.sleep(0.2)

            if not self.temp_audio_path or not os.path.exists(self.temp_audio_path):
                self._set_status("Keine Aufnahme", "#ff0000")
                return

            # Check file size (too small = no actual recording)
            file_size = os.path.getsize(self.temp_audio_path)
            if file_size < 1000:  # Less than 1KB
                self._set_status("Aufnahme zu kurz", "#ff0000")
                return

            logger.info(f"Processing audio: {self.temp_audio_path} ({file_size} bytes)")

            # 2. NPU exklusiv für Sprache reservieren
            self._set_status("NPU reservieren...", "#ffff00")
            try:
                hailo_mgr = get_hailo_manager()
                if hailo_mgr.acquire_for_voice(timeout=10.0):
                    self._npu_acquired_for_voice = True
                    logger.info("[PTT] NPU exklusiv für Sprache - Vision gestoppt")
                else:
                    logger.warning("[PTT] NPU acquire failed - Whisper nutzt CPU")
            except Exception as e:
                logger.error(f"[PTT] NPU acquire error: {e}")

            # 3. Raw audio direkt an Whisper (kein Preprocessing!)
            # Audio-Pipeline war schuld an schlechter Qualität (SNR:0dB, Quality:50%)
            audio_for_whisper = self.temp_audio_path

            # 4. Transcription
            backend = self.whisper.backend if self.whisper else "unknown"
            self._set_status(f"Transkribiere ({backend})...", "#ffff00")
            text = self.whisper.transcribe(
                audio_for_whisper,
                language="de",
                timeout_ms=30000,  # 30s
                npu_already_acquired=self._npu_acquired_for_voice
            )

            if not text:
                self._set_status("Nichts verstanden", "#ff0000")
                return

            logger.info(f"Transcribed: {text}")
            self._append_response(f"Du: {text}", "user")

            # Log STT event
            get_timeline().stt_transcribe(len(text))
            get_timeline().user_input(len(text), interface="voice")

            # 5. NPU freigeben VOR Claude-API (braucht keine NPU mehr!)
            #    So kann Vision schon während Claude-Request weiterlaufen
            if self._npu_acquired_for_voice:
                try:
                    hailo_mgr = get_hailo_manager()
                    hailo_mgr.release_voice(restart_vision=True)
                    logger.info("[PTT] NPU freigegeben - Vision wird neu gestartet")
                except Exception as e:
                    logger.error(f"[PTT] NPU release error: {e}\n{traceback.format_exc()}")
                finally:
                    self._npu_acquired_for_voice = False

            # 6. Send to Claude (kein NPU nötig)
            self._set_status("M.O.L.O.C.H. denkt...", "#ffff00")
            response = self._chat_with_claude(text)

            if response:
                self._append_response(f"M.O.L.O.C.H.: {response}", "moloch")
                get_timeline().assistant_response(len(response))
                self._set_status("Bereit", "#00ff88")

                # 7. Speak response (kein NPU nötig)
                if self.tts and self.tts.enabled:
                    self._set_status("Spricht...", "#00ffff")
                    self._speak(response, blocking=True)
                    self._set_status("Bereit", "#00ff88")
            else:
                self._set_status("Keine Antwort", "#ff0000")

        except Exception as e:
            logger.error(f"[PTT] Processing error: {e}\n{traceback.format_exc()}")
            self._set_status(f"Fehler: {e}", "#ff0000")

        finally:
            # GARANTIERT: Cleanup + NPU release
            try:
                if self.temp_audio_path:
                    os.unlink(self.temp_audio_path)
            except:
                pass
            try:
                if processed_path and processed_path != self.temp_audio_path:
                    os.unlink(processed_path)
            except:
                pass

            # NPU release (falls nicht schon freigegeben in Schritt 5)
            if self._npu_acquired_for_voice:
                try:
                    hailo_mgr = get_hailo_manager()
                    hailo_mgr.release_voice(restart_vision=True)
                    logger.info("[PTT] NPU freigegeben (cleanup)")
                except Exception as e:
                    logger.error(f"[PTT] NPU release cleanup error: {e}")
                finally:
                    self._npu_acquired_for_voice = False

            # Update UI
            def update_ui_after_voice():
                if self.hailo_pose_detector and self.hailo_pose_detector._running:
                    self.sonoff_label.config(text="640x640 POSE @ 15fps", fg="#00ffff")
                elif self.hailo_detector and self.hailo_detector._running:
                    self.sonoff_label.config(text="640x640 NPU @ 15fps", fg="#00ff88")
                else:
                    self.sonoff_label.config(text="Vision startet...", fg="#ffaa00")

            self.window.after(500, update_ui_after_voice)
            self._processing = False


    def _send_text_message(self, event):
        """Send typed text message to Claude."""
        text = self.text_input.get().strip()
        logger.info(f"[TEXT] _send_text_message called, text='{text}', claude_client={self.claude_client is not None}")
        if not text:
            return

        self.text_input.delete(0, tk.END)
        self._append_response(f"Du: {text}", "user")  # ROT
        self._set_status("Sende...", "#ffaa00")

        def process():
            try:
                logger.info(f"[TEXT] Sending to Claude: '{text[:50]}'")
                response = self._chat_with_claude(text)
                logger.info(f"[TEXT] Claude response: '{response[:80] if response else 'None'}'")
                self.window.after(0, lambda r=response: self._append_response(f"M.O.L.O.C.H.: {r}", "moloch"))  # GRÜN
                self.window.after(0, lambda: self._set_status("Bereit", "#00ff88"))
                if self.tts:
                    self._speak(response, blocking=True)
            except Exception as e:
                logger.error(f"[TEXT] Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.window.after(0, lambda err=e: self._append_response(f"Fehler: {err}", "system"))  # ORANGE
                self.window.after(0, lambda: self._set_status("Fehler", "#ff0000"))

        threading.Thread(target=process, daemon=True).start()

    def _chat_with_claude(self, user_text: str) -> str:
        """Send message to Claude and get response."""
        if not self.claude_client:
            return "Claude API nicht verfügbar"

        try:
            # Add vision context from VisionContext
            message_content = user_text
            vision_info = None

            # Try VisionContext (from VisionWorker/Hailo)
            try:
                from context.vision_context import get_vision_context
                ctx = get_vision_context()
                state = ctx.get_state()
                if state.camera_connected and state.last_update > 0:
                    vision_info = ctx.describe()
            except Exception:
                pass

            # Fallback to legacy Vision
            if not vision_info and self.vision and self.vision.connected:
                vision_info = self.vision.describe_what_i_see()

            if vision_info:
                message_content = f"[Vision: {vision_info}]\n\nMarkus sagt: {user_text}"

            self.conversation_history.append({
                "role": "user",
                "content": message_content
            })

            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,  # Shorter for voice
                system=self.system_prompt or "Du bist M.O.L.O.C.H., ein frecher Hauskobold.",
                messages=self.conversation_history
            )

            assistant_message = response.content[0].text

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            logger.error(f"Claude error: {e}")
            return f"Fehler: {e}"

    # === HYBRID VISION CALLBACKS ===

    def _on_person_identified(self, event: PersonEvent):
        """Handle known person identification."""
        logger.info(f"Person identified: {event.person_name} ({event.confidence:.0%})")
        self.last_recognized_person = event.person_name
        self.last_recognition_time = time.time()

        # Update Sonoff label with face recognition result
        self.window.after(0, lambda: self.sonoff_label.config(
            text=f"Erkannt: {event.person_name} ({event.confidence:.0%})",
            fg="#00ff88"
        ))

        # TTS greeting (only if not recently announced)
        if self.tts and self.tts.enabled:
            self._speak(f"Hallo {event.person_name}!", blocking=False)

    def _on_unknown_person(self, event: PersonEvent):
        """Handle unknown person detection."""
        logger.info(f"Unknown person detected (conf={event.confidence:.0%})")

        # Update Sonoff label
        self.window.after(0, lambda: self.sonoff_label.config(
            text="Unbekannte Person",
            fg="#ffaa00"
        ))

    def _on_person_lost(self):
        """Handle person leaving camera view."""
        logger.info("Person lost from view")
        self.last_recognized_person = None

        # Update Sonoff label
        self.window.after(0, lambda: self.sonoff_label.config(
            text="Niemand sichtbar",
            fg="#888888"
        ))

    def _start_opencv_fallback(self):
        """Start OpenCV RTSP fallback for display when Hailo fails."""
        if self._fallback_started:
            return
        self._fallback_started = True

        logger.warning("=" * 60)
        logger.warning("[FALLBACK] Hailo not delivering frames - starting OpenCV RTSP")
        logger.warning("=" * 60)

        try:
            import cv2
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

            self._sonoff_cap = cv2.VideoCapture(self._sonoff_rtsp, cv2.CAP_FFMPEG)
            self._sonoff_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self._sonoff_cap.isOpened():
                self._fallback_active = True
                logger.info("[FALLBACK] OpenCV RTSP connected!")
                self.sonoff_label.config(text="RTSP Fallback", fg="#ffaa00")
            else:
                logger.error("[FALLBACK] OpenCV RTSP failed to open!")
                self.sonoff_label.config(text="RTSP FEHLER", fg="#ff4444")
        except Exception as e:
            logger.error(f"[FALLBACK] OpenCV error: {e}")

    def _update_fallback_frame(self):
        """Get and display frame from OpenCV fallback capture."""
        if not self._sonoff_cap or not self._sonoff_cap.isOpened():
            logger.warning("[FALLBACK] Capture not open!")
            return False

        try:
            import cv2

            # Track fallback frames
            if not hasattr(self, '_fallback_frame_count'):
                self._fallback_frame_count = 0

            # Read frame directly (grab already done by buffer flush)
            ret, frame = self._sonoff_cap.read()

            if not ret or frame is None:
                logger.warning(f"[FALLBACK] read() failed: ret={ret}")
                # Try to reconnect
                self._sonoff_cap.release()
                self._sonoff_cap = cv2.VideoCapture(self._sonoff_rtsp, cv2.CAP_FFMPEG)
                self._sonoff_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return False

            self._fallback_frame_count += 1
            if self._fallback_frame_count <= 3:
                logger.info(f"[FALLBACK] Frame {self._fallback_frame_count}: {frame.shape}")

            # Resize and convert BGR -> RGB
            img = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Add "FALLBACK" indicator + frame count
            cv2.putText(img, f"FALLBACK #{self._fallback_frame_count}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

            pil_img = Image.fromarray(img)
            self.sonoff_photo = ImageTk.PhotoImage(pil_img)

            if self.sonoff_image_id:
                self.sonoff_canvas.itemconfig(self.sonoff_image_id, image=self.sonoff_photo)
            else:
                # Clear any placeholder first
                self.sonoff_canvas.delete("all")
                self.sonoff_image_id = self.sonoff_canvas.create_image(
                    0, 0, anchor="nw", image=self.sonoff_photo)
            return True

        except Exception as e:
            logger.error(f"[FALLBACK] Frame error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return False

    def _process_frame_queue(self):
        """Process queued frames from Hailo callback in main thread."""
        try:
            # Debug: First call log
            if not hasattr(self, '_frame_process_count'):
                self._frame_process_count = 0
                self._hailo_first_frame_time = time.time()
                logger.info("[FRAME_PROC] Queue processor started!")

            # === NO FALLBACK - Hailo only! Log errors ===
            if self._gui_frame_count == 0:
                elapsed = time.time() - self._hailo_first_frame_time
                if elapsed > 3.0 and int(elapsed) % 3 == 0:
                    logger.error(f"[ERROR] NO HAILO FRAMES after {elapsed:.0f}s! Check detector.")

            # === HAILO MODE: Process queue ===
            processed = 0
            for _ in range(3):
                try:
                    frame, result = self._frame_queue.get_nowait()
                    processed += 1
                    # Log first few frames
                    if processed == 1 and self._frame_process_count < 5:
                        logger.info(f"[FRAME_PROC] Got frame! shape={frame.shape if frame is not None else 'None'}")
                    if result is not None:
                        # Pose mode with PoseDetectionResult
                        self._update_sonoff_display_pose(frame, result)
                    else:
                        # Detection mode (no pose data)
                        self._update_sonoff_display_fast(frame)
                except queue.Empty:
                    break

            self._frame_process_count += 1
            # Log every 30 cycles (~1 second)
            if self._frame_process_count % 30 == 0:
                qsize = self._frame_queue.qsize()
                logger.info(f"[QUEUE] processed={processed}, qsize={qsize}, gui_frames={self._gui_frame_count}")
        except Exception as e:
            logger.error(f"Frame queue processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Reschedule every 33ms (~30 FPS)
            self.window.after(33, self._process_frame_queue)

    def run(self):
        """Start the GUI."""
        logger.info("Starting Push-to-Talk GUI")
        timeline = get_timeline()
        timeline.system_startup(component="push_to_talk")
        timeline.system_start("push_to_talk")

        # Start frame queue processor in main loop
        self.window.after(100, self._process_frame_queue)

        self.window.mainloop()

        # Cleanup
        logger.info("Shutting down...")
        self.camera_running = False

        if self.autonomous_tracker:
            self.autonomous_tracker.stop()

        if self.hailo_pose_detector:
            self.hailo_pose_detector.stop()

        if self.hailo_detector:
            self.hailo_detector.stop()

        if self.hybrid_vision:
            self.hybrid_vision.stop()

        if self.vision_worker:
            self.vision_worker.stop()

        # Release OpenCV fallback capture
        if self._sonoff_cap:
            self._sonoff_cap.release()
            logger.info("OpenCV fallback capture released")

        timeline.system_stop("push_to_talk")
        timeline.system_shutdown(reason="manual")


def main():
    """Main entry point."""
    gui = PushToTalkGUI()
    gui.run()


if __name__ == "__main__":
    main()
