#!/usr/bin/env python3
"""
M.O.L.O.C.H. Vision Calibration Lab
====================================

Standalone tool for visual calibration and model testing.
Does NOT interfere with push_to_talk, STT, or other systems.

Usage:
    python scripts/moloch_vision_lab.py

Keyboard Controls:
    0 = RAW (no detection - latency test)
    1 = PERSON detection
    2 = FACE detection
    3 = HANDS detection
    4 = IDENTITY (face + embedding match)
    5 = FULL STACK (all models)
    6 = POSE (keypoint detection)
    R = Reload config
    S = Save runtime config
    +/- = Adjust active threshold (mode-dependent)
    Q = Quit

Author: M.O.L.O.C.H. System
Date: 2026-02-05
Updated: 2026-02-06 (Usability Fix: trackbars, per-mode params, confidence, POSE mode)
"""

import sys
import os
import json
import time
import copy
import logging
import atexit
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

# PTZ control imports (optional - Vision Lab works without PTZ)
try:
    from core.hardware.camera import get_camera_controller as get_ptz_controller
    from core.mpo.autonomous_tracker import get_autonomous_tracker
    PTZ_AVAILABLE = True
except ImportError:
    PTZ_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION PATHS
# ============================================================================
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.json"
VISION_MODES_PATH = CONFIG_DIR / "vision_modes.json"
VISION_PIPELINE_PATH = CONFIG_DIR / "vision_pipeline.json"
RUNTIME_CONFIG_PATH = CONFIG_DIR / "vision_runtime_config.json"

# RTSP URL
RTSP_URL = os.environ.get("MOLOCH_RTSP_URL", "rtsp://USER:PASS@CAMERA_IP:554/av_stream/ch0")

# Window names
WIN_MAIN = "M.O.L.O.C.H. Vision Lab"
WIN_CONTROLS = "Controls"


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================
class VisionLabMode(Enum):
    """Available vision lab modes."""
    RAW = 0           # No detection - pure video latency test
    PERSON = 1
    FACE = 2
    HANDS = 3
    IDENTITY = 4
    FULL_STACK = 5
    POSE = 6


@dataclass
class RuntimeConfig:
    """Runtime-adjustable parameters."""
    person_bbox_shrink_factor: float = 0.2
    face_roi_padding: float = 0.1
    identity_threshold: float = 0.65
    min_face_size: int = 80
    hand_confidence_threshold: float = 0.5
    pose_confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_bbox_shrink_factor": self.person_bbox_shrink_factor,
            "face_roi_padding": self.face_roi_padding,
            "identity_threshold": self.identity_threshold,
            "min_face_size": self.min_face_size,
            "hand_confidence_threshold": self.hand_confidence_threshold,
            "pose_confidence_threshold": self.pose_confidence_threshold,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeConfig":
        return cls(
            person_bbox_shrink_factor=data.get("person_bbox_shrink_factor", 0.2),
            face_roi_padding=data.get("face_roi_padding", 0.1),
            identity_threshold=data.get("identity_threshold", 0.65),
            min_face_size=data.get("min_face_size", 80),
            hand_confidence_threshold=data.get("hand_confidence_threshold", 0.5),
            pose_confidence_threshold=data.get("pose_confidence_threshold", 0.25),
            keypoint_confidence_threshold=data.get("keypoint_confidence_threshold", 0.3)
        )

    def __eq__(self, other):
        if not isinstance(other, RuntimeConfig):
            return False
        return self.to_dict() == other.to_dict()

    def diff(self, saved: "RuntimeConfig") -> Dict[str, Tuple[Any, Any]]:
        """Return dict of changed fields: {field: (current, saved)}."""
        changes = {}
        for key in self.to_dict():
            cur = getattr(self, key)
            sav = getattr(saved, key)
            if cur != sav:
                changes[key] = (cur, sav)
        return changes


@dataclass
class DetectionResult:
    """Results from current detection."""
    persons: List[Tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    person_weights: List[float] = field(default_factory=list)  # HOG weights
    faces: List[Tuple[int, int, int, int]] = field(default_factory=list)
    hands: List[Tuple[int, int, int, int]] = field(default_factory=list)
    hand_areas: List[int] = field(default_factory=list)  # contour areas
    landmarks_hand: List[List[Tuple[int, int]]] = field(default_factory=list)
    identity_name: Optional[str] = None
    identity_score: float = 0.0
    # Pose results
    pose_keypoints: List[List[Tuple[float, float, float]]] = field(default_factory=list)  # (x, y, conf) per person
    pose_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    pose_confidences: List[float] = field(default_factory=list)


# ============================================================================
# COLORS (BGR for OpenCV)
# ============================================================================
COLOR_PERSON = (0, 255, 0)      # Green
COLOR_FACE = (255, 0, 0)        # Blue
COLOR_HAND = (0, 255, 255)      # Yellow
COLOR_LANDMARK = (255, 0, 255)  # Magenta
COLOR_CROSSHAIR = (0, 255, 255) # Yellow
COLOR_DEADZONE = (128, 128, 128)  # Gray
COLOR_TEXT_BG = (0, 0, 0)       # Black
COLOR_TEXT = (255, 255, 255)    # White
COLOR_IDENTITY = (0, 255, 0)    # Green for matched identity
COLOR_MODIFIED = (0, 255, 255)  # Yellow for unsaved changes
COLOR_POSE = (0, 200, 255)      # Orange for pose

# COCO 17 keypoint skeleton pairs
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # Head
    (5, 6), (5, 7), (7, 9), (6, 8),     # Arms
    (8, 10), (5, 11), (6, 12),          # Torso
    (11, 12), (11, 13), (13, 15),       # Left leg
    (12, 14), (14, 16)                  # Right leg
]


# ============================================================================
# VISION LAB CLASS
# ============================================================================
class MolochVisionLab:
    """
    Vision Calibration Lab - standalone model tester.

    Uses hailo_manager for NPU access.
    Single-threaded, one model at a time.
    """

    def __init__(self):
        self.running = False
        self.mode = VisionLabMode.PERSON
        self.config = RuntimeConfig()
        self.saved_config = RuntimeConfig()  # Track saved state
        self.model_registry: Dict[str, Any] = {}
        self.current_model_name: str = ""
        self.current_hef_path: str = ""

        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None

        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.fps_start_time = 0.0

        # Detection results
        self.detection = DetectionResult()

        # Frame skip: only run detection every N frames (keeps display responsive)
        self._detect_every_n = 3
        self._frame_index = 0

        # Hailo manager (lazy load)
        self._hailo_manager = None
        self._hailo_acquired = False

        # Identity manager (lazy load)
        self._identity_manager = None

        # Pose detector (lazy load)
        self._pose_detector = None

        # GStreamer pipeline (for Hailo inference)
        self._pipeline = None
        self._pipeline_running = False

        # Button positions for mouse clicks
        self._buttons: List[Tuple[int, int, int, int, VisionLabMode]] = []
        self._ptz_buttons: List[Tuple[int, int, int, int, str]] = []

        # PTZ control state
        self._ptz_enabled = False
        self._ptz_controller = None
        self._ptz_connected = False
        self._ptz_status_text = ""
        self._ptz_status_color = (128, 128, 128)
        self._ptz_status_time = 0.0
        self._ptz_arrow_dir = None
        self._ptz_arrow_time = 0.0

        # Load configs
        self._load_configs()

        logger.info("MolochVisionLab initialized")

    def _load_configs(self):
        """Load all configuration files."""
        # Model Registry
        if MODEL_REGISTRY_PATH.exists():
            with open(MODEL_REGISTRY_PATH) as f:
                data = json.load(f)
                self.model_registry = data.get("models", {})
            logger.info(f"Loaded {len(self.model_registry)} models from registry")
        else:
            logger.warning(f"Model registry not found: {MODEL_REGISTRY_PATH}")

        # Runtime Config
        if RUNTIME_CONFIG_PATH.exists():
            with open(RUNTIME_CONFIG_PATH) as f:
                data = json.load(f)
                self.config = RuntimeConfig.from_dict(data)
                self.saved_config = RuntimeConfig.from_dict(data)
            logger.info("Loaded runtime config")
        else:
            logger.info("Using default runtime config")
            self.saved_config = RuntimeConfig()

    def _save_runtime_config(self):
        """Save runtime config to JSON."""
        try:
            with open(RUNTIME_CONFIG_PATH, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            self.saved_config = RuntimeConfig.from_dict(self.config.to_dict())
            logger.info(f"Runtime config saved to {RUNTIME_CONFIG_PATH}")
            self._sync_trackbars()
            return True
        except Exception as e:
            logger.error(f"Failed to save runtime config: {e}")
            return False

    def _get_hailo_manager(self):
        """Get HailoManager singleton (lazy load)."""
        if self._hailo_manager is None:
            try:
                from core.hardware.hailo_manager import get_hailo_manager
                self._hailo_manager = get_hailo_manager()
                logger.info("HailoManager loaded")
            except ImportError as e:
                logger.error(f"Cannot import HailoManager: {e}")
        return self._hailo_manager

    def _get_identity_manager(self):
        """Get IdentityManager singleton (lazy load)."""
        if self._identity_manager is None:
            try:
                from core.vision.identity_manager import get_identity_manager
                self._identity_manager = get_identity_manager()
                logger.info("IdentityManager loaded")
            except ImportError as e:
                logger.error(f"Cannot import IdentityManager: {e}")
        return self._identity_manager

    def _get_pose_detector(self):
        """Get GstHailoPoseDetector singleton (lazy load)."""
        if self._pose_detector is None:
            try:
                from core.vision.gst_hailo_pose_detector import get_gst_pose_detector
                self._pose_detector = get_gst_pose_detector()
                logger.info("PoseDetector loaded")
            except ImportError as e:
                logger.error(f"Cannot import PoseDetector: {e}")
        return self._pose_detector

    def _acquire_hailo(self) -> bool:
        """Acquire Hailo NPU via manager."""
        if self._hailo_acquired:
            return True

        manager = self._get_hailo_manager()
        if manager is None:
            logger.warning("No HailoManager - running without NPU")
            return False

        if manager.acquire_for_vision(timeout=5.0):
            self._hailo_acquired = True
            logger.info("Hailo NPU acquired for VisionLab")
            return True
        else:
            logger.error("Failed to acquire Hailo NPU")
            return False

    def _release_hailo(self):
        """Release Hailo NPU."""
        if not self._hailo_acquired:
            return

        manager = self._get_hailo_manager()
        if manager:
            manager.release_vision()
            logger.info("Hailo NPU released")

        self._hailo_acquired = False

    def _get_model_for_mode(self, mode: VisionLabMode) -> Tuple[str, str]:
        """Get model name and HEF path for given mode."""
        mode_to_model = {
            VisionLabMode.PERSON: "person_detector",
            VisionLabMode.FACE: "face_detector",
            VisionLabMode.HANDS: "hand_detector",
            VisionLabMode.IDENTITY: "face_embedding",
            VisionLabMode.FULL_STACK: "person_detector",
            VisionLabMode.POSE: "pose_detector"
        }

        model_name = mode_to_model.get(mode, "person_detector")
        model_info = self.model_registry.get(model_name, {})
        hef_path = model_info.get("hef_path", "")

        return model_name, hef_path

    def _stop_pipeline(self):
        """Stop current Hailo pipeline."""
        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, 'set_state'):
                    import gi
                    gi.require_version('Gst', '1.0')
                    from gi.repository import Gst
                    self._pipeline.set_state(Gst.State.NULL)
                elif hasattr(self._pipeline, 'stop'):
                    self._pipeline.stop()

                self._pipeline = None
                logger.info("Pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

        self._pipeline_running = False

    def _switch_mode(self, new_mode: VisionLabMode):
        """Switch to new vision mode with proper cleanup."""
        if new_mode == self.mode and self._pipeline_running:
            logger.info(f"Already in mode {new_mode.name}")
            return

        logger.info(f"Switching mode: {self.mode.name} -> {new_mode.name}")

        # 1. Stop current pipeline
        self._stop_pipeline()

        # 2. Release Hailo (if acquired)
        self._release_hailo()

        # 3. Small delay for cleanup
        time.sleep(0.3)

        # 4. Update mode
        self.mode = new_mode

        # RAW mode = no model needed
        if new_mode == VisionLabMode.RAW:
            self.current_model_name = "NONE (RAW)"
            self.current_hef_path = ""
            self._pipeline_running = True
            logger.info("RAW mode active - no detection, pure video latency test")
            return

        # 5. Get model info
        model_name, hef_path = self._get_model_for_mode(new_mode)

        # 6. Check if model exists
        if not hef_path or not os.path.exists(hef_path):
            logger.error(f"MODEL NOT FOUND: {hef_path or model_name}")
            self.current_model_name = f"NOT FOUND: {model_name}"
            self.current_hef_path = ""
            return

        self.current_model_name = model_name
        self.current_hef_path = hef_path

        # 7. Acquire Hailo
        if not self._acquire_hailo():
            logger.warning("Running in CPU-only mode")

        self._pipeline_running = True
        logger.info(f"Mode {new_mode.name} active, model: {model_name}")

    # ========================================================================
    # PTZ CONTROL
    # ========================================================================

    def _init_ptz(self) -> bool:
        """Initialize PTZ controller."""
        if not PTZ_AVAILABLE:
            logger.warning("PTZ nicht verfuegbar (Imports fehlen)")
            return False
        try:
            self._ptz_controller = get_ptz_controller()
            if not self._ptz_controller.is_connected:
                self._ptz_controller.connect()
            if not self._ptz_controller.is_connected:
                logger.error("PTZ Verbindung fehlgeschlagen")
                return False
            self._ptz_connected = True
            logger.info("PTZ Controller verbunden")
            return True
        except Exception as e:
            logger.error(f"PTZ init Fehler: {e}")
            return False

    def _toggle_ptz(self):
        """Toggle PTZ control mode."""
        if self._ptz_enabled:
            self._disable_ptz()
        else:
            self._enable_ptz()

    def _enable_ptz(self):
        """Enable exclusive PTZ control, stop tracker."""
        if not self._ptz_controller and not self._init_ptz():
            self._set_ptz_status("PTZ: FEHLER", (0, 0, 255))
            return
        # Stop autonomous tracker
        try:
            tracker = get_autonomous_tracker()
            if tracker._running:
                tracker.stop()
                logger.info("AutonomousTracker gestoppt")
        except Exception as e:
            logger.warning(f"Tracker stopp: {e}")
        # Acquire exclusive lock
        if not self._ptz_controller.acquire_exclusive("vision_lab"):
            self._set_ptz_status("PTZ: GESPERRT", (0, 0, 255))
            return
        self._ptz_enabled = True
        self._set_ptz_status("PTZ: BEREIT", (0, 255, 0))
        logger.info("PTZ Steuerung AKTIVIERT (exklusiv)")

    def _disable_ptz(self):
        """Disable PTZ, release lock."""
        if self._ptz_controller:
            self._ptz_controller.stop()
            self._ptz_controller.release_exclusive("vision_lab")
        self._ptz_enabled = False
        self._set_ptz_status("PTZ: AUS", (128, 128, 128))
        logger.info("PTZ Steuerung DEAKTIVIERT")

    def _ptz_move(self, direction: str):
        """Send PTZ move command."""
        if not self._ptz_enabled or not self._ptz_controller:
            return
        dir_names = {"left": "LINKS", "right": "RECHTS", "up": "HOCH", "down": "RUNTER"}
        try:
            self._ptz_controller.move_manual(direction, speed=0.4)
            self._set_ptz_status(f"PTZ: {dir_names.get(direction, direction)}", (0, 255, 0))
            self._ptz_arrow_dir = direction
            self._ptz_arrow_time = time.time()
            logger.info(f"PTZ Bewegung: {dir_names.get(direction, direction)}")
        except Exception as e:
            self._set_ptz_status("PTZ: FEHLER", (0, 0, 255))
            logger.error(f"PTZ Bewegung fehlgeschlagen: {e}")

    def _ptz_center(self):
        """Move PTZ to center."""
        if not self._ptz_enabled or not self._ptz_controller:
            return
        try:
            self._ptz_controller.center()
            self._set_ptz_status("PTZ: MITTE", (0, 255, 255))
            logger.info("PTZ: Zurueck zur Mitte")
        except Exception as e:
            self._set_ptz_status("PTZ: FEHLER", (0, 0, 255))
            logger.error(f"PTZ center fehlgeschlagen: {e}")

    def _ptz_stop(self):
        """Stop PTZ movement."""
        if self._ptz_controller:
            self._ptz_controller.stop()
            self._set_ptz_status("PTZ: STOPP", (0, 255, 255))
            logger.info("PTZ: Gestoppt")

    def _set_ptz_status(self, text: str, color: Tuple[int, int, int]):
        """Set PTZ status for overlay."""
        self._ptz_status_text = text
        self._ptz_status_color = color
        self._ptz_status_time = time.time()

    # ========================================================================
    # CPU DETECTION METHODS
    # ========================================================================

    def _detect_persons_cpu(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """CPU fallback for person detection using HOG. Returns (boxes, weights)."""
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            scale = 0.3
            small = cv2.resize(frame, None, fx=scale, fy=scale)

            boxes, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.1)

            persons = []
            weight_list = []
            for i, (x, y, w, h) in enumerate(boxes):
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                persons.append((x, y, w, h))
                weight_list.append(float(weights[i]) if i < len(weights) else 0.0)

            return persons, weight_list
        except Exception as e:
            logger.debug(f"HOG detection error: {e}")
            return [], []

    def _detect_faces_cpu(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """CPU fallback for face detection using Haar Cascade."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config.min_face_size, self.config.min_face_size)
            )

            return [(x, y, w, h) for (x, y, w, h) in faces]
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []

    def _detect_hands_cpu(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        """CPU fallback for hand detection. Returns (boxes, areas)."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hands = []
            areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    hands.append((x, y, w, h))
                    areas.append(int(area))

            return hands[:2], areas[:2]
        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
            return [], []

    def _compute_embedding_cpu(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Compute face embedding (CPU placeholder)."""
        try:
            face_resized = cv2.resize(face_crop, (112, 112))
            face_normalized = face_resized.astype(np.float32) / 255.0

            flat = face_normalized.flatten()
            embedding = np.zeros(512, dtype=np.float32)

            for i in range(512):
                idx = (i * len(flat)) // 512
                embedding[i] = flat[idx]

            return embedding
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return None

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================

    def _process_frame(self, frame: np.ndarray):
        """Process frame based on current mode."""
        self.detection = DetectionResult()

        if self.mode == VisionLabMode.RAW:
            return

        if self.mode == VisionLabMode.PERSON:
            persons, weights = self._detect_persons_cpu(frame)
            self.detection.persons = persons
            self.detection.person_weights = weights

        elif self.mode == VisionLabMode.FACE:
            self.detection.faces = self._detect_faces_cpu(frame)

        elif self.mode == VisionLabMode.HANDS:
            hands, areas = self._detect_hands_cpu(frame)
            self.detection.hands = hands
            self.detection.hand_areas = areas

        elif self.mode == VisionLabMode.IDENTITY:
            self.detection.faces = self._detect_faces_cpu(frame)
            if self.detection.faces:
                x, y, fw, fh = self.detection.faces[0]
                face_crop = frame[y:y+fh, x:x+fw]
                embedding = self._compute_embedding_cpu(face_crop)
                if embedding is not None:
                    id_mgr = self._get_identity_manager()
                    if id_mgr:
                        name, score = id_mgr.match(embedding)
                        if name and score >= self.config.identity_threshold:
                            self.detection.identity_name = name
                            self.detection.identity_score = score

        elif self.mode == VisionLabMode.FULL_STACK:
            persons, weights = self._detect_persons_cpu(frame)
            self.detection.persons = persons
            self.detection.person_weights = weights
            self.detection.faces = self._detect_faces_cpu(frame)
            hands, areas = self._detect_hands_cpu(frame)
            self.detection.hands = hands
            self.detection.hand_areas = areas

            if self.detection.faces:
                x, y, fw, fh = self.detection.faces[0]
                face_crop = frame[y:y+fh, x:x+fw]
                embedding = self._compute_embedding_cpu(face_crop)
                if embedding is not None:
                    id_mgr = self._get_identity_manager()
                    if id_mgr:
                        name, score = id_mgr.match(embedding)
                        if name and score >= self.config.identity_threshold:
                            self.detection.identity_name = name
                            self.detection.identity_score = score

        elif self.mode == VisionLabMode.POSE:
            # Pose uses GstHailoPoseDetector if available, otherwise CPU person detection
            detector = self._get_pose_detector()
            if detector and hasattr(detector, 'latest_result') and detector._running:
                result = detector.latest_result
                if result and result.detections:
                    for det in result.detections:
                        if det.confidence >= self.config.pose_confidence_threshold:
                            bx, by, bw, bh = det.bbox
                            h, w = frame.shape[:2]
                            self.detection.pose_boxes.append((
                                int(bx * w), int(by * h),
                                int(bw * w), int(bh * h)
                            ))
                            self.detection.pose_confidences.append(det.confidence)
                            kps = []
                            for kp in det.keypoints:
                                kps.append((kp.x * w, kp.y * h, kp.confidence))
                            self.detection.pose_keypoints.append(kps)
            else:
                # CPU fallback: just person detection
                persons, weights = self._detect_persons_cpu(frame)
                self.detection.persons = persons
                self.detection.person_weights = weights

    # ========================================================================
    # DRAW OVERLAYS
    # ========================================================================

    def _draw_text_bg(self, frame, text, pos, font_scale, color, thickness=1, pad=3):
        """Draw text with black background for readability."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = pos
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), COLOR_TEXT_BG, -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return th + pad * 2

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw all visual overlays on frame."""
        h, w = frame.shape[:2]
        F = 0.4  # Basis-Schriftgroesse fuer 640x360

        # ─── DETECTION BOXES ───────────────────────────────────

        # Person (gruen)
        for i, (x, y, bw, bh) in enumerate(self.detection.persons):
            shrink = self.config.person_bbox_shrink_factor
            sx = int(x + bw * shrink / 2)
            sy = int(y + bh * shrink / 2)
            sw = int(bw * (1 - shrink))
            sh = int(bh * (1 - shrink))
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), COLOR_PERSON, 2)
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), COLOR_PERSON, 1)
            weight = self.detection.person_weights[i] if i < len(self.detection.person_weights) else 0.0
            label = f"PERSON {weight:.1f}" if weight > 0 else "PERSON"
            cv2.putText(frame, label, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, F, COLOR_PERSON, 1)

        # Face (blau)
        for i, (x, y, fw, fh) in enumerate(self.detection.faces):
            pad = int(self.config.face_roi_padding * max(fw, fh))
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), COLOR_FACE, 2)
            cv2.rectangle(frame, (max(0,x-pad), max(0,y-pad)),
                         (min(w,x+fw+pad), min(h,y+fh+pad)), COLOR_FACE, 1)
            if self.detection.identity_name and i == 0:
                cv2.putText(frame, f"{self.detection.identity_name} ({self.detection.identity_score:.2f})",
                           (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, F, COLOR_IDENTITY, 1)
            else:
                cv2.putText(frame, f"FACE {fw}x{fh}", (x, y-4),
                           cv2.FONT_HERSHEY_SIMPLEX, F, COLOR_FACE, 1)

        # Hand (gelb)
        for i, (x, y, hw, hh) in enumerate(self.detection.hands):
            cv2.rectangle(frame, (x, y), (x+hw, y+hh), COLOR_HAND, 2)
            area = self.detection.hand_areas[i] if i < len(self.detection.hand_areas) else 0
            cv2.putText(frame, f"HAND {area}" if area else "HAND",
                       (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, F, COLOR_HAND, 1)

        # Hand Landmarks
        for landmarks in self.detection.landmarks_hand:
            for (lx, ly) in landmarks:
                cv2.circle(frame, (lx, ly), 2, COLOR_HAND, -1)

        # Pose (orange)
        for pi, kps in enumerate(self.detection.pose_keypoints):
            conf = self.detection.pose_confidences[pi] if pi < len(self.detection.pose_confidences) else 0.0
            bbox = self.detection.pose_boxes[pi] if pi < len(self.detection.pose_boxes) else None
            if bbox:
                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), COLOR_POSE, 2)
                cv2.putText(frame, f"POSE {conf:.1f}", (bx, by-4),
                           cv2.FONT_HERSHEY_SIMPLEX, F, COLOR_POSE, 1)
            valid_kps = {}
            for ki, (kx, ky, kc) in enumerate(kps):
                if kc >= self.config.keypoint_confidence_threshold:
                    pt = (int(kx), int(ky))
                    cv2.circle(frame, pt, 3, COLOR_POSE, -1)
                    valid_kps[ki] = pt
            for (a, b) in SKELETON_PAIRS:
                if a in valid_kps and b in valid_kps:
                    cv2.line(frame, valid_kps[a], valid_kps[b], COLOR_POSE, 2)

        # ─── FADENKREUZ ───────────────────────────────────────
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx-15, cy), (cx+15, cy), COLOR_CROSSHAIR, 1)
        cv2.line(frame, (cx, cy-15), (cx, cy+15), COLOR_CROSSHAIR, 1)

        # ─── STATUSLEISTE OBEN (1 Zeile, kompakt) ─────────────
        is_modified = self.config != self.saved_config
        mod_tag = " *" if is_modified else ""
        ptz_tag = " | PTZ AN" if self._ptz_enabled else (" | P=PTZ" if PTZ_AVAILABLE else "")
        status = f"{self.mode.name}{mod_tag}{ptz_tag}  |  {self.fps:.0f} FPS"

        # Detection counts inline
        det_parts = []
        if self.detection.persons:
            det_parts.append(f"P:{len(self.detection.persons)}")
        if self.detection.faces:
            det_parts.append(f"F:{len(self.detection.faces)}")
        if self.detection.hands:
            det_parts.append(f"H:{len(self.detection.hands)}")
        if self.detection.pose_boxes:
            det_parts.append(f"Pose:{len(self.detection.pose_boxes)}")
        if det_parts:
            status += "  |  " + " ".join(det_parts)

        status_color = COLOR_MODIFIED if is_modified else COLOR_TEXT
        self._draw_text_bg(frame, status, (5, 16), 0.45, status_color, 1)

        # ─── PARAMETER-ANZEIGE (oben rechts, kompakt) ─────────
        changes = self.config.diff(self.saved_config)
        param_lines = self._get_mode_params()

        y_off = 16
        for label, field_name, fmt in param_lines:
            cur = getattr(self.config, field_name)
            changed = field_name in changes
            color = COLOR_MODIFIED if changed else (180, 180, 180)
            text = f"{label}: {fmt.format(cur)}"
            if changed:
                text += f" (war {fmt.format(changes[field_name][1])})"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, F, 1)
            self._draw_text_bg(frame, text, (w - tw - 5, y_off), F, color)
            y_off += 18

        # Speicher-Hinweis nur wenn geaendert
        if is_modified:
            hint = "S=Speichern  R=Zuruecksetzen"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            self._draw_text_bg(frame, hint, (w - tw - 5, y_off), 0.35, COLOR_MODIFIED)

        # ─── MODUS-BUTTONS (unten) ────────────────────────────
        self._buttons = []
        button_names = ["RAW", "PERSON", "FACE", "HANDS", "IDENT", "FULL", "POSE"]
        button_modes = [VisionLabMode.RAW, VisionLabMode.PERSON, VisionLabMode.FACE,
                       VisionLabMode.HANDS, VisionLabMode.IDENTITY,
                       VisionLabMode.FULL_STACK, VisionLabMode.POSE]

        total = len(button_names)
        margin = 3
        avail = w - 10
        bw = (avail - (total - 1) * margin) // total
        bh = 30
        sx = 5
        sy = h - bh - 3

        for i, (name, mode) in enumerate(zip(button_names, button_modes)):
            x1 = sx + i * (bw + margin)
            x2 = x1 + bw
            y1, y2 = sy, sy + bh
            self._buttons.append((x1, y1, x2, y2, mode))

            active = (mode == self.mode)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (0, 100, 0) if active else (40, 40, 40), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                         (0, 220, 0) if active else (80, 80, 80), 1)

            ts = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            tx = x1 + (bw - ts[0]) // 2
            ty = y1 + (bh + ts[1]) // 2
            cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

        # ─── PTZ CONTROLS (nur wenn PTZ aktiv) ──────────────
        if self._ptz_enabled:
            self._ptz_buttons = []
            ptz_names = ["< Links", "^ Hoch", "v Runter", "> Rechts", "O Mitte"]
            ptz_actions = ["left", "up", "down", "right", "center"]
            ptz_total = len(ptz_names)
            ptz_bw = (avail - (ptz_total - 1) * margin) // ptz_total
            ptz_bh = 26
            ptz_sy = sy - ptz_bh - 4

            for i, (name, action) in enumerate(zip(ptz_names, ptz_actions)):
                x1 = sx + i * (ptz_bw + margin)
                x2 = x1 + ptz_bw
                y1, y2 = ptz_sy, ptz_sy + ptz_bh
                self._ptz_buttons.append((x1, y1, x2, y2, action))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 1)
                ts = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                tx = x1 + (ptz_bw - ts[0]) // 2
                ty = y1 + (ptz_bh + ts[1]) // 2
                cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1)

            # PTZ status text (fades after 1.5s)
            now = time.time()
            if now - self._ptz_status_time < 1.5:
                self._draw_text_bg(frame, self._ptz_status_text, (5, ptz_sy - 8),
                                  0.4, self._ptz_status_color)

            # Direction arrow overlay (fades after 0.5s)
            if self._ptz_arrow_dir and (now - self._ptz_arrow_time < 0.5):
                self._draw_ptz_arrow(frame, self._ptz_arrow_dir)

            # PTZ hint
            hint = "WASD=Bewegen  C=Mitte  Leertaste=Stopp  P=PTZ aus"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.putText(frame, hint, ((w-tw)//2, ptz_sy - 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # ─── TASTATUR-HILFE (nur im RAW-Modus ohne PTZ) ─────
        elif self.mode == VisionLabMode.RAW:
            hint = "Pfeile: L/R Wert  U/D Modus | P=PTZ | S Speichern | Q Beenden"
            (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(frame, hint, ((w-tw)//2, sy - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        return frame

    def _draw_ptz_arrow(self, frame, direction: str):
        """Draw directional arrow overlay on video edge."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        color = (0, 220, 0)
        if direction == "left":
            pts = np.array([[10, cy], [30, cy-12], [30, cy+12]], np.int32)
        elif direction == "right":
            pts = np.array([[w-10, cy], [w-30, cy-12], [w-30, cy+12]], np.int32)
        elif direction == "up":
            pts = np.array([[cx, 30], [cx-12, 50], [cx+12, 50]], np.int32)
        elif direction == "down":
            pts = np.array([[cx, h-70], [cx-12, h-90], [cx+12, h-90]], np.int32)
        else:
            return
        cv2.fillPoly(frame, [pts], color)

    def _get_mode_params(self) -> List[Tuple[str, str, str]]:
        """Get (label, field_name, format_str) for current mode's relevant params."""
        params = []
        if self.mode == VisionLabMode.PERSON:
            params.append(("Shrink", "person_bbox_shrink_factor", "{:.2f}"))
        elif self.mode == VisionLabMode.FACE:
            params.append(("Face Pad", "face_roi_padding", "{:.2f}"))
            params.append(("Min Face", "min_face_size", "{}px"))
        elif self.mode == VisionLabMode.HANDS:
            params.append(("Hand Conf", "hand_confidence_threshold", "{:.2f}"))
        elif self.mode == VisionLabMode.IDENTITY:
            params.append(("ID Threshold", "identity_threshold", "{:.2f}"))
            params.append(("Face Pad", "face_roi_padding", "{:.2f}"))
            params.append(("Min Face", "min_face_size", "{}px"))
        elif self.mode == VisionLabMode.FULL_STACK:
            params.append(("ID Threshold", "identity_threshold", "{:.2f}"))
            params.append(("Shrink", "person_bbox_shrink_factor", "{:.2f}"))
            params.append(("Hand Conf", "hand_confidence_threshold", "{:.2f}"))
        elif self.mode == VisionLabMode.POSE:
            params.append(("Pose Conf", "pose_confidence_threshold", "{:.2f}"))
            params.append(("KP Conf", "keypoint_confidence_threshold", "{:.2f}"))
        return params

    # ========================================================================
    # TRACKBAR CONTROLS
    # ========================================================================

    # Trackbar names (used for create + sync, must match exactly)
    _TB_SHRINK = "Person Box verkleinern %"
    _TB_FACEPAD = "Gesicht Rand-Puffer %"
    _TB_IDTHRESH = "Erkennung Schwelle %"
    _TB_MINFACE = "Min Gesicht Pixel"
    _TB_HANDCONF = "Hand Sicherheit %"
    _TB_POSECONF = "Pose Sicherheit %"
    _TB_KPCONF = "Keypoint Sicherheit %"

    def _create_trackbars(self):
        """Create OpenCV trackbar controls in separate window."""
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 620, 350)

        # All trackbars map integer range to float via callbacks
        cv2.createTrackbar(self._TB_SHRINK, WIN_CONTROLS,
                          int(self.config.person_bbox_shrink_factor * 100), 50,
                          lambda v: setattr(self.config, 'person_bbox_shrink_factor', v / 100.0))

        cv2.createTrackbar(self._TB_FACEPAD, WIN_CONTROLS,
                          int(self.config.face_roi_padding * 100), 50,
                          lambda v: setattr(self.config, 'face_roi_padding', v / 100.0))

        cv2.createTrackbar(self._TB_IDTHRESH, WIN_CONTROLS,
                          int(self.config.identity_threshold * 100), 100,
                          lambda v: setattr(self.config, 'identity_threshold', v / 100.0))

        cv2.createTrackbar(self._TB_MINFACE, WIN_CONTROLS,
                          self.config.min_face_size, 200,
                          lambda v: setattr(self.config, 'min_face_size', max(20, v)))

        cv2.createTrackbar(self._TB_HANDCONF, WIN_CONTROLS,
                          int(self.config.hand_confidence_threshold * 100), 100,
                          lambda v: setattr(self.config, 'hand_confidence_threshold', v / 100.0))

        cv2.createTrackbar(self._TB_POSECONF, WIN_CONTROLS,
                          int(self.config.pose_confidence_threshold * 100), 100,
                          lambda v: setattr(self.config, 'pose_confidence_threshold', v / 100.0))

        cv2.createTrackbar(self._TB_KPCONF, WIN_CONTROLS,
                          int(self.config.keypoint_confidence_threshold * 100), 100,
                          lambda v: setattr(self.config, 'keypoint_confidence_threshold', v / 100.0))

    def _sync_trackbars(self):
        """Sync trackbar positions to current config values (after load/save)."""
        try:
            cv2.setTrackbarPos(self._TB_SHRINK, WIN_CONTROLS, int(self.config.person_bbox_shrink_factor * 100))
            cv2.setTrackbarPos(self._TB_FACEPAD, WIN_CONTROLS, int(self.config.face_roi_padding * 100))
            cv2.setTrackbarPos(self._TB_IDTHRESH, WIN_CONTROLS, int(self.config.identity_threshold * 100))
            cv2.setTrackbarPos(self._TB_MINFACE, WIN_CONTROLS, self.config.min_face_size)
            cv2.setTrackbarPos(self._TB_HANDCONF, WIN_CONTROLS, int(self.config.hand_confidence_threshold * 100))
            cv2.setTrackbarPos(self._TB_POSECONF, WIN_CONTROLS, int(self.config.pose_confidence_threshold * 100))
            cv2.setTrackbarPos(self._TB_KPCONF, WIN_CONTROLS, int(self.config.keypoint_confidence_threshold * 100))
        except cv2.error:
            pass  # Window not yet ready

    def _draw_controls_panel(self) -> np.ndarray:
        """Draw a label panel image for the Controls window showing current values."""
        panel_w, panel_h = 620, 210
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        labels = [
            ("1  Person Box verkleinern", f"{self.config.person_bbox_shrink_factor:.2f}", (0, 200, 0)),
            ("2  Gesicht Rand-Puffer", f"{self.config.face_roi_padding:.2f}", (200, 150, 0)),
            ("3  Erkennung Schwelle", f"{self.config.identity_threshold:.2f}", (200, 100, 0)),
            ("4  Min Gesicht Pixel", f"{self.config.min_face_size}", (200, 150, 0)),
            ("5  Hand Sicherheit", f"{self.config.hand_confidence_threshold:.2f}", (0, 200, 200)),
            ("6  Pose Sicherheit", f"{self.config.pose_confidence_threshold:.2f}", (0, 140, 255)),
            ("7  Keypoint Sicherheit", f"{self.config.keypoint_confidence_threshold:.2f}", (0, 140, 255)),
        ]

        y = 25
        for name, val, color in labels:
            cv2.putText(panel, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(panel, val, (panel_w - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 28

        return panel

    # ========================================================================
    # INPUT HANDLING
    # ========================================================================

    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

    def _handle_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks on buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # PTZ buttons first
            if self._ptz_enabled:
                for (x1, y1, x2, y2, action) in self._ptz_buttons:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if action == "center":
                            self._ptz_center()
                        else:
                            self._ptz_move(action)
                        return
            # Mode buttons
            for (x1, y1, x2, y2, mode) in self._buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._switch_mode(mode)
                    break

    def _get_active_threshold_field(self) -> Optional[str]:
        """Get the primary threshold field for current mode (for +/- keys)."""
        mode_to_field = {
            VisionLabMode.PERSON: "person_bbox_shrink_factor",
            VisionLabMode.FACE: "face_roi_padding",
            VisionLabMode.HANDS: "hand_confidence_threshold",
            VisionLabMode.IDENTITY: "identity_threshold",
            VisionLabMode.FULL_STACK: "identity_threshold",
            VisionLabMode.POSE: "pose_confidence_threshold",
        }
        return mode_to_field.get(self.mode)

    def _adjust_threshold(self, direction: int):
        """Adjust active threshold by direction (+1 or -1)."""
        field = self._get_active_threshold_field()
        if not field:
            return
        cur = getattr(self.config, field)
        if field == "min_face_size":
            new_val = max(20, min(200, cur + direction * 5))
        else:
            new_val = max(0.0, min(1.0, cur + direction * 0.05))
        setattr(self.config, field, new_val)
        self._sync_trackbars()
        logger.info(f"{field}: {new_val}")

    def _switch_mode_by_offset(self, offset: int):
        """Switch to next/previous mode."""
        modes = list(VisionLabMode)
        cur_idx = modes.index(self.mode)
        new_idx = (cur_idx + offset) % len(modes)
        self._switch_mode(modes[new_idx])

    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == -1:
            return True

        # Arrow keys (Linux GTK key codes)
        if key == 65362:      # Up arrow = previous mode
            self._switch_mode_by_offset(-1)
            return True
        elif key == 65364:    # Down arrow = next mode
            self._switch_mode_by_offset(1)
            return True
        elif key == 65361:    # Left arrow = threshold down
            self._adjust_threshold(-1)
            return True
        elif key == 65363:    # Right arrow = threshold up
            self._adjust_threshold(1)
            return True

        key_char = chr(key & 0xFF).upper()

        # PTZ toggle (always available)
        if key_char == 'P' and PTZ_AVAILABLE:
            self._toggle_ptz()
            return True

        # PTZ movement (when PTZ active, WASD + C + Space)
        if self._ptz_enabled:
            if key_char == 'W':
                self._ptz_move("up")
                return True
            elif key_char == 'A':
                self._ptz_move("left")
                return True
            elif key_char == 'S':
                self._ptz_move("down")
                return True
            elif key_char == 'D':
                self._ptz_move("right")
                return True
            elif key_char == 'C':
                self._ptz_center()
                return True
            elif key_char == ' ':
                self._ptz_stop()
                return True

        # Mode switches
        if key_char == '0':
            self._switch_mode(VisionLabMode.RAW)
        elif key_char == '1':
            self._switch_mode(VisionLabMode.PERSON)
        elif key_char == '2':
            self._switch_mode(VisionLabMode.FACE)
        elif key_char == '3':
            self._switch_mode(VisionLabMode.HANDS)
        elif key_char == '4':
            self._switch_mode(VisionLabMode.IDENTITY)
        elif key_char == '5':
            self._switch_mode(VisionLabMode.FULL_STACK)
        elif key_char == '6':
            self._switch_mode(VisionLabMode.POSE)

        # Config controls (S only when PTZ off)
        elif key_char == 'R':
            logger.info("Reloading config...")
            self._load_configs()
            self._sync_trackbars()
        elif key_char == 'S' and not self._ptz_enabled:
            if self._save_runtime_config():
                logger.info("Config saved!")
        elif key_char == '+' or key_char == '=':
            self._adjust_threshold(1)
        elif key_char == '-':
            self._adjust_threshold(-1)

        # Quit
        elif key_char == 'Q' or key == 27:
            return False

        return True

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        """Main loop."""
        logger.info("Starting MolochVisionLab...")
        logger.info(f"RTSP URL: {RTSP_URL}")

        # Use GStreamer pipeline for MINIMUM LATENCY
        gst_pipeline = (
            f"rtspsrc location={RTSP_URL} latency=0 buffer-mode=none protocols=tcp do-retransmission=false ! "
            "rtph264depay ! h264parse ! "
            "v4l2h264dec ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        )

        logger.info("GStreamer pipeline (HW decode): trying...")
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            logger.warning("Hardware decode failed, trying software decode...")
            gst_pipeline = (
                f"rtspsrc location={RTSP_URL} latency=0 buffer-mode=none protocols=tcp do-retransmission=false ! "
                "rtph264depay ! h264parse ! "
                "avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            logger.warning("GStreamer failed, using OpenCV RTSP...")
            self.cap = cv2.VideoCapture(RTSP_URL)
            if not self.cap.isOpened():
                logger.error(f"Cannot open RTSP stream: {RTSP_URL}")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info("Video capture opened")

        # Initialize to RAW mode for latency testing
        self._switch_mode(VisionLabMode.RAW)

        # Create windows
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_MAIN, self._handle_mouse)

        # Create trackbar controls
        self._create_trackbars()

        # Target display size (smaller = faster, less CPU for resize + detection)
        self._display_width = 640
        self._display_height = 360

        self.running = True
        self.fps_start_time = time.time()

        # Ensure PTZ is released even on crash
        atexit.register(lambda: self._disable_ptz() if self._ptz_enabled else None)

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame read failed, reconnecting...")
                    time.sleep(0.5)
                    self.cap.release()
                    gst_pipeline = (
                        f"rtspsrc location={RTSP_URL} latency=0 buffer-mode=none protocols=tcp do-retransmission=false ! "
                        "rtph264depay ! h264parse ! v4l2h264dec ! "
                        "videoconvert ! video/x-raw,format=BGR ! "
                        "appsink drop=true sync=false max-buffers=1"
                    )
                    self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(RTSP_URL)
                    continue

                # Scale frame down for display
                frame = cv2.resize(frame, (self._display_width, self._display_height), interpolation=cv2.INTER_NEAREST)

                # Process frame (only every N-th frame to keep display responsive)
                self._frame_index += 1
                if self._frame_index >= self._detect_every_n:
                    self._frame_index = 0
                    self._process_frame(frame)

                # Draw overlays
                display = self._draw_overlays(frame)

                # Update FPS
                self._update_fps()

                # Show frame
                cv2.imshow(WIN_MAIN, display)

                # Update controls panel with current values
                cv2.imshow(WIN_CONTROLS, self._draw_controls_panel())

                # Handle keyboard
                key = cv2.waitKey(1)
                if not self._handle_key(key):
                    self.running = False

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        # Release PTZ exclusive control
        if self._ptz_enabled:
            self._disable_ptz()

        self._stop_pipeline()
        self._release_hailo()

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        logger.info("Cleanup complete")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          M.O.L.O.C.H. VISION CALIBRATION LAB            ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Modi: RAW | PERSON | FACE | HANDS | IDENT | FULL | POSE ║
    ║                                                           ║
    ║  Steuerung:                                               ║
    ║    0-6 = Modus    Pfeile = Modus/Werte    +/- = Werte    ║
    ║    R = Neu laden   S = Speichern   Q = Beenden           ║
    ║    P = PTZ ein/aus  WASD = Kamera  C = Mitte             ║
    ║    Schieberegler im Controls-Fenster                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    lab = MolochVisionLab()
    lab.run()


if __name__ == "__main__":
    main()
