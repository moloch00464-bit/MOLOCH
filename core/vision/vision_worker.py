#!/usr/bin/env python3
"""
M.O.L.O.C.H. Vision Worker
===========================

Persistent process that runs Hailo-10H inference on camera feeds
and publishes semantic events to VisionContext.

Uses GStreamer pipeline with hailonet for reliable inference.

Usage:
    python -m core.vision.vision_worker
"""

import os
import sys
import cv2
import time
import json
import signal
import logging
import threading
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from functools import partial

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [VisionWorker] %(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

# Hailo model directory (secondary SSD)
HAILO_MODELS_DIR = "/mnt/moloch-data/hailo/models"

# Try to import Hailo inference
HAILO_AVAILABLE = False
try:
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    HAILO_AVAILABLE = True
    log.info("Hailo inference available")
except ImportError as e:
    log.warning(f"Hailo inference not available: {e}")


class HailoPersonDetector:
    """Hailo-10H based person detector using YOLOv8."""

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self.hailo_infer = None
        self.input_shape = None
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize Hailo inference engine."""
        if self._initialized:
            return True

        if not HAILO_AVAILABLE:
            log.warning("Hailo not available, cannot initialize")
            return False

        with self._lock:
            if self._initialized:
                return True

            try:
                log.info(f"Loading HEF model: {self.hef_path}")
                self.hailo_infer = HailoInfer(self.hef_path, batch_size=1)
                self.input_shape = self.hailo_infer.get_input_shape()
                log.info(f"Hailo initialized. Input shape: {self.input_shape}")
                self._initialized = True
                return True

            except Exception as e:
                log.error(f"Hailo initialization failed: {e}")
                return False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference with letterboxing."""
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape[:2]

        # Calculate scale maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size (letterbox)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return rgb

    def detect(self, frame: np.ndarray, score_threshold: float = 0.4) -> List[Dict]:
        """
        Run person detection on frame using Hailo-10H.

        Returns list of detections: [{class_id, class_name, confidence, bbox}]
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            # Preprocess
            input_frame = self.preprocess(frame)
            orig_h, orig_w = frame.shape[:2]

            # Storage for async result
            detections_result = []
            event = threading.Event()

            def inference_callback(completion_info, bindings_list):
                nonlocal detections_result
                try:
                    for binding in bindings_list:
                        outputs = {
                            name: binding.output(name).get_buffer()
                            for name in self.hailo_infer.output_type
                        }
                        detections_result.append(outputs)
                except Exception as e:
                    log.error(f"Inference callback error: {e}")
                finally:
                    event.set()

            # Run async inference
            self.hailo_infer.run([input_frame], inference_callback)

            # Wait for completion
            if not event.wait(timeout=5.0):
                log.warning("Hailo inference timeout")
                return []

            if not detections_result:
                return []

            # Parse YOLO output for persons
            return self._parse_persons(detections_result[0], orig_h, orig_w, score_threshold)

        except Exception as e:
            log.error(f"Hailo detection error: {e}")
            return []

    def _parse_persons(self, outputs: Dict, orig_h: int, orig_w: int,
                       score_threshold: float) -> List[Dict]:
        """Parse Hailo YOLOv8 NMS output for person detections.
        
        Output format: List of 80 arrays (one per COCO class).
        Each array contains detections for that class.
        Detection format: [x1_norm, y1_norm, x2_norm, y2_norm, conf]
        Coordinates are normalized (0-1).
        """
        detections = []

        try:
            for name, output in outputs.items():
                # output is a list of 80 numpy arrays (one per class)
                if isinstance(output, list) and len(output) >= 1:
                    # Class 0 = person
                    person_dets = output[0]
                    
                    if isinstance(person_dets, np.ndarray) and len(person_dets) > 0:
                        for det in person_dets:
                            if len(det) >= 5:
                                x1_n, y1_n, x2_n, y2_n, conf = det[:5]
                                
                                if conf >= score_threshold:
                                    # Convert normalized coords to original image
                                    x1 = int(x1_n * orig_w)
                                    y1 = int(y1_n * orig_h)
                                    x2 = int(x2_n * orig_w)
                                    y2 = int(y2_n * orig_h)
                                    
                                    detections.append({
                                        "class_id": 0,
                                        "class_name": "person",
                                        "confidence": float(conf),
                                        "bbox": {
                                            "x": x1,
                                            "y": y1,
                                            "w": x2 - x1,
                                            "h": y2 - y1
                                        }
                                    })

        except Exception as e:
            log.error(f"Parse error: {e}")

        return detections

    def shutdown(self):
        """Release resources."""
        self._initialized = False
        self.hailo_infer = None


class VisionWorker:
    """
    M.O.L.O.C.H. Vision Worker - Hailo-10H Person Detection.

    Uses Hailo-10H NPU for YOLO inference (falls back to HOG on CPU).
    Publishes results to VisionContext for brain integration.
    """

    # Configuration - Heimnetzwerk
    SONOFF_RTSP = os.environ.get("MOLOCH_RTSP_URL", "rtsp://USER:PASS@CAMERA_IP:554/av_stream/ch0")
    MODEL_PATH = f"{HAILO_MODELS_DIR}/yolov8m_h10.hef"

    # Detection settings
    PERSON_CLASS_ID = 0  # COCO class 0 = person
    CONFIDENCE_THRESHOLD = 0.4
    DETECTION_INTERVAL = 0.2  # 5 FPS detection

    def __init__(self):
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_connected = False

        # State
        self._running = False
        self._stop_event = threading.Event()

        # Vision context (lazy loaded)
        self._vision_context = None

        # Hailo detector (primary)
        self.hailo_detector: Optional[HailoPersonDetector] = None
        self.use_hailo = False

        # HOG detector for CPU-based person detection (fallback)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Stats
        self.frames_processed = 0
        self.detections_count = 0
        self.last_detection_time = 0

        # Last state (for change detection)
        self._last_person_detected = False

        log.info("VisionWorker initialized")

    def _get_vision_context(self):
        """Lazy load VisionContext."""
        if self._vision_context is None:
            try:
                from context.vision_context import get_vision_context
                self._vision_context = get_vision_context()
            except ImportError as e:
                log.error(f"Failed to import VisionContext: {e}")
        return self._vision_context

    def _init_camera(self) -> bool:
        """Initialize Sonoff RTSP camera."""
        try:
            log.info(f"Connecting to camera...")

            # TCP transport for stability
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            self.cap = cv2.VideoCapture(self.SONOFF_RTSP, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                ret, frame = self.cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    self.camera_connected = True
                    log.info(f"Camera connected: {w}x{h}")
                    return True

            log.error("Camera connection failed")
            return False

        except Exception as e:
            log.error(f"Camera init failed: {e}")
            return False

    def _detect_persons_hog(self, frame) -> List[Dict[str, Any]]:
        """Detect persons using HOG detector (CPU fallback)."""
        try:
            # Resize for faster detection
            scale = 0.5
            small = cv2.resize(frame, None, fx=scale, fy=scale)

            # Detect
            boxes, weights = self.hog.detectMultiScale(
                small,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )

            detections = []
            for (x, y, w, h), weight in zip(boxes, weights):
                conf = float(weight[0]) if hasattr(weight, '__len__') else float(weight)
                if conf > 0.3:  # HOG threshold (lower than YOLO)
                    detections.append({
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": conf,
                        "bbox": {
                            "x": int(x / scale),
                            "y": int(y / scale),
                            "w": int(w / scale),
                            "h": int(h / scale)
                        }
                    })

            return detections

        except Exception as e:
            log.error(f"HOG detection error: {e}")
            return []

    def _detect_motion(self, frame, prev_frame) -> float:
        """Simple motion detection to detect presence."""
        if prev_frame is None:
            return 0.0

        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur to reduce noise
            gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

            # Compute difference
            diff = cv2.absdiff(gray1, gray2)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

            # Calculate motion percentage
            motion = (thresh > 0).sum() / thresh.size

            return motion

        except Exception as e:
            return 0.0

    def _update_context(self, person_detected: bool, person_count: int, confidence: float, detections: list = None):
        """Update VisionContext with detection results including position."""
        ctx = self._get_vision_context()
        if ctx is None:
            return

        # Face detected if confidence is high
        face_detected = confidence > 0.6
        
        # Calculate target center X from best detection
        target_x = 960  # Default center
        frame_w = 1920
        if detections and len(detections) > 0:
            best_det = max(detections, key=lambda d: d.get("confidence", 0))
            bbox = best_det.get("bbox", {})
            if bbox:
                # bbox format: {x, y, w, h} where x,y is top-left corner
                x = bbox.get("x", 0)
                w = bbox.get("w", 0)
                target_x = x + w // 2  # center of bbox
            elif "x" in best_det:
                # Alternative format: center x
                target_x = best_det.get("x", frame_w // 2)

        ctx.update_detection(
            person_detected=person_detected,
            person_count=person_count,
            face_detected=face_detected,
            confidence=confidence,
            source="sonoff",
            camera_connected=self.camera_connected,
            npu_active=self.use_hailo,
            target_center_x=target_x,
            frame_width=frame_w
        )

        self.detections_count = person_count

        # Log state changes
        if person_detected != self._last_person_detected:
            if person_detected:
                log.info(f"Person detected! Count: {person_count}, Confidence: {confidence:.2f}, target_x={target_x}")
            else:
                log.info("Person lost")
            self._last_person_detected = person_detected

    def _init_hailo(self) -> bool:
        """Initialize Hailo-10H detector."""
        if not HAILO_AVAILABLE:
            log.warning("Hailo not available, using HOG fallback")
            return False

        try:
            self.hailo_detector = HailoPersonDetector(self.MODEL_PATH)
            if self.hailo_detector.initialize():
                self.use_hailo = True
                log.info("Hailo-10H detector initialized!")
                return True
            else:
                log.warning("Hailo initialization failed, using HOG fallback")
                return False
        except Exception as e:
            log.error(f"Hailo init error: {e}, using HOG fallback")
            return False

    def start(self) -> bool:
        """Start the vision worker."""
        log.info("Starting VisionWorker...")

        # Init Camera (Sonoff)
        if not self._init_camera():
            log.error("Failed to initialize camera")
            return False

        # Init Hailo (optional)
        self._init_hailo()

        self._running = True
        self._stop_event.clear()

        # Log detection method
        if self.use_hailo:
            log.info("VisionWorker started - using Hailo-10H NPU (primary) + HOG fallback")
        else:
            log.info("VisionWorker started - using HOG (CPU)")
        self._detection_loop()

        return True

    def _detection_loop(self):
        """Main detection loop."""
        last_detection = 0
        prev_frame = None
        consecutive_motion = 0

        while self._running and not self._stop_event.is_set():
            try:
                # Rate limit
                now = time.time()
                if now - last_detection < self.DETECTION_INTERVAL:
                    time.sleep(0.05)
                    continue

                # Capture frame
                if not self.cap or not self.cap.isOpened():
                    self.camera_connected = False
                    ctx = self._get_vision_context()
                    if ctx:
                        ctx.update_detection(
                            person_detected=False,
                            camera_connected=False,
                            source="sonoff"
                        )
                    time.sleep(1)
                    # Try to reconnect
                    self._init_camera()
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.camera_connected = True

                # Motion detection first (fast)
                motion = self._detect_motion(frame, prev_frame)
                prev_frame = frame.copy()

                # Track consecutive motion frames
                if motion > 0.01:  # 1% of pixels changed
                    consecutive_motion += 1
                else:
                    consecutive_motion = max(0, consecutive_motion - 1)

                # Run person detection
                person_detected = False
                person_count = 0
                confidence = 0.0
                detections = []

                # Priority: Hailo (NPU) > HOG (CPU fallback)
                # 1. Try Hailo NPU first (best accuracy)
                if self.use_hailo and self.hailo_detector:
                    detections = self.hailo_detector.detect(frame)
                    if detections:
                        log.debug(f"Hailo detected {len(detections)} person(s)")

                # 2. Fallback to HOG if Hailo found nothing
                if not detections:
                    detections = self._detect_persons_hog(frame)

                person_count = len(detections)
                person_detected = person_count > 0

                if detections:
                    confidence = max(d["confidence"] for d in detections)

                # Update context
                self._update_context(person_detected, person_count, confidence, detections)

                self.frames_processed += 1
                last_detection = now

                # Log periodically
                if self.frames_processed % 100 == 0:
                    log.info(f"Processed {self.frames_processed} frames, "
                            f"motion: {motion:.3f}, persons: {self.detections_count}")

            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(0.5)

    def stop(self):
        """Stop the vision worker."""
        log.info("Stopping VisionWorker...")
        self._running = False
        self._stop_event.set()

        # Clear context
        ctx = self._get_vision_context()
        if ctx:
            ctx.clear()

        # Shutdown Hailo
        if self.hailo_detector:
            self.hailo_detector.shutdown()
            self.hailo_detector = None
            self.use_hailo = False

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        log.info("VisionWorker stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "running": self._running,
            "camera_connected": self.camera_connected,
            "frames_processed": self.frames_processed,
            "detections_count": self.detections_count,
            "last_person_detected": self._last_person_detected
        }


# Global instance
_worker: Optional[VisionWorker] = None


def get_vision_worker() -> VisionWorker:
    """Get or create VisionWorker instance."""
    global _worker
    if _worker is None:
        _worker = VisionWorker()
    return _worker


def main():
    """Main entry point."""
    worker = get_vision_worker()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        log.info("Shutdown signal received")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start worker
    if not worker.start():
        log.error("Failed to start VisionWorker")
        sys.exit(1)


if __name__ == "__main__":
    main()
