#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Unified Vision Pipeline
=====================================

Konsolidierte Vision-Pipeline:
- RTSP-Verbindung zur Sonoff PTZ Kamera
- Integriertes Overlay-Rendering
- Hailo-10H NPU Inference mit Fallback-Kette

Autor: M.O.L.O.C.H. System
"""

import os
import cv2
import time
import json
import signal
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [UnifiedPipeline] %(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

# Hailo model directory (secondary SSD)
HAILO_MODELS_DIR = "/mnt/moloch-data/hailo/models"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detection:
    """Eine einzelne Erkennung."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, int]  # {x, y, w, h}
    source: str = "hailo"  # hailo or hog

    @property
    def center_x(self) -> int:
        return self.bbox["x"] + self.bbox["w"] // 2

    @property
    def center_y(self) -> int:
        return self.bbox["y"] + self.bbox["h"] // 2


@dataclass
class FrameResult:
    """Ergebnis einer Frame-Verarbeitung."""
    frame: np.ndarray
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    source: str = "sonoff"

    @property
    def person_detected(self) -> bool:
        return any(d.class_id == 0 for d in self.detections)

    @property
    def person_count(self) -> int:
        return sum(1 for d in self.detections if d.class_id == 0)

    @property
    def best_confidence(self) -> float:
        persons = [d for d in self.detections if d.class_id == 0]
        return max((d.confidence for d in persons), default=0.0)


class PipelineState(Enum):
    """Pipeline-Zustand."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Konfiguration der Unified Pipeline."""

    # Sonoff RTSP
    sonoff_url: str = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
    sonoff_transport: str = "udp"  # udp oder tcp
    sonoff_buffer_size: int = 1

    # Hailo NPU
    hailo_model_path: str = f"{HAILO_MODELS_DIR}/yolov8m_h10.hef"
    hailo_confidence_threshold: float = 0.4

    # Detection
    detection_interval: float = 0.1  # 10 FPS max
    motion_threshold: float = 0.01

    # HOG Fallback
    hog_confidence_threshold: float = 0.3
    hog_scale: float = 0.5

    # Overlay
    overlay_enabled: bool = True
    overlay_show_fps: bool = True
    overlay_show_inference_time: bool = True
    overlay_bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Gruen
    overlay_text_color: Tuple[int, int, int] = (255, 255, 255)  # Weiss


def load_config() -> PipelineConfig:
    """Lade Konfiguration aus JSON falls vorhanden."""
    config_path = Path("/home/molochzuhause/moloch/config/sonoff_camera.json")
    config = PipelineConfig()

    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
                if "camera" in data and "connection" in data["camera"]:
                    config.sonoff_url = data["camera"]["connection"].get("url", config.sonoff_url)
        except Exception as e:
            log.warning(f"Config load error: {e}")

    return config


# =============================================================================
# HAILO DETECTOR (aus vision_worker.py extrahiert)
# =============================================================================

HAILO_AVAILABLE = False
try:
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    HAILO_AVAILABLE = True
    log.info("Hailo inference available")
except ImportError as e:
    log.warning(f"Hailo inference not available: {e}")


class HailoDetector:
    """Hailo-10H NPU Detektor."""

    def __init__(self, hef_path: str, confidence_threshold: float = 0.4):
        self.hef_path = hef_path
        self.confidence_threshold = confidence_threshold
        self.hailo_infer = None
        self.input_shape = None
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialisiere Hailo NPU."""
        if self._initialized:
            return True

        if not HAILO_AVAILABLE:
            return False

        with self._lock:
            if self._initialized:
                return True

            try:
                log.info(f"Loading HEF: {self.hef_path}")
                self.hailo_infer = HailoInfer(self.hef_path, batch_size=1)
                self.input_shape = self.hailo_infer.get_input_shape()
                log.info(f"Hailo initialized. Input: {self.input_shape}")
                self._initialized = True
                return True
            except Exception as e:
                log.error(f"Hailo init failed: {e}")
                return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Fuehre Inference durch."""
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            start = time.time()

            # Preprocess (letterbox)
            input_frame = self._preprocess(frame)
            orig_h, orig_w = frame.shape[:2]

            # Async inference
            detections = []
            event = threading.Event()

            def callback(completion_info, bindings_list):
                nonlocal detections
                try:
                    for binding in bindings_list:
                        outputs = {
                            name: binding.output(name).get_buffer()
                            for name in self.hailo_infer.output_type
                        }
                        detections = self._parse_output(outputs, orig_h, orig_w)
                except Exception as e:
                    log.error(f"Callback error: {e}")
                finally:
                    event.set()

            self.hailo_infer.run([input_frame], callback)

            if not event.wait(timeout=5.0):
                log.warning("Hailo timeout")
                return []

            inference_time = (time.time() - start) * 1000
            log.debug(f"Hailo inference: {inference_time:.1f}ms, {len(detections)} detections")

            return detections

        except Exception as e:
            log.error(f"Hailo detect error: {e}")
            return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox preprocessing."""
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    def _parse_output(self, outputs: Dict, orig_h: int, orig_w: int) -> List[Detection]:
        """Parse YOLOv8 NMS output."""
        detections = []

        try:
            for name, output in outputs.items():
                if isinstance(output, list) and len(output) >= 1:
                    person_dets = output[0]  # Class 0 = person

                    if isinstance(person_dets, np.ndarray) and len(person_dets) > 0:
                        for det in person_dets:
                            if len(det) >= 5:
                                x1_n, y1_n, x2_n, y2_n, conf = det[:5]

                                if conf >= self.confidence_threshold:
                                    x1 = int(x1_n * orig_w)
                                    y1 = int(y1_n * orig_h)
                                    x2 = int(x2_n * orig_w)
                                    y2 = int(y2_n * orig_h)

                                    detections.append(Detection(
                                        class_id=0,
                                        class_name="person",
                                        confidence=float(conf),
                                        bbox={"x": x1, "y": y1, "w": x2-x1, "h": y2-y1},
                                        source="hailo"
                                    ))
        except Exception as e:
            log.error(f"Parse error: {e}")

        return detections

    def shutdown(self):
        """Release resources."""
        self._initialized = False
        self.hailo_infer = None


# =============================================================================
# UNIFIED VISION PIPELINE
# =============================================================================

class UnifiedVisionPipeline:
    """
    Konsolidierte Vision Pipeline fuer M.O.L.O.C.H.

    SINGLE SOURCE OF TRUTH fuer alle Vision-Operationen:
    - RTSP-Verbindung zur Sonoff PTZ
    - Integriertes Hailo NPU Inference
    - Overlay-Rendering
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or load_config()

        # State
        self.state = PipelineState.STOPPED
        self._running = False
        self._stop_event = threading.Event()

        # Single RTSP connection
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_connected = False

        # Legacy compatibility
        self.xiao_connected = False

        # Hailo NPU
        self.hailo: Optional[HailoDetector] = None
        self.hailo_available = False

        # HOG Fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Current frame result (thread-safe)
        self._current_result: Optional[FrameResult] = None
        self._result_lock = threading.Lock()

        # Stats
        self.stats = {
            "frames_processed": 0,
            "detections_total": 0,
            "hailo_inferences": 0,
            "hog_fallbacks": 0,
            "fps": 0.0,
            "avg_inference_ms": 0.0
        }
        self._fps_counter = 0
        self._fps_start = time.time()
        self._inference_times = []

        # Vision Context (lazy load)
        self._vision_context = None

        # Callbacks
        self.on_person_detected: Optional[Callable[[FrameResult], None]] = None
        self.on_person_lost: Optional[Callable[[], None]] = None

        # Previous state for change detection
        self._last_person_detected = False

        log.info("UnifiedVisionPipeline initialized")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _init_camera(self) -> bool:
        """Initialisiere EINZIGE RTSP-Verbindung."""
        try:
            log.info(f"Connecting to Sonoff...")

            # Transport settings
            if self.config.sonoff_transport == "udp":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            self.cap = cv2.VideoCapture(self.config.sonoff_url, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.sonoff_buffer_size)
                ret, frame = self.cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    self.camera_connected = True
                    log.info(f"Sonoff connected: {w}x{h}")
                    return True

            log.error("Sonoff connection failed")
            return False

        except Exception as e:
            log.error(f"Camera init error: {e}")
            return False

    def _init_hailo(self) -> bool:
        """Initialisiere Hailo NPU."""
        if not HAILO_AVAILABLE:
            return False

        try:
            self.hailo = HailoDetector(
                self.config.hailo_model_path,
                self.config.hailo_confidence_threshold
            )
            if self.hailo.initialize():
                self.hailo_available = True
                log.info("Hailo-10H initialized")
                return True
            return False
        except Exception as e:
            log.error(f"Hailo init error: {e}")
            return False

    def _get_vision_context(self):
        """Lazy load VisionContext."""
        if self._vision_context is None:
            try:
                from context.vision_context import get_vision_context
                self._vision_context = get_vision_context()
            except ImportError as e:
                log.error(f"VisionContext import failed: {e}")
        return self._vision_context

    # =========================================================================
    # DETECTION METHODS
    # =========================================================================

    def _detect_hailo(self, frame: np.ndarray) -> List[Detection]:
        """Hailo NPU Detection (Primary)."""
        if not self.hailo_available or not self.hailo:
            return []

        detections = self.hailo.detect(frame)
        if detections:
            self.stats["hailo_inferences"] += 1
        return detections

    def _detect_hog(self, frame: np.ndarray) -> List[Detection]:
        """HOG CPU Detection (Fallback)."""
        try:
            scale = self.config.hog_scale
            small = cv2.resize(frame, None, fx=scale, fy=scale)

            boxes, weights = self.hog.detectMultiScale(
                small, winStride=(8, 8), padding=(4, 4), scale=1.05
            )

            detections = []
            for (x, y, w, h), weight in zip(boxes, weights):
                conf = float(weight[0]) if hasattr(weight, '__len__') else float(weight)
                if conf > self.config.hog_confidence_threshold:
                    detections.append(Detection(
                        class_id=0,
                        class_name="person",
                        confidence=min(conf, 1.0),
                        bbox={"x": int(x/scale), "y": int(y/scale),
                              "w": int(w/scale), "h": int(h/scale)},
                        source="hog"
                    ))

            if detections:
                self.stats["hog_fallbacks"] += 1
            return detections

        except Exception as e:
            log.error(f"HOG error: {e}")
            return []

    # =========================================================================
    # OVERLAY RENDERING
    # =========================================================================

    def render_overlay(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Rendere Detection-Overlay auf Frame."""
        if not self.config.overlay_enabled:
            return frame

        overlay = frame.copy()

        # Bounding Boxes
        for det in result.detections:
            bbox = det.bbox
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Box
            color = self.config.overlay_bbox_color
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)

            # Label
            label = f"{det.class_name} {det.confidence:.0%} [{det.source}]"
            cv2.putText(overlay, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Stats Overlay (top-left)
        y_offset = 30
        text_color = self.config.overlay_text_color

        if self.config.overlay_show_fps:
            cv2.putText(overlay, f"FPS: {self.stats['fps']:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 25

        if self.config.overlay_show_inference_time:
            cv2.putText(overlay, f"Inference: {result.inference_time_ms:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 25

        # Detection count
        cv2.putText(overlay, f"Persons: {result.person_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        y_offset += 25

        # Active model
        active = "Hailo" if self.hailo_available else "HOG"
        cv2.putText(overlay, f"Model: {active}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        return overlay

    # =========================================================================
    # CONTEXT UPDATE
    # =========================================================================

    def _update_context(self, result: FrameResult):
        """Update VisionContext mit Detection-Ergebnis."""
        ctx = self._get_vision_context()
        if ctx is None:
            return

        # Target center for PTZ
        target_x = result.frame.shape[1] // 2  # Default: frame center
        if result.detections:
            best = max(result.detections, key=lambda d: d.confidence)
            target_x = best.center_x

        ctx.update_detection(
            person_detected=result.person_detected,
            person_count=result.person_count,
            face_detected=result.best_confidence > 0.6,
            confidence=result.best_confidence,
            source="unified",
            camera_connected=self.camera_connected,
            npu_active=self.hailo_available,
            target_center_x=target_x,
            frame_width=result.frame.shape[1]
        )

        # Fire callbacks on state change
        if result.person_detected != self._last_person_detected:
            if result.person_detected:
                log.info(f"Person detected! Count: {result.person_count}, "
                        f"Conf: {result.best_confidence:.0%}")
                if self.on_person_detected:
                    self.on_person_detected(result)
            else:
                log.info("Person lost")
                if self.on_person_lost:
                    self.on_person_lost()

            self._last_person_detected = result.person_detected

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Verarbeite einen Frame durch die komplette Pipeline."""
        start = time.time()
        detections = []

        # Detection Priority: Hailo > HOG
        # 1. Hailo NPU (best accuracy)
        if self.hailo_available:
            detections = self._detect_hailo(frame)

        # 2. HOG fallback
        if not detections:
            detections = self._detect_hog(frame)

        inference_time = (time.time() - start) * 1000
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        return FrameResult(
            frame=frame,
            timestamp=time.time(),
            detections=detections,
            inference_time_ms=inference_time,
            source="unified"
        )

    def _main_loop(self):
        """Haupt-Verarbeitungsloop."""
        last_process = 0

        while self._running and not self._stop_event.is_set():
            try:
                # Rate limiting
                now = time.time()
                if now - last_process < self.config.detection_interval:
                    time.sleep(0.01)
                    continue

                # Check camera
                if not self.cap or not self.cap.isOpened():
                    self.camera_connected = False
                    ctx = self._get_vision_context()
                    if ctx:
                        ctx.update_detection(
                            person_detected=False,
                            camera_connected=False,
                            source="unified"
                        )
                    time.sleep(1)
                    self._init_camera()
                    continue

                # Grab frame (low latency: grab + retrieve)
                if not self.cap.grab():
                    continue

                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    continue

                self.camera_connected = True

                # Process frame
                result = self._process_frame(frame)

                # Update shared state
                with self._result_lock:
                    self._current_result = result

                # Update context
                self._update_context(result)

                # Update stats
                self.stats["frames_processed"] += 1
                self.stats["detections_total"] += len(result.detections)
                self._fps_counter += 1

                # FPS calculation (every second)
                if now - self._fps_start >= 1.0:
                    self.stats["fps"] = self._fps_counter / (now - self._fps_start)
                    self._fps_counter = 0
                    self._fps_start = now

                    if self._inference_times:
                        self.stats["avg_inference_ms"] = sum(self._inference_times) / len(self._inference_times)

                last_process = now

                # Periodic log
                if self.stats["frames_processed"] % 100 == 0:
                    log.info(f"Frames: {self.stats['frames_processed']}, "
                            f"FPS: {self.stats['fps']:.1f}, "
                            f"Persons: {result.person_count}")

            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(0.5)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self) -> bool:
        """Starte die Pipeline."""
        log.info("Starting UnifiedVisionPipeline...")
        self.state = PipelineState.STARTING

        # Initialize components
        if not self._init_camera():
            self.state = PipelineState.ERROR
            return False

        self._init_hailo()

        # Start main loop
        self._running = True
        self._stop_event.clear()
        self.state = PipelineState.RUNNING

        # Log active components
        components = []
        if self.hailo_available:
            components.append("Hailo-10H")
        components.append("HOG-Fallback")

        log.info(f"Pipeline running: {' + '.join(components)}")

        # Run loop
        self._main_loop()
        return True

    def start_async(self) -> threading.Thread:
        """Starte Pipeline in Background-Thread."""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stoppe die Pipeline."""
        log.info("Stopping UnifiedVisionPipeline...")
        self._running = False
        self._stop_event.set()

        # Clear context
        ctx = self._get_vision_context()
        if ctx:
            ctx.clear()

        # Shutdown Hailo
        if self.hailo:
            self.hailo.shutdown()

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        self.state = PipelineState.STOPPED
        log.info("Pipeline stopped")

    def get_current_frame(self) -> Optional[FrameResult]:
        """Hole aktuelles Frame-Ergebnis (thread-safe)."""
        with self._result_lock:
            return self._current_result

    def get_frame_with_overlay(self) -> Optional[np.ndarray]:
        """Hole Frame mit gerendertem Overlay."""
        result = self.get_current_frame()
        if result is None:
            return None
        return self.render_overlay(result.frame, result)

    def get_stats(self) -> Dict[str, Any]:
        """Hole Pipeline-Statistiken."""
        return {
            **self.stats,
            "state": self.state.value,
            "camera_connected": self.camera_connected,
            "hailo_available": self.hailo_available
        }


# =============================================================================
# SINGLETON & CLI
# =============================================================================

_pipeline: Optional[UnifiedVisionPipeline] = None


def get_unified_pipeline(config: PipelineConfig = None) -> UnifiedVisionPipeline:
    """Hole oder erstelle Pipeline Singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedVisionPipeline(config)
    return _pipeline


def main():
    """CLI Entry Point."""
    pipeline = get_unified_pipeline()

    def signal_handler(sig, frame):
        log.info("Shutdown signal received")
        pipeline.stop()
        import sys
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not pipeline.start():
        log.error("Failed to start pipeline")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
