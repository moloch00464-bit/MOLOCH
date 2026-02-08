#!/usr/bin/env python3
"""
M.O.L.O.C.H. GStreamer Hailo Detector
=====================================

Person detection using GStreamer + Hailo-10H NPU.
Uses hailonet element for inference and hailofilter for NMS postprocessing.

Tested: 19.9 FPS with YOLOv8m on Hailo-10H (2026-02-04)
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

import numpy as np
import threading
import time
import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Callable
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# Debug flags
DEBUG_FRAME_LOGGING = True  # Log every N frames
DEBUG_FRAME_INTERVAL = 30   # Log every 30 frames
DEBUG_WATCHDOG_TIMEOUT = 3.0  # Warn if no frame for N seconds

# Hailo model directory (secondary SSD)
HAILO_MODELS_DIR = "/mnt/moloch-data/hailo/models"

# Initialize GStreamer
Gst.init(None)

# Import Hailo for detection parsing
try:
    import hailo
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo module not available - detections won't be parsed")


def check_hailo_device_free() -> bool:
    """Check if Hailo device is available (not in use by another process)."""
    try:
        from hailo_platform import VDevice, HailoRTException
        import gc
        # Try to create a virtual device - this will fail if NPU is busy
        vdevice = VDevice()
        # If we get here, device is free - release it immediately
        del vdevice
        gc.collect()  # Force cleanup to avoid race condition
        import time
        time.sleep(0.1)  # Small delay to let NPU fully release
        return True
    except Exception as e:
        error_msg = str(e)
        if "HAILO_OUT_OF_PHYSICAL_DEVICES" in error_msg or "not enough free devices" in error_msg:
            logger.warning("Hailo NPU is busy (probably used by Whisper)")
            return False
        # Other errors - log but assume available
        logger.debug(f"Hailo check: {e}")
        return True


@dataclass
class Detection:
    """Detection result."""
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) normalized 0-1
    center_x: float  # normalized 0-1
    center_y: float  # normalized 0-1


@dataclass
class DetectionResult:
    """Frame detection result."""
    detections: List[Dict] = field(default_factory=list)
    frame_width: int = 640
    frame_height: int = 640
    inference_time_ms: float = 0
    fps: float = 0
    timestamp: float = field(default_factory=time.time)
    error: str = ""
    frame: Optional[Any] = None  # BGR numpy array for display


class GstHailoDetector:
    """
    GStreamer-based person detector using Hailo-10H.

    Uses a GStreamer pipeline with:
    - rtspsrc for RTSP input
    - avdec_h264 for decoding
    - hailonet for Hailo inference
    - hailofilter for NMS postprocessing
    - appsink for getting results
    """

    # Available models (using secondary SSD)
    MODELS = {
        "yolov8m": {
            "hef": f"{HAILO_MODELS_DIR}/yolov8m_h10.hef",
            "postprocess": "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
            "function": "yolov8m",
            "description": "YOLOv8 Medium - Object Detection (21MB)"
        },
        "yolov8s_pose": {
            "hef": f"{HAILO_MODELS_DIR}/yolov8s_pose_h10.hef",
            "postprocess": "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so",
            "function": "yolov8_pose_estimation",
            "description": "YOLOv8 Small Pose - Keypoint Detection (13MB)"
        },
        "yolov8m_pose": {
            "hef": f"{HAILO_MODELS_DIR}/yolov8m_pose_h10.hef",
            "postprocess": "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so",
            "function": "yolov8_pose_estimation",
            "description": "YOLOv8 Medium Pose - Keypoint Detection (29MB)"
        },
        "yolov11m": {
            "hef": f"{HAILO_MODELS_DIR}/yolov11m_h10.hef",
            "postprocess": "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
            "function": "yolov8m",  # Uses same post-processor
            "description": "YOLOv11 Medium - Object Detection (28MB)"
        }
    }

    # Default model
    DEFAULT_MODEL = "yolov8m"

    # Model paths (set from MODELS dict)
    HEF_PATH = f"{HAILO_MODELS_DIR}/yolov8m_h10.hef"
    POSTPROCESS_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
    POSTPROCESS_FUNC = "yolov8m"

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.45

    def __init__(self, rtsp_url: str = None, model: str = None):
        """
        Initialize detector.

        Args:
            rtsp_url: RTSP URL for camera input
            model: Model name from MODELS dict (default: yolov8m)
        """
        self.rtsp_url = rtsp_url or "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"

        # Set model configuration
        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name in self.MODELS:
            model_config = self.MODELS[self.model_name]
            self.HEF_PATH = model_config["hef"]
            self.POSTPROCESS_SO = model_config["postprocess"]
            self.POSTPROCESS_FUNC = model_config["function"]
            logger.info(f"Model selected: {self.model_name} - {model_config['description']}")
        else:
            logger.warning(f"Unknown model '{model}', using default: {self.DEFAULT_MODEL}")
            self.model_name = self.DEFAULT_MODEL

        self._pipeline = None
        self._bus = None
        self._running = False
        self._main_loop = None
        self._loop_thread = None

        # Results
        self._latest_result: DetectionResult = DetectionResult()
        self._result_lock = threading.Lock()

        # Callbacks
        self._on_detection: Optional[Callable[[DetectionResult], None]] = None

        # Stats
        self._frame_count = 0
        self._start_time = 0
        self._last_fps_time = 0
        self._fps_frame_count = 0
        self._last_frame_time = 0  # Watchdog: last frame received
        self._watchdog_warned = False  # Avoid spam

        logger.info(f"GstHailoDetector initialized (model: {self.model_name})")
        logger.info(f"  HEF: {self.HEF_PATH}")
        logger.info(f"  PostProcess: {self.POSTPROCESS_SO}")

    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline string."""
        pipeline = f"""
            rtspsrc location={self.rtsp_url} latency=200 !
            rtph264depay !
            h264parse !
            avdec_h264 !
            videoconvert !
            videoscale !
            video/x-raw,width=640,height=640,format=RGB !
            queue leaky=downstream max-size-buffers=2 !
            hailonet hef-path={self.HEF_PATH}
                batch-size=1
                nms-score-threshold={self.CONFIDENCE_THRESHOLD}
                nms-iou-threshold={self.NMS_THRESHOLD} !
            queue leaky=downstream max-size-buffers=2 !
            hailofilter so-path={self.POSTPROCESS_SO}
                function-name={self.POSTPROCESS_FUNC}
                qos=false !
            videoconvert !
            appsink name=appsink emit-signals=true max-buffers=2 drop=true sync=false
        """
        return " ".join(pipeline.split())

    def start(self) -> bool:
        """Start the detection pipeline."""
        if self._running:
            logger.warning("Detector already running")
            return True

        # === DEBUG: Log startup parameters ===
        logger.info("=" * 60)
        logger.info("[DEBUG] GstHailoDetector START")
        logger.info(f"[DEBUG] RTSP URL: {self.rtsp_url}")
        logger.info(f"[DEBUG] Model: {self.model_name}")
        logger.info(f"[DEBUG] HEF Path: {self.HEF_PATH}")
        logger.info(f"[DEBUG] PostProcess SO: {self.POSTPROCESS_SO}")

        # === DEBUG: Verify files exist ===
        if os.path.exists(self.HEF_PATH):
            hef_size = os.path.getsize(self.HEF_PATH) / (1024 * 1024)
            logger.info(f"[DEBUG] HEF exists: {hef_size:.1f} MB")
        else:
            logger.error(f"[DEBUG] HEF NOT FOUND: {self.HEF_PATH}")
            return False

        if os.path.exists(self.POSTPROCESS_SO):
            logger.info(f"[DEBUG] PostProcess SO exists: OK")
        else:
            logger.error(f"[DEBUG] PostProcess SO NOT FOUND: {self.POSTPROCESS_SO}")
            return False

        # Check if Hailo device is available BEFORE trying GStreamer
        # This prevents segfault when NPU is already in use (e.g., by Whisper)
        if not check_hailo_device_free():
            logger.warning("Hailo NPU not available - skipping GstHailoDetector")
            self._latest_result = DetectionResult(error="Hailo NPU busy")
            return False
        logger.info("[DEBUG] Hailo NPU: available")

        try:
            pipeline_str = self._build_pipeline()
            logger.info(f"[DEBUG] Creating GStreamer pipeline...")

            self._pipeline = Gst.parse_launch(pipeline_str)
            if not self._pipeline:
                logger.error("Failed to create pipeline")
                return False

            # Get appsink and connect callback
            appsink = self._pipeline.get_by_name("appsink")
            if appsink:
                appsink.connect("new-sample", self._on_new_sample)
            else:
                logger.error("Appsink not found")
                return False

            # Setup bus
            self._bus = self._pipeline.get_bus()
            self._bus.add_signal_watch()
            self._bus.connect("message", self._on_bus_message)

            # Start pipeline with state transition logging
            logger.info("[DEBUG] Pipeline state: NULL -> PLAYING")
            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("[DEBUG] Pipeline state change FAILED")
                return False
            elif ret == Gst.StateChangeReturn.ASYNC:
                logger.info("[DEBUG] Pipeline state change: ASYNC (waiting for stream)")
            elif ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("[DEBUG] Pipeline state change: SUCCESS")
            else:
                logger.info(f"[DEBUG] Pipeline state change: {ret}")

            self._running = True
            self._start_time = time.time()
            self._last_fps_time = self._start_time
            self._last_frame_time = self._start_time  # Initialize watchdog

            # Start main loop in thread
            self._main_loop = GLib.MainLoop()
            self._loop_thread = threading.Thread(target=self._run_main_loop, daemon=True)
            self._loop_thread.start()

            logger.info("GstHailoDetector started")
            return True

        except Exception as e:
            logger.error(f"Failed to start detector: {e}")
            return False

    def _run_main_loop(self):
        """Run GLib main loop."""
        try:
            # Add watchdog timer (check every second)
            GLib.timeout_add(1000, self._watchdog_check)
            self._main_loop.run()
        except Exception as e:
            logger.error(f"Main loop error: {e}")

    def _watchdog_check(self) -> bool:
        """Check if frames are being received (watchdog timer)."""
        if not self._running:
            return False  # Stop timer

        if self._last_frame_time > 0:
            elapsed = time.time() - self._last_frame_time
            if elapsed > DEBUG_WATCHDOG_TIMEOUT and not self._watchdog_warned:
                logger.warning(f"[DEBUG] WATCHDOG: No frame received for {elapsed:.1f}s!")
                logger.warning(f"[DEBUG] Last frame count: {self._frame_count}")
                self._watchdog_warned = True

        return self._running  # Continue timer if still running

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new frame from appsink."""
        try:
            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            # Parse Hailo detections
            detections = []
            if HAILO_AVAILABLE:
                try:
                    roi = hailo.get_roi_from_buffer(buffer)
                    if roi:
                        hailo_dets = hailo.get_hailo_detections(roi)
                        for d in hailo_dets:
                            label = d.get_label()
                            conf = d.get_confidence()
                            bbox = d.get_bbox()

                            # Only include persons with good confidence
                            if label == "person" and conf >= 0.5:
                                # bbox is HailoBBox with xmin, ymin, width, height (normalized)
                                x1 = bbox.xmin()
                                y1 = bbox.ymin()
                                w = bbox.width()
                                h = bbox.height()
                                x2 = x1 + w
                                y2 = y1 + h

                                detections.append({
                                    "class": label,
                                    "confidence": conf,
                                    "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
                                    "center_x": (x1 + x2) / 2,
                                    "center_y": (y1 + y2) / 2
                                })

                except Exception as e:
                    if self._frame_count == 0:
                        logger.debug(f"Hailo detection parse error: {e}")

            # Extract frame for display (BGR format for OpenCV/tkinter)
            frame_bgr = None
            try:
                import numpy as np
                # Map buffer to numpy array
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    # Buffer is in RGB format from videoconvert
                    frame_rgb = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data
                    ).copy()  # Copy so we can unmap
                    buffer.unmap(map_info)

                    # Convert RGB to BGR for OpenCV/display
                    import cv2
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                if self._frame_count == 0:
                    logger.debug(f"Frame extraction error: {e}")

            # Update FPS and watchdog
            self._frame_count += 1
            self._fps_frame_count += 1
            now = time.time()
            self._last_frame_time = now  # Update watchdog
            self._watchdog_warned = False  # Reset warning

            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = now
            else:
                fps = self._fps_frame_count / elapsed if elapsed > 0 else 0

            # === DEBUG: Log every N frames ===
            if DEBUG_FRAME_LOGGING and self._frame_count % DEBUG_FRAME_INTERVAL == 0:
                frame_status = "OK" if frame_bgr is not None else "NONE"
                logger.info(f"[DEBUG] Frame {self._frame_count}: {len(detections)} det, "
                           f"FPS={fps:.1f}, frame={frame_status}, "
                           f"size={width}x{height}")

            # Create result with frame
            result = DetectionResult(
                detections=detections,
                frame_width=width,
                frame_height=height,
                fps=fps,
                timestamp=now,
                frame=frame_bgr
            )

            # Store result (thread-safe)
            with self._result_lock:
                self._latest_result = result

            # Callback
            if self._on_detection:
                try:
                    self._on_detection(result)
                except Exception as e:
                    logger.error(f"Detection callback error: {e}")

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"Sample callback error: {e}")
            return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages with debug logging."""
        t = message.type
        src = message.src.get_name() if message.src else "unknown"

        if t == Gst.MessageType.EOS:
            logger.info(f"[DEBUG] GStreamer EOS from {src}")
            self.stop()

        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[DEBUG] GStreamer ERROR from {src}: {err.message}")
            logger.error(f"[DEBUG] Error debug: {debug}")

        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning(f"[DEBUG] GStreamer WARNING from {src}: {err.message}")
            logger.warning(f"[DEBUG] Warning debug: {debug}")

        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self._pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info(f"[DEBUG] Pipeline state: {old.value_nick} -> {new.value_nick} "
                           f"(pending: {pending.value_nick})")

        elif t == Gst.MessageType.STREAM_START:
            logger.info(f"[DEBUG] Stream started from {src}")

        elif t == Gst.MessageType.LATENCY:
            logger.debug(f"[DEBUG] Latency update from {src}")

    def stop(self):
        """Stop the detection pipeline."""
        if not self._running:
            return

        self._running = False

        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        if self._main_loop and self._main_loop.is_running():
            self._main_loop.quit()

        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None

        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"GstHailoDetector stopped ({self._frame_count} frames, {fps:.1f} FPS avg)")

    def get_latest_result(self) -> DetectionResult:
        """Get latest detection result."""
        with self._result_lock:
            return self._latest_result

    def set_detection_callback(self, callback: Callable[[DetectionResult], None]):
        """Set callback for detection results."""
        self._on_detection = callback

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        with self._result_lock:
            return self._latest_result.fps


# Singleton
_detector: Optional[GstHailoDetector] = None
_detector_lock = threading.Lock()


def get_gst_detector(rtsp_url: str = None, model: str = None) -> GstHailoDetector:
    """Get or create GstHailoDetector singleton."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = GstHailoDetector(rtsp_url, model=model)
    return _detector


def list_available_models() -> Dict[str, str]:
    """List available Hailo models."""
    return {name: cfg["description"] for name, cfg in GstHailoDetector.MODELS.items()}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=== GStreamer Hailo Detector Test ===")

    detector = GstHailoDetector()

    def on_detect(result):
        if result.detections:
            for d in result.detections[:3]:
                print(f"  {d['class']}: {d['confidence']*100:.0f}% @ ({d['center_x']:.2f}, {d['center_y']:.2f})")

    detector.set_detection_callback(on_detect)

    if not detector.start():
        print("Failed to start!")
        sys.exit(1)

    print("Running for 15 seconds...")
    try:
        for i in range(150):
            time.sleep(0.1)
            if i % 30 == 0:
                r = detector.get_latest_result()
                print(f"[{i/10:.0f}s] FPS: {r.fps:.1f}, Persons: {len(r.detections)}")
    except KeyboardInterrupt:
        pass

    detector.stop()
    print("Done!")
