#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hybrid Vision Pipeline
====================================

Orchestrates Hailo-10H for face recognition.

Architecture:
- Hailo-10H NPU: On-demand face recognition via GStreamer pose detector

States:
- IDLE: Waiting for trigger
- TRIGGERED: Face detected
- ANALYZING: Hailo active, processing face
- IDENTIFIED: Person recognized (transient)
"""

import time
import logging
import threading
import base64
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class HybridVisionState(Enum):
    """Pipeline states."""
    IDLE = "idle"
    TRIGGERED = "triggered"
    ANALYZING = "analyzing"
    ERROR = "error"


@dataclass
class PersonEvent:
    """Event when a person is detected/identified."""
    timestamp: float
    person_id: str
    person_name: str
    confidence: float  # 0.0 - 1.0
    is_known: bool
    face_count: int
    source: str  # "hailo"
    bbox: Optional[Dict] = None
    image_base64: Optional[str] = None


class HybridVision:
    """
    M.O.L.O.C.H. Hybrid Vision Pipeline.

    Singleton pattern for hardware access coordination.
    Event-driven architecture with callbacks.

    Energy Management:
    - Hailo on-demand (wakes on face, sleeps after idle timeout)
    """

    _instance = None
    _lock = threading.Lock()

    # Configuration
    HAILO_IDLE_TIMEOUT = 30.0  # Seconds before Hailo goes standby
    RECOGNITION_INTERVAL = 2.0  # Seconds between recognition attempts

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # State
        self.state = HybridVisionState.IDLE
        self._state_lock = threading.Lock()

        # Components (lazy loaded)
        self._hailo_analyzer = None
        self._timeline = None

        # Callbacks
        self._on_person_detected: Optional[Callable[[PersonEvent], None]] = None
        self._on_person_identified: Optional[Callable[[PersonEvent], None]] = None
        self._on_unknown_person: Optional[Callable[[PersonEvent], None]] = None
        self._on_person_lost: Optional[Callable[[], None]] = None
        self._on_state_change: Optional[Callable[[HybridVisionState, HybridVisionState], None]] = None

        # Timers
        self._idle_timer = None
        self._last_face_time = 0
        self._last_recognition_time = 0

        # Control
        self._running = False
        self._stop_event = threading.Event()
        self._analysis_thread = None

        # Last recognized person (to avoid repeat announcements)
        self._last_person_id = None
        self._last_person_time = 0

        logger.info("HybridVision orchestrator initialized")

    def _get_timeline(self):
        """Get Timeline instance."""
        if self._timeline is None:
            try:
                from core.timeline import get_timeline
                self._timeline = get_timeline()
            except ImportError:
                pass
        return self._timeline

    def _get_hailo_analyzer(self):
        """Get HailoAnalyzer instance."""
        if self._hailo_analyzer is None:
            from .hailo_analyzer import get_hailo_analyzer
            self._hailo_analyzer = get_hailo_analyzer()
        return self._hailo_analyzer

    def _set_state(self, new_state: HybridVisionState):
        """Set pipeline state with callback."""
        with self._state_lock:
            old_state = self.state
            if old_state == new_state:
                return

            self.state = new_state
            logger.debug(f"HybridVision state: {old_state.value} -> {new_state.value}")

            if self._on_state_change:
                try:
                    self._on_state_change(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    def start(self,
              on_person_detected: Callable[[PersonEvent], None] = None,
              on_person_identified: Callable[[PersonEvent], None] = None,
              on_unknown_person: Callable[[PersonEvent], None] = None,
              on_person_lost: Callable[[], None] = None,
              on_state_change: Callable = None) -> bool:
        """
        Start hybrid vision pipeline.

        Args:
            on_person_detected: Called when any face is detected (before ID)
            on_person_identified: Called when known person is identified
            on_unknown_person: Called when face detected but not recognized
            on_person_lost: Called when face disappears
            on_state_change: Called when pipeline state changes

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("HybridVision already running")
            return True

        # Set callbacks
        self._on_person_detected = on_person_detected
        self._on_person_identified = on_person_identified
        self._on_unknown_person = on_unknown_person
        self._on_person_lost = on_person_lost
        self._on_state_change = on_state_change

        self._running = True
        self._stop_event.clear()
        self._set_state(HybridVisionState.IDLE)

        # Log startup
        timeline = self._get_timeline()
        if timeline:
            timeline.system_start("hybrid_vision")

        logger.info("HybridVision pipeline started")
        return True

    def stop(self):
        """Stop pipeline and release resources."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Put Hailo in standby
        analyzer = self._get_hailo_analyzer()
        if analyzer and analyzer.is_active:
            active_time = analyzer.standby()
            timeline = self._get_timeline()
            if timeline:
                timeline.log("system", "hailo_standby", active_seconds=active_time)

        # Stop analysis thread
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)
            self._analysis_thread = None

        # Cancel idle timer
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None

        self._set_state(HybridVisionState.IDLE)

        # Log shutdown
        timeline = self._get_timeline()
        if timeline:
            timeline.system_stop("hybrid_vision")

        logger.info("HybridVision pipeline stopped")

    def _wake_hailo_and_analyze(self):
        """Wake Hailo and start face recognition."""
        analyzer = self._get_hailo_analyzer()

        if not analyzer.is_available:
            logger.warning("Hailo not available, skipping recognition")
            self._set_state(HybridVisionState.IDLE)
            return

        # Wake Hailo if not active
        if not analyzer.is_active:
            if analyzer.wake():
                timeline = self._get_timeline()
                if timeline:
                    timeline.log("system", "hailo_wake")

        self._set_state(HybridVisionState.ANALYZING)

        # Run recognition in background
        if self._analysis_thread is None or not self._analysis_thread.is_alive():
            self._analysis_thread = threading.Thread(
                target=self._recognition_loop,
                name="HybridVisionAnalysis",
                daemon=True
            )
            self._analysis_thread.start()

    def _recognition_loop(self):
        """Background recognition loop while face is visible."""
        analyzer = self._get_hailo_analyzer()

        while self._running:
            try:
                # Rate limit recognition attempts
                if time.time() - self._last_recognition_time < self.RECOGNITION_INTERVAL:
                    self._stop_event.wait(0.5)
                    continue

                self._last_recognition_time = time.time()

                # Legacy: This loop is no longer used without XIAO trigger
                break

                detections = frame_data.get("detections", [])
                image_b64 = frame_data.get("image")

                if not detections:
                    continue

                # Try to recognize
                result = analyzer.recognize(
                    image_data=base64.b64decode(image_b64) if image_b64 else b'',
                    detections=detections
                )

                if result:
                    self._handle_recognition_result(result, image_b64)

            except Exception as e:
                logger.error(f"Recognition loop error: {e}")

            self._stop_event.wait(0.5)

        # No longer analyzing
        if self.state == HybridVisionState.ANALYZING:
            self._set_state(HybridVisionState.IDLE)

    def _handle_recognition_result(self, result, image_b64: str = None):
        """Handle face recognition result."""
        # Check if this is a new person or same as before
        is_new_person = (
            result.person_id != self._last_person_id or
            time.time() - self._last_person_time > 60  # Re-announce after 60s
        )

        event = PersonEvent(
            timestamp=time.time(),
            person_id=result.person_id,
            person_name=result.person_name,
            confidence=result.confidence,
            is_known=result.is_known,
            face_count=result.face_count,
            source="hailo",
            bbox=result.bbox,
            image_base64=image_b64
        )

        # Log to timeline
        timeline = self._get_timeline()
        if timeline:
            if result.is_known:
                timeline.log("vision", "person_identified",
                            person_name=result.person_name,
                            person_id=result.person_id,
                            confidence=round(result.confidence, 2))
            else:
                timeline.log("vision", "unknown_person",
                            confidence=round(result.confidence, 2))

        # Notify callbacks (only for new person)
        if is_new_person:
            self._last_person_id = result.person_id
            self._last_person_time = time.time()

            if result.is_known and self._on_person_identified:
                try:
                    self._on_person_identified(event)
                except Exception as e:
                    logger.error(f"on_person_identified callback error: {e}")

            elif not result.is_known and self._on_unknown_person:
                try:
                    self._on_unknown_person(event)
                except Exception as e:
                    logger.error(f"on_unknown_person callback error: {e}")

    def _start_idle_timer(self):
        """Start timer for Hailo standby."""
        if self._idle_timer:
            self._idle_timer.cancel()

        self._idle_timer = threading.Timer(
            self.HAILO_IDLE_TIMEOUT,
            self._idle_timer_expired
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_timer_expired(self):
        """Called when idle timer expires - put Hailo in standby."""
        if not self._running:
            return

        analyzer = self._get_hailo_analyzer()
        if analyzer and analyzer.is_active:
            active_time = analyzer.standby()
            logger.info(f"Hailo standby after {self.HAILO_IDLE_TIMEOUT}s idle (was active {active_time:.1f}s)")

            timeline = self._get_timeline()
            if timeline:
                timeline.log("system", "hailo_standby", active_seconds=active_time)

        self._set_state(HybridVisionState.IDLE)

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        analyzer = self._get_hailo_analyzer()

        status = {
            "state": self.state.value,
            "running": self._running,
            "hailo_available": analyzer.is_available if analyzer else False,
            "hailo_state": analyzer.state.value if analyzer else "unknown"
        }

        if analyzer:
            hailo_status = analyzer.get_status()
            status["known_persons"] = hailo_status.get("known_persons", 0)

        return status

    def get_known_persons(self) -> List[Dict[str, Any]]:
        """Get list of known persons."""
        analyzer = self._get_hailo_analyzer()
        if analyzer:
            return analyzer.list_known_persons()
        return []

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running


# Singleton accessor
_hybrid_vision: Optional[HybridVision] = None
_hv_lock = threading.Lock()


def get_hybrid_vision() -> HybridVision:
    """Get or create HybridVision instance."""
    global _hybrid_vision
    if _hybrid_vision is None:
        with _hv_lock:
            if _hybrid_vision is None:
                _hybrid_vision = HybridVision()
    return _hybrid_vision
