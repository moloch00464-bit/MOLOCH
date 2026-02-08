#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. System Autonomy Layer
==================================

Controlled autonomy for self-monitoring, self-repair, and resource management.

Capabilities:
- Health monitoring (NPU, RTSP, CPU, memory)
- Self-repair for pipeline failures
- NPU resource allocation (speech vs vision)
- Autonomous state transitions
- Safety rules enforcement

Author: M.O.L.O.C.H. System
Date: 2026-02-04
"""

import os
import time
import logging
import threading
import subprocess
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND STATE DEFINITIONS
# ============================================================================

class NPUMode(Enum):
    """NPU resource allocation mode."""
    IDLE = "idle"           # NPU not in use
    VISION = "vision"       # NPU used for pose/detection
    SPEECH = "speech"       # NPU used for Whisper
    SWITCHING = "switching" # Transitioning between modes


class SystemHealth(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class PipelineState(Enum):
    """Vision pipeline state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    RESTARTING = "restarting"


class TrackingMode(Enum):
    """Tracking behavior mode."""
    DISABLED = "disabled"
    TRACKING = "tracking"
    SEARCHING = "searching"
    LOCKED = "locked"


# ============================================================================
# HEALTH METRICS
# ============================================================================

@dataclass
class HealthMetrics:
    """System health metrics snapshot."""
    # NPU Status
    npu_available: bool = False
    npu_mode: NPUMode = NPUMode.IDLE
    npu_temperature: float = 0.0

    # RTSP Stream
    rtsp_connected: bool = False
    rtsp_fps: float = 0.0
    rtsp_errors: int = 0

    # System Resources
    cpu_percent: float = 0.0
    cpu_temperature: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0

    # Pipeline Status
    pipeline_state: PipelineState = PipelineState.STOPPED
    last_frame_time: float = 0.0
    frame_count: int = 0

    # Perception Status
    perception_active: bool = False
    last_detection_time: float = 0.0
    user_visible: bool = False

    # Tracking Status
    tracking_mode: TrackingMode = TrackingMode.DISABLED
    tracking_active: bool = False

    # Timestamps
    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0


@dataclass
class SystemState:
    """Complete system state for autonomy decisions."""
    health: SystemHealth = SystemHealth.HEALTHY
    metrics: HealthMetrics = field(default_factory=HealthMetrics)

    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_recovery_time: float = 0.0
    total_recoveries: int = 0

    # Mode flags
    speech_active: bool = False
    vision_active: bool = False
    autonomous_mode: bool = True

    # Safety
    user_override_active: bool = False
    last_user_interaction: float = 0.0


# ============================================================================
# SYSTEM AUTONOMY MANAGER
# ============================================================================

class SystemAutonomy:
    """
    M.O.L.O.C.H. System Autonomy Manager.

    Monitors system health, manages resources, and performs self-repair
    while respecting safety rules and user overrides.
    """

    # Health check intervals
    HEALTH_CHECK_INTERVAL = 2.0  # seconds
    PERCEPTION_TIMEOUT = 5.0     # seconds before pipeline reset
    RTSP_TIMEOUT = 10.0          # seconds before RTSP reconnect

    # Thresholds
    CPU_WARNING_THRESHOLD = 80.0
    CPU_CRITICAL_THRESHOLD = 95.0
    MEMORY_WARNING_THRESHOLD = 85.0
    MEMORY_CRITICAL_THRESHOLD = 95.0
    TEMP_WARNING_THRESHOLD = 75.0
    TEMP_CRITICAL_THRESHOLD = 85.0

    # Recovery limits
    MAX_CONSECUTIVE_FAILURES = 5
    RECOVERY_COOLDOWN = 30.0  # seconds between recovery attempts

    def __init__(self):
        self._state = SystemState()
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_time = time.time()

        # Callbacks
        self._on_health_change: Optional[Callable[[SystemHealth, str], None]] = None
        self._on_recovery_action: Optional[Callable[[str], None]] = None

        # Component references (set by external modules)
        self._pose_detector = None
        self._whisper = None
        self._tracker = None
        self._perception_state = None

        # Action history for logging
        self._action_history: List[Dict[str, Any]] = []

        logger.info("[AUTONOMY] System Autonomy Manager initialized")

    # ========================================================================
    # COMPONENT REGISTRATION
    # ========================================================================

    def register_pose_detector(self, detector) -> None:
        """Register pose detector for management."""
        self._pose_detector = detector
        logger.info("[AUTONOMY] Pose detector registered")

    def register_whisper(self, whisper) -> None:
        """Register Whisper for NPU management."""
        self._whisper = whisper
        logger.info("[AUTONOMY] Whisper registered")

    def register_tracker(self, tracker) -> None:
        """Register autonomous tracker."""
        self._tracker = tracker
        logger.info("[AUTONOMY] Tracker registered")

    def register_perception_state(self, perception) -> None:
        """Register perception state."""
        self._perception_state = perception
        logger.info("[AUTONOMY] Perception state registered")

    # ========================================================================
    # HEALTH MONITORING
    # ========================================================================

    def start(self) -> None:
        """Start autonomy monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemAutonomy"
        )
        self._monitor_thread.start()
        logger.info("[AUTONOMY] Monitoring started")

    def stop(self) -> None:
        """Stop autonomy monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None
        logger.info("[AUTONOMY] Monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._update_health_metrics()
                self._evaluate_health()
                self._check_recovery_needed()
            except Exception as e:
                logger.error(f"[AUTONOMY] Monitor error: {e}")

            time.sleep(self.HEALTH_CHECK_INTERVAL)

    def _update_health_metrics(self) -> None:
        """Update all health metrics."""
        with self._lock:
            metrics = self._state.metrics
            metrics.timestamp = time.time()
            metrics.uptime_seconds = time.time() - self._start_time

            # NPU Status
            metrics.npu_available = self._check_npu_available()
            metrics.npu_mode = self._get_npu_mode()

            # System Resources
            self._update_system_resources(metrics)

            # Pipeline Status
            self._update_pipeline_status(metrics)

            # Perception Status
            self._update_perception_status(metrics)

            # Tracking Status
            self._update_tracking_status(metrics)

    def _check_npu_available(self) -> bool:
        """Check if Hailo NPU device is available."""
        try:
            return os.path.exists("/dev/hailo0")
        except:
            return False

    def _get_npu_mode(self) -> NPUMode:
        """Determine current NPU usage mode."""
        with self._lock:
            if self._state.speech_active:
                return NPUMode.SPEECH
            elif self._state.vision_active:
                return NPUMode.VISION
            return NPUMode.IDLE

    def _update_system_resources(self, metrics: HealthMetrics) -> None:
        """Update CPU, memory, temperature metrics."""
        try:
            # CPU percent
            with open("/proc/loadavg", "r") as f:
                load = float(f.read().split()[0])
                # Approximate CPU percent from load average
                cpu_count = os.cpu_count() or 4
                metrics.cpu_percent = min(100.0, (load / cpu_count) * 100)

            # Memory
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1])
                        meminfo[key] = value

                total = meminfo.get("MemTotal", 1)
                available = meminfo.get("MemAvailable", 0)
                metrics.memory_available_mb = available / 1024
                metrics.memory_percent = 100.0 * (1 - available / total)

            # CPU Temperature
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    metrics.cpu_temperature = int(f.read().strip()) / 1000.0
            except:
                metrics.cpu_temperature = 0.0

        except Exception as e:
            logger.debug(f"[AUTONOMY] Resource check error: {e}")

    def _update_pipeline_status(self, metrics: HealthMetrics) -> None:
        """Update vision pipeline status."""
        if self._pose_detector:
            try:
                metrics.pipeline_state = (
                    PipelineState.RUNNING if self._pose_detector._running
                    else PipelineState.STOPPED
                )
                metrics.frame_count = self._pose_detector._frame_count
                metrics.last_frame_time = self._pose_detector._last_frame_time
                metrics.rtsp_fps = self._pose_detector.fps
                metrics.rtsp_connected = self._pose_detector._running
            except:
                pass

    def _update_perception_status(self, metrics: HealthMetrics) -> None:
        """Update perception state status."""
        if self._perception_state:
            try:
                snap = self._perception_state.get_snapshot()
                metrics.perception_active = True
                metrics.last_detection_time = snap.last_user_seen
                metrics.user_visible = snap.user_visible
            except:
                pass

    def _update_tracking_status(self, metrics: HealthMetrics) -> None:
        """Update tracker status."""
        if self._tracker:
            try:
                metrics.tracking_active = self._tracker.tracking_active
                state = self._tracker.state
                if state.value == "tracking":
                    metrics.tracking_mode = TrackingMode.TRACKING
                elif state.value == "searching":
                    metrics.tracking_mode = TrackingMode.SEARCHING
                elif state.value == "locked":
                    metrics.tracking_mode = TrackingMode.LOCKED
                else:
                    metrics.tracking_mode = TrackingMode.DISABLED
            except:
                pass

    # ========================================================================
    # HEALTH EVALUATION
    # ========================================================================

    def _evaluate_health(self) -> None:
        """Evaluate overall system health."""
        with self._lock:
            metrics = self._state.metrics
            old_health = self._state.health

            issues = []

            # Check CPU
            if metrics.cpu_percent > self.CPU_CRITICAL_THRESHOLD:
                issues.append(("critical", "CPU critical"))
            elif metrics.cpu_percent > self.CPU_WARNING_THRESHOLD:
                issues.append(("warning", "CPU high"))

            # Check Memory
            if metrics.memory_percent > self.MEMORY_CRITICAL_THRESHOLD:
                issues.append(("critical", "Memory critical"))
            elif metrics.memory_percent > self.MEMORY_WARNING_THRESHOLD:
                issues.append(("warning", "Memory high"))

            # Check Temperature
            if metrics.cpu_temperature > self.TEMP_CRITICAL_THRESHOLD:
                issues.append(("critical", "Temperature critical"))
            elif metrics.cpu_temperature > self.TEMP_WARNING_THRESHOLD:
                issues.append(("warning", "Temperature high"))

            # Check NPU
            if not metrics.npu_available:
                issues.append(("critical", "NPU unavailable"))

            # Check Pipeline
            if metrics.pipeline_state == PipelineState.ERROR:
                issues.append(("critical", "Pipeline error"))
            elif (metrics.pipeline_state == PipelineState.RUNNING and
                  time.time() - metrics.last_frame_time > self.PERCEPTION_TIMEOUT):
                issues.append(("warning", "Pipeline stalled"))

            # Determine overall health
            critical_count = sum(1 for level, _ in issues if level == "critical")
            warning_count = sum(1 for level, _ in issues if level == "warning")

            if critical_count > 0:
                new_health = SystemHealth.CRITICAL
            elif warning_count > 0:
                new_health = SystemHealth.DEGRADED
            elif self._state.consecutive_failures > 0:
                new_health = SystemHealth.RECOVERING
            else:
                new_health = SystemHealth.HEALTHY

            # Log health change
            if new_health != old_health:
                issue_str = ", ".join(msg for _, msg in issues) if issues else "all clear"
                logger.info(f"[AUTONOMY] Health: {old_health.value} -> {new_health.value} ({issue_str})")
                self._state.health = new_health

                if self._on_health_change:
                    self._on_health_change(new_health, issue_str)

    # ========================================================================
    # SELF-REPAIR LOGIC
    # ========================================================================

    def _check_recovery_needed(self) -> None:
        """Check if recovery actions are needed."""
        if not self._state.autonomous_mode:
            return

        if self._state.user_override_active:
            return

        now = time.time()
        metrics = self._state.metrics

        # Check cooldown
        if now - self._state.last_recovery_time < self.RECOVERY_COOLDOWN:
            return

        # Check failure limit
        if self._state.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning("[AUTONOMY] Max failures reached, waiting for manual intervention")
            return

        # === SAFETY: Never recover while voice/STT is using the NPU! ===
        # Pipeline was stopped intentionally for voice priority.
        try:
            from core.hardware.hailo_manager import get_hailo_manager, HailoConsumer
            mgr = get_hailo_manager()
            if mgr.current_consumer == HailoConsumer.VOICE:
                return  # Voice active - all recovery disabled
        except Exception:
            pass

        # Check for stalled pipeline
        if (metrics.pipeline_state == PipelineState.RUNNING and
            metrics.last_frame_time > 0 and
            now - metrics.last_frame_time > self.PERCEPTION_TIMEOUT):
            self._trigger_recovery("pipeline_stalled", "No frames for >5s")
            return

        # Check for stopped pipeline that should be running
        if (metrics.pipeline_state == PipelineState.STOPPED and
            self._state.vision_active and
            self._pose_detector):
            self._trigger_recovery("pipeline_stopped", "Pipeline stopped unexpectedly")
            return

        # Check for perception silence
        if (metrics.perception_active and
            metrics.last_detection_time > 0 and
            now - metrics.last_detection_time > self.PERCEPTION_TIMEOUT * 2):
            # This might be normal if no one is in view
            pass

    def _trigger_recovery(self, reason: str, description: str) -> None:
        """Trigger a recovery action."""
        with self._lock:
            now = time.time()
            self._state.consecutive_failures += 1
            self._state.last_failure_time = now

            action = {
                "timestamp": now,
                "reason": reason,
                "description": description,
                "attempt": self._state.consecutive_failures
            }
            self._action_history.append(action)

            logger.warning(f"[AUTONOMY] Recovery triggered: {reason} - {description} "
                          f"(attempt {self._state.consecutive_failures})")

        # Perform recovery based on reason
        success = False

        if reason in ["pipeline_stalled", "pipeline_stopped"]:
            success = self._recover_pipeline()
        elif reason == "npu_busy":
            success = self._recover_npu()
        elif reason == "tracker_inactive":
            success = self._recover_tracker()

        with self._lock:
            if success:
                self._state.last_recovery_time = time.time()
                self._state.total_recoveries += 1
                self._state.consecutive_failures = 0
                logger.info(f"[AUTONOMY] Recovery successful: {reason}")
            else:
                logger.error(f"[AUTONOMY] Recovery failed: {reason}")

        if self._on_recovery_action:
            self._on_recovery_action(reason)

    def _recover_pipeline(self) -> bool:
        """Attempt to recover the vision pipeline."""
        if not self._pose_detector:
            return False

        # SAFETY: Don't recover if voice is using NPU!
        try:
            from core.hardware.hailo_manager import get_hailo_manager, HailoConsumer
            mgr = get_hailo_manager()
            if mgr.current_consumer == HailoConsumer.VOICE:
                logger.info("[AUTONOMY] Skipping recovery - voice has NPU priority")
                return False
        except Exception:
            pass

        try:
            logger.info("[AUTONOMY] Attempting pipeline recovery...")

            # Stop existing pipeline
            self._pose_detector.stop()
            time.sleep(1.0)

            # Restart pipeline (use normal NPU check, not skip!)
            success = self._pose_detector.start(skip_npu_check=False)

            if success:
                logger.info("[AUTONOMY] Pipeline restarted successfully")
            return success

        except Exception as e:
            logger.error(f"[AUTONOMY] Pipeline recovery error: {e}")
            return False

    def _recover_npu(self) -> bool:
        """Attempt to recover NPU access."""
        try:
            logger.info("[AUTONOMY] Attempting NPU recovery...")

            # Release Whisper if holding NPU
            if self._whisper and hasattr(self._whisper, 'release'):
                self._whisper.release()

            time.sleep(0.5)

            # Check if NPU is now available
            if self._check_npu_available():
                logger.info("[AUTONOMY] NPU recovered")
                return True

            return False

        except Exception as e:
            logger.error(f"[AUTONOMY] NPU recovery error: {e}")
            return False

    def _recover_tracker(self) -> bool:
        """Attempt to recover the tracker."""
        if not self._tracker:
            return False

        try:
            logger.info("[AUTONOMY] Attempting tracker recovery...")

            # Restart tracker
            if hasattr(self._tracker, 'stop'):
                self._tracker.stop()
            time.sleep(0.5)
            if hasattr(self._tracker, 'start'):
                self._tracker.start()

            return True

        except Exception as e:
            logger.error(f"[AUTONOMY] Tracker recovery error: {e}")
            return False

    # ========================================================================
    # NPU RESOURCE MANAGEMENT
    # ========================================================================

    def request_npu_for_speech(self) -> bool:
        """
        Request NPU for speech processing.

        Stops vision pipeline if running and returns True when NPU is available.
        """
        with self._lock:
            if self._state.speech_active:
                return True  # Already in speech mode

            logger.info("[AUTONOMY] NPU requested for speech")

            # Stop vision pipeline
            if self._pose_detector and self._pose_detector._running:
                logger.info("[AUTONOMY] Stopping vision for speech...")
                self._pose_detector.stop()
                self._state.vision_active = False

            time.sleep(0.5)
            self._state.speech_active = True
            self._state.metrics.npu_mode = NPUMode.SPEECH

            return True

    def release_npu_from_speech(self) -> None:
        """
        Release NPU after speech processing.

        Restarts vision pipeline if it was running.
        """
        with self._lock:
            if not self._state.speech_active:
                return

            logger.info("[AUTONOMY] Releasing NPU from speech")
            self._state.speech_active = False

            # Release Whisper resources
            if self._whisper and hasattr(self._whisper, 'release'):
                self._whisper.release()

            time.sleep(0.5)

            # Restart vision pipeline
            if self._pose_detector and self._state.autonomous_mode:
                logger.info("[AUTONOMY] Restarting vision after speech...")
                if self._pose_detector.start(skip_npu_check=True):
                    self._state.vision_active = True
                    self._state.metrics.npu_mode = NPUMode.VISION
                else:
                    logger.error("[AUTONOMY] Failed to restart vision after speech")
                    self._state.metrics.npu_mode = NPUMode.IDLE

    def start_vision(self) -> bool:
        """Start vision pipeline (if NPU available)."""
        with self._lock:
            if self._state.speech_active:
                logger.warning("[AUTONOMY] Cannot start vision - speech active")
                return False

            if self._state.vision_active:
                return True

            if self._pose_detector:
                if self._pose_detector.start():
                    self._state.vision_active = True
                    self._state.metrics.npu_mode = NPUMode.VISION
                    logger.info("[AUTONOMY] Vision started")
                    return True

            return False

    def stop_vision(self) -> None:
        """Stop vision pipeline."""
        with self._lock:
            if self._pose_detector and self._pose_detector._running:
                self._pose_detector.stop()
            self._state.vision_active = False
            self._state.metrics.npu_mode = NPUMode.IDLE
            logger.info("[AUTONOMY] Vision stopped")

    # ========================================================================
    # STATE ACCESS
    # ========================================================================

    def get_state(self) -> SystemState:
        """Get current system state."""
        with self._lock:
            return self._state

    def get_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        with self._lock:
            return self._state.metrics

    def get_health(self) -> SystemHealth:
        """Get current health status."""
        with self._lock:
            return self._state.health

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        with self._lock:
            return self._state.health in [SystemHealth.HEALTHY, SystemHealth.RECOVERING]

    def set_autonomous_mode(self, enabled: bool) -> None:
        """Enable/disable autonomous recovery."""
        with self._lock:
            self._state.autonomous_mode = enabled
            logger.info(f"[AUTONOMY] Autonomous mode: {'enabled' if enabled else 'disabled'}")

    def set_user_override(self, active: bool) -> None:
        """Set user override flag (pauses autonomous actions)."""
        with self._lock:
            self._state.user_override_active = active
            self._state.last_user_interaction = time.time()
            if active:
                logger.info("[AUTONOMY] User override ACTIVE")

    def set_health_callback(self, callback: Callable[[SystemHealth, str], None]) -> None:
        """Set callback for health status changes."""
        self._on_health_change = callback

    def set_recovery_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for recovery actions."""
        self._on_recovery_action = callback

    # ========================================================================
    # STATUS REPORTING
    # ========================================================================

    def describe(self) -> str:
        """Get natural language description of system status."""
        with self._lock:
            metrics = self._state.metrics
            health = self._state.health

            parts = []

            # Health
            if health == SystemHealth.HEALTHY:
                parts.append("Alle Systeme normal")
            elif health == SystemHealth.DEGRADED:
                parts.append("System leicht belastet")
            elif health == SystemHealth.CRITICAL:
                parts.append("WARNUNG: Systemprobleme")
            elif health == SystemHealth.RECOVERING:
                parts.append("System erholt sich")

            # NPU Mode
            if metrics.npu_mode == NPUMode.VISION:
                parts.append("NPU: Sehen")
            elif metrics.npu_mode == NPUMode.SPEECH:
                parts.append("NPU: Hören")
            else:
                parts.append("NPU: Bereit")

            # Resources
            if metrics.cpu_percent > 70:
                parts.append(f"CPU {metrics.cpu_percent:.0f}%")
            if metrics.memory_percent > 70:
                parts.append(f"RAM {metrics.memory_percent:.0f}%")
            if metrics.cpu_temperature > 60:
                parts.append(f"Temp {metrics.cpu_temperature:.0f}°C")

            # Vision
            if metrics.pipeline_state == PipelineState.RUNNING:
                parts.append(f"Kamera: {metrics.rtsp_fps:.0f} FPS")
            elif metrics.pipeline_state == PipelineState.ERROR:
                parts.append("Kamera: FEHLER")

            return ", ".join(parts) + "."

    def get_status_dict(self) -> Dict[str, Any]:
        """Get complete status as dictionary."""
        with self._lock:
            metrics = self._state.metrics
            return {
                "health": self._state.health.value,
                "npu_mode": metrics.npu_mode.value,
                "npu_available": metrics.npu_available,
                "pipeline_state": metrics.pipeline_state.value,
                "rtsp_fps": metrics.rtsp_fps,
                "cpu_percent": metrics.cpu_percent,
                "cpu_temperature": metrics.cpu_temperature,
                "memory_percent": metrics.memory_percent,
                "user_visible": metrics.user_visible,
                "tracking_mode": metrics.tracking_mode.value,
                "autonomous_mode": self._state.autonomous_mode,
                "consecutive_failures": self._state.consecutive_failures,
                "total_recoveries": self._state.total_recoveries,
                "uptime_seconds": metrics.uptime_seconds
            }


# ============================================================================
# SINGLETON
# ============================================================================

_autonomy: Optional[SystemAutonomy] = None
_autonomy_lock = threading.Lock()


def get_system_autonomy() -> SystemAutonomy:
    """Get or create singleton SystemAutonomy."""
    global _autonomy
    if _autonomy is None:
        with _autonomy_lock:
            if _autonomy is None:
                _autonomy = SystemAutonomy()
    return _autonomy


# Convenience functions
def is_system_healthy() -> bool:
    """Quick check if system is healthy."""
    return get_system_autonomy().is_healthy()


def get_system_health() -> SystemHealth:
    """Get current system health."""
    return get_system_autonomy().get_health()


def request_npu_for_speech() -> bool:
    """Request NPU for speech processing."""
    return get_system_autonomy().request_npu_for_speech()


def release_npu_from_speech() -> None:
    """Release NPU after speech."""
    get_system_autonomy().release_npu_from_speech()
