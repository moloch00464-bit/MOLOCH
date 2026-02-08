#!/usr/bin/env python3
"""
M.O.L.O.C.H. Mode Manager
==========================

Central controller for Hailo task switching between VISION and VOICE modes.
Ensures exclusive access to Hailo NPU - only one HEF loaded at a time.

Modes:
    IDLE    - No active inference, NPU available
    VISION  - Pose detection/tracking active
    VOICE   - Whisper speech recognition active

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import time
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operating modes."""
    IDLE = "idle"       # No active inference
    VISION = "vision"   # Pose detection active
    VOICE = "voice"     # Speech recognition active
    SWITCHING = "switching"  # Transitioning between modes


@dataclass
class ModeConfig:
    """Mode manager configuration."""
    # Model paths (secondary SSD)
    vision_hef: str = "/mnt/moloch-data/hailo/models/yolov8m_pose_h10.hef"
    voice_hef: str = "/mnt/moloch-data/hailo/models/whisper_h10.hef"

    # Timing
    voice_timeout_sec: float = 3.0      # Return to vision after no speech for this long
    mode_switch_delay_sec: float = 0.5  # Delay between unload/load HEF

    # Safety
    max_switch_retries: int = 3
    vision_restart_on_freeze_sec: float = 5.0  # Restart vision if no frames for this long


@dataclass
class ModeState:
    """Current mode state."""
    mode: SystemMode = SystemMode.IDLE
    mode_since: float = field(default_factory=time.time)
    last_voice_activity: float = 0.0
    last_vision_frame: float = 0.0
    switch_count: int = 0
    errors: List[str] = field(default_factory=list)

    def mode_duration_sec(self) -> float:
        return time.time() - self.mode_since


class ModeManager:
    """
    Central controller for Hailo task switching.

    Ensures exclusive NPU access by managing HEF loading/unloading.
    Only one mode can be active at a time.
    """

    def __init__(self, config: ModeConfig = None):
        self.config = config or ModeConfig()
        self.state = ModeState()

        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Callbacks for mode transitions
        self._on_mode_change: List[Callable[[SystemMode, SystemMode], None]] = []

        # Pipeline references (set externally)
        self._vision_pipeline = None
        self._voice_pipeline = None

        # Statistics
        self.stats = {
            "mode_switches": 0,
            "voice_sessions": 0,
            "vision_restarts": 0,
            "errors": 0
        }

        logger.info("ModeManager initialized")

    def register_vision_pipeline(self, pipeline):
        """Register the vision pipeline for control."""
        self._vision_pipeline = pipeline
        logger.info(f"Vision pipeline registered: {type(pipeline).__name__}")

    def register_voice_pipeline(self, pipeline):
        """Register the voice pipeline for control."""
        self._voice_pipeline = pipeline
        logger.info(f"Voice pipeline registered: {type(pipeline).__name__}")

    def add_mode_change_listener(self, callback: Callable[[SystemMode, SystemMode], None]):
        """Add callback for mode changes (old_mode, new_mode)."""
        self._on_mode_change.append(callback)

    def start(self) -> bool:
        """Start the mode manager monitoring thread."""
        if self._running:
            return True

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        # Start in VISION mode by default
        self._switch_to_vision()

        logger.info("ModeManager started")
        return True

    def stop(self):
        """Stop the mode manager."""
        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        # Switch to IDLE
        self._set_mode(SystemMode.IDLE)

        logger.info(f"ModeManager stopped (switches={self.stats['mode_switches']})")

    # === Mode Switching ===

    def request_voice_mode(self) -> bool:
        """
        Request switch to VOICE mode (e.g., push-to-talk pressed).

        Returns:
            True if switch successful or already in VOICE mode
        """
        with self._lock:
            if self.state.mode == SystemMode.VOICE:
                self.state.last_voice_activity = time.time()
                return True

            if self.state.mode == SystemMode.SWITCHING:
                logger.warning("Mode switch already in progress")
                return False

            return self._switch_to_voice()

    def request_vision_mode(self) -> bool:
        """
        Request switch to VISION mode.

        Returns:
            True if switch successful or already in VISION mode
        """
        with self._lock:
            if self.state.mode == SystemMode.VISION:
                return True

            if self.state.mode == SystemMode.SWITCHING:
                logger.warning("Mode switch already in progress")
                return False

            return self._switch_to_vision()

    def notify_voice_activity(self):
        """Notify that voice activity is ongoing (resets voice timeout)."""
        with self._lock:
            self.state.last_voice_activity = time.time()

    def notify_voice_complete(self):
        """Notify that voice session is complete - switch back to vision."""
        with self._lock:
            logger.info("[MODE] Voice complete - returning to vision")
            self._switch_to_vision()

    def notify_vision_frame(self):
        """Notify that a vision frame was received (for health monitoring)."""
        with self._lock:
            self.state.last_vision_frame = time.time()

    # === Internal Mode Switching ===

    def _switch_to_vision(self) -> bool:
        """Internal: Switch to VISION mode."""
        old_mode = self.state.mode

        try:
            self._set_mode(SystemMode.SWITCHING)

            # Stop voice pipeline if running
            if self._voice_pipeline:
                logger.info("[MODE] Stopping voice pipeline...")
                try:
                    self._voice_pipeline.stop()
                except Exception as e:
                    logger.error(f"Error stopping voice pipeline: {e}")

            # Small delay for NPU release
            time.sleep(self.config.mode_switch_delay_sec)

            # Start vision pipeline
            if self._vision_pipeline:
                logger.info("[MODE] Starting vision pipeline...")
                try:
                    self._vision_pipeline.start()
                except Exception as e:
                    logger.error(f"Error starting vision pipeline: {e}")
                    self._set_mode(SystemMode.IDLE)
                    return False

            self._set_mode(SystemMode.VISION)
            self.stats["mode_switches"] += 1

            # Notify listeners
            self._notify_mode_change(old_mode, SystemMode.VISION)

            logger.info(f"[MODE] Switched to VISION (from {old_mode.value})")
            return True

        except Exception as e:
            logger.error(f"[MODE] Switch to VISION failed: {e}")
            self.state.errors.append(f"vision_switch: {e}")
            self.stats["errors"] += 1
            self._set_mode(SystemMode.IDLE)
            return False

    def _switch_to_voice(self) -> bool:
        """Internal: Switch to VOICE mode."""
        old_mode = self.state.mode

        try:
            self._set_mode(SystemMode.SWITCHING)
            self.state.last_voice_activity = time.time()

            # Stop vision pipeline
            if self._vision_pipeline:
                logger.info("[MODE] Stopping vision pipeline...")
                try:
                    self._vision_pipeline.stop()
                except Exception as e:
                    logger.error(f"Error stopping vision pipeline: {e}")

            # Small delay for NPU release
            time.sleep(self.config.mode_switch_delay_sec)

            # Start voice pipeline
            if self._voice_pipeline:
                logger.info("[MODE] Starting voice pipeline...")
                try:
                    self._voice_pipeline.start()
                except Exception as e:
                    logger.error(f"Error starting voice pipeline: {e}")
                    # Fallback to vision
                    self._switch_to_vision()
                    return False

            self._set_mode(SystemMode.VOICE)
            self.stats["mode_switches"] += 1
            self.stats["voice_sessions"] += 1

            # Notify listeners
            self._notify_mode_change(old_mode, SystemMode.VOICE)

            logger.info(f"[MODE] Switched to VOICE (from {old_mode.value})")
            return True

        except Exception as e:
            logger.error(f"[MODE] Switch to VOICE failed: {e}")
            self.state.errors.append(f"voice_switch: {e}")
            self.stats["errors"] += 1
            # Fallback to vision
            self._switch_to_vision()
            return False

    def _set_mode(self, mode: SystemMode):
        """Internal: Set mode and update timestamp."""
        self.state.mode = mode
        self.state.mode_since = time.time()
        self.state.switch_count += 1

    def _notify_mode_change(self, old_mode: SystemMode, new_mode: SystemMode):
        """Notify all listeners of mode change."""
        for callback in self._on_mode_change:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Mode change callback error: {e}")

    # === Monitoring ===

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("[MODE] Monitor loop started")

        while self._running:
            try:
                self._check_mode_health()
            except Exception as e:
                logger.error(f"[MODE] Monitor error: {e}")

            time.sleep(1.0)  # Check every second

    def _check_mode_health(self):
        """Check mode health and perform automatic transitions."""
        with self._lock:
            now = time.time()

            # === VOICE timeout check ===
            if self.state.mode == SystemMode.VOICE:
                voice_idle_time = now - self.state.last_voice_activity
                if voice_idle_time > self.config.voice_timeout_sec:
                    logger.info(f"[MODE] Voice timeout ({voice_idle_time:.1f}s) - returning to vision")
                    self._switch_to_vision()

            # === VISION health check ===
            elif self.state.mode == SystemMode.VISION:
                if self.state.last_vision_frame > 0:
                    vision_idle_time = now - self.state.last_vision_frame
                    if vision_idle_time > self.config.vision_restart_on_freeze_sec:
                        logger.warning(f"[MODE] Vision freeze detected ({vision_idle_time:.1f}s) - restarting")
                        self.stats["vision_restarts"] += 1
                        self._restart_vision()

    def _restart_vision(self):
        """Restart vision pipeline after freeze."""
        if self._vision_pipeline:
            try:
                logger.info("[MODE] Restarting vision pipeline...")
                self._vision_pipeline.stop()
                time.sleep(0.5)
                self._vision_pipeline.start()
                self.state.last_vision_frame = time.time()
                logger.info("[MODE] Vision pipeline restarted")
            except Exception as e:
                logger.error(f"[MODE] Vision restart failed: {e}")

    # === Status ===

    def get_mode(self) -> SystemMode:
        """Get current mode."""
        return self.state.mode

    def is_vision_active(self) -> bool:
        """Check if vision mode is active."""
        return self.state.mode == SystemMode.VISION

    def is_voice_active(self) -> bool:
        """Check if voice mode is active."""
        return self.state.mode == SystemMode.VOICE

    def get_status(self) -> Dict[str, Any]:
        """Get full status."""
        with self._lock:
            return {
                "mode": self.state.mode.value,
                "mode_duration_sec": self.state.mode_duration_sec(),
                "switch_count": self.state.switch_count,
                "last_voice_activity": self.state.last_voice_activity,
                "last_vision_frame": self.state.last_vision_frame,
                "stats": self.stats.copy(),
                "errors": self.state.errors[-5:],  # Last 5 errors
                "config": {
                    "vision_hef": self.config.vision_hef,
                    "voice_hef": self.config.voice_hef,
                    "voice_timeout_sec": self.config.voice_timeout_sec
                }
            }


# === Singleton ===
_mode_manager: Optional[ModeManager] = None
_mode_manager_lock = threading.Lock()


def get_mode_manager() -> ModeManager:
    """Get or create singleton ModeManager."""
    global _mode_manager
    with _mode_manager_lock:
        if _mode_manager is None:
            _mode_manager = ModeManager()
    return _mode_manager
