#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hailo NPU Resource Manager
========================================

Central manager for Hailo-10H NPU resource allocation.
Ensures exclusive access - only one consumer (Vision OR Voice) at a time.

Problem solved:
    [HailoRT] [error] Failed to create vdevice. there are not enough free devices.
    [HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_OUT_OF_PHYSICAL_DEVICES(74)

Usage:
    manager = get_hailo_manager()

    # Acquire for vision (blocking if voice is using it)
    if manager.acquire_for_vision():
        # Use Hailo for vision...
        manager.release_vision()

    # Acquire for voice (will stop vision first)
    if manager.acquire_for_voice():
        # Use Hailo for voice...
        manager.release_voice()

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import threading
import time
import logging
import subprocess
import os
from enum import Enum
from typing import Optional, Callable, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Hailo device path
HAILO_DEVICE = "/dev/hailo0"


class HailoConsumer(Enum):
    """Who is currently using the Hailo NPU."""
    NONE = "none"
    VISION = "vision"      # GstHailoPoseDetector
    VOICE = "voice"        # HailoWhisper


@dataclass
class HailoState:
    """Current state of Hailo resource."""
    consumer: HailoConsumer = HailoConsumer.NONE
    acquired_at: float = 0.0
    released_at: float = 0.0
    acquire_count: int = 0
    release_count: int = 0
    switch_count: int = 0  # Vision<->Voice switches
    last_error: str = ""


class HailoManager:
    """
    Central Hailo NPU resource manager.

    Ensures exclusive access to the Hailo-10H NPU.
    Only one consumer (Vision or Voice) can use it at a time.

    Priority: Voice > Vision
    When voice acquisition is requested, vision is stopped first.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = HailoState()

        # Registered pipelines for cleanup
        self._vision_pipeline: Optional[Any] = None  # GstHailoPoseDetector
        self._voice_pipeline: Optional[Any] = None   # HailoWhisper

        # Callbacks for lifecycle events
        self._on_vision_stop: Optional[Callable[[], None]] = None
        self._on_vision_start: Optional[Callable[[], bool]] = None
        self._on_voice_stop: Optional[Callable[[], None]] = None
        self._on_voice_start: Optional[Callable[[], bool]] = None

        # Condition for waiting
        self._available = threading.Condition(self._lock)

        # Check device on startup
        device_free, pids = self._check_device_free()
        if not device_free:
            logger.warning(f"[HAILO_MGR] Device in use by PIDs: {pids} on startup")

        logger.info("[HAILO_MGR] HailoManager initialized")

    def _check_device_free(self) -> Tuple[bool, List[int]]:
        """Check if Hailo device file is free (not held by OTHER processes).

        Our own PID is excluded because we manage the device internally
        through HailoManager state - the VDevice fd may linger briefly
        after release() but that's normal and shouldn't block re-acquire.

        Returns:
            Tuple of (is_free, list_of_other_pids_holding_device)
        """
        if not os.path.exists(HAILO_DEVICE):
            logger.warning(f"[HAILO_MGR] Device {HAILO_DEVICE} does not exist!")
            return True, []

        try:
            result = subprocess.run(
                ["lsof", HAILO_DEVICE],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                # lsof returns non-zero if no process holds the file
                return True, []

            # Parse PIDs from lsof output, exclude our own process
            my_pid = os.getpid()
            pids = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        if pid != my_pid:
                            pids.append(pid)
                    except ValueError:
                        pass

            return len(pids) == 0, pids

        except subprocess.TimeoutExpired:
            logger.error("[HAILO_MGR] lsof timeout checking device")
            return False, []
        except Exception as e:
            logger.error(f"[HAILO_MGR] Error checking device: {e}")
            return True, []  # Assume free on error

    def _force_release_device(self, exclude_pid: int = None) -> bool:
        """Force release Hailo device by killing processes holding it.

        Args:
            exclude_pid: PID to exclude from killing (usually current process)

        Returns:
            True if device is now free
        """
        logger.warning("[HAILO_MGR] FORCE RELEASE DEVICE called")

        device_free, pids = self._check_device_free()
        if device_free:
            logger.info("[HAILO_MGR] Device already free")
            return True

        current_pid = os.getpid()
        killed = []

        for pid in pids:
            if pid == current_pid or pid == exclude_pid:
                logger.info(f"[HAILO_MGR] Skipping own/excluded PID {pid}")
                continue

            try:
                # Try SIGTERM first
                logger.warning(f"[HAILO_MGR] Sending SIGTERM to PID {pid}")
                os.kill(pid, 15)  # SIGTERM
                killed.append(pid)
            except ProcessLookupError:
                logger.info(f"[HAILO_MGR] PID {pid} already gone")
            except PermissionError:
                logger.error(f"[HAILO_MGR] No permission to kill PID {pid}")
            except Exception as e:
                logger.error(f"[HAILO_MGR] Error killing PID {pid}: {e}")

        # Wait a moment for processes to terminate
        if killed:
            time.sleep(0.5)

        # Check again
        device_free, remaining_pids = self._check_device_free()

        if not device_free and remaining_pids:
            # Force kill remaining
            for pid in remaining_pids:
                if pid == current_pid or pid == exclude_pid:
                    continue
                try:
                    logger.warning(f"[HAILO_MGR] Sending SIGKILL to PID {pid}")
                    os.kill(pid, 9)  # SIGKILL
                except Exception:
                    pass

            time.sleep(0.3)
            device_free, _ = self._check_device_free()

        if device_free:
            logger.info("[HAILO_MGR] Device successfully released")
        else:
            logger.error("[HAILO_MGR] Failed to release device!")

        return device_free

    def is_device_free(self) -> bool:
        """Check if Hailo device is actually free at system level."""
        free, _ = self._check_device_free()
        return free

    def ensure_device_free(self, timeout: float = 3.0) -> bool:
        """Ensure device is free, waiting or forcing if necessary.

        Args:
            timeout: Max time to wait before forcing

        Returns:
            True if device is free
        """
        start = time.time()

        while time.time() - start < timeout:
            if self.is_device_free():
                return True
            time.sleep(0.2)

        # Timeout - try force release
        logger.warning("[HAILO_MGR] Device not free after timeout, forcing release")
        return self._force_release_device()

    @property
    def current_consumer(self) -> HailoConsumer:
        """Get current consumer of Hailo NPU."""
        with self._lock:
            return self._state.consumer

    @property
    def is_available(self) -> bool:
        """Check if Hailo NPU is available."""
        with self._lock:
            return self._state.consumer == HailoConsumer.NONE

    def register_vision_pipeline(self, pipeline,
                                  on_stop: Callable[[], None] = None,
                                  on_start: Callable[[], bool] = None):
        """Register vision pipeline for managed lifecycle.

        Args:
            pipeline: GstHailoPoseDetector instance
            on_stop: Callback to stop the pipeline
            on_start: Callback to start the pipeline (returns success)
        """
        with self._lock:
            self._vision_pipeline = pipeline
            self._on_vision_stop = on_stop
            self._on_vision_start = on_start
            logger.info(f"[HAILO_MGR] Vision pipeline registered: {type(pipeline).__name__}")

    def register_voice_pipeline(self, pipeline,
                                 on_stop: Callable[[], None] = None,
                                 on_start: Callable[[], bool] = None):
        """Register voice pipeline for managed lifecycle.

        Args:
            pipeline: HailoWhisper instance
            on_stop: Callback to stop the pipeline
            on_start: Callback to start the pipeline (returns success)
        """
        with self._lock:
            self._voice_pipeline = pipeline
            self._on_voice_stop = on_stop
            self._on_voice_start = on_start
            logger.info(f"[HAILO_MGR] Voice pipeline registered: {type(pipeline).__name__}")

    def acquire_for_vision(self, timeout: float = 5.0) -> bool:
        """Acquire Hailo NPU for vision pipeline.

        Will wait if voice is currently using it (voice has priority).

        Args:
            timeout: Max time to wait for availability (seconds)

        Returns:
            True if acquired, False if timeout or error
        """
        logger.info("[HAILO_MGR] Vision acquire requested")

        with self._available:
            start = time.time()

            # Wait for availability (voice may be using it)
            while self._state.consumer == HailoConsumer.VOICE:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    logger.warning("[HAILO_MGR] Vision acquire timeout - voice still active")
                    return False
                logger.info(f"[HAILO_MGR] Vision waiting for voice to finish ({remaining:.1f}s remaining)")
                self._available.wait(timeout=min(remaining, 1.0))

            # Check if already owned by vision
            if self._state.consumer == HailoConsumer.VISION:
                logger.debug("[HAILO_MGR] Vision already owns Hailo")
                return True

            # CRITICAL: Check if device is actually free at system level
            if not self.is_device_free():
                logger.warning("[HAILO_MGR] Device not free at system level, state says NONE")
                # Try to ensure it's free
                if not self.ensure_device_free(timeout=2.0):
                    logger.error("[HAILO_MGR] Cannot acquire - device locked at system level")
                    self._state.last_error = "Device locked at system level"
                    return False

            # Acquire
            self._state.consumer = HailoConsumer.VISION
            self._state.acquired_at = time.time()
            self._state.acquire_count += 1

            logger.info(f"[HAILO_MGR] Device acquired for VISION (count={self._state.acquire_count})")
            return True

    def acquire_for_voice(self, timeout: float = 10.0) -> bool:
        """Acquire Hailo NPU for voice pipeline.

        Voice has priority - will stop vision if necessary.

        Args:
            timeout: Max time to wait for release (seconds)

        Returns:
            True if acquired, False if error
        """
        logger.info("[HAILO_MGR] Voice acquire requested")

        with self._available:
            # If vision is running, stop it first
            if self._state.consumer == HailoConsumer.VISION:
                logger.info("[HAILO_MGR] Voice requested - stopping vision first")
                self._stop_vision_internal()
                self._state.consumer = HailoConsumer.NONE  # Mark as released
                self._state.switch_count += 1

            # Check if already owned by voice
            if self._state.consumer == HailoConsumer.VOICE:
                logger.debug("[HAILO_MGR] Voice already owns Hailo")
                return True

            # Verify device is actually free (should be after _stop_vision_internal)
            if not self.is_device_free():
                logger.warning("[HAILO_MGR] Device not free after vision stop, waiting...")
                if not self.ensure_device_free(timeout=2.0):
                    logger.error("[HAILO_MGR] Cannot acquire for voice - device still locked")
                    self._state.last_error = "Device locked for voice"
                    return False

            # Mark as NONE temporarily while device is released
            self._state.consumer = HailoConsumer.NONE

            # Acquire
            self._state.consumer = HailoConsumer.VOICE
            self._state.acquired_at = time.time()
            self._state.acquire_count += 1

            logger.info(f"[HAILO_MGR] Device acquired for VOICE (count={self._state.acquire_count}, switches={self._state.switch_count})")
            return True

    def release_vision(self) -> bool:
        """Release Hailo NPU from vision pipeline.

        Returns:
            True if released, False if wasn't held by vision
        """
        with self._available:
            if self._state.consumer != HailoConsumer.VISION:
                logger.warning(f"[HAILO_MGR] Vision release called but consumer is {self._state.consumer.value}")
                return False

            self._stop_vision_internal()

            self._state.consumer = HailoConsumer.NONE
            self._state.released_at = time.time()
            self._state.release_count += 1

            duration = self._state.released_at - self._state.acquired_at
            logger.info(f"[HAILO_MGR] RELEASED from VISION (held {duration:.1f}s, count={self._state.release_count})")

            # Notify waiters
            self._available.notify_all()
            return True

    def release_voice(self, restart_vision: bool = True) -> bool:
        """Release Hailo NPU from voice pipeline.

        Args:
            restart_vision: If True, automatically restart vision pipeline

        Returns:
            True if released, False if wasn't held by voice
        """
        with self._available:
            if self._state.consumer != HailoConsumer.VOICE:
                logger.warning(f"[HAILO_MGR] Voice release called but consumer is {self._state.consumer.value}")
                return False

            self._stop_voice_internal()

            self._state.consumer = HailoConsumer.NONE
            self._state.released_at = time.time()
            self._state.release_count += 1

            duration = self._state.released_at - self._state.acquired_at
            logger.info(f"[HAILO_MGR] RELEASED from VOICE (held {duration:.1f}s, count={self._state.release_count})")

            # Notify waiters
            self._available.notify_all()

        # Restart vision if requested (outside lock to avoid deadlock)
        if restart_vision and self._on_vision_start:
            logger.info("[HAILO_MGR] Auto-restarting vision after voice release")

            # Wait for Hailo device to be fully released
            max_wait = 3.0
            wait_start = time.time()
            while time.time() - wait_start < max_wait:
                if self.is_device_free():
                    logger.info(f"[HAILO_MGR] Device free after {time.time() - wait_start:.1f}s")
                    break
                time.sleep(0.2)
            else:
                logger.warning(f"[HAILO_MGR] Device not free after {max_wait}s, trying anyway...")

            # Try restart with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Acquire for vision
                    if not self.acquire_for_vision(timeout=5.0):
                        logger.warning(f"[HAILO_MGR] Vision acquire failed (attempt {attempt + 1}/{max_retries})")
                        time.sleep(0.5)
                        continue

                    # Start vision pipeline
                    success = self._on_vision_start()
                    if success:
                        logger.info(f"[HAILO_MGR] Vision restarted successfully (attempt {attempt + 1})")
                        break
                    else:
                        logger.warning(f"[HAILO_MGR] Vision start returned False (attempt {attempt + 1}/{max_retries})")
                        self.release_vision()
                        time.sleep(0.5)

                except Exception as e:
                    logger.error(f"[HAILO_MGR] Vision restart error (attempt {attempt + 1}): {e}")
                    try:
                        self.release_vision()
                    except:
                        pass
                    time.sleep(0.5)
            else:
                logger.error(f"[HAILO_MGR] Vision restart FAILED after {max_retries} attempts!")

        return True

    def _stop_vision_internal(self):
        """Stop vision pipeline (internal, called with lock held)."""
        if self._on_vision_stop:
            logger.info("[HAILO_MGR] Stopping vision pipeline...")
            try:
                self._on_vision_stop()

                # Wait for GStreamer to fully release /dev/hailo0
                # GStreamer pipelines need time to reach NULL state and release resources
                max_wait = 2.0
                wait_start = time.time()
                while time.time() - wait_start < max_wait:
                    time.sleep(0.3)
                    if self.is_device_free():
                        logger.info(f"[HAILO_MGR] Vision pipeline stopped, device free after {time.time() - wait_start:.1f}s")
                        return
                    logger.debug(f"[HAILO_MGR] Waiting for device release...")

                # Device still not free after timeout - log warning but continue
                logger.warning(f"[HAILO_MGR] Vision pipeline stopped but device not free after {max_wait}s")

            except Exception as e:
                logger.error(f"[HAILO_MGR] Vision stop error: {e}")
                self._state.last_error = str(e)

    def _stop_voice_internal(self):
        """Stop voice pipeline (internal, called with lock held)."""
        if self._on_voice_stop:
            logger.info("[HAILO_MGR] Stopping voice pipeline...")
            try:
                self._on_voice_stop()
                logger.info("[HAILO_MGR] Voice pipeline stopped")
            except Exception as e:
                logger.error(f"[HAILO_MGR] Voice stop error: {e}")
                self._state.last_error = str(e)

    def force_release(self) -> bool:
        """Force release of Hailo NPU regardless of current state.

        Use only in error recovery scenarios.

        Returns:
            True if anything was released
        """
        logger.warning("[HAILO_MGR] ===== FORCE RELEASE INITIATED =====")

        with self._available:
            prev = self._state.consumer

            if prev == HailoConsumer.VISION:
                self._stop_vision_internal()
            elif prev == HailoConsumer.VOICE:
                self._stop_voice_internal()

            self._state.consumer = HailoConsumer.NONE
            self._state.released_at = time.time()
            self._state.release_count += 1

            logger.warning(f"[HAILO_MGR] FORCE RELEASED from {prev.value}")

            self._available.notify_all()

        # Also ensure device is actually free at system level
        time.sleep(0.3)  # Wait for pipeline to fully stop
        if not self.is_device_free():
            logger.warning("[HAILO_MGR] Device still held after force release, forcing at system level")
            self._force_release_device()

        logger.info("[HAILO_MGR] ===== FORCE RELEASE COMPLETE =====")
        return prev != HailoConsumer.NONE

    def force_reset(self) -> bool:
        """Complete force reset - stop everything and ensure device is free.

        Use when HAILO_OUT_OF_PHYSICAL_DEVICES error occurs.

        Returns:
            True if device is now free
        """
        logger.error("[HAILO_MGR] ===== FORCE RESET EXECUTED =====")

        # First try graceful release
        self.force_release()

        # Wait for any pending operations
        time.sleep(0.5)

        # Force release at device level
        success = self._force_release_device()

        if success:
            logger.info("[HAILO_MGR] Force reset successful - device is free")
        else:
            logger.error("[HAILO_MGR] Force reset FAILED - device may still be locked")

        return success

    def get_status(self) -> dict:
        """Get current status for debugging."""
        with self._lock:
            return {
                "consumer": self._state.consumer.value,
                "acquired_at": self._state.acquired_at,
                "released_at": self._state.released_at,
                "acquire_count": self._state.acquire_count,
                "release_count": self._state.release_count,
                "switch_count": self._state.switch_count,
                "last_error": self._state.last_error,
                "vision_registered": self._vision_pipeline is not None,
                "voice_registered": self._voice_pipeline is not None,
            }


# Singleton
_hailo_manager: Optional[HailoManager] = None
_manager_lock = threading.Lock()


def get_hailo_manager() -> HailoManager:
    """Get or create HailoManager singleton."""
    global _hailo_manager
    if _hailo_manager is None:
        with _manager_lock:
            if _hailo_manager is None:
                _hailo_manager = HailoManager()
    return _hailo_manager


# Convenience functions for common operations
def acquire_hailo_for_vision(timeout: float = 5.0) -> bool:
    """Acquire Hailo for vision pipeline."""
    return get_hailo_manager().acquire_for_vision(timeout)


def acquire_hailo_for_voice(timeout: float = 10.0) -> bool:
    """Acquire Hailo for voice pipeline (stops vision if needed)."""
    return get_hailo_manager().acquire_for_voice(timeout)


def release_hailo_from_vision() -> bool:
    """Release Hailo from vision pipeline."""
    return get_hailo_manager().release_vision()


def release_hailo_from_voice(restart_vision: bool = True) -> bool:
    """Release Hailo from voice pipeline."""
    return get_hailo_manager().release_voice(restart_vision)


def is_hailo_available() -> bool:
    """Check if Hailo NPU is currently available."""
    return get_hailo_manager().is_available


def get_hailo_consumer() -> str:
    """Get current Hailo consumer as string."""
    return get_hailo_manager().current_consumer.value


def force_hailo_reset() -> bool:
    """Force reset Hailo device - use when status=74 occurs."""
    return get_hailo_manager().force_reset()


def is_hailo_device_free() -> bool:
    """Check if Hailo device is free at system level."""
    return get_hailo_manager().is_device_free()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)

    manager = get_hailo_manager()
    print(f"Status: {manager.get_status()}")

    # Simulate vision acquire
    print("\n--- Acquire for vision ---")
    manager.acquire_for_vision()
    print(f"Status: {manager.get_status()}")

    # Simulate voice acquire (should stop vision)
    print("\n--- Acquire for voice (should stop vision) ---")
    manager.acquire_for_voice()
    print(f"Status: {manager.get_status()}")

    # Release voice
    print("\n--- Release voice ---")
    manager.release_voice(restart_vision=False)
    print(f"Status: {manager.get_status()}")

    print("\nDone!")
