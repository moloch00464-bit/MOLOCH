#!/usr/bin/env python3
"""
M.O.L.O.C.H. Thermal Manager
============================

Layer 1 (Raw Facts) - Hardware homeostasis control.
Machine manages its own temperature and reports state changes as facts.

Architecture:
- Modular design for integration with status.py
- Read-only monitoring (fan controlled by kernel thermal_zone)
- Optional manual fan control (requires udev permissions)

Features:
- CPU temperature monitoring with moving average smoothing
- 5°C hysteresis to prevent fan oscillation
- Event emission for thermal state changes
- TTS integration for important thermal events

Hardware Endpoints:
- CPU temp: /sys/class/thermal/thermal_zone0/temp (read-only)
- Fan level: /sys/class/thermal/cooling_device0/cur_state (read/write)

Permission Note:
- Reading is always possible
- Writing to fan requires either root or udev rules
"""

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

# Setup logging
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "thermal.log"

# Configure logger for this module only
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler if not already present
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(fh)


# =============================================================================
# UDEV RULES FOR FAN CONTROL
# =============================================================================

UDEV_RULE_CONTENT = """\
# M.O.L.O.C.H. Thermal Manager - Fan Control Permissions
# File: /etc/udev/rules.d/99-moloch-thermal.rules
#
# Allows non-root users in 'moloch' group to control the fan
#
SUBSYSTEM=="thermal", KERNEL=="cooling_device*", ACTION=="add", \\
    RUN+="/bin/chmod 0664 /sys/class/thermal/%k/cur_state", \\
    RUN+="/bin/chgrp moloch /sys/class/thermal/%k/cur_state"

# Also allow thermal zone trip point control
SUBSYSTEM=="thermal", KERNEL=="thermal_zone*", ACTION=="add", \\
    RUN+="/bin/chmod 0664 /sys/class/thermal/%k/trip_point_*_temp", \\
    RUN+="/bin/chgrp moloch /sys/class/thermal/%k/trip_point_*_temp"
"""

UDEV_SETUP_INSTRUCTIONS = """
================================================================================
FEHLENDE SCHREIBRECHTE FÜR LÜFTERSTEUERUNG
================================================================================

Um die manuelle Lüftersteuerung zu aktivieren, führe folgende Befehle aus:

1. Erstelle die Gruppe 'moloch' (falls nicht vorhanden):
   sudo groupadd moloch

2. Füge deinen Benutzer zur Gruppe hinzu:
   sudo usermod -aG moloch $USER

3. Erstelle die udev-Regel:
   sudo tee /etc/udev/rules.d/99-moloch-thermal.rules << 'EOF'
# M.O.L.O.C.H. Thermal Manager - Fan Control Permissions
SUBSYSTEM=="thermal", KERNEL=="cooling_device*", ACTION=="add", \\
    RUN+="/bin/chmod 0664 /sys/class/thermal/%k/cur_state", \\
    RUN+="/bin/chgrp moloch /sys/class/thermal/%k/cur_state"
EOF

4. Lade die udev-Regeln neu:
   sudo udevadm control --reload-rules
   sudo udevadm trigger

5. Melde dich ab und wieder an (oder starte neu).

HINWEIS: Die Temperaturüberwachung funktioniert auch ohne diese Rechte.
         Nur die manuelle Lüftersteuerung erfordert Schreibzugriff.
================================================================================
"""


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================

class ThermalState(Enum):
    """Thermal status levels."""
    STABLE = "stable"           # Within target range
    WARMING = "warming"         # Rising toward warning
    WARNING = "warning"         # Soft limit exceeded (65°C)
    THROTTLING = "throttling"   # Throttling risk (75°C)
    CRITICAL = "critical"       # Critical - maximum cooling (85°C)


@dataclass
class ThermalEvent:
    """Thermal event data for callbacks and logging."""
    timestamp: float
    event_type: str
    temp_c: float
    fan_level: int
    state: ThermalState
    previous_state: Optional[ThermalState] = None
    message: Optional[str] = None


@dataclass
class ThermalTelemetry:
    """
    Thermal telemetry data - modular format for status.py integration.

    This dataclass can be directly integrated into status.py's telemetry system.
    """
    temp_c: float               # Smoothed CPU temperature
    temp_raw_c: float           # Raw (unsmoothed) temperature
    fan_level: int              # Current fan PWM level (0-4)
    fan_max: int                # Maximum fan level
    fan_percent: int            # Fan speed as percentage
    state: str                  # ThermalState value
    status: str                 # ok, warning, critical (for dashboard)
    can_write_fan: bool         # Whether we have fan control permissions


# =============================================================================
# CONFIGURATION
# =============================================================================

class ThermalConfig:
    """Thermal management configuration - all thresholds in one place."""

    # === MONITORING ===
    MONITORING_INTERVAL = 5.0       # Seconds between checks
    SMOOTHING_WINDOW = 5            # Moving average window size

    # === HYSTERESIS ===
    # Fan springt bei X°C an, geht erst bei (X - 5)°C wieder runter
    HYSTERESIS_OFFSET = 5.0         # Degrees Celsius

    # === TARGET RANGE ===
    TARGET_IDLE_MIN = 48.0          # Optimal minimum
    TARGET_IDLE_MAX = 55.0          # Optimal maximum

    # === THERMAL LIMITS ===
    SOFT_LIMIT = 65.0               # Warning threshold
    THROTTLING_RISK = 75.0          # Performance may degrade
    CRITICAL_SHUTDOWN = 85.0        # Critical threshold

    # === FAN TRIP POINTS (updated 2026-02-04) ===
    # Temperature at which each fan level activates
    # More conservative curve - less noise, still safe
    FAN_TRIP_POINTS = {
        1: 50.0,    # Level 1: Low - starts at 50°C
        2: 55.0,    # Level 2: Medium
        3: 65.0,    # Level 3: High
        4: 75.0,    # Level 4: Maximum
    }

    # === HARDWARE PATHS ===
    CPU_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
    FAN_STATE_PATH = Path("/sys/class/thermal/cooling_device0/cur_state")
    FAN_MAX_STATE_PATH = Path("/sys/class/thermal/cooling_device0/max_state")

    # === TTS MESSAGES (German) ===
    TTS_MESSAGES = {
        "fan_level_3": "Ich erhöhe die Kühlung, Markus. Mir wird gerade etwas warm.",
        "thermal_stable_after_heat": "Temperatur wieder im Normalbereich.",
        "thermal_warning": "Achtung, ich werde warm. {} Grad.",
        "thermal_critical": "Warnung! Kritische Temperatur erreicht. {} Grad."
    }


# =============================================================================
# PERMISSION CHECKER
# =============================================================================

class PermissionChecker:
    """Check and report hardware access permissions."""

    @staticmethod
    def can_read_temp() -> bool:
        """Check if we can read CPU temperature."""
        path = ThermalConfig.CPU_TEMP_PATH
        return path.exists() and os.access(path, os.R_OK)

    @staticmethod
    def can_read_fan() -> bool:
        """Check if we can read fan state."""
        path = ThermalConfig.FAN_STATE_PATH
        return path.exists() and os.access(path, os.R_OK)

    @staticmethod
    def can_write_fan() -> bool:
        """Check if we can write to fan control."""
        path = ThermalConfig.FAN_STATE_PATH
        return path.exists() and os.access(path, os.W_OK)

    @staticmethod
    def check_all() -> Dict[str, bool]:
        """Check all permissions and return status dict."""
        return {
            "read_temp": PermissionChecker.can_read_temp(),
            "read_fan": PermissionChecker.can_read_fan(),
            "write_fan": PermissionChecker.can_write_fan(),
        }

    @staticmethod
    def print_status():
        """Print permission status to console."""
        perms = PermissionChecker.check_all()

        print("\n=== M.O.L.O.C.H. Thermal Permissions ===")
        print(f"  CPU Temperatur lesen:  {'✓' if perms['read_temp'] else '✗'}")
        print(f"  Lüfter-Status lesen:   {'✓' if perms['read_fan'] else '✗'}")
        print(f"  Lüfter steuern:        {'✓' if perms['write_fan'] else '✗'}")

        if not perms['write_fan']:
            print(UDEV_SETUP_INSTRUCTIONS)

        return perms


# =============================================================================
# THERMAL MANAGER
# =============================================================================

class ThermalManager:
    """
    M.O.L.O.C.H. Thermal Manager.

    Singleton pattern for hardware access coordination.
    Reports state changes as facts - machine homeostasis.

    Hysteresis Logic (5°C):
    - Fan level increases when temp crosses trip point upward
    - Fan level only decreases when temp drops 5°C BELOW that trip point
    - Example: Level 2 activates at 40°C, deactivates at 35°C
    """

    _instance = None
    _lock = threading.Lock()

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
        self.state = ThermalState.STABLE
        self._state_lock = threading.Lock()

        # Temperature smoothing (moving average)
        self._temp_buffer: deque = deque(maxlen=ThermalConfig.SMOOTHING_WINDOW)
        self._current_temp = 0.0
        self._smoothed_temp = 0.0

        # Fan state
        self._fan_level = 0
        self._last_fan_level = 0
        self._fan_max_state = 4

        # Hysteresis tracking: remember the "committed" fan level
        # This prevents oscillation: fan won't drop until temp is well below trip point
        self._hysteresis_fan_level = 0

        # Permissions
        self._can_write_fan = False

        # Control
        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread = None

        # Callbacks
        self._on_thermal_event: Optional[Callable[[ThermalEvent], None]] = None
        self._on_state_change: Optional[Callable[[ThermalState, ThermalState], None]] = None

        # Components (lazy loaded)
        self._timeline = None
        self._tts = None
        self._tts_enabled = True

        # Recovery tracking
        self._was_above_warning = False

        logger.info("ThermalManager initialized")

    # =========================================================================
    # LAZY LOADERS
    # =========================================================================

    def _get_timeline(self):
        """Get Timeline instance (lazy load)."""
        if self._timeline is None:
            try:
                from core.timeline import get_timeline
                self._timeline = get_timeline()
            except ImportError:
                logger.debug("Timeline not available")
        return self._timeline

    def _get_tts(self):
        """Get TTS instance (lazy load)."""
        if self._tts is None:
            try:
                from core.tts import speak
                self._tts = speak
            except ImportError:
                logger.debug("TTS not available")
        return self._tts

    # =========================================================================
    # HARDWARE READ/WRITE
    # =========================================================================

    def _read_temp(self) -> float:
        """Read CPU temperature (millidegrees -> degrees)."""
        try:
            content = ThermalConfig.CPU_TEMP_PATH.read_text().strip()
            return int(content) / 1000.0
        except Exception as e:
            logger.error(f"Failed to read temperature: {e}")
            return 0.0

    def _read_fan_state(self) -> int:
        """Read current fan PWM level."""
        try:
            content = ThermalConfig.FAN_STATE_PATH.read_text().strip()
            return int(content)
        except Exception as e:
            logger.error(f"Failed to read fan state: {e}")
            return 0

    def _read_fan_max_state(self) -> int:
        """Read maximum fan level."""
        try:
            content = ThermalConfig.FAN_MAX_STATE_PATH.read_text().strip()
            return int(content)
        except Exception:
            return 4

    def _write_fan_state(self, level: int) -> bool:
        """
        Write fan level (requires permissions).

        Returns True if successful, False otherwise.
        """
        if not self._can_write_fan:
            return False

        try:
            level = max(0, min(level, self._fan_max_state))
            ThermalConfig.FAN_STATE_PATH.write_text(str(level))
            logger.info(f"Set fan level to {level}")
            return True
        except PermissionError:
            logger.warning("No permission to write fan state")
            self._can_write_fan = False
            return False
        except Exception as e:
            logger.error(f"Failed to write fan state: {e}")
            return False

    # =========================================================================
    # TEMPERATURE SMOOTHING
    # =========================================================================

    def _update_smoothed_temp(self, raw_temp: float) -> float:
        """Update and return smoothed temperature using moving average."""
        self._temp_buffer.append(raw_temp)
        self._smoothed_temp = sum(self._temp_buffer) / len(self._temp_buffer)
        return self._smoothed_temp

    # =========================================================================
    # HYSTERESIS LOGIC
    # =========================================================================

    def _calculate_target_fan_level(self, temp: float) -> int:
        """
        Calculate target fan level with 5°C hysteresis.

        Hysteresis prevents oscillation:
        - Level UP: when temp >= trip_point
        - Level DOWN: only when temp < (trip_point - 5°C)

        Example with current level 2 (trip at 40°C):
        - Temp rises to 45°C → Level 3 (trip at 45°C crossed)
        - Temp drops to 42°C → Stay at Level 3 (42 > 40, not below 45-5=40)
        - Temp drops to 38°C → Stay at Level 3 (38 > 35, still not below 40-5=35)
        - Temp drops to 34°C → Level 2 (34 < 40-5=35, but 34 > 35-5=30, so level 2)
        """
        trip_points = ThermalConfig.FAN_TRIP_POINTS
        hysteresis = ThermalConfig.HYSTERESIS_OFFSET
        current_level = self._hysteresis_fan_level

        # Check if we should increase (no hysteresis for going up)
        new_level = 0
        for level in sorted(trip_points.keys()):
            if temp >= trip_points[level]:
                new_level = level

        # If new level is higher, immediately adopt it
        if new_level > current_level:
            return new_level

        # If new level is lower, apply hysteresis
        # Only decrease if temp is below (trip_point - hysteresis)
        if new_level < current_level:
            # Check if we should keep current level due to hysteresis
            current_trip = trip_points.get(current_level, 0)
            if temp >= (current_trip - hysteresis):
                # Stay at current level - not cool enough yet
                return current_level

            # We're below hysteresis threshold, check each level down
            for level in range(current_level - 1, -1, -1):
                if level == 0:
                    # Level 0 has no trip point, check if we're below level 1's threshold
                    if temp < (trip_points.get(1, 35) - hysteresis):
                        return 0
                    return 1
                else:
                    level_trip = trip_points.get(level, 0)
                    if temp >= (level_trip - hysteresis):
                        return level

        return current_level

    # =========================================================================
    # STATE DETERMINATION
    # =========================================================================

    def _determine_thermal_state(self, temp: float) -> ThermalState:
        """
        Determine thermal state with hysteresis.

        States are determined by temperature thresholds with hysteresis
        to prevent rapid state changes.
        """
        current = self.state
        hysteresis = ThermalConfig.HYSTERESIS_OFFSET

        # Critical - immediate response, no hysteresis
        if temp >= ThermalConfig.CRITICAL_SHUTDOWN:
            return ThermalState.CRITICAL

        # Throttling risk
        if temp >= ThermalConfig.THROTTLING_RISK:
            return ThermalState.THROTTLING
        elif current == ThermalState.THROTTLING:
            if temp >= (ThermalConfig.THROTTLING_RISK - hysteresis):
                return ThermalState.THROTTLING

        # Warning
        if temp >= ThermalConfig.SOFT_LIMIT:
            return ThermalState.WARNING
        elif current == ThermalState.WARNING:
            if temp >= (ThermalConfig.SOFT_LIMIT - hysteresis):
                return ThermalState.WARNING

        # Warming (above target range but below warning)
        if temp > ThermalConfig.TARGET_IDLE_MAX:
            return ThermalState.WARMING

        # Stable
        return ThermalState.STABLE

    def _state_to_status(self, state: ThermalState) -> str:
        """Convert ThermalState to dashboard status string."""
        mapping = {
            ThermalState.STABLE: "ok",
            ThermalState.WARMING: "ok",
            ThermalState.WARNING: "warning",
            ThermalState.THROTTLING: "warning",
            ThermalState.CRITICAL: "critical",
        }
        return mapping.get(state, "unknown")

    # =========================================================================
    # EVENT EMISSION
    # =========================================================================

    def _emit_event(self, event_type: str, message: str = None):
        """Emit a thermal event to timeline and callbacks."""
        event = ThermalEvent(
            timestamp=time.time(),
            event_type=event_type,
            temp_c=self._smoothed_temp,
            fan_level=self._fan_level,
            state=self.state,
            message=message
        )

        # Log to timeline
        timeline = self._get_timeline()
        if timeline:
            timeline.log(
                "thermal",
                event_type,
                temp_c=round(self._smoothed_temp, 1),
                fan_level=self._fan_level,
                state=self.state.value
            )

        # Call callback
        if self._on_thermal_event:
            try:
                self._on_thermal_event(event)
            except Exception as e:
                logger.error(f"Thermal event callback error: {e}")

        logger.info(f"THERMAL: {event_type} | {self._smoothed_temp:.1f}°C | Fan {self._fan_level}/{self._fan_max_state} | {self.state.value}")

    def _speak(self, message: str):
        """Speak message via TTS (if enabled)."""
        if not self._tts_enabled:
            return

        tts = self._get_tts()
        if tts:
            try:
                tts(message)
            except Exception as e:
                logger.error(f"TTS error: {e}")

    # =========================================================================
    # THERMAL CHECK CYCLE
    # =========================================================================

    def _check_thermal(self):
        """Perform one thermal check cycle."""
        # Read current values
        raw_temp = self._read_temp()
        self._current_temp = raw_temp
        self._update_smoothed_temp(raw_temp)

        self._fan_level = self._read_fan_state()

        # Calculate target fan level with hysteresis
        target_fan = self._calculate_target_fan_level(self._smoothed_temp)
        if target_fan != self._hysteresis_fan_level:
            old_hysteresis = self._hysteresis_fan_level
            self._hysteresis_fan_level = target_fan
            logger.debug(f"Hysteresis fan level: {old_hysteresis} -> {target_fan} at {self._smoothed_temp:.1f}°C")

        # Determine thermal state
        new_state = self._determine_thermal_state(self._smoothed_temp)
        old_state = self.state

        # Handle state transitions
        if new_state != old_state:
            self._handle_state_transition(old_state, new_state)

        # Check fan level changes (from kernel, not our hysteresis)
        if self._fan_level != self._last_fan_level:
            self._handle_fan_change(self._last_fan_level, self._fan_level)
            self._last_fan_level = self._fan_level

    def _handle_state_transition(self, old_state: ThermalState, new_state: ThermalState):
        """Handle thermal state transition."""
        with self._state_lock:
            self.state = new_state

        logger.info(f"Thermal state: {old_state.value} -> {new_state.value} at {self._smoothed_temp:.1f}°C")

        # Call state change callback
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        # Emit events and TTS based on transition
        if new_state == ThermalState.STABLE:
            if self._was_above_warning:
                self._emit_event("thermal_stable")
                self._speak(ThermalConfig.TTS_MESSAGES["thermal_stable_after_heat"])
                self._was_above_warning = False
            else:
                self._emit_event("thermal_stable")

        elif new_state == ThermalState.WARMING:
            self._emit_event("thermal_warming")

        elif new_state == ThermalState.WARNING:
            self._was_above_warning = True
            self._emit_event("thermal_warning")
            self._speak(ThermalConfig.TTS_MESSAGES["thermal_warning"].format(int(self._smoothed_temp)))

        elif new_state == ThermalState.THROTTLING:
            self._was_above_warning = True
            self._emit_event("thermal_throttling")

        elif new_state == ThermalState.CRITICAL:
            self._was_above_warning = True
            self._emit_event("critical_cooling_active")
            self._speak(ThermalConfig.TTS_MESSAGES["thermal_critical"].format(int(self._smoothed_temp)))

    def _handle_fan_change(self, old_level: int, new_level: int):
        """Handle fan level change (reported by kernel)."""
        self._emit_event("fan_level_changed")
        logger.info(f"Fan level: {old_level} -> {new_level}")

        # TTS for significant fan changes
        if new_level >= 3 and old_level < 3:
            self._speak(ThermalConfig.TTS_MESSAGES["fan_level_3"])

    # =========================================================================
    # MONITORING LOOP
    # =========================================================================

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("Thermal monitoring started")

        # Initial readings
        self._fan_max_state = self._read_fan_max_state()
        self._check_thermal()
        self._last_fan_level = self._fan_level
        self._hysteresis_fan_level = self._fan_level

        while self._running:
            try:
                self._check_thermal()
            except Exception as e:
                logger.error(f"Thermal check error: {e}")

            self._stop_event.wait(ThermalConfig.MONITORING_INTERVAL)

        logger.info("Thermal monitoring stopped")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def start(self,
              on_thermal_event: Callable[[ThermalEvent], None] = None,
              on_state_change: Callable[[ThermalState, ThermalState], None] = None,
              enable_tts: bool = True) -> bool:
        """
        Start thermal monitoring.

        Args:
            on_thermal_event: Callback for thermal events
            on_state_change: Callback for state changes
            enable_tts: Enable TTS announcements

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("ThermalManager already running")
            return True

        # Set callbacks
        self._on_thermal_event = on_thermal_event
        self._on_state_change = on_state_change
        self._tts_enabled = enable_tts

        # Check permissions
        if not PermissionChecker.can_read_temp():
            logger.error(f"Cannot read CPU temperature: {ThermalConfig.CPU_TEMP_PATH}")
            return False

        self._can_write_fan = PermissionChecker.can_write_fan()
        if not self._can_write_fan:
            logger.info("Fan control: Read-only mode (kernel controls fan)")

        self._running = True
        self._stop_event.clear()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ThermalMonitor",
            daemon=True
        )
        self._monitor_thread.start()

        # Log startup
        timeline = self._get_timeline()
        if timeline:
            timeline.system_start("thermal_manager")

        logger.info("ThermalManager started")
        return True

    def stop(self):
        """Stop thermal monitoring."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        timeline = self._get_timeline()
        if timeline:
            timeline.system_stop("thermal_manager")

        logger.info("ThermalManager stopped")

    def get_telemetry(self) -> ThermalTelemetry:
        """
        Get thermal telemetry in modular format.

        This method provides data compatible with status.py integration.
        """
        return ThermalTelemetry(
            temp_c=round(self._smoothed_temp, 1),
            temp_raw_c=round(self._current_temp, 1),
            fan_level=self._fan_level,
            fan_max=self._fan_max_state,
            fan_percent=int((self._fan_level / self._fan_max_state) * 100) if self._fan_max_state > 0 else 0,
            state=self.state.value,
            status=self._state_to_status(self.state),
            can_write_fan=self._can_write_fan,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current thermal status as dictionary."""
        telemetry = self.get_telemetry()
        return {
            "running": self._running,
            **asdict(telemetry),
            "target_range": f"{ThermalConfig.TARGET_IDLE_MIN}-{ThermalConfig.TARGET_IDLE_MAX}°C",
            "limits": {
                "soft": ThermalConfig.SOFT_LIMIT,
                "throttling": ThermalConfig.THROTTLING_RISK,
                "critical": ThermalConfig.CRITICAL_SHUTDOWN
            },
            "hysteresis": ThermalConfig.HYSTERESIS_OFFSET,
        }

    @property
    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._running

    @property
    def current_temp(self) -> float:
        """Get current smoothed temperature."""
        return self._smoothed_temp

    @property
    def current_fan_level(self) -> int:
        """Get current fan level."""
        return self._fan_level


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_thermal_manager: Optional[ThermalManager] = None
_tm_lock = threading.Lock()


def get_thermal_manager() -> ThermalManager:
    """Get or create ThermalManager singleton instance."""
    global _thermal_manager
    if _thermal_manager is None:
        with _tm_lock:
            if _thermal_manager is None:
                _thermal_manager = ThermalManager()
    return _thermal_manager


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    print("M.O.L.O.C.H. Thermal Manager")
    print("=" * 60)

    # Check permissions first
    perms = PermissionChecker.print_status()

    if not perms['read_temp']:
        print("\nERROR: Kann CPU-Temperatur nicht lesen!")
        sys.exit(1)

    manager = get_thermal_manager()

    def on_event(event: ThermalEvent):
        print(f"  EVENT: {event.event_type}")

    def on_state_change(old: ThermalState, new: ThermalState):
        print(f"  STATE: {old.value} -> {new.value}")

    if not manager.start(
        on_thermal_event=on_event,
        on_state_change=on_state_change,
        enable_tts=False
    ):
        print("ERROR: Failed to start thermal manager")
        sys.exit(1)

    print("\nInitial Status:")
    status = manager.get_status()
    for key, value in status.items():
        if key != 'limits':
            print(f"  {key}: {value}")

    print(f"\nHysterese: {ThermalConfig.HYSTERESIS_OFFSET}°C")
    print("  Fan springt an bei: 35°C (L1), 40°C (L2), 45°C (L3), 50°C (L4)")
    print("  Fan geht aus bei:   30°C (L0), 35°C (L1), 40°C (L2), 45°C (L3)")

    print(f"\nMonitoring alle {ThermalConfig.MONITORING_INTERVAL}s (Ctrl+C zum Beenden)...")

    try:
        while True:
            time.sleep(5)
            tel = manager.get_telemetry()
            print(f"  {tel.temp_c:.1f}°C | Fan {tel.fan_level}/{tel.fan_max} | {tel.state}")
    except KeyboardInterrupt:
        print("\nStopping...")

    manager.stop()
    print("Done.")
