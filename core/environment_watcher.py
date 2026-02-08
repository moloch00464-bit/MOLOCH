"""
M.O.L.O.C.H. Environment & Hardware Change Detection Module

Passively monitors the system for new hardware, devices, and resources.
Does NOT automatically activate anything - only detects and logs changes.

Principle: Observe, don't interfere. M.O.L.O.C.H. learns what's available.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
import hashlib


# Setup logging
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "environment.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# State file
STATE_DIR = Path.home() / "moloch" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "environment_state.json"


class DeviceChange:
    """Represents a detected change in the environment."""

    def __init__(self, change_type: str, category: str, details: str, timestamp: Optional[str] = None):
        self.change_type = change_type  # "added", "removed", "modified"
        self.category = category  # "audio_device", "video_device", etc.
        self.details = details
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "change_type": self.change_type,
            "category": self.category,
            "details": self.details
        }

    def __repr__(self):
        return f"DeviceChange({self.change_type}, {self.category}, {self.details})"


class EnvironmentSnapshot:
    """Snapshot of the current system environment."""

    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.dev_devices: Set[str] = set()
        self.video_devices: Set[str] = set()
        self.audio_devices: Set[str] = set()
        self.models: Set[str] = set()
        self.hardware_files: Set[str] = set()
        self.usb_devices: Set[str] = set()

    def capture(self):
        """Capture current system state."""
        # Capture /dev devices (filtered)
        self._capture_dev_devices()

        # Capture video devices
        self._capture_video_devices()

        # Capture audio devices
        self._capture_audio_devices()

        # Capture model files
        self._capture_models()

        # Capture hardware config files
        self._capture_hardware_files()

        # Capture USB devices
        self._capture_usb_devices()

        logger.debug(f"Snapshot captured: {len(self.dev_devices)} /dev, "
                    f"{len(self.video_devices)} video, {len(self.audio_devices)} audio, "
                    f"{len(self.models)} models, {len(self.hardware_files)} hardware files, "
                    f"{len(self.usb_devices)} USB devices")

    def _capture_dev_devices(self):
        """Capture interesting devices from /dev."""
        dev_path = Path("/dev")
        if not dev_path.exists():
            return

        # Only capture interesting device types
        interesting_patterns = [
            "video*", "audio*", "snd/*", "input/*",
            "i2c-*", "spidev*", "gpiochip*",
            "ttyUSB*", "ttyACM*", "ttyS*",
            "hailo*", "dri/*"
        ]

        for pattern in interesting_patterns:
            for device in dev_path.glob(pattern):
                if device.exists():
                    # Store relative path from /dev
                    rel_path = str(device.relative_to(dev_path))
                    self.dev_devices.add(rel_path)

    def _capture_video_devices(self):
        """Capture video4linux devices."""
        v4l_path = Path("/sys/class/video4linux")
        if not v4l_path.exists():
            return

        for device in v4l_path.iterdir():
            if device.is_symlink() or device.is_dir():
                # Get device name
                dev_name = device.name

                # Try to get device info
                name_file = device / "name"
                if name_file.exists():
                    try:
                        device_name = name_file.read_text().strip()
                        self.video_devices.add(f"{dev_name}: {device_name}")
                    except:
                        self.video_devices.add(dev_name)
                else:
                    self.video_devices.add(dev_name)

    def _capture_audio_devices(self):
        """Capture ALSA audio devices."""
        asound_path = Path("/proc/asound")
        if not asound_path.exists():
            return

        # Capture card directories
        for item in asound_path.iterdir():
            if item.is_dir() and item.name.startswith("card"):
                card_id = item / "id"
                if card_id.exists():
                    try:
                        card_name = card_id.read_text().strip()
                        self.audio_devices.add(f"{item.name}: {card_name}")
                    except:
                        self.audio_devices.add(item.name)

    def _capture_models(self):
        """Capture model files from ~/moloch/models."""
        models_path = Path.home() / "moloch" / "models"
        if not models_path.exists():
            return

        # Recursively find model files
        for model_file in models_path.rglob("*"):
            if model_file.is_file():
                # Store relative path from models directory
                rel_path = str(model_file.relative_to(models_path))
                # Only store certain extensions to avoid clutter
                if any(model_file.suffix.endswith(ext) for ext in ['.onnx', '.pt', '.pth', '.tflite', '.pb', '.h5']):
                    self.models.add(rel_path)

    def _capture_hardware_files(self):
        """Capture hardware configuration files."""
        hardware_path = Path.home() / "moloch" / "hardware"
        if not hardware_path.exists():
            return

        for hw_file in hardware_path.rglob("*"):
            if hw_file.is_file():
                rel_path = str(hw_file.relative_to(hardware_path))
                self.hardware_files.add(rel_path)

    def _capture_usb_devices(self):
        """Capture USB device information."""
        usb_path = Path("/sys/bus/usb/devices")
        if not usb_path.exists():
            return

        for device in usb_path.iterdir():
            if device.is_symlink() or device.is_dir():
                # Try to get manufacturer and product
                manufacturer_file = device / "manufacturer"
                product_file = device / "product"

                if manufacturer_file.exists() and product_file.exists():
                    try:
                        manufacturer = manufacturer_file.read_text().strip()
                        product = product_file.read_text().strip()
                        self.usb_devices.add(f"{device.name}: {manufacturer} {product}")
                    except:
                        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "dev_devices": sorted(list(self.dev_devices)),
            "video_devices": sorted(list(self.video_devices)),
            "audio_devices": sorted(list(self.audio_devices)),
            "models": sorted(list(self.models)),
            "hardware_files": sorted(list(self.hardware_files)),
            "usb_devices": sorted(list(self.usb_devices))
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentSnapshot':
        """Create snapshot from dictionary."""
        snapshot = cls()
        snapshot.timestamp = data.get("timestamp", "")
        snapshot.dev_devices = set(data.get("dev_devices", []))
        snapshot.video_devices = set(data.get("video_devices", []))
        snapshot.audio_devices = set(data.get("audio_devices", []))
        snapshot.models = set(data.get("models", []))
        snapshot.hardware_files = set(data.get("hardware_files", []))
        snapshot.usb_devices = set(data.get("usb_devices", []))
        return snapshot

    def compare(self, previous: 'EnvironmentSnapshot') -> List[DeviceChange]:
        """
        Compare this snapshot with a previous one and detect changes.

        Returns:
            List of DeviceChange objects
        """
        changes = []

        # Compare /dev devices
        added_dev = self.dev_devices - previous.dev_devices
        removed_dev = previous.dev_devices - self.dev_devices

        for device in added_dev:
            category = self._classify_device(device)
            changes.append(DeviceChange("added", category, f"/dev/{device}"))

        for device in removed_dev:
            category = self._classify_device(device)
            changes.append(DeviceChange("removed", category, f"/dev/{device}"))

        # Compare video devices
        added_video = self.video_devices - previous.video_devices
        removed_video = previous.video_devices - self.video_devices

        for device in added_video:
            changes.append(DeviceChange("added", "video_device", device))

        for device in removed_video:
            changes.append(DeviceChange("removed", "video_device", device))

        # Compare audio devices
        added_audio = self.audio_devices - previous.audio_devices
        removed_audio = previous.audio_devices - self.audio_devices

        for device in added_audio:
            changes.append(DeviceChange("added", "audio_device", device))

        for device in removed_audio:
            changes.append(DeviceChange("removed", "audio_device", device))

        # Compare models
        added_models = self.models - previous.models
        removed_models = previous.models - self.models

        for model in added_models:
            changes.append(DeviceChange("added", "model", f"models/{model}"))

        for model in removed_models:
            changes.append(DeviceChange("removed", "model", f"models/{model}"))

        # Compare hardware files
        added_hw = self.hardware_files - previous.hardware_files
        removed_hw = previous.hardware_files - self.hardware_files

        for hw_file in added_hw:
            changes.append(DeviceChange("added", "hardware_config", f"hardware/{hw_file}"))

        for hw_file in removed_hw:
            changes.append(DeviceChange("removed", "hardware_config", f"hardware/{hw_file}"))

        # Compare USB devices
        added_usb = self.usb_devices - previous.usb_devices
        removed_usb = previous.usb_devices - self.usb_devices

        for usb in added_usb:
            changes.append(DeviceChange("added", "usb_device", usb))

        for usb in removed_usb:
            changes.append(DeviceChange("removed", "usb_device", usb))

        return changes

    def _classify_device(self, device: str) -> str:
        """Classify a device based on its name."""
        device_lower = device.lower()

        if "video" in device_lower:
            return "video_device"
        elif "audio" in device_lower or "snd" in device_lower:
            return "audio_device"
        elif "i2c" in device_lower or "spi" in device_lower or "gpio" in device_lower:
            return "sensor"
        elif "hailo" in device_lower:
            return "ai_accelerator"
        elif "tty" in device_lower or "usb" in device_lower or "acm" in device_lower:
            return "serial_device"
        elif "dri" in device_lower:
            return "gpu_device"
        else:
            return "unknown_device"


class EnvironmentWatcher:
    """
    Watches the system environment for hardware and resource changes.
    Passive observer - does NOT modify system state.
    """

    def __init__(self):
        self.state_file = STATE_FILE
        self.last_snapshot: Optional[EnvironmentSnapshot] = None

        # Load previous state if exists
        self._load_state()

        logger.info("Environment Watcher initialized")

    def _load_state(self):
        """Load the last known state from disk."""
        if not self.state_file.exists():
            logger.info("No previous state found - this is the first run")
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            self.last_snapshot = EnvironmentSnapshot.from_dict(data)
            logger.info(f"Loaded previous state from {self.last_snapshot.timestamp}")

        except Exception as e:
            logger.error(f"Failed to load previous state: {e}")

    def _save_state(self, snapshot: EnvironmentSnapshot):
        """Save current state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)

            logger.debug(f"State saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def check(self) -> List[DeviceChange]:
        """
        Check for environment changes.

        Returns:
            List of detected changes (empty if no changes)
        """
        logger.debug("Checking environment for changes...")

        # Capture current state
        current_snapshot = EnvironmentSnapshot()
        current_snapshot.capture()

        changes = []

        # If we have a previous snapshot, compare
        if self.last_snapshot:
            changes = current_snapshot.compare(self.last_snapshot)

            if changes:
                logger.info(f"Detected {len(changes)} change(s)")
                for change in changes:
                    logger.info(f"  [{change.change_type.upper()}] {change.category}: {change.details}")
            else:
                logger.debug("No changes detected")
        else:
            logger.info("First check - establishing baseline")

        # Save current snapshot as last known state
        self.last_snapshot = current_snapshot
        self._save_state(current_snapshot)

        return changes

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current environment state as a dictionary."""
        if self.last_snapshot:
            return self.last_snapshot.to_dict()
        return {}

    def force_baseline(self):
        """Force a new baseline snapshot (ignoring any previous state)."""
        logger.info("Forcing new baseline snapshot")
        self.last_snapshot = None
        return self.check()


# Global watcher instance
_watcher: Optional[EnvironmentWatcher] = None


def get_watcher() -> EnvironmentWatcher:
    """Get or create the global environment watcher instance."""
    global _watcher
    if _watcher is None:
        _watcher = EnvironmentWatcher()
    return _watcher


def check_environment() -> List[DeviceChange]:
    """
    Convenience function to check for environment changes.

    Returns:
        List of detected changes
    """
    watcher = get_watcher()
    return watcher.check()


def get_current_state() -> Dict[str, Any]:
    """Get current environment state."""
    watcher = get_watcher()
    return watcher.get_current_state()


if __name__ == "__main__":
    # Quick test
    print("M.O.L.O.C.H. Environment Watcher")
    print("=" * 70)

    watcher = get_watcher()

    print("\nPerforming environment check...")
    changes = watcher.check()

    if changes:
        print(f"\nDetected {len(changes)} change(s):")
        for change in changes:
            print(f"  [{change.change_type.upper()}] {change.category}: {change.details}")
    else:
        print("\nNo changes detected (or establishing baseline)")

    print("\nCurrent environment state:")
    state = watcher.get_current_state()
    for key, values in state.items():
        if key != "timestamp" and values:
            print(f"  {key}: {len(values)} items")
