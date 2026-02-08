#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hardware Telemetry Engine

Collects and bundles raw data from:
- Hailo-10H NPU (Temperature, Power, TOPS utilization) - 40 TOPS
- Samsung 980 NVMe (Disk I/O, Temperature, Usage)
- Raspberry Pi 5 CPU (Temperature, Usage, Frequency)
- Audio channels status

Provides safety checks against hardware limits.
M.O.L.O.C.H. must know its physical boundaries.
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "telemetry.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config file
CONFIG_DIR = Path.home() / "moloch" / "config"
CONFIG_FILE = CONFIG_DIR / "moloch_context.json"


class Status(Enum):
    """Hardware status levels."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"
    OFFLINE = "offline"


@dataclass
class CPUTelemetry:
    """CPU telemetry data."""
    temp_c: float
    usage_percent: float
    freq_mhz: int
    cores: int
    status: str
    per_core_usage: list


@dataclass
class NPUTelemetry:
    """Hailo NPU telemetry data."""
    temp_c: float
    power_w: float
    utilization_percent: float
    tops_available: float
    device: str
    status: str
    online: bool


@dataclass
class StorageTelemetry:
    """NVMe storage telemetry data."""
    temp_c: float
    used_gb: float
    total_gb: float
    used_percent: float
    read_mbps: float
    write_mbps: float
    device: str
    mount: str
    status: str
    name: str = "Storage"


@dataclass
class MultiStorageTelemetry:
    """Multiple storage devices telemetry."""
    devices: list  # List of StorageTelemetry
    status: str


@dataclass
class AudioChannel:
    """Audio channel status."""
    id: int
    name: str
    device: str
    status: str
    volume_percent: int


@dataclass
class AudioTelemetry:
    """Audio system telemetry."""
    channels: list
    active_count: int
    status: str


@dataclass
class RAMTelemetry:
    """RAM telemetry data."""
    used_gb: float
    total_gb: float
    used_percent: float
    available_gb: float
    status: str


@dataclass
class FanTelemetry:
    """Fan telemetry data."""
    state: int           # Current PWM state (0-4)
    max_state: int       # Maximum PWM state
    speed_percent: int   # Speed as percentage
    temp_c: float        # CPU temperature
    status: str          # ok, warning, critical


@dataclass
class SystemTelemetry:
    """Complete system telemetry bundle."""
    timestamp: str
    cpu: CPUTelemetry
    npu: NPUTelemetry
    storage: MultiStorageTelemetry
    ram: RAMTelemetry
    audio: AudioTelemetry
    overall_status: str


class HardwareLimits:
    """Hardware limits from config."""

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config = {}
        self.limits = {}
        self._load_config(config_path)

    def _load_config(self, config_path: Path):
        """Load configuration from JSON."""
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            self._set_defaults()
            return

        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.limits = self.config.get("limits", {})
            logger.debug("Hardware limits loaded from config")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._set_defaults()

    def _set_defaults(self):
        """Set default safe limits."""
        self.limits = {
            "cpu": {"temp_warning_c": 70, "temp_critical_c": 80, "temp_shutdown_c": 85},
            "npu": {"temp_warning_c": 75, "temp_critical_c": 85, "temp_shutdown_c": 95},
            "storage": {"temp_warning_c": 60, "temp_critical_c": 70},
            "ram": {"usage_warning_percent": 80, "usage_critical_percent": 90}
        }

    def check_cpu_temp(self, temp: float) -> Status:
        """Check CPU temperature against limits."""
        limits = self.limits.get("cpu", {})
        if temp >= limits.get("temp_shutdown_c", 85):
            return Status.SHUTDOWN
        elif temp >= limits.get("temp_critical_c", 80):
            return Status.CRITICAL
        elif temp >= limits.get("temp_warning_c", 70):
            return Status.WARNING
        return Status.OK

    def check_npu_temp(self, temp: float) -> Status:
        """Check NPU temperature against limits."""
        limits = self.limits.get("npu", {})
        if temp >= limits.get("temp_shutdown_c", 95):
            return Status.SHUTDOWN
        elif temp >= limits.get("temp_critical_c", 85):
            return Status.CRITICAL
        elif temp >= limits.get("temp_warning_c", 75):
            return Status.WARNING
        return Status.OK

    def check_storage_temp(self, temp: float) -> Status:
        """Check storage temperature against limits."""
        limits = self.limits.get("storage", {})
        if temp >= limits.get("temp_critical_c", 70):
            return Status.CRITICAL
        elif temp >= limits.get("temp_warning_c", 60):
            return Status.WARNING
        return Status.OK

    def check_ram_usage(self, percent: float) -> Status:
        """Check RAM usage against limits."""
        limits = self.limits.get("ram", {})
        if percent >= limits.get("usage_critical_percent", 90):
            return Status.CRITICAL
        elif percent >= limits.get("usage_warning_percent", 80):
            return Status.WARNING
        return Status.OK


class TelemetryEngine:
    """
    M.O.L.O.C.H. Hardware Telemetry Engine

    Collects real-time hardware data and checks safety limits.
    """

    def __init__(self):
        self.limits = HardwareLimits()
        self._last_disk_stats = None
        self._last_disk_time = None
        logger.info("Telemetry Engine initialized")

    def get_cpu_telemetry(self) -> CPUTelemetry:
        """Collect CPU telemetry."""
        try:
            import psutil

            # Temperature
            temp = 0.0
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            temp = entry.current
                            break
                    if temp > 0:
                        break

            # If no temp from psutil, try thermal zone
            if temp == 0:
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        temp = float(f.read().strip()) / 1000.0
                except:
                    pass

            # Usage
            usage = psutil.cpu_percent(interval=0.1)
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            # Frequency
            freq = psutil.cpu_freq()
            freq_mhz = int(freq.current) if freq else 0

            # Core count
            cores = psutil.cpu_count(logical=True)

            # Check status
            status = self.limits.check_cpu_temp(temp)

            return CPUTelemetry(
                temp_c=round(temp, 1),
                usage_percent=round(usage, 1),
                freq_mhz=freq_mhz,
                cores=cores,
                status=status.value,
                per_core_usage=[round(c, 1) for c in per_core]
            )

        except Exception as e:
            logger.error(f"CPU telemetry error: {e}")
            return CPUTelemetry(0, 0, 0, 0, Status.UNKNOWN.value, [])

    def get_npu_telemetry(self) -> NPUTelemetry:
        """Collect Hailo NPU telemetry."""
        device = "/dev/hailo0"
        online = os.path.exists(device)

        if not online:
            return NPUTelemetry(
                temp_c=0, power_w=0, utilization_percent=0,
                tops_available=0, device=device,
                status=Status.OFFLINE.value, online=False
            )

        try:
            # Get Hailo info via hailortcli
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True, text=True, timeout=5
            )

            # Try to get temperature
            temp = 0.0
            power = 0.0

            # Read RP1 temperature (board chip near NPU) as proxy for NPU area temp
            try:
                rp1_temp_path = Path("/sys/class/hwmon/hwmon1/temp1_input")
                if rp1_temp_path.exists():
                    temp = int(rp1_temp_path.read_text().strip()) / 1000.0
            except:
                pass

            # Estimate utilization (placeholder - real monitoring needs hailo-monitor)
            utilization = 0.0

            status = self.limits.check_npu_temp(temp) if temp > 0 else Status.OK

            return NPUTelemetry(
                temp_c=round(temp, 1),
                power_w=round(power, 2),
                utilization_percent=round(utilization, 1),
                tops_available=40.0,  # Hailo-10H spec
                device=device,
                status=status.value,
                online=True
            )

        except Exception as e:
            logger.debug(f"NPU telemetry limited: {e}")
            return NPUTelemetry(
                temp_c=0, power_w=0, utilization_percent=0,
                tops_available=40.0, device=device,
                status=Status.OK.value, online=True
            )

    def _get_single_storage_telemetry(self, device: str, mount: str, name: str, disk_key: str) -> StorageTelemetry:
        """Collect telemetry for a single storage device."""
        try:
            import psutil

            # Disk usage
            if os.path.exists(mount):
                usage = psutil.disk_usage(mount)
                used_gb = usage.used / (1024**3)
                total_gb = usage.total / (1024**3)
                used_percent = usage.percent
            else:
                return StorageTelemetry(0, 0, 0, 0, 0, 0, device, mount, Status.OFFLINE.value, name)

            # Disk I/O
            read_mbps = write_mbps = 0.0
            try:
                io = psutil.disk_io_counters(perdisk=True)
                if disk_key in io:
                    current_stats = io[disk_key]
                    current_time = datetime.now().timestamp()

                    stats_key = f"_last_{disk_key}_stats"
                    time_key = f"_last_{disk_key}_time"

                    last_stats = getattr(self, stats_key, None)
                    last_time = getattr(self, time_key, None)

                    if last_stats and last_time:
                        time_delta = current_time - last_time
                        if time_delta > 0:
                            read_bytes = current_stats.read_bytes - last_stats.read_bytes
                            write_bytes = current_stats.write_bytes - last_stats.write_bytes
                            read_mbps = (read_bytes / time_delta) / (1024**2)
                            write_mbps = (write_bytes / time_delta) / (1024**2)

                    setattr(self, stats_key, current_stats)
                    setattr(self, time_key, current_time)
            except:
                pass

            # Temperature (try hwmon)
            temp = 0.0
            try:
                for hwmon in Path("/sys/class/hwmon").iterdir():
                    name_file = hwmon / "name"
                    if name_file.exists():
                        hwmon_name = name_file.read_text().strip()
                        if "nvme" in hwmon_name.lower() or "jmicron" in hwmon_name.lower():
                            temp_file = hwmon / "temp1_input"
                            if temp_file.exists():
                                temp = float(temp_file.read_text().strip()) / 1000.0
                                break
            except:
                pass

            status = self.limits.check_storage_temp(temp) if temp > 0 else Status.OK

            return StorageTelemetry(
                temp_c=round(temp, 1),
                used_gb=round(used_gb, 1),
                total_gb=round(total_gb, 1),
                used_percent=round(used_percent, 1),
                read_mbps=round(read_mbps, 1),
                write_mbps=round(write_mbps, 1),
                device=device,
                mount=mount,
                status=status.value,
                name=name
            )

        except Exception as e:
            logger.error(f"Storage telemetry error for {name}: {e}")
            return StorageTelemetry(0, 0, 0, 0, 0, 0, device, mount, Status.UNKNOWN.value, name)

    def get_storage_telemetry(self) -> MultiStorageTelemetry:
        """Collect telemetry for all storage devices."""
        # Define storage devices to monitor
        storage_configs = [
            {"device": "/dev/sdb", "mount": "/", "name": "System SSD", "disk_key": "sdb"},
            {"device": "/dev/sda", "mount": "/mnt/moloch-data", "name": "Daten SSD", "disk_key": "sda"},
        ]

        devices = []
        worst_status = Status.OK

        for config in storage_configs:
            telemetry = self._get_single_storage_telemetry(
                config["device"], config["mount"], config["name"], config["disk_key"]
            )
            devices.append(telemetry)

            # Track worst status
            if telemetry.status == Status.CRITICAL.value:
                worst_status = Status.CRITICAL
            elif telemetry.status == Status.WARNING.value and worst_status != Status.CRITICAL:
                worst_status = Status.WARNING

        return MultiStorageTelemetry(
            devices=devices,
            status=worst_status.value
        )

    def get_ram_telemetry(self) -> RAMTelemetry:
        """Collect RAM telemetry."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_percent = mem.percent

            status = self.limits.check_ram_usage(used_percent)

            return RAMTelemetry(
                used_gb=round(used_gb, 2),
                total_gb=round(total_gb, 2),
                used_percent=round(used_percent, 1),
                available_gb=round(available_gb, 2),
                status=status.value
            )

        except Exception as e:
            logger.error(f"RAM telemetry error: {e}")
            return RAMTelemetry(0, 0, 0, 0, Status.UNKNOWN.value)

    def get_audio_telemetry(self) -> AudioTelemetry:
        """Collect audio channel status."""
        channels = []

        try:
            # Check ALSA cards
            asound_path = Path("/proc/asound")
            if asound_path.exists():
                for item in sorted(asound_path.iterdir()):
                    if item.is_dir() and item.name.startswith("card"):
                        card_id = item / "id"
                        if card_id.exists():
                            try:
                                name = card_id.read_text().strip()
                                card_num = int(item.name.replace("card", ""))

                                # Check if card is accessible
                                status = Status.OK.value
                                volume = 100  # Default

                                channels.append(AudioChannel(
                                    id=card_num,
                                    name=name,
                                    device=f"card{card_num}",
                                    status=status,
                                    volume_percent=volume
                                ))
                            except:
                                continue

            # Pad to 8 channels
            while len(channels) < 8:
                channels.append(AudioChannel(
                    id=len(channels),
                    name=f"Channel {len(channels)}",
                    device="",
                    status=Status.OFFLINE.value,
                    volume_percent=0
                ))

            active_count = sum(1 for c in channels if c.status == Status.OK.value)
            overall_status = Status.OK.value if active_count > 0 else Status.OFFLINE.value

            return AudioTelemetry(
                channels=[asdict(c) for c in channels[:8]],
                active_count=active_count,
                status=overall_status
            )

        except Exception as e:
            logger.error(f"Audio telemetry error: {e}")
            return AudioTelemetry([], 0, Status.UNKNOWN.value)

    def get_fan_telemetry(self) -> FanTelemetry:
        """
        Collect fan status.

        Tries to use ThermalManager (with smoothing/hysteresis) if running,
        otherwise falls back to direct sysfs reading.
        """
        # Try ThermalManager first (provides smoothed temp, hysteresis tracking)
        try:
            from core.hardware.thermal_manager import get_thermal_manager
            tm = get_thermal_manager()
            if tm.is_running:
                tel = tm.get_telemetry()
                return FanTelemetry(
                    state=tel.fan_level,
                    max_state=tel.fan_max,
                    speed_percent=tel.fan_percent,
                    temp_c=tel.temp_c,
                    status=tel.status
                )
        except ImportError:
            pass  # ThermalManager not available
        except Exception as e:
            logger.debug(f"ThermalManager not available: {e}")

        # Fallback: Direct sysfs reading
        try:
            cooling_path = Path("/sys/class/thermal/cooling_device0")

            if not cooling_path.exists():
                return FanTelemetry(
                    state=0, max_state=0, speed_percent=0,
                    temp_c=0, status=Status.OFFLINE.value
                )

            # Read current state
            state = int((cooling_path / "cur_state").read_text().strip())
            max_state = int((cooling_path / "max_state").read_text().strip())

            # Calculate percentage
            speed_percent = int((state / max_state) * 100) if max_state > 0 else 0

            # Get CPU temperature
            temp_c = 0.0
            try:
                result = subprocess.run(
                    ["vcgencmd", "measure_temp"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
                    temp_c = float(temp_str)
            except:
                pass

            # Determine status based on temperature
            if temp_c >= 80:
                status = Status.CRITICAL.value
            elif temp_c >= 70:
                status = Status.WARNING.value
            else:
                status = Status.OK.value

            return FanTelemetry(
                state=state,
                max_state=max_state,
                speed_percent=speed_percent,
                temp_c=temp_c,
                status=status
            )

        except Exception as e:
            logger.error(f"Fan telemetry error: {e}")
            return FanTelemetry(0, 0, 0, 0, Status.UNKNOWN.value)

    def collect(self) -> SystemTelemetry:
        """Collect all telemetry data."""
        cpu = self.get_cpu_telemetry()
        npu = self.get_npu_telemetry()
        storage = self.get_storage_telemetry()
        ram = self.get_ram_telemetry()
        audio = self.get_audio_telemetry()

        # Determine overall status (worst of all)
        statuses = [cpu.status, npu.status, storage.status, ram.status]
        # Also check individual storage devices
        for dev in storage.devices:
            statuses.append(dev.status)

        if Status.SHUTDOWN.value in statuses:
            overall = Status.SHUTDOWN.value
        elif Status.CRITICAL.value in statuses:
            overall = Status.CRITICAL.value
        elif Status.WARNING.value in statuses:
            overall = Status.WARNING.value
        elif Status.UNKNOWN.value in statuses:
            overall = Status.UNKNOWN.value
        else:
            overall = Status.OK.value

        return SystemTelemetry(
            timestamp=datetime.now().isoformat(),
            cpu=cpu,
            npu=npu,
            storage=storage,
            ram=ram,
            audio=audio,
            overall_status=overall
        )

    def check_safety(self) -> Tuple[bool, str]:
        """
        Check if system is within safe operating limits.

        Returns:
            Tuple of (is_safe, reason)
        """
        telemetry = self.collect()

        if telemetry.overall_status == Status.SHUTDOWN.value:
            return False, "SHUTDOWN: Hardware limits exceeded!"

        if telemetry.overall_status == Status.CRITICAL.value:
            reasons = []
            if telemetry.cpu.status == Status.CRITICAL.value:
                reasons.append(f"CPU temp critical: {telemetry.cpu.temp_c}°C")
            if telemetry.npu.status == Status.CRITICAL.value:
                reasons.append(f"NPU temp critical: {telemetry.npu.temp_c}°C")
            if telemetry.storage.status == Status.CRITICAL.value:
                reasons.append(f"Storage temp critical: {telemetry.storage.temp_c}°C")
            if telemetry.ram.status == Status.CRITICAL.value:
                reasons.append(f"RAM usage critical: {telemetry.ram.used_percent}%")
            return False, "CRITICAL: " + "; ".join(reasons)

        return True, "System within safe limits"

    def to_dict(self) -> Dict[str, Any]:
        """Get telemetry as dictionary."""
        telemetry = self.collect()
        fan = self.get_fan_telemetry()
        return {
            "timestamp": telemetry.timestamp,
            "cpu": asdict(telemetry.cpu),
            "npu": asdict(telemetry.npu),
            "storage": {
                "devices": [asdict(dev) for dev in telemetry.storage.devices],
                "status": telemetry.storage.status
            },
            "ram": asdict(telemetry.ram),
            "audio": asdict(telemetry.audio),
            "fan": asdict(fan),
            "overall_status": telemetry.overall_status
        }


# Global instance
_engine: Optional[TelemetryEngine] = None


def get_engine() -> TelemetryEngine:
    """Get or create the global telemetry engine."""
    global _engine
    if _engine is None:
        _engine = TelemetryEngine()
    return _engine


def collect() -> Dict[str, Any]:
    """Collect all telemetry data."""
    return get_engine().to_dict()


def check_safety() -> Tuple[bool, str]:
    """Check system safety."""
    return get_engine().check_safety()


if __name__ == "__main__":
    print("M.O.L.O.C.H. Telemetry Engine")
    print("=" * 60)

    engine = get_engine()
    data = engine.to_dict()

    print(f"\nTimestamp: {data['timestamp']}")
    print(f"Overall Status: {data['overall_status'].upper()}")

    print(f"\nCPU:")
    print(f"  Temperature: {data['cpu']['temp_c']}°C")
    print(f"  Usage: {data['cpu']['usage_percent']}%")
    print(f"  Frequency: {data['cpu']['freq_mhz']} MHz")

    print(f"\nNPU (Hailo-10H):")
    print(f"  Online: {data['npu']['online']}")
    print(f"  Temperature: {data['npu']['temp_c']}°C")
    print(f"  TOPS Available: {data['npu']['tops_available']}")

    print(f"\nStorage:")
    for dev in data['storage']['devices']:
        print(f"  {dev['name']} ({dev['mount']}):")
        print(f"    Used: {dev['used_gb']:.1f} / {dev['total_gb']:.1f} GB ({dev['used_percent']}%)")
        print(f"    Read: {dev['read_mbps']:.1f} MB/s | Write: {dev['write_mbps']:.1f} MB/s")

    print(f"\nRAM:")
    print(f"  Used: {data['ram']['used_gb']:.2f} / {data['ram']['total_gb']:.2f} GB ({data['ram']['used_percent']}%)")

    print(f"\nAudio:")
    print(f"  Active Channels: {data['audio']['active_count']}/8")

    is_safe, reason = engine.check_safety()
    print(f"\nSafety Check: {'PASS' if is_safe else 'FAIL'}")
    print(f"  {reason}")
