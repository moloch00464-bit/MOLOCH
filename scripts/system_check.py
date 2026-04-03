#!/usr/bin/env python3
"""
M.O.L.O.C.H. System Check Script
Monitors Raspberry Pi 5 with Hailo-10H (40 TOPS) and Samsung 980 NVMe
"""

import os
import subprocess
import psutil
import datetime
from pathlib import Path

LOG_FILE = Path.home() / "moloch" / "logs" / "system_check.log"

def get_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.datetime.now().isoformat()

def check_hailo():
    """Check Hailo-10H NPU status"""
    try:
        # Check if Hailo device exists
        hailo_devices = list(Path("/dev").glob("hailo*"))
        if hailo_devices:
            status = f"Hailo devices found: {[str(d) for d in hailo_devices]}"
        else:
            status = "WARNING: No Hailo devices found in /dev"

        # Try hailortcli if available
        try:
            result = subprocess.run(
                ["hailortcli", "fw-control", "identify"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status += f" | hailortcli: OK"
            else:
                status += f" | hailortcli: {result.stderr.strip()}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            status += " | hailortcli: not available"

        return status
    except Exception as e:
        return f"ERROR checking Hailo: {e}"

def check_nvme():
    """Check NVMe disk space (Samsung 980)"""
    try:
        # Find NVMe mount points (Samsung 980 may show as /dev/sda or /dev/nvme*)
        nvme_info = []
        for partition in psutil.disk_partitions():
            # Check for nvme in device name OR sda (Samsung 980 NVMe)
            if 'nvme' in partition.device or 'sda' in partition.device:
                usage = psutil.disk_usage(partition.mountpoint)
                nvme_info.append(
                    f"{partition.device} ({partition.mountpoint}): "
                    f"{usage.percent}% used, "
                    f"{usage.free // (1024**3)}GB free / "
                    f"{usage.total // (1024**3)}GB total"
                )

        if nvme_info:
            return " | ".join(nvme_info)
        else:
            return "WARNING: No NVMe partitions found"
    except Exception as e:
        return f"ERROR checking NVMe: {e}"

def get_system_metrics():
    """Get CPU, RAM, and temperature metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # RAM usage
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)

        # Temperature (Raspberry Pi specific)
        temp = "N/A"
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_raw = int(f.read().strip())
                temp = f"{temp_raw / 1000:.1f}°C"
        except:
            pass

        return (
            f"CPU: {cpu_percent}% ({cpu_count} cores) | "
            f"RAM: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB ({mem.percent}%) | "
            f"Temp: {temp}"
        )
    except Exception as e:
        return f"ERROR getting system metrics: {e}"

def log_entry(message):
    """Append log entry to file"""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"ERROR writing to log: {e}")

def main():
    """Run system check and log results"""
    timestamp = get_timestamp()

    # Gather all metrics
    hailo_status = check_hailo()
    nvme_status = check_nvme()
    system_metrics = get_system_metrics()

    # Format log entry
    log_message = (
        f"[{timestamp}]\n"
        f"  Hailo: {hailo_status}\n"
        f"  NVMe: {nvme_status}\n"
        f"  System: {system_metrics}\n"
    )

    # Write to log
    log_entry(log_message)

    # Also print to stdout for cron emails/debugging
    print(log_message)

if __name__ == "__main__":
    main()
