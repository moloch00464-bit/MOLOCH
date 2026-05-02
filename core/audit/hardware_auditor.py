"""Hardware-Auditor (Welle 12 Schritt 4).

Pi-eigene Hardware-Probes:
- Kamera RTSP via ffprobe (timeout 5s)
- Kamera Ping (ICMP)
- ReSpeaker Mic UDP-Heartbeat
- Disk-Free /mnt/moloch-data
- vcgencmd get_throttled (Pi CPU-Throttle)
- CPU-Temp + RAM (Cross-Check zu pi-Layer)

Schreibt audit_state.layers.hardware Schema:
  {camera_reachable, camera_rtsp_ok, camera_ping_ms,
   audio_mic_pegel, mic_connected, disk_free_gb, cpu_throttled,
   cpu_temp, status, score, max, detail}

Status-Logik:
- PASS: camera+mic+disk+throttle alle ok
- WARN: 1 Komponente warn (z.B. cpu_throttled flag, disk_free <20GB)
- FAIL: camera oder mic unreachable
"""
from __future__ import annotations

import shutil
import subprocess
import socket
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("hardware_auditor")

CAMERA_IP = "192.168.178.25"
CAMERA_RTSP = "rtsp://Moloch_4.5:Auge666@192.168.178.25:554/av_stream/ch0"
RESPEAKER_IP = "10.42.0.2"


def _ping(host: str, count: int = 2) -> Optional[float]:
    """Returns avg ping ms or None."""
    try:
        r = subprocess.run(
            ["ping", "-c", str(count), "-W", "2", host],
            capture_output=True, text=True, timeout=8,
        )
        for ln in r.stdout.splitlines():
            if "avg" in ln:
                # rtt min/avg/max/mdev = 10/20/30/5 ms
                parts = ln.split("=")[-1].strip().split()[0].split("/")
                return float(parts[1]) if len(parts) >= 2 else None
    except Exception:
        pass
    return None


def _rtsp_ok(rtsp_url: str, timeout: int = 5) -> bool:
    """ffprobe RTSP-Stream — exit 0 = ok."""
    try:
        r = subprocess.run(
            ["ffprobe", "-rtsp_transport", "tcp", "-v", "error",
             "-timeout", "3000000", rtsp_url],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


def _tcp_open(host: str, port: int, timeout: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _vcgencmd_throttled() -> Dict[str, Any]:
    try:
        r = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=3,
        )
        # Format: "throttled=0x0" oder "throttled=0x50000"
        out = r.stdout.strip()
        if "=" in out:
            val_hex = out.split("=", 1)[1]
            val_int = int(val_hex, 16) if val_hex.startswith("0x") else int(val_hex)
            return {"raw": out, "value": val_int, "throttled_now": bool(val_int & 0x4)}
    except Exception as e:
        return {"error": str(e)[:80]}
    return {"value": 0, "throttled_now": False}


def collect() -> Dict[str, Any]:
    """Sammelt Hardware-Layer-Daten."""
    detail: Dict[str, Any] = {}

    # 1. Kamera
    cam_ping = _ping(CAMERA_IP)
    cam_rtsp = _rtsp_ok(CAMERA_RTSP)
    cam_reachable = cam_ping is not None
    detail["camera_ping_ms"] = round(cam_ping, 1) if cam_ping else None
    detail["camera_rtsp_ok"] = cam_rtsp

    # 2. ReSpeaker Mic (TCP:80)
    mic_connected = _tcp_open(RESPEAKER_IP, 80)
    detail["mic_connected_http"] = mic_connected

    # 3. Disk
    try:
        usage = shutil.disk_usage("/mnt/moloch-data")
        disk_free_gb = round(usage.free / (1024**3), 1)
        disk_total_gb = round(usage.total / (1024**3), 1)
    except Exception:
        disk_free_gb = -1
        disk_total_gb = -1
    detail["disk_total_gb"] = disk_total_gb

    # 4. vcgencmd throttle
    throttled = _vcgencmd_throttled()
    detail["throttled"] = throttled

    # 5. CPU-Temp aus thermal_zone
    cpu_temp = None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = round(int(f.read().strip()) / 1000, 1)
    except Exception:
        pass
    detail["cpu_temp_c"] = cpu_temp

    # Status-Berechnung
    score = 0
    max_score = 5
    if cam_reachable:
        score += 1
    if cam_rtsp:
        score += 1
    if mic_connected:
        score += 1
    if disk_free_gb >= 20:
        score += 1
    if not throttled.get("throttled_now") and (cpu_temp is None or cpu_temp < 70):
        score += 1

    # Pi-essentielle Hardware: Camera + Disk. Bei Verlust = FAIL.
    # ESP32-Mic ist external (separates WiFi-Gerät, kann selbst rebooten).
    # Mic-Outage = WARN (bewusst keine FAIL — vergleiche tentacle_auditor).
    if not cam_reachable or not cam_rtsp:
        status = "FAIL"
    elif disk_free_gb < 5:
        status = "FAIL"
    elif not mic_connected:
        # ESP32 nicht erreichbar — externe Outage, Pi-Hardware OK.
        status = "WARN"
    elif throttled.get("throttled_now") or disk_free_gb < 20 or (cpu_temp and cpu_temp >= 65):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "camera_reachable": cam_reachable,
        "camera_rtsp_ok": cam_rtsp,
        "camera_ping_ms": detail["camera_ping_ms"],
        "mic_connected": mic_connected,
        "disk_free_gb": disk_free_gb,
        "cpu_throttled": throttled.get("throttled_now", False),
        "cpu_temp": cpu_temp,
        "detail": detail,
    }
