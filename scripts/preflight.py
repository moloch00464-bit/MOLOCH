#!/usr/bin/env python3
"""
M.O.L.O.C.H. PRE-FLIGHT CHECK
================================
Erfasst System-Baseline VOR einer Code-Aenderung.
Speichert Snapshot nach /tmp/moloch_preflight.json.

Aufruf: python3 ~/moloch/scripts/preflight.py
Exit:   0 = alles gruen, 1 = Problem erkannt
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STATUS_FILE = "/dev/shm/moloch_status.json"
OUTPUT_FILE = "/tmp/moloch_preflight.json"
THERMAL_FILE = "/sys/class/thermal/thermal_zone0/temp"

# Schwellwerte (gleich wie moloch_audit.py)
LIMITS = {
    "min_fps": 10.0,
    "max_ram_mb": 3500,
    "max_cpu_c": 80.0,
    "max_frame_age": 5.0,
    "max_threads": 50,
}


def get_service_status() -> str:
    """systemctl is-active moloch"""
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "moloch"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def get_git_status() -> dict:
    """Git-Status: clean/dirty + Anzahl geaenderter Dateien."""
    moloch_dir = os.path.expanduser("~/moloch")
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=moloch_dir
        )
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        return {"clean": len(lines) == 0, "dirty_files": len(lines)}
    except Exception:
        return {"clean": False, "dirty_files": -1}


def get_status_json() -> dict:
    """Liest /dev/shm/moloch_status.json."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def get_cpu_temp() -> float:
    """CPU-Temperatur aus /sys."""
    try:
        with open(THERMAL_FILE) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


def get_ram_used_mb() -> int:
    """RAM-Verbrauch aus /proc/meminfo."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        return (total - available) // 1024  # KB → MB
    except Exception:
        return -1


def get_npu_status() -> str:
    """NPU erreichbar?"""
    try:
        r = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True, text=True, timeout=10
        )
        return "reachable" if r.returncode == 0 else "error"
    except Exception:
        return "unreachable"


def get_thread_count() -> int:
    """Thread-Anzahl des moloch-Service."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "moloch_service"],
            capture_output=True, text=True, timeout=5
        )
        pid = r.stdout.strip().split("\n")[0]
        if pid:
            task_dir = Path(f"/proc/{pid}/task")
            if task_dir.exists():
                return len(list(task_dir.iterdir()))
    except Exception:
        pass
    return -1


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PREFLIGHT] {ts}")
    print()

    # Daten sammeln
    service = get_service_status()
    git = get_git_status()
    status = get_status_json()
    cpu_temp = get_cpu_temp()
    ram_mb = get_ram_used_mb()
    npu = get_npu_status()
    threads = get_thread_count()

    # FPS aus Status-JSON
    fps_data = status.get("fps", {})
    if isinstance(fps_data, dict):
        fps = fps_data.get("total", 0.0)
    else:
        fps = float(fps_data) if fps_data else 0.0

    frame_age = status.get("frame_age", -1.0)

    # Bewertung
    checks = {}
    has_fail = False

    def check(name, value, fmt, ok_cond, limit_str=""):
        nonlocal has_fail
        passed = ok_cond
        status_str = "OK" if passed else "FAIL"
        if not passed:
            has_fail = True
        checks[name] = {"value": value, "ok": passed}
        pad_name = f"{name}:".ljust(14)
        pad_val = f"{fmt}".ljust(18)
        lim = f"({limit_str})" if limit_str else ""
        print(f"  {pad_name}{pad_val}{status_str} {lim}")

    check("Service", service, service,
          service == "active")
    check("Git", "clean" if git["clean"] else f"dirty ({git['dirty_files']} Dateien)",
          "clean" if git["clean"] else f"dirty ({git['dirty_files']})",
          git["clean"])
    check("FPS", fps, f"{fps:.1f}",
          fps >= LIMITS["min_fps"],
          f"min: {LIMITS['min_fps']}")
    check("RAM", ram_mb, f"{ram_mb} MB",
          0 < ram_mb < LIMITS["max_ram_mb"],
          f"max: {LIMITS['max_ram_mb']}")
    check("CPU", cpu_temp, f"{cpu_temp:.1f} C",
          0 < cpu_temp < LIMITS["max_cpu_c"],
          f"max: {LIMITS['max_cpu_c']}")
    check("Frame Age", frame_age, f"{frame_age:.1f}s",
          0 <= frame_age < LIMITS["max_frame_age"],
          f"max: {LIMITS['max_frame_age']}")
    check("NPU", npu, npu,
          npu == "reachable")
    check("Threads", threads, str(threads),
          0 < threads < LIMITS["max_threads"],
          f"max: {LIMITS['max_threads']}")

    # Snapshot speichern
    snapshot = {
        "timestamp": ts,
        "service": service,
        "git": git,
        "fps": fps,
        "ram_mb": ram_mb,
        "cpu_c": cpu_temp,
        "frame_age": frame_age,
        "npu": npu,
        "threads": threads,
        "checks": checks,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print()
    if has_fail:
        print(f"  Ergebnis: FAIL — Baseline gespeichert in {OUTPUT_FILE}")
        print("  WARNUNG: System nicht in optimalem Zustand vor Aenderung!")
        return 1
    else:
        print(f"  Ergebnis: PASS — Baseline gespeichert in {OUTPUT_FILE}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
