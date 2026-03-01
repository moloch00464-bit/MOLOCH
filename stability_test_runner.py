#!/usr/bin/env python3
"""
Gate 0 Phase 10 — 6h Stabilitaetstest Runner.
Separater Prozess. Liest NUR von aussen:
  - /dev/shm/moloch_status.json
  - psutil (CPU, RAM, Threads)
  - /sys/class/thermal/thermal_zone0/temp
Loggt alle 5 Sekunden nach ~/moloch/logs/stability_log.jsonl
"""

import json
import os
import sys
import time
import signal
import psutil
from datetime import datetime, timezone
from pathlib import Path

# --- Konfiguration ---
STATUS_PATH = "/dev/shm/moloch_status.json"
THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_FILE = LOG_DIR / "stability_log.jsonl"
INTERVAL = 5  # Sekunden
SERVICE_NAME = "moloch_service"

running = True


def signal_handler(sig, frame):
    global running
    running = False
    print(f"\n[STOPP] Signal {sig} empfangen. Beende Runner.")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def read_status():
    """Liest /dev/shm/moloch_status.json — reines Lesen, kein Schreiben."""
    try:
        with open(STATUS_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_cpu_temp():
    """Liest CPU-Temperatur aus /sys/thermal."""
    try:
        with open(THERMAL_PATH, "r") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError, OSError):
        return None


def find_moloch_process():
    """Findet den Moloch-Service Prozess via psutil."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any(SERVICE_NAME in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def collect_sample(moloch_proc, start_time):
    """Sammelt einen Messpunkt."""
    now = datetime.now(timezone.utc)
    elapsed_s = (now - start_time).total_seconds()

    sample = {
        "timestamp": now.isoformat(),
        "elapsed_s": round(elapsed_s, 1),
    }

    # --- System-Metriken (psutil) ---
    sample["system"] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 1),
        "ram_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
        "ram_percent": psutil.virtual_memory().percent,
        "swap_used_mb": round(psutil.swap_memory().used / 1024 / 1024, 1),
    }

    # --- CPU Temperatur ---
    cpu_temp = read_cpu_temp()
    sample["cpu_temp_c"] = round(cpu_temp, 1) if cpu_temp is not None else None

    # --- Moloch Prozess-Metriken ---
    if moloch_proc and moloch_proc.is_running():
        try:
            mem = moloch_proc.memory_info()
            sample["moloch_process"] = {
                "pid": moloch_proc.pid,
                "rss_mb": round(mem.rss / 1024 / 1024, 1),
                "vms_mb": round(mem.vms / 1024 / 1024, 1),
                "cpu_percent": moloch_proc.cpu_percent(interval=None),
                "threads": moloch_proc.num_threads(),
                "status": moloch_proc.status(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            sample["moloch_process"] = {"error": "process_lost"}
    else:
        sample["moloch_process"] = {"error": "not_found"}

    # --- moloch_status.json ---
    status = read_status()
    if status:
        fps_data = status.get("fps", {})
        perception = status.get("perception", {})
        core = status.get("core", {})

        sample["moloch_status"] = {
            "fps_total": fps_data.get("total", 0),
            "fps_scrfd": fps_data.get("scrfd", 0),
            "fps_yolo": fps_data.get("yolov8m", 0),
            "fps_arcface": fps_data.get("arcface", 0),
            "npu_stage": perception.get("npu_stage", "unknown"),
            "active_models": status.get("active_models", []),
            "personality_mode": status.get("personality_mode", "unknown"),
            "tension": status.get("tension", -1),
            "autonomous_mode": status.get("autonomous_mode", False),
            "frozen_restarts": status.get("frozen_restarts", 0),
            "frame_age": status.get("frame_age", -1),
        }

        # PTZ Conflict: moloch_has_control und manual_mode gleichzeitig
        moloch_ctrl = status.get("moloch_has_control", False)
        manual = status.get("manual_mode", False)
        sample["moloch_status"]["ptz_conflict"] = moloch_ctrl and manual
    else:
        sample["moloch_status"] = {"error": "status_file_unreadable"}

    return sample


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GATE 0 PHASE 10 — STABILITAETSTEST RUNNER")
    print("  Intervall: 5 Sekunden")
    print(f"  Log: {LOG_FILE}")
    print(f"  Status: {STATUS_PATH}")
    print("  Stoppen: Ctrl+C oder kill")
    print("=" * 60)

    # Moloch-Prozess finden
    moloch_proc = find_moloch_process()
    if moloch_proc:
        print(f"  Moloch PID: {moloch_proc.pid}")
    else:
        print("  WARNUNG: Moloch-Prozess nicht gefunden!")
        print("  Runner laeuft trotzdem — Prozess wird bei jedem Sample gesucht.")

    # CPU-Percent einmal aufrufen (erster Aufruf ist immer 0)
    psutil.cpu_percent(interval=None)
    if moloch_proc and moloch_proc.is_running():
        try:
            moloch_proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    start_time = datetime.now(timezone.utc)
    sample_count = 0

    print(f"\n  Start: {start_time.isoformat()}")
    print("  Laeuft...\n")

    while running:
        # Moloch-Prozess ggf. neu finden (nach Crash/Restart)
        if not moloch_proc or not moloch_proc.is_running():
            moloch_proc = find_moloch_process()
            if moloch_proc:
                try:
                    moloch_proc.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        sample = collect_sample(moloch_proc, start_time)
        sample_count += 1

        # In JSONL schreiben
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  FEHLER beim Schreiben: {e}")

        # Kompakte Status-Zeile alle 60s
        if sample_count % 12 == 0:
            elapsed_min = sample["elapsed_s"] / 60
            ms = sample.get("moloch_status", {})
            mp = sample.get("moloch_process", {})
            temp = sample.get("cpu_temp_c", "?")
            fps = ms.get("fps_total", "?")
            rss = mp.get("rss_mb", "?")
            threads = mp.get("threads", "?")
            stage = ms.get("npu_stage", "?")
            print(f"  [{elapsed_min:6.1f}min] FPS={fps} RSS={rss}MB Threads={threads} "
                  f"Temp={temp}°C Stage={stage}")

        time.sleep(INTERVAL)

    # Abschluss
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    print(f"\n  Ende: {end_time.isoformat()}")
    print(f"  Dauer: {duration / 3600:.2f}h ({sample_count} Samples)")
    print(f"  Log: {LOG_FILE}")
    print(f"\n  Analyse starten: python3 ~/moloch/analyze_stability.py")


if __name__ == "__main__":
    main()
