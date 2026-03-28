#!/usr/bin/env python3
"""
M.O.L.O.C.H. POST-FLIGHT CHECK
=================================
Vergleicht aktuellen System-Zustand mit Preflight-Baseline.
Fuehrt Audit aus. Zeigt Delta-Tabelle.

Aufruf: python3 ~/moloch/scripts/postflight.py
Exit:   0 = PASS, 1 = FAIL
Voraussetzung: preflight.py muss vorher gelaufen sein.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PREFLIGHT_FILE = "/tmp/moloch_preflight.json"
STATUS_FILE = "/dev/shm/moloch_status.json"
THERMAL_FILE = "/sys/class/thermal/thermal_zone0/temp"

# Delta-Schwellwerte
WARN_THRESHOLDS = {
    "ram_mb": 50,       # +50 MB = WARN
    "fps": -5,          # -5 FPS = WARN
    "cpu_c": 5.0,       # +5 C = WARN
    "threads": 5,       # +5 Threads = WARN
}

FAIL_THRESHOLDS = {
    "ram_mb": 200,      # +200 MB = FAIL
    "min_fps": 10.0,    # Unter 10 = FAIL
}


def load_preflight() -> dict:
    """Laedt Preflight-Snapshot."""
    if not os.path.exists(PREFLIGHT_FILE):
        return {}
    with open(PREFLIGHT_FILE) as f:
        return json.load(f)


def get_current_values() -> dict:
    """Aktuelle Systemwerte erfassen (gleiche Quellen wie preflight)."""
    values = {}

    # Service
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "moloch"],
            capture_output=True, text=True, timeout=5
        )
        values["service"] = r.stdout.strip()
    except Exception:
        values["service"] = "unknown"

    # Status-JSON
    try:
        with open(STATUS_FILE) as f:
            status = json.load(f)
        fps_data = status.get("fps", {})
        values["fps"] = fps_data.get("total", 0.0) if isinstance(fps_data, dict) else float(fps_data)
        values["frame_age"] = status.get("frame_age", -1.0)
    except Exception:
        values["fps"] = 0.0
        values["frame_age"] = -1.0

    # RAM
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        values["ram_mb"] = (total - available) // 1024
    except Exception:
        values["ram_mb"] = -1

    # CPU Temp
    try:
        with open(THERMAL_FILE) as f:
            values["cpu_c"] = int(f.read().strip()) / 1000.0
    except Exception:
        values["cpu_c"] = -1.0

    # Threads
    try:
        r = subprocess.run(
            ["pgrep", "-f", "moloch_service"],
            capture_output=True, text=True, timeout=5
        )
        pid = r.stdout.strip().split("\n")[0]
        if pid:
            task_dir = Path(f"/proc/{pid}/task")
            values["threads"] = len(list(task_dir.iterdir())) if task_dir.exists() else -1
        else:
            values["threads"] = -1
    except Exception:
        values["threads"] = -1

    return values


def run_audit() -> tuple:
    """moloch_audit.py --auto ausfuehren. Gibt (passed, summary) zurueck."""
    audit_script = os.path.expanduser("~/moloch/moloch_audit.py")
    if not os.path.exists(audit_script):
        return False, "audit script nicht gefunden"
    try:
        r = subprocess.run(
            ["python3", audit_script, "--auto"],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode == 0:
            return True, "PASS"
        last_lines = r.stdout.strip().split("\n")[-3:]
        return False, " | ".join(l.strip() for l in last_lines if l.strip())[:80]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (>120s)"
    except Exception as e:
        return False, str(e)[:80]


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Preflight laden
    preflight = load_preflight()
    if not preflight:
        print(f"[POSTFLIGHT] {ts}")
        print()
        print("  FEHLER: Keine Preflight-Daten gefunden!")
        print("  Bitte zuerst: python3 ~/moloch/scripts/preflight.py")
        return 1

    pre_ts = preflight.get("timestamp", "?")
    print(f"[POSTFLIGHT] {ts}")
    print(f"  Baseline von: {pre_ts}")
    print()

    # Aktuelle Werte
    current = get_current_values()

    # Delta-Tabelle
    has_fail = False
    has_warn = False

    header = f"  {'Metrik':<14}| {'Vorher':>8} | {'Nachher':>8} | {'Delta':>8} | Status"
    separator = f"  {'-'*14}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*8}"
    print(header)
    print(separator)

    def row(name, pre_val, cur_val, fmt="d", higher_bad=True):
        nonlocal has_fail, has_warn

        if pre_val is None or pre_val < 0 or cur_val is None or cur_val < 0:
            delta_str = "--"
            status = "?"
        else:
            delta = cur_val - pre_val
            if fmt == "f":
                delta_str = f"{delta:+.1f}"
                pre_str = f"{pre_val:.1f}"
                cur_str = f"{cur_val:.1f}"
            else:
                delta_str = f"{delta:+d}"
                pre_str = str(pre_val)
                cur_str = str(cur_val)

            # Bewertung
            status = "OK"
            threshold_key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

            if name == "FPS":
                if cur_val < FAIL_THRESHOLDS.get("min_fps", 10):
                    status = "FAIL"
                    has_fail = True
                elif delta < WARN_THRESHOLDS.get("fps", -5):
                    status = "WARN"
                    has_warn = True
            elif name == "RAM (MB)":
                if delta > FAIL_THRESHOLDS.get("ram_mb", 200):
                    status = "FAIL"
                    has_fail = True
                elif delta > WARN_THRESHOLDS.get("ram_mb", 50):
                    status = "WARN"
                    has_warn = True
            elif name == "CPU (C)":
                if delta > WARN_THRESHOLDS.get("cpu_c", 5.0):
                    status = "WARN"
                    has_warn = True
            elif name == "Threads":
                if delta > WARN_THRESHOLDS.get("threads", 5):
                    status = "WARN"
                    has_warn = True

            print(f"  {name:<14}| {pre_str:>8} | {cur_str:>8} | {delta_str:>8} | {status}")
            return

        # Fallback fuer fehlende Werte
        print(f"  {name:<14}| {'?':>8} | {'?':>8} | {delta_str:>8} | {status}")

    row("RAM (MB)", preflight.get("ram_mb"), current["ram_mb"])
    row("FPS", preflight.get("fps"), current["fps"], fmt="f", higher_bad=False)
    row("CPU (C)", preflight.get("cpu_c"), current["cpu_c"], fmt="f")
    row("Threads", preflight.get("threads"), current["threads"])

    # Service-Status
    pre_service = preflight.get("service", "?")
    cur_service = current.get("service", "?")
    svc_status = "OK" if cur_service == "active" else "FAIL"
    if cur_service != "active":
        has_fail = True
    print(f"  {'Service':<14}| {pre_service:>8} | {cur_service:>8} | {'--':>8} | {svc_status}")

    # Audit
    print()
    print("  Audit laeuft...")
    audit_passed, audit_summary = run_audit()
    audit_status = "PASS" if audit_passed else "FAIL"
    if not audit_passed:
        has_fail = True
    print(f"  {'Audit':<14}| {'--':>8} | {audit_summary[:8]:>8} | {'--':>8} | {audit_status}")

    # Zusammenfassung
    print()
    if has_fail:
        print("  Ergebnis: FAIL")
        print("  EMPFEHLUNG: git checkout -- [datei] und Root-Cause analysieren.")
        return 1
    elif has_warn:
        print("  Ergebnis: WARN — Metriken verschlechtert, aber noch akzeptabel.")
        return 0
    else:
        print("  Ergebnis: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
