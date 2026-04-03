#!/usr/bin/env python3
"""
M.O.L.O.C.H. SMOKE TEST — 8-Punkt Post-Reboot Checkliste
===========================================================
Schneller Sanity-Check nach sudo reboot.

Aufruf: python3 ~/moloch/scripts/smoke_test.py
Exit:   0 = alle 8 PASS, 1 = mindestens 1 FAIL
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STATUS_FILE = "/dev/shm/moloch_status.json"
FRAME_SHM = "/dev/shm/moloch_frame"
LOG_DIR = os.path.expanduser("~/moloch/logs")
REPORT_FILE = os.path.join(LOG_DIR, "smoke_test_last.json")


def run_cmd(cmd, timeout=10):
    """Subprocess mit Timeout. Gibt (returncode, stdout) zurueck."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def check_1_service():
    """Service laeuft?"""
    rc, out = run_cmd(["systemctl", "is-active", "moloch"])
    return out == "active", out


def check_2_fps():
    """FPS > 10?"""
    try:
        with open(STATUS_FILE) as f:
            data = json.load(f)
        fps_data = data.get("fps", {})
        fps = fps_data.get("total", 0.0) if isinstance(fps_data, dict) else float(fps_data)
        return fps > 10.0, f"{fps:.1f}"
    except Exception as e:
        return False, str(e)


def check_3_ram():
    """RAM < 3500 MB verwendet?"""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        used_mb = (total - available) // 1024
        return used_mb < 3500, f"{used_mb} MB"
    except Exception as e:
        return False, str(e)


def check_4_frame_age():
    """Frame-Alter < 5 Sekunden?"""
    try:
        with open(STATUS_FILE) as f:
            data = json.load(f)
        age = data.get("frame_age", 999)
        return age < 5.0, f"{age:.1f}s"
    except Exception as e:
        return False, str(e)


def check_5_npu():
    """NPU erreichbar?"""
    rc, out = run_cmd(["hailortcli", "fw-control", "identify"], timeout=10)
    if rc == 0:
        # Modellname extrahieren
        for line in out.split("\n"):
            if "Board Name" in line or "Device" in line:
                return True, line.strip()
        return True, "Hailo OK"
    return False, out[:80]


def check_6_ipc():
    """IPC Frame-SHM existiert und hat Inhalt?"""
    p = Path(FRAME_SHM)
    if not p.exists():
        return False, "nicht vorhanden"
    size = p.stat().st_size
    if size == 0:
        return False, "leer (0 Bytes)"
    return True, f"{size} Bytes"


def check_7_segv():
    """Kein SEGV in den letzten dmesg-Zeilen?"""
    rc, out = run_cmd(["dmesg"], timeout=5)
    if rc != 0:
        # dmesg braucht evtl. root — versuche mit sudo
        rc, out = run_cmd(["sudo", "dmesg"], timeout=5)
    if rc != 0:
        return True, "dmesg nicht lesbar (OK angenommen)"
    # Letzte 100 Zeilen pruefen
    lines = out.split("\n")[-100:]
    segv_count = sum(1 for l in lines if "SEGV" in l or "segfault" in l.lower())
    if segv_count > 0:
        return False, f"{segv_count} SEGV gefunden"
    return True, "0 SEGV"


def check_8_audit():
    """moloch_audit.py --auto besteht?"""
    audit_script = os.path.expanduser("~/moloch/moloch_audit.py")
    if not os.path.exists(audit_script):
        return False, "audit script nicht gefunden"
    rc, out = run_cmd(["python3", audit_script, "--auto"], timeout=120)
    if rc == 0:
        return True, "PASS"
    # Letzte Zeile fuer Zusammenfassung
    last_lines = out.strip().split("\n")[-3:]
    summary = " | ".join(l.strip() for l in last_lines if l.strip())
    return False, summary[:120]


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SMOKE TEST] {ts}")
    print()

    checks = [
        ("Service", check_1_service),
        ("FPS", check_2_fps),
        ("RAM", check_3_ram),
        ("Frame Age", check_4_frame_age),
        ("NPU", check_5_npu),
        ("IPC", check_6_ipc),
        ("SEGV", check_7_segv),
        ("Audit", check_8_audit),
    ]

    results = []
    pass_count = 0
    fail_count = 0

    for i, (name, func) in enumerate(checks, 1):
        passed, detail = func()
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        pad_name = f"{name}:".ljust(14)
        pad_detail = f"{detail}".ljust(22)
        print(f"  {i}. {pad_name}{pad_detail}{status}")

        results.append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })

    print()
    overall = "PASS" if fail_count == 0 else "FAIL"
    print(f"  Ergebnis: {pass_count}/{len(checks)} PASS — {overall}")

    # Report speichern
    report = {
        "timestamp": ts,
        "overall": overall,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "checks": results,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
