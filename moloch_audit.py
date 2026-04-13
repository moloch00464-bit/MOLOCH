#!/usr/bin/env python3
"""
M.O.L.O.C.H. Audit-Skript
Prüft System- und Service-Status, speichert Ergebnis in logs/audit_last.json
"""

import subprocess
import sys
import json
import os
import time
import logging
import shlex
import tempfile
from pathlib import Path
import urllib.request
import urllib.error

# Konfiguration
LOG_DIR = Path.home() / "moloch" / "logs"
AUDIT_JSON = LOG_DIR / "audit_last.json"
EVENT_TRACE = Path.home() / "moloch" / "logs" / "event_trace.log"
SERVICE_NAME = "moloch"
QDRANT_URL = "http://localhost:6333/"

def run_cmd(cmd):
    """Führt Shell-Befehl aus, gibt stdout zurück."""
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=5)
        return result.stdout.strip(), result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1
    except Exception as e:
        return "", str(e), -1

def check_service():
    """Prüft ob moloch-Service aktiv ist."""
    out, err, code = run_cmd(f"systemctl is-active {SERVICE_NAME}")
    if code == 0 and out == "active":
        return "PASS", f"Service {SERVICE_NAME} ist aktiv"
    else:
        return "FAIL", f"Service {SERVICE_NAME} nicht aktiv (out:{out}, err:{err})"

def check_ram():
    """Prüft RAM-Nutzung unter 80%."""
    out, err, code = run_cmd("free -h")
    if code != 0:
        return "WARN", f"RAM-Check fehlgeschlagen: {err}"
    lines = out.splitlines()
    if len(lines) < 2:
        return "WARN", "Unerwartetes free-Format"
    mem_line = lines[1].split()
    # total, used, free, shared, buff/cache, available
    if len(mem_line) >= 7:
        total_str = mem_line[1]
        used_str = mem_line[2]
        # Entferne Einheiten (M, G) für Berechnung
        try:
            # Entferne Einheiten: G, Gi, M, Mi
            total = float(total_str.rstrip('iMGTK'))
            used = float(used_str.rstrip('iMGTK'))
            unit = total_str.lstrip('0123456789.')
            # Auf gleiche Einheit bringen (alles in MiB)
            if unit.startswith('G'):
                total *= 1024
                used *= 1024
            percent = (used / total) * 100 if total > 0 else 0
            if percent < 80:
                return "PASS", f"RAM {percent:.1f}% (<80%)"
            else:
                return "FAIL", f"RAM {percent:.1f}% (>=80%)"
        except ValueError:
            return "WARN", f"Konnte RAM-Werte nicht parsen: {total_str}, {used_str}"
    return "WARN", "RAM-Daten unvollständig"

def check_temp():
    """Prüft CPU-Temperatur unter 70°C."""
    out, err, code = run_cmd("vcgencmd measure_temp")
    if code != 0:
        return "WARN", f"Temperatur-Check fehlgeschlagen: {err}"
    # Ausgabe: temp=47.2'C
    if "temp=" in out:
        try:
            temp_str = out.split('=')[1].split("'")[0]
            temp = float(temp_str)
            if temp < 70:
                return "PASS", f"CPU {temp}°C (<70°C)"
            else:
                return "FAIL", f"CPU {temp}°C (>=70°C)"
        except (IndexError, ValueError):
            return "WARN", f"Temperatur konnte nicht gelesen werden: {out}"
    return "WARN", f"Unerwartetes Temperaturformat: {out}"

def check_qdrant():
    """Prüft Qdrant Health Endpoint."""
    try:
        req = urllib.request.Request(QDRANT_URL, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                return "PASS", "Qdrant health OK (200)"
            else:
                return "FAIL", f"Qdrant health status {response.status}"
    except urllib.error.URLError as e:
        return "FAIL", f"Qdrant nicht erreichbar: {e.reason}"
    except Exception as e:
        return "WARN", f"Qdrant-Check Fehler: {e}"

def check_event_log():
    """Prüft letzte 50 Zeilen der event_trace.log auf ERROR."""
    if not EVENT_TRACE.exists():
        return "WARN", f"Logdatei nicht gefunden: {EVENT_TRACE}"
    try:
        # Letzte 50 Zeilen lesen
        with open(EVENT_TRACE, 'r') as f:
            lines = f.readlines()[-50:]
        error_lines = [l.strip() for l in lines if "ERROR" in l.upper()]
        if error_lines:
            return "FAIL", f"{len(error_lines)} ERROR(s) in letzten 50 Zeilen"
        else:
            return "PASS", "Keine ERRORs in letzten 50 Zeilen"
    except Exception as e:
        return "WARN", f"Log-Check fehlgeschlagen: {e}"

def main():
    """Hauptfunktion für --auto Modus."""
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # Checks durchführen
        checks = [
            ("Service", check_service()),
            ("RAM", check_ram()),
            ("CPU Temp", check_temp()),
            ("Qdrant", check_qdrant()),
            ("Event Log", check_event_log()),
        ]
        
        # Gesamtstatus: FAIL > WARN > PASS
        status_order = {"FAIL": 2, "WARN": 1, "PASS": 0}
        overall = "PASS"
        for name, (status, _) in checks:
            if status_order[status] > status_order[overall]:
                overall = status
        
        # Ergebnis-Dict
        result = {
            "timestamp": time.time(),
            "overall": overall,
            "checks": {
                name: {"status": status, "message": msg}
                for name, (status, msg) in checks
            }
        }
        
        # JSON atomic speichern (NEVER #6)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=LOG_DIR, delete=False, suffix='.tmp') as tf:
            json.dump(result, tf, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, AUDIT_JSON)
        
        # Kurze Ausgabe für Terminal
        print(f"MOLOCH AUDIT --auto")
        print(f"Gesamtstatus: {overall}")
        for name, (status, msg) in checks:
            print(f"  {name}: {status} - {msg}")
        
        # Exit-Code für Script-Nutzung
        sys.exit(0 if overall == "PASS" else 1)
    else:
        print("Nutzung: python3 moloch_audit.py --auto")
        sys.exit(1)

if __name__ == "__main__":
    main()
