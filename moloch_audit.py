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
STATUS_JSON = Path("/dev/shm/moloch_status.json")

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


# ---------------------------------------------------------------------------
# KATEGORIE: SUBSYSTEMS
# Prüft ob alle Kern-Subsysteme geladen und aktiv sind
# ---------------------------------------------------------------------------

def _load_status_json():
    """Lädt /dev/shm/moloch_status.json, gibt Dict oder None zurück."""
    try:
        if not STATUS_JSON.exists():
            return None
        with open(STATUS_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _check_npu_worker(worker_name: str):
    """Hilfsfunktion: Prüft ob ein NPU-Worker im Status-JSON als running gemeldet wird."""
    data = _load_status_json()
    if data is None:
        return "FAIL", f"Status-JSON nicht lesbar ({STATUS_JSON})"
    npu_workers = data.get("npu_workers", {})
    if not isinstance(npu_workers, dict):
        return "FAIL", "npu_workers kein Dict im Status-JSON"
    if worker_name not in npu_workers:
        return "FAIL", f"{worker_name} nicht in npu_workers vorhanden"
    worker_data = npu_workers[worker_name]
    if isinstance(worker_data, dict):
        running = worker_data.get("running", False)
    else:
        # Einfaches bool-Format
        running = bool(worker_data)
    if running:
        return "PASS", f"{worker_name} ist running"
    else:
        return "FAIL", f"{worker_name} vorhanden aber running=False"

def check_activity_worker_loaded():
    """Prüft ob ActivityWorker im Status-JSON als running gemeldet wird."""
    return _check_npu_worker("ActivityWorker")

def check_depth_worker_loaded():
    """Prüft ob DepthWorker im Status-JSON als running gemeldet wird."""
    return _check_npu_worker("DepthWorker")

def check_yoloworld_worker_loaded():
    """Prüft ob YOLOWorldWorker im Status-JSON als running gemeldet wird."""
    return _check_npu_worker("YOLOWorldWorker")

def check_person_attr_worker_loaded():
    """Prüft ob PersonAttrWorker im Status-JSON als running gemeldet wird."""
    return _check_npu_worker("PersonAttrWorker")

def check_unconscious_engine_ticking():
    """Prüft ob UnconsciousEngine tickt: /dev/shm/moloch_impulse.json muss existieren
    und jünger als 60 Sekunden sein (Engine tickt alle 10s)."""
    impulse_path = Path("/dev/shm/moloch_impulse.json")
    if not impulse_path.exists():
        return "FAIL", "moloch_impulse.json nicht vorhanden — UnconsciousEngine läuft nicht"
    try:
        age_s = time.time() - impulse_path.stat().st_mtime
        if age_s <= 60:
            return "PASS", f"moloch_impulse.json aktuell (Alter: {age_s:.0f}s)"
        else:
            return "FAIL", f"moloch_impulse.json veraltet ({age_s:.0f}s > 60s) — Engine tickt nicht"
    except Exception as e:
        return "WARN", f"Konnte moloch_impulse.json nicht prüfen: {e}"

def check_event_bus_alive():
    """Prüft ob EventBus Events schreibt: logs/events/ muss existieren
    und mindestens eine Datei jünger als 24h enthalten."""
    events_dir = Path.home() / "moloch" / "logs" / "events"
    if not events_dir.exists():
        return "FAIL", "logs/events/ Verzeichnis existiert nicht — EventBus schreibt keine Events"
    try:
        cutoff = time.time() - 86400  # 24 Stunden
        recent_files = [f for f in events_dir.iterdir()
                        if f.is_file() and f.stat().st_mtime > cutoff]
        if recent_files:
            return "PASS", f"{len(recent_files)} Event-Datei(en) jünger als 24h in logs/events/"
        else:
            return "FAIL", "Keine Event-Datei jünger als 24h — EventBus inaktiv"
    except Exception as e:
        return "WARN", f"events/-Verzeichnis nicht lesbar: {e}"

# Gültige Mood-Werte laut Persönlichkeits-Architektur
VALID_MOODS = {"calm", "focused", "alert", "agitated", "euphoric", "dark"}

def check_mood_engine_state():
    """Prüft ob mood_state im Status-JSON existiert und ein gültiger Wert ist."""
    data = _load_status_json()
    if data is None:
        return "FAIL", f"Status-JSON nicht lesbar ({STATUS_JSON})"
    mood = data.get("mood_state")
    if mood is None:
        return "FAIL", "mood_state fehlt im Status-JSON"
    if mood in VALID_MOODS:
        return "PASS", f"mood_state='{mood}' (gültig)"
    else:
        return "WARN", f"mood_state='{mood}' — unbekannter Wert (erwartet: {sorted(VALID_MOODS)})"

def check_llm_bridge_reachable():
    """Prüft ob hailo-ollama auf Port 8000 erreichbar ist.
    WARN statt FAIL — Service kann optional laufen."""
    try:
        req = urllib.request.Request("http://localhost:8000/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                return "PASS", "hailo-ollama Port 8000 erreichbar (200)"
            else:
                return "WARN", f"hailo-ollama antwortet mit Status {response.status}"
    except urllib.error.URLError as e:
        return "WARN", f"hailo-ollama nicht erreichbar (Port 8000): {e.reason}"
    except Exception as e:
        return "WARN", f"LLM-Bridge-Check Fehler: {e}"

def check_decision_engine_initialized():
    """Prüft ob decision_engine Eintrag im Status-JSON existiert."""
    data = _load_status_json()
    if data is None:
        return "FAIL", f"Status-JSON nicht lesbar ({STATUS_JSON})"
    if "decision_engine" in data:
        return "PASS", "decision_engine im Status-JSON vorhanden"
    else:
        return "FAIL", "decision_engine fehlt im Status-JSON — DecisionEngine nicht initialisiert"

def check_core_integrator_ticking():
    """Prüft ob tension im Status-JSON existiert und ein numerischer Wert im
    Bereich -1.0 bis 1.0 ist. dominance wird geprüft wenn vorhanden (optional).
    Tatsächlicher tension-Bereich laut Status-JSON: -1.0 bis 1.0."""
    data = _load_status_json()
    if data is None:
        return "FAIL", f"Status-JSON nicht lesbar ({STATUS_JSON})"
    tension = data.get("tension")
    if tension is None:
        return "FAIL", "tension fehlt im Status-JSON — CoreIntegrator tickt nicht"
    if not isinstance(tension, (int, float)):
        return "FAIL", f"tension kein numerischer Wert ({type(tension).__name__})"
    if not (-1.0 <= tension <= 1.0):
        return "FAIL", f"tension={tension:.3f} außerhalb [-1.0, 1.0]"
    # dominance ist optional — falls vorhanden auch prüfen
    dominance = data.get("dominance")
    if dominance is not None:
        if not isinstance(dominance, (int, float)):
            return "WARN", f"dominance kein numerischer Wert ({type(dominance).__name__})"
        if not (-1.0 <= dominance <= 1.0):
            return "WARN", f"dominance={dominance:.3f} außerhalb [-1.0, 1.0]"
        return "PASS", f"tension={tension:.3f}, dominance={dominance:.3f} (beide im Bereich)"
    return "PASS", f"tension={tension:.3f} im Bereich [-1.0, 1.0] (dominance nicht im Status-JSON)"


def main():
    """Hauptfunktion für --auto Modus."""
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        # --- Basis-Checks ---
        checks = [
            ("Service", check_service()),
            ("RAM", check_ram()),
            ("CPU Temp", check_temp()),
            ("Qdrant", check_qdrant()),
            ("Event Log", check_event_log()),
        ]

        # --- Kategorie: SUBSYSTEMS ---
        subsystem_checks = [
            ("SUBSYSTEMS/ActivityWorker",       check_activity_worker_loaded()),
            ("SUBSYSTEMS/DepthWorker",           check_depth_worker_loaded()),
            ("SUBSYSTEMS/YOLOWorldWorker",        check_yoloworld_worker_loaded()),
            ("SUBSYSTEMS/PersonAttrWorker",      check_person_attr_worker_loaded()),
            ("SUBSYSTEMS/UnconsciousEngine",     check_unconscious_engine_ticking()),
            ("SUBSYSTEMS/EventBus",              check_event_bus_alive()),
            ("SUBSYSTEMS/MoodEngine",            check_mood_engine_state()),
            ("SUBSYSTEMS/LLMBridge",             check_llm_bridge_reachable()),
            ("SUBSYSTEMS/DecisionEngine",        check_decision_engine_initialized()),
            ("SUBSYSTEMS/CoreIntegrator",        check_core_integrator_ticking()),
        ]

        all_checks = checks + subsystem_checks

        # Gesamtstatus: FAIL > WARN > PASS
        status_order = {"FAIL": 2, "WARN": 1, "PASS": 0}
        overall = "PASS"
        for name, (status, _) in all_checks:
            if status_order[status] > status_order[overall]:
                overall = status

        # Ergebnis-Dict
        result = {
            "timestamp": time.time(),
            "overall": overall,
            "checks": {
                name: {"status": status, "message": msg}
                for name, (status, msg) in all_checks
            }
        }

        # JSON atomic speichern (NEVER #6)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=LOG_DIR, delete=False, suffix='.tmp') as tf:
            json.dump(result, tf, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, AUDIT_JSON)

        # Ausgabe: Basis-Checks
        print("MOLOCH AUDIT --auto")
        print(f"Gesamtstatus: {overall}")
        print("--- BASIS ---")
        for name, (status, msg) in checks:
            print(f"  {name}: {status} - {msg}")

        # Ausgabe: SUBSYSTEMS
        print("--- SUBSYSTEMS ---")
        for name, (status, msg) in subsystem_checks:
            short = name.replace("SUBSYSTEMS/", "")
            print(f"  {short}: {status} - {msg}")

        # Exit-Code für Script-Nutzung
        sys.exit(0 if overall == "PASS" else 1)
    else:
        print("Nutzung: python3 moloch_audit.py --auto")
        sys.exit(1)

if __name__ == "__main__":
    main()
