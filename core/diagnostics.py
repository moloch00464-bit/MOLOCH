#!/usr/bin/env python3
"""
M.O.L.O.C.H. Self-Diagnostics API
====================================

Sammelt Systemzustand aus allen Quellen und erkennt Probleme automatisch.

Datenquellen:
  - /dev/shm/moloch_status.json (IPC Status)
  - homeostasis.get_state() (RAM, CPU, FPS, Disk)
  - action_bridge.get_status() (FSM State)
  - /proc, /sys (Hardware-Metriken)

Zwei Hauptfunktionen:
  - collect_diagnostics() → Kompletter Systemzustand als Dict
  - self_diagnose() → Probleme erkennen, Warnungen als Liste

HTTP-Endpoint: /moloch/diagnostics auf Port 5000 (Daemon-Thread)
"""

import json
import logging
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MolochDiagnostics")

# Schwellwerte fuer Problemerkennung
DIAG_THRESHOLDS = {
    "fps_min": 15.0,
    "ram_percent_max": 85.0,
    "cpu_temp_max": 70.0,
    "tension_max": 0.9,
    "tension_max_duration_s": 60.0,
}

# Tension-Tracking (fuer Dauer-Check)
_tension_high_since: Optional[float] = None
_tension_lock = threading.Lock()


def collect_diagnostics() -> Dict[str, Any]:
    """Kompletter Systemzustand aus allen Quellen.

    Liest: moloch_status.json, homeostasis, action_bridge, /proc, /sys.
    Ziel: < 50ms, keine blockierenden Calls.
    """
    diag: Dict[str, Any] = {}

    # 1. Status-JSON lesen (Hauptquelle, bereits von Service geschrieben)
    status = _read_status_json()

    # 2. Hardware-Metriken direkt aus /proc und /sys
    hw = _read_hardware()
    diag["fps"] = _extract_fps(status)
    diag["cpu_temp"] = hw["cpu_temp"]
    diag["ram_mb"] = hw["ram_used_mb"]
    diag["ram_percent"] = hw["ram_percent"]
    diag["luefter_stufe"] = hw["fan_level"]
    diag["disk_free_gb"] = hw["disk_free_gb"]
    diag["uptime"] = hw["uptime"]
    diag["thread_count"] = threading.active_count()

    # 3. NPU Status
    diag["npu_status"] = "aktiv" if os.path.exists("/dev/hailo0") else "offline"

    # 4. Core-State (Tension, Dominance, Mood, Zone)
    core = status.get("core", {})
    diag["tension"] = core.get("tension", 0.0)
    diag["dominance"] = core.get("dominance", 0.5)
    diag["mood"] = _get_mood(status)
    diag["personality_zone"] = status.get("personality_mode", "unknown")

    # 5. Action Bridge State
    bridge = status.get("bridge", {})
    diag["bridge_state"] = bridge.get("state", "unknown")

    # 6. Face Detection
    diag["face_id"] = status.get("face_id", None)
    diag["face_similarity"] = status.get("face_similarity", 0.0)
    diag["person_detected"] = status.get("person_detected", False)

    # 7. Aktive Modelle
    diag["aktive_modelle"] = status.get("active_models", [])

    # 8. Event Bus
    bus_stats = status.get("bus_stats", {})
    diag["event_bus_subscribers"] = bus_stats.get("subscribers", 0)

    # 9. Pipeline
    diag["pipeline_alive"] = status.get("pipeline_alive", False)

    return diag


def self_diagnose() -> List[str]:
    """Probleme erkennen und als Warnungsliste zurueckgeben.

    Prueft:
      - FPS unter 15
      - RAM ueber 85%
      - CPU-Temp ueber 70
      - Face unbekannt obwohl Person da
      - Tension ueber 0.9 laenger als 60s
      - NPU offline
      - Pipeline tot
    """
    global _tension_high_since

    diag = collect_diagnostics()
    warnungen: List[str] = []

    # FPS zu niedrig
    fps = diag.get("fps", 0.0)
    if 0 < fps < DIAG_THRESHOLDS["fps_min"]:
        warnungen.append(f"FPS niedrig: {fps:.1f} (Schwelle: {DIAG_THRESHOLDS['fps_min']})")

    # RAM zu hoch
    ram = diag.get("ram_percent", 0.0)
    if ram > DIAG_THRESHOLDS["ram_percent_max"]:
        warnungen.append(f"RAM kritisch: {ram:.1f}% (Schwelle: {DIAG_THRESHOLDS['ram_percent_max']}%)")

    # CPU zu heiss
    temp = diag.get("cpu_temp", 0.0)
    if temp > DIAG_THRESHOLDS["cpu_temp_max"]:
        warnungen.append(f"CPU heiss: {temp:.1f}C (Schwelle: {DIAG_THRESHOLDS['cpu_temp_max']}C)")

    # Face unbekannt obwohl Person da
    if diag.get("person_detected") and not diag.get("face_id"):
        warnungen.append("Person erkannt aber kein Gesicht identifiziert")

    # Tension zu hoch zu lange
    tension = diag.get("tension", 0.0)
    now = time.time()
    with _tension_lock:
        if tension > DIAG_THRESHOLDS["tension_max"]:
            if _tension_high_since is None:
                _tension_high_since = now
            elif (now - _tension_high_since) > DIAG_THRESHOLDS["tension_max_duration_s"]:
                dauer = int(now - _tension_high_since)
                warnungen.append(
                    f"Tension kritisch hoch: {tension:.2f} seit {dauer}s "
                    f"(Schwelle: >{DIAG_THRESHOLDS['tension_max']} fuer "
                    f">{DIAG_THRESHOLDS['tension_max_duration_s']:.0f}s)"
                )
        else:
            _tension_high_since = None

    # NPU offline
    if diag.get("npu_status") != "aktiv":
        warnungen.append("NPU (Hailo) offline!")

    # Pipeline tot
    if not diag.get("pipeline_alive"):
        warnungen.append("Vision-Pipeline nicht aktiv!")

    return warnungen


def get_diagnostics_text() -> str:
    """Diagnose als lesbaren Text fuer Chat/TTS zurueckgeben."""
    diag = collect_diagnostics()
    warnungen = self_diagnose()

    teile = []
    teile.append(f"FPS: {diag['fps']:.1f}")
    teile.append(f"CPU: {diag['cpu_temp']:.0f}C")
    teile.append(f"RAM: {diag['ram_mb']}MB ({diag['ram_percent']:.0f}%)")
    teile.append(f"Luefter: Stufe {diag['luefter_stufe']}")
    teile.append(f"NPU: {diag['npu_status']}")
    teile.append(f"Tension: {diag['tension']:.2f}")
    teile.append(f"Dominance: {diag['dominance']:.2f}")
    teile.append(f"Stimmung: {diag['mood']}")
    teile.append(f"Bridge: {diag['bridge_state']}")
    teile.append(f"Uptime: {diag['uptime']}")

    if diag.get("face_id"):
        teile.append(f"Gesicht: {diag['face_id']} ({diag['face_similarity']:.2f})")
    elif diag.get("person_detected"):
        teile.append("Person da, Gesicht unbekannt")

    text = "Systemstatus: " + ", ".join(teile) + "."

    if warnungen:
        text += " WARNUNGEN: " + "; ".join(warnungen) + "."
    else:
        text += " Alles im gruenen Bereich."

    return text


# =========================================================================
# Hilfsfunktionen (alle < 5ms, kein Blocking)
# =========================================================================

def _read_status_json() -> Dict[str, Any]:
    """moloch_status.json aus Shared Memory lesen."""
    try:
        with open("/dev/shm/moloch_status.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_hardware() -> Dict[str, Any]:
    """Hardware-Metriken direkt aus /proc und /sys."""
    hw: Dict[str, Any] = {
        "cpu_temp": 0.0,
        "ram_used_mb": 0,
        "ram_percent": 0.0,
        "fan_level": 0,
        "disk_free_gb": 0.0,
        "uptime": "0m",
    }

    # CPU-Temperatur
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            hw["cpu_temp"] = round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_mb = meminfo.get("MemTotal", 1) // 1024
            avail_mb = meminfo.get("MemAvailable", meminfo.get("MemFree", total_mb)) // 1024
            used_mb = total_mb - avail_mb
            hw["ram_used_mb"] = used_mb
            hw["ram_percent"] = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
    except Exception:
        pass

    # Luefter
    try:
        with open("/sys/class/thermal/cooling_device0/cur_state") as f:
            hw["fan_level"] = int(f.read().strip())
    except Exception:
        pass

    # Disk
    try:
        st = os.statvfs(os.path.expanduser("~/moloch"))
        hw["disk_free_gb"] = round((st.f_bavail * st.f_frsize) / (1024**3), 1)
    except Exception:
        pass

    # Uptime
    try:
        with open("/proc/uptime") as f:
            uptime_sec = float(f.read().split()[0])
            hours = int(uptime_sec // 3600)
            mins = int((uptime_sec % 3600) // 60)
            hw["uptime"] = f"{hours}h{mins:02d}m" if hours > 0 else f"{mins}m"
    except Exception:
        pass

    return hw


def _extract_fps(status: Dict[str, Any]) -> float:
    """FPS aus Status-JSON extrahieren."""
    fps_data = status.get("fps", {})
    if isinstance(fps_data, dict):
        return fps_data.get("current", fps_data.get("avg", 0.0))
    return 0.0


def _get_mood(status: Dict[str, Any]) -> str:
    """Aktuelle Stimmung/Mood aus Status ableiten."""
    # MoodEngine Mood (falls vorhanden)
    core = status.get("core", {})
    mood = core.get("mood", None)
    if mood:
        return mood

    # Fallback: aus Tension/Zone ableiten
    tension = core.get("tension", 0.0)
    zone = status.get("personality_mode", "guardian")
    if tension > 0.8:
        return "angespannt"
    elif tension > 0.5:
        return "wachsam"
    elif zone == "guardian":
        return "entspannt"
    elif zone == "shadow":
        return "dunkel"
    else:
        return "neutral"


# =========================================================================
# HTTP-Server (leichtgewichtig, stdlib, kein Flask)
# =========================================================================

class _DiagnosticsHandler(BaseHTTPRequestHandler):
    """HTTP-Handler fuer /moloch/diagnostics Endpoint."""

    def do_GET(self):
        if self.path == "/moloch/diagnostics":
            data = collect_diagnostics()
            data["warnungen"] = self_diagnose()
            data["timestamp"] = time.time()
            body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/moloch/health":
            # Schneller Health-Check (nur OK/WARN)
            warnungen = self_diagnose()
            ok = len(warnungen) == 0
            data = {"status": "ok" if ok else "warn", "warnungen": warnungen}
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """HTTP-Logging unterdruecken (zu viel Noise)."""
        pass


_server_thread: Optional[threading.Thread] = None
_server_instance: Optional[HTTPServer] = None


def start_diagnostics_server(port: int = 5000):
    """HTTP-Server als Daemon-Thread starten."""
    global _server_thread, _server_instance

    if _server_thread and _server_thread.is_alive():
        logger.info("[DIAG] Server laeuft bereits")
        return

    try:
        _server_instance = HTTPServer(("0.0.0.0", port), _DiagnosticsHandler)
        _server_thread = threading.Thread(
            target=_server_instance.serve_forever,
            daemon=True,
            name="DiagnosticsHTTP",
        )
        _server_thread.start()
        logger.info(f"[DIAG] HTTP-Server gestartet auf Port {port}")
    except Exception as e:
        logger.error(f"[DIAG] HTTP-Server Start fehlgeschlagen: {e}")


def stop_diagnostics_server():
    """HTTP-Server sauber stoppen."""
    global _server_instance
    if _server_instance:
        _server_instance.shutdown()
        _server_instance = None
        logger.info("[DIAG] HTTP-Server gestoppt")
