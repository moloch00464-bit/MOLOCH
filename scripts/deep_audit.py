#!/usr/bin/env python3
"""
M.O.L.O.C.H. Deep Audit — Innenleben-Monitor ueber mehrere Stunden.

Nimmt alle 30 Minuten einen Snapshot aller internen Systeme:
  - Event Bus (Events/min, Subscriber, Deduplizierung)
  - Perception State (PFrame Luecken, FPS Stabilitaet)
  - Action Bridge FSM (State-Verteilung, Haengenbleiben)
  - NPU Pipeline (FPS, Model Health, Temperatur)
  - Face Recognition (Trefferquote, Fehlversuche)
  - PTZ Tracker (Smooth vs. ruckartig, Befehle/min)
  - Whisper (Aufrufe, Latenz)
  - Hardware (CPU, RAM, Disk, Temp)

Ergebnis: ~/moloch/logs/deep_audit_YYYYMMDD.log

Autor: Deep Audit Agent (Claude Code)
Datum: 2026-03-10
"""

import json
import os
import sys
import time
import subprocess
import signal
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict

# === KONFIGURATION ===
AUDIT_DURATION_H = 3          # Gesamtdauer in Stunden
SNAPSHOT_INTERVAL_MIN = 30    # Intervall zwischen Snapshots
STATUS_JSON = "/dev/shm/moloch_status.json"
EVENTS_DIR = Path.home() / "moloch/logs/events"
LOG_DIR = Path.home() / "moloch/logs"

# === GLOBALS ===
_running = True
_snapshots = []


def signal_handler(sig, frame):
    """Graceful shutdown bei Ctrl+C."""
    global _running
    print("\n[AUDIT] Signal empfangen, schreibe Abschlussbericht...")
    _running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =====================================================================
#  DATENSAMMLER
# =====================================================================

def read_status_json() -> dict:
    """Liest /dev/shm/moloch_status.json — das Herzstueck."""
    try:
        with open(STATUS_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def get_hardware_metrics() -> dict:
    """CPU-Temp, RAM, Disk, Load."""
    metrics = {}

    # CPU Temperatur
    try:
        temp = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        metrics["cpu_temp_c"] = round(int(temp) / 1000, 1)
    except:
        metrics["cpu_temp_c"] = -1

    # NPU Temperatur (aus hailo_temp Sensor)
    try:
        result = subprocess.run(
            ["cat", "/sys/class/hwmon/hwmon4/temp1_input"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            metrics["npu_temp_c"] = round(int(result.stdout.strip()) / 1000, 1)
        else:
            metrics["npu_temp_c"] = -1
    except:
        metrics["npu_temp_c"] = -1

    # RAM
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:", "Buffers:", "Cached:"):
                    mem[parts[0].rstrip(":")] = int(parts[1])
            total = mem.get("MemTotal", 0)
            available = mem.get("MemAvailable", 0)
            metrics["ram_total_mb"] = total // 1024
            metrics["ram_used_mb"] = (total - available) // 1024
            metrics["ram_percent"] = round((total - available) / total * 100, 1) if total else 0
    except:
        metrics["ram_total_mb"] = 0
        metrics["ram_used_mb"] = 0
        metrics["ram_percent"] = 0

    # Moloch Service RSS
    try:
        result = subprocess.run(
            ["bash", "-c", "ps -o rss= -p $(pgrep -f moloch_service) 2>/dev/null | head -1"],
            capture_output=True, text=True, timeout=3
        )
        rss = result.stdout.strip()
        metrics["moloch_rss_mb"] = int(rss) // 1024 if rss else -1
    except:
        metrics["moloch_rss_mb"] = -1

    # Load Average
    try:
        load = os.getloadavg()
        metrics["load_1m"] = round(load[0], 2)
        metrics["load_5m"] = round(load[1], 2)
    except:
        metrics["load_1m"] = 0
        metrics["load_5m"] = 0

    # Disk
    try:
        st = os.statvfs("/home/molochzuhause/moloch")
        metrics["disk_code_free_gb"] = round(st.f_bavail * st.f_frsize / (1024**3), 1)
    except:
        metrics["disk_code_free_gb"] = -1

    return metrics


def get_event_bus_stats(status: dict) -> dict:
    """Event Bus Metriken aus Status-JSON extrahieren."""
    bus = status.get("bus_stats", {})
    return {
        "total_published": bus.get("total_published", 0),
        "total_delivered": bus.get("total_delivered", 0),
        "total_deduplicated": bus.get("total_deduplicated", 0),
        "total_errors": bus.get("total_errors", 0),
        "total_silenced": bus.get("total_silenced", 0),
        "dedup_cache_size": bus.get("dedup_cache_size", 0),
        "subscriber_count": len(bus.get("subscribers", {})),
        "silence_level": status.get("silence_level", 0),
    }


def get_perception_state(status: dict) -> dict:
    """Perception State aus Status-JSON."""
    perception = status.get("perception", {})
    return {
        "person_detected": status.get("person_detected", False),
        "face_detected": status.get("face_detected", False),
        "face_id": status.get("face_id"),
        "face_confidence": round(status.get("face_confidence", 0), 3),
        "face_similarity": round(status.get("face_similarity", 0), 3),
        "npu_stage": perception.get("npu_stage", status.get("npu_stage", "unknown")),
        "mode": status.get("mode", "unknown"),
        "pipeline_alive": status.get("pipeline_alive", False),
        "active_models": status.get("active_models", []),
    }


def get_fps_data(status: dict) -> dict:
    """FPS aus Status-JSON."""
    fps = status.get("fps", {})
    return {
        "total": round(fps.get("total", 0), 1),
        "scrfd": round(fps.get("scrfd", 0), 1),
        "arcface": round(fps.get("arcface", 0), 1),
        "yolov8m": round(fps.get("yolov8m", 0), 1),
    }


def get_bridge_state(status: dict) -> dict:
    """Action Bridge FSM aus Status-JSON."""
    bridge = status.get("bridge", {})
    return {
        "state": bridge.get("state", "unknown"),
        "prev_state": bridge.get("prev_state", "unknown"),
        "state_age_s": round(bridge.get("state_age_s", 0), 1),
        "person_detected": bridge.get("person_detected", False),
        "face_confirmed": bridge.get("face_confirmed", False),
        "owner_detected": bridge.get("owner_detected", False),
        "owner_name": bridge.get("owner_name"),
        "decisions": bridge.get("decisions", 0),
    }


def get_ptz_state(status: dict) -> dict:
    """PTZ Tracker aus Status-JSON."""
    ptz = status.get("ptz", {})
    return {
        "current_pan": round(ptz.get("current_pan", 0), 1),
        "current_tilt": round(ptz.get("current_tilt", 0), 1),
        "tracker_state": ptz.get("tracker_state", "unknown"),
        "tracking_moves": ptz.get("tracking_moves", 0),
        "search_moves": ptz.get("search_moves", 0),
        "ptz_velocity": round(ptz.get("ptz_velocity", 0), 1),
        "ptz_restless_score": round(ptz.get("ptz_restless_score", 0), 3),
    }


def get_core_state(status: dict) -> dict:
    """Core Integrator (Personality) aus Status-JSON."""
    core = status.get("core", {})
    return {
        "tension": round(core.get("tension", 0), 3),
        "dominance": round(core.get("dominance", 0), 3),
        "zone": core.get("zone", "unknown"),
    }


def get_audio_state(status: dict) -> dict:
    """Audio/Voice aus Status-JSON."""
    audio = status.get("audio", {})
    wifi_mic = status.get("wifi_mic", {})
    return {
        "mic_gain": audio.get("mic_gain", 0),
        "noise_gate_db": audio.get("noise_gate_db", 0),
        "level": round(audio.get("level", 0), 3),
        "wifi_mic_connected": wifi_mic.get("connected_16k", False) if wifi_mic else False,
        "wifi_mic_packets": wifi_mic.get("packets_recv_16k", 0) if wifi_mic else 0,
    }


def count_recent_events(minutes: int = 30) -> dict:
    """Zaehlt Events aus JSONL-Logfiles der letzten N Minuten."""
    today = datetime.now().strftime("%Y-%m-%d")
    event_file = EVENTS_DIR / f"events_{today}.jsonl"

    if not event_file.exists():
        return {"_error": f"Keine Event-Datei: {event_file}"}

    cutoff = time.time() - (minutes * 60)
    event_types = Counter()
    total = 0
    errors = 0

    try:
        # Nur letzte Zeilen lesen (Performance bei grossen Dateien)
        result = subprocess.run(
            ["tail", "-5000", str(event_file)],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
                ts = evt.get("timestamp", 0)
                if ts >= cutoff:
                    total += 1
                    event_types[evt.get("event_type", "unknown")] += 1
            except json.JSONDecodeError:
                errors += 1
    except Exception as e:
        return {"_error": str(e)}

    events_per_min = round(total / minutes, 1) if minutes > 0 else 0

    return {
        "total_in_window": total,
        "events_per_min": events_per_min,
        "parse_errors": errors,
        "top_types": dict(event_types.most_common(10)),
    }


def count_face_matches_in_journal(minutes: int = 30) -> dict:
    """Zaehlt FACE-MATCH Ergebnisse aus journalctl."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "moloch", "--no-pager",
             "--since", f"{minutes} min ago",
             "--grep", "FACE-MATCH"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        total = len(lines)
        matches = sum(1 for l in lines if "✓" in l)
        no_match = sum(1 for l in lines if "✗" in l or "kein Match" in l)

        return {
            "total_attempts": total,
            "matches": matches,
            "no_match": no_match,
            "match_rate": round(matches / total * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"_error": str(e)}


def count_whisper_calls(minutes: int = 30) -> dict:
    """Zaehlt Whisper-Transkriptionen aus journalctl."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "moloch", "--no-pager",
             "--since", f"{minutes} min ago",
             "--grep", "Transkription"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]

        latencies = []
        for l in lines:
            # Format: [VOICE] Transkription (1250ms): Text
            if "ms)" in l:
                try:
                    ms_str = l.split("(")[1].split("ms)")[0]
                    latencies.append(float(ms_str))
                except:
                    pass

        return {
            "calls": len(lines),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
            "max_latency_ms": round(max(latencies), 0) if latencies else 0,
            "min_latency_ms": round(min(latencies), 0) if latencies else 0,
        }
    except Exception as e:
        return {"_error": str(e)}


def count_errors_in_journal(minutes: int = 30) -> dict:
    """Zaehlt Errors/Warnings aus journalctl."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "moloch", "--no-pager",
             "--since", f"{minutes} min ago",
             "--priority", "err"],
            capture_output=True, text=True, timeout=10
        )
        error_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]

        result2 = subprocess.run(
            ["journalctl", "-u", "moloch", "--no-pager",
             "--since", f"{minutes} min ago",
             "--grep", "Traceback|Exception|ERROR"],
            capture_output=True, text=True, timeout=10
        )
        exception_lines = [l for l in result2.stdout.strip().split("\n") if l.strip()]

        return {
            "errors": len(error_lines),
            "exceptions": len(exception_lines),
        }
    except Exception as e:
        return {"_error": str(e)}


# =====================================================================
#  SNAPSHOT
# =====================================================================

def take_snapshot(snapshot_num: int, minutes_since_start: int) -> dict:
    """Nimmt einen kompletten Innenleben-Snapshot."""
    ts = datetime.now()
    status = read_status_json()

    snapshot = {
        "snapshot_num": snapshot_num,
        "timestamp": ts.isoformat(),
        "minutes_since_start": minutes_since_start,

        # Kern-Systeme
        "hardware": get_hardware_metrics(),
        "fps": get_fps_data(status),
        "event_bus": get_event_bus_stats(status),
        "perception": get_perception_state(status),
        "bridge": get_bridge_state(status),
        "ptz": get_ptz_state(status),
        "core": get_core_state(status),
        "audio": get_audio_state(status),

        # Analyse aus Logs (letzte 30 min bzw. seit Start)
        "events_recent": count_recent_events(min(minutes_since_start, 30) or 1),
        "face_matches": count_face_matches_in_journal(min(minutes_since_start, 30) or 1),
        "whisper": count_whisper_calls(min(minutes_since_start, 30) or 1),
        "errors": count_errors_in_journal(min(minutes_since_start, 30) or 1),
    }

    return snapshot


# =====================================================================
#  FORMATIERUNG
# =====================================================================

def format_snapshot(snap: dict) -> str:
    """Formatiert einen Snapshot als lesbaren Text-Block."""
    lines = []
    ts = snap["timestamp"]
    num = snap["snapshot_num"]
    mins = snap["minutes_since_start"]

    lines.append(f"\n{'='*72}")
    lines.append(f"  SNAPSHOT #{num}  |  {ts}  |  +{mins} min")
    lines.append(f"{'='*72}")

    # Hardware
    hw = snap["hardware"]
    lines.append(f"\n  HARDWARE:")
    lines.append(f"    CPU: {hw['cpu_temp_c']}°C  |  NPU: {hw['npu_temp_c']}°C")
    lines.append(f"    RAM: {hw['ram_used_mb']}/{hw['ram_total_mb']} MB ({hw['ram_percent']}%)")
    lines.append(f"    Moloch RSS: {hw['moloch_rss_mb']} MB  |  Load: {hw['load_1m']}/{hw['load_5m']}")
    lines.append(f"    Disk (Code SSD): {hw['disk_code_free_gb']} GB frei")

    # FPS
    fps = snap["fps"]
    lines.append(f"\n  NPU PIPELINE:")
    lines.append(f"    FPS: {fps['total']}  (YOLO={fps['yolov8m']}, SCRFD={fps['scrfd']}, ArcFace={fps['arcface']})")
    perc = snap["perception"]
    lines.append(f"    Stage: {perc['npu_stage']}  |  Pipeline: {'ALIVE' if perc['pipeline_alive'] else 'DEAD!'}")
    lines.append(f"    Modelle aktiv: {', '.join(perc['active_models']) if perc['active_models'] else 'KEINE'}")

    # Perception
    lines.append(f"\n  PERCEPTION STATE:")
    lines.append(f"    Person: {'JA' if perc['person_detected'] else 'nein'}  |  "
                 f"Face: {'JA' if perc['face_detected'] else 'nein'}  |  "
                 f"ID: {perc['face_id'] or '-'}")
    lines.append(f"    Face Confidence: {perc['face_confidence']}  |  "
                 f"Similarity: {perc['face_similarity']}")

    # Face Recognition
    fm = snap["face_matches"]
    if "_error" not in fm:
        rate_str = f"{fm['match_rate']}%" if fm['total_attempts'] > 0 else "n/a"
        lines.append(f"    Face-Matches (letzte {min(mins, 30)} min): "
                     f"{fm['matches']}/{fm['total_attempts']} = {rate_str}")
    else:
        lines.append(f"    Face-Matches: ERROR ({fm['_error']})")

    # Action Bridge
    br = snap["bridge"]
    lines.append(f"\n  ACTION BRIDGE FSM:")
    lines.append(f"    State: {br['state'].upper()} (seit {br['state_age_s']}s, vorher: {br['prev_state']})")
    lines.append(f"    Person={br['person_detected']}  Face={br['face_confirmed']}  "
                 f"Owner={br['owner_detected']} ({br['owner_name'] or '-'})")
    lines.append(f"    Decisions gesamt: {br['decisions']}")

    # PTZ
    ptz = snap["ptz"]
    lines.append(f"\n  PTZ TRACKER:")
    lines.append(f"    State: {ptz['tracker_state'].upper()}  |  "
                 f"Pan={ptz['current_pan']}°  Tilt={ptz['current_tilt']}°")
    lines.append(f"    Moves: {ptz['tracking_moves']} tracking, {ptz['search_moves']} search")
    lines.append(f"    Velocity: {ptz['ptz_velocity']}°/s  |  "
                 f"Restless: {ptz['ptz_restless_score']}")

    # Core
    core = snap["core"]
    lines.append(f"\n  CORE / PERSONALITY:")
    lines.append(f"    Zone: {core['zone'].upper()}  |  "
                 f"Tension: {core['tension']}  |  Dominance: {core['dominance']}")

    # Event Bus
    eb = snap["event_bus"]
    lines.append(f"\n  EVENT BUS:")
    lines.append(f"    Published: {eb['total_published']}  |  "
                 f"Delivered: {eb['total_delivered']}  |  "
                 f"Deduplicated: {eb['total_deduplicated']}")
    lines.append(f"    Errors: {eb['total_errors']}  |  "
                 f"Silenced: {eb['total_silenced']}  |  "
                 f"Subscribers: {eb['subscriber_count']}")

    ev = snap["events_recent"]
    if "_error" not in ev:
        lines.append(f"    Events/min (letzte {min(mins, 30)} min): {ev['events_per_min']}")
        if ev["top_types"]:
            top3 = list(ev["top_types"].items())[:5]
            lines.append(f"    Top Event-Types: {', '.join(f'{t}={c}' for t, c in top3)}")
    else:
        lines.append(f"    Events: ERROR ({ev['_error']})")

    # Whisper
    wh = snap["whisper"]
    if "_error" not in wh:
        lines.append(f"\n  WHISPER STT:")
        if wh["calls"] > 0:
            lines.append(f"    Aufrufe: {wh['calls']}  |  "
                         f"Latenz: {wh['avg_latency_ms']}ms avg, "
                         f"{wh['max_latency_ms']}ms max, {wh['min_latency_ms']}ms min")
        else:
            lines.append(f"    Aufrufe: 0 (kein PTT in diesem Zeitraum)")
    else:
        lines.append(f"\n  WHISPER: ERROR ({wh['_error']})")

    # Audio
    au = snap["audio"]
    lines.append(f"\n  AUDIO:")
    lines.append(f"    WiFi-Mic: {'verbunden' if au['wifi_mic_connected'] else 'OFFLINE'}  |  "
                 f"Pakete: {au['wifi_mic_packets']}  |  Level: {au['level']}")

    # Errors
    errs = snap["errors"]
    if "_error" not in errs:
        if errs["errors"] > 0 or errs["exceptions"] > 0:
            lines.append(f"\n  ⚠ FEHLER: {errs['errors']} Errors, {errs['exceptions']} Exceptions")
        else:
            lines.append(f"\n  ✓ Keine Fehler im Zeitraum")
    else:
        lines.append(f"\n  FEHLER-CHECK: ERROR ({errs['_error']})")

    return "\n".join(lines)


def generate_final_report(snapshots: list) -> str:
    """Erzeugt den 3-Stunden-Abschlussbericht mit Trends und Empfehlungen."""
    lines = []
    lines.append(f"\n{'#'*72}")
    lines.append(f"#  M.O.L.O.C.H. DEEP AUDIT — ABSCHLUSSBERICHT")
    lines.append(f"#  Start: {snapshots[0]['timestamp']}")
    lines.append(f"#  Ende:  {snapshots[-1]['timestamp']}")
    lines.append(f"#  Snapshots: {len(snapshots)}")
    lines.append(f"{'#'*72}")

    # === TREND-ANALYSE ===
    lines.append(f"\n{'='*40}")
    lines.append(f"  TREND-ANALYSE")
    lines.append(f"{'='*40}")

    # FPS Trend
    fps_values = [s["fps"]["total"] for s in snapshots]
    fps_min = min(fps_values)
    fps_max = max(fps_values)
    fps_avg = round(sum(fps_values) / len(fps_values), 1)
    fps_stable = (fps_max - fps_min) < 3.0

    lines.append(f"\n  FPS: avg={fps_avg}, min={fps_min}, max={fps_max}")
    lines.append(f"    Stabilitaet: {'✓ STABIL' if fps_stable else '⚠ INSTABIL (Schwankung > 3 FPS)'}")

    # CPU Temp Trend
    cpu_temps = [s["hardware"]["cpu_temp_c"] for s in snapshots if s["hardware"]["cpu_temp_c"] > 0]
    if cpu_temps:
        lines.append(f"\n  CPU Temp: avg={round(sum(cpu_temps)/len(cpu_temps),1)}°C, "
                     f"min={min(cpu_temps)}°C, max={max(cpu_temps)}°C")
        if max(cpu_temps) > 70:
            lines.append(f"    ⚠ WARNUNG: CPU ueber 70°C!")
        else:
            lines.append(f"    ✓ Thermisch OK")

    # NPU Temp Trend
    npu_temps = [s["hardware"]["npu_temp_c"] for s in snapshots if s["hardware"]["npu_temp_c"] > 0]
    if npu_temps:
        lines.append(f"\n  NPU Temp: avg={round(sum(npu_temps)/len(npu_temps),1)}°C, "
                     f"min={min(npu_temps)}°C, max={max(npu_temps)}°C")

    # RAM Trend
    ram_values = [s["hardware"]["ram_percent"] for s in snapshots]
    lines.append(f"\n  RAM: avg={round(sum(ram_values)/len(ram_values),1)}%, "
                 f"min={min(ram_values)}%, max={max(ram_values)}%")
    if max(ram_values) > 85:
        lines.append(f"    ⚠ WARNUNG: RAM ueber 85%!")
    else:
        lines.append(f"    ✓ RAM OK")

    # Moloch RSS Trend
    rss_values = [s["hardware"]["moloch_rss_mb"] for s in snapshots if s["hardware"]["moloch_rss_mb"] > 0]
    if rss_values:
        rss_growth = rss_values[-1] - rss_values[0]
        lines.append(f"\n  Moloch RSS: start={rss_values[0]}MB, end={rss_values[-1]}MB, "
                     f"delta={rss_growth:+d}MB")
        if rss_growth > 50:
            lines.append(f"    ⚠ MEMORY LEAK? RSS wuchs um {rss_growth}MB")
        else:
            lines.append(f"    ✓ Kein Memory-Leak erkennbar")

    # Bridge State Verteilung
    lines.append(f"\n{'='*40}")
    lines.append(f"  BRIDGE STATE VERTEILUNG")
    lines.append(f"{'='*40}")
    state_counter = Counter(s["bridge"]["state"] for s in snapshots)
    for state, count in state_counter.most_common():
        pct = round(count / len(snapshots) * 100, 1)
        lines.append(f"    {state:20s}  {count}x ({pct}%)")

    # Bridge haengenbleiben?
    max_age = max(s["bridge"]["state_age_s"] for s in snapshots)
    lines.append(f"    Laengster State: {max_age}s")
    if max_age > 300:
        lines.append(f"    ⚠ Bridge haengt — State >5 min unveraendert!")

    # PTZ Analyse
    lines.append(f"\n{'='*40}")
    lines.append(f"  PTZ TRACKER ANALYSE")
    lines.append(f"{'='*40}")
    tracker_states = Counter(s["ptz"]["tracker_state"] for s in snapshots)
    for state, count in tracker_states.most_common():
        pct = round(count / len(snapshots) * 100, 1)
        lines.append(f"    {state:20s}  {count}x ({pct}%)")

    restless_scores = [s["ptz"]["ptz_restless_score"] for s in snapshots]
    avg_restless = round(sum(restless_scores) / len(restless_scores), 3)
    lines.append(f"    Restless Score avg: {avg_restless}")
    if avg_restless > 0.6:
        lines.append(f"    ⚠ PTZ ruckartig! Gain-Tuning pruefen (G1-T05)")
    else:
        lines.append(f"    ✓ PTZ smooth")

    total_moves = snapshots[-1]["ptz"]["tracking_moves"]
    lines.append(f"    Total Tracking Moves: {total_moves}")

    # Face Recognition Zusammenfassung
    lines.append(f"\n{'='*40}")
    lines.append(f"  FACE RECOGNITION")
    lines.append(f"{'='*40}")
    total_attempts = sum(s["face_matches"].get("total_attempts", 0) for s in snapshots
                         if "_error" not in s["face_matches"])
    total_matches = sum(s["face_matches"].get("matches", 0) for s in snapshots
                        if "_error" not in s["face_matches"])
    overall_rate = round(total_matches / total_attempts * 100, 1) if total_attempts > 0 else 0
    lines.append(f"    Gesamt: {total_matches}/{total_attempts} Treffer = {overall_rate}%")
    if total_attempts > 0 and overall_rate < 20:
        lines.append(f"    ⚠ Trefferquote unter 20% — ArcFace Threshold oder Enrollment pruefen!")
    elif total_attempts == 0:
        lines.append(f"    ℹ Kein Face-Matching stattgefunden")

    # Event Bus
    lines.append(f"\n{'='*40}")
    lines.append(f"  EVENT BUS")
    lines.append(f"{'='*40}")
    final_bus = snapshots[-1]["event_bus"]
    lines.append(f"    Total Published: {final_bus['total_published']}")
    lines.append(f"    Total Delivered: {final_bus['total_delivered']}")
    lines.append(f"    Errors: {final_bus['total_errors']}")
    if final_bus["total_errors"] > 0:
        lines.append(f"    ⚠ Event Bus Fehler aufgetreten!")

    epm_values = [s["events_recent"].get("events_per_min", 0) for s in snapshots
                  if "_error" not in s["events_recent"]]
    if epm_values:
        lines.append(f"    Events/min: avg={round(sum(epm_values)/len(epm_values),1)}, "
                     f"min={min(epm_values)}, max={max(epm_values)}")

    # Whisper
    lines.append(f"\n{'='*40}")
    lines.append(f"  WHISPER STT")
    lines.append(f"{'='*40}")
    total_whisper = sum(s["whisper"].get("calls", 0) for s in snapshots
                        if "_error" not in s["whisper"])
    latencies = [s["whisper"].get("avg_latency_ms", 0) for s in snapshots
                 if "_error" not in s["whisper"] and s["whisper"].get("calls", 0) > 0]
    lines.append(f"    Gesamtaufrufe: {total_whisper}")
    if latencies:
        lines.append(f"    Avg Latenz: {round(sum(latencies)/len(latencies))}ms")
    if total_whisper == 0:
        lines.append(f"    ℹ Whisper wurde nicht aufgerufen (kein PTT)")

    # Fehler
    lines.append(f"\n{'='*40}")
    lines.append(f"  FEHLER & WARNUNGEN")
    lines.append(f"{'='*40}")
    total_errors = sum(s["errors"].get("errors", 0) for s in snapshots
                       if "_error" not in s["errors"])
    total_exceptions = sum(s["errors"].get("exceptions", 0) for s in snapshots
                           if "_error" not in s["errors"])
    lines.append(f"    Errors gesamt: {total_errors}")
    lines.append(f"    Exceptions gesamt: {total_exceptions}")
    if total_errors == 0 and total_exceptions == 0:
        lines.append(f"    ✓ Fehlerfrei!")

    # === EMPFEHLUNGEN ===
    lines.append(f"\n{'='*40}")
    lines.append(f"  EMPFEHLUNGEN")
    lines.append(f"{'='*40}")

    recommendations = []

    if not fps_stable:
        recommendations.append("FPS instabil — GStreamer Queue-Groessen pruefen, RTSP-Verbindung checken")
    if max(cpu_temps) > 70 if cpu_temps else False:
        recommendations.append("CPU thermisch grenzwertig — Kuehlkoerper/Luefter pruefen")
    if max(ram_values) > 85:
        recommendations.append("RAM-Auslastung hoch — Module auf Memory-Leaks pruefen")
    if rss_values and (rss_values[-1] - rss_values[0]) > 50:
        recommendations.append(f"Memory Leak verdacht: RSS wuchs um {rss_values[-1] - rss_values[0]}MB")
    if max_age > 300:
        recommendations.append("Action Bridge haengt in einem State — FSM-Timeouts pruefen")
    if avg_restless > 0.6:
        recommendations.append("PTZ ruckartig — TRACKING_GAIN und MAX_STEP reduzieren (G1-T05)")
    if total_attempts > 10 and overall_rate < 20:
        recommendations.append("ArcFace Trefferquote zu niedrig — Re-Enrollment durch TAPPAS-Pipeline noetig")
    if final_bus["total_errors"] > 0:
        recommendations.append("Event Bus Fehler — Subscriber-Callbacks auf Exceptions pruefen")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"    {i}. {rec}")
    else:
        lines.append(f"    ✓ System laeuft rund — keine Aktion noetig")

    lines.append(f"\n{'#'*72}")
    lines.append(f"#  AUDIT ENDE")
    lines.append(f"{'#'*72}\n")

    return "\n".join(lines)


# =====================================================================
#  HAUPTPROGRAMM
# =====================================================================

def main():
    global _running, _snapshots

    today = datetime.now().strftime("%Y%m%d")
    logfile = LOG_DIR / f"deep_audit_{today}.log"

    total_snapshots = (AUDIT_DURATION_H * 60 // SNAPSHOT_INTERVAL_MIN) + 1
    end_time = datetime.now() + timedelta(hours=AUDIT_DURATION_H)

    # Header
    header = [
        f"\n{'#'*72}",
        f"#  M.O.L.O.C.H. DEEP AUDIT — Innenleben-Monitor",
        f"#  Start: {datetime.now().isoformat()}",
        f"#  Dauer: {AUDIT_DURATION_H}h ({total_snapshots} Snapshots, alle {SNAPSHOT_INTERVAL_MIN} min)",
        f"#  Ende geplant: {end_time.isoformat()}",
        f"#  Log: {logfile}",
        f"{'#'*72}",
    ]

    header_text = "\n".join(header)
    print(header_text)

    with open(logfile, "a") as f:
        f.write(header_text + "\n")

    # Erster Snapshot sofort
    snapshot_num = 0
    start_time = time.time()

    while _running:
        minutes_elapsed = int((time.time() - start_time) / 60)

        print(f"\n[AUDIT] Snapshot #{snapshot_num} bei +{minutes_elapsed} min ...")
        snap = take_snapshot(snapshot_num, minutes_elapsed)
        _snapshots.append(snap)

        formatted = format_snapshot(snap)
        print(formatted)

        with open(logfile, "a") as f:
            f.write(formatted + "\n")

        snapshot_num += 1

        # Sind wir fertig?
        if time.time() >= end_time.timestamp():
            break

        # Warte bis naechster Snapshot (interruptible)
        next_snapshot = time.time() + (SNAPSHOT_INTERVAL_MIN * 60)
        while _running and time.time() < next_snapshot:
            time.sleep(5)

    # Abschlussbericht
    if len(_snapshots) >= 2:
        report = generate_final_report(_snapshots)
        print(report)
        with open(logfile, "a") as f:
            f.write(report + "\n")
    elif len(_snapshots) == 1:
        print("\n[AUDIT] Nur 1 Snapshot — kein Trend-Bericht moeglich.")

    print(f"\n[AUDIT] Fertig. Log: {logfile}")


if __name__ == "__main__":
    main()
