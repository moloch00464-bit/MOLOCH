#!/usr/bin/env python3
"""
M.O.L.O.C.H. System Check — moloch check
==========================================
Prüft alle Subsysteme nach Reboot oder Deploy.
Kombiniert ChatGPT Validation Tests + Gemini Meridian Checks.

Usage:
    python3 scripts/moloch_system_check.py          # Vollständiger Check
    python3 scripts/moloch_system_check.py --quick   # Nur Imports + Health
    python3 scripts/moloch_system_check.py --json    # Maschinenlesbar

Erstellt: 06.03.2026
Quellen: ChatGPT (BRAIN_DEBUG_AND_SYSTEM_VALIDATION),
         Gemini+DeepSeek (MASTER_INTEGRATION_PLAN),
         Opus (Pragmatismus-Filter)
"""

import sys
import os
import time
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Moloch Root
MOLOCH_ROOT = Path.home() / "moloch"
sys.path.insert(0, str(MOLOCH_ROOT))


# ── Result Tracking ──────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str  # PASS, FAIL, WARN, SKIP
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class SystemReport:
    timestamp: float = field(default_factory=time.time)
    hostname: str = ""
    checks: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, result: CheckResult):
        self.checks.append(asdict(result))

    def finalize(self):
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["status"] == "PASS")
        failed = sum(1 for c in self.checks if c["status"] == "FAIL")
        warned = sum(1 for c in self.checks if c["status"] == "WARN")
        skipped = sum(1 for c in self.checks if c["status"] == "SKIP")
        self.summary = {
            "total": total,
            "pass": passed,
            "fail": failed,
            "warn": warned,
            "skip": skipped,
            "verdict": "PASS" if failed == 0 else "FAIL",
        }


report = SystemReport()


# ── Helper ───────────────────────────────────────────────────

ICONS = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}


def check(name: str, fn, critical=True):
    """Führt einen Check aus, fängt Exceptions, misst Zeit."""
    t0 = time.time()
    try:
        ok, detail = fn()
        latency = (time.time() - t0) * 1000
        status = "PASS" if ok else ("FAIL" if critical else "WARN")
        result = CheckResult(name, status, detail, round(latency, 1))
    except Exception as e:
        latency = (time.time() - t0) * 1000
        status = "FAIL" if critical else "WARN"
        result = CheckResult(name, status, f"Exception: {e}", round(latency, 1))

    report.add(result)
    icon = ICONS.get(result.status, "?")
    print(f"  {icon} {name}: {result.status} ({result.latency_ms:.0f}ms) — {result.detail}")
    return result.status == "PASS"


# ══════════════════════════════════════════════════════════════
# CHECKS — Reihenfolge: Hardware → Pipeline → Core → Memory →
#           Awareness → Personality → Autonomy → System Health
# ══════════════════════════════════════════════════════════════


# ── 1. HARDWARE & SYSTEM ─────────────────────────────────────

def check_pi_hardware():
    """CPU, RAM, Temperatur, Disk."""
    import psutil

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # Temperatur
    temp = "?"
    temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp_file.exists():
        temp_raw = int(temp_file.read_text().strip())
        temp = f"{temp_raw / 1000:.1f}°C"

    ram_used_mb = ram.used / (1024 * 1024)
    ram_total_mb = ram.total / (1024 * 1024)
    disk_free_gb = disk.free / (1024 * 1024 * 1024)

    detail = (
        f"CPU {cpu}%, RAM {ram_used_mb:.0f}/{ram_total_mb:.0f}MB "
        f"({ram.percent}%), Temp {temp}, Disk {disk_free_gb:.1f}GB frei"
    )

    # FAIL wenn RAM > 90% oder Disk < 1GB
    ok = ram.percent < 90 and disk_free_gb > 1.0
    return ok, detail


def check_hailo_device():
    """Hailo NPU erreichbar?"""
    result = subprocess.run(
        ["hailortcli", "fw-control", "identify"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        # Erste Zeile mit Board Name
        lines = result.stdout.strip().split("\n")
        info = lines[0] if lines else "OK"
        return True, f"NPU online: {info[:80]}"
    return False, f"NPU nicht erreichbar: {result.stderr.strip()[:100]}"


def check_rtsp_stream():
    """RTSP-Stream der Kamera erreichbar?"""
    # Kurzer ffprobe Check — Timeout 5s
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-rtsp_transport", "tcp",
            "-i", "rtsp://192.168.178.75/av_stream/ch0",
            "-show_entries", "stream=width,height,codec_name",
            "-of", "json",
        ],
        capture_output=True, text=True, timeout=8,
    )
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            w = stream.get("width", "?")
            h = stream.get("height", "?")
            codec = stream.get("codec_name", "?")
            return True, f"Stream OK: {w}x{h} {codec}"
        except Exception:
            return True, "Stream erreichbar (Parse-Fehler)"
    return False, f"RTSP nicht erreichbar"


def check_service_running():
    """systemd Service Status."""
    result = subprocess.run(
        ["systemctl", "is-active", "moloch"],
        capture_output=True, text=True, timeout=5,
    )
    active = result.stdout.strip() == "active"
    return active, f"Service: {result.stdout.strip()}"


# ── 2. VISION PIPELINE (Gemini: Vision_Channel Meridian) ─────

def check_tappas_pipeline():
    """TAPPAS Pipeline importierbar + Feature-Flag aktiv?"""
    tappas_flag = os.environ.get("MOLOCH_USE_TAPPAS", "0")
    from core.perception.tappas_pipeline import TappasPipeline
    return True, f"TappasPipeline OK, MOLOCH_USE_TAPPAS={tappas_flag}"


def check_vision_models():
    """HEF-Dateien für YOLO, SCRFD, ArcFace vorhanden?"""
    # Alle bekannten HEF-Verzeichnisse in Prioritaetsreihenfolge
    hef_candidates = [
        Path("/mnt/moloch-data/hailo/models"),
        MOLOCH_ROOT / "models" / "hailo",
        MOLOCH_ROOT / "models",
        Path("/usr/share/hailo-models"),
        MOLOCH_ROOT / "hefs",
    ]
    hef_dir = None
    for candidate in hef_candidates:
        if candidate.exists() and list(candidate.glob("*.hef")):
            hef_dir = candidate
            break
    if hef_dir is None:
        # Fallback auf ersten existierenden Pfad
        for candidate in hef_candidates:
            if candidate.exists():
                hef_dir = candidate
                break

    hefs = list(hef_dir.rglob("*.hef")) if hef_dir and hef_dir.exists() else []
    names = [h.name for h in hefs]

    found = []
    missing = []
    for model in ["yolo", "scrfd", "arcface"]:
        if any(model in n.lower() for n in names):
            found.append(model)
        else:
            missing.append(model)

    if missing:
        return False, f"HEFs gefunden: {found}, FEHLEND: {missing}"
    return True, f"Alle HEFs vorhanden: {found} ({len(hefs)} total in {hef_dir})"


# ── 3. EVENT BUS ─────────────────────────────────────────────

def check_event_bus():
    """Event Bus Singleton + PubSub funktioniert?"""
    from core.moloch_event_bus import get_event_bus

    bus = get_event_bus()
    received = []
    bus.subscribe("system.check", lambda e: received.append(e))
    bus.emit("system.check", {"test": True}, source="moloch_check")
    time.sleep(0.15)

    if received:
        return True, f"PubSub OK, {len(bus._subscribers)} Subscriber registriert"
    return False, "Event nicht empfangen"


# ── 4. ACTION BRIDGE (Gate 1) ────────────────────────────────

def check_action_bridge():
    """Action Bridge FSM importierbar + Status abfragbar?"""
    from core.action_bridge import get_action_bridge

    ab = get_action_bridge()
    status = ab.get_status()
    if isinstance(status, dict):
        state = status.get("state", "UNKNOWN")
        return True, f"Bridge OK, State: {state}"
    return True, f"Bridge OK, Status: {status}"


# ── 5. MEMORY SYSTEM (Gemini: Memory_Core Meridian) ──────────

def check_qdrant():
    """Qdrant Vector DB erreichbar?"""
    try:
        from qdrant_client import QdrantClient

        # Lokaler Qdrant — entweder gRPC oder REST
        client = QdrantClient(host="localhost", port=6333, timeout=5)
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        return True, f"Qdrant OK, Collections: {names}"
    except ImportError:
        return False, "qdrant-client nicht installiert"
    except Exception as e:
        # Qdrant läuft vielleicht nicht als Server sondern embedded
        return False, f"Qdrant nicht erreichbar: {e}"


def check_music_memory():
    """MusicMemory JSON laden?"""
    # Prüfe ob MusicMemory existiert und ladbar ist
    music_paths = [
        MOLOCH_ROOT / "state" / "music_memory.json",
        Path("/mnt/moloch-data/memory/music_memory.json"),
        MOLOCH_ROOT / "config" / "music_memory.json",
    ]
    for p in music_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                entries = len(data) if isinstance(data, (list, dict)) else "?"
                return True, f"MusicMemory OK: {p.name} ({entries} entries)"
            except json.JSONDecodeError as e:
                return False, f"MusicMemory CORRUPT: {p} — {e}"

    # Kein File gefunden — probiere Import
    try:
        from core.memory.music_memory import get_music_memory
        mm = get_music_memory()
        return True, "MusicMemory Modul OK (kein JSON-File gefunden)"
    except ImportError:
        return True, "MusicMemory nicht implementiert (Gate 2+ Feature)"
    except Exception as e:
        return False, f"MusicMemory Fehler: {e}"


def check_episodic_memory():
    """Episodisches Gedächtnis erreichbar?"""
    try:
        from core.memory.episodic_memory import get_episodic_memory
        em = get_episodic_memory()
        status = em.get_status() if hasattr(em, "get_status") else "importiert"
        return True, f"EpisodicMemory OK: {status}"
    except ImportError:
        return True, "EpisodicMemory nicht implementiert (Gate 2+ Feature)"
    except Exception as e:
        return False, f"EpisodicMemory Fehler: {e}"


# ── 6. AWARENESS CHAIN (Gate 3) ──────────────────────────────

def check_awareness_chain():
    """RoomMap → MotionAnalyzer → ActivityClassifier → ContextEvaluator."""
    modules = [
        ("RoomMap", "core.awareness.room_map", "get_room_map"),
        ("MotionAnalyzer", "core.awareness.motion_analyzer", "get_motion_analyzer"),
        ("ActivityClassifier", "core.awareness.activity_classifier", "get_activity_classifier"),
        ("ContextEvaluator", "core.awareness.context_evaluator", "get_context_evaluator"),
    ]
    loaded = []
    missing = []
    for name, module_path, factory in modules:
        try:
            mod = __import__(module_path, fromlist=[factory])
            getattr(mod, factory)()
            loaded.append(name)
        except (ImportError, AttributeError):
            missing.append(name)
        except Exception:
            missing.append(name)

    if missing:
        if loaded:
            return False, f"Teilweise: {loaded} OK, FEHLEND: {missing}"
        return True, "Awareness Chain nicht implementiert (Gate 3 Feature)"
    return True, f"Kette komplett: {' → '.join(loaded)}"


# ── 7. PERSONALITY SYSTEM (Gate 4) ───────────────────────────

def check_personality():
    """TensionIntegrator → MoodEngine."""
    parts = []
    try:
        from core.personality.tension_integrator import get_tension_integrator
        ti = get_tension_integrator()
        status = ti.get_status() if hasattr(ti, "get_status") else {}
        tension = status.get("tension", "?") if isinstance(status, dict) else "?"
        parts.append(f"Tension={tension}")
    except (ImportError, Exception) as e:
        parts.append(f"TensionIntegrator: {e}")

    try:
        from core.personality.mood_engine import get_mood_engine
        me = get_mood_engine()
        status = me.get_status() if hasattr(me, "get_status") else {}
        mood = status.get("mood", "?") if isinstance(status, dict) else "?"
        parts.append(f"Mood={mood}")
    except (ImportError, Exception) as e:
        parts.append(f"MoodEngine: {e}")

    detail = ", ".join(parts)
    ok = not any("Error" in p or "Exception" in p for p in parts)
    return ok, detail


# ── 8. AUTONOMY SYSTEM (Gate 5) ──────────────────────────────

def check_autonomy():
    """DecisionEngine, AtmosphereController, Homeostasis, NightCycle."""
    modules = {
        "DecisionEngine": "core.autonomy.decision_engine.get_decision_engine",
        "Atmosphere": "core.autonomy.atmosphere_controller.get_atmosphere_controller",
        "Homeostasis": "core.autonomy.homeostasis.get_homeostasis",
        "NightCycle": "core.autonomy.night_cycle.get_night_cycle",
    }
    loaded = []
    failed = []
    for name, import_path in modules.items():
        try:
            parts = import_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            factory = getattr(mod, parts[1])
            factory()
            loaded.append(name)
        except (ImportError, AttributeError):
            failed.append(name)
        except Exception as e:
            failed.append(f"{name}({e})")

    if failed and not loaded:
        return True, "Autonomy nicht implementiert (Gate 5 Feature)"
    if failed:
        return False, f"Teilweise: {loaded} OK, FEHLEND: {failed}"
    return True, f"Alle 4 Module OK: {', '.join(loaded)}"


# ── 9. SYSTEM HEALTH (Gemini: Meridian Checks) ───────────────

def check_homeostasis_status():
    """Homeostasis Health-Daten abfragen."""
    try:
        from core.autonomy.homeostasis import get_homeostasis
        h = get_homeostasis()
        status = h.get_status() if hasattr(h, "get_status") else None
        if status and isinstance(status, dict):
            ram = status.get("ram_percent", "?")
            cpu = status.get("cpu_percent", "?")
            temp = status.get("temperature", "?")
            return True, f"Homeostasis: RAM {ram}%, CPU {cpu}%, Temp {temp}"
        return True, "Homeostasis importiert (kein Status verfügbar)"
    except ImportError:
        return True, "Homeostasis nicht verfügbar"
    except Exception as e:
        return False, f"Homeostasis Fehler: {e}"


def check_shm_frame():
    """/dev/shm/moloch_frame existiert? (TAPPAS Frame-IPC)"""
    shm_path = Path("/dev/shm/moloch_frame")
    if shm_path.exists():
        size = shm_path.stat().st_size
        age = time.time() - shm_path.stat().st_mtime
        return True, f"SHM Frame OK: {size} bytes, {age:.1f}s alt"
    return False, "Kein SHM Frame — TAPPAS schreibt nicht?"


def check_logs_disk():
    """Log-Verzeichnis nicht überlaufen?"""
    log_dir = MOLOCH_ROOT / "logs"
    if not log_dir.exists():
        return True, "Kein Log-Verzeichnis"
    total = sum(f.stat().st_size for f in log_dir.rglob("*") if f.is_file())
    total_mb = total / (1024 * 1024)
    ok = total_mb < 500  # Warnung über 500MB
    return ok, f"Logs: {total_mb:.0f}MB (Limit: 500MB)"


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

def run_full():
    """Alle Checks durchlaufen."""
    print()
    print("═" * 55)
    print("  M.O.L.O.C.H. SYSTEM CHECK")
    print("═" * 55)

    # Hardware & System
    print("\n  ── HARDWARE & SYSTEM ──")
    check("Pi5 Hardware", check_pi_hardware)
    check("Hailo NPU", check_hailo_device)
    check("RTSP Stream", check_rtsp_stream, critical=False)
    check("Service Status", check_service_running)

    # Vision Pipeline
    print("\n  ── VISION PIPELINE ──")
    check("TAPPAS Pipeline", check_tappas_pipeline)
    check("Vision HEF Models", check_vision_models, critical=False)

    # Core Systems
    print("\n  ── CORE SYSTEMS ──")
    check("Event Bus", check_event_bus)
    check("Action Bridge", check_action_bridge)

    # Memory
    print("\n  ── MEMORY ──")
    check("Qdrant VectorDB", check_qdrant, critical=False)
    check("Music Memory", check_music_memory, critical=False)
    check("Episodic Memory", check_episodic_memory, critical=False)

    # Awareness
    print("\n  ── AWARENESS (Gate 3) ──")
    check("Awareness Chain", check_awareness_chain, critical=False)

    # Personality
    print("\n  ── PERSONALITY (Gate 4) ──")
    check("Personality System", check_personality, critical=False)

    # Autonomy
    print("\n  ── AUTONOMY (Gate 5) ──")
    check("Autonomy Modules", check_autonomy, critical=False)
    check("Homeostasis Status", check_homeostasis_status, critical=False)

    # System Health
    print("\n  ── SYSTEM HEALTH ──")
    check("SHM Frame IPC", check_shm_frame, critical=False)
    check("Log Disk Usage", check_logs_disk, critical=False)

    # Finalize
    report.finalize()
    verdict = report.summary["verdict"]
    icon = "✅" if verdict == "PASS" else "❌"

    print()
    print("═" * 55)
    print(f"  {icon} VERDICT: {verdict}")
    print(
        f"  {report.summary['pass']} Pass, "
        f"{report.summary['fail']} Fail, "
        f"{report.summary['warn']} Warn, "
        f"{report.summary['skip']} Skip"
    )
    print("═" * 55)
    print()

    return report


def run_quick():
    """Nur die kritischen Checks."""
    print("\n  M.O.L.O.C.H. QUICK CHECK\n")
    check("Pi5 Hardware", check_pi_hardware)
    check("Service Status", check_service_running)
    check("Event Bus", check_event_bus)
    check("Action Bridge", check_action_bridge)
    check("SHM Frame", check_shm_frame, critical=False)

    report.finalize()
    verdict = report.summary["verdict"]
    print(f"\n  {'✅' if verdict == 'PASS' else '❌'} {verdict}\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="M.O.L.O.C.H. System Check — prüft alle Subsysteme"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Nur kritische Checks")
    parser.add_argument("--json", action="store_true",
                        help="Output als JSON")
    args = parser.parse_args()

    report.hostname = os.uname().nodename

    if args.quick:
        result = run_quick()
    else:
        result = run_full()

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))

    # Exit Code
    sys.exit(0 if result.summary.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
