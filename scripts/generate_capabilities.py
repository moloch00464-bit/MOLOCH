#!/usr/bin/env python3
"""
M.O.L.O.C.H. System Capabilities Generator
=============================================
Erzeugt config/system_capabilities.json mit echten Import-Tests.
Laeuft beim Service-Start oder manuell: python3 scripts/generate_capabilities.py

Prueft:
- Welche Python-Module tatsaechlich importierbar sind
- Welche HEF-Modelle auf der SSD vorhanden sind
- Welche Hardware-Interfaces verfuegbar sind
- Welche Gates aktiv/bestanden sind
- Feature-Flags aus Environment
"""

import json
import os
import sys
import time
import importlib
from pathlib import Path
from datetime import datetime

# Projekt-Root bestimmen
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_PATH = PROJECT_ROOT / "config" / "system_capabilities.json"

# ========================================================================
# Modul-Definitionen: (import_path, gate, beschreibung)
# ========================================================================
CORE_MODULES = [
    # Gate 0 — Basis
    ("core.moloch_service", "0", "Service-Kern"),
    ("core.camera_manager", "0", "ONVIF/RTSP Kamera"),
    ("core.ptz_tracker", "0", "PTZ Tracker"),
    ("core.ptz_arbiter", "0", "PTZ Arbiter (Auto/Manuell)"),
    ("core.perception_engine", "0", "Perception Stage Machine"),
    ("core.model_orchestrator", "0", "Model Scheduling (legacy)"),
    ("core.inference_engine", "0", "NPU Inference (legacy)"),
    ("core.led_controller", "0", "eWeLink LED"),
    ("core.voice_pipeline", "0", "Voice Pipeline"),
    ("core.longterm_memory", "0", "Langzeitgedaechtnis"),
    ("core.ipc_router", "0", "Service/Panel IPC"),
    ("core.core_integrator", "0", "Tension/Dominance Core"),

    # Gate 0.5 — TAPPAS
    ("core.perception.tappas_pipeline", "0.5", "GStreamer TAPPAS Pipeline"),
    ("core.perception.perception_frame", "0.5", "PFrame Datenstruktur"),
    ("core.perception.perception_buffer", "0.5", "PFrame Ring-Buffer"),
    ("core.perception.hailo_postprocess", "0.5", "TAPPAS Postprocessing"),
    ("core.perception.perception_manager", "0.5", "Perception Manager"),
    ("core.perception.model_health", "0.5", "NPU Health Monitor"),
    ("core.speech.hailo_whisper", "0.5", "Hailo Whisper (shared VDevice)"),

    # Gate 1 — Action Bridge + Autonomie
    ("core.action_bridge", "1", "Action Bridge FSM"),
    ("core.moloch_event_bus", "1", "Priority Event Bus"),
    ("core.capability_monitor", "1", "System Capability Monitor"),
    ("core.keyword_handler", "1", "Sprach-Keyword Triggers"),
    ("core.autonomy.decision_engine", "1", "Decision Engine"),
    ("core.autonomy.atmosphere_controller", "1", "Atmosphaeren-Controller"),
    ("core.autonomy.homeostasis", "1", "Selbstregulation"),
    ("core.awareness.room_map", "1", "Raum-Kartierung"),
    ("core.awareness.motion_analyzer", "1", "Bewegungs-Analyse"),
    ("core.awareness.activity_analyzer", "1", "Aktivitaets-Klassifikation"),
    ("core.awareness.context_evaluator", "1", "Kontext-Evaluator"),
    ("core.personality.mood_engine", "1", "Mood Engine"),
    ("core.personality.personality_engine", "1", "Personality Engine"),
    ("core.personality.behavior_rules", "1", "Behavior Rules"),

    # Gate 2+ — Vorbereitung
    ("core.memory.episodic_memory", "2", "Episodisches Gedaechtnis (Qdrant)"),
    ("core.memory.vector_memory", "2", "Semantisches Vector Memory"),
    ("core.memory.person_reid", "2", "Person ReID"),
    ("core.daily_learner", "2", "Daily Learner"),
    ("core.einpraegen", "2", "Batch Enrollment"),
]

# Externe Abhaengigkeiten
EXTERNAL_DEPS = [
    ("hailo", "Hailo NPU SDK"),
    ("gi", "GObject Introspection"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("tkinter", "Tkinter GUI"),
    ("pygame", "PyGame"),
    ("spotipy", "Spotify API"),
    ("qdrant_client", "Qdrant Vector DB"),
    ("piper", "Piper TTS"),
    ("rich", "Rich TUI"),
]

# HEF-Modelle (aktiv + Pfad)
HEF_MODELS = [
    ("yolov8m_h10.hef", "Person Detection", True),
    ("scrfd_10g.hef", "Face Detection", True),
    ("arcface_mobilefacenet.hef", "Face Recognition", True),
    ("yolov8s_pose_h10.hef", "Pose Estimation", False),
]
HEF_DIR = Path("/mnt/moloch-data/hailo/models")

# Piper Voice Models
VOICE_DIR = PROJECT_ROOT / "models" / "voices"


def test_import(module_path: str) -> bool:
    """Versuche ein Modul zu importieren. Gibt True/False zurueck."""
    try:
        importlib.import_module(module_path)
        return True
    except Exception:
        return False


def test_external_deps() -> dict:
    """Teste externe Abhaengigkeiten."""
    results = {}
    for mod, desc in EXTERNAL_DEPS:
        results[mod] = {
            "available": test_import(mod),
            "description": desc,
        }
    return results


def test_core_modules() -> dict:
    """Teste alle Core-Module mit echten Imports."""
    results = {}
    for mod_path, gate, desc in CORE_MODULES:
        available = test_import(mod_path)
        results[mod_path] = {
            "available": available,
            "gate": gate,
            "description": desc,
        }
    return results


def check_hef_models() -> list:
    """Pruefe welche HEF-Modelle auf der SSD vorhanden sind."""
    models = []
    for filename, desc, active in HEF_MODELS:
        path = HEF_DIR / filename
        exists = path.exists()
        size_mb = round(path.stat().st_size / 1024 / 1024, 1) if exists else 0
        models.append({
            "name": filename.replace(".hef", ""),
            "file": filename,
            "description": desc,
            "exists": exists,
            "active_in_pipeline": active and exists,
            "size_mb": size_mb,
        })
    return models


def check_voice_models() -> list:
    """Pruefe welche Piper Voice Models vorhanden sind."""
    voices = []
    if VOICE_DIR.exists():
        # Piper Voices liegen als .onnx Dateien direkt im Ordner
        for f in sorted(VOICE_DIR.glob("*.onnx")):
            voices.append({
                "name": f.stem,
                "has_config": (f.parent / (f.name + ".json")).exists(),
            })
    return voices


def check_feature_flags() -> dict:
    """Feature-Flags aus Environment lesen."""
    return {
        "MOLOCH_USE_TAPPAS": os.environ.get("MOLOCH_USE_TAPPAS", "0") == "1",
        "MOLOCH_CAMERA_HOST": os.environ.get("MOLOCH_CAMERA_HOST", "192.168.178.25"),
        "MOLOCH_RTSP_URL_SET": bool(os.environ.get("MOLOCH_RTSP_URL", "")),
    }


def check_hardware() -> dict:
    """Hardware-Checks (CPU, RAM, Hailo, Storage)."""
    hw = {}
    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    hw["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        hw["ram_gb"] = None

    # CPU Model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Model"):
                    hw["cpu_model"] = line.split(":")[1].strip()
                    break
    except Exception:
        hw["cpu_model"] = None

    # Hailo Device
    hw["hailo_device"] = Path("/dev/hailo0").exists()

    # Storage
    for name, path in [("ssd1_code", str(PROJECT_ROOT)), ("ssd2_data", "/mnt/moloch-data")]:
        try:
            st = os.statvfs(path)
            total_gb = round((st.f_blocks * st.f_frsize) / (1024**3), 1)
            free_gb = round((st.f_bavail * st.f_frsize) / (1024**3), 1)
            hw[name] = {"total_gb": total_gb, "free_gb": free_gb}
        except Exception:
            hw[name] = None

    # Kuehlung und Stromversorgung
    hw["kuehlung"] = "Noctua NF-A2x20 PWM (30% reicht fuer 48\u00b0C unter Volllast)"
    hw["stromversorgung"] = "Pico Power 5 USV mit 7.5V Akku (Schutz vor Stromausfall)"

    # Lokale LLM-Modelle (hailo-ollama)
    import shutil
    hw["local_llm"] = {
        "hailo_ollama_installed": shutil.which("hailo-ollama") is not None,
        "port": 8000,
        "models": [
            {"name": "qwen2.5-instruct:1.5b", "rolle": "Kommunikation",
             "beschreibung": "Lokales Sprachmodell fuer Konversation auf Deutsch"},
            {"name": "deepseek_r1_distill_qwen:1.5b", "rolle": "Reasoning",
             "beschreibung": "Lokales Denkmodell fuer Selbstdiagnose und Logik"},
        ],
    }

    return hw


def check_audio_input() -> dict:
    """Audio-Eingang pruefen (WiFi-Mic ESP32 oder USB Fallback)."""
    audio = {
        "primary": "ESP32 WiFi ReSpeaker Lite",
        "samplerates": [16000, 48000],
        "protokoll": "UDP",
        "latenz_ms": 5,
        "fallback": "USB ALSA",
        "status": "unbekannt",
    }
    # Pruefen ob WiFi-Mic Modul importierbar
    try:
        from core.audio.wifi_mic import get_wifi_mic
        wm = get_wifi_mic()
        audio["status"] = "aktiv" if wm.connected else "bereit (nicht verbunden)"
    except Exception:
        audio["status"] = "modul_nicht_verfuegbar"
    return audio


def determine_gates(modules: dict) -> dict:
    """Gate-Status anhand der verfuegbaren Module bestimmen."""
    gate_modules = {}
    for mod_path, info in modules.items():
        gate = info["gate"]
        if gate not in gate_modules:
            gate_modules[gate] = {"total": 0, "available": 0}
        gate_modules[gate]["total"] += 1
        if info["available"]:
            gate_modules[gate]["available"] += 1

    gates = {}
    gate_info = {
        "0": {"name": "Vier Inseln", "date": "2026-03-01", "status": "PASS"},
        "0.5": {"name": "TAPPAS Pipeline", "date": "2026-03-05", "status": "PASS"},
        "1": {"name": "Action Bridge + Autonomie", "date": "2026-03-06", "status": "AKTIV"},
        "2": {"name": "Identity (ReID + Qdrant)", "date": None, "status": "GEPLANT"},
    }
    for gate_id, info in gate_info.items():
        counts = gate_modules.get(gate_id, {"total": 0, "available": 0})
        gates[f"gate_{gate_id}"] = {
            "name": info["name"],
            "date": info["date"],
            "status": info["status"],
            "modules_total": counts["total"],
            "modules_available": counts["available"],
            "ready": counts["available"] == counts["total"] and counts["total"] > 0,
        }
    return gates


def generate():
    """Hauptfunktion: Alle Tests ausfuehren, JSON schreiben."""
    print("[CAPABILITIES] Starte System-Capability-Tests...")
    t0 = time.time()

    modules = test_core_modules()
    externals = test_external_deps()
    hef_models = check_hef_models()
    voice_models = check_voice_models()
    flags = check_feature_flags()
    hardware = check_hardware()
    audio_input = check_audio_input()
    gates = determine_gates(modules)

    # Zusammenfassung
    mod_ok = sum(1 for m in modules.values() if m["available"])
    mod_total = len(modules)
    ext_ok = sum(1 for e in externals.values() if e["available"])
    ext_total = len(externals)
    hef_ok = sum(1 for h in hef_models if h["exists"])

    result = {
        "version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "generation_ms": 0,
        "summary": {
            "core_modules": f"{mod_ok}/{mod_total}",
            "external_deps": f"{ext_ok}/{ext_total}",
            "hef_models": f"{hef_ok}/{len(hef_models)}",
            "voice_models": len(voice_models),
            "tappas_active": flags["MOLOCH_USE_TAPPAS"],
        },
        "gates": gates,
        "feature_flags": flags,
        "hardware": hardware,
        "core_modules": modules,
        "external_dependencies": externals,
        "audio_input": audio_input,
        "npu_models": hef_models,
        "voice_models": voice_models,
    }

    elapsed = round((time.time() - t0) * 1000, 1)
    result["generation_ms"] = elapsed

    # Atomar speichern
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    tmp = str(OUTPUT_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, str(OUTPUT_PATH))

    print(f"[CAPABILITIES] Fertig in {elapsed:.0f}ms")
    print(f"  Core:   {mod_ok}/{mod_total} Module verfuegbar")
    print(f"  Extern: {ext_ok}/{ext_total} Abhaengigkeiten")
    print(f"  HEFs:   {hef_ok}/{len(hef_models)} Modelle")
    print(f"  Voices: {len(voice_models)} Modelle")
    print(f"  Gates:  {', '.join(g + '=' + v['status'] for g, v in gates.items())}")
    print(f"  → {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    generate()
