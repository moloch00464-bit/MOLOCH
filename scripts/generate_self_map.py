#!/usr/bin/env python3
"""
generate_self_map.py — MOLOCHs Koerperwissen generieren.
Scannt core/, liest settings.json und schreibt config/self_map.json (atomar).

Aufruf:
    python3 ~/moloch/scripts/generate_self_map.py
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MOLOCH_ROOT = Path(__file__).parent.parent
CORE_DIR = MOLOCH_ROOT / "core"
SETTINGS_PATH = MOLOCH_ROOT / "config" / "settings.json"
REGISTRY_PATH = MOLOCH_ROOT / "config" / "self_tune_registry.json"
OUTPUT_PATH = MOLOCH_ROOT / "config" / "self_map.json"

# Risiko-Stufe pro Datei (aus CLAUDE.md / Agenten-Wissen)
RISK_MAP = {
    "moloch_service.py": "RED",
    "perception/tappas_pipeline.py": "RED",
    "hardware/camera.py": "RED",
    "voice_pipeline.py": "YELLOW",
    "tts.py": "YELLOW",
    "personality/personality_engine.py": "YELLOW",
    "unconscious_engine.py": "YELLOW",
    "core_integrator.py": "YELLOW",
    "daily_learner.py": "YELLOW",
    "perception/einpraegen.py": "YELLOW",
    "ipc_router.py": "YELLOW",
}

# NEVER-Regeln (aus CLAUDE.md)
NEVER_RULES = [
    "NEVER 1: GStreamer-Pipeline-String nicht blind aendern",
    "NEVER 2: Pan-Vorzeichen in camera.py Zeile ~732 ist TABU",
    "NEVER 3: ArcFace-Threshold nicht als Quick-Fix erhoehen",
    "NEVER 4: Nie mehrere ROT-Dateien in einem Commit",
    "NEVER 5: Immer timeout= bei subprocess.run/Popen",
    "NEVER 6: Immer atomic JSON write (tmp + os.replace)",
    "NEVER 7: Keine Runtime-State-Dateien committen",
    "NEVER 8: Kein shell=True in Produktionscode",
    "NEVER 9: HailoRT uint8 vs float32 vor Inferenz pruefen",
    "NEVER 10: Kein np.ndarray in moloch_service.py Signaturen",
    "NEVER 11: __pycache__ loeschen vor Neustart nach Aenderung",
    "NEVER 12: Nicht im Worktree testen — auf Pi deployen",
]


# ===========================================================================
# Module scannen
# ===========================================================================

def _get_risk(rel_path: str) -> str:
    for key, risk in RISK_MAP.items():
        if key in rel_path:
            return risk
    return "GREEN"


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def _extract_imports(path: Path) -> list:
    """Direkte Top-Level imports aus einer .py-Datei extrahieren."""
    imports = set()
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
    except Exception:
        pass
    return sorted(imports)


def scan_modules() -> list:
    modules = []
    for py_file in sorted(CORE_DIR.rglob("*.py")):
        rel = py_file.relative_to(MOLOCH_ROOT).as_posix()
        lines = _count_lines(py_file)
        risk = _get_risk(rel)
        imports = _extract_imports(py_file)
        modules.append({
            "path": rel,
            "lines": lines,
            "risk": risk,
            "imports": [i for i in imports if not i.startswith("_")],
        })
    return modules


# ===========================================================================
# Parameter aus settings.json + Registry
# ===========================================================================

def collect_parameters() -> list:
    settings = {}
    try:
        settings = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        pass

    registry = {"parameters": []}
    try:
        registry = json.loads(REGISTRY_PATH.read_text())
    except Exception:
        pass

    params = []
    for p in registry.get("parameters", []):
        section = p.get("section", "")
        key = p.get("key", "")
        current = settings.get(section, {}).get(key)
        params.append({
            "key": f"{section}.{key}",
            "value": current if current is not None else p.get("default"),
            "default": p.get("default"),
            "min": p.get("min"),
            "max": p.get("max"),
            "step": p.get("step"),
            "type": p.get("type"),
            "risk": p.get("risk", "GREEN"),
            "description": p.get("description", ""),
        })
    return params


# ===========================================================================
# Systemzustand live lesen
# ===========================================================================

def read_ram_mb() -> dict:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
        total = info.get("MemTotal", 0) // 1024
        avail = info.get("MemAvailable", 0) // 1024
        used = total - avail
        return {"total_mb": total, "available_mb": avail, "used_mb": used}
    except Exception:
        return {}


def read_cpu_temp() -> float:
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return int(raw) / 1000.0
    except Exception:
        return 0.0


def _run(cmd: list, timeout: int = 5) -> tuple:
    """Fuehrt Kommando aus, gibt (returncode, stdout) zurueck."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except Exception as e:
        return -1, str(e)


# ===========================================================================
# Health Checks
# ===========================================================================

HEALTH_CHECKS_DEF = [
    {"id": "service_running", "cmd": ["systemctl", "is-active", "moloch"],
     "expect_rc": 0, "severity": "critical",
     "diagnosis": "Moloch-Service nicht aktiv — Crash oder manueller Stop"},
    {"id": "camera_reachable", "cmd": ["ping", "-c1", "-W2", "192.168.178.25"],
     "expect_rc": 0, "severity": "critical",
     "diagnosis": "Kamera nicht erreichbar — Netzwerk oder Kamera-Neustart noetig"},
    {"id": "npu_accessible", "cmd": ["hailortcli", "fw-control", "identify"],
     "expect_rc": 0, "severity": "critical",
     "diagnosis": "NPU nicht erreichbar — PCIe-Problem oder Treiber-Fehler"},
]


def run_health_checks() -> list:
    results = []
    for chk in HEALTH_CHECKS_DEF:
        rc, out = _run(chk["cmd"])
        passed = (rc == chk["expect_rc"])
        results.append({
            "id": chk["id"],
            "passed": passed,
            "severity": chk["severity"],
            "output": out[:120] if out else "",
            "diagnosis": chk["diagnosis"] if not passed else "OK",
        })

    # Status-JSON Checks (FPS, RAM, Tracking)
    status = {}
    try:
        status = json.loads(Path("/dev/shm/moloch_status.json").read_text())
    except Exception:
        pass

    fps = float(status.get("fps", 0))
    results.append({
        "id": "fps_check",
        "passed": fps >= 10,
        "severity": "critical",
        "output": f"fps={fps:.1f}",
        "diagnosis": "OK" if fps >= 10 else f"FPS {fps:.1f} < 10 — Pipeline-Problem",
    })

    moves = float(status.get("tracking_moves_per_minute", 0))
    results.append({
        "id": "tracking_stable",
        "passed": moves < 60,
        "severity": "warning",
        "output": f"moves/min={moves:.0f}",
        "diagnosis": "OK" if moves < 60 else f"Tracking hektisch ({moves:.0f}/min) — Dead Zone pruefen",
    })

    return results


# ===========================================================================
# Hauptfunktion
# ===========================================================================

def generate() -> dict:
    ram = read_ram_mb()
    cpu_temp = read_cpu_temp()

    self_map = {
        "version": "1.0",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "brain": "Raspberry Pi 5",
            "ram_mb": 4096,
            "ram_usable_mb": 3800,
            "ram_live": ram,
            "cpu_temp_c": cpu_temp,
            "npu": "Hailo-10H",
            "npu_tops": 40,
            "npu_ram_mb": 8192,
            "npu_interface": "PCIe Gen2 x1 (Pi5 Limit, ~500MB/s)",
            "camera": {
                "model": "Sonoff CAM-PT2",
                "ip": "192.168.178.25",
                "protocol": "RTSP/ONVIF",
                "resolution": "1920x1080",
                "fps": 20,
                "ptz": True,
                "pan_inverted": True,
                "pan_range": [-168.4, 170.0],
                "tilt_range": [-78.0, 78.8],
                "note": "NEVER 2: Pan-Vorzeichen in camera.py TABU",
            },
            "audio": {
                "input": ["SmartMic BT (WiFi, ReSpeaker firmware)", "ReSpeaker Lite (USB Fallback)"],
                "output": "Piper TTS via HDMI/PipeWire",
                "whisper_model": "turbo",
            },
            "storage": {
                "ssd1": {"mount": "/", "fs": "ext4", "size_gb": 465, "purpose": "Code, Configs, Voices"},
                "ssd2": {"mount": "/mnt/moloch-data", "fs": "NTFS", "size_gb": 477,
                         "purpose": "AI-Modelle, Qdrant DB",
                         "note": "NTFS: kein chmod moeglich"},
            },
            "cooling": {
                "noctua": {"gpio": 18, "pwm_hz": 25000, "control": "scripts/fan_control.py",
                           "activation_temp_c": 42},
                "cpu_cooler": {"path": "/sys/class/thermal/cooling_device0", "levels": 4},
            },
            "ups": "Pico Power 5 (7.5V LiPo, Schutz vor Stromausfall)",
        },
        "modules": scan_modules(),
        "parameters": collect_parameters(),
        "limits": {
            "ram_max_mb": 3800,
            "ram_warning_mb": 3200,
            "ram_critical_mb": 3500,
            "cpu_temp_warning_c": 65,
            "cpu_temp_critical_c": 75,
            "fps_target": 20,
            "fps_minimum": 10,
            "rtsp_slots": 1,
            "vdevice_count": 1,
            "max_subprocess_timeout_s": 60,
            "npu_llm_blocks_vision": True,
            "never_rules": NEVER_RULES,
        },
        "health_checks": run_health_checks(),
    }
    return self_map


def main():
    print("[SELF-MAP] Generiere Koerperwissen...", file=sys.stderr)
    data = generate()

    # Atomar schreiben
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, str(OUTPUT_PATH))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    modules_count = len(data["modules"])
    params_count = len(data["parameters"])
    health_pass = sum(1 for h in data["health_checks"] if h["passed"])
    health_total = len(data["health_checks"])

    print(f"[SELF-MAP] Fertig: {modules_count} Module, {params_count} Parameter, "
          f"{health_pass}/{health_total} Health-Checks OK", file=sys.stderr)
    print(f"[SELF-MAP] Geschrieben: {OUTPUT_PATH}", file=sys.stderr)

    # Auch auf stdout: JSON fuer Pipes
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
