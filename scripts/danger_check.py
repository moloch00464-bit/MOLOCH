#!/usr/bin/env python3
"""
M.O.L.O.C.H. DANGER CHECK — Pre-Commit Hook
===============================================
Prueft staged Files auf gefaehrliche Patterns.
Blockiert Commits mit 2+ ROT-Dateien oder Runtime-State.

Installation:
  ln -sf ~/moloch/scripts/danger_check.py ~/moloch/.git/hooks/pre-commit
  chmod +x ~/moloch/scripts/danger_check.py

Exit: 0 = Commit erlaubt, 1 = Commit blockiert
"""

import re
import subprocess
import sys

# ROT-Dateien: System-Crash Risk bei Fehler
ROT_FILES = {
    "core/moloch_service.py",
    "core/perception/tappas_pipeline.py",
    "core/hardware/camera.py",
    "core/hardware/hailo_manager.py",
    "core/core_integrator.py",
    "core/voice_pipeline.py",
    "core/mpo/autonomous_tracker.py",
    "core/gui/moloch_unified_panel.py",
    "core/speech/audio_pipeline.py",
    "core/inference_engine.py",
    "core/camera_manager.py",
    "core/model_orchestrator.py",
    "core/perception_engine.py",
    "core/ipc_router.py",
    "core/hardware/thermal_manager.py",
    "core/ptz_tracker.py",
    "core/perception/model_scheduler.py",
    "core/memory/episodic_memory.py",
    "core/memory/person_reid.py",
    "config/settings.json",
}

# Runtime-State: Gehoert NICHT in Git
RUNTIME_STATE_FILES = {
    "config/last_face_position.json",
    "config/learned_patrol_positions.json",
    "config/kontext.json",
}

# Pattern die in neuen Zeilen gewarnt werden
DANGEROUS_PATTERNS = [
    (r"subprocess\.Popen\(", "subprocess.Popen ohne timeout"),
    (r"shell\s*=\s*True", "shell=True (Injection-Risiko)"),
]

# Spezifische Zeilen die TABU sind
TABU_CHECKS = [
    ("core/hardware/camera.py", r"pan_delta\s*=",
     "Pan-Vorzeichen Aenderung! MINUS IST KORREKT (camera.py:732)"),
]


def get_staged_files():
    """Liste der staged Files relativ zum Repo-Root."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10
        )
        return [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def get_staged_diff():
    """Staged Diff fuer Pattern-Analyse."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "-U0"],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout
    except Exception:
        return ""


def check_rot_files(staged):
    """Prueft ob 2+ ROT-Dateien im selben Commit sind."""
    rot_staged = [f for f in staged if f in ROT_FILES]
    if len(rot_staged) >= 2:
        return False, rot_staged
    return True, rot_staged


def check_runtime_state(staged):
    """Prueft ob Runtime-State-Dateien gestaged sind."""
    state_staged = [f for f in staged if f in RUNTIME_STATE_FILES]
    if state_staged:
        return False, state_staged
    return True, []


def check_dangerous_patterns(diff_text):
    """Sucht nach gefaehrlichen Patterns in neuen Zeilen."""
    warnings = []
    current_file = ""

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            # Dateiname extrahieren
            parts = line.split(" b/")
            current_file = parts[-1] if len(parts) > 1 else ""
        elif line.startswith("+") and not line.startswith("+++"):
            # Neue Zeile (hinzugefuegt)
            for pattern, desc in DANGEROUS_PATTERNS:
                if re.search(pattern, line):
                    warnings.append(f"{current_file}: {desc}")

    return warnings


def check_tabu_lines(diff_text):
    """Prueft ob TABU-Zeilen geaendert wurden."""
    warnings = []
    current_file = ""

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            current_file = parts[-1] if len(parts) > 1 else ""
        elif line.startswith("+") and not line.startswith("+++"):
            for file_pattern, regex, msg in TABU_CHECKS:
                if current_file == file_pattern and re.search(regex, line):
                    warnings.append(msg)

    return warnings


def main():
    staged = get_staged_files()
    if not staged:
        return 0  # Nichts gestaged, nichts zu pruefen

    diff_text = get_staged_diff()

    fails = []
    warns = []

    # Check 1: ROT-Dateien
    rot_ok, rot_files = check_rot_files(staged)
    if not rot_ok:
        fails.append(f"2+ ROT-Dateien im selben Commit:\n" +
                      "\n".join(f"    - {f}" for f in rot_files) +
                      "\n    Regel: Maximal 1 ROT-Datei pro Commit.")
    elif rot_files:
        warns.append(f"1 ROT-Datei: {rot_files[0]} — Vorsicht!")

    # Check 2: Runtime-State
    state_ok, state_files = check_runtime_state(staged)
    if not state_ok:
        fails.append(f"Runtime-State-Dateien gestaged:\n" +
                      "\n".join(f"    - {f}" for f in state_files) +
                      "\n    Diese gehoeren nach /dev/shm/, nicht in Git.")

    # Check 3: Gefaehrliche Patterns
    pattern_warns = check_dangerous_patterns(diff_text)
    for w in pattern_warns:
        warns.append(w)

    # Check 4: TABU-Zeilen
    tabu_warns = check_tabu_lines(diff_text)
    for w in tabu_warns:
        warns.append(f"TABU: {w}")

    # Check 5: Viele Dateien
    if len(staged) > 5:
        warns.append(f"{len(staged)} Dateien in einem Commit — aufteilen?")

    # Output
    if fails or warns:
        print("[DANGER CHECK] Pre-Commit Hook")
        print()

    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
            print()

    if warns:
        for w in warns:
            print(f"  WARN: {w}")
        print()

    if fails:
        print("  Commit BLOCKIERT. Bitte aufteilen oder --no-verify (nur wenn sicher).")
        return 1

    if warns:
        print("  Commit erlaubt, aber bitte Warnungen beachten.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
