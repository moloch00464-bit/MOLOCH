#!/bin/bash
# PreToolUse Hook fuer Edit|Write: Prueft ROT-Dateien und NEVER-Regeln
# Exit 2 = blockiert die Aktion, Exit 0 = erlaubt

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null)
OLD_STRING=$(echo "$INPUT" | jq -r '.tool_input.old_string // ""' 2>/dev/null)
NEW_STRING=$(echo "$INPUT" | jq -r '.tool_input.new_string // ""' 2>/dev/null)

# Kein File = kein Check
[ -z "$FILE" ] && exit 0

BASENAME=$(basename "$FILE")

# ============================================================
# ROT-Dateien Warnung (blockiert NICHT, aber warnt deutlich)
# ============================================================
ROT_FILES=(
    "moloch_service.py"
    "tappas_pipeline.py"
    "camera.py"
    "hailo_manager.py"
    "core_integrator.py"
    "voice_pipeline.py"
    "autonomous_tracker.py"
    "moloch_unified_panel.py"
    "audio_pipeline.py"
    "inference_engine.py"
    "camera_manager.py"
    "model_orchestrator.py"
    "perception_engine.py"
    "ipc_router.py"
    "thermal_manager.py"
    "ptz_tracker.py"
    "model_scheduler.py"
    "episodic_memory.py"
    "person_reid.py"
    "settings.json"
    "super_res_worker.py"
    "low_light_processor.py"
)

for ROT in "${ROT_FILES[@]}"; do
    if [ "$BASENAME" = "$ROT" ]; then
        echo "WARNUNG: $BASENAME ist eine ROT-Datei (System-Crash Risk)! Pre-Flight Check und BACKUP Pflicht."
        break
    fi
done

# ============================================================
# NEVER 2: Pan-Vorzeichen in camera.py
# ============================================================
if [ "$BASENAME" = "camera.py" ]; then
    if echo "$OLD_STRING" | grep -q "pan_delta.*=.*-error_x\|pan_delta.*=.*error_x"; then
        echo "BLOCKIERT: NEVER 2 — Pan-Vorzeichen (pan_delta = -error_x) ist TABU. Das Minus ist KORREKT (Sonoff invertiert)." >&2
        exit 2
    fi
fi

# ============================================================
# NEVER: panel_styles.py nicht aendern (ausser explizit beauftragt)
# ============================================================
if [ "$BASENAME" = "panel_styles.py" ]; then
    echo "BLOCKIERT: panel_styles.py darf nur geaendert werden wenn explizit beauftragt." >&2
    exit 2
fi

# ============================================================
# NEVER 7: Runtime-State Dateien nicht committen
# ============================================================
if [ "$BASENAME" = "last_face_position.json" ] || [ "$BASENAME" = "learned_patrol_positions.json" ]; then
    echo "BLOCKIERT: NEVER 7 — Runtime-State Dateien gehoeren nicht in den Code." >&2
    exit 2
fi

# ============================================================
# NEVER 10: np.ndarray Type-Hints in moloch_service.py
# ============================================================
if [ "$BASENAME" = "moloch_service.py" ]; then
    if echo "$NEW_STRING" | grep -qE "np\.ndarray|numpy\.ndarray"; then
        if echo "$NEW_STRING" | grep -qE "def .*\(.*np\.ndarray"; then
            echo "BLOCKIERT: NEVER 10 — Kein np.ndarray als Type-Hint in moloch_service.py Signaturen. Nutze String-Annotation." >&2
            exit 2
        fi
    fi
fi

# ============================================================
# NEVER 8: shell=True in subprocess
# ============================================================
if echo "$NEW_STRING" | grep -q "shell=True"; then
    echo "WARNUNG: NEVER 8 — shell=True erkannt! Command Injection Risiko. Nutze Liste statt String."
fi

# ============================================================
# NEVER 6: JSON direkt schreiben (json.dump ohne atomic)
# ============================================================
if echo "$NEW_STRING" | grep -qE "json\.dump\(.*open\("; then
    if ! echo "$NEW_STRING" | grep -q "tempfile\|os\.replace\|atomic"; then
        echo "WARNUNG: NEVER 6 — JSON direkt schreiben erkannt. Nutze atomic write (tempfile + os.replace)."
    fi
fi

# ============================================================
# NEVER 5: subprocess.Popen ohne timeout
# ============================================================
if echo "$NEW_STRING" | grep -q "subprocess\.Popen\|subprocess\.run"; then
    if ! echo "$NEW_STRING" | grep -q "timeout"; then
        echo "WARNUNG: NEVER 5 — subprocess ohne timeout erkannt. Immer timeout=30 setzen."
    fi
fi

exit 0
