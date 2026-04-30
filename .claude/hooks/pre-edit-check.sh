#!/bin/bash
# PreToolUse Hook fuer Edit|Write: Agent-Lock, Domain-Check, ROT-Dateien, NEVER-Regeln
# Nutzt Python3 statt jq (jq nicht installiert auf Pi)
# Exit 2 = blockiert die Aktion, Exit 0 = erlaubt

INPUT=$(cat)

# JSON-Parsing via Python3
FILE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except:
    print('')
" 2>/dev/null)

NEW_STRING=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('new_string', ''))
except:
    print('')
" 2>/dev/null)

OLD_STRING=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('old_string', ''))
except:
    print('')
" 2>/dev/null)

# Kein File = kein Check
[ -z "$FILE" ] && exit 0

BASENAME=$(basename "$FILE")
DIRPATH=$(dirname "$FILE")

# ============================================================
# PFLICHT-STARTPROTOKOLL LOCK
# ============================================================
LOCK_FILE="/tmp/moloch_session_lock"
if [ -f "$LOCK_FILE" ]; then
    case "$FILE" in
        */.claude/*|*/tmp/*|*.md)
            ;;
        *)
            echo "BLOCKIERT: PFLICHT-STARTPROTOKOLL nicht ausgefuehrt!" >&2
            echo "Du MUSST zuerst moloch_status() ausfuehren. Lies CLAUDE.md Abschnitt 1." >&2
            exit 2
            ;;
    esac
fi

# ============================================================
# GRUEN-Liste: Diese Dateien/Pfade brauchen keinen Agent-Lock
# ============================================================
is_exempt() {
    case "$FILE" in
        */.claude/*|*/tmp/*|*/logs/*|*/docs/*) return 0 ;;
    esac
    case "$BASENAME" in
        *.md) return 0 ;;
    esac
    echo "$FILE" | grep -q "/scripts/" && return 0
    echo "$FILE" | grep -q "/config/.*\.json$" && [ "$BASENAME" != "settings.json" ] && return 0
    return 1
}

# ============================================================
# DOMAIN-BESTIMMUNG
# ============================================================
get_file_domain() {
    # Nach Verzeichnis
    case "$DIRPATH" in
        */perception*|*/hailo*)   echo "vision";      return ;;
        */mpo*)                   echo "tracking";    return ;;
        */mcp*)                   echo "service";     return ;;
        */personality*)           echo "personality"; return ;;
        */autonomy*)              echo "autonomy";    return ;;
        */bridge*)                echo "bridge";      return ;;
        */audit*)                 echo "audit";       return ;;
        */awareness*)             echo "awareness";   return ;;
        */memory*)                echo "memory";      return ;;
        */music*)                 echo "music";       return ;;
        */speech*|*/tts*)         echo "voice";       return ;;
        */hardware*)
            case "$BASENAME" in
                audio_pipeline.py) echo "voice"; return ;;
            esac
            echo "hardware"; return ;;
        */gui*|*/panel*)          echo "gui";         return ;;
    esac
    # Nach Dateiname
    case "$BASENAME" in
        moloch_service.py|core_integrator.py|ipc_router.py|\
        status.py|environment_watcher.py|\
        moloch_mcp_server.py|settings.json)
            echo "service" ;;
        tappas_pipeline.py|vision_workers.py|roi_dispatcher.py|face_pipeline.py|\
        pose_worker.py|perception_engine.py|inference_engine.py|model_scheduler.py|\
        model_orchestrator.py|hailo_manager.py|super_res_worker.py|low_light_processor.py)
            echo "vision" ;;
        autonomous_tracker.py|ptz_arbiter.py|ptz_tracker.py|action_bridge.py|\
        ptz_orchestrator.py|mode_manager.py|arbitration.py|person_reid.py)
            echo "tracking" ;;
        moloch_unified_panel.py|moloch_console.py|panel_*.py|popup_*.py)
            echo "gui" ;;
        voice_pipeline.py|hailo_whisper.py|tts.py|\
        moloch_sprache.py|keyword_handler.py)
            echo "voice" ;;
        camera.py|ptz_calibration.py|thermal_manager.py|\
        led_controller.py|rgb_led_controller.py|cloud_controller.py|\
        ewelink_controller.py|camera_manager.py)
            echo "hardware" ;;
        unconscious_engine.py|tao_engine.py|anima_mappings.py)
            echo "unconscious" ;;
        system_watchdog.py|diagnostics.py|capability_monitor.py)
            echo "watchdog" ;;
        local_llm_bridge.py|deepseek_client.py|llm_response.py)
            echo "deepseek" ;;
        spotify_controller.py)
            echo "music" ;;
        wifi_mic.py|camera_cloud_bridge.py)
            echo "tentacle" ;;
        longterm_memory.py|daily_learner.py|episodic_memory.py)
            echo "memory" ;;
        personality_engine.py|mood.py|tension_integrator.py|event_bus.py)
            echo "personality" ;;
        decision_engine.py|homeostasis.py|introspection.py|llm_bridge.py|\
        night_cycle.py|atmosphere.py|preference_learner.py)
            echo "autonomy" ;;
        activity_worker.py|context_evaluator.py|motion_analyzer.py|\
        room_map.py|world_state.py)
            echo "awareness" ;;
        moloch_audit.py|stress_test*.py|chaos_*.py)
            echo "stresstest" ;;
        *)
            echo "unknown" ;;
    esac
}

# ============================================================
# AGENT-PFLICHT: Main Claude ist NUR Koordinator
# ============================================================
if ! is_exempt; then
    # Aktiven Agent-Lock suchen
    ACTIVE_AGENT=""
    for LOCK in /tmp/moloch_agent_*; do
        if [ -f "$LOCK" ]; then
            ACTIVE_AGENT=$(basename "$LOCK" | sed 's/moloch_agent_//')
            break
        fi
    done

    # Kein Lock -> BLOCKIERT
    if [ -z "$ACTIVE_AGENT" ]; then
        FILE_DOMAIN=$(get_file_domain)
        echo "BLOCKIERT: Kein Agent-Lock aktiv fuer '$BASENAME'!" >&2
        echo "" >&2
        echo "Main Claude ist NUR Koordinator — kein direktes Code-Editieren." >&2
        echo "Spawne den zustaendigen Domain-Agenten:" >&2
        echo "  Zustaendige Domain: ${FILE_DOMAIN:-unbekannt}" >&2
        echo "" >&2
        echo "  service     -> moloch_service.py, IPC, core_integrator" >&2
        echo "  vision      -> TAPPAS, GStreamer, NPU, perception" >&2
        echo "  tracking    -> PTZ, Tracker, Arbiter, autonomous_tracker" >&2
        echo "  gui         -> Tkinter, panel_*.py, popups" >&2
        echo "  voice       -> Whisper, TTS, Piper, audio_pipeline" >&2
        echo "  hardware    -> ONVIF, Kamera, Thermal, LED, eWeLink" >&2
        echo "  personality -> PersonalityEngine, Mood, Tension, EventBus" >&2
        echo "  autonomy    -> DecisionEngine, Homeostasis, LLM-Bridge" >&2
        echo "  awareness   -> Activity, Context, Motion, RoomMap" >&2
        echo "  memory      -> Episodic, Persistent, Vector, ReID, Longterm" >&2
        echo "  watchdog    -> SystemWatchdog, Diagnostics, CapabilityMonitor" >&2
        echo "  music       -> Spotify, Track-Index, MusicMemory" >&2
        echo "  deepseek    -> hailo-ollama, LLM-Bridge, DeepSeek API" >&2
        echo "  tentacle    -> ESP32 WiFi-Mic, wifi_mic.py, camera_cloud_bridge" >&2
        echo "  unconscious -> TaoEngine, Unterbewusstsein, anima_mappings" >&2
        echo "  stresstest  -> Scripts, Tests, Chaos Engineering" >&2
        echo "" >&2
        echo "Agent setzt Lock: touch /tmp/moloch_agent_[name]" >&2
        exit 2
    fi

    # Domain-Check: Passt der Agent zur Datei?
    FILE_DOMAIN=$(get_file_domain)
    if [ "$FILE_DOMAIN" != "unknown" ] && [ "$ACTIVE_AGENT" != "$FILE_DOMAIN" ]; then
        echo "BLOCKIERT: Domain-Konflikt!" >&2
        echo "" >&2
        echo "  Aktiver Agent : moloch_agent_${ACTIVE_AGENT}" >&2
        echo "  Datei-Domain  : ${FILE_DOMAIN}  (${BASENAME})" >&2
        echo "" >&2
        echo "${BASENAME} gehoert NICHT zum ${ACTIVE_AGENT}-Agenten." >&2
        echo "Beende diesen Agenten (rm /tmp/moloch_agent_${ACTIVE_AGENT})," >&2
        echo "dann spawne: moloch_agent_${FILE_DOMAIN}" >&2
        exit 2
    fi
fi

# ============================================================
# ROT-Dateien Warnung
# ============================================================
ROT_FILES="moloch_service.py tappas_pipeline.py camera.py hailo_manager.py
core_integrator.py voice_pipeline.py autonomous_tracker.py
audio_pipeline.py ipc_router.py person_reid.py vision_workers.py
face_pipeline.py roi_dispatcher.py settings.json"

for ROT in $ROT_FILES; do
    if [ "$BASENAME" = "$ROT" ]; then
        echo "WARNUNG: $BASENAME ist eine ROT-Datei (System-Crash Risk)! Pre-Flight Check und BACKUP Pflicht."
        break
    fi
done

# ============================================================
# NEVER 2: Pan-Vorzeichen
# ============================================================
if [ "$BASENAME" = "camera.py" ]; then
    if echo "$OLD_STRING" | grep -q "pan_delta.*=.*-error_x\|pan_delta.*=.*error_x"; then
        echo "BLOCKIERT: NEVER 2 — Pan-Vorzeichen (pan_delta = -error_x) ist TABU." >&2
        exit 2
    fi
fi

# ============================================================
# NEVER: panel_styles.py
# ============================================================
if [ "$BASENAME" = "panel_styles.py" ]; then
    echo "BLOCKIERT: panel_styles.py darf nur geaendert werden wenn explizit beauftragt." >&2
    exit 2
fi

# ============================================================
# NEVER 7: Runtime-State
# ============================================================
if [ "$BASENAME" = "last_face_position.json" ] || [ "$BASENAME" = "learned_patrol_positions.json" ]; then
    echo "BLOCKIERT: NEVER 7 — Runtime-State Dateien gehoeren nicht in den Code." >&2
    exit 2
fi

# ============================================================
# NEVER 10: np.ndarray in moloch_service.py
# ============================================================
if [ "$BASENAME" = "moloch_service.py" ]; then
    if echo "$NEW_STRING" | grep -qE "def .*\(.*np\.ndarray"; then
        echo "BLOCKIERT: NEVER 10 — Kein np.ndarray als Type-Hint in moloch_service.py." >&2
        exit 2
    fi
fi

# ============================================================
# NEVER 3: ArcFace-Threshold nicht erhoehen
# ============================================================
if echo "$NEW_STRING" | grep -qE "arcface.*threshold|ARCFACE.*THRESH|similarity_threshold"; then
    if echo "$NEW_STRING" | grep -qE "0\.[7-9]|[1-9]\.[0-9]"; then
        echo "BLOCKIERT: NEVER 3 — ArcFace-Threshold NICHT erhoehen! Enrollment muss via gleichen Code-Pfad wie Live-Inference." >&2
        exit 2
    fi
fi

# ============================================================
# NEVER 4: Nicht mehrere ROT-Dateien in einer Session
# ============================================================
ROT_TRACK="/tmp/moloch_rot_edited"
for ROT in $ROT_FILES; do
    if [ "$BASENAME" = "$ROT" ]; then
        if [ -f "$ROT_TRACK" ]; then
            PREV_ROT=$(cat "$ROT_TRACK")
            if [ "$PREV_ROT" != "$BASENAME" ]; then
                echo "WARNUNG: NEVER 4 — Bereits ROT-Datei '$PREV_ROT' editiert! Jetzt auch '$BASENAME'." >&2
                echo "1 Commit = 1 ROT-Datei. Erst committen, dann naechste Datei." >&2
            fi
        else
            echo "$BASENAME" > "$ROT_TRACK"
        fi
        break
    fi
done

# ============================================================
# NEVER 8: shell=True
# ============================================================
if echo "$NEW_STRING" | grep -q "shell=True"; then
    echo "WARNUNG: NEVER 8 — shell=True erkannt! Nutze Liste statt String."
fi

# ============================================================
# NEVER 6: JSON nicht-atomic
# ============================================================
if echo "$NEW_STRING" | grep -qE "json\.dump\(.*open\("; then
    if ! echo "$NEW_STRING" | grep -q "tempfile\|os\.replace"; then
        echo "WARNUNG: NEVER 6 — JSON direkt schreiben. Nutze atomic write (tempfile + os.replace)."
    fi
fi

# ============================================================
# NEVER 5: subprocess ohne timeout
# ============================================================
if echo "$NEW_STRING" | grep -q "subprocess\.Popen\|subprocess\.run"; then
    if ! echo "$NEW_STRING" | grep -q "timeout"; then
        echo "WARNUNG: NEVER 5 — subprocess ohne timeout. Immer timeout=30 setzen."
    fi
fi

exit 0
