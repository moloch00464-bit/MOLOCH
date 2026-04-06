#!/bin/bash
# UserPromptSubmit Hook: LOKOMOTIVE Auto-Audit bei jedem Auftrags-Start
# Prueft ob das Protokoll eingehalten wird — schnell, kein Block, nur Bericht.

INPUT=$(cat)

# Prueft nur wenn es ein echter Auftrag ist (kein internes Tool-Feedback)
PROMPT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', '')[:120])
except:
    print('')
" 2>/dev/null)

[ -z "$PROMPT" ] && exit 0

# ============================================================
# CHECK 1: Session-Lock (moloch_status noch nicht ausgefuehrt?)
# ============================================================
LOKOMOTIVE_OK=true
WARNINGS=""

if [ -f "/tmp/moloch_session_lock" ]; then
    WARNINGS="${WARNINGS}\n  [!] Session-Lock aktiv — moloch_status() noch nicht ausgefuehrt!"
    LOKOMOTIVE_OK=false
fi

# ============================================================
# CHECK 2: Veraltete Agent-Locks (vergessen aufzuraeumen?)
# ============================================================
STALE_LOCKS=""
for LOCK in /tmp/moloch_agent_*; do
    if [ -f "$LOCK" ]; then
        AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo 0) ))
        AGENT=$(basename "$LOCK" | sed 's/moloch_agent_//')
        if [ "$AGE" -gt 3600 ]; then
            STALE_LOCKS="${STALE_LOCKS} ${AGENT}(${AGE}s)"
            LOKOMOTIVE_OK=false
        fi
    fi
done
[ -n "$STALE_LOCKS" ] && WARNINGS="${WARNINGS}\n  [!] Veraltete Agent-Locks:${STALE_LOCKS} — rm /tmp/moloch_agent_*"

# ============================================================
# CHECK 3: Git-Status (uncommitted ROT-Dateien?)
# ============================================================
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || cd /home/molochzuhause/moloch
ROT_FILES="moloch_service.py tappas_pipeline.py camera.py hailo_manager.py core_integrator.py voice_pipeline.py autonomous_tracker.py"

DIRTY_ROT=""
for ROT in $ROT_FILES; do
    if git status --porcelain 2>/dev/null | grep -q "$ROT"; then
        DIRTY_ROT="${DIRTY_ROT} ${ROT}"
        LOKOMOTIVE_OK=false
    fi
done
[ -n "$DIRTY_ROT" ] && WARNINGS="${WARNINGS}\n  [!] Uncommitted ROT-Dateien:${DIRTY_ROT}"

# ============================================================
# CHECK 4: Alle Agenten-Dateien vorhanden?
# ============================================================
REQUIRED_AGENTS="vision hardware gui tracking voice service coordinates"
MISSING_AGENTS=""
for AGENT in $REQUIRED_AGENTS; do
    if [ ! -f "$CLAUDE_PROJECT_DIR/.claude/agents/${AGENT}.md" ]; then
        MISSING_AGENTS="${MISSING_AGENTS} ${AGENT}"
        LOKOMOTIVE_OK=false
    fi
done
[ -n "$MISSING_AGENTS" ] && WARNINGS="${WARNINGS}\n  [!] Fehlende Agent-Dateien:${MISSING_AGENTS}"

# ============================================================
# AUSGABE
# ============================================================
if [ "$LOKOMOTIVE_OK" = "true" ]; then
    echo "[LOKOMOTIVE] Protokoll-Check OK — alle Bedingungen erfuellt."
else
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║   LOKOMOTIVE AUDIT — Protokoll-Verletzung!   ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "$WARNINGS"
    echo ""
    echo "Behebe die Punkte BEVOR Du Code schreibst."
    echo "Lies: CLAUDE.md → PFLICHT-STARTPROTOKOLL"
    echo ""
fi

exit 0
