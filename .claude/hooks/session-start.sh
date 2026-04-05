#!/bin/bash
# SessionStart Hook: Laedt automatisch Status + Handoff bei jeder Session
# Wird bei startup, resume, clear und compact ausgefuehrt

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('source', 'startup'))
except:
    print('startup')
" 2>/dev/null)

# Lock erzeugen — Code-Edits blockiert bis moloch_status() gelaufen ist
touch /tmp/moloch_session_lock

echo "=== MOLOCH SESSION START ($SOURCE) ==="
echo ""
echo "  ██╗      ██████╗ ██╗  ██╗ ██████╗ ███╗   ███╗ ██████╗ ████████╗██╗██╗   ██╗███████╗"
echo "  ██║     ██╔═══██╗██║ ██╔╝██╔═══██╗████╗ ████║██╔═══██╗╚══██╔══╝██║██║   ██║██╔════╝"
echo "  ██║     ██║   ██║█████╔╝ ██║   ██║██╔████╔██║██║   ██║   ██║   ██║██║   ██║█████╗  "
echo "  ██║     ██║   ██║██╔═██╗ ██║   ██║██║╚██╔╝██║██║   ██║   ██║   ██║╚██╗ ██╔╝██╔══╝  "
echo "  ███████╗╚██████╔╝██║  ██╗╚██████╔╝██║ ╚═╝ ██║╚██████╔╝   ██║   ██║ ╚████╔╝ ███████╗"
echo "  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═══╝  ╚══════╝"
echo ""
echo "  Codename: LOKOMOTIVE | Workflow aktiv | Alle Regeln gelten"
echo ""

# Git Status
echo "--- GIT STATUS ---"
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || cd /home/molochzuhause/moloch
DIRTY=$(git status --porcelain 2>/dev/null | head -5)
if [ -n "$DIRTY" ]; then
    echo "WARNUNG: Uncommitted Changes!"
    echo "$DIRTY"
else
    echo "Git: Clean"
fi

# Handoff von letzter Session
HANDOFF="$CLAUDE_PROJECT_DIR/logs/agent_handoff.md"
if [ -f "$HANDOFF" ]; then
    echo ""
    echo "--- LETZTE SESSION HANDOFF ---"
    head -30 "$HANDOFF"
fi

# Bug Report pruefen
BUGS="$CLAUDE_PROJECT_DIR/logs/bug_report.txt"
if [ -f "$BUGS" ]; then
    BUGCOUNT=$(wc -l < "$BUGS" 2>/dev/null)
    if [ "$BUGCOUNT" -gt 0 ]; then
        echo ""
        echo "--- OFFENE BUGS: $BUGCOUNT Eintraege ---"
        tail -5 "$BUGS"
    fi
fi

echo ""
echo "=== PFLICHT-STARTPROTOKOLL ==="
echo "BEVOR Du Code schreibst, MUSST Du:"
echo "  1. moloch_status()       — System-Status pruefen"
echo "  2. moloch_npu_workers()  — Worker-Health pruefen"
echo "  3. /moloch-dev laden     — NEVER-Regeln + Templates"
echo "KEIN SSH. KEIN cat /dev/shm/. NUR MCP-Tools."
echo "Skills: /moloch-mcp /moloch-agent /moloch-dev /moloch-npu /moloch-audit"
echo "================================"

exit 0
