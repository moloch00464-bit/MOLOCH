#!/bin/bash
# PreToolUse Hook fuer Bash: Prueft gefaehrliche Befehle
# Exit 2 = blockiert, Exit 0 = erlaubt

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null)

# Kein Command = kein Check
[ -z "$COMMAND" ] && exit 0

# ============================================================
# NEVER 11: __pycache__ nach Code-Aenderungen pruefen
# (Erinnerung bei Service-Restart ohne Cache-Clear)
# ============================================================
if echo "$COMMAND" | grep -q "systemctl restart moloch"; then
    if ! echo "$COMMAND" | grep -q "pycache\|__pycache__"; then
        echo "ERINNERUNG: NEVER 11 — Vor Service-Restart __pycache__ loeschen:"
        echo "  find ~/moloch/core -name '__pycache__' -exec rm -rf {} + 2>/dev/null"
    fi
fi

# ============================================================
# NEVER 12: Worktree-Warnung
# ============================================================
if echo "$COMMAND" | grep -q "systemctl.*moloch"; then
    CWD=$(echo "$INPUT" | jq -r '.cwd // ""' 2>/dev/null)
    if echo "$CWD" | grep -q "worktrees\|worktree"; then
        echo "WARNUNG: NEVER 12 — Du bist in einem Worktree! Service laeuft von ~/moloch/, nicht vom Worktree."
    fi
fi

# ============================================================
# Git Amend Warnung
# ============================================================
if echo "$COMMAND" | grep -q "git commit.*--amend"; then
    echo "WARNUNG: git amend erkannt — sicher dass kein neuer Commit besser waere?"
fi

# ============================================================
# Git add -A / git add . Warnung (Runtime-State)
# ============================================================
if echo "$COMMAND" | grep -qE "git add -A|git add \."; then
    echo "WARNUNG: NEVER 7 — 'git add -A' koennte Runtime-State Dateien (last_face_position.json etc.) einschliessen. Besser spezifische Dateien adden."
fi

exit 0
