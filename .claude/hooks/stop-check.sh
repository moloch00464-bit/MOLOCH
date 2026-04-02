#!/bin/bash
# Stop Hook: Prueft ob uncommitted Changes oder Syntax-Fehler bleiben
# Exit 2 = Claude darf nicht aufhoeren, Exit 0 = OK

INPUT=$(cat)

# Endlos-Loop vermeiden
STOP_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
if [ "$STOP_ACTIVE" = "true" ]; then
    exit 0
fi

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || cd /home/user/MOLOCH

# Uncommitted Changes pruefen
DIRTY=$(git status --porcelain 2>/dev/null | grep -v "^??" | head -5)
if [ -n "$DIRTY" ]; then
    echo "HINWEIS: Es gibt noch uncommitted Changes:"
    echo "$DIRTY"
    echo ""
    echo "Vergiss nicht zu committen wenn die Aenderungen fertig sind."
fi

exit 0
