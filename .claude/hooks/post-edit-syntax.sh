#!/bin/bash
# PostToolUse Hook fuer Edit|Write: Automatischer Syntax-Check nach jeder Aenderung
# Kann nicht blockieren (Edit ist schon passiert), aber warnt bei Fehlern

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null)

# Kein File oder kein Python = kein Check
[ -z "$FILE" ] && exit 0
[[ "$FILE" != *.py ]] && exit 0

# Syntax-Check via py_compile
RESULT=$(python3 -m py_compile "$FILE" 2>&1)
if [ $? -ne 0 ]; then
    echo "SYNTAX-FEHLER in $FILE:"
    echo "$RESULT"
    echo ""
    echo "SOFORT fixen bevor weitergemacht wird!"
else
    echo "Syntax OK: $(basename "$FILE")"
fi

exit 0
