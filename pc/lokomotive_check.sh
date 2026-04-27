#!/usr/bin/env bash
# Lokomotive PC-Cowork Check — vollstaendiger Session-Start-Workflow als Bash-Skript.
# Output strukturiert mit ✓/⚠/✗. Ruft alle 8 Schritte der pc-cowork-startup Skill auf
# plus Auto-Pipeline-Detection (was hat Daemon ohne uns gemacht).

set -u
REPO="${MOLOCH_REPO:-$HOME/moloch_repo}"
LOG_DIR="$HOME/moloch_logs"

echo "==========================================="
echo " LOKOMOTIVE PC-COWORK CHECK"
echo "==========================================="

# [1] Repo-Sync
echo ""
echo "[1] Repo-Sync"
if ! cd "$REPO" 2>/dev/null; then echo "  ✗ repo not found: $REPO"; exit 1; fi
git fetch -q origin main 2>&1
HEAD_LOCAL=$(git rev-parse HEAD)
HEAD_ORIGIN=$(git rev-parse origin/main)
if [ "$HEAD_LOCAL" != "$HEAD_ORIGIN" ]; then
    echo "  ⚠ behind origin — running pull --rebase"
    git pull --rebase 2>&1 | tail -2
fi
echo "  ✓ HEAD: $(git log -1 --oneline)"

# [2] Diff since last 24h
echo ""
echo "[2] Commits letzte 24h"
SINCE=$(date -d "1 day ago" '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d' 2>/dev/null || echo "")
if [ -n "$SINCE" ]; then
    COMMITS_24H=$(git log --since="$SINCE" --oneline 2>/dev/null | wc -l | tr -d ' ')
    echo "  $COMMITS_24H Commits"
    git log --since="$SINCE" --oneline 2>/dev/null | head -5 | sed 's/^/    /'
fi

# [3] Smoke
echo ""
echo "[3] Smoke-Test"
if cmd //c "pc\\smoke.cmd" >/tmp/lkmt_smoke.log 2>&1; then
    echo "  ✓ smoke OK"
else
    echo "  ✗ smoke FAILED:"
    tail -8 /tmp/lkmt_smoke.log | sed 's/^/    /'
fi

# [4] Mailboxen
echo ""
echo "[4] Mailboxen"
PC_OPEN=$(grep -c "^status: open" docs/PC_TO_PI.md 2>/dev/null || echo 0)
PI_OPEN=$(grep -c "^status: open" docs/PI_TO_PC.md 2>/dev/null || echo 0)
echo "  PC_TO_PI ($PC_OPEN open) — top:"
grep -nE "^## \[" docs/PC_TO_PI.md 2>/dev/null | head -3 | sed 's/^/    /'
echo "  PI_TO_PC ($PI_OPEN open) — top:"
grep -nE "^## \[" docs/PI_TO_PC.md 2>/dev/null | head -3 | sed 's/^/    /'

# [5] Service-Health
echo ""
echo "[5] Service-Health (5 Endpoints)"
for entry in "adapter|11600/health" "dashboard|11700/api/state" "avatar|11800/api/state" "pi-cockpit|9000/feedback_stats" "ollama|11434/api/tags"; do
    name="${entry%%|*}"
    path="${entry#*|}"
    code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:$path" 2>/dev/null || echo "ERR")
    if [ "$code" = "200" ]; then echo "  ✓ $name :$path"
    else echo "  ✗ $name :$path HTTP=$code"; fi
done

# [6] Auto-Pipeline-Detection — was hat Daemon autonom gemacht?
echo ""
echo "[6] Daemon Auto-Aktivitaet (24h)"
AUTO_COMMITS=$(git log --since="$SINCE" --oneline --author="cowork-monitor\\|cowork-claude-auto" 2>/dev/null | wc -l | tr -d ' ')
echo "  Auto-Commits: $AUTO_COMMITS"
git log --since="$SINCE" --oneline --author="cowork-monitor\\|cowork-claude-auto" 2>/dev/null | head -3 | sed 's/^/    /'
if [ -f "$LOG_DIR/cross_session.jsonl" ]; then
    FED_REPLIES=$(grep -c '"kind": "federation_reply"' "$LOG_DIR/cross_session.jsonl" 2>/dev/null || echo 0)
    echo "  federation_reply events (lifetime): $FED_REPLIES"
fi

# [7] Pool + Adapter
echo ""
echo "[7] Pool + Adapter"
POOL=$(curl -sS --max-time 3 http://localhost:9000/feedback_stats 2>/dev/null)
if [ -n "$POOL" ]; then echo "  Pool: $POOL"
else echo "  ✗ pool stats not reachable"; fi
ADAPTERS=$(curl -sS --max-time 3 http://localhost:11600/list 2>/dev/null)
if [ -n "$ADAPTERS" ]; then echo "  Adapter: $ADAPTERS"
else echo "  ✗ adapter list not reachable"; fi

# [8] Daemon-Heartbeat-Alter
echo ""
echo "[8] Daemon-Heartbeat"
if [ -f "$LOG_DIR/cross_session.jsonl" ]; then
    LAST_TS=$(tail -1 "$LOG_DIR/cross_session.jsonl" | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('iso',''))" 2>/dev/null)
    if [ -n "$LAST_TS" ]; then
        AGE=$(python -c "import datetime; print(int((datetime.datetime.now() - datetime.datetime.fromisoformat('$LAST_TS')).total_seconds()))" 2>/dev/null)
        if [ -n "$AGE" ]; then
            if [ "$AGE" -lt 90 ]; then echo "  ✓ heartbeat ${AGE}s alt — Daemon lebt"
            elif [ "$AGE" -lt 600 ]; then echo "  ⚠ heartbeat ${AGE}s alt — verzoegert"
            else echo "  ✗ heartbeat ${AGE}s alt — Daemon vermutlich tot"; fi
        fi
    fi
fi

# [9] Federation status
echo ""
echo "[9] Federation"
if [ -f "$LOG_DIR/fed_kill" ]; then echo "  fed_kill marker AKTIV — Federation aus (Stand 2026-04-27)"
else echo "  fed_kill marker fehlt — Federation evtl. aktiv (Daemon-Trigger werden auf claude -p geprueft)"; fi

# [10] Summary
echo ""
echo "==========================================="
echo " SUMMARY"
echo "==========================================="
echo " HEAD: $(git log -1 --oneline)"
echo " Letzte Pi-Mailbox-Top: $(grep -m1 -E "^## \[" docs/PI_TO_PC.md 2>/dev/null | sed 's/^## //')"
echo " Letzte PC-Mailbox-Top: $(grep -m1 -E "^## \[" docs/PC_TO_PI.md 2>/dev/null | sed 's/^## //')"
echo " Open: PC=$PC_OPEN PI=$PI_OPEN"
echo " Auto-Pipeline 24h: $AUTO_COMMITS commits"
echo "==========================================="
