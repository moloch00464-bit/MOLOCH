"""Audit-Orchestrator — Welle 8 Fundament.

Aggregiert vier Layer in `/dev/shm/audit_state.json`:
1. **Pi-Health** aus `moloch_audit.py --auto` (`logs/audit_last.json`)
2. **PC-Health** aus Mailbox-POST `/mailbox/audit/pc_health` (W9 PC-Side)
3. **Persona-Drift** aus character_journal (W10, vorerst leer/optional)
4. **Mailbox-Hygiene** aus Mailbox-POST `/mailbox/audit/hygiene` (W9)

Schema:
```
{
  "overall": "green|warn|red",
  "updated_at": "ISO-Timestamp",
  "layers": {
    "pi": {"score": int, "max": int, "status": "PASS|WARN|FAIL", "detail": {...}},
    "pc": {"score": int, "max": int, "status": "...", "detail": {...}},
    "persona": {"avg": float, "sparkline": [...], "status": "..."},
    "mailbox": {"backlog_pc": int, "backlog_pi": int, "stale": int, "dups": int, "status": "..."}
  },
  "drift_events": [{"ts", "layer", "signal", "severity"}, ...],
  "alarm_tier": "silent|warn|alert"
}
```

CLI:
- `python3 -m core.audit.audit_orchestrator --once` → ein Tick + exit
- `python3 -m core.audit.audit_orchestrator --loop` → Endlos-Loop, 60s Intervall

Atomic-Write via tempfile + os.replace (NEVER-Regel 6).
Subprocess immer mit timeout=30 (NEVER-Regel 5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("audit_orchestrator")

MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
AUDIT_STATE_PATH = Path("/dev/shm/audit_state.json")
PI_AUDIT_JSON = MOLOCH_DIR / "logs" / "audit_last.json"
CROSS_SESSION_LOG = Path(os.path.expanduser("~/moloch_logs/cross_session.jsonl"))
LOOP_INTERVAL_S = 60
SUBPROCESS_TIMEOUT_S = 30
DRIFT_EVENT_MAX = 50
SPARKLINE_LEN = 50


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> bool:
    """Atomic JSON-Schreibe via tempfile + os.replace (NEVER 6)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent),
                                         prefix=path.name + ".",
                                         suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(path))
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning("[audit] Schreibe-Fehler %s: %s", path, e)
        return False


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Best-effort JSON-Read. None bei Fehler."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _run_pi_audit() -> Dict[str, Any]:
    """Subprocess `moloch_audit.py --auto` -> parse audit_last.json -> Layer."""
    try:
        subprocess.run(
            ["python3", str(MOLOCH_DIR / "moloch_audit.py"), "--auto"],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            cwd=str(MOLOCH_DIR),
        )
    except subprocess.TimeoutExpired:
        return {"score": 0, "max": 0, "status": "FAIL",
                "detail": {"error": "audit timeout"}}
    except Exception as e:
        return {"score": 0, "max": 0, "status": "FAIL",
                "detail": {"error": str(e)[:200]}}
    data = _read_json_safe(PI_AUDIT_JSON) or {}
    # moloch_audit.py-Schema: {overall: 'PASS', checks: {Name: {status, message}, ...}}
    # Plus defensive Fallbacks fuer aeltere/andere Schemata.
    overall = (data.get("overall") or data.get("gesamtstatus")
               or data.get("Gesamtstatus") or data.get("status") or "").upper()
    checks = data.get("checks") or data.get("tests") or data.get("Tests") or {}
    passed, total = 0, 0
    if isinstance(checks, dict):
        for c in checks.values():
            if isinstance(c, dict):
                total += 1
                if (c.get("status") or "").upper() == "PASS":
                    passed += 1
    elif isinstance(checks, list):
        for c in checks:
            if isinstance(c, dict):
                total += 1
                if (c.get("status") or "").upper() == "PASS":
                    passed += 1
    if overall in ("PASS", "OK", "GREEN"):
        status = "PASS"
    elif overall in ("WARN", "WARNING", "YELLOW"):
        status = "WARN"
    elif overall:
        status = "FAIL"
    else:
        status = "WARN"  # Audit lief, kein klares overall
    return {
        "score": passed,
        "max": total,
        "status": status,
        "detail": {"overall_raw": overall, "tests_total": total},
    }


def _read_persona_layer(prev_layer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persona-Score aus character_journal lesen (W10 — vorerst leer wenn keine Daten)."""
    sparkline: List[float] = []
    avg: Optional[float] = None
    try:
        from core.memory.character_journal import get_journal  # type: ignore
        j = get_journal()
        # Best-effort: Suche persona_score-Events. Falls API anders heisst,
        # bleibt Layer leer (kein Crash).
        getter = getattr(j, "get_recent_events", None) or getattr(j, "recent_events", None)
        events = []
        if callable(getter):
            try:
                events = getter(type_filter="persona_score", limit=SPARKLINE_LEN) or []
            except TypeError:
                try:
                    events = getter(SPARKLINE_LEN) or []
                except Exception:
                    events = []
        for ev in events:
            score = ev.get("score") if isinstance(ev, dict) else None
            if isinstance(score, (int, float)):
                sparkline.append(float(score))
        if sparkline:
            avg = sum(sparkline) / len(sparkline)
    except Exception:
        pass
    if not sparkline:
        # Vorlauf-Daten aus prev_layer behalten falls vorhanden
        if prev_layer and isinstance(prev_layer.get("sparkline"), list):
            sparkline = list(prev_layer["sparkline"])[-SPARKLINE_LEN:]
            avg = (sum(sparkline) / len(sparkline)) if sparkline else None
    if avg is None:
        return {"avg": None, "sparkline": sparkline, "status": "PENDING"}
    if avg >= 7:
        status = "PASS"
    elif avg >= 5:
        status = "WARN"
    else:
        status = "FAIL"
    return {"avg": round(avg, 2), "sparkline": sparkline, "status": status}


def _compute_overall(layers: Dict[str, Dict[str, Any]]) -> str:
    """green wenn alle PASS/PENDING, warn bei einem WARN, red bei FAIL."""
    statuses = []
    for layer in layers.values():
        if isinstance(layer, dict):
            statuses.append((layer.get("status") or "").upper())
    if any(s == "FAIL" for s in statuses):
        return "red"
    if any(s == "WARN" for s in statuses):
        return "warn"
    return "green"


def _compute_alarm_tier(state: Dict[str, Any]) -> str:
    """Alarm-Tier basierend auf drift_events der letzten Stunde + persona_avg."""
    now = time.time()
    one_hour_ago = now - 3600
    fails_last_hour = 0
    for ev in state.get("drift_events", []) or []:
        ts_str = ev.get("ts", "")
        sev = (ev.get("severity") or "").upper()
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts_dt.timestamp() >= one_hour_ago and sev in ("FAIL", "ALERT"):
                fails_last_hour += 1
        except Exception:
            continue
    persona = (state.get("layers") or {}).get("persona") or {}
    persona_avg = persona.get("avg")
    if persona_avg is not None and persona_avg < 3:
        return "alert"
    if fails_last_hour >= 5:
        return "alert"
    if fails_last_hour >= 3:
        return "warn"
    if persona_avg is not None and persona_avg < 5:
        # nur wenn 10+ datapoints, sonst silent
        sparkline = persona.get("sparkline") or []
        if len(sparkline) >= 10:
            return "warn"
    return "silent"


def _collect_drift_events(prev: Optional[Dict[str, Any]],
                          current_layers: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Diff prev vs current Layer-Status — neue Events mit ts."""
    events: List[Dict[str, Any]] = []
    prev_layers = (prev or {}).get("layers") or {}
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, cur in current_layers.items():
        if not isinstance(cur, dict):
            continue
        prev_layer = prev_layers.get(name) or {}
        cur_status = (cur.get("status") or "").upper()
        prev_status = (prev_layer.get("status") or "").upper()
        if prev_status and cur_status and prev_status != cur_status:
            severity = "ALERT" if cur_status == "FAIL" else (
                "WARN" if cur_status == "WARN" else "INFO"
            )
            events.append({
                "ts": now_iso,
                "layer": name,
                "signal": f"{prev_status} -> {cur_status}",
                "severity": severity,
            })
    # Append neue Events an existierende Liste, max DRIFT_EVENT_MAX behalten
    existing = (prev or {}).get("drift_events") or []
    combined = (existing + events)[-DRIFT_EVENT_MAX:]
    return combined


def run_once() -> Dict[str, Any]:
    """Ein Tick — sammelt alle Layer + schreibt audit_state.json atomic."""
    prev = _read_json_safe(AUDIT_STATE_PATH) or {}
    prev_layers = prev.get("layers") or {}

    pi_layer = _run_pi_audit()
    # PC + Mailbox bleiben aus prev-Lauf (Mailbox-Receiver merget rein).
    # Wenn noch nie gepostet -> leer/PENDING.
    pc_layer = prev_layers.get("pc") or {
        "score": 0, "max": 0, "status": "PENDING", "detail": {}
    }
    mailbox_layer = prev_layers.get("mailbox") or {
        "backlog_pc": 0, "backlog_pi": 0, "stale": 0, "dups": 0, "status": "PENDING"
    }
    persona_layer = _read_persona_layer(prev_layers.get("persona"))

    layers = {
        "pi": pi_layer,
        "pc": pc_layer,
        "persona": persona_layer,
        "mailbox": mailbox_layer,
    }
    drift_events = _collect_drift_events(prev, layers)
    state = {
        "overall": _compute_overall(layers),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": layers,
        "drift_events": drift_events,
        "alarm_tier": "silent",
    }
    state["alarm_tier"] = _compute_alarm_tier(state)
    _atomic_write_json(AUDIT_STATE_PATH, state)
    return state


def merge_component(component: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Receiver-Hilfsfunktion: merget ein Component-Update in audit_state.layers.

    Wird vom chat_server.py-Endpoint POST /mailbox/audit/{component} genutzt.
    component in {pc_health, hygiene, persona}.
    """
    valid = {"pc_health": "pc", "hygiene": "mailbox", "persona": "persona"}
    layer_key = valid.get(component)
    if layer_key is None:
        return None
    state = _read_json_safe(AUDIT_STATE_PATH) or {
        "overall": "warn",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": {},
        "drift_events": [],
        "alarm_tier": "silent",
    }
    state.setdefault("layers", {})
    state["layers"][layer_key] = data
    state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["overall"] = _compute_overall(state["layers"])
    state["alarm_tier"] = _compute_alarm_tier(state)
    _atomic_write_json(AUDIT_STATE_PATH, state)
    return state


def _main() -> int:
    parser = argparse.ArgumentParser(description="Audit-Orchestrator (Welle 8)")
    parser.add_argument("--once", action="store_true", help="ein Tick + exit")
    parser.add_argument("--loop", action="store_true",
                        help=f"Endlos-Loop, {LOOP_INTERVAL_S}s Intervall")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.once or (not args.once and not args.loop):
        state = run_once()
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0

    if args.loop:
        logger.info("[audit] Loop start, Intervall=%ss", LOOP_INTERVAL_S)
        while True:
            try:
                state = run_once()
                logger.info("[audit] tick overall=%s tier=%s",
                            state.get("overall"), state.get("alarm_tier"))
            except Exception as e:
                logger.warning("[audit] tick-Fehler: %s", e)
            time.sleep(LOOP_INTERVAL_S)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
