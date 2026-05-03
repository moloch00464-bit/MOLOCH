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


def _safe_collect(module_name: str, fallback_status: str = "PENDING") -> Dict[str, Any]:
    """Best-effort Sub-Auditor-Aufruf. Bei Fehler: PENDING-Layer mit error-detail."""
    try:
        mod = __import__(f"core.audit.{module_name}", fromlist=["collect"])
        return mod.collect()
    except Exception as e:
        logger.warning("[audit] %s collect-Fehler: %s", module_name, e)
        return {"score": 0, "max": 0, "status": fallback_status,
                "detail": {"error": str(e)[:200]}}


def _safe_collect_self_diagnosis() -> Dict[str, Any]:
    """W14: lese /dev/shm/audit_self_diagnosis.json (Snapshot vom Timer-Run)."""
    snap = _read_json_safe(Path("/dev/shm/audit_self_diagnosis.json"))
    if not snap:
        return {"score": 0, "max": 0, "status": "PENDING",
                "detail": {"reason": "snapshot_missing — Timer noch nicht gelaufen"}}
    # Snapshot-Wrapper {ts, iso, result: {...}} -> result entpacken
    return snap.get("result", snap)


def _safe_collect_expression_state() -> Dict[str, Any]:
    """W16: liest expression_state primaer aus /dev/shm/expression_state.json
    (cross-prozess-fix), fallback auf Singleton-Getter."""
    EXPR_PATH = Path("/dev/shm/expression_state.json")
    snap = _read_json_safe(EXPR_PATH)
    if not snap:
        # Fallback: Singleton-Getter (gibt PENDING wenn audit in eigenem Prozess laeuft)
        try:
            from core.audit.expression.expression_orchestrator import get_expression_state  # type: ignore
            snap = get_expression_state() or {}
        except Exception as e:
            return {"score": 0, "max": 5, "status": "PENDING",
                    "detail": {"error": str(e)[:200]}}
        if not snap:
            return {"score": 0, "max": 5, "status": "PENDING",
                    "detail": {"reason": "expression_state.json fehlt + Singleton leer"}}

    # Adaption auf Audit-Schema (analog zu vorher)
    alive_count = int(snap.get("alive_count", 0) or 0)
    modules = snap.get("modules", {}) or {}
    total = len(modules)
    if total == 0:
        return {"score": 0, "max": 5, "status": "PENDING",
                "detail": {"reason": "expression_orchestrator nicht gestartet"}}
    if alive_count == total:
        status = "PASS"
    elif alive_count >= total // 2:
        status = "WARN"
    else:
        status = "FAIL"
    return {"score": alive_count, "max": total, "status": status,
            "alive_count": alive_count, "modules": list(modules.keys()),
            "detail": snap}


def _append_pi_heartbeat(state: Dict[str, Any]) -> None:
    """Drift 7 Pi-Side Heartbeat: appendet pro audit-Tick eine Zeile in
    cross_session.jsonl, damit transition.federation_heartbeat alive bleibt
    (PC's cross_session_monitor schreibt nur PC-Side, Pi muss selbst pingen).
    Best-effort, crasht NIE den Orchestrator."""
    try:
        CROSS_SESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": "moloch",
            "event": "pi_audit_tick",
            "source": "audit_orchestrator",
            "overall": state.get("overall"),
            "alarm_tier": state.get("alarm_tier"),
            "layer_count": len(state.get("layers", {})),
        }
        with open(CROSS_SESSION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"[audit] heartbeat-append fail: {e}")


def _safe_collect_capabilities() -> Dict[str, Any]:
    """W17: lese capability_inventory aus self_awareness."""
    try:
        from core.audit.self_awareness.capability_inventory import collect_capabilities  # type: ignore
        return collect_capabilities()
    except Exception as e:
        return {"score": 0, "max": 0, "status": "PENDING",
                "can_do": [], "cannot_do": [], "summary_de": "",
                "detail": {"error": str(e)[:200]}}


def _safe_collect_reflections() -> Dict[str, Any]:
    """W17: lese failure_reflection."""
    try:
        from core.audit.self_awareness.failure_reflection import reflect_on_failures  # type: ignore
        return reflect_on_failures()
    except Exception as e:
        return {"score": 0, "max": 0, "status": "PENDING",
                "reflections_de": [], "incidents_24h": [],
                "detail": {"error": str(e)[:200]}}


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

    # Welle 12: Pi-Side Sub-Auditoren live (vision/npu/spotify/hardware)
    vision_layer = _safe_collect("vision_auditor")
    npu_layer = _safe_collect("npu_auditor")
    spotify_layer = _safe_collect("spotify_auditor")
    hardware_layer = _safe_collect("hardware_auditor")

    # PC-Side W12 (PC-Cowork POSTet via /mailbox/audit/{pc_hardware,web_ui})
    pc_hardware_layer = prev_layers.get("pc_hardware") or {
        "score": 0, "max": 0, "status": "PENDING", "detail": {}
    }
    web_ui_layer = prev_layers.get("web_ui") or {
        "score": 0, "max": 0, "status": "PENDING", "detail": {}
    }
    # W19 Web-Pipeline: PC web_pipeline_auditor POSTet alle 5 Min nach
    # /mailbox/audit/web_search — bleibt PENDING bis erster POST eintrifft.
    web_search_layer = prev_layers.get("web_search") or {
        "score": 0, "max": 0, "status": "PENDING", "detail": {}
    }

    # Welle 13: Innere Subsysteme L0-L2
    personality_layer = _safe_collect("personality_auditor")
    memory_layer = _safe_collect("memory_auditor")
    tracking_layer = _safe_collect("tracking_auditor")
    autonomy_layer = _safe_collect("autonomy_auditor")
    awareness_layer = _safe_collect("awareness_auditor")
    voice_layer = _safe_collect("voice_auditor")

    # Welle 14: Restkern + Cross-Cutting + Self-Diagnose-Snapshot
    unconscious_layer = _safe_collect("unconscious_auditor")
    bridge_layer = _safe_collect("bridge_auditor")
    tentacle_layer = _safe_collect("tentacle_auditor")
    cross_layer = _safe_collect("cross_auditor")
    self_diagnosis_layer = _safe_collect_self_diagnosis()

    # Welle 16: Expression-Lifecycle-Status (best-effort; PENDING bis service start_all_expressions ruft)
    expression_layer = _safe_collect_expression_state()

    # Welle 17: Self-Awareness — Capabilities + Failure-Reflection
    capability_layer = _safe_collect_capabilities()
    reflection_layer = _safe_collect_reflections()

    # Welle 21 B4: Agent-Tools-Layer (Smoketest der 5 W21-Tools)
    agent_tools_layer = _safe_collect("agent_tools_auditor")

    # Transition: 7-Kanaele-Health fuer Pi<->PC-Uebergang
    transition_layer = _safe_collect("transition_auditor")

    # Phase 1 Drei-Hirn-Synthese: state_engine 4-Tests
    state_engine_layer = _safe_collect("state_engine_auditor")

    layers = {
        "pi": pi_layer,
        "pc": pc_layer,
        "persona": persona_layer,
        "mailbox": mailbox_layer,
        # W12 Pi-Side
        "vision": vision_layer,
        "npu": npu_layer,
        "spotify": spotify_layer,
        "hardware": hardware_layer,
        # W12 PC-Side (von Cowork-POST)
        "pc_hardware": pc_hardware_layer,
        "web_ui": web_ui_layer,
        # W19 Web-Pipeline (von PC-Cowork-POST alle 5 Min)
        "web_search": web_search_layer,
        # W13
        "personality": personality_layer,
        "memory": memory_layer,
        "tracking": tracking_layer,
        "autonomy": autonomy_layer,
        "awareness": awareness_layer,
        "voice": voice_layer,
        # W14
        "unconscious": unconscious_layer,
        "bridge": bridge_layer,
        "tentacle": tentacle_layer,
        "cross": cross_layer,
        "self_diagnosis": self_diagnosis_layer,
        # W16 Expression-Lifecycle (Hardware-als-Ausdruck)
        "expression": expression_layer,
        # W17 Self-Awareness
        "capability": capability_layer,
        "reflection": reflection_layer,
        # W21 B4 Agent-Tools-Smoketest
        "agent_tools": agent_tools_layer,
        # Transition: Pi<->PC-Uebergang (7 Kanaele + e2e)
        "transition": transition_layer,
        # Phase 1 Drei-Hirn-Synthese (state_engine + transition + logger + identity)
        "state_engine": state_engine_layer,
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
    # Drift 7: Pi-Side Heartbeat fuer transition.federation_heartbeat
    _append_pi_heartbeat(state)
    return state


def merge_component(component: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Receiver-Hilfsfunktion: merget ein Component-Update in audit_state.layers.

    Wird vom chat_server.py-Endpoint POST /mailbox/audit/{component} genutzt.
    component in {pc_health, hygiene, persona}.
    """
    valid = {
        # W8 (existing)
        "pc_health": "pc", "hygiene": "mailbox", "persona": "persona",
        # W12 — PC-Side + Pi-Side Sub-Auditoren landen in eigenen layers
        "pc_hardware": "pc_hardware", "web_ui": "web_ui",
        "vision": "vision", "npu": "npu",
        "spotify": "spotify", "hardware": "hardware",
        # W19 — Web-Pipeline-Audit (PC-Cowork web_pipeline_auditor)
        "web_search": "web_search",
        # W13 — alle Sub-Domains als eigene Layer
        "personality": "personality", "memory": "memory", "tracking": "tracking",
        "autonomy": "autonomy", "awareness": "awareness", "voice": "voice",
        # W14
        "bridge": "bridge", "tentacle": "tentacle", "unconscious": "unconscious",
        "cross": "cross", "self_diagnosis": "self_diagnosis",
        # W16 / W17
        "expression": "expression", "capability": "capability",
        "reflection": "reflection",
        # W21 B4
        "agent_tools": "agent_tools",
        # Transition (Pi<->PC)
        "transition": "transition",
    }
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
