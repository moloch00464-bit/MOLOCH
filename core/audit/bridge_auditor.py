"""Bridge-Auditor (Welle 14).

Pullt:
- L0: GET http://localhost:9100/health (chat_server alive)
- L1: PC-Heartbeat via audit_state.layers.pc.detail.last_seen_age_s
      ODER ~/moloch_logs/cross_session.jsonl (letzte Zeile)
- L2: Mailbox-Latenz docs/PC_TO_PI.md mtime
      Tentakel-Reachability http://192.168.178.20:11434/api/tags (best-effort)

Schreibt audit_state.layers.bridge Schema:
  {chat_server_alive, pc_heartbeat_age_s, mailbox_pc_to_pi_age_s,
   tentakel_reachable, score, max, status, detail}

Status-Logik:
- PASS: alles erreichbar, Heartbeat <90s
- WARN: Tentakel offline ODER Heartbeat stale >5min
- FAIL: chat_server selber nicht erreichbar
"""
from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("bridge_auditor")

_AUDIT_STATE = "/dev/shm/audit_state.json"
_CROSS_SESSION = os.path.expanduser("~/moloch_logs/cross_session.jsonl")
_MAILBOX_PC_TO_PI = "/home/molochzuhause/moloch/docs/PC_TO_PI.md"
_SETTINGS_PATH = "/home/molochzuhause/moloch/config/settings.json"


def _http_ok(url: str, timeout: int = 3) -> Optional[int]:
    """GET url, returnt Status-Code oder None bei Fehler."""
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        return r.status_code
    except Exception:
        return None


def _tentakel_url() -> str:
    """Liest tentacle_llm.host:port aus settings.json, default 192.168.178.20:11434."""
    host = "192.168.178.20"
    port = 11434
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        t = s.get("tentacle_llm", {}) or {}
        host = t.get("host", host)
        port = int(t.get("port", port))
    except Exception:
        pass
    return f"http://{host}:{port}/api/tags"


def _pc_heartbeat_from_audit_state() -> Optional[float]:
    try:
        with open(_AUDIT_STATE, "r", encoding="utf-8") as f:
            st = json.load(f)
        layers = st.get("layers", {}) or {}
        pc = layers.get("pc", {}) or {}
        d = pc.get("detail", {}) or {}
        for key in ("last_seen_age_s", "heartbeat_age_s", "age_s"):
            if key in d and d[key] is not None:
                return float(d[key])
    except Exception:
        pass
    return None


def _pc_heartbeat_from_jsonl() -> Optional[float]:
    if not os.path.exists(_CROSS_SESSION):
        return None
    try:
        # Letzte Zeile lesen
        with open(_CROSS_SESSION, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 4096)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore").splitlines()
        for ln in reversed(tail):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                ts = obj.get("ts") or obj.get("timestamp") or obj.get("time")
                if ts is None:
                    continue
                if isinstance(ts, (int, float)):
                    return max(0.0, time.time() - float(ts))
                # ISO-String
                try:
                    import datetime
                    dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    return max(0.0, time.time() - dt.timestamp())
                except Exception:
                    continue
            except Exception:
                continue
    except Exception:
        return None
    return None


def collect() -> Dict[str, Any]:
    """Sammelt Bridge-Layer-Daten."""
    detail: Dict[str, Any] = {}

    # L0: chat_server health
    code = _http_ok("http://localhost:9100/health", timeout=3)
    chat_alive = code == 200
    detail["chat_server_status_code"] = code

    # L1: PC-Heartbeat
    pc_age = _pc_heartbeat_from_audit_state()
    if pc_age is None:
        pc_age = _pc_heartbeat_from_jsonl()
    detail["pc_heartbeat_source"] = (
        "audit_state" if _pc_heartbeat_from_audit_state() is not None
        else ("cross_session_jsonl" if pc_age is not None else "none")
    )

    # L2a: Mailbox PC_TO_PI mtime
    mailbox_age = None
    if os.path.exists(_MAILBOX_PC_TO_PI):
        try:
            mailbox_age = round(time.time() - os.path.getmtime(_MAILBOX_PC_TO_PI), 1)
        except Exception:
            pass
    detail["mailbox_pc_to_pi_path"] = _MAILBOX_PC_TO_PI

    # L2b: Tentakel-Reachability
    tentakel_url = _tentakel_url()
    tentakel_code = _http_ok(tentakel_url, timeout=3)
    tentakel_reachable = tentakel_code == 200
    detail["tentakel_url"] = tentakel_url
    detail["tentakel_status_code"] = tentakel_code

    # Status-Berechnung
    # PC_HEARTBEAT_STALE_THRESHOLD_S: 7200s (2h). Konsistent mit transition_auditor.
    # Frueher 300s (5min) — zu aggressiv fuer nachts-aus-PC. PC kann mehrere
    # Stunden idle sein ohne dass das ein Bug ist.
    PC_HEARTBEAT_STALE = 7200
    score = 0
    max_score = 4
    if chat_alive:
        score += 1
    if pc_age is not None and pc_age < 90:
        score += 1
    if mailbox_age is not None and mailbox_age < 24 * 3600:
        score += 1
    if tentakel_reachable:
        score += 1

    if not chat_alive:
        status = "FAIL"
    elif not tentakel_reachable or (pc_age is not None and pc_age > PC_HEARTBEAT_STALE):
        status = "WARN"
    elif pc_age is None:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "chat_server_alive": chat_alive,
        "pc_heartbeat_age_s": round(pc_age, 1) if pc_age is not None else None,
        "mailbox_pc_to_pi_age_s": mailbox_age,
        "tentakel_reachable": tentakel_reachable,
        "detail": detail,
    }
