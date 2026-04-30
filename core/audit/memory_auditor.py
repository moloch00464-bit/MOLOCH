"""Memory-Layer-Auditor (Welle 13).

Misst Gedaechtnis-Subsysteme:
- Langzeitgedaechtnis (core.longterm_memory)
- CharacterJournal (core.memory.character_journal)
- FaceDB (worker_health.FaceWorker.face_db_entries)
- Qdrant (HTTP /collections, optional)

Schreibt audit_state.layers.memory:
  {face_db_entries, journal_events_1h, qdrant_alive, qdrant_collections,
   score, max, status, detail}

Status-Logik:
- PASS: journal alive + face_db >=10 + Qdrant erreichbar
- WARN: journal events_1h=0 ODER Qdrant nicht erreichbar
- FAIL: Modul-Import scheitert ODER face_db=0
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger("memory_auditor")

_STATUS_PATH = "/dev/shm/moloch_status.json"
_QDRANT_URL = "http://localhost:6333/collections"


def _parse_ts(ts: Any) -> float:
    """Robust ISO/Float-Timestamp parsen, return UNIX-time oder 0.0."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            # ISO 8601 mit oder ohne Z
            s = ts.rstrip("Z")
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            try:
                return float(ts)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def collect() -> Dict[str, Any]:
    """Sammelt Memory-Layer-Daten."""
    detail: Dict[str, Any] = {}
    face_db_entries = 0
    journal_events_1h = 0
    qdrant_alive = False
    qdrant_collections: List[str] = []
    longterm_alive = False
    journal_alive = False

    # 1. longterm_memory (L0)
    try:
        from core.longterm_memory import get_memory  # type: ignore
        mem = get_memory()
        longterm_alive = mem is not None
    except Exception as e:
        detail["longterm_import_error"] = str(e)[:120]

    # 2. character_journal (L1)
    try:
        from core.memory.character_journal import get_journal  # type: ignore
        jr = get_journal()
        journal_alive = jr is not None
        try:
            recent = jr.read_recent(50) or []
            now = time.time()
            cnt = 0
            for ev in recent:
                if not isinstance(ev, dict):
                    continue
                ts = _parse_ts(ev.get("timestamp") or ev.get("ts"))
                if ts > 0 and (now - ts) <= 3600:
                    cnt += 1
            journal_events_1h = cnt
            detail["journal_total_recent"] = len(recent)
        except Exception as ee:
            detail["journal_read_error"] = str(ee)[:100]
    except Exception as e:
        detail["journal_import_error"] = str(e)[:120]

    # 3. FaceDB aus moloch_status.json (L2)
    try:
        with open(_STATUS_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        wh = st.get("worker_health") or {}
        if isinstance(wh, dict):
            fw = wh.get("FaceWorker") or {}
            if isinstance(fw, dict):
                face_db_entries = int(fw.get("face_db_entries", 0) or 0)
        # Fallback: face_id-Liste oder einpraegen-Status
        if face_db_entries == 0:
            ep = st.get("einpraegen_progress") or {}
            if isinstance(ep, dict):
                face_db_entries = int(ep.get("enrolled", 0) or 0)
    except Exception as e:
        detail["status_json_error"] = str(e)[:100]

    # 4. Qdrant ping (best-effort, requests timeout=5)
    try:
        import requests  # type: ignore
        r = requests.get(_QDRANT_URL, timeout=5)
        if r.status_code == 200:
            qdrant_alive = True
            try:
                payload = r.json()
                cols = (payload.get("result", {}) or {}).get("collections", []) or []
                qdrant_collections = [
                    c.get("name") for c in cols if isinstance(c, dict) and c.get("name")
                ]
            except Exception as ee:
                detail["qdrant_parse_error"] = str(ee)[:100]
        else:
            detail["qdrant_status"] = r.status_code
    except Exception as e:
        detail["qdrant_error"] = str(e)[:80]

    detail["longterm_alive"] = longterm_alive
    detail["journal_alive"] = journal_alive

    # 5. Status berechnen
    score = 0
    max_score = 4
    if journal_alive:
        score += 1
    if longterm_alive:
        score += 1
    if face_db_entries >= 10:
        score += 1
    if qdrant_alive:
        score += 1

    if not journal_alive or not longterm_alive or face_db_entries == 0:
        status = "FAIL"
    elif journal_events_1h == 0 or not qdrant_alive:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "face_db_entries": face_db_entries,
        "journal_events_1h": journal_events_1h,
        "qdrant_alive": qdrant_alive,
        "qdrant_collections": qdrant_collections,
        "detail": detail,
    }
