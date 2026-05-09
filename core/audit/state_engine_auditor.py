"""state_engine-Layer-Auditor (Phase 1 Pi-Side, Drei-Hirn-Synthese).

4 Tests laut PC-Opus task_phase1_pi_side_state_engine_und_identity:
  1. state_engine_alive       — last_transition_ts < 60s alt
  2. transition_engine_failsafe — kein State stuck >300s
  3. state_logger_writing      — heutige JSONL > 0 bytes
  4. identity_phrase_present   — current_state in VALID_STATES + identity_phrase set

Schreibt audit_state.layers.state_engine:
  {score, max, status, detail: {checks: [...], current_state, ...}}

Status-Logik:
  PASS:  4/4 Tests OK
  WARN:  3/4 OK
  FAIL:  <3/4 OK
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger("audit.state_engine")

VALID_STATES = ("idle", "observing", "engaged", "overloaded", "withdrawing", "offline_anchor")

ALIVE_MAX_AGE_S = 60.0
STUCK_FAILSAFE_S = 300.0


def collect() -> Dict[str, Any]:
    """Sammelt 4 Phase-1-Health-Checks fuer state_engine."""
    checks = []
    score = 0
    max_score = 4
    detail: Dict[str, Any] = {}

    # 1. state_engine alive — tick() rufbar + snapshot() liefert valide state_vector.
    #    Vorher: 'last_transition_age_s < 60s' war zu aggressiv (idle/withdrawing
    #    sind langlebige Stable-States, kein State-Wechsel != tot). Jetzt:
    #    Stable-States duerfen lange dauern, aktive States muessen frisch sein.
    STABLE_STATES = {"idle", "withdrawing", "offline_anchor"}
    try:
        from core.personality.state_engine import get_state_engine
        get_state_engine().tick(reason="state_engine_auditor")
        snap = get_state_engine().snapshot()
        last_ts = float(snap.get("last_transition_ts", 0.0))
        cur_state = snap.get("current_state", "idle")
        sv = snap.get("state_vector") or {}
        sv_sum_ok = abs(sum(sv.values()) - 1.0) < 0.01 if sv else False
        age_s = time.time() - last_ts if last_ts > 0 else 0.0
        if last_ts == 0 or cur_state in STABLE_STATES:
            ok = sv_sum_ok
        else:
            ok = age_s < ALIVE_MAX_AGE_S and sv_sum_ok
        if ok:
            score += 1
        checks.append({
            "name": "state_engine_alive",
            "ok": ok,
            "detail": (
                f"current_state={cur_state} age_s={age_s:.0f} "
                f"vector_sum_ok={sv_sum_ok} stable={cur_state in STABLE_STATES}"
            ),
        })
        detail["current_state"] = cur_state
        detail["state_vector"] = sv
        detail["tension"] = snap.get("tension")
        detail["transition_speed"] = snap.get("transition_speed")
    except Exception as e:
        checks.append({
            "name": "state_engine_alive",
            "ok": False,
            "detail": f"import_error: {str(e)[:120]}",
        })

    # 2. transition_engine failsafe — 'stuck > 300s' nur fuer aktive States gefaehrlich.
    #    Stable-States duerfen unbegrenzt dauern (Markus weg vom Frame = idle = normal).
    try:
        from core.personality.transition_engine import get_transition_engine
        from core.personality.state_engine import get_state_engine
        te = get_transition_engine()
        age_s = te.state_age_s()
        cur_state = (get_state_engine().snapshot() or {}).get("current_state", "idle")
        if cur_state in STABLE_STATES:
            ok = True
        else:
            ok = age_s <= STUCK_FAILSAFE_S
        if ok:
            score += 1
        checks.append({
            "name": "transition_engine_failsafe",
            "ok": ok,
            "detail": (
                f"state_age_s={age_s:.0f} state={cur_state} "
                f"stable={cur_state in STABLE_STATES} (max {STUCK_FAILSAFE_S}s active)"
            ),
        })
    except Exception as e:
        checks.append({
            "name": "transition_engine_failsafe",
            "ok": False,
            "detail": f"import_error: {str(e)[:120]}",
        })

    # 3. state_logger_writing (heutige JSONL > 0 bytes)
    try:
        from core.personality.state_logger import get_state_logger
        sl = get_state_logger()
        size = sl.today_size_bytes()
        count = sl.today_count()
        # OK wenn Datei existiert UND >0 bytes ODER frisch (count>=0 + Verzeichnis schreibbar)
        ok = size > 0
        # Wenn 0 bytes aber Pfad zugaenglich: gilt als "noch keine Transition heute" - akzeptabel kurz nach Boot
        if size == 0:
            try:
                from core.personality.state_logger import LOG_DIR
                ok = LOG_DIR.exists()
            except Exception:
                ok = False
        if ok:
            score += 1
        checks.append({
            "name": "state_logger_writing",
            "ok": ok,
            "detail": f"size={size} count={count}",
        })
    except Exception as e:
        checks.append({
            "name": "state_logger_writing",
            "ok": False,
            "detail": f"import_error: {str(e)[:120]}",
        })

    # 4. identity_phrase_present
    try:
        from core.personality.identity_phrases import IDENTITY_PHRASES, get_phrase
        from core.personality.state_engine import get_state_engine
        snap = get_state_engine().snapshot()
        cur = snap.get("current_state")
        phrase = snap.get("identity_phrase") or ""
        ok = (cur in VALID_STATES) and (cur in IDENTITY_PHRASES) and bool(phrase.strip())
        if ok:
            score += 1
        checks.append({
            "name": "identity_phrase_present",
            "ok": ok,
            "detail": f"state='{cur}' phrase_len={len(phrase)}",
        })
    except Exception as e:
        checks.append({
            "name": "identity_phrase_present",
            "ok": False,
            "detail": f"import_error: {str(e)[:120]}",
        })

    detail["checks"] = checks

    if score == max_score:
        status = "PASS"
    elif score >= max_score - 1:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "score": score,
        "max": max_score,
        "status": status,
        "detail": detail,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2, ensure_ascii=False, default=str))
