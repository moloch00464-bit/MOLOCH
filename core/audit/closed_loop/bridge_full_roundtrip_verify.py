"""W15.X Bridge Full-Roundtrip Closed-Loop-Verifier — End-to-End Multi-Hop.

Trifft den kompletten Pfad:
  POST /chat -> Klassifikator -> Specialist-Router -> Tentakel/DeepSeek
  -> response -> Pi-Memory-Save (last_turn.json)

PASS  : provider + prompt_type + response + memory_saved
WARN  : 2-3 von 4 Indikatoren erfuellt
FAIL  : <2 Indikatoren ODER /chat-Endpoint nicht erreichbar
SKIP  : nie (Verifier ist Best-Effort)

Lesen:
- chat_response (von /chat)
- /dev/shm/last_turn.json (Memory-Save check)
- /dev/shm/audit_state.json layers.transition.alive_count

Best-effort: Verifier crasht nie. Bei Exception -> FAIL mit Reason.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from ._common import fail_result, now

logger = logging.getLogger("bridge_full_roundtrip_verify")

_CHAT_URL = "http://localhost:9100/chat"
_LAST_TURN_PATH = "/dev/shm/last_turn.json"
_AUDIT_STATE_PATH = "/dev/shm/audit_state.json"
_TEST_QUERY = "Sag eins"


def verify(timeout_s: int = 45) -> Dict[str, Any]:
    """W15.X End-to-End Roundtrip — Markus-Frage durch komplette Pipeline."""
    started = time.time()
    cmd = f"POST {_CHAT_URL} text='{_TEST_QUERY}'"

    # 1. Chat-Trigger (kompletter Pfad)
    try:
        import requests  # type: ignore
    except Exception as e:
        return fail_result(
            f"requests_unavailable:{str(e)[:120]}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    try:
        r = requests.post(
            _CHAT_URL,
            json={"text": _TEST_QUERY},
            timeout=timeout_s,
        )
    except Exception as e:
        return fail_result(
            f"chat_endpoint_error:{str(e)[:200]}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    if not r.ok:
        return fail_result(
            f"chat_endpoint_status_{r.status_code}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    try:
        chat_resp = r.json()
    except Exception as e:
        return fail_result(
            f"chat_response_not_json:{str(e)[:120]}",
            duration_s=round(time.time() - started, 2),
            command_sent=cmd,
        )

    # 2. Validate response shape
    response_text = ""
    if isinstance(chat_resp, dict):
        response_text = (
            chat_resp.get("response")
            or chat_resp.get("text")
            or ""
        )
    response_text = str(response_text)
    provider = ""
    prompt_type = ""
    if isinstance(chat_resp, dict):
        provider = str(chat_resp.get("provider", ""))
        prompt_type = str(chat_resp.get("prompt_type", ""))

    # 3. Memory-Save check (last_turn.json) — async write, kurz warten
    memory_saved = False
    try:
        time.sleep(0.5)
        with open(_LAST_TURN_PATH, "r", encoding="utf-8") as f:
            lt = json.load(f)
        if isinstance(lt, dict):
            user_text = str(lt.get("user_text", "")).lower()
            prompt = str(lt.get("prompt", "")).lower()
            needle = _TEST_QUERY.lower()
            memory_saved = needle in user_text or needle in prompt
    except Exception as e:
        logger.debug("last_turn read failed: %s", e)

    # 4. transition-Layer alive-check (Quervalidierung)
    transition_alive = 0
    try:
        with open(_AUDIT_STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        if isinstance(st, dict):
            layers = st.get("layers") or {}
            if isinstance(layers, dict):
                tr = layers.get("transition") or {}
                if isinstance(tr, dict):
                    transition_alive = int(tr.get("alive_count", 0) or 0)
    except Exception as e:
        logger.debug("audit_state read failed: %s", e)

    duration = time.time() - started

    # Score-Berechnung
    score = 0
    if response_text:
        score += 1
    if provider:
        score += 1
    if prompt_type:
        score += 1
    if memory_saved:
        score += 1
    max_s = 4

    if score == max_s:
        status = "PASS"
    elif score >= 2:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "score": score,
        "max": max_s,
        "status": status,
        "command_sent": cmd,
        "baseline": {},
        "after": {
            "provider": provider,
            "prompt_type": prompt_type,
            "response_chars": len(response_text),
            "memory_saved": memory_saved,
            "transition_alive": transition_alive,
        },
        "delta": {
            "provider": provider,
            "prompt_type": prompt_type,
            "response_chars": len(response_text),
            "memory_saved": memory_saved,
            "transition_alive": transition_alive,
        },
        "duration_s": round(duration, 2),
        "detail": {
            "prompt": _TEST_QUERY,
            "response_excerpt": response_text[:200],
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
