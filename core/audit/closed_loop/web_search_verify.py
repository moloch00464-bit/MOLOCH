"""W19 Closed-Loop Web-Search-Verifier — End-to-End-Test der Web-Pipeline.

Triggert eine Test-Frage am chat_server-API und verifiziert dass:
- Search-Proxy /stats zeigt seconds_since_last_call < 30 (Pipeline aktiv)
- Antwort enthaelt URL ODER festival/Zahl (echte WGT-Daten)
- Antwort enthaelt NICHT klassische Spotify-Stats-Bands (Halluzination)

SKIP wenn Search-Proxy unerreichbar (PC-Cowork down).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

from ._common import fail_result, skip_result

logger = logging.getLogger("closed_loop.web_search")

CHAT_URL = "http://localhost:9100/chat"
SEARCH_PROXY_STATS_URL = "http://192.168.178.20:11650/stats"
TEST_QUERY = "Wieviel Bands spielen aufm WGT 2026?"
SPOTIFY_HALLUCINATION_BANDS = {
    "suicide commando", "vomito negro", "chainreactor", "esa", "geistform",
}


def verify(timeout_s: int = 30) -> Dict[str, Any]:
    started = time.time()

    # 0. Search-Proxy erreichbar?
    try:
        r0 = requests.get(SEARCH_PROXY_STATS_URL, timeout=5)
        if not r0.ok:
            return skip_result("search_proxy_unreachable",
                               duration_s=time.time() - started)
        baseline_stats = r0.json() if r0.content else {}
    except Exception as e:
        return skip_result(f"search_proxy_unreachable: {e}",
                           duration_s=time.time() - started)

    # 1. chat-Trigger
    try:
        r = requests.post(CHAT_URL, json={"text": TEST_QUERY}, timeout=timeout_s)
        if not r.ok:
            return fail_result("chat_endpoint_error",
                               detail={"status": r.status_code},
                               duration_s=time.time() - started)
        chat_response = r.json()
        answer = (chat_response.get("response")
                  or chat_response.get("text") or "").lower()
    except Exception as e:
        return fail_result(f"chat_endpoint_timeout: {e}",
                           duration_s=time.time() - started)

    # 2. Search-Proxy /stats nochmal
    try:
        r2 = requests.get(SEARCH_PROXY_STATS_URL, timeout=5)
        after_stats = r2.json() if r2.ok and r2.content else {}
    except Exception:
        after_stats = {}

    secs_since = after_stats.get("seconds_since_last_call", 999)

    # 3. Bewerten
    has_url = "http" in answer
    has_festival = "festival" in answer or "wgt" in answer
    has_number = any(str(n) in answer for n in range(100, 200))
    has_hallucination = any(b in answer for b in SPOTIFY_HALLUCINATION_BANDS)

    duration = time.time() - started

    if has_hallucination:
        return fail_result(
            "spotify_hallucination_detected",
            detail={"answer_excerpt": answer[:300]},
            duration_s=duration,
        )
    if secs_since > 30:
        return fail_result(
            "search_proxy_not_called",
            detail={"seconds_since_last_call": secs_since},
            duration_s=duration,
        )

    # PASS-Logik
    score = 0
    if has_url:
        score += 1
    if has_festival:
        score += 1
    if has_number:
        score += 1
    if secs_since < 30:
        score += 1
    max_s = 4
    if score >= 3:
        status = "PASS"
    elif score >= 2:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "status": status,
        "score": score,
        "max": max_s,
        "duration_s": duration,
        "command_sent": TEST_QUERY,
        "baseline": baseline_stats,
        "after": after_stats,
        "delta": {
            "has_url": has_url,
            "has_festival": has_festival,
            "has_number": has_number,
            "secs_since_last_call": secs_since,
        },
        "detail": {
            "answer_excerpt": answer[:200],
            "search_proxy_stats_after": after_stats,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
