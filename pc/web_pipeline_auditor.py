"""Web-Pipeline-Auditor (PC-Side, Welle-19).

End-to-End-Verifikation der Web-Recherche-Pipeline:
  1. Search-Proxy /health erreichbar
  2. Search-Proxy /stats Endpoint vorhanden + zeigt sinnvolle Werte
  3. POST /search mit Test-Query liefert echte URLs (kein Hallu)
  4. Optional: zeigt seconds_since_last_call -> Pi-Routing aktiv?

Hintergrund: 2026-04-30 entdeckt — Audit zeigte Search-Proxy "PASS HTTP 200"
aber Pi-Routing ruft ihn fuer prompt_type=web NICHT an. LLM halluziniert
WGT-Lineup aus Spotify-Stats. Health-Check != Functional-Test.

Periodic-Mode: alle 5 min POST an /mailbox/audit/web_search auf Pi (sobald
Pi-Whitelist erweitert ist). CLI-Mode: --once -> JSON auf stdout.

Atomic-Write nicht noetig (kein lokaler State-File).
NEVER 5: requests.timeout=15. NEVER 8: kein shell=True (kein subprocess).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("web-pipeline-auditor")

SEARCH_PROXY_URL = "http://localhost:11650"
PI_AUDIT_ENDPOINT = "http://192.168.178.30:9100/mailbox/audit/web_search"
TEST_QUERY = "Wave Gotik Treffen Leipzig 2026 Lineup"
TEST_FETCH_URL = "https://www.wave-gotik-treffen.de/bands.php"  # WGT-Bands-Liste
LOOP_INTERVAL_S = 300  # 5 min
HTTP_TIMEOUT_S = 15
FETCH_TIMEOUT_S = 25

# Audit-relevant: ist last_call zu lang her -> Pi-Routing dead (kein Web-Routing)
STALE_THRESHOLD_SEC = 3600  # 1h ohne /search-Call = Routing wahrscheinlich kaputt


def _check_health() -> tuple[bool, str]:
    try:
        r = requests.get(f"{SEARCH_PROXY_URL}/health", timeout=5)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _check_stats() -> tuple[bool, Dict[str, Any]]:
    try:
        r = requests.get(f"{SEARCH_PROXY_URL}/stats", timeout=5)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)[:80]}


def _check_e2e_search() -> tuple[bool, Dict[str, Any]]:
    try:
        r = requests.post(
            f"{SEARCH_PROXY_URL}/search",
            json={"query": TEST_QUERY, "max_results": 3},
            timeout=HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            return False, {"error": f"HTTP {r.status_code}"}
        data = r.json()
        results = data.get("results", [])
        urls = [x.get("url", "") for x in results]
        valid_urls = [u for u in urls if u.startswith("http")]
        ok = len(valid_urls) >= 1
        return ok, {
            "result_count": len(results),
            "valid_url_count": len(valid_urls),
            "sample_url": valid_urls[0] if valid_urls else None,
            "duration_ms": data.get("duration_ms"),
            "cached": data.get("cached"),
        }
    except Exception as e:
        return False, {"error": str(e)[:120]}


def _check_e2e_fetch() -> tuple[bool, Dict[str, Any]]:
    """Welle 20a: testet /fetch Endpoint mit WGT-Bands-Liste."""
    try:
        r = requests.post(
            f"{SEARCH_PROXY_URL}/fetch",
            json={"url": TEST_FETCH_URL, "max_chars": 6000},
            timeout=FETCH_TIMEOUT_S,
        )
        if r.status_code != 200:
            return False, {"error": f"HTTP {r.status_code}"}
        data = r.json()
        text = data.get("text", "")
        chars = data.get("chars", 0)
        # Sanity: WGT-bands-Seite muss "Wave-Gotik" oder typische Band-Namen enthalten
        ok_marker = ("Wave-Gotik" in text) or ("Lacrimosa" in text) or ("Suicide Commando" in text)
        ok = chars > 500 and ok_marker
        return ok, {
            "chars": chars,
            "title": data.get("title", "")[:80],
            "duration_ms": data.get("duration_ms"),
            "cached": data.get("cached"),
            "marker_found": ok_marker,
            "text_snippet": text[:200] if text else None,
        }
    except Exception as e:
        return False, {"error": str(e)[:120]}


def collect() -> Dict[str, Any]:
    """Audit-Sub-Auditor Pattern. Returns {score, max, status, detail}."""
    detail: Dict[str, Any] = {}
    score = 0
    total = 0

    # Layer 1: Health
    total += 1
    health_ok, health_msg = _check_health()
    detail["health"] = health_msg
    if health_ok:
        score += 1

    # Layer 2: Stats-Endpoint vorhanden + Pi-Routing-Activity-Indikator
    total += 1
    stats_ok, stats_data = _check_stats()
    if stats_ok:
        score += 1
        detail["stats"] = {
            "request_count": stats_data.get("request_count", 0),
            "last_query": stats_data.get("last_query"),
            "seconds_since_last_call": stats_data.get("seconds_since_last_call"),
            "uptime_sec": stats_data.get("uptime_sec"),
        }
        # Wichtigster Indikator: wurde Search-Proxy ueberhaupt schon gerufen?
        seconds_since = stats_data.get("seconds_since_last_call")
        if seconds_since is None:
            detail["pi_routing_active"] = False
            detail["pi_routing_note"] = "Search-Proxy nie gerufen seit start"
        elif seconds_since > STALE_THRESHOLD_SEC:
            detail["pi_routing_active"] = False
            detail["pi_routing_note"] = (
                f"Letzter Call vor {seconds_since}s — Pi-Routing inaktiv"
            )
        else:
            detail["pi_routing_active"] = True
    else:
        detail["stats"] = stats_data

    # Layer 3: End-to-End Search
    total += 1
    e2e_ok, e2e_data = _check_e2e_search()
    detail["e2e_search"] = e2e_data
    if e2e_ok:
        score += 1

    # Layer 4: End-to-End Fetch (Welle 20a)
    total += 1
    fetch_ok, fetch_data = _check_e2e_fetch()
    detail["e2e_fetch"] = fetch_data
    if fetch_ok:
        score += 1

    # Status-Mapping
    if total == 0:
        status = "PENDING"
    elif score == total:
        status = "PASS"
    elif score >= total * 0.6:
        status = "WARN"
    else:
        status = "FAIL"

    return {"score": score, "max": total, "status": status, "detail": detail}


def post_to_pi(result: Dict[str, Any]) -> bool:
    """Sendet Audit-Layer an Pi-Audit-Receiver. Whitelist-Erweiterung Pi-Side noetig."""
    try:
        payload = {**result, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        r = requests.post(PI_AUDIT_ENDPOINT, json=payload, timeout=HTTP_TIMEOUT_S)
        if r.status_code == 200:
            return True
        if r.status_code == 400:
            logger.debug("[post] Pi-Whitelist nicht erweitert (web_search), 400 erwartet bis Pi-Patch")
        else:
            logger.warning(f"[post] HTTP {r.status_code}: {r.text[:120]}")
        return False
    except Exception as e:
        logger.warning(f"[post] error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Web-Pipeline Auditor (Welle-19)")
    parser.add_argument("--once", action="store_true",
                        help="ein Audit-Run + JSON auf stdout, dann exit")
    parser.add_argument("--no-post", action="store_true",
                        help="auch im loop nicht zu Pi posten (lokal-only)")
    args = parser.parse_args()

    if args.once:
        result = collect()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] in ("PASS", "WARN") else 1

    logger.info(f"web-pipeline-auditor loop start, intervall={LOOP_INTERVAL_S}s")
    while True:
        try:
            result = collect()
            logger.info(
                f"[tick] status={result['status']} score={result['score']}/{result['max']}"
            )
            if not args.no_post:
                ok = post_to_pi(result)
                logger.info(f"[tick] post={'ok' if ok else 'fail'}")
        except Exception as e:
            logger.warning(f"[tick] error: {e}")
        time.sleep(LOOP_INTERVAL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
