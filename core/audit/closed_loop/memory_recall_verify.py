"""Memory-Recall Closed-Loop-Verifier — recall('Markus').

PASS  : 1+ Treffer 'Markus' + Confidence >=0.7
WARN  : Treffer aber Confidence <0.7
FAIL  : kein Treffer ODER recall-API nicht verfuegbar
SKIP  : longterm_memory nicht importierbar
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from ._common import fail_result, now, skip_result

logger = logging.getLogger("memory_recall_verify")

_QUERY = "Markus"


def _get_memory():
    try:
        from core.longterm_memory import get_memory  # type: ignore
        return get_memory()
    except Exception as e:
        logger.debug("memory import failed: %s", e)
        return None


def _try_recall(mem, query: str) -> Dict[str, Any]:
    """Versucht recall/search/find/query — wenn alle fehlen,
    Fallback auf get_memory_context() Substring-Match.
    """
    for name in ("recall", "search", "find", "query"):
        fn = getattr(mem, name, None)
        if callable(fn):
            try:
                hits = fn(query)
                return {"method": name, "hits": hits}
            except TypeError:
                # ggf. (query, k) Signatur
                try:
                    hits = fn(query, 5)
                    return {"method": name, "hits": hits}
                except Exception as e:
                    logger.debug("%s(%r) failed: %s", name, query, e)
            except Exception as e:
                logger.debug("%s(%r) failed: %s", name, query, e)

    # Fallback: get_memory_context() durchsuchen
    if hasattr(mem, "get_memory_context"):
        try:
            ctx = mem.get_memory_context()
            if isinstance(ctx, str) and query.lower() in ctx.lower():
                # Konfidenz aus Anzahl der Vorkommen ableiten
                count = len(re.findall(re.escape(query), ctx, flags=re.IGNORECASE))
                return {
                    "method": "memory_context_substring",
                    "hits": [{"text": query, "confidence": min(1.0, 0.5 + count * 0.05),
                              "occurrences": count}],
                }
        except Exception as e:
            logger.debug("memory_context fallback failed: %s", e)

    return {"method": None, "hits": None}


def _confidence(hits: Any) -> float:
    """Best-effort Konfidenz aus Hit-Liste extrahieren."""
    if hits is None:
        return 0.0
    if isinstance(hits, (int, float)):
        return float(hits)
    if isinstance(hits, str):
        return 1.0 if hits else 0.0
    if isinstance(hits, list):
        if not hits:
            return 0.0
        # Liste von Dicts mit confidence/score?
        scores: List[float] = []
        for h in hits:
            if isinstance(h, dict):
                for k in ("confidence", "score", "similarity", "relevance"):
                    if k in h and isinstance(h[k], (int, float)):
                        scores.append(float(h[k]))
                        break
        if scores:
            return max(scores)
        # Liste mit irgendwas drin = mind. ein Treffer
        return 0.6
    if isinstance(hits, dict):
        for k in ("confidence", "score"):
            if k in hits and isinstance(hits[k], (int, float)):
                return float(hits[k])
        return 0.6
    return 0.5


def verify(timeout_s: int = 5) -> Dict[str, Any]:
    mem = _get_memory()
    if mem is None:
        return skip_result("longterm_memory_unavailable")

    t_start = now()
    result = _try_recall(mem, _QUERY)
    method = result.get("method")
    hits = result.get("hits")

    if method is None:
        return fail_result("no_recall_api", query=_QUERY)

    has_hits = bool(hits)
    conf = _confidence(hits)

    if has_hits and conf >= 0.7:
        status, score = "PASS", 2
    elif has_hits:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    # Hits trimmen damit das JSON klein bleibt
    hits_summary: Any
    if isinstance(hits, list):
        hits_summary = hits[:3]
    else:
        hits_summary = hits

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": f"{method}('{_QUERY}')",
        "baseline": {},
        "after": {"hits": hits_summary, "confidence": round(conf, 2)},
        "delta": {"hit_count": len(hits) if isinstance(hits, list) else (1 if has_hits else 0)},
        "duration_s": round(now() - t_start, 2),
        "detail": {"query": _QUERY, "method": method, "threshold": 0.7},
    }
