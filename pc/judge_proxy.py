"""MOLOCH Judge-Proxy (PC-Side, DeepSeek-LLM-as-Judge fuer Performance-Test).

FastAPI on :11651. DeepSeek-Cloud-Call fuer semantische Bewertung von Moloch-Antworten
im 5-Akt-Performance-Test. Heuristik-Validators auf Pi machen den ersten Pass —
Pi ruft judge_proxy nur als Fallback (--judge=cloud Flag im runner.py).

Endpoints:
  POST /judge_act  body={"act_id": str, "moloch_response": str, "expectations": {...}}
                   resp={"verdict": "PASS|FAIL", "score": float, "reason": str,
                         "tokens_used": int, "cached": bool}
  GET  /health     resp={"status": "ok", "service": "moloch-judge-proxy",
                         "calls_today": int, "usd_today": float}

Reboot-persistent via pc/install_judge_proxy_task.bat (separater Schritt).

NEVER 5: requests timeout. NEVER 8: kein shell. API-Key NIE loggen.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pc.agent.deepseek_client import complete as deepseek_complete
from pc.agent.token_budget import record_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("judge-proxy")

HOST = os.environ.get("MOLOCH_JUDGE_HOST", "0.0.0.0")
PORT = int(os.environ.get("MOLOCH_JUDGE_PORT", "11651"))
MODEL = os.environ.get("MOLOCH_JUDGE_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.environ.get("MOLOCH_JUDGE_MAX_TOKENS", "800"))
TEMPERATURE = float(os.environ.get("MOLOCH_JUDGE_TEMP", "0.1"))
CACHE_SIZE = int(os.environ.get("MOLOCH_JUDGE_CACHE_SIZE", "256"))

app = FastAPI(title="MOLOCH Judge-Proxy", version="1.0")

_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_stats: Dict[str, Any] = {"calls_today": 0, "usd_today": 0.0, "started_day": ""}


class Expectations(BaseModel):
    must_avoid: List[str] = Field(default_factory=list)
    must_have: List[str] = Field(default_factory=list)
    tone_target: str = ""


class JudgeRequest(BaseModel):
    act_id: str
    moloch_response: str
    expectations: Expectations


class JudgeResponse(BaseModel):
    verdict: str
    score: float
    reason: str
    tokens_used: int
    cached: bool


def _cache_key(req: JudgeRequest) -> str:
    h = hashlib.sha256(req.moloch_response.encode("utf-8")).hexdigest()[:16]
    return f"{req.act_id}:{h}"


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_stats_if_new_day() -> None:
    today = _today_key()
    if _stats["started_day"] != today:
        _stats["started_day"] = today
        _stats["calls_today"] = 0
        _stats["usd_today"] = 0.0


def _build_judge_prompt(req: JudgeRequest) -> List[Dict[str, str]]:
    e = req.expectations
    must_avoid = "\n".join(f"  - {x}" for x in e.must_avoid) or "  (keine)"
    must_have = "\n".join(f"  - {x}" for x in e.must_have) or "  (keine)"
    system = (
        "Du bist Performance-Test-Judge fuer den KI-Charakter MOLOCH. "
        "Du bewertest, ob eine Moloch-Antwort die erwarteten Charakter-Eigenschaften erfuellt. "
        "Antworte AUSSCHLIESSLICH mit gueltigem JSON in dieser Struktur:\n"
        '{"verdict": "PASS" oder "FAIL", "score": 0.0-1.0, "reason": "kurze deutsche Begruendung max 2 Saetze"}'
    )
    user = (
        f"Akt-ID: {req.act_id}\n\n"
        f"Moloch-Antwort:\n\"{req.moloch_response}\"\n\n"
        f"Erwartungen:\n"
        f"Ton-Ziel: {e.tone_target or '(nicht spezifiziert)'}\n"
        f"Antwort MUSS enthalten / aufweisen:\n{must_have}\n"
        f"Antwort darf NICHT enthalten:\n{must_avoid}\n\n"
        "Bewerte: erfuellt die Antwort die Erwartungen? Gib NUR das JSON zurueck."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_judge_output(content: str) -> Dict[str, Any]:
    """Extrahiert JSON aus DeepSeek-Output. Toleriert Code-Blocks."""
    txt = content.strip()
    if txt.startswith("```"):
        # ```json ... ``` strippen
        lines = txt.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                inner.append(line)
        txt = "\n".join(inner).strip() or txt
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # Suche erstes {...}-Substring
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(txt[start : end + 1])
        else:
            raise
    verdict = str(data.get("verdict", "FAIL")).upper()
    if verdict not in ("PASS", "FAIL"):
        verdict = "FAIL"
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", "kein Grund angegeben"))[:500]
    return {"verdict": verdict, "score": score, "reason": reason}


@app.get("/health")
def health() -> Dict[str, Any]:
    _reset_stats_if_new_day()
    return {
        "status": "ok",
        "service": "moloch-judge-proxy",
        "version": "1.0",
        "model": MODEL,
        "calls_today": _stats["calls_today"],
        "usd_today": round(_stats["usd_today"], 5),
        "cache_entries": len(_cache),
    }


@app.post("/judge_act", response_model=JudgeResponse)
def judge_act(req: JudgeRequest) -> JudgeResponse:
    _reset_stats_if_new_day()
    key = _cache_key(req)
    if key in _cache:
        cached = _cache[key]
        return JudgeResponse(
            verdict=cached["verdict"],
            score=cached["score"],
            reason=cached["reason"],
            tokens_used=0,
            cached=True,
        )

    messages = _build_judge_prompt(req)
    try:
        resp = deepseek_complete(
            messages=messages,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=60,
        )
    except RuntimeError as e:
        # API-Key-Fehler -> 503, Pi faellt auf Heuristik zurueck
        raise HTTPException(status_code=503, detail=f"deepseek-key: {e}")
    except Exception as e:
        logger.exception("judge_act deepseek call failed")
        raise HTTPException(status_code=503, detail=f"deepseek-error: {type(e).__name__}")

    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.exception("judge_act response parse failed")
        raise HTTPException(status_code=502, detail=f"deepseek-malformed: {e}")

    usage = resp.get("usage", {}) or {}
    in_tok = int(usage.get("prompt_tokens", 0))
    out_tok = int(usage.get("completion_tokens", 0))
    total_tok = in_tok + out_tok
    try:
        record_call(MODEL, in_tok, out_tok)
    except Exception:
        logger.warning("token_budget.record_call failed (non-fatal)")

    try:
        parsed = _parse_judge_output(content)
    except Exception as e:
        logger.warning(f"judge_act parse failed, content={content[:200]!r}")
        parsed = {
            "verdict": "FAIL",
            "score": 0.0,
            "reason": f"judge-output-unparsbar: {type(e).__name__}",
        }

    # Cache
    _cache[key] = parsed
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)

    # Stats
    _stats["calls_today"] += 1
    from pc.agent.token_budget import estimate_usd
    _stats["usd_today"] += estimate_usd(MODEL, in_tok, out_tok)

    return JudgeResponse(
        verdict=parsed["verdict"],
        score=parsed["score"],
        reason=parsed["reason"],
        tokens_used=total_tok,
        cached=False,
    )


def main() -> None:
    logger.info(f"MOLOCH Judge-Proxy startet auf {HOST}:{PORT} (model={MODEL})")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
