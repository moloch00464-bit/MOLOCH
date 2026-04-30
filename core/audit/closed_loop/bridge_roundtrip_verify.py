"""Bridge-Roundtrip Closed-Loop-Verifier — Tentakel-LLM Ping.

Sendet 'Sag eins' an Markus-PC Ollama. Misst RTT.
PASS  : Response <5 s + non-empty content
WARN  : 5-15 s
FAIL  : Timeout ODER kein content
SKIP  : Tentakel-Endpoint nicht erreichbar (DNS/Port)
"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

from ._common import fail_result, now, skip_result

logger = logging.getLogger("bridge_roundtrip_verify")

_DEFAULT_HOST = "markus-pc.local"
_DEFAULT_PORT = 11434
_TIMEOUT_S = 15
_PROMPT = "Sag eins"


def _load_tentakel_target() -> Tuple[str, int, str]:
    """Liest tentacle_llm aus settings.json. Returns (host, port, model)."""
    host, port, model = _DEFAULT_HOST, _DEFAULT_PORT, ""
    try:
        with open("/home/molochzuhause/moloch/config/settings.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        t = cfg.get("tentacle_llm") or {}
        if isinstance(t, dict):
            h = t.get("host", "")
            if isinstance(h, str) and ":" in h:
                host, p = h.rsplit(":", 1)
                try:
                    port = int(p)
                except Exception:
                    pass
            elif isinstance(h, str) and h:
                host = h
            elif isinstance(t.get("hostname"), str):
                host = t["hostname"]
            if isinstance(t.get("port"), int):
                port = t["port"]
            if isinstance(t.get("model"), str):
                model = t["model"]
    except Exception as e:
        logger.debug("settings load failed: %s", e)
    return host, port, model


def _resolve(host: str) -> bool:
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False


def _send_prompt(host: str, port: int, model: str, prompt: str, timeout: int) -> Dict[str, Any]:
    """Ollama /api/generate Aufruf. Returns dict mit rtt/content/error."""
    url = f"http://{host}:{port}/api/generate"
    body = {
        "model": model or "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            rtt = time.time() - t0
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            content = ""
            if isinstance(payload, dict):
                content = str(payload.get("response") or payload.get("message") or "")
            return {"rtt": rtt, "content": content[:500], "ok": True}
    except urllib.error.URLError as e:
        return {"rtt": time.time() - t0, "content": "", "ok": False, "error": f"URLError:{e.reason}"}
    except socket.timeout:
        return {"rtt": time.time() - t0, "content": "", "ok": False, "error": "socket_timeout"}
    except Exception as e:
        return {"rtt": time.time() - t0, "content": "", "ok": False, "error": str(e)[:120]}


def verify(timeout_s: int = 15) -> Dict[str, Any]:
    host, port, model = _load_tentakel_target()

    if not _resolve(host):
        return skip_result("dns_unresolvable", host=host)

    t_start = now()
    cmd = f"POST http://{host}:{port}/api/generate model={model or 'auto'}"
    res = _send_prompt(host, port, model, _PROMPT, _TIMEOUT_S)

    rtt = float(res.get("rtt", 0.0))
    content = res.get("content", "")
    ok = bool(res.get("ok"))

    if not ok:
        return {
            "score": 0,
            "max": 2,
            "status": "FAIL",
            "command_sent": cmd,
            "baseline": {},
            "after": {"rtt_s": round(rtt, 2), "error": res.get("error", "")},
            "delta": {},
            "duration_s": round(now() - t_start, 2),
            "detail": {"prompt": _PROMPT, "host": host, "port": port},
        }

    if content and rtt < 5.0:
        status, score = "PASS", 2
    elif content and rtt < 15.0:
        status, score = "WARN", 1
    else:
        status, score = "FAIL", 0

    return {
        "score": score,
        "max": 2,
        "status": status,
        "command_sent": cmd,
        "baseline": {},
        "after": {
            "rtt_s": round(rtt, 2),
            "content_len": len(content),
            "content_preview": content[:80],
        },
        "delta": {"rtt_class": "<5s" if rtt < 5 else "5-15s" if rtt < 15 else ">=15s"},
        "duration_s": round(now() - t_start, 2),
        "detail": {"prompt": _PROMPT, "host": host, "port": port, "model": model},
    }
