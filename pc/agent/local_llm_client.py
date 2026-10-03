"""Lokaler LLM-Client (Ollama /api/chat) — gleiche Signatur wie deepseek_client.

Damit laesst sich der Orchestrator ohne Cloud betreiben. Die Antwort wird ins
OpenAI-Format uebersetzt ({choices:[{message}], usage}), das der Loop erwartet.

Konfig (env):
  MOLOCH_LOCAL_LLM_URL     default http://127.0.0.1:11434
  MOLOCH_LOCAL_MODEL       default dolphin-llama3:8b
  MOLOCH_LOCAL_KEEP_ALIVE  default 30m (Modell bleibt geladen, sonst ~11 s Ladezeit)

Messwerte 2026-10-03 (Ryzen 9 3900X, CPU-only): ~5.7 Tok/s Antwort, ~37 Tok/s
Prompt-Verarbeitung, warmer Praefix 0.3-0.6 s. Deshalb: fester System-Prompt
vorne, kurze Antworten (max_tokens klein halten).

NEVER 5: requests timeout. Kein Cloud-Call, kein Token-Budget-Eintrag.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("local-llm-client")

API_URL = os.environ.get("MOLOCH_LOCAL_LLM_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("MOLOCH_LOCAL_MODEL", "dolphin-llama3:8b")
DEFAULT_KEEP_ALIVE = os.environ.get("MOLOCH_LOCAL_KEEP_ALIVE", "30m")
DEFAULT_MAX_TOKENS = 80
DEFAULT_TEMP = 0.6
DEFAULT_NUM_CTX = 4096
HTTP_TIMEOUT = 120


def complete(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMP,
    timeout: int = HTTP_TIMEOUT,
    response_format: Optional[Any] = None,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    stop: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Sendet messages an Ollama, returnt Response im OpenAI-Format.

    tools: natives Function-Calling — nur Modelle, die es koennen (z.B. mistral).
           Lehnt das Modell Tools ab (HTTP 400), wird ohne Tools wiederholt.
    response_format: "json" oder ein JSON-Schema (Ollama-Parameter `format`).
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "num_ctx": DEFAULT_NUM_CTX,
        },
    }
    if stop:
        payload["options"]["stop"] = stop
    if tools:
        payload["tools"] = tools
    if response_format is not None:
        payload["format"] = response_format

    r = requests.post(f"{API_URL}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 400 and tools and "tools" in r.text.lower():
        logger.warning(f"[local] {model} kann kein natives Tool-Calling — Wiederholung ohne Tools")
        payload.pop("tools", None)
        r = requests.post(f"{API_URL}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
    return _to_openai(r.json())


def _to_openai(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Ollama-Antwort -> OpenAI-Form, die orchestrator.run() erwartet."""
    msg = dict(resp.get("message") or {})
    msg.setdefault("role", "assistant")
    msg.setdefault("content", "")
    # Ollama liefert tool_calls ohne id und mit arguments als dict
    calls = msg.get("tool_calls") or []
    for i, tc in enumerate(calls):
        tc.setdefault("id", f"local-{i}")
        tc.setdefault("type", "function")
    if not calls:
        msg.pop("tool_calls", None)
    prompt = int(resp.get("prompt_eval_count") or 0)
    completion = int(resp.get("eval_count") or 0)
    eval_s = (resp.get("eval_duration") or 0) / 1e9
    return {
        "choices": [{"message": msg}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "model": resp.get("model"),
        "timing": {
            "total_s": (resp.get("total_duration") or 0) / 1e9,
            "load_s": (resp.get("load_duration") or 0) / 1e9,
            "prompt_s": (resp.get("prompt_eval_duration") or 0) / 1e9,
            "eval_s": eval_s,
            "tok_per_s": (completion / eval_s) if eval_s > 0 else 0.0,
        },
    }


def extract_message(response: Dict[str, Any]) -> Dict[str, Any]:
    """Holt assistant-Message aus der Response."""
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("Ollama: keine choices")
    return choices[0].get("message", {})


def extract_usage(response: Dict[str, Any]) -> Dict[str, int]:
    """Token-Usage aus Response. Returns {prompt_tokens, completion_tokens, total_tokens}."""
    return response.get("usage", {})
