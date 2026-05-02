"""DeepSeek-Client mit function-calling (OpenAI-kompatible API).

Liest API-Key aus ~/moloch/config/api_keys.json oder env DEEPSEEK_API_KEY.
NEVER 5: requests timeout. NEVER 8: kein shell. API-Key NIE loggen.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("deepseek-client")

API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMP = 0.3
HTTP_TIMEOUT = 90


def _load_api_key() -> str:
    """Lese DeepSeek-Key aus api_keys.json oder env. NIE loggen."""
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    keys_paths = [
        Path.home() / "moloch_repo" / "config" / "api_keys.json",
        Path("C:/Users/49179/moloch_repo/config/api_keys.json"),
    ]
    for p in keys_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                key = (
                    data.get("deepseek")
                    or data.get("api_deepseek")
                    or data.get("DEEPSEEK_API_KEY")
                    or ""
                ).strip()
                if key:
                    return key
            except Exception as e:
                logger.warning(f"[key] api_keys.json parse-Fehler: {e}")
    raise RuntimeError(
        "DeepSeek-API-Key nicht gefunden. Setze env DEEPSEEK_API_KEY oder "
        "config/api_keys.json:deepseek"
    )


def complete(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMP,
    timeout: int = HTTP_TIMEOUT,
) -> Dict[str, Any]:
    """Sendet messages an DeepSeek, returnt Response-Dict.

    messages: OpenAI-Format [{role, content}, ...]
    tools: function-calling-Tools im OpenAI-Schema oder None
    Returns: {choices: [{message: {role, content, tool_calls?}}], usage: {...}, ...}
    """
    api_key = _load_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        resp = r.json()
        # Welle 21 Phase 5: Token-Budget-Tracking
        try:
            from pc.agent.token_budget import record_call
            usage = resp.get("usage", {})
            record_call(
                model=model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        except Exception as e:
            logger.warning(f"[budget] record_call-Fehler: {e}")
        return resp
    except requests.HTTPError as e:
        body = ""
        try:
            body = r.text[:300]
        except Exception:
            pass
        raise RuntimeError(f"DeepSeek HTTP {r.status_code}: {body}") from e


def extract_message(response: Dict[str, Any]) -> Dict[str, Any]:
    """Holt assistant-Message aus DeepSeek-Response."""
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("DeepSeek: keine choices")
    return choices[0].get("message", {})


def extract_usage(response: Dict[str, Any]) -> Dict[str, int]:
    """Token-Usage aus Response. Returns {prompt_tokens, completion_tokens, total_tokens}."""
    return response.get("usage", {})
