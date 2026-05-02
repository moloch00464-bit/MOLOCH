"""Transition-Auditor — 7-Kanaele-Health-Layer fuer Pi<->PC-Uebergang.

Aggregiert alle Health-Kanaele die fuer den Pi<->PC-Handoff relevant sind
in einen einzigen Layer. Ersetzt zerstreute Sicht ueber bridge/pc/pc_hardware/
web_ui/web_search/mailbox-Layer durch konsolidierte 1-Glance-Sicht.

Soll-Spezifikation: pc-pi-handoff/SKILL.md (Endpoints + Latency-Budget).

7 Kanaele:
  1. chat_server (Pi-Side localhost:9100/health)
  2. search_proxy (PC 192.168.178.20:11650/stats)
  3. ollama_tentakel (PC 192.168.178.20:11434/api/tags)
  4. adapter_inference (PC settings.adapter_inference.host:port/health)
  5. mailbox_freshness (docs/PC_TO_PI.md mtime + pc.last_seen_age_s)
  6. federation_heartbeat (~/moloch_logs/cross_session.jsonl mtime)
  7. tool_api (Pi-Side localhost:9100/api/agent/tools — W21)

Optional E2E-Roundtrip wenn chat_server + tool_api alive sind:
  POST /api/agent/dispatch {tool_name: "get_mood", params: {}}

Schema:
{
  "score": int (alive_count),
  "max": int (total channels),
  "status": "PASS|WARN|FAIL",
  "channels": {name: {alive, latency_ms, detail}, ...},
  "alive_count": int,
  "detail": {"e2e_roundtrip": {...}}
}

Status-Logik:
- PASS: alle 7 Kanaele alive
- WARN: 1 Kanal fail (z.B. cross_session.jsonl fehlt)
- WARN: alive_count >= total // 2
- FAIL: alive_count < total // 2

NEVER 5: requests-Calls timeout<=5s, gesamt-Auditor <=30s.
Best-effort: Auditor crasht NIE — fehlende Datei/HTTP-Fehler -> alive=False.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("audit.transition")

MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
SETTINGS_PATH = MOLOCH_DIR / "config" / "settings.json"
PC_TO_PI_PATH = MOLOCH_DIR / "docs" / "PC_TO_PI.md"
CROSS_SESSION_LOG = Path(os.path.expanduser("~/moloch_logs/cross_session.jsonl"))
AUDIT_STATE_PATH = Path("/dev/shm/audit_state.json")

# Defaults — werden via settings.json ueberschrieben falls vorhanden
DEFAULT_PC_HOST = "192.168.178.20"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_SEARCH_PROXY_PORT = 11650
DEFAULT_ADAPTER_PORT = 11600
DEFAULT_CHAT_SERVER_PORT = 9100

HTTP_TIMEOUT_S = 5
E2E_TIMEOUT_S = 10
MAILBOX_STALE_THRESHOLD_S = 1800  # 30 min
FEDERATION_STALE_THRESHOLD_S = 7200  # 2h


def _load_settings() -> Dict[str, Any]:
    """Best-effort settings.json lesen."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _http_get(url: str, timeout: float = HTTP_TIMEOUT_S) -> Dict[str, Any]:
    """Best-effort HTTP-GET. Returns {alive, latency_ms, detail, status_code}."""
    t0 = time.time()
    try:
        import requests  # lazy import
        r = requests.get(url, timeout=timeout)
        latency = round((time.time() - t0) * 1000, 1)
        if r.status_code == 200:
            return {"alive": True, "latency_ms": latency, "detail": "200 OK",
                    "status_code": 200}
        return {"alive": False, "latency_ms": latency,
                "detail": f"HTTP {r.status_code}", "status_code": r.status_code}
    except Exception as e:
        latency = round((time.time() - t0) * 1000, 1)
        return {"alive": False, "latency_ms": latency,
                "detail": f"err:{type(e).__name__}:{str(e)[:100]}",
                "status_code": None}


def _check_chat_server() -> Dict[str, Any]:
    """Pi-Side chat_server localhost:9100/health."""
    return _http_get(f"http://localhost:{DEFAULT_CHAT_SERVER_PORT}/health")


def _check_search_proxy(pc_host: str) -> Dict[str, Any]:
    """PC search_proxy /stats."""
    return _http_get(f"http://{pc_host}:{DEFAULT_SEARCH_PROXY_PORT}/stats")


def _check_ollama_tentakel(host: str, port: int) -> Dict[str, Any]:
    """PC Ollama-Tentakel /api/tags."""
    return _http_get(f"http://{host}:{port}/api/tags")


def _check_adapter_inference(host: str, port: int) -> Dict[str, Any]:
    """PC adapter_inference /health (best-effort)."""
    return _http_get(f"http://{host}:{port}/health")


def _check_mailbox_freshness() -> Dict[str, Any]:
    """docs/PC_TO_PI.md mtime + audit_state.pc.last_seen_age_s."""
    detail_parts = []
    alive = True
    pc_to_pi_age = None
    try:
        if PC_TO_PI_PATH.exists():
            pc_to_pi_age = round(time.time() - PC_TO_PI_PATH.stat().st_mtime, 1)
            detail_parts.append(f"PC_TO_PI age={pc_to_pi_age:.0f}s")
            if pc_to_pi_age > MAILBOX_STALE_THRESHOLD_S:
                alive = False
                detail_parts.append("STALE>30m")
        else:
            alive = False
            detail_parts.append("PC_TO_PI.md missing")
    except Exception as e:
        alive = False
        detail_parts.append(f"err:{type(e).__name__}")

    pc_last_seen_age = None
    try:
        if AUDIT_STATE_PATH.exists():
            with open(AUDIT_STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            pc_layer = (state.get("layers") or {}).get("pc") or {}
            pc_detail = pc_layer.get("detail") or {}
            pc_last_seen_age = pc_detail.get("last_seen_age_s")
            if pc_last_seen_age is not None:
                detail_parts.append(f"pc_heartbeat_age={pc_last_seen_age}s")
    except Exception:
        pass

    return {
        "alive": alive,
        "latency_ms": 0.0,
        "detail": " | ".join(detail_parts) if detail_parts else "no data",
        "pc_to_pi_age_s": pc_to_pi_age,
        "pc_last_seen_age_s": pc_last_seen_age,
    }


def _check_federation_heartbeat() -> Dict[str, Any]:
    """~/moloch_logs/cross_session.jsonl mtime (best-effort, Datei darf fehlen)."""
    try:
        if not CROSS_SESSION_LOG.exists():
            return {"alive": False, "latency_ms": 0.0,
                    "detail": "cross_session.jsonl missing",
                    "age_s": None}
        age = round(time.time() - CROSS_SESSION_LOG.stat().st_mtime, 1)
        alive = age <= FEDERATION_STALE_THRESHOLD_S
        return {"alive": alive, "latency_ms": 0.0,
                "detail": f"age={age:.0f}s" + (
                    "" if alive else f" STALE>{FEDERATION_STALE_THRESHOLD_S}s"),
                "age_s": age}
    except Exception as e:
        return {"alive": False, "latency_ms": 0.0,
                "detail": f"err:{type(e).__name__}:{str(e)[:100]}",
                "age_s": None}


def _check_tool_api() -> Dict[str, Any]:
    """W21 Pi-Side /api/agent/tools — 200 + count > 0."""
    t0 = time.time()
    try:
        import requests
        r = requests.get(
            f"http://localhost:{DEFAULT_CHAT_SERVER_PORT}/api/agent/tools",
            timeout=HTTP_TIMEOUT_S,
        )
        latency = round((time.time() - t0) * 1000, 1)
        if r.status_code != 200:
            return {"alive": False, "latency_ms": latency,
                    "detail": f"HTTP {r.status_code}",
                    "tool_count": 0}
        try:
            data = r.json()
        except Exception:
            return {"alive": False, "latency_ms": latency,
                    "detail": "non-json response", "tool_count": 0}
        tools = data.get("tools") if isinstance(data, dict) else None
        count = len(tools) if isinstance(tools, list) else 0
        if count <= 0:
            return {"alive": False, "latency_ms": latency,
                    "detail": "tool_count=0", "tool_count": 0}
        return {"alive": True, "latency_ms": latency,
                "detail": f"tools={count}", "tool_count": count}
    except Exception as e:
        latency = round((time.time() - t0) * 1000, 1)
        return {"alive": False, "latency_ms": latency,
                "detail": f"err:{type(e).__name__}:{str(e)[:100]}",
                "tool_count": 0}


def _e2e_roundtrip() -> Dict[str, Any]:
    """E2E-Mini: POST /api/agent/dispatch tool=get_mood. Erwartet result.zone."""
    t0 = time.time()
    try:
        import requests
        r = requests.post(
            f"http://localhost:{DEFAULT_CHAT_SERVER_PORT}/api/agent/dispatch",
            json={"tool_name": "get_mood", "params": {}},
            timeout=E2E_TIMEOUT_S,
        )
        latency = round((time.time() - t0) * 1000, 1)
        if r.status_code != 200:
            return {"ok": False, "latency_ms": latency,
                    "detail": f"HTTP {r.status_code}"}
        try:
            data = r.json()
        except Exception:
            return {"ok": False, "latency_ms": latency,
                    "detail": "non-json response"}
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict) or "zone" not in result:
            return {"ok": False, "latency_ms": latency,
                    "detail": f"no zone in result: {str(result)[:120]}"}
        return {"ok": True, "latency_ms": latency,
                "detail": f"zone={result.get('zone')}"}
    except Exception as e:
        latency = round((time.time() - t0) * 1000, 1)
        return {"ok": False, "latency_ms": latency,
                "detail": f"err:{type(e).__name__}:{str(e)[:100]}"}


def collect() -> Dict[str, Any]:
    """7-Kanaele-Health-Layer fuer Pi<->PC-Uebergang."""
    settings = _load_settings()

    # Hosts/Ports aus settings (Fallbacks beibehalten)
    tentacle = settings.get("tentacle_llm") or {}
    adapter = settings.get("adapter_inference") or {}
    pc_host_ollama = tentacle.get("host") or DEFAULT_PC_HOST
    pc_port_ollama = int(tentacle.get("port") or DEFAULT_OLLAMA_PORT)
    pc_host_adapter = adapter.get("host") or DEFAULT_PC_HOST
    pc_port_adapter = int(adapter.get("port") or DEFAULT_ADAPTER_PORT)
    # search_proxy gilt als PC-Service auf gleichem Host wie tentacle (Konvention)
    pc_host_search = pc_host_ollama

    channels: Dict[str, Dict[str, Any]] = {}

    # Sequenziell — Gesamtbudget HTTP_TIMEOUT_S * 6 = 30s max
    channels["chat_server"] = _check_chat_server()
    channels["search_proxy"] = _check_search_proxy(pc_host_search)
    channels["ollama_tentakel"] = _check_ollama_tentakel(pc_host_ollama, pc_port_ollama)
    channels["adapter_inference"] = _check_adapter_inference(pc_host_adapter, pc_port_adapter)
    channels["mailbox_freshness"] = _check_mailbox_freshness()
    channels["federation_heartbeat"] = _check_federation_heartbeat()
    channels["tool_api"] = _check_tool_api()

    alive_count = sum(1 for c in channels.values() if c.get("alive"))
    total = len(channels)

    # Status-Logik
    if alive_count == total:
        status = "PASS"
    elif alive_count >= total - 1:
        # 1 Kanal darf fehlen (z.B. federation_heartbeat)
        status = "WARN"
    elif alive_count >= total // 2:
        status = "WARN"
    else:
        status = "FAIL"

    # E2E-Roundtrip nur wenn beide lokalen Endpoints alive
    e2e_result: Dict[str, Any] = {"skipped": True,
                                   "reason": "chat_server or tool_api down"}
    if (channels["chat_server"].get("alive")
            and channels["tool_api"].get("alive")):
        e2e_result = _e2e_roundtrip()

    return {
        "score": alive_count,
        "max": total,
        "status": status,
        "channels": channels,
        "alive_count": alive_count,
        "detail": {"e2e_roundtrip": e2e_result},
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(collect(), indent=2, ensure_ascii=False))
    sys.exit(0)
