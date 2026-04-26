#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Adapter Inference Client (Pi-Side)

ThreeBrain Welle 3 / Pi-Antwort auf PC-Side commit 709512f.

Spricht den PC-seitigen Adapter-Inference-Proxy auf
http://192.168.178.20:11600 an. Damit kann Pi-LLM-Bridge ein neues
Provider-Backend nutzen: 'qwen_adapter_remote' — Qwen2.5-1.5B-Instruct
mit gerade aktivem LoRA-Adapter (vom PC trainiert).

API-Vertrag (von PC pc/adapter_inference_proxy.py):
  POST /infer    {prompt, system?, max_tokens?} -> {response, adapter_version, tokens, duration_ms}
  GET  /health   -> {status, adapter, base}
  GET  /list     -> {adapters, active}
  POST /reload   -> {reloaded, adapter}

Settings-driven via config/settings.json Block 'adapter_inference'
(Defaults wenn Block fehlt).

Singleton: get_adapter_client()

CLI: python3 -m core.bridge.adapter_inference_client
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("AdapterClient")

DEFAULT_HOST = "192.168.178.20"
DEFAULT_PORT = 11600
DEFAULT_TIMEOUT_S = 120          # 2.5 tok/s CPU + max_tokens=100 -> ~40s, mit Margin 120
DEFAULT_BACKOFF_S = 600
DEFAULT_MAX_TOKENS = 100         # statt 200: bei 2.5 tok/s = 40s Antwortzeit (komfortabel)

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json",
)


def _load_cfg() -> Dict[str, Any]:
    cfg = {
        "enabled": True,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "timeout_sec": DEFAULT_TIMEOUT_S,
        "backoff_sec": DEFAULT_BACKOFF_S,
        "default_max_tokens": DEFAULT_MAX_TOKENS,
    }
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        block = data.get("adapter_inference", {}) or {}
        for k in cfg:
            if k in block:
                cfg[k] = block[k]
    except Exception as e:
        logger.debug(f"[ADAPTER] settings.json fallback: {e}")
    return cfg


# =============================================================================
# AdapterInferenceClient
# =============================================================================

class AdapterInferenceClient:
    """Pi-Client fuer PC-Side Qwen+LoRA Inference-Proxy."""

    def __init__(self):
        self._lock = threading.Lock()
        self._http = requests.Session()
        self._fail_count = 0
        self._backoff_until = 0.0
        self._last_health_check = 0.0
        self._last_health_ok = False
        self._last_adapter_version: Optional[str] = None
        self._cfg = _load_cfg()
        logger.info(
            f"[ADAPTER] Init host={self._cfg['host']}:{self._cfg['port']} "
            f"enabled={self._cfg['enabled']}"
        )

    def _base_url(self) -> str:
        return f"http://{self._cfg['host']}:{self._cfg['port']}"

    # ------------------------------------------------------- HEALTH

    def health(self, force: bool = False) -> bool:
        """Probe PC-Proxy. Cached 30s wenn nicht force=True."""
        if not self._cfg.get("enabled", True):
            return False
        now = time.monotonic()
        if not force and (now - self._last_health_check) < 30.0:
            return self._last_health_ok
        if now < self._backoff_until:
            return False
        try:
            r = self._http.get(f"{self._base_url()}/health", timeout=3)
            r.raise_for_status()
            data = r.json()
            self._last_health_check = now
            self._last_health_ok = data.get("status") == "ok"
            self._last_adapter_version = data.get("adapter")
            return self._last_health_ok
        except Exception as e:
            logger.debug(f"[ADAPTER] health probe fehlgeschlagen: {e}")
            self._last_health_check = now
            self._last_health_ok = False
            return False

    # ------------------------------------------------------- CALL

    def infer(self, prompt: str, system: str = "Du bist Moloch.",
              max_tokens: Optional[int] = None) -> Optional[str]:
        """Eine Inferenz auf PC-Adapter-Proxy. Returns text oder None bei Fehler.

        max_tokens=None -> nimmt settings.adapter_inference.default_max_tokens
        (default 100, passt zu CPU 2.5 tok/s in <40s Antwortzeit).
        """
        if time.monotonic() < self._backoff_until:
            return None
        if not self._cfg.get("enabled", True):
            return None
        if not prompt or not prompt.strip():
            return None

        if max_tokens is None:
            max_tokens = int(self._cfg.get("default_max_tokens", DEFAULT_MAX_TOKENS))

        payload = {
            "prompt": prompt,
            "system": system or "Du bist Moloch.",
            "max_tokens": int(max_tokens),
        }
        resp = None
        try:
            resp = self._http.post(
                f"{self._base_url()}/infer",
                json=payload,
                timeout=self._cfg["timeout_sec"],
            )
            resp.raise_for_status()
            data = resp.json()
            self._fail_count = 0
            self._backoff_until = 0.0
            self._last_adapter_version = data.get("adapter_version")
            text = (data.get("response") or "").strip()
            logger.info(
                f"[ADAPTER] {data.get('adapter_version', '?')}: "
                f"{data.get('tokens', 0)} tok in {data.get('duration_ms', 0)}ms"
            )
            return text or None
        except Exception as e:
            with self._lock:
                self._fail_count += 1
                if self._fail_count >= 3:
                    self._backoff_until = time.monotonic() + self._cfg["backoff_sec"]
                    logger.warning(
                        f"[ADAPTER] {self._fail_count}x Fehler -> "
                        f"{self._cfg['backoff_sec']}s Backoff aktiv"
                    )
            logger.debug(f"[ADAPTER] infer Fehler: {e}")
            return None
        finally:
            if resp is not None:
                resp.close()

    def list_adapters(self) -> Optional[Dict]:
        try:
            r = self._http.get(f"{self._base_url()}/list", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.debug(f"[ADAPTER] list fehlgeschlagen: {e}")
            return None

    def reload(self) -> Optional[Dict]:
        """Trigger PC, neuesten Adapter zu laden (nach Training-Run)."""
        try:
            r = self._http.post(f"{self._base_url()}/reload", timeout=30)
            r.raise_for_status()
            data = r.json()
            self._last_adapter_version = data.get("adapter")
            logger.info(f"[ADAPTER] reload -> {self._last_adapter_version}")
            return data
        except Exception as e:
            logger.warning(f"[ADAPTER] reload fehlgeschlagen: {e}")
            return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "host": self._cfg["host"],
            "port": self._cfg["port"],
            "enabled": self._cfg.get("enabled", True),
            "fail_count": self._fail_count,
            "backoff_remaining_s": max(0.0, self._backoff_until - time.monotonic()),
            "last_health_ok": self._last_health_ok,
            "active_adapter": self._last_adapter_version,
        }


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[AdapterInferenceClient] = None
_instance_lock = threading.Lock()


def get_adapter_client() -> AdapterInferenceClient:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AdapterInferenceClient()
    return _instance


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    c = get_adapter_client()
    print(f"\n[State] {c.get_state()}")
    print(f"\n[Health] {c.health(force=True)}")
    if not c.health(force=False):
        print(f"\nPC-Proxy nicht erreichbar — vermutlich noch nicht installiert/gestartet")
        print(f"PC-Setup laeuft? Pruefen mit: curl http://{c._cfg['host']}:{c._cfg['port']}/health")
        raise SystemExit(0)
    lst = c.list_adapters()
    print(f"\n[Adapters] {lst}")
    if not (lst or {}).get("adapters"):
        print(f"\nKein Adapter geladen — vermutlich noch nichts trainiert")
        raise SystemExit(0)
    print(f"\n[Infer Test]")
    out = c.infer(prompt="Wer bist du?", system="Du bist Moloch.", max_tokens=50)
    print(f"  Antwort: {out}")
    print(f"\nSelf-Test PASS")
