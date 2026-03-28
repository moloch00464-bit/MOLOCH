#!/usr/bin/env python3
"""
M.O.L.O.C.H. Local LLM Bridge — Gate 7
========================================
Einheitliche Schnittstelle fuer lokale + Cloud LLM Reasoning.

Prioritaet:
  1. hailo-ollama (Qwen2.5-1.5B auf NPU, 8GB LPDDR4) — offline, schnell
  2. DeepSeek API (Cloud) — online, guenstig
  3. Claude API (Cloud) — online, Fallback
  4. Stille — kein Crash, kein Fehler, nur keine Antwort

WICHTIG: Lokales LLM erfordert Vision-Pipeline PAUSE!
NPU kann nicht gleichzeitig Vision + LLM. Schicht-3-Architektur
aus NPU_DREI_SCHICHTEN_ARCHITEKTUR.md beachten.

Singleton: get_llm_bridge()
"""

import logging
import os
import subprocess
import threading
import time
from typing import Optional, Dict, Callable

logger = logging.getLogger("LocalLLMBridge")

# hailo-ollama Konfiguration
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 30  # Sekunden


class LocalLLMBridge:
    """Einheitliche LLM-Schnittstelle mit Fallback-Kette."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ollama_available: Optional[bool] = None
        self._vision_pause_callback: Optional[Callable] = None
        self._vision_resume_callback: Optional[Callable] = None
        self._last_provider: str = "none"
        self._request_count: int = 0
        self._check_ollama()
        logger.info(f"[LLM-BRIDGE] Init — hailo-ollama={'JA' if self._ollama_available else 'NEIN'}")

    def _check_ollama(self):
        """Pruefen ob hailo-ollama installiert und erreichbar ist."""
        try:
            result = subprocess.run(
                ["which", "hailo-ollama"], capture_output=True, timeout=5)
            if result.returncode == 0:
                self._ollama_available = True
                return
        except Exception:
            pass
        # Alternativ: ollama direkt
        try:
            result = subprocess.run(
                ["which", "ollama"], capture_output=True, timeout=5)
            if result.returncode == 0:
                self._ollama_available = True
                return
        except Exception:
            pass
        self._ollama_available = False

    def set_vision_callbacks(self, pause_fn: Callable, resume_fn: Callable):
        """Callbacks fuer Vision-Pipeline Pause/Resume registrieren.

        KRITISCH: Lokales LLM braucht vollen NPU-Zugriff.
        Vision muss pausiert werden bevor LLM laeuft.
        """
        self._vision_pause_callback = pause_fn
        self._vision_resume_callback = resume_fn
        logger.info("[LLM-BRIDGE] Vision-Callbacks registriert")

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 512, use_local: bool = False) -> Optional[str]:
        """Text generieren — waehlt automatisch den besten Provider.

        Args:
            prompt: User/System Prompt
            system: System-Prompt (optional)
            max_tokens: Maximale Antwortlaenge
            use_local: True = lokales LLM erzwingen (pausiert Vision!)

        Returns: Generierter Text oder None bei Fehler
        """
        with self._lock:
            self._request_count += 1

        # 1. Lokal (hailo-ollama) wenn verfuegbar UND gewuenscht
        if use_local and self._ollama_available:
            result = self._generate_local(prompt, system, max_tokens)
            if result:
                return result

        # 2. DeepSeek API (Cloud)
        result = self._generate_deepseek(prompt, system, max_tokens)
        if result:
            return result

        # 3. Claude API (Fallback)
        result = self._generate_claude(prompt, system, max_tokens)
        if result:
            return result

        # 4. Stille
        self._last_provider = "stille"
        return None

    def _generate_local(self, prompt: str, system: str,
                        max_tokens: int) -> Optional[str]:
        """Lokales LLM via ollama API. PAUSIERT VISION!"""
        if not self._ollama_available:
            return None

        try:
            # Vision pausieren (NPU freigeben)
            if self._vision_pause_callback:
                logger.info("[LLM-BRIDGE] Vision pausieren fuer lokales LLM...")
                self._vision_pause_callback()
                time.sleep(1.0)  # NPU-Freigabe abwarten

            import requests
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._last_provider = "local_ollama"
            return data.get("response", "").strip()

        except Exception as e:
            logger.warning(f"[LLM-BRIDGE] Lokales LLM Fehler: {e}")
            return None
        finally:
            # Vision IMMER wieder starten
            if self._vision_resume_callback:
                logger.info("[LLM-BRIDGE] Vision fortsetzen...")
                self._vision_resume_callback()

    def _generate_deepseek(self, prompt: str, system: str,
                           max_tokens: int) -> Optional[str]:
        """DeepSeek API (Cloud, guenstig)."""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        try:
            import requests
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "deepseek-chat", "messages": messages,
                      "max_tokens": max_tokens},
                timeout=15)
            resp.raise_for_status()
            self._last_provider = "deepseek"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"[LLM-BRIDGE] DeepSeek Fehler: {e}")
            return None

    def _generate_claude(self, prompt: str, system: str,
                         max_tokens: int) -> Optional[str]:
        """Claude API (Fallback)."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import requests
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001",
                      "max_tokens": max_tokens,
                      "system": system,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=15)
            resp.raise_for_status()
            self._last_provider = "claude"
            return resp.json()["content"][0]["text"].strip()
        except Exception as e:
            logger.debug(f"[LLM-BRIDGE] Claude Fehler: {e}")
            return None

    def get_status(self) -> Dict:
        return {
            "ollama_available": self._ollama_available,
            "last_provider": self._last_provider,
            "request_count": self._request_count,
            "vision_callbacks": self._vision_pause_callback is not None,
        }


# Singleton
_instance: Optional[LocalLLMBridge] = None

def get_llm_bridge() -> LocalLLMBridge:
    global _instance
    if _instance is None:
        _instance = LocalLLMBridge()
    return _instance
