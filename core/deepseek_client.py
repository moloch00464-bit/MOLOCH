#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. DeepSeek API Client

Minimaler Client fuer die DeepSeek Chat API (OpenAI-kompatibel).
Nutzt requests direkt — kein openai-Paket noetig, spart RAM auf Pi5.

Singleton: get_deepseek() -> globale Instanz

API Docs: https://api-docs.deepseek.com/
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)

# Config-Pfad (gleicher Ort wie Anthropic Key)
API_KEYS_PATH = Path.home() / "moloch" / "config" / "api_keys.json"

# Defaults
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 30  # Sekunden
MAX_RETRIES = 2


class DeepSeekClient:
    """
    DeepSeek Chat API Client fuer M.O.L.O.C.H.

    Laedt API-Key aus config/api_keys.json.
    OpenAI-kompatibles Chat-Completion Interface.
    Thread-safe, sparsam, robust.
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._base_url: str = DEFAULT_BASE_URL
        self._model: str = DEFAULT_MODEL
        self._session: Optional[requests.Session] = None
        self._lock = threading.Lock()

        self._load_config()
        self._init_session()

        if self._api_key:
            logger.info("DeepSeek Client initialisiert (Model: %s)", self._model)
        else:
            logger.warning("DeepSeek API Key NICHT gefunden in %s", API_KEYS_PATH)

    def _load_config(self):
        """Laedt DeepSeek-Config aus api_keys.json."""
        if not API_KEYS_PATH.exists():
            logger.error("api_keys.json nicht gefunden: %s", API_KEYS_PATH)
            return

        try:
            with open(API_KEYS_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)

            ds_config = config.get("deepseek", {})
            self._api_key = ds_config.get("api_key")
            self._base_url = ds_config.get("base_url", DEFAULT_BASE_URL)
            self._model = ds_config.get("model", DEFAULT_MODEL)

        except (json.JSONDecodeError, IOError) as e:
            logger.error("Fehler beim Laden von api_keys.json: %s", e)

    def _init_session(self):
        """Erstellt requests Session mit Auth-Header."""
        self._session = requests.Session()
        if self._api_key:
            self._session.headers.update({
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            })

    @property
    def available(self) -> bool:
        """Prueft ob der Client einsatzbereit ist."""
        return self._api_key is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Optional[str]:
        """
        Sendet Chat-Completion Request an DeepSeek API.

        Args:
            messages: Liste von {"role": "user"|"assistant", "content": "..."}
            model: Modell-Override (default: aus Config)
            temperature: Kreativitaet (0.0-2.0)
            max_tokens: Max Antwort-Laenge
            system: System-Prompt (wird als erste Message eingefuegt)
            stream: Streaming-Modus (noch nicht implementiert)

        Returns:
            Antwort-Text oder None bei Fehler
        """
        if not self.available:
            logger.error("DeepSeek nicht verfuegbar — kein API Key")
            return None

        # System-Prompt vorne einfuegen
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": model or self._model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,  # Streaming spaeter
        }

        url = f"{self._base_url}/chat/completions"

        # Retry-Logik
        for attempt in range(MAX_RETRIES + 1):
            try:
                with self._lock:
                    response = self._session.post(
                        url,
                        json=payload,
                        timeout=DEFAULT_TIMEOUT,
                    )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    logger.debug(
                        "DeepSeek OK — Tokens: %d prompt + %d completion",
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )
                    return content

                elif response.status_code == 429:
                    # Rate Limit — kurz warten und retry
                    wait = min(2 ** attempt, 8)
                    logger.warning("DeepSeek Rate Limit — warte %ds", wait)
                    time.sleep(wait)
                    continue

                elif response.status_code == 401:
                    logger.error("DeepSeek Auth Fehler — API Key ungueltig?")
                    return None

                else:
                    logger.error(
                        "DeepSeek Fehler %d: %s",
                        response.status_code,
                        response.text[:200],
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.warning("DeepSeek Timeout (Versuch %d/%d)", attempt + 1, MAX_RETRIES + 1)
                if attempt < MAX_RETRIES:
                    continue
                return None

            except requests.exceptions.ConnectionError as e:
                logger.error("DeepSeek Verbindungsfehler: %s", e)
                return None

            except Exception as e:
                logger.error("DeepSeek unerwarteter Fehler: %s", e)
                return None

        return None

    def ask(self, question: str, system: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Kurzform: Einzelne Frage stellen.

        Args:
            question: Die Frage
            system: Optionaler System-Prompt
            **kwargs: Weitere Parameter fuer chat()

        Returns:
            Antwort-Text oder None
        """
        messages = [{"role": "user", "content": question}]
        return self.chat(messages, system=system, **kwargs)

    def stop(self):
        """Session sauber schliessen."""
        if self._session:
            self._session.close()
            logger.info("DeepSeek Session geschlossen")


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[DeepSeekClient] = None
_instance_lock = threading.Lock()


def get_deepseek() -> DeepSeekClient:
    """Globale DeepSeekClient Instanz."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DeepSeekClient()
    return _instance
