#!/usr/bin/env python3
"""
M.O.L.O.C.H. Local LLM Bridge — Gate 7
========================================
Einheitliche Schnittstelle fuer lokale + Cloud LLM Reasoning.

Prioritaet (Fallback-Kette):
  1. hailo-ollama (Port 8000) — Qwen2.5 oder DeepSeek R1 lokal auf NPU
  2. DeepSeek API (Cloud) — online, guenstig
  3. Stille — kein Crash, kein Fehler, nur keine Antwort
  (Claude API wurde entfernt — nur DeepSeek als Cloud-Fallback)

Zwei Rollen:
  - ask_external(prompt) → DeepSeek R1 fuer Konversation (Deutsch)
  - reason_internal(prompt) → DeepSeek R1 fuer Selbstdiagnose/Logik

WICHTIG: hailo-ollama muss separat laufen (systemd oder manuell).
Vision laeuft weiter waehrend hailo-ollama antwortet — hailo-ollama
managed den NPU-Zugriff selbst via shared VDevice.

Singleton: get_llm_bridge()
"""

import json
import logging
import os
import requests
import subprocess
import threading
import time
from typing import Optional, Dict, Callable

logger = logging.getLogger("LocalLLMBridge")

# hailo-ollama Konfiguration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:8000")
OLLAMA_MODEL_CHAT = "deepseek_r1_distill_qwen:1.5b"    # R1 = Hauptmodell fuer alles
OLLAMA_MODEL_REASON = "deepseek_r1_distill_qwen:1.5b"
OLLAMA_TIMEOUT_CHAT = 90      # R1 braucht ~80s (Chain-of-Thought) — 30s war zu kurz
OLLAMA_TIMEOUT_REASON = 120   # DeepSeek R1 braucht ~80s (Chain-of-Thought)
OLLAMA_MAX_INPUT_CHARS = 12000  # ~3000 Tokens Safety-Limit (Qwen2.5-1.5B: 4096 Kontext, 256 Output)


class LocalLLMBridge:
    """Einheitliche LLM-Schnittstelle mit Fallback-Kette."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ollama_available: Optional[bool] = None
        self._vision_pause_callback: Optional[Callable] = None
        self._vision_resume_callback: Optional[Callable] = None
        self._last_provider: str = "none"
        self._request_count: int = 0
        # Circuit-Breaker: Ollama automatisch ueberbruecken wenn wiederholt offline
        self._ollama_fail_count: int = 0
        self._ollama_backoff_until: float = 0.0
        self.OLLAMA_BACKOFF_SEC: int = 300  # 5 Minuten Cloud-Backoff
        # Wiederverwendbare HTTP-Session — verhindert RAM-Leak durch offene Sockets
        self._http = requests.Session()
        self._check_ollama()
        logger.info(
            f"[LLM-BRIDGE] Init — hailo-ollama={'JA' if self._ollama_available else 'NEIN'}"
        )

    def _check_ollama(self):
        """Pruefen ob hailo-ollama installiert ist."""
        try:
            result = subprocess.run(
                ["which", "hailo-ollama"], capture_output=True, timeout=5)
            self._ollama_available = result.returncode == 0
        except Exception:
            self._ollama_available = False

    def _is_ollama_running(self) -> bool:
        """Pruefen ob hailo-ollama Prozess laeuft (Port 8000 erreichbar)."""
        resp = None
        try:
            resp = self._http.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False
        finally:
            if resp is not None:
                resp.close()

    def set_vision_callbacks(self, pause_fn: Callable, resume_fn: Callable):
        """Callbacks fuer Vision-Pipeline Pause/Resume registrieren."""
        self._vision_pause_callback = pause_fn
        self._vision_resume_callback = resume_fn
        logger.info("[LLM-BRIDGE] Vision-Callbacks registriert")

    # === Oeffentliche Methoden: Zwei Rollen ===

    def ask_external(self, prompt: str, system: str = "",
                     max_tokens: int = 256,
                     temperature: float = 0.8,
                     top_p: float = 0.95,
                     force_local: bool = False,
                     use_reason_model: bool = False) -> Optional[str]:
        """Konversation: lokal auf NPU → DeepSeek API → Stille.

        Fuer Echtzeit-Dialog mit Markus. Kurze Antworten, Deutsch.
        temperature/top_p steuern Guardian- vs Shadow-Tonalität.
        force_local=True: kein Cloud-Fallback, Prompt wird gekuerzt wenn noetig.
        use_reason_model=True: DeepSeek R1 statt Qwen2.5 (laenger, aber besser).
        """
        with self._lock:
            self._request_count += 1

        model = OLLAMA_MODEL_REASON if use_reason_model else OLLAMA_MODEL_CHAT
        timeout = OLLAMA_TIMEOUT_REASON if use_reason_model else OLLAMA_TIMEOUT_CHAT

        # 1. hailo-ollama lokal auf NPU (R1 oder Qwen2.5)
        result = self._generate_ollama(prompt, system, max_tokens,
                                       model=model,
                                       timeout=timeout,
                                       temperature=temperature,
                                       top_p=top_p,
                                       force_local=force_local)
        if result:
            return result

        # 2. DeepSeek API (Cloud-Fallback) — nur wenn nicht force_local
        if not force_local:
            result = self._generate_deepseek(prompt, system, max_tokens)
            if result:
                return result

        # 3. Stille
        self._last_provider = "stille"
        return None

    def reason_internal(self, prompt: str, system: str = "",
                        max_tokens: int = 512) -> Optional[str]:
        """Internes Reasoning: DeepSeek R1 lokal → DeepSeek API → None.

        Fuer Selbstdiagnose, Entscheidungen, Systemchecks. Nicht fuer TTS.
        """
        with self._lock:
            self._request_count += 1

        # 1. hailo-ollama DeepSeek R1 (lokal)
        result = self._generate_ollama(prompt, system, max_tokens,
                                       model=OLLAMA_MODEL_REASON,
                                       timeout=OLLAMA_TIMEOUT_REASON)
        if result:
            return result

        # 2. DeepSeek API als Fallback
        result = self._generate_deepseek(prompt, system, max_tokens)
        if result:
            return result

        # 3. Stille (kein Claude fuer internes Reasoning)
        self._last_provider = "stille"
        return None

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 512, use_local: bool = False) -> Optional[str]:
        """Legacy-Methode: Waehlt automatisch den besten Provider.

        Bei use_local=True wird Qwen2.5 lokal bevorzugt.
        """
        if use_local:
            return self.ask_external(prompt, system, max_tokens)
        # Ohne use_local: DeepSeek Cloud direkt
        result = self._generate_deepseek(prompt, system, max_tokens)
        if result:
            return result
        self._last_provider = "stille"
        return None

    # === Private: Provider-Implementierungen ===

    def _generate_ollama(self, prompt: str, system: str,
                         max_tokens: int, model: str,
                         timeout: int,
                         temperature: float = 0.8,
                         top_p: float = 0.95,
                         force_local: bool = False) -> Optional[str]:
        """hailo-ollama Chat API (Port 8000) mit Circuit-Breaker."""
        if not self._ollama_available:
            return None

        # Circuit-Breaker: Backoff aktiv?
        # Bei force_local trotzdem versuchen — Moloch soll lokal antworten
        if time.monotonic() < self._ollama_backoff_until:
            if not force_local:
                verbleibend = int(self._ollama_backoff_until - time.monotonic())
                logger.info(f"[LLM] Ollama Backoff aktiv ({verbleibend}s), direkt Cloud")
                return None
            logger.info("[LLM] force_local: ignoriere Backoff, versuche lokal")

        # Health-Check: nicht erreichbar → Fehlerzaehler erhoehen
        if not self._is_ollama_running():
            self._ollama_fail_count += 1
            if self._ollama_fail_count >= 3:
                self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                logger.warning(
                    f"[LLM] Ollama {self._ollama_fail_count}x down → "
                    f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff"
                )
            logger.debug("[LLM-BRIDGE] hailo-ollama nicht erreichbar")
            return None

        # Input-Length-Check
        input_len = len(system) + len(prompt)
        if input_len > OLLAMA_MAX_INPUT_CHARS:
            if not force_local:
                logger.info(f"[LLM] Input zu lang ({input_len} Zeichen > {OLLAMA_MAX_INPUT_CHARS}) → Cloud-Fallback")
                return None
            # force_local: Prompt kuerzen statt zur Cloud zu fallen
            # System-Prompt bleibt intact, User-Prompt wird von hinten beibehalten
            allowed = OLLAMA_MAX_INPUT_CHARS - len(system) - 100
            if allowed < 300:
                logger.warning("[LLM] force_local: System-Prompt zu lang, kein Platz fuer User-Input")
                return None
            prompt = prompt[-allowed:]
            logger.info(f"[LLM] force_local: Prompt auf {len(prompt)} Zeichen gekuerzt (Tension/Shadow/Berserker)")

        # Vision pausieren damit hailo-ollama die NPU exklusiv nutzen kann
        _vision_paused = False
        if self._vision_pause_callback:
            try:
                self._vision_pause_callback()
                _vision_paused = True
                logger.info("[LLM] Vision pausiert fuer LLM-Inference")
                time.sleep(3)  # 3s warten bis TAPPAS NPU vollstaendig freigegeben hat
            except Exception as e:
                logger.warning(f"[LLM] Vision-Pause fehlgeschlagen: {e}")

        try:
            resp = None
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                resp = self._http.post(
                    f"{OLLAMA_HOST}/api/chat",
                    json={"model": model, "messages": messages, "stream": False,
                          "options": {"num_predict": max_tokens,
                                      "temperature": temperature,
                                      "top_p": top_p}},
                    timeout=timeout)
                resp.raise_for_status()
                # Explizit UTF-8 dekodieren — resp.json() kann bei fehlendem charset-Header
                # Latin-1 waehlen → Umlaute werden als Ã¼ statt ü dargestellt
                data = json.loads(resp.content.decode('utf-8'))
                text = data.get("message", {}).get("content", "").strip()

                # DeepSeek R1 <think>...</think> Block entfernen (nur Antwort behalten)
                import re
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

                if not text:
                    return None

                # Erfolg: Circuit-Breaker zuruecksetzen
                self._ollama_fail_count = 0
                self._ollama_backoff_until = 0.0
                self._last_provider = f"lokal_{model.split(':')[0]}"
                logger.info(
                    f"[LLM-BRIDGE] {model}: {len(text)} Zeichen in "
                    f"{data.get('total_duration', 0) // 1_000_000}ms"
                )
                return text

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                # Verbindungsfehler waehrend Generation → Fehlerzaehler
                self._ollama_fail_count += 1
                if self._ollama_fail_count >= 3:
                    self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                    logger.warning(
                        f"[LLM] Ollama {self._ollama_fail_count}x Verbindungsfehler → "
                        f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff"
                    )
                logger.warning(f"[LLM-BRIDGE] hailo-ollama ({model}) Verbindungsfehler: {e}")
                return None

            except Exception as e:
                # HTTP 500 und andere Fehler: auch im Circuit-Breaker zaehlen
                self._ollama_fail_count += 1
                if self._ollama_fail_count >= 3:
                    self._ollama_backoff_until = time.monotonic() + self.OLLAMA_BACKOFF_SEC
                    logger.warning(
                        f"[LLM] Ollama {self._ollama_fail_count}x Fehler → "
                        f"{self.OLLAMA_BACKOFF_SEC}s Cloud-Backoff aktiv"
                    )
                logger.warning(f"[LLM-BRIDGE] hailo-ollama ({model}) Fehler: {e}")
                return None

            finally:
                if resp is not None:
                    resp.close()

        finally:
            # Vision IMMER wieder starten — auch bei Fehler
            if _vision_paused and self._vision_resume_callback:
                try:
                    # hailo-ollama mit KEEP_ALIVE=0 gibt NPU sofort frei nach Inference
                    # Trotzdem 2s warten fuer saubere Freigabe
                    time.sleep(2)
                    self._vision_resume_callback()
                    logger.info("[LLM] Vision wieder aktiv")
                except Exception as e:
                    logger.warning(f"[LLM] Vision-Resume fehlgeschlagen: {e}")

    def _load_api_key(self, provider: str) -> Optional[str]:
        """API Key aus config/api_keys.json laden."""
        keys_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "config", "api_keys.json")
        # Env-Var hat Vorrang
        env_key = os.environ.get(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key
        try:
            import json
            with open(keys_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(provider, {}).get("api_key")
        except Exception:
            return None

    def _generate_deepseek(self, prompt: str, system: str,
                           max_tokens: int) -> Optional[str]:
        """DeepSeek API (Cloud, guenstig)."""
        api_key = self._load_api_key("deepseek")
        if not api_key:
            return None
        resp = None
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._http.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "deepseek-chat", "messages": messages,
                      "max_tokens": max_tokens},
                timeout=15)
            resp.raise_for_status()
            self._last_provider = "api_deepseek"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.debug(f"[LLM-BRIDGE] DeepSeek Fehler: {e}")
            return None
        finally:
            if resp is not None:
                resp.close()

    def get_status(self) -> Dict:
        now = time.monotonic()
        backoff_remaining = max(0.0, self._ollama_backoff_until - now)
        return {
            "ollama_installed": self._ollama_available,
            "ollama_running": self._is_ollama_running() if self._ollama_available else False,
            "ollama_fail_count": self._ollama_fail_count,
            "ollama_backoff_sec": round(backoff_remaining),
            "last_provider": self._last_provider,
            "request_count": self._request_count,
            "models": {
                "chat": OLLAMA_MODEL_CHAT,
                "reason": OLLAMA_MODEL_REASON,
            },
        }


# Singleton
_instance: Optional[LocalLLMBridge] = None

def get_llm_bridge() -> LocalLLMBridge:
    global _instance
    if _instance is None:
        _instance = LocalLLMBridge()
    return _instance
