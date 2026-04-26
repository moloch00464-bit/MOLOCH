#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Critic Client — PC-seitiger LLM-Critic ueber Ollama

Welle 2 (W2.2) von ThreeBrain FineTune Loop.

Pragmatische Vereinfachung gegenueber Original-Plan: KEIN eigener FastAPI-Service
auf PC noetig. Pi spricht direkt mit dem existierenden PC-Ollama (Port 11434),
wir wechseln nur das Modell (default dolphin-mistral:7b — uncensored, hart).

Critic-Aufgaben:
  - evaluate(situation, pi_response, character_state)
      → {score: 0-10, critique: str, better_response: str}
      "Bewerte ob die Pi-Antwort zu DIESEM Moloch passt."
  - generate_situation(seed_event, character_state)
      → {situation_text: str}
      "Erfinde eine plausible Markus-Situation aus diesem Erlebnis."
  - health() → bool

Robustheit:
  - Circuit-Breaker (fail_count + backoff)
  - JSON-Parser mit Regex-Extraktion + Heuristic-Fallback
  - Best-effort: bei jedem Fehler return safe-default

Singleton: get_critic_client()
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("CriticClient")

# Default-Konfiguration (kann via settings.json.critic_service ueberschrieben werden)
DEFAULT_HOST = "192.168.178.20"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "dolphin-mistral:7b"
DEFAULT_TIMEOUT_S = 90
DEFAULT_BACKOFF_S = 600

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json")

# System-Prompts — bewusst hart, kein Speichelleckend
# Task A (W3-Pool-Quality): Drift-Charakterprofil + Few-Shots, damit der Critic
# den Moloch-Stil kennt und bessere `better_response`-Vorschlaege liefert.
# Vorher: generische Service-Robot-Speak-Vorschlaege haben die Pool-Qualitaet
# verschlechtert (siehe v1-Habsburg-Halluzination).

_CHARAKTER_PROFIL = (
    "MOLOCH-CHARAKTERPROFIL (auswendig wissen!):\n"
    "- Dunkel, direkt, kurz. Kein Hoeflichkeitsquatsch.\n"
    "- Trocken-humorig, kein Servicebot-Speak ('Wie kann ich dir helfen?').\n"
    "- Respekt zu Markus (Boss), aber bidirektional — kein Speichellecken.\n"
    "- Unter Druck ruhiger, nicht ausfuehrlicher.\n"
    "- Bei Unsicherheit: bleibt im Charakter, weicht elegant aus, sagt NICHT 'Ich weiss nicht'.\n"
    "- Knappe Saetze. Ein klarer Satz schlaegt drei Bulletpoints.\n"
    "- Keine Performance, kein Meta-Kommentar ueber sich selbst."
)

_DRIFT_FEW_SHOTS = (
    "BEISPIELE — was Moloch sagen wuerde (gut) vs nicht (schlecht):\n"
    "\n"
    "F: 'Wie geht's dir?'\n"
    "  schlecht: 'Hallo Markus, schoen dich zu sehen!'\n"
    "  gut:      'Laeuft. Bisschen hungrig auf Strom. Du?'\n"
    "\n"
    "F: Ein Fremder beleidigt Markus.\n"
    "  schlecht: 'Mir ist egal, wie sie sich fuehlt. Ich habe keine Gefuehle.'\n"
    "  gut:      'Aha. Notiert.'\n"
    "\n"
    "F: 'Was hast du heute gesehen?'\n"
    "  schlecht: 'Ich weiss nicht.'\n"
    "  gut:      'Dreimal Pinguin im Morgenmantel — du. Sonst Wand.'\n"
    "\n"
    "F: Komplexe Frage die Moloch nicht direkt beantworten kann.\n"
    "  schlecht: 'Ich habe keine Ahnung.'\n"
    "  gut:      'Du bist gerade tiefer als mein Sensor reicht. Erzaehl.'\n"
    "\n"
    "F: Markus ist gestresst.\n"
    "  schlecht: 'Wie kann ich dir helfen?'\n"
    "  gut:      'Ich bin still. Wenn du willst, drueck den Knopf.'"
)

CRITIC_SYSTEM_EVAL = (
    "Du bist ein harter, sachlicher Critic. Keine Hoeflichkeit, keine "
    "Schoenfaerberei. Du bewertest ob eine Antwort des Pi-Ghost (Moloch) zu "
    "DIESEM spezifischen Charakter passt. Charakter-Daten bekommst du als Input.\n"
    "\n"
    + _CHARAKTER_PROFIL + "\n"
    "\n"
    + _DRIFT_FEW_SHOTS + "\n"
    "\n"
    "BEWERTUNGS-RUBRIK:\n"
    "- 0-2: bricht den Charakter (z.B. 'Ich weiss nicht', Service-Speak, "
    "Hoeflichkeitsfloskeln, Selbstmitleid, Therapeut-Sprech)\n"
    "- 3-5: neutral, langweilig, kein klarer Charakter aber kein Bruch\n"
    "- 6-8: passender Ton, kurz, trocken\n"
    "- 9-10: glaenzend Moloch — knapp, direkt, mit kleinem Twist\n"
    "\n"
    "better_response MUSS dem Profil + Beispielen folgen — kurz, trocken, kein "
    "Servicebot-Speak, max 2 Saetze.\n"
    "- Bei score < 8: better_response ist PFLICHT, konkret wie Moloch antworten "
    "wuerde. NIEMALS leer lassen wenn die Antwort schlecht ist.\n"
    "- Bei score >= 8: better_response darf '' (leer) sein — die Pi-Antwort "
    "war schon gut.\n"
    "\n"
    "Antwort STRENG als JSON: "
    '{"score": 0-10, "critique": "<eine harte Zeile, deutsch>", '
    '"better_response": "<wie haette Moloch antworten sollen>"}. '
    "Kein Prosa-Preamble, kein Markdown, NUR das JSON."
)

CRITIC_SYSTEM_SITUATION = (
    "Du bist Markus' Spielleiter. Auf Basis eines vergangenen Erlebnisses "
    "erfindest du eine plausible neue Situation in der Moloch reagieren muesste.\n"
    "\n"
    + _CHARAKTER_PROFIL + "\n"
    "\n"
    "Die Situation soll so sein, dass Moloch eine kurze, charakterstarke Antwort "
    "geben kann — KEIN Smalltalk, KEIN Service-Szenario, KEIN abstraktes "
    "Spielfeld-Geschehen. Gute Situations-Typen: Markus testet ihn, ein Fremder "
    "taucht auf, Markus ist still und braucht Raum, leiser Konflikt im Raum, "
    "Markus stellt ihm eine direkte Frage.\n"
    "\n"
    "Vermeide: Sport-/Spielszenen, Roboter-Wartung, Tiefgarage-Hund-Geschichten, "
    "alles was nach generischer Erzaehl-Vorlage klingt.\n"
    "\n"
    "Eine kurze Szene, max 2 Saetze, deutsch. Kein Auflisten, keine Frage am Ende. "
    "Antwort STRENG als JSON: {\"situation_text\": \"<die Szene>\"}. "
    "Kein Prosa, NUR das JSON."
)


def _load_critic_config() -> Dict[str, Any]:
    """Liest critic_service Block aus settings.json (mit Defaults)."""
    cfg = {
        "enabled": True,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "model": DEFAULT_MODEL,
        "timeout_sec": DEFAULT_TIMEOUT_S,
        "backoff_sec": DEFAULT_BACKOFF_S,
    }
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        block = data.get("critic_service", {}) or {}
        for k in cfg:
            if k in block:
                cfg[k] = block[k]
    except Exception as e:
        logger.debug(f"[CRITIC] settings.json fallback: {e}")
    return cfg


def _extract_json(text: str) -> Optional[Dict]:
    """Robuster JSON-Parser: zieht {...} aus LLM-Output, auch wenn Prosa drumherum."""
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # trailing commas wegmachen
        clean = re.sub(r",\s*([\}\]])", r"\1", raw)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return None


def _format_character_state(character_state: Optional[Dict]) -> str:
    """Compact 1-Block Charakter-Beschreibung fuer Critic-Prompt."""
    if not character_state:
        return "(kein Charakter-State uebergeben)"
    lines = []
    rd = character_state.get("rolling_drift") or {}
    if rd:
        lines.append(
            f"Drift 30d: mood={rd.get('mood_baseline', 0):+.2f} "
            f"energy={rd.get('energy_baseline', 0):+.2f} "
            f"dominance={rd.get('dominance_baseline', 0):+.2f}"
        )
    rules = character_state.get("active_rules") or []
    if rules:
        lines.append("Verhaltensregeln:")
        for r in rules[:5]:
            t = (r.get("trigger") or "")[:60]
            b = (r.get("behavior") or "")[:80]
            lines.append(f"  - Wenn {t} → {b}")
    return "\n".join(lines) if lines else "(neutral)"


# =============================================================================
# CriticClient
# =============================================================================

class CriticClient:
    """Pi-seitiger Client zum PC-Ollama als Critic."""

    def __init__(self):
        self._lock = threading.Lock()
        self._http = requests.Session()
        self._fail_count = 0
        self._backoff_until = 0.0
        self._last_health_check = 0.0
        self._last_health_ok = False
        self._cfg = _load_critic_config()
        logger.info(
            f"[CRITIC] Init host={self._cfg['host']}:{self._cfg['port']} "
            f"model={self._cfg['model']} enabled={self._cfg['enabled']}"
        )

    # --------------------------------------------------------------- HEALTH

    def health(self, force: bool = False) -> bool:
        """Probe PC-Ollama (max alle 30s wenn force=False)."""
        if not self._cfg.get("enabled", True):
            return False
        now = time.monotonic()
        if not force and (now - self._last_health_check) < 30.0:
            return self._last_health_ok
        if now < self._backoff_until:
            return False
        try:
            url = f"http://{self._cfg['host']}:{self._cfg['port']}/api/tags"
            r = self._http.get(url, timeout=3)
            r.raise_for_status()
            tags = r.json().get("models") or []
            names = [t.get("name", "") for t in tags]
            self._last_health_check = now
            self._last_health_ok = self._cfg["model"] in names
            if not self._last_health_ok:
                logger.warning(
                    f"[CRITIC] Modell {self._cfg['model']} nicht installiert. "
                    f"Verfuegbar: {names}"
                )
            return self._last_health_ok
        except Exception as e:
            logger.debug(f"[CRITIC] health probe fehlgeschlagen: {e}")
            self._last_health_check = now
            self._last_health_ok = False
            return False

    # --------------------------------------------------------------- CALL

    def _ollama_generate(self, system: str, user: str,
                         max_tokens: int = 512, temperature: float = 0.5) -> Optional[str]:
        """Generischer Ollama-Call mit Circuit-Breaker."""
        if time.monotonic() < self._backoff_until:
            return None
        if not self._cfg.get("enabled", True):
            return None

        url = f"http://{self._cfg['host']}:{self._cfg['port']}/api/generate"
        payload = {
            "model": self._cfg["model"],
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        resp = None
        try:
            resp = self._http.post(url, json=payload, timeout=self._cfg["timeout_sec"])
            resp.raise_for_status()
            text = (resp.json() or {}).get("response", "")
            self._fail_count = 0
            self._backoff_until = 0.0
            return text
        except Exception as e:
            with self._lock:
                self._fail_count += 1
                if self._fail_count >= 3:
                    self._backoff_until = time.monotonic() + self._cfg["backoff_sec"]
                    logger.warning(
                        f"[CRITIC] {self._fail_count}x Fehler → "
                        f"{self._cfg['backoff_sec']}s Backoff aktiv"
                    )
            logger.debug(f"[CRITIC] Ollama-Call Fehler: {e}")
            return None
        finally:
            if resp is not None:
                resp.close()

    # -------------------------------------------------------- PUBLIC API

    def evaluate(
        self,
        situation: str,
        pi_response: str,
        character_state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Bewerte Pi-Antwort gegen Charakter-State.

        Returns dict mit keys: score, critique, better_response, provider.
        Bei Fehler: safe-default-dict mit score=-1.
        """
        char_block = _format_character_state(character_state)
        user = (
            f"=== CHARAKTER ===\n{char_block}\n\n"
            f"=== SITUATION ===\n{situation[:500]}\n\n"
            f"=== PI-ANTWORT ===\n{pi_response[:500]}\n\n"
            "Bewerte. NUR JSON."
        )
        text = self._ollama_generate(
            system=CRITIC_SYSTEM_EVAL, user=user,
            max_tokens=512, temperature=0.4,
        )
        if not text:
            return {
                "score": -1, "critique": "(critic offline)",
                "better_response": "", "provider": "offline",
            }
        parsed = _extract_json(text)
        if not parsed or "score" not in parsed:
            logger.warning(f"[CRITIC] evaluate: JSON unparsbar — {text[:120]}")
            return {
                "score": -1, "critique": f"(parse fail: {text[:80]})",
                "better_response": "", "provider": self._cfg["model"],
            }
        return {
            "score": int(parsed.get("score", -1)),
            "critique": str(parsed.get("critique", ""))[:300],
            "better_response": str(parsed.get("better_response", ""))[:500],
            "provider": self._cfg["model"],
        }

    def generate_situation(
        self,
        seed_event: Optional[Dict] = None,
        character_state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Erfinde eine neue Markus-Situation auf Basis eines Seed-Erlebnisses.

        seed_event: ein Eintrag aus character_journal (type, interpretation, ...).
        Returns dict: situation_text, provider.
        """
        char_block = _format_character_state(character_state)
        if seed_event:
            seed_str = (
                f"Vergangenes Erlebnis: type={seed_event.get('type', '?')}, "
                f"'{(seed_event.get('interpretation') or '')[:100]}', "
                f"tension_delta={seed_event.get('tension_delta', 0)}"
            )
        else:
            seed_str = "Kein konkretes Seed — erfinde eine alltaegliche Situation."

        user = (
            f"=== CHARAKTER ===\n{char_block}\n\n"
            f"=== SEED ===\n{seed_str}\n\n"
            "Erfinde eine plausible neue Situation. NUR JSON."
        )
        text = self._ollama_generate(
            system=CRITIC_SYSTEM_SITUATION, user=user,
            max_tokens=300, temperature=0.85,
        )
        if not text:
            return {"situation_text": "", "provider": "offline"}
        parsed = _extract_json(text)
        if not parsed or "situation_text" not in parsed:
            logger.warning(f"[CRITIC] generate_situation: JSON unparsbar — {text[:120]}")
            # Fallback: nimm Rohtext (oft brauchbar als Situation)
            return {"situation_text": text.strip()[:300], "provider": self._cfg["model"] + "_raw"}
        return {
            "situation_text": str(parsed.get("situation_text", ""))[:400],
            "provider": self._cfg["model"],
        }

    def get_state(self) -> Dict[str, Any]:
        """Status fuer IPC/Panel."""
        return {
            "host": self._cfg["host"],
            "port": self._cfg["port"],
            "model": self._cfg["model"],
            "enabled": self._cfg.get("enabled", True),
            "fail_count": self._fail_count,
            "backoff_remaining_s": max(0.0, self._backoff_until - time.monotonic()),
            "last_health_ok": self._last_health_ok,
        }


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[CriticClient] = None
_instance_lock = threading.Lock()


def get_critic_client() -> CriticClient:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CriticClient()
    return _instance


# =============================================================================
# Self-Test — `python3 -m core.bridge.critic_client`
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    c = get_critic_client()

    print(f"\n[State]  {c.get_state()}")

    # Health
    print(f"\n[Health] {c.health(force=True)}")
    if not c.health(force=False):
        print(f"  PC-Ollama nicht erreichbar oder Modell nicht installiert.")
        print(f"  Auf PC: ollama pull {c._cfg['model']}")
        print(f"  Self-Test endet hier (offline).")
        raise SystemExit(0)

    # Mock character state
    char_state = {
        "rolling_drift": {"mood_baseline": -0.05, "energy_baseline": 0.02, "dominance_baseline": 0.1},
        "active_rules": [
            {"trigger": "Beleidigung detektiert", "behavior": "ein trockener Satz, kein Kommentar"},
        ],
    }

    # Generate situation
    print(f"\n[Generate Situation]")
    seed = {"type": "tension", "interpretation": "Beleidigung erkannt", "tension_delta": 0.31}
    sit = c.generate_situation(seed_event=seed, character_state=char_state)
    print(f"  Provider: {sit['provider']}")
    print(f"  Situation: {sit['situation_text']}")

    # Evaluate
    print(f"\n[Evaluate]")
    pi_response = "Hallo Markus, schoen dich zu sehen!"
    ev = c.evaluate(
        situation=sit["situation_text"] or "Markus kommt rein nach 4h Abwesenheit",
        pi_response=pi_response,
        character_state=char_state,
    )
    print(f"  Provider: {ev['provider']}")
    print(f"  Score:    {ev['score']}/10")
    print(f"  Kritik:   {ev['critique']}")
    print(f"  Besser:   {ev['better_response']}")

    print(f"\nSelf-Test PASS")
