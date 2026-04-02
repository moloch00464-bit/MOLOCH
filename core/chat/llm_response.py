#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. LLM Response — Personality-aware Chat
====================================================

Baut den vollständigen System-Prompt basierend auf:
  - Aktueller Persönlichkeit (Guardian / Shadow / Berserker)
  - Vision-Kontext (wen/was sieht Moloch gerade)
  - Innerem Zustand (Tension, Dominance, Zone)

Ruft LocalLLMBridge.ask_external() auf und gibt die Antwort zurück.

Singleton: get_llm_response()
"""

import json
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger("LLMResponse")

# Pfade
_MOLOCH_DIR = os.path.expanduser("~/moloch")
_TEMPLATES_PATH = os.path.join(_MOLOCH_DIR, "core", "chat", "prompt_templates.json")
_SETTINGS_PATH = os.path.join(_MOLOCH_DIR, "config", "llm_settings.json")
_STATUS_PATH = os.path.join(_MOLOCH_DIR, "moloch_status.json")


def _load_json(path: str) -> Dict:
    """JSON-Datei laden, leeres Dict bei Fehler."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"[LLM-RESPONSE] JSON laden fehlgeschlagen ({path}): {e}")
        return {}


def _get_personality_mode() -> str:
    """Aktuellen Persönlichkeitsmodus als String ('guardian'/'shadow'/'berserker')."""
    try:
        from core.personality.personality_engine import get_personality_engine
        pe = get_personality_engine()
        return pe.mode.value
    except Exception:
        return "guardian"


def _get_inner_state() -> Dict:
    """Tension, Dominance und Zone aus CoreIntegrator."""
    try:
        from core.core_integrator import get_core_integrator
        ci = get_core_integrator()
        return {
            "tension": ci.get_tension(),
            "dominance": ci.get_dominance(),
            "zone": ci.get_personality_zone(),
        }
    except Exception:
        return {"tension": 0.0, "dominance": 0.5, "zone": "guardian"}


def _get_vision_context() -> Dict:
    """Aktuelle Detektionen aus moloch_status.json."""
    status = _load_json(_STATUS_PATH)
    detections = status.get("panel_detections", {})
    return {
        "person_detected": detections.get("person_detected", False),
        "face_detected": detections.get("face_detected", False),
        "face_id": detections.get("face_id"),
        "similarity": detections.get("similarity", 0.0),
        "person_count": detections.get("person_count", 0),
    }


def _build_vision_text(vision: Dict, templates: Dict) -> str:
    """Vision-Kontext als Text für den System-Prompt formatieren."""
    vt = templates.get("vision_context", {})
    lines = [vt.get("prefix", "\n--- WAS ICH SEHE ---\n")]

    count = vision.get("person_count", 0)
    face_id = vision.get("face_id")
    sim = vision.get("similarity", 0.0)

    if count > 1:
        lines.append(vt.get("multiple", "Ich sehe {count} Personen.").format(count=count))
    elif vision.get("face_detected"):
        if face_id and face_id != "unknown" and sim >= 0.5:
            tpl = vt.get("person_known", "Ich erkenne {name} (Ähnlichkeit: {similarity:.0%}).")
            lines.append(tpl.format(name=face_id, similarity=sim))
        elif vision.get("person_detected"):
            lines.append(vt.get("person_unknown", "Eine unbekannte Person ist im Bild."))
        else:
            lines.append(vt.get("face_only", "Ich sehe ein Gesicht."))
    elif vision.get("person_detected"):
        lines.append(vt.get("person_unknown", "Eine unbekannte Person ist im Bild."))
    else:
        lines.append(vt.get("no_person", "Niemand ist im Bild."))

    lines.append(vt.get("suffix", "\n"))
    return "".join(lines)


def _build_inner_state_text(state: Dict, templates: Dict) -> str:
    """Inneren Zustand als Text für den System-Prompt formatieren."""
    it = templates.get("inner_state", {})
    prefix = it.get("prefix", "\n--- INNERER ZUSTAND ---\n")
    tpl = it.get("template", "Zone: {zone} | Tension: {tension:.2f} | Dominance: {dominance:+.2f}\n")
    return prefix + tpl.format(**state)


def build_system_prompt(mode: str, vision: Dict, state: Dict,
                        templates: Dict, settings: Dict) -> str:
    """Vollständigen System-Prompt zusammenbauen."""
    base = templates.get("system_prompts", {}).get(mode, "")

    features = settings.get("features", {})
    parts = [base]

    if features.get("vision_context_in_prompt", True):
        parts.append(_build_vision_text(vision, templates))

    if features.get("inner_state_in_prompt", True):
        parts.append(_build_inner_state_text(state, templates))

    return "".join(parts)


def ask(user_text: str, max_tokens: Optional[int] = None) -> Optional[str]:
    """Hauptfunktion: Antwort vom LLM holen, personality-aware.

    Args:
        user_text: Die Eingabe des Benutzers.
        max_tokens: Überschreibt den Wert aus llm_settings.json.

    Returns:
        Antwort-Text oder None wenn LLM nicht verfügbar.
    """
    templates = _load_json(_TEMPLATES_PATH)
    settings = _load_json(_SETTINGS_PATH)

    mode = _get_personality_mode()
    vision = _get_vision_context()
    state = _get_inner_state()

    system = build_system_prompt(mode, vision, state, templates, settings)

    chat_cfg = settings.get("chat", {})
    tokens = max_tokens or chat_cfg.get("max_tokens", 256)

    logger.info(f"[LLM-RESPONSE] mode={mode} tension={state['tension']:.2f} tokens={tokens}")

    try:
        from core.autonomy.local_llm_bridge import get_llm_bridge
        bridge = get_llm_bridge()
        answer = bridge.ask_external(user_text, system=system, max_tokens=tokens)
        if answer:
            logger.info(f"[LLM-RESPONSE] Antwort: {len(answer)} Zeichen via {bridge._last_provider}")
        return answer
    except Exception as e:
        logger.error(f"[LLM-RESPONSE] Fehler: {e}")
        return None


# Singleton-Wrapper für konsistente Nutzung
class LLMResponse:
    """Thin wrapper um ask() für Dependency-Injection."""

    def ask(self, user_text: str, max_tokens: Optional[int] = None) -> Optional[str]:
        return ask(user_text, max_tokens)

    def get_current_mode(self) -> str:
        return _get_personality_mode()

    def get_vision_context(self) -> Dict:
        return _get_vision_context()

    def get_inner_state(self) -> Dict:
        return _get_inner_state()


_instance: Optional[LLMResponse] = None


def get_llm_response() -> LLMResponse:
    global _instance
    if _instance is None:
        _instance = LLMResponse()
    return _instance
