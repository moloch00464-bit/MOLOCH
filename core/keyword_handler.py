#!/usr/bin/env python3
"""
M.O.L.O.C.H. Lokale Keyword-Erkennung
========================================

Prueft eingehende Nachrichten BEVOR sie an die Claude API gehen.
Einfache Keywords werden lokal behandelt — spart Tokens und Latenz.

Keyword-Liste in config/keywords.json (einfach erweiterbar).
Matching: Substring, case-insensitive.

Aktionen:
  - owner_confirm: CoreIntegrator.owner_override() → tension -0.3, dominance +0.3
  - calm_down:     CoreIntegrator.calm_down() → tension -0.3
  - light_on/off:  Flutlicht via IPC Command
  - alarm_on/off:  Alarm via IPC Command
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("KeywordHandler")

_CONFIG_PATH = Path.home() / "moloch" / "config" / "keywords.json"


class KeywordHandler:
    """Lokale Keyword-Erkennung fuer M.O.L.O.C.H."""

    def __init__(self):
        self._categories: List[Dict] = []
        self._load_keywords()

    def _load_keywords(self):
        """Keywords aus config/keywords.json laden."""
        try:
            if _CONFIG_PATH.exists():
                with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._categories = data.get("categories", [])
                total = sum(len(c.get("keywords", [])) for c in self._categories)
                logger.info(f"[KEYWORD] {len(self._categories)} Kategorien, {total} Keywords geladen")
            else:
                logger.warning(f"[KEYWORD] {_CONFIG_PATH} nicht gefunden")
        except Exception as e:
            logger.error(f"[KEYWORD] Laden fehlgeschlagen: {e}")

    def reload(self):
        """Keywords neu laden (z.B. nach Config-Aenderung)."""
        self._load_keywords()

    def check(self, text: str) -> Optional[Tuple[str, str]]:
        """Text gegen Keywords pruefen.

        Args:
            text: Eingabetext (wird lowercase fuer Matching)

        Returns:
            (action_id, response_text) wenn Match, None wenn kein Match.
            Erste Kategorie die matcht gewinnt.
        """
        if not text or not self._categories:
            return None

        lower = text.lower().strip()
        if len(lower) < 3:
            return None

        for cat in self._categories:
            keywords = cat.get("keywords", [])
            for kw in keywords:
                if kw in lower:
                    action = cat.get("action", "")
                    response = cat.get("response", "OK.")
                    logger.info(
                        f"[KEYWORD] Match: '{kw}' -> {action} "
                        f"(text='{text[:60]}')"
                    )
                    return (action, response)

        return None

    def execute(self, text: str) -> Optional[str]:
        """Text pruefen UND Aktion ausfuehren.

        Returns:
            Antwort-String wenn lokal behandelt, None wenn an API weiterleiten.
        """
        result = self.check(text)
        if result is None:
            return None

        action, response = result

        if action == "owner_confirm":
            return self._action_owner_confirm(response)
        elif action == "calm_down":
            return self._action_calm_down(response)
        elif action == "light_on":
            return self._action_light(True, response)
        elif action == "light_off":
            return self._action_light(False, response)
        elif action == "alarm_on":
            return self._action_alarm(True, response)
        elif action == "alarm_off":
            return self._action_alarm(False, response)
        elif action == "enrollment_start":
            return self._action_enrollment(text, response)
        elif action == "diagnostics":
            return self._action_diagnostics()
        else:
            logger.warning(f"[KEYWORD] Unbekannte Aktion: {action}")
            return None

    # =========================================================================
    # Aktionen
    # =========================================================================

    def _action_owner_confirm(self, response: str) -> str:
        """Owner bestaetigt sich — CoreIntegrator + ArbitrationEngine updaten."""
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            ci.owner_override()
            logger.info("[KEYWORD] Owner-Override ausgefuehrt")
        except Exception as e:
            logger.error(f"[KEYWORD] Owner-Override fehlgeschlagen: {e}")
        # ArbitrationEngine: User Override + Identity Confirmed
        try:
            from core.arbitration import get_arbitration
            arbi = get_arbitration()
            arbi.user_override("guardian")
            arbi.identity_confirmed()
            logger.info("[KEYWORD] ArbitrationEngine: user_override + identity_confirmed")
        except Exception as e:
            logger.error(f"[KEYWORD] ArbitrationEngine fehlgeschlagen: {e}")
        return response

    def _action_calm_down(self, response: str) -> str:
        """Beruhigung — Tension senken + ArbitrationEngine Guardian Override."""
        try:
            from core.core_integrator import get_core_integrator
            ci = get_core_integrator()
            ci.calm_down()
            logger.info("[KEYWORD] Calm-Down ausgefuehrt")
        except Exception as e:
            logger.error(f"[KEYWORD] Calm-Down fehlgeschlagen: {e}")
        # ArbitrationEngine: Guardian Override 20s + 5s Fade
        try:
            from core.arbitration import get_arbitration
            get_arbitration().user_override("guardian")
            logger.info("[KEYWORD] ArbitrationEngine: user_override(guardian)")
        except Exception as e:
            logger.error(f"[KEYWORD] ArbitrationEngine fehlgeschlagen: {e}")
        return response

    def _action_light(self, on: bool, response: str) -> str:
        """Flutlicht steuern via IPC Command."""
        # level 2 = Farb-Nachtsicht (weisse LEDs AN), 0 = IR-only (AUS)
        level = 2 if on else 0
        self._send_ipc_command("cloud_led", level=level)
        return response

    def _action_alarm(self, on: bool, response: str) -> str:
        """Alarm steuern via IPC Command."""
        self._send_ipc_command("cloud_alarm", on=on)
        return response

    def _action_enrollment(self, text: str, response_template: str) -> str:
        """Face-Enrollment via Chat starten.

        Extrahiert den Namen aus dem Text:
          "merk dir das ist Peter" → name=peter
          "das ist Hans" → name=hans
          "enrollment markus" → name=markus
          "gesicht merken" → name=unbekannt (Fallback)
        """
        name = self._extract_enrollment_name(text)
        self._send_ipc_command("enrollment_start", name=name, n=20)
        response = response_template.replace("{name}", name.capitalize())
        logger.info(f"[KEYWORD] Enrollment gestartet fuer '{name}'")
        return response

    def _extract_enrollment_name(self, text: str) -> str:
        """Name aus Enrollment-Befehl extrahieren."""
        lower = text.lower().strip()

        # Pattern: "merk dir das ist <NAME>"
        m = re.search(r"merk dir das ist\s+(\w+)", lower)
        if m:
            return m.group(1)

        # Pattern: "das ist <NAME>" (aber nicht "das ist gut/ok/etc")
        m = re.search(r"das ist\s+(\w+)", lower)
        if m and m.group(1) not in ("gut", "ok", "okay", "toll", "super", "richtig", "falsch", "egal"):
            return m.group(1)

        # Pattern: "enrollment <NAME>" / "enroll <NAME>"
        m = re.search(r"enroll(?:ment)?\s+(\w+)", lower)
        if m:
            return m.group(1)

        # Pattern: "einpraegen <NAME>"
        m = re.search(r"einpr[aä]gen\s+(\w+)", lower)
        if m:
            return m.group(1)

        return "unbekannt"

    def _action_diagnostics(self) -> str:
        """Selbstdiagnose ausfuehren und Ergebnis als Text zurueckgeben."""
        try:
            from core.diagnostics import get_diagnostics_text
            return get_diagnostics_text()
        except Exception as e:
            logger.error(f"[KEYWORD] Diagnostics fehlgeschlagen: {e}")
            return "Diagnose konnte nicht ausgefuehrt werden."

    # =========================================================================
    # IPC Helper
    # =========================================================================

    def _send_ipc_command(self, action: str, **kwargs):
        """Command an MolochService senden via IPC (file-based).

        Schreibt /tmp/moloch_cmd_<timestamp>.json, Service pollt alle 200ms.
        """
        cmd = {"action": action}
        cmd.update(kwargs)
        path = f"/tmp/moloch_cmd_{int(time.time() * 1000)}.json"
        try:
            with open(path, "w") as f:
                json.dump(cmd, f)
            logger.info(f"[KEYWORD] IPC Command: {action} -> {path}")
        except Exception as e:
            logger.error(f"[KEYWORD] IPC Command fehlgeschlagen: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[KeywordHandler] = None


def get_keyword_handler() -> KeywordHandler:
    """Singleton-Zugriff auf den KeywordHandler."""
    global _instance
    if _instance is None:
        _instance = KeywordHandler()
    return _instance
