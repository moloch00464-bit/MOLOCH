#!/usr/bin/env python3
"""
M.O.L.O.C.H. State Arbitration Engine
========================================

Sitzt ZWISCHEN CoreIntegrator und allen Consumern (Avatar, LED, Voice).
Arbitriert Konflikte nach Prioritaet:

  Priority 100: User Command ("beruhig dich", "ich bin's")
  Priority  80: Identity Confirmed (ArcFace erkennt Markus)
  Priority  50: Perception (Kamera sieht Unbekannten) — durchgereicht
  Priority  20: Ambient (Musik, Tageszeit) — durchgereicht

User Commands schlagen ALLES. Perception wird ueberstimmt.
CoreIntegrator berechnet weiter den rohen State — ArbitrationEngine
filtert nur den OUTPUT der an Consumer geht.

Override-Logik:
  - user_override("guardian", 20s) -> Guardian fuer 20 Sekunden
  - Dominance Floor auf +0.2, Shadow gecappt auf max -0.1
  - Nach 20s: 5s Fade zurueck zu Raw State
  - identity_confirmed() -> Shadow gecappt auf -0.2 fuer 10 Minuten

Consumer-Integration:
  - Avatar liest FinalState (via apply())
  - LED liest Zone (via get_zone())
  - Voice liest Zone (via get_zone())
  - NICHT mehr direkt CoreIntegrator fuer Zone-Entscheidungen
"""

import time
import threading
import logging
from typing import Optional, Dict

_logger = logging.getLogger("Arbitration")


class ArbitrationEngine:
    """State Arbitration — entscheidet was Consumer sehen."""

    # === Override-Parameter ===
    USER_OVERRIDE_DURATION = 20.0    # Sekunden
    USER_OVERRIDE_FADE = 5.0         # Sekunden Fade nach Override
    USER_DOMINANCE_FLOOR = 0.2       # Mindest-Dominance waehrend Override
    USER_SHADOW_CAP = -0.1           # Max Shadow waehrend Override (unused, Floor reicht)

    IDENTITY_DURATION = 600.0        # 10 Minuten
    IDENTITY_SHADOW_CAP = -0.2       # Max Shadow wenn Markus bestaetigt

    ZONE_HYSTERESIS = 0.15           # Gleich wie CoreIntegrator

    def __init__(self):
        self._lock = threading.Lock()

        # User Override State
        self._user_active = False
        self._user_zone = "guardian"
        self._user_until = 0.0        # monotonic: Override endet
        self._user_fade_until = 0.0   # monotonic: Fade endet

        # Identity Override State
        self._identity_active = False
        self._identity_until = 0.0

        # Cache des letzten apply()-Ergebnisses (fuer get_zone())
        self._last_zone = "guardian"
        self._last_dominance = 0.5

        _logger.info("[ARBI] ArbitrationEngine initialisiert")

    # =================================================================
    # Public API — Override setzen
    # =================================================================

    def user_override(self, zone: str = "guardian",
                      duration: float = None, fade: float = None):
        """User-Command Override (Prioritaet 100).

        Erzwingt Zone fuer duration Sekunden, dann fade Sekunden Uebergang.
        Schlaegt ALLES — Perception, Identity, Ambient.
        """
        if duration is None:
            duration = self.USER_OVERRIDE_DURATION
        if fade is None:
            fade = self.USER_OVERRIDE_FADE

        now = time.monotonic()
        with self._lock:
            self._user_active = True
            self._user_zone = zone
            self._user_until = now + duration
            self._user_fade_until = now + duration + fade
            # Cache sofort updaten fuer get_zone()
            self._last_zone = zone
            self._last_dominance = self.USER_DOMINANCE_FLOOR

        _logger.info(
            f"[ARBI] User Override: zone={zone} "
            f"duration={duration:.0f}s fade={fade:.0f}s"
        )

    def identity_confirmed(self, duration: float = None):
        """Identity Confirmed Override (Prioritaet 80).

        Cappt Shadow-Einfluss fuer duration Sekunden.
        Typisch: ArcFace erkennt Markus, oder User sagt "ich bin's".
        """
        if duration is None:
            duration = self.IDENTITY_DURATION

        now = time.monotonic()
        with self._lock:
            self._identity_active = True
            self._identity_until = now + duration

        _logger.info(f"[ARBI] Identity Confirmed: duration={duration:.0f}s")

    def clear_user_override(self):
        """User Override sofort beenden."""
        with self._lock:
            self._user_active = False
        _logger.info("[ARBI] User Override geloescht")

    def clear_identity(self):
        """Identity Override sofort beenden."""
        with self._lock:
            self._identity_active = False
        _logger.info("[ARBI] Identity Override geloescht")

    # =================================================================
    # Public API — State lesen
    # =================================================================

    def apply(self, raw_status: dict) -> dict:
        """Raw CoreIntegrator Status-Dict modifizieren.

        Nimmt get_status_dict() Output und wendet Arbitration an.
        Gibt modifiziertes Dict zurueck (Kopie, Original bleibt intakt).

        Muss regelmaessig aufgerufen werden (z.B. in _write_status_json).
        """
        result = dict(raw_status)
        now = time.monotonic()

        with self._lock:
            # Timer pruefen
            if self._user_active and now > self._user_fade_until:
                self._user_active = False
                _logger.info("[ARBI] User Override + Fade abgelaufen")

            if self._identity_active and now > self._identity_until:
                self._identity_active = False
                _logger.info("[ARBI] Identity Override abgelaufen")

            # === Prioritaet 100: User Override ===
            if self._user_active:
                if now < self._user_until:
                    # Voller Override: Zone erzwingen, Dominance Floor
                    result["zone"] = self._user_zone
                    dom = float(result.get("dominance", 0.0))
                    result["dominance"] = round(
                        max(dom, self.USER_DOMINANCE_FLOOR), 4
                    )
                    remaining = self._user_until - now
                    result["override"] = {
                        "active": True,
                        "source": "user_command",
                        "remaining": round(remaining, 1),
                        "zone": self._user_zone,
                    }
                    # Cache updaten
                    self._last_zone = result["zone"]
                    self._last_dominance = result["dominance"]
                    return result

                elif now < self._user_fade_until:
                    # Fade-Phase: Uebergang von Override zu Raw
                    fade_total = self.USER_OVERRIDE_FADE
                    fade_elapsed = now - self._user_until
                    fade_t = min(1.0, fade_elapsed / max(0.01, fade_total))
                    # 0=voller Override, 1=zurueck bei Raw

                    raw_dom = float(result.get("dominance", 0.0))
                    raw_zone = result.get("zone", "guardian")

                    # Dominance: Blend von Floor zu Raw
                    blended_dom = (
                        self.USER_DOMINANCE_FLOOR
                        + (raw_dom - self.USER_DOMINANCE_FLOOR) * fade_t
                    )
                    result["dominance"] = round(blended_dom, 4)

                    # Zone: bleibt Override solange blended_dom > Hysterese
                    if blended_dom > self.ZONE_HYSTERESIS:
                        result["zone"] = self._user_zone
                    else:
                        result["zone"] = raw_zone

                    remaining = self._user_fade_until - now
                    result["override"] = {
                        "active": True,
                        "source": "user_command_fade",
                        "remaining": round(remaining, 1),
                        "fade_progress": round(fade_t, 2),
                    }
                    # Cache updaten
                    self._last_zone = result["zone"]
                    self._last_dominance = result["dominance"]
                    return result

            # === Prioritaet 80: Identity Override ===
            if self._identity_active:
                dom = float(result.get("dominance", 0.0))
                # Shadow gecappt auf max -0.2
                capped_dom = max(dom, self.IDENTITY_SHADOW_CAP)
                result["dominance"] = round(capped_dom, 4)

                # Identity confirmed = Shadow komplett unterdrueckt
                # Markus ist bestaetigt, Perception-Shadow hat keinen Grund
                zone = result.get("zone", "guardian")
                if zone == "shadow":
                    result["zone"] = "guardian"

                remaining = self._identity_until - now
                result["override"] = {
                    "active": True,
                    "source": "identity",
                    "remaining": round(remaining, 1),
                }
                # Cache updaten
                self._last_zone = result["zone"]
                self._last_dominance = result["dominance"]
                return result

        # Kein Override aktiv — Raw durchreichen
        result["override"] = {"active": False, "source": "", "remaining": 0}
        # Cache updaten
        with self._lock:
            self._last_zone = result.get("zone", "guardian")
            self._last_dominance = float(result.get("dominance", 0.5))
        return result

    def get_zone(self) -> str:
        """Aktuelle arbitrierte Zone (cached, fuer LED/Voice).

        Nutzt den Cache vom letzten apply()-Aufruf.
        Zwischen apply()-Aufrufen kann die Zone veraltet sein
        (max ~1s bei 1Hz Status-Schreibrate).
        """
        now = time.monotonic()
        with self._lock:
            # Schnell-Check: User Override noch aktiv?
            if self._user_active:
                if now < self._user_until:
                    return self._user_zone
                elif now > self._user_fade_until:
                    self._user_active = False

            return self._last_zone

    def get_dominance(self) -> float:
        """Aktuelle arbitrierte Dominance (cached)."""
        with self._lock:
            return self._last_dominance

    def is_override_active(self) -> bool:
        """Ob irgendein Override gerade aktiv ist."""
        now = time.monotonic()
        with self._lock:
            if self._user_active and now < self._user_fade_until:
                return True
            if self._identity_active and now < self._identity_until:
                return True
        return False

    def get_override_info(self) -> dict:
        """Aktuellen Override-Status zurueckgeben (fuer Diagnostik)."""
        now = time.monotonic()
        with self._lock:
            if self._user_active:
                if now < self._user_until:
                    return {
                        "active": True,
                        "source": "user_command",
                        "zone": self._user_zone,
                        "remaining": round(self._user_until - now, 1),
                    }
                elif now < self._user_fade_until:
                    return {
                        "active": True,
                        "source": "user_command_fade",
                        "remaining": round(self._user_fade_until - now, 1),
                    }
            if self._identity_active and now < self._identity_until:
                return {
                    "active": True,
                    "source": "identity",
                    "remaining": round(self._identity_until - now, 1),
                }
        return {"active": False, "source": "", "remaining": 0}


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[ArbitrationEngine] = None
_instance_lock = threading.Lock()


def get_arbitration() -> ArbitrationEngine:
    """Singleton-Zugriff auf die ArbitrationEngine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ArbitrationEngine()
    return _instance
