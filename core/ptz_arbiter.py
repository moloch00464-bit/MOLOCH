#!/usr/bin/env python3
"""
M.O.L.O.C.H. PTZ Arbiter — Zwei-Modus Kamerasteuerung.

Gate 0 Phase 2: Smart Tracking KOMPLETT AUS. Moloch steuert ALLES.

Modi:
  1. MOLOCH_AUTONOM  — Moloch steuert Kamera (Default)
  2. MOLOCH_MANUELL  — User steuert per GUI, 30s Timeout → zurueck zu AUTONOM

Alte Modi entfernt:
  - KAMERA_FUEHRT (Smart Tracking) — existiert nicht mehr
  - MOLOCH_KORRIGIERT — nicht mehr noetig ohne ST

Status wird in /dev/shm/moloch_status.json exportiert.
"""

import time
import logging
import threading
from enum import Enum

logger = logging.getLogger("PTZArbiter")


class ArbiterMode(Enum):
    MOLOCH_AUTONOM = "moloch_autonom"
    MOLOCH_MANUELL = "moloch_manuell"
    # Legacy-Kompatibilitaet (fuer alte Status-Reads)
    KAMERA_FUEHRT = "kamera_fuehrt"
    MOLOCH_KORRIGIERT = "moloch_korrigiert"
    MOLOCH_UEBERNIMMT = "moloch_uebernimmt"


class PTZArbiter:
    """Zwei-Modus PTZ Arbiter. Thread-safe. Gate 0."""

    # Mindestzeit zwischen Modus-Wechseln
    SWITCH_COOLDOWN_SEC = 0.2

    # Manuell-Timeout: nach 30s zurueck zu AUTONOM
    MANUAL_TIMEOUT_SEC = 30.0

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = ArbiterMode.MOLOCH_AUTONOM
        self._mode_since = time.time()
        self._last_switch_time = 0.0
        self._switch_reason = "init"

        # Manuell: wann zuletzt manuell gesteuert
        self._last_manual_time = 0.0

        # Smart Tracking ist PERMANENT AUS (Gate 0)
        self._smart_tracking_on = False
        self._moloch_tracking_on = True

        # G1-T03: Callback bei Auto-Resume (Manuell -> Autonom)
        self.on_auto_resume = None  # Callable[[], None]

    # =========================================================================
    # Properties (Thread-safe)
    # =========================================================================

    @property
    def mode(self) -> ArbiterMode:
        with self._lock:
            return self._mode

    @property
    def mode_name(self) -> str:
        with self._lock:
            return self._mode.value

    @property
    def smart_tracking_on(self) -> bool:
        # Gate 0: IMMER False
        return False

    @property
    def moloch_tracking_on(self) -> bool:
        with self._lock:
            return self._moloch_tracking_on

    # =========================================================================
    # Modus-Wechsel
    # =========================================================================

    def _switch_mode(self, new_mode: ArbiterMode, reason: str):
        """Interner Modus-Wechsel (muss unter Lock aufgerufen werden)."""
        now = time.time()
        if now - self._last_switch_time < self.SWITCH_COOLDOWN_SEC:
            return False
        old = self._mode
        if old == new_mode:
            return False
        self._mode = new_mode
        self._mode_since = now
        self._last_switch_time = now
        self._switch_reason = reason
        logger.info(f"[ARBITER] {old.value} -> {new_mode.value} grund={reason}")
        return True

    def set_moloch_autonom(self, reason: str = "default"):
        """Moloch steuert Kamera autonom."""
        with self._lock:
            self._smart_tracking_on = False
            self._moloch_tracking_on = True
            return self._switch_mode(ArbiterMode.MOLOCH_AUTONOM, reason)

    def set_moloch_manuell(self, reason: str = "gui_steuerung"):
        """User steuert manuell. 30s Timeout → zurueck zu AUTONOM."""
        with self._lock:
            self._smart_tracking_on = False
            self._moloch_tracking_on = False
            self._last_manual_time = time.time()
            return self._switch_mode(ArbiterMode.MOLOCH_MANUELL, reason)

    # Legacy-Kompatibilitaet (andere Module rufen diese noch auf)
    def set_kamera_fuehrt(self, reason: str = "default"):
        """Legacy → wird zu MOLOCH_AUTONOM."""
        return self.set_moloch_autonom(f"legacy_kamera_fuehrt_{reason}")

    def set_moloch_korrigiert(self, reason: str = "head_off_center"):
        """Legacy → wird zu MOLOCH_AUTONOM."""
        return self.set_moloch_autonom(f"legacy_korrigiert_{reason}")

    def set_moloch_uebernimmt(self, reason: str = "person_detected"):
        """Legacy → wird zu MOLOCH_AUTONOM."""
        return self.set_moloch_autonom(f"legacy_uebernimmt_{reason}")

    # =========================================================================
    # PTZ-Befehl erlaubt?
    # =========================================================================

    def may_send_ptz(self) -> bool:
        """Darf MOLOCH jetzt einen PTZ-Befehl senden?

        Gate 0: JA wenn AUTONOM, NEIN wenn MANUELL.
        """
        with self._lock:
            return self._mode == ArbiterMode.MOLOCH_AUTONOM

    def record_correction(self):
        """Legacy-Kompatibilitaet. Noop in Gate 0."""
        pass

    def record_takeover_reason(self):
        """Legacy-Kompatibilitaet. Noop in Gate 0."""
        pass

    def record_manual_activity(self):
        """User hat manuell gesteuert — Timeout reset."""
        with self._lock:
            self._last_manual_time = time.time()

    # =========================================================================
    # Timeout-Check (sollte periodisch aufgerufen werden)
    # =========================================================================

    def check_timeout(self):
        """Manuell-Timeout: zurueck zu AUTONOM nach 30s. G1-T03: mit Callback."""
        with self._lock:
            if self._mode != ArbiterMode.MOLOCH_MANUELL:
                return
            now = time.time()
            elapsed = now - self._last_manual_time
            if elapsed > self.MANUAL_TIMEOUT_SEC:
                self._moloch_tracking_on = True
                self._switch_mode(ArbiterMode.MOLOCH_AUTONOM,
                                  f"manual_timeout_{elapsed:.0f}s")
                # G1-T03: Auto-Resume Callback (TTS Spruch etc.)
                if self.on_auto_resume:
                    try:
                        self.on_auto_resume()
                    except Exception as e:
                        logger.warning(f"[ARBITER] on_auto_resume Fehler: {e}")

    # =========================================================================
    # Sync (Legacy-Kompatibilitaet)
    # =========================================================================

    def sync_smart_tracking(self, on: bool):
        """Legacy. Smart Tracking bleibt AUS in Gate 0."""
        pass

    # =========================================================================
    # Status-Export (fuer /dev/shm/moloch_status.json)
    # =========================================================================

    def get_status(self) -> dict:
        """Status-Dict fuer SHM Export."""
        with self._lock:
            return {
                "ptz_arbiter_mode": self._mode.value,
                "cam_smart_tracking": False,
                "moloch_tracking": self._moloch_tracking_on,
                "ptz_last_switch": time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(self._mode_since)
                ),
                "ptz_switch_reason": self._switch_reason,
            }


# Singleton
_instance = None
_lock = threading.Lock()


def get_ptz_arbiter() -> PTZArbiter:
    """Singleton PTZ Arbiter."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PTZArbiter()
    return _instance
