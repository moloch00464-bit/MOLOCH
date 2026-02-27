#!/usr/bin/env python3
"""
M.O.L.O.C.H. PTZ Arbiter — Drei-Modus Kamerasteuerung.

Verhindert Konflikte zwischen Sonoff Smart Tracking und MOLOCH PTZ.

Modi:
  1. KAMERA_FUEHRT   — Smart Tracking AN, MOLOCH beobachtet nur (Default)
  2. MOLOCH_KORRIGIERT — ST bleibt AN, MOLOCH sendet max 1 Korrektur / 5s
  3. MOLOCH_UEBERNIMMT — ST AUS, MOLOCH steuert via ONVIF

Uebergangslogik:
  - 200ms Pause zwischen Modus-Wechseln
  - Modus 3 > 10s ohne Grund -> auto zurueck zu Modus 1

Status wird in /dev/shm/moloch_status.json exportiert.
"""

import time
import logging
import threading
from enum import Enum

logger = logging.getLogger("PTZArbiter")


class ArbiterMode(Enum):
    KAMERA_FUEHRT = "kamera_fuehrt"
    MOLOCH_KORRIGIERT = "moloch_korrigiert"
    MOLOCH_UEBERNIMMT = "moloch_uebernimmt"


class PTZArbiter:
    """Drei-Modus PTZ Arbiter. Thread-safe."""

    # Mindestzeit zwischen Modus-Wechseln
    SWITCH_COOLDOWN_SEC = 0.2

    # Max 1 Korrektur pro Intervall in Modus 2
    CORRECTION_INTERVAL_SEC = 5.0

    # Modus 3 Timeout: zurueck zu Modus 1 wenn kein Grund mehr besteht
    TAKEOVER_TIMEOUT_SEC = 10.0

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = ArbiterMode.KAMERA_FUEHRT
        self._mode_since = time.time()
        self._last_switch_time = 0.0
        self._switch_reason = "init"

        # Modus 2: letzte Korrektur-Zeit
        self._last_correction_time = 0.0

        # Modus 3: Grund-Tracking (wann zuletzt ein Grund vorlag)
        self._last_takeover_reason_time = 0.0

        # Externe Flags (von camera_manager gesetzt)
        self._smart_tracking_on = True
        self._moloch_tracking_on = False

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
        with self._lock:
            return self._smart_tracking_on

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

    def set_kamera_fuehrt(self, reason: str = "default"):
        """Modus 1: Kamera fuehrt, MOLOCH beobachtet."""
        with self._lock:
            self._smart_tracking_on = True
            self._moloch_tracking_on = False
            return self._switch_mode(ArbiterMode.KAMERA_FUEHRT, reason)

    def set_moloch_korrigiert(self, reason: str = "head_off_center"):
        """Modus 2: ST bleibt an, MOLOCH darf 1x korrigieren."""
        with self._lock:
            self._smart_tracking_on = True
            self._moloch_tracking_on = True
            return self._switch_mode(ArbiterMode.MOLOCH_KORRIGIERT, reason)

    def set_moloch_uebernimmt(self, reason: str = "person_detected"):
        """Modus 3: ST aus, MOLOCH steuert."""
        with self._lock:
            self._smart_tracking_on = False
            self._moloch_tracking_on = True
            self._last_takeover_reason_time = time.time()
            return self._switch_mode(ArbiterMode.MOLOCH_UEBERNIMMT, reason)

    # =========================================================================
    # PTZ-Befehl erlaubt?
    # =========================================================================

    def may_send_ptz(self) -> bool:
        """Darf MOLOCH jetzt einen PTZ-Befehl senden?

        Returns True wenn:
          - Modus 3 (UEBERNIMMT): immer
          - Modus 2 (KORRIGIERT): nur wenn letzte Korrektur > 5s her
          - Modus 1 (KAMERA_FUEHRT): nie
        """
        with self._lock:
            if self._mode == ArbiterMode.MOLOCH_UEBERNIMMT:
                return True
            if self._mode == ArbiterMode.MOLOCH_KORRIGIERT:
                now = time.time()
                if now - self._last_correction_time >= self.CORRECTION_INTERVAL_SEC:
                    return True
                return False
            return False

    def record_correction(self):
        """Modus 2: Korrektur wurde gesendet — Zeitstempel aktualisieren."""
        with self._lock:
            self._last_correction_time = time.time()
            # Nach Korrektur zurueck zu Modus 1
            self._switch_mode(ArbiterMode.KAMERA_FUEHRT, "korrektur_gesendet")

    def record_takeover_reason(self):
        """Modus 3: Es gibt noch einen Grund fuer Takeover (z.B. Person sichtbar)."""
        with self._lock:
            self._last_takeover_reason_time = time.time()

    # =========================================================================
    # Timeout-Check (sollte periodisch aufgerufen werden)
    # =========================================================================

    def check_timeout(self):
        """Modus 3 Timeout: zurueck zu Modus 1 wenn kein Grund seit 10s."""
        with self._lock:
            if self._mode != ArbiterMode.MOLOCH_UEBERNIMMT:
                return
            now = time.time()
            elapsed = now - self._last_takeover_reason_time
            if elapsed > self.TAKEOVER_TIMEOUT_SEC:
                self._smart_tracking_on = True
                self._moloch_tracking_on = False
                self._switch_mode(ArbiterMode.KAMERA_FUEHRT, f"timeout_{elapsed:.0f}s")

    # =========================================================================
    # Sync mit externem State
    # =========================================================================

    def sync_smart_tracking(self, on: bool):
        """Von camera_manager aufgerufen wenn ST-State sich aendert."""
        with self._lock:
            self._smart_tracking_on = on

    # =========================================================================
    # Status-Export (fuer /dev/shm/moloch_status.json)
    # =========================================================================

    def get_status(self) -> dict:
        """Status-Dict fuer SHM Export."""
        with self._lock:
            return {
                "ptz_arbiter_mode": self._mode.value,
                "cam_smart_tracking": self._smart_tracking_on,
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
