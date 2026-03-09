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
from collections import deque
from enum import Enum
from typing import Optional, Tuple

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

        # === SMART_SEARCH_RETURN (G1-T04) ===
        # Letzte bekannte Zielposition (pan, tilt)
        self._last_known_position: Optional[Tuple[float, float]] = None
        # Zeitpunkt des Ziel-Verlusts
        self._target_lost_time: Optional[float] = None
        # True waehrend 3s Wartezeit nach Rueckkehr zur letzten Position
        self._smart_search_waiting: bool = False
        # True wenn Wartezeit abgelaufen → patrol_scan soll starten
        self.smart_search_patrol_ready: bool = False
        # Wartezeit in Sekunden bevor patrol_scan aktiviert wird
        SMART_SEARCH_WAIT_SEC: float = 3.0
        self._smart_search_wait_sec: float = SMART_SEARCH_WAIT_SEC

        # === FACE_LOCK_MODE (SCRFD-basiert) ===
        # Wie viele aufeinanderfolgende Frames face_confidence > 0.8
        self._face_lock_confidence_count: int = 0
        # Ab N Frames wird face_lock aktiviert
        FACE_LOCK_FRAMES: int = 5
        self._face_lock_frames: int = FACE_LOCK_FRAMES
        # Face-Lock aktiv: PTZ zielt auf face_bbox_center statt body_center
        self.face_lock_active: bool = False
        # Schwellwerte
        self._face_lock_enter_conf: float = 0.8
        self._face_lock_exit_conf: float = 0.5

        # === MULTI_PERSON_SCENE ===
        # Prioritaet: markus > groesste BBox > zuletzt getracktes Target
        self._last_tracked_target_id: Optional[str] = None

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
                pass
            else:
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
        # SmartSearch Wartezeit pruefen (ausserhalb des Mode-Checks)
        self.check_smart_search_timeout()

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
                "face_lock_active": self.face_lock_active,
                "smart_search_patrol_ready": self.smart_search_patrol_ready,
                "last_known_pan": self._last_known_position[0] if self._last_known_position else None,
                "last_known_tilt": self._last_known_position[1] if self._last_known_position else None,
            }

    # =========================================================================
    # SMART_SEARCH_RETURN (G1-T04)
    # =========================================================================

    def update_target_position(self, pan: float, tilt: float):
        """Aktuelle Zielposition merken (aufgerufen wenn Track aktiv).

        Muss vom Tracker bei jedem Frame mit gueltiger Zielposition aufgerufen werden.
        """
        with self._lock:
            self._last_known_position = (pan, tilt)
            self._target_lost_time = None
            self._smart_search_waiting = False
            self.smart_search_patrol_ready = False

    def on_target_lost(self):
        """Ziel verloren — startet SMART_SEARCH_RETURN Ablauf.

        1. Speichert letzte bekannte Position (bereits gespeichert)
        2. Setzt Verlust-Zeitpunkt
        3. smart_search_waiting = True (PTZ soll zur letzten Position fahren)
        4. Nach 3s: smart_search_patrol_ready = True (patrol_scan aktivieren)
        """
        with self._lock:
            if self._last_known_position is None:
                # Noch nie ein Ziel gehabt → direkt zu patrol
                self.smart_search_patrol_ready = True
                return
            self._target_lost_time = time.time()
            self._smart_search_waiting = True
            self.smart_search_patrol_ready = False
            logger.info(
                f"[ARBITER] SmartSearch: Rueckkehr zu "
                f"pan={self._last_known_position[0]:.1f} "
                f"tilt={self._last_known_position[1]:.1f}"
            )

    def check_smart_search_timeout(self):
        """Wartezeit pruefen (periodisch aufrufen, z.B. aus check_timeout).

        Nach 3s Wartezeit → smart_search_patrol_ready = True.
        """
        with self._lock:
            if not self._smart_search_waiting:
                return
            if self._target_lost_time is None:
                return
            elapsed = time.time() - self._target_lost_time
            if elapsed >= self._smart_search_wait_sec:
                self._smart_search_waiting = False
                self.smart_search_patrol_ready = True
                logger.info(
                    f"[ARBITER] SmartSearch: {elapsed:.1f}s gewartet → patrol_scan"
                )

    @property
    def smart_search_return_position(self) -> Optional[Tuple[float, float]]:
        """Gibt letzte bekannte Zielposition zurueck (oder None)."""
        with self._lock:
            if self._smart_search_waiting:
                return self._last_known_position
            return None

    # =========================================================================
    # FACE_LOCK_MODE
    # =========================================================================

    def update_face_confidence(self, face_confidence: float):
        """SCRFD Face-Confidence einpflegen. Aktiviert/deaktiviert Face-Lock.

        face_lock_active = True  wenn face_confidence > 0.8 fuer 5 Frames
        face_lock_active = False wenn face_confidence < 0.5
        """
        with self._lock:
            if face_confidence >= self._face_lock_enter_conf:
                self._face_lock_confidence_count += 1
                if (not self.face_lock_active
                        and self._face_lock_confidence_count >= self._face_lock_frames):
                    self.face_lock_active = True
                    logger.info(
                        f"[ARBITER] Face-Lock AKTIV "
                        f"(conf={face_confidence:.2f}, frames={self._face_lock_confidence_count})"
                    )
            elif face_confidence < self._face_lock_exit_conf:
                if self.face_lock_active:
                    logger.info(
                        f"[ARBITER] Face-Lock AUS (conf={face_confidence:.2f})"
                    )
                self.face_lock_active = False
                self._face_lock_confidence_count = 0
            # Zwischen 0.5 und 0.8: Zaehler nicht resetten (Hysterese)

    # =========================================================================
    # MULTI_PERSON_SCENE
    # =========================================================================

    def select_tracking_target(self, persons: list) -> Optional[dict]:
        """Waehlt bestes Ziel aus mehreren erkannten Personen.

        Prioritaet:
          1. face_id == "markus" (oder "Markus")
          2. Groesste Bounding Box (bbox_area)
          3. Zuletzt getracktes Target (last_tracked_target_id)

        Args:
            persons: Liste von Dicts mit Feldern:
                     - id: str (optional)
                     - face_id: str (optional, "markus"|"unknown"|None)
                     - bbox: (x1, y1, x2, y2) normalisiert (optional)
        Returns:
            Bestes Ziel-Dict oder None wenn Liste leer
        """
        if not persons:
            return None

        # Prioritaet 1: Markus erkannt
        for p in persons:
            fid = (p.get("face_id") or "").lower()
            if fid == "markus":
                with self._lock:
                    self._last_tracked_target_id = p.get("id")
                logger.debug("[ARBITER] MultiPerson: Markus erkannt → Prioritaet 1")
                return p

        # Prioritaet 2: Groesste BBox
        best = None
        best_area = -1.0
        for p in persons:
            bbox = p.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = p

        if best is not None:
            # Prioritaet 3: Falls gleich gross, zuletzt getracktes bevorzugen
            last_id = None
            with self._lock:
                last_id = self._last_tracked_target_id
            if last_id is not None:
                for p in persons:
                    if p.get("id") == last_id:
                        bbox = p.get("bbox")
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            area = (x2 - x1) * (y2 - y1)
                            # Nur wechseln wenn neue BBox groesser als 20% besser
                            if best_area > 0 and (best_area - area) / best_area < 0.20:
                                best = p
                                break

            with self._lock:
                self._last_tracked_target_id = best.get("id")
            logger.debug(
                f"[ARBITER] MultiPerson: {len(persons)} Personen → "
                f"groesste BBox (area={best_area:.3f})"
            )
            return best

        # Fallback: erste Person in Liste
        return persons[0]


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
