"""BodyController: entscheidet, welche Pose wann an den Koerper geht.

Rangfolge (hoeher gewinnt):
    Not-Aus  >  Grenzen (immer)  >  MANUAL  >  DIALOGUE  >  IDLE

Sicherheitsregeln:
- Not-Aus rastet ein: bis release_estop() geht ausser stop() nichts raus.
- Jeder Befehl hat eine Lebensdauer (TTL). Abgelaufene oder schon beim
  Eintreffen veraltete Befehle werden verworfen.
- Ohne gueltigen Befehl faehrt der Koerper einmal in die Ruhepose und bleibt dort.
- Ohne Verbindung wird nichts aufgestaut: Befehle verfallen, nach dem
  Wiederverbinden beginnt die erste Bewegung langsam (Startpose unbekannt).
- Jede Pose wird geklemmt, in der Geschwindigkeit begrenzt und gedrosselt.

tick() wird vom Aufrufer zyklisch gerufen (Konzept: 5 Hz). Der Controller
startet selbst keinen Thread.
"""

import logging
import threading
import time
from dataclasses import replace
from enum import IntEnum
from typing import Callable, Dict, Optional

from .backend import MotionBackend
from .pose import NEUTRAL, BodyPose, clamp_pose, pose_to_dict

logger = logging.getLogger(__name__)

MAX_TTL_S = 10.0  # laenger darf kein Befehl gueltig bleiben


class Priority(IntEnum):
    IDLE = 0        # Grundausdruck aus compute_pose()
    DIALOGUE = 1    # Reaktion im Gespraech (Blick, Sprech-Animation)
    MANUAL = 2      # Markus steuert direkt


class BodyController:

    def __init__(self, backend: MotionBackend, mappings: Dict,
                 clock: Callable[[], float] = time.monotonic):
        self._backend = backend
        self._limits = mappings["limits"]
        self._m = mappings["motion"]
        self._clock = clock
        self._lock = threading.Lock()
        self._slots: Dict[Priority, tuple] = {}   # Priority -> (pose, expires_at)
        self._estop = False
        self._stop_sent = False
        self._quiet = False
        self._last_sent: Optional[BodyPose] = None   # None = Pose des Koerpers unbekannt
        self._last_send_at = float("-inf")
        self._last_command_at = float("-inf")
        self._at_neutral = False
        self._next_connect_at = float("-inf")
        self._last_result = "init"

    # === Eingaben ===

    def submit(self, priority: Priority, pose: BodyPose, ttl_s: float = 1.0,
               created_at: Optional[float] = None) -> bool:
        """Befehl einreichen. False wenn abgelehnt (Not-Aus, TTL, veraltet).

        created_at: Entstehungszeit in der Uhr des Controllers. Ist der Befehl
        beim Eintreffen schon aelter als seine TTL, wird er verworfen.
        """
        with self._lock:
            now = self._clock()
            if self._estop:
                return False
            if not (0.0 < ttl_s <= MAX_TTL_S):
                return False
            born = now if created_at is None else created_at
            if born > now + 0.5 or now - born >= ttl_s:
                return False  # aus der Zukunft oder schon abgelaufen
            self._slots[Priority(priority)] = (pose, born + ttl_s)
            return True

    def estop(self) -> None:
        """Not-Aus: alles verwerfen, beim naechsten tick() stop() senden."""
        with self._lock:
            self._estop = True
            self._stop_sent = False
            self._slots.clear()
        logger.warning("[BODY] Not-Aus aktiv")

    def release_estop(self) -> None:
        with self._lock:
            self._estop = False
            self._slots.clear()
            self._last_sent = None      # Motoren waren entlastet: Pose unbekannt
            self._at_neutral = False
        logger.info("[BODY] Not-Aus aufgehoben")

    def set_quiet(self, quiet: bool) -> None:
        """Ruhemodus (z.B. Nacht): nur MANUAL bewegt, sonst Ruhepose."""
        with self._lock:
            self._quiet = bool(quiet)

    # === Zyklus ===

    def tick(self) -> str:
        """Einen Schritt ausfuehren. Gibt zurueck, was passiert ist."""
        with self._lock:
            self._last_result = self._tick(self._clock())
            return self._last_result

    def _tick(self, now: float) -> str:
        if self._estop:
            if not self._stop_sent and self._ensure_connected(now):
                try:
                    self._backend.stop()
                    self._stop_sent = True
                except Exception as e:
                    logger.error(f"[BODY] stop() fehlgeschlagen, neuer Versuch: {e}")
            return "estop"

        if not self._ensure_connected(now):
            self._slots.clear()
            self._last_sent = None
            self._at_neutral = False
            return "offline"

        pose = self._select(now)
        if pose is None:
            if self._at_neutral:
                return "idle"
            if now - self._last_command_at < self._m["idle_timeout_s"]:
                return "idle"
            pose, to_neutral = replace(NEUTRAL, duration_s=1.0), True
        else:
            self._last_command_at = now
            to_neutral = False

        if now - self._last_send_at < self._m["min_interval_s"]:
            return "rate_limited"

        pose = self._shape(clamp_pose(pose, self._limits))
        if (self._last_sent is not None
                and pose.max_delta(self._last_sent) < self._m["deadband"]):
            self._at_neutral = to_neutral
            return "unchanged"

        try:
            self._backend.goto(pose)
        except Exception as e:
            logger.warning(f"[BODY] goto fehlgeschlagen, Koerper offline: {e}")
            self._last_sent = None
            self._at_neutral = False
            self._next_connect_at = now + self._m["reconnect_backoff_s"]
            return "offline"

        self._last_sent = pose
        self._last_send_at = now
        self._at_neutral = to_neutral
        return "neutral" if to_neutral else "sent"

    # === intern ===

    def _ensure_connected(self, now: float) -> bool:
        if self._backend.is_connected():
            return True
        if now < self._next_connect_at:
            return False
        try:
            ok = bool(self._backend.connect())
        except Exception as e:
            logger.debug(f"[BODY] connect fehlgeschlagen: {e}")
            ok = False
        if not ok:
            self._next_connect_at = now + self._m["reconnect_backoff_s"]
        return ok

    def _select(self, now: float) -> Optional[BodyPose]:
        """Hoechstrangigen, noch gueltigen Befehl waehlen; Abgelaufenes loeschen."""
        for prio in [p for p, (_, exp) in self._slots.items() if exp <= now]:
            del self._slots[prio]
        floor = Priority.MANUAL if self._quiet else Priority.IDLE
        valid = [p for p in self._slots if p >= floor]
        return self._slots[max(valid)][0] if valid else None

    def _shape(self, pose: BodyPose) -> BodyPose:
        """Dauer so setzen, dass die Hoechstgeschwindigkeit eingehalten wird."""
        m = self._m
        dur = min(max(pose.duration_s, m["min_duration_s"]), m["max_duration_s"])
        if self._last_sent is None:
            dur = m["max_duration_s"]   # Startpose unbekannt: langsam anfahren
        else:
            dur = max(dur, pose.max_delta(self._last_sent) / m["max_speed_deg_s"])
        return replace(pose, duration_s=dur)

    def status(self) -> Dict:
        with self._lock:
            return {
                "connected": self._backend.is_connected(),
                "estop": self._estop,
                "quiet": self._quiet,
                "last_result": self._last_result,
                "pending": sorted(p.name for p in self._slots),
                "last_pose": pose_to_dict(self._last_sent) if self._last_sent else None,
            }
