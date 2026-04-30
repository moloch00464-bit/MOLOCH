"""
MoodToSpotify — Mood-Wechsel triggert Spotify-Zone-Bias

Subscribed: mood_changed
Mapping Mood -> Zone (nur signifikante Wechsel):
  calm/sleepy: keine Action
  alert:       guardian (Futurepop)
  focused:     keine Action
  shadow:      shadow (Dark Electro/EBM)
  berserker:   berserker (Aggrotech)

Wartet 30s no-action nach Mood-Wechsel, dann setzt Bias.
Cooldown: pro Mood-Wechsel max 1x / 5min.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("expression.mood_to_spotify")

_MOOD_TO_ZONE = {
    "calm": None,
    "sleepy": None,
    "alert": "guardian",
    "focused": None,
    "shadow": "shadow",
    "berserker": "berserker",
}

_NO_ACTION_DELAY = 30.0  # Sekunden bis Bias gesetzt wird
_COOLDOWN_S = 300.0  # 5 Minuten Cooldown


class MoodToSpotify:
    """Spotify-Zone-Bias bei Mood-Wechseln (Cooldown + Verzoegerung)."""

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._bus = None
        self._last_mood: Optional[str] = None
        self._last_apply_ts: float = 0.0
        self._pending_zone: Optional[str] = None
        self._pending_ts: float = 0.0
        self._timer: Optional[threading.Timer] = None
        self._subscribed = False

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            try:
                from core.moloch_event_bus import get_event_bus
                self._bus = get_event_bus()
                self._bus.subscribe("mood_changed", self._on_mood_event, priority=5)
                self._subscribed = True
                self._running = True
                logger.info("MoodToSpotify gestartet (subscribed: mood_changed)")
                return True
            except Exception as e:
                logger.warning(f"MoodToSpotify start fehlgeschlagen: {e}")
                return False

    def stop(self):
        with self._lock:
            self._running = False
            self._subscribed = False
            if self._timer:
                try:
                    self._timer.cancel()
                except Exception:
                    pass
                self._timer = None
            logger.info("MoodToSpotify gestoppt")

    def _on_mood_event(self, payload):
        try:
            data = payload if isinstance(payload, dict) else {}
            old = data.get("old_mood") or data.get("from") or self._last_mood
            new = data.get("new_mood") or data.get("to") or data.get("mood")
            if not new:
                return
            self.on_mood_change(old, new)
        except Exception as e:
            logger.debug(f"MoodToSpotify _on_mood_event Fehler: {e}")

    def on_mood_change(self, old: Optional[str], new: str):
        """Externe API: triggert verzoegerten Bias-Switch."""
        new = (new or "").lower().strip()
        zone = _MOOD_TO_ZONE.get(new)
        if zone is None:
            self._last_mood = new
            return
        # Signifikanz pruefen: nur calm<->shadow, shadow<->berserker etc.
        old_l = (old or "").lower().strip()
        if old_l == new:
            return
        with self._lock:
            now = time.time()
            if (now - self._last_apply_ts) < _COOLDOWN_S:
                logger.debug(f"MoodToSpotify: Cooldown aktiv, skip {new} (zone={zone})")
                self._last_mood = new
                return
            # Pending Switch — neuer Wechsel cancelt alten
            if self._timer:
                try:
                    self._timer.cancel()
                except Exception:
                    pass
            self._pending_zone = zone
            self._pending_ts = now
            self._timer = threading.Timer(_NO_ACTION_DELAY, self._delayed_apply, args=[zone])
            self._timer.daemon = True
            self._timer.start()
            self._last_mood = new
            logger.info(f"MoodToSpotify: Bias '{zone}' geplant in {_NO_ACTION_DELAY}s (mood {old_l}->{new})")

    def _delayed_apply(self, zone: str):
        """Wird nach 30s aufgerufen — setzt Bias falls noch aktuell."""
        with self._lock:
            if self._pending_zone != zone:
                return
            self._last_apply_ts = time.time()
            self._pending_zone = None
        self._apply_zone_bias(zone)

    def _apply_zone_bias(self, zone: str):
        """Best-effort: ruft spotify_controller.set_zone_bias() oder play_by_mood()."""
        try:
            from core.spotify_controller import get_spotify
            sp = get_spotify()
            if hasattr(sp, "set_zone_bias"):
                sp.set_zone_bias(zone)
                logger.info(f"MoodToSpotify: set_zone_bias({zone}) erfolgreich")
            else:
                logger.info(
                    f"MoodToSpotify: spotify_controller.set_zone_bias() fehlt — "
                    f"zone='{zone}' nur registriert"
                )
        except Exception as e:
            logger.debug(f"MoodToSpotify _apply_zone_bias: {e}")

    def get_state(self) -> dict:
        with self._lock:
            return {
                "alive": self._running,
                "subscribed": self._subscribed,
                "last_mood": self._last_mood,
                "pending_zone": self._pending_zone,
                "last_apply_ts": self._last_apply_ts,
                "last_action_age": time.time() - self._last_apply_ts if self._last_apply_ts else None,
            }


_instance: Optional[MoodToSpotify] = None
_instance_lock = threading.Lock()


def get_mood_to_spotify() -> MoodToSpotify:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = MoodToSpotify()
        return _instance
