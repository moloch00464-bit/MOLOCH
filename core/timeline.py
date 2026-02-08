"""
M.O.L.O.C.H. Timeline - Append-Only Event Log
==============================================

Zeitgefühl, nicht Erinnerung.
Loggt Fakten, keine Inhalte.

Event-Typen:
- system: Start, Stop, Fehler
- state: TTS, Voice, Mode-Änderungen
- interaction: Konversation, Input, Output (nur Metadaten)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import logging

logger = logging.getLogger(__name__)

TIMELINE_PATH = Path.home() / "moloch" / "state" / "timeline.jsonl"


class Timeline:
    """
    Append-only Timeline für M.O.L.O.C.H.

    Thread-safe, append-only, keine Inhalte.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._file_lock = threading.Lock()

        # Ensure directory exists
        TIMELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Timeline initialized: {TIMELINE_PATH}")

    def log(self, event_type: str, event: str, **meta) -> None:
        """
        Log an event to the timeline.

        Args:
            event_type: "system", "state", or "interaction"
            event: Short event name (e.g., "console_start", "tts_speak")
            **meta: Additional metadata (no content, just facts)
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "event": event
        }

        # Add metadata (filter out None values)
        for key, value in meta.items():
            if value is not None:
                entry[key] = value

        with self._file_lock:
            try:
                with open(TIMELINE_PATH, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.error(f"Timeline write error: {e}")

    # === SYSTEM EVENTS ===

    def system_start(self, component: str, **meta):
        """Log system component start."""
        self.log("system", f"{component}_start", **meta)

    def system_stop(self, component: str, **meta):
        """Log system component stop."""
        self.log("system", f"{component}_stop", **meta)

    def system_error(self, component: str, error_type: str, **meta):
        """Log system error (no error content, just type)."""
        self.log("system", f"{component}_error", error_type=error_type, **meta)

    def system_startup(self, **meta):
        """
        Log M.O.L.O.C.H. system startup.

        Automatically calculates offline duration since last event.
        Detects if last shutdown was clean or a crash.
        """
        last_event = _get_last_event()

        startup_meta = {**meta}

        if last_event:
            last_ts = last_event.get("ts")
            last_type = last_event.get("event", "")

            if last_ts:
                try:
                    last_time = datetime.fromisoformat(last_ts)
                    offline_seconds = (datetime.now() - last_time).total_seconds()
                    startup_meta["offline_seconds"] = round(offline_seconds, 1)
                    startup_meta["offline_minutes"] = round(offline_seconds / 60, 1)

                    # Detect crash: last event wasn't a shutdown
                    if "shutdown" not in last_type and "stop" not in last_type:
                        startup_meta["crash_recovery"] = True
                        startup_meta["last_event"] = last_type
                    else:
                        startup_meta["clean_boot"] = True
                except (ValueError, TypeError):
                    pass

        self.log("system", "moloch_startup", **startup_meta)

    def system_shutdown(self, reason: str = "manual", **meta):
        """
        Log M.O.L.O.C.H. system shutdown.

        Args:
            reason: "manual", "reboot", "poweroff", "signal"
        """
        self.log("system", "moloch_shutdown", reason=reason, **meta)

    # === STATE EVENTS ===

    def state_change(self, what: str, value: Any, **meta):
        """Log state change."""
        self.log("state", f"{what}_changed", value=str(value), **meta)

    def tts_speak(self, chars: int, voice: str, **meta):
        """Log TTS output (length only, no content)."""
        self.log("state", "tts_speak", chars=chars, voice=voice, **meta)

    def voice_change(self, voice: str, **meta):
        """Log voice change."""
        self.log("state", "voice_changed", voice=voice, **meta)

    # === INTERACTION EVENTS ===

    def user_input(self, chars: int, interface: str = "console", **meta):
        """Log user input (length only, no content)."""
        self.log("interaction", "user_input", chars=chars, interface=interface, **meta)

    def assistant_response(self, chars: int, **meta):
        """Log assistant response (length only, no content)."""
        self.log("interaction", "assistant_response", chars=chars, **meta)

    def conversation_start(self, interface: str = "console", **meta):
        """Log conversation start."""
        self.log("interaction", "conversation_start", interface=interface, **meta)

    def conversation_end(self, turns: int, **meta):
        """Log conversation end."""
        self.log("interaction", "conversation_end", turns=turns, **meta)

    def stt_transcribe(self, chars: int, duration_sec: float = None, **meta):
        """Log STT transcription (length only)."""
        self.log("interaction", "stt_transcribe", chars=chars, duration_sec=duration_sec, **meta)

    # === VISION EVENTS ===

    def person_detected(self, face_count: int = 1, confidence: float = 0.0, **meta):
        """Log person detection (before identification)."""
        self.log("vision", "person_detected", face_count=face_count, confidence=confidence, **meta)

    def person_identified(self, person_name: str, person_id: str = None, confidence: float = 0.0, **meta):
        """Log known person identification."""
        self.log("vision", "person_identified",
                 person_name=person_name,
                 person_id=person_id,
                 confidence=confidence,
                 **meta)

    def unknown_person(self, confidence: float = 0.0, **meta):
        """Log unknown person detection."""
        self.log("vision", "unknown_person", confidence=confidence, **meta)

    def person_left(self, person_name: str = None, duration_visible: float = None, **meta):
        """Log person leaving camera view."""
        self.log("vision", "person_left",
                 person_name=person_name,
                 duration_visible=duration_visible,
                 **meta)

    def face_training(self, person_name: str, num_samples: int, **meta):
        """Log face training event."""
        self.log("vision", "face_training",
                 person_name=person_name,
                 num_samples=num_samples,
                 **meta)

    def hailo_wake(self, **meta):
        """Log Hailo NPU wake event."""
        self.log("system", "hailo_wake", **meta)

    def hailo_standby(self, active_seconds: float = 0, **meta):
        """Log Hailo NPU standby event."""
        self.log("system", "hailo_standby", active_seconds=round(active_seconds, 1), **meta)


# Global singleton
_timeline: Optional[Timeline] = None


def get_timeline() -> Timeline:
    """Get or create Timeline instance."""
    global _timeline
    if _timeline is None:
        _timeline = Timeline()
    return _timeline


# Convenience functions
def log_event(event_type: str, event: str, **meta):
    """Quick event logging."""
    get_timeline().log(event_type, event, **meta)


# =============================================================================
# ZEITEMPFINDEN - Faktenbasiert, keine Emotion
# =============================================================================

def _read_timeline_events() -> list:
    """Read all events from timeline (newest last)."""
    if not TIMELINE_PATH.exists():
        return []

    events = []
    try:
        with open(TIMELINE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error reading timeline: {e}")

    return events


def _get_last_event() -> Optional[Dict[str, Any]]:
    """Get the most recent event from timeline."""
    if not TIMELINE_PATH.exists():
        return None

    try:
        # Read last line efficiently
        with open(TIMELINE_PATH, 'rb') as f:
            # Seek to end
            f.seek(0, 2)
            file_size = f.tell()

            if file_size == 0:
                return None

            # Read last 4KB (should be enough for last event)
            read_size = min(4096, file_size)
            f.seek(-read_size, 2)
            last_chunk = f.read().decode('utf-8', errors='ignore')

            # Find last complete line
            lines = last_chunk.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        logger.error(f"Error reading last event: {e}")

    return None


def _find_last_stop() -> Optional[datetime]:
    """Find the timestamp of the last system stop/shutdown event."""
    events = _read_timeline_events()

    # Search backwards for last shutdown or stop event
    for event in reversed(events):
        event_name = event.get("event", "")
        if event.get("type") == "system":
            if "shutdown" in event_name or "_stop" in event_name:
                try:
                    return datetime.fromisoformat(event["ts"])
                except (ValueError, KeyError):
                    continue

    return None


def _find_last_startup() -> Optional[Dict[str, Any]]:
    """Find the most recent startup event."""
    events = _read_timeline_events()

    for event in reversed(events):
        if event.get("event") == "moloch_startup":
            return event

    return None


def _classify_offline_duration(seconds: float) -> str:
    """
    Classify offline duration into categories.

    Schwellen (hart codiert):
    - < 5 Minuten       → "kurz"
    - 5 min - 2 Stunden → "eine_weile"
    - 2 - 12 Stunden    → "laenger"
    - > 12 Stunden      → "sehr_lange"
    """
    minutes = seconds / 60
    hours = minutes / 60

    if minutes < 5:
        return "kurz"
    elif hours < 2:
        return "eine_weile"
    elif hours < 12:
        return "laenger"
    else:
        return "sehr_lange"


def describe_last_offline_duration() -> Optional[str]:
    """
    Beschreibt die letzte Offline-Dauer in natürlicher Sprache.

    Wird nur bei Bedarf genutzt:
    - beim ersten Gespräch nach Boot
    - auf Nachfrage ("warst du weg?")

    Returns:
        Sprachliche Beschreibung oder None wenn keine Daten.

    KEINE Emotion, KEINE Schuld, KEIN Drama.
    Nur Fakten, sprachlich eingefärbt.
    """
    last_stop = _find_last_stop()

    if last_stop is None:
        return None

    now = datetime.now()
    offline_seconds = (now - last_stop).total_seconds()

    if offline_seconds < 0:
        return None

    category = _classify_offline_duration(offline_seconds)

    # Check for crash
    last_startup = _find_last_startup()
    was_crash = last_startup.get("crash_recovery", False) if last_startup else False

    # Sprach-Mapping (neutral, faktenbasiert)
    if was_crash:
        phrases = {
            "kurz": "Bin abgestürzt. War nur kurz weg.",
            "eine_weile": "Bin abgestürzt. War eine Weile offline.",
            "laenger": "Bin abgestürzt. War länger nicht da.",
            "sehr_lange": "Bin abgestürzt. War ziemlich lange weg."
        }
    else:
        phrases = {
            "kurz": "Ich war nur kurz weg.",
            "eine_weile": "Ich war eine Weile offline.",
            "laenger": "Ich war länger nicht da.",
            "sehr_lange": "Ich war ziemlich lange weg."
        }

    return phrases.get(category, "Ich bin wieder da.")


def get_offline_info() -> Dict[str, Any]:
    """
    Get raw offline information for programmatic use.

    Returns dict with:
    - last_stop: ISO timestamp or None
    - offline_seconds: float or None
    - category: "kurz", "eine_weile", "laenger", "sehr_lange" or None
    - was_crash: bool - True if last shutdown was not clean
    - last_event: str - name of last event before current startup
    """
    last_event = _get_last_event()
    last_stop = _find_last_stop()

    # Check last startup for crash info
    last_startup = _find_last_startup()
    was_crash = False
    if last_startup:
        was_crash = last_startup.get("crash_recovery", False)

    if last_stop is None and last_event is None:
        return {
            "last_stop": None,
            "offline_seconds": None,
            "category": None,
            "was_crash": False,
            "last_event": None
        }

    # Use last event timestamp if no explicit stop found
    if last_stop is None and last_event:
        try:
            last_stop = datetime.fromisoformat(last_event["ts"])
        except (ValueError, KeyError):
            pass

    if last_stop is None:
        return {
            "last_stop": None,
            "offline_seconds": None,
            "category": None,
            "was_crash": was_crash,
            "last_event": last_event.get("event") if last_event else None
        }

    now = datetime.now()
    offline_seconds = (now - last_stop).total_seconds()

    return {
        "last_stop": last_stop.isoformat(),
        "offline_seconds": offline_seconds,
        "category": _classify_offline_duration(offline_seconds),
        "was_crash": was_crash,
        "last_event": last_event.get("event") if last_event else None
    }
