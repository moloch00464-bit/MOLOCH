#!/usr/bin/env python3
"""
M.O.L.O.C.H. Behavior Rules — Mood→Verhalten Mapping
=======================================================

Regelwerk das Mood-States auf konkrete Aktionen mappt:
  - dark + tension > 0.8 → Shadow-Modus, Sirene-Trigger
  - calm + markus_present → Guardian-Modus, blaues Licht
  - euphoric → Musik lauter, LED-Puls schneller
  - agitated → LED blinken, Wachsamkeit
  - alert → LED an, moderate Aufmerksamkeit
  - focused → LED Standlicht, ruhig

Publiziert behavior_trigger Event (Priority 3) mit konkreten Aktionen.

Singleton: get_behavior_rules()
Gate 4: Emergent Personality
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochBehaviorRules")

# Cooldown zwischen gleichen Behavior-Triggers (Sekunden)
TRIGGER_COOLDOWN = 10.0


class BehaviorRules:
    """Mood→Verhalten Regelwerk."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_triggers: Dict[str, float] = {}  # trigger_name → timestamp
        self._active_behavior: Optional[str] = None

    def evaluate(self, mood: str, tension: float = 0.0,
                 dominance: float = 0.5, personality_zone: str = "guardian",
                 face_id: Optional[str] = None,
                 music_energy: float = 0.0) -> List[Dict[str, Any]]:
        """Mood evaluieren und passende Behavior-Triggers generieren.

        Args:
            mood: Aktueller Mood aus MoodEngine
            tension: CoreIntegrator Tension (0.0-1.0)
            dominance: CoreIntegrator Dominance (-1.0 bis +1.0)
            personality_zone: "guardian" / "shadow" / "berserker"
            face_id: Erkannte Person
            music_energy: Spotify Audio Energy (0.0-1.0)

        Returns:
            Liste von Trigger-Dicts die publiziert werden sollen
        """
        triggers = []
        markus_present = face_id and face_id != "unknown"

        # === Dark + hohe Tension → Shadow-Modus, Sirene ===
        if mood == "dark" and tension > 0.8:
            triggers.append({
                "action": "shadow_mode",
                "led": "blink_fast",
                "sirene": True,
                "personality_zone": "shadow",
                "reason": "dark mood + hohe tension",
            })

        # === Agitated → LED blinken, Wachsamkeit ===
        elif mood == "agitated":
            triggers.append({
                "action": "agitated_mode",
                "led": "blink",
                "sirene": False,
                "personality_zone": "shadow",
                "reason": "agitated mood",
            })

        # === Alert → LED an, aufmerksam ===
        elif mood == "alert":
            triggers.append({
                "action": "alert_mode",
                "led": "on",
                "sirene": False,
                "personality_zone": personality_zone,
                "reason": "alert mood",
            })

        # === Euphoric → Musik lauter, LED-Puls ===
        elif mood == "euphoric":
            triggers.append({
                "action": "euphoric_mode",
                "led": "blink_slow",
                "music_volume": "up",
                "sirene": False,
                "personality_zone": "guardian",
                "reason": "euphoric mood",
            })

        # === Calm + Markus → Guardian, blaues Licht ===
        elif mood == "calm" and markus_present:
            triggers.append({
                "action": "guardian_calm",
                "led": "on",
                "sirene": False,
                "personality_zone": "guardian",
                "reason": "calm + markus present",
            })

        # === Focused → Standlicht, ruhig ===
        elif mood == "focused":
            triggers.append({
                "action": "focused_mode",
                "led": "on",
                "sirene": False,
                "personality_zone": "guardian",
                "reason": "focused mood",
            })

        # === Default (calm ohne Markus, away) → LED aus ===
        else:
            triggers.append({
                "action": "idle_mode",
                "led": "off",
                "sirene": False,
                "personality_zone": personality_zone,
                "reason": f"default ({mood})",
            })

        # Cooldown-Filter und Publish
        published = []
        now = time.time()
        for trigger in triggers:
            action = trigger["action"]

            with self._lock:
                last = self._last_triggers.get(action, 0.0)
                if (now - last) < TRIGGER_COOLDOWN and action == self._active_behavior:
                    continue
                self._last_triggers[action] = now
                self._active_behavior = action

            # Event publizieren
            try:
                from core.moloch_event_bus import get_event_bus
                get_event_bus().publish(
                    event_type="behavior_trigger",
                    source="behavior_rules",
                    priority=3,
                    payload=trigger,
                )
                published.append(trigger)
                logger.info(f"[BEHAVIOR] {action}: {trigger.get('reason')}")
            except Exception as e:
                logger.debug(f"[BEHAVIOR] Event publish: {e}")

        return published

    @property
    def active_behavior(self) -> Optional[str]:
        with self._lock:
            return self._active_behavior

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "active_behavior": self._active_behavior,
                "trigger_count": len(self._last_triggers),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[BehaviorRules] = None
_instance_lock = threading.Lock()


def get_behavior_rules() -> BehaviorRules:
    """Singleton-Zugriff auf Behavior Rules."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = BehaviorRules()
    return _instance
