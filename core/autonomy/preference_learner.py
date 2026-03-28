#!/usr/bin/env python3
"""
M.O.L.O.C.H. Preference Learner — Gate 10
============================================
Lernt Praeferenzen aus Beobachtung und Feedback.

Statt fester IF/THEN Regeln: MOLOCH lernt welche Aktionen
in welchem Kontext gut ankommen.

Lern-Signale:
  POSITIV: Markus bleibt ruhig sitzen, laechelt, sagt "gut/ja/cool"
  NEGATIV: Markus steht auf und geht, sagt "nein/stopp/aus", Tension steigt

Gelernte Praeferenzen:
  - Musik: Genre/Artist pro Tageszeit + Stimmung
  - Licht: Helligkeit/Farbe pro Aktivitaet
  - Verhalten: Wie viel soll MOLOCH reden/reagieren
  - PTZ: Soll Kamera folgen oder ruhig bleiben

Persistent: /mnt/moloch-data/memory/preferences.json
Singleton: get_preference_learner()
"""

import json
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("PreferenceLearner")

PERSIST_PATH = "/mnt/moloch-data/memory/preferences.json"

# Kontext-Dimensionen
CONTEXTS = ["morning", "afternoon", "evening", "night",
            "alone", "working", "conversation", "party"]

# Lernbare Praeferenzen mit Default-Werten (0.5 = neutral)
DEFAULT_PREFERENCES = {
    "music_volume": 0.5,        # 0=leise, 1=laut
    "music_energy": 0.5,        # 0=ruhig, 1=energetisch
    "light_brightness": 0.5,    # 0=dunkel, 1=hell
    "ptz_follow_rate": 0.5,     # 0=ruhig, 1=aktiv folgen
    "speech_frequency": 0.5,    # 0=still, 1=gespraechig
    "reaction_speed": 0.5,      # 0=traege, 1=sofort
}


class PreferenceLearner:
    """Lernt Praeferenzen aus Reinforcement-Signalen."""

    LEARN_RATE = 0.05           # Wie schnell anpassen (0-1)
    DECAY_RATE = 0.001          # Langsamer Drift zurueck zur Mitte
    POSITIVE_BOOST = 0.02       # Reward pro positivem Signal
    NEGATIVE_PENALTY = 0.03     # Penalty pro negativem Signal

    def __init__(self):
        self._lock = threading.Lock()
        # Praeferenzen pro Kontext
        self._prefs: Dict[str, Dict[str, float]] = {}
        self._current_context: str = "alone"
        self._reward_history: List[Dict] = []  # Letzte 100 Rewards
        self._total_rewards: int = 0
        self._total_penalties: int = 0
        self._load()

    def _load(self):
        try:
            with open(PERSIST_PATH, "r") as f:
                data = json.load(f)
            self._prefs = data.get("preferences", {})
            self._total_rewards = data.get("total_rewards", 0)
            self._total_penalties = data.get("total_penalties", 0)
            logger.info(f"[PREF] {len(self._prefs)} Kontext-Profile geladen "
                        f"(R={self._total_rewards} P={self._total_penalties})")
        except FileNotFoundError:
            logger.info("[PREF] Kein Profil — starte mit Defaults")
        except Exception as e:
            logger.warning(f"[PREF] Laden: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(PERSIST_PATH), exist_ok=True)
            with open(PERSIST_PATH, "w") as f:
                json.dump({
                    "preferences": self._prefs,
                    "total_rewards": self._total_rewards,
                    "total_penalties": self._total_penalties,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"[PREF] Speichern: {e}")

    def set_context(self, activity: str = "alone", hour: int = -1):
        """Aktuellen Kontext setzen (aus Awareness-Modulen)."""
        import datetime
        if hour < 0:
            hour = datetime.datetime.now().hour

        if 6 <= hour < 12:
            time_ctx = "morning"
        elif 12 <= hour < 17:
            time_ctx = "afternoon"
        elif 17 <= hour < 22:
            time_ctx = "evening"
        else:
            time_ctx = "night"

        # Kombinierter Kontext: Tageszeit + Aktivitaet
        self._current_context = f"{time_ctx}_{activity}"

    def reward(self, preference_key: str, amount: float = None):
        """Positives Signal — aktuelle Einstellung war gut.

        Args:
            preference_key: z.B. "music_volume", "speech_frequency"
            amount: Staerke des Rewards (default: POSITIVE_BOOST)
        """
        if amount is None:
            amount = self.POSITIVE_BOOST
        with self._lock:
            self._apply_signal(preference_key, amount)
            self._total_rewards += 1
            if self._total_rewards % 10 == 0:
                self._save()

    def penalty(self, preference_key: str, amount: float = None):
        """Negatives Signal — aktuelle Einstellung war schlecht.

        Bewegt Praeferenz in Gegenrichtung (zurueck zur Mitte).
        """
        if amount is None:
            amount = -self.NEGATIVE_PENALTY
        with self._lock:
            self._apply_signal(preference_key, amount)
            self._total_penalties += 1
            if self._total_penalties % 10 == 0:
                self._save()

    def _apply_signal(self, key: str, delta: float):
        """Signal auf aktuelle Kontext-Praeferenz anwenden."""
        ctx = self._current_context
        if ctx not in self._prefs:
            self._prefs[ctx] = dict(DEFAULT_PREFERENCES)
        if key not in self._prefs[ctx]:
            self._prefs[ctx][key] = DEFAULT_PREFERENCES.get(key, 0.5)

        old = self._prefs[ctx][key]
        new = max(0.0, min(1.0, old + delta))
        self._prefs[ctx][key] = new

    def get_preference(self, key: str) -> float:
        """Aktuelle Praeferenz fuer den aktuellen Kontext holen."""
        ctx = self._current_context
        if ctx in self._prefs and key in self._prefs[ctx]:
            return self._prefs[ctx][key]
        return DEFAULT_PREFERENCES.get(key, 0.5)

    def get_all_preferences(self) -> Dict[str, float]:
        """Alle Praeferenzen fuer aktuellen Kontext."""
        ctx = self._current_context
        if ctx in self._prefs:
            result = dict(DEFAULT_PREFERENCES)
            result.update(self._prefs[ctx])
            return result
        return dict(DEFAULT_PREFERENCES)

    def tick(self):
        """Einmal pro Minute: langsamer Drift zurueck zur Mitte (Vergessen)."""
        with self._lock:
            for ctx in self._prefs:
                for key in self._prefs[ctx]:
                    val = self._prefs[ctx][key]
                    # Drift Richtung 0.5
                    if val > 0.5:
                        self._prefs[ctx][key] = max(0.5, val - self.DECAY_RATE)
                    elif val < 0.5:
                        self._prefs[ctx][key] = min(0.5, val + self.DECAY_RATE)

    def get_status(self) -> Dict:
        return {
            "context": self._current_context,
            "preferences": self.get_all_preferences(),
            "total_rewards": self._total_rewards,
            "total_penalties": self._total_penalties,
            "contexts_learned": len(self._prefs),
        }


# Singleton
_instance: Optional[PreferenceLearner] = None

def get_preference_learner() -> PreferenceLearner:
    global _instance
    if _instance is None:
        _instance = PreferenceLearner()
    return _instance
