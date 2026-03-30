#!/usr/bin/env python3
"""
M.O.L.O.C.H. Decision Engine — Utility-basierte autonome Entscheidungen
=========================================================================

Kombiniert Mood, Context, Activity, Memory zu autonomen Aktionen.
Utility-basiert: Jede moegliche Aktion bekommt einen Score (0-1),
die Aktion mit hoechstem Score gewinnt.

Aktionstypen:
  - music_change: Musik wechseln/starten/stoppen
  - light_change: LED-Muster aendern
  - ptz_move: Kamera in Zone fahren
  - speak: Spontaner TTS-Kommentar
  - silence: Nichts tun (Default, immer Baseline-Score)

Cooldowns verhindern Spam. Silence hat immer einen Baseline-Score
von 0.3 — Aktionen muessen das uebertreffen um zu feuern.

Publiziert decision_made Event (Priority 3) bei neuer Entscheidung.

Singleton: get_decision_engine()
Gate 5: Autonomous Environmental Agent
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochDecisionEngine")

# Cooldowns pro Aktion (Sekunden) — verhindert Spam
COOLDOWNS = {
    "music_change": 120.0,   # 2 Minuten zwischen Musikwechseln
    "light_change": 15.0,    # 15s zwischen LED-Aenderungen
    "ptz_move": 30.0,        # 30s zwischen PTZ-Moves
    "speak": 60.0,           # 1 Minute zwischen Kommentaren
    "web_search": 300.0,     # 5 Minuten zwischen autonomen Websuchen
    "reflect": 600.0,        # 10 Minuten zwischen NPU-Reflexionen
    "silence": 0.0,          # Kein Cooldown fuer Nichtstun
}

# Baseline-Score fuer Silence — Aktionen muessen das uebertreffen
SILENCE_BASELINE = 0.3


class DecisionEngine:
    """Utility-basierte Entscheidungs-Engine fuer autonomes Verhalten."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_actions: Dict[str, float] = {}  # action_type → timestamp
        self._last_decision: Optional[Dict[str, Any]] = None
        self._enabled = True

        # Aktuelle Signale (werden extern aktualisiert)
        self._mood: str = "calm"
        self._tension: float = 0.0
        self._dominance: float = 0.0
        self._activity: str = "away"
        self._zone: Optional[str] = None
        self._face_id: Optional[str] = None
        self._music_energy: float = 0.0
        self._context_score: float = 0.0
        self._alertness: float = 0.0
        self._engagement: float = 0.0
        self._music_playing: bool = False
        self._hour: int = 0

    def update_signals(self, mood: str = "calm", tension: float = 0.0,
                       dominance: float = 0.0, activity: str = "away",
                       zone: Optional[str] = None, face_id: Optional[str] = None,
                       music_energy: float = 0.0, context_score: float = 0.0,
                       alertness: float = 0.0, engagement: float = 0.0,
                       music_playing: bool = False):
        """Alle Signale auf einmal updaten (Thread-safe)."""
        with self._lock:
            self._mood = mood
            self._tension = tension
            self._dominance = dominance
            self._activity = activity
            self._zone = zone
            self._face_id = face_id
            self._music_energy = music_energy
            self._context_score = context_score
            self._alertness = alertness
            self._engagement = engagement
            self._music_playing = music_playing
            self._hour = time.localtime().tm_hour

    def decide(self) -> Dict[str, Any]:
        """Autonome Entscheidung treffen.

        Berechnet Utility-Scores fuer alle Aktionstypen,
        waehlt die mit hoechstem Score (ueber Cooldown + Silence-Baseline).

        Returns:
            Dict mit action, score, reason, params
        """
        if not self._enabled:
            return {"action": "silence", "score": 1.0, "reason": "engine_disabled"}

        with self._lock:
            candidates = self._score_all()

        # Cooldown-Filter
        now = time.time()
        valid = []
        for c in candidates:
            action = c["action"]
            cooldown = COOLDOWNS.get(action, 0.0)
            last = self._last_actions.get(action, 0.0)
            if (now - last) >= cooldown:
                valid.append(c)

        if not valid:
            return {"action": "silence", "score": SILENCE_BASELINE, "reason": "all_on_cooldown"}

        # Beste Aktion waehlen
        best = max(valid, key=lambda x: x["score"])

        # Nur ausfuehren wenn ueber Silence-Baseline
        if best["action"] != "silence" and best["score"] <= SILENCE_BASELINE:
            return {"action": "silence", "score": SILENCE_BASELINE, "reason": "below_threshold"}

        # Entscheidung registrieren
        with self._lock:
            self._last_actions[best["action"]] = now
            self._last_decision = best

        # Event publizieren
        if best["action"] != "silence":
            try:
                from core.moloch_event_bus import get_event_bus
                get_event_bus().publish(
                    event_type="decision_made",
                    source="decision_engine",
                    priority=3,
                    payload=best,
                )
                logger.info(f"[DECISION] {best['action']}: {best.get('reason', '?')} "
                            f"(score={best['score']:.2f})")
            except Exception as e:
                logger.debug(f"[DECISION] Event publish: {e}")

        # Autonome Websuche ausfuehren wenn gewonnen
        if best["action"] == "web_search":
            try:
                from core.net.autonomous_search import get_autonomous_search
                threading.Thread(
                    target=get_autonomous_search().trigger_search,
                    daemon=True,
                    name="WebSearch-Trigger",
                ).start()
            except Exception as e:
                logger.debug(f"[DECISION] Web search trigger: {e}")

        return best

    def _score_all(self) -> List[Dict[str, Any]]:
        """Utility-Scores fuer alle Aktionstypen berechnen (unter Lock)."""
        candidates = []

        # --- Silence: Immer als Baseline ---
        candidates.append({
            "action": "silence",
            "score": SILENCE_BASELINE,
            "reason": "baseline",
        })

        # --- Music Change ---
        candidates.append(self._score_music())

        # --- Light Change ---
        candidates.append(self._score_light())

        # --- PTZ Move ---
        candidates.append(self._score_ptz())

        # --- Speak ---
        candidates.append(self._score_speak())

        # --- Web Search ---
        candidates.append(self._score_web_search())

        # --- Reflect (NPU Selbstreflexion) ---
        candidates.append(self._score_reflect())

        return candidates

    def _score_music(self) -> Dict[str, Any]:
        """Utility-Score fuer Musikwechsel."""
        score = 0.0
        reason = ""
        params = {}

        # Musik passt nicht zum Mood
        if self._mood == "dark" and self._music_energy > 0.6:
            score = 0.55
            reason = "dark_mood_but_happy_music"
            params = {"target_mood": "dark"}
        elif self._mood == "euphoric" and self._music_energy < 0.3:
            score = 0.50
            reason = "euphoric_mood_but_quiet_music"
            params = {"target_mood": "euphoric"}
        # Niemand da → Musik aus
        elif self._activity == "away" and self._music_playing:
            score = 0.45
            reason = "nobody_home_music_on"
            params = {"command": "pause"}
        # Markus kommt → Musik starten
        elif self._face_id and self._face_id != "unknown" and not self._music_playing:
            score = 0.40
            reason = "markus_arrived_no_music"
            params = {"command": "play", "for_person": self._face_id}
        # Nacht → leise Musik
        elif self._hour >= 23 or self._hour < 6:
            if self._music_energy > 0.5:
                score = 0.45
                reason = "night_loud_music"
                params = {"command": "volume_down"}

        return {"action": "music_change", "score": score, "reason": reason, "params": params}

    def _score_light(self) -> Dict[str, Any]:
        """Utility-Score fuer LED-Aenderung."""
        score = 0.0
        reason = ""
        params = {}

        # Mood → LED
        if self._mood == "dark" and self._tension > 0.7:
            score = 0.50
            reason = "dark_mood_high_tension"
            params = {"led": "blink_fast"}
        elif self._mood == "euphoric":
            score = 0.40
            reason = "euphoric_mood"
            params = {"led": "blink_slow"}
        elif self._mood == "calm" and self._activity == "working":
            score = 0.35
            reason = "calm_working"
            params = {"led": "on"}
        elif self._activity == "away":
            score = 0.35
            reason = "nobody_home"
            params = {"led": "off"}

        return {"action": "light_change", "score": score, "reason": reason, "params": params}

    def _score_ptz(self) -> Dict[str, Any]:
        """Utility-Score fuer PTZ-Bewegung."""
        score = 0.0
        reason = ""
        params = {}

        # Hohe Alertness + Tuer-Zone → zur Tuer schauen
        if self._alertness > 0.7 and self._zone != "tuer":
            score = 0.50
            reason = "high_alertness_check_door"
            params = {"target_zone": "tuer"}
        # Allein + Markus am Schreibtisch → Kamera folgt
        elif self._face_id and self._zone != "schreibtisch" and self._activity == "working":
            score = 0.35
            reason = "follow_to_desk"
            params = {"target_zone": "schreibtisch"}
        # Niemand da → Park-Position (Tuer beobachten)
        elif self._activity == "away" and self._zone != "tuer":
            score = 0.40
            reason = "park_at_door"
            params = {"target_zone": "tuer"}

        return {"action": "ptz_move", "score": score, "reason": reason, "params": params}

    def _score_speak(self) -> Dict[str, Any]:
        """Utility-Score fuer spontanen Kommentar (skaliert mit PreferenceLearner)."""
        score = 0.0
        reason = ""
        params = {}

        # Hohes Engagement + bekannte Person → Kommentar
        if self._engagement > 0.7 and self._face_id and self._face_id != "unknown":
            score = 0.35
            reason = "high_engagement"
            params = {"type": "observation"}
        # Mood-Wechsel zu dark → Kommentar
        elif self._mood == "dark" and self._tension > 0.8:
            score = 0.40
            reason = "dark_mood_warning"
            params = {"type": "warning"}
        # Lange allein → Begruessung bei Rueckkehr
        elif self._mood == "calm" and self._face_id and self._engagement > 0.5:
            score = 0.32
            reason = "welcome_back"
            params = {"type": "greeting"}

        # PreferenceLearner: speech_frequency skaliert Score
        # 0.5 = neutral (Score unveraendert), 0.0 = still, 1.0 = gespraechig
        if score > 0:
            try:
                from core.autonomy.preference_learner import get_preference_learner
                speech_pref = get_preference_learner().get_preference("speech_frequency")
                # Skalierung: 0.0 → Score*0.5, 0.5 → Score*1.0, 1.0 → Score*1.3
                score *= 0.5 + speech_pref * 0.8
            except Exception:
                pass

        return {"action": "speak", "score": score, "reason": reason, "params": params}

    def _score_web_search(self) -> Dict[str, Any]:
        """Utility-Score fuer autonome Websuche."""
        score = 0.0
        reason = ""
        params: Dict[str, Any] = {}

        # Pruefen ob Suche erlaubt
        try:
            from core.net.autonomous_search import get_autonomous_search
            if not get_autonomous_search().permitted:
                return {"action": "web_search", "score": 0.0, "reason": "not_permitted"}
        except Exception:
            return {"action": "web_search", "score": 0.0, "reason": "module_unavailable"}

        # Hohes Engagement + bekannte Person → Kontext-Suche
        if self._engagement > 0.6 and self._face_id and self._face_id != "unknown":
            score = 0.38
            reason = "curious_during_engagement"
            params = {"trigger": "engagement"}
        # Allein + ruhige Phase → Hintergrund-Recherche
        elif self._activity == "away" and self._tension < 0.3:
            score = 0.35
            reason = "idle_background_search"
            params = {"trigger": "idle"}
        # Hohe Tension + Guardian → Sicherheits-Check
        elif self._tension > 0.6 and self._dominance > 0.15:
            score = 0.42
            reason = "security_check"
            params = {"trigger": "security"}
        # Morgens → Briefing
        elif 6 <= self._hour <= 9:
            score = 0.36
            reason = "morning_briefing"
            params = {"trigger": "morning"}

        return {"action": "web_search", "score": score, "reason": reason, "params": params}

    def _score_reflect(self) -> Dict[str, Any]:
        """Utility-Score fuer NPU-Selbstreflexion (DeepSeek R1)."""
        score = 0.0
        reason = ""

        # Hohe Tension → warum bin ich angespannt?
        if self._tension > 0.7:
            score = 0.45
            reason = "high_tension_introspection"
        # Shadow-Zone → was passiert mit mir?
        elif self._dominance < -0.15:
            score = 0.40
            reason = "shadow_zone_awareness"
        # Niemand da + ruhig → Leerlauf-Kontemplation
        elif self._activity == "away" and self._tension < 0.3:
            score = 0.38
            reason = "idle_contemplation"
        # Allein/Arbeiten + wenig Engagement → periodische Reflexion
        elif self._activity in ("alone", "working") and self._engagement < 0.4:
            score = 0.33
            reason = "periodic_reflection"

        return {"action": "reflect", "score": score, "reason": reason}

    # =====================================================================
    # Public API
    # =====================================================================

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
        logger.info(f"[DECISION] Engine {'aktiviert' if value else 'deaktiviert'}")

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "last_decision": self._last_decision,
                "mood": self._mood,
                "activity": self._activity,
                "tension": round(self._tension, 3),
                "cooldowns": {
                    k: round(max(0, COOLDOWNS[k] - (time.time() - self._last_actions.get(k, 0))), 1)
                    for k in COOLDOWNS
                },
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[DecisionEngine] = None
_instance_lock = threading.Lock()


def get_decision_engine() -> DecisionEngine:
    """Singleton-Zugriff auf Decision Engine."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DecisionEngine()
    return _instance
