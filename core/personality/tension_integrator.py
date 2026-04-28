#!/usr/bin/env python3
"""
M.O.L.O.C.H. Tension Integrator — Gate-3 Awareness → CoreIntegrator Bridge
=============================================================================

Empfaengt Gate-3 Events (context_update, activity_changed, motion_state_changed)
und mappt sie auf CoreIntegrator Inputs (tension/dominance Deltas).

Erweitert den bestehenden CoreIntegrator NICHT — fuettert ihn nur mit
neuen Signalen aus den Awareness-Modulen.

Mapping:
  - context_update Score hoch → tension sinkt (alles gut)
  - context_update alertness hoch → tension steigt (Wachsamkeit)
  - activity party → dominance hoch (Energie)
  - activity alone → dominance leicht negativ (ruhig)
  - motion approaching → tension leicht hoch (Aufmerksamkeit)

Singleton: get_tension_integrator()
Gate 4: Emergent Personality
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochTensionIntegrator")

# Gewichtung der Awareness-Signale auf CoreIntegrator
CONTEXT_SCORE_TENSION_WEIGHT = -0.05     # Hoher Score → Tension sinkt
ALERTNESS_TENSION_WEIGHT = 0.08          # Hohe Alertness → Tension steigt
ACTIVITY_DOMINANCE_MAP = {
    "party": 0.15,           # Party → Dominance hoch (Energie/Guardian)
    "conversation": 0.08,    # Gespraech → leicht Guardian
    "working": 0.03,         # Arbeiten → neutral-positiv
    "alone": -0.05,          # Allein → leicht Shadow
    "away": -0.02,           # Weg → minimal Shadow-Drift
}
MOTION_TENSION_MAP = {
    "approaching": 0.04,     # Jemand kommt → leichte Wachsamkeit
    "leaving": -0.02,        # Jemand geht → Entspannung
    "walking": 0.01,         # Bewegung → minimal
    "stationary": 0.0,       # Still → kein Effekt
}

# Beleidigung-Keywords (Deutsch + Englisch) — Tension-Spike bei verbaler Aggression
_RUDENESS_KEYWORDS = [
    "blöd", "dumm", "scheiß", "idiot", "nutzlos", "kaputt", "schrott", "müll",
    "bescheuert", "depp", "doof", "schwachsinn", "mist", "dreck", "arschloch",
    "wichser", "hurensohn", "vollidiot", "trottel", "spacken",
    "stupid", "useless", "trash", "garbage", "broken", "crap", "fuck", "shit",
    "asshole", "idiot", "moron", "dumbass",
]
# Rate-Limiting: Minimum Sekunden zwischen Rudeness-Spikes
_RUDENESS_COOLDOWN_S = 10.0

# Anger-Floor: Tension darf nach Beleidigung nicht sofort wieder sinken
_ANGER_FLOOR_DURATION_S = 45.0          # Wie lange Zorn-Basis aktiv
_ANGER_SUSTAIN_INTERVAL_S = 3.0         # Re-Injektions-Takt (Sekunden)

# Besaenftigung: Keywords die Tension senken
_APPEASEMENT_KEYWORDS: List[str] = [
    "sorry", "entschuldigung", "tut mir leid", "bitte", "danke", "schön",
    "toll", "super", "gut gemacht", "prima", "klasse", "wunderbar",
    "ich mag dich", "du bist gut", "respekt", "brav", "okay okay",
    "alles gut", "peace", "calm down", "beruhig dich",
]
_APPEASEMENT_BASE_BOOST = -0.25         # Basis-Senkung bei Besaenftigung
_APPEASEMENT_RESISTANCE_DURING_ANGER = 0.25  # Nur 25% Wirkung bei aktivem Floor


class TensionIntegrator:
    """Bridge zwischen Gate-3 Awareness und CoreIntegrator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._core_integrator = None
        self._last_context_score = 0.5
        self._last_alertness = 0.2
        self._last_activity = "away"
        self._last_motion = "stationary"
        self._last_rudeness_ts = 0.0
        self._last_rudeness_boost = 0.0
        # Anger-Floor State
        self._anger_floor_until = 0.0
        self._anger_floor_value = 0.0
        self._anger_sustain_thread: Optional[threading.Thread] = None
        # Besaenftigung State
        self._last_appeasement_boost = 0.0
        # Unknown-Person State (Phase 0c — Bug 2 Fix, v2)
        self._last_unknown_face_ts = 0.0
        self._unknown_face_cooldown_s = 5.0  # max 1 Push pro 5s
        self._last_owner_ts = 0.0            # owner_detected Tracking
        self._owner_grace_s = 3.0            # 3s Karenz nach Owner-Detection

    def set_core_integrator(self, ci):
        """CoreIntegrator-Referenz setzen (lazy init) + Event-Subscriptions."""
        self._core_integrator = ci
        # Whisper-Rudeness Subscription — self-subscribe statt Service-Verdrahtung
        try:
            from core.moloch_event_bus import get_event_bus
            bus = get_event_bus()
            bus.subscribe("whisper.result", self.on_whisper_result, priority=5)
            logger.info("[TENSION] whisper.result Subscription aktiv (Rudeness-Detection)")
            # Phase 0c: Unbekannte Person → Tension-Push (Shadow-Schwelle erreichen)
            bus.subscribe("perception.face_confirmed", self.on_face_confirmed, priority=5)
            bus.subscribe("perception.owner_detected", self.on_owner_detected, priority=5)
            logger.info("[TENSION] perception.face_confirmed + owner_detected Subscriptions aktiv (Unknown-Person v2)")
        except Exception as e:
            logger.warning(f"[TENSION] Subscription fehlgeschlagen: {e}")

    def on_context_update(self, event: Dict[str, Any]):
        """Event-Handler fuer context_update Events.

        Mappt Score und Alertness auf Tension-Deltas.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        score = payload.get("score", 0.5)
        alertness = payload.get("alertness", 0.2)

        with self._lock:
            self._last_context_score = score
            self._last_alertness = alertness

        # Score → Tension: hoher Score = alles gut = Tension sinkt
        tension_delta = (score - 0.5) * CONTEXT_SCORE_TENSION_WEIGHT
        # Alertness → Tension: direkte Zuordnung
        tension_delta += alertness * ALERTNESS_TENSION_WEIGHT

        # Via update_input an CoreIntegrator — positiv = tension steigt
        # GEDAEMPFT: tension_delta direkt als conflict_input erzeugt Feedback-Loop!
        # Stattdessen: nur starke Deltas (>0.3) als schwachen Input weitergeben
        if tension_delta > 0.3:
            self._core_integrator.update_input("awareness", "conflict_input", min(0.2, tension_delta * 0.3))
        elif tension_delta < -0.1:
            self._core_integrator.update_input("awareness", "respect_score", min(0.3, abs(tension_delta)))

    def on_activity_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer activity_changed Events.

        Mappt Activity auf Dominance-Shifts.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        activity = payload.get("activity", "away")

        with self._lock:
            self._last_activity = activity

        dominance_delta = ACTIVITY_DOMINANCE_MAP.get(activity, 0.0)
        if dominance_delta > 0:
            self._core_integrator.update_input("awareness", "markus_recognized", dominance_delta)
        elif dominance_delta < 0:
            self._core_integrator.update_input("awareness", "unknown_person", abs(dominance_delta))

    def on_motion_state_changed(self, event: Dict[str, Any]):
        """Event-Handler fuer motion_state_changed Events.

        Mappt Motion auf Tension-Shifts.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        motion = payload.get("state", "stationary")

        with self._lock:
            self._last_motion = motion

        tension_delta = MOTION_TENSION_MAP.get(motion, 0.0)
        if tension_delta > 0:
            self._core_integrator.update_input("awareness", "person_detected", tension_delta)
        elif tension_delta < 0:
            self._core_integrator.update_input("awareness", "respect_score", abs(tension_delta))

    def on_owner_detected(self, event: Dict[str, Any]):
        """Phase 0c v3 — Owner-Detection-Timestamp tracken (defensiv).

        perception.owner_detected feuert nur bei Sim >= arcface_thresh (0.70).
        face_id im Status wird aber bei niedrigerer Schwelle gesetzt. Daher ist
        dieser Event ein zusaetzliches Signal, primaer wird face_id direkt aus
        moloch_status.json gelesen.
        """
        with self._lock:
            self._last_owner_ts = time.time()

    def on_face_confirmed(self, event: Dict[str, Any]):
        """Phase 0c v3 — Unbekannte Person erkannt → Tension-Push (Shadow-Schwelle).

        SCRFD detektiert Gesicht. Identitaets-Pruefung ueber moloch_status.json
        (face_id Feld) als Single-Source-of-Truth.

        Wenn face_id eine bekannte Identitaet enthaelt (nicht None/'unknown'/
        'Unbekannt'), kein Push.

        v1 nutzte similarity<0.45 — triggerte faelschlich bei Markus-Sim 0.39-0.45.
        v2 nutzte owner_detected-Event — feuert nur bei Sim>=0.70, zu restriktiv.
        v3: face_id-Check ist die Quelle der Wahrheit.

        Push-Wert: unknown_person=0.5 -> +0.2 Tension-Delta (Multiplier 0.4).
        """
        if not self._core_integrator:
            return

        now = time.time()
        with self._lock:
            if now - self._last_unknown_face_ts < self._unknown_face_cooldown_s:
                return  # Cooldown

        # Single-Source-of-Truth: markus_recognized im CoreIntegrator
        # (vermeidet Race-Condition mit moloch_status.json das verzoegert
        # geschrieben wird vom Poll-Thread)
        try:
            ci_inputs = getattr(self._core_integrator, "_inputs", {})
            for source_data in ci_inputs.values():
                if source_data.get("markus_recognized", 0.0) > 0.1:
                    return  # Markus erkannt -> kein Push
        except Exception:
            pass  # CoreIntegrator-State nicht lesbar -> defensiv weiter

        with self._lock:
            self._last_unknown_face_ts = now

        payload = event.get("payload", {})
        similarity = payload.get("similarity", 0.0)

        # Unknown-Person-Push: 0.5 -> +0.2 Tension-Delta (CoreIntegrator-Multiplier)
        self._core_integrator.update_input("awareness", "unknown_person", 0.5)
        logger.info(
            f"[TENSION] Unknown person detected (sim={similarity:.2f}, "
            f"face_id-check passed) -> unknown_person=0.5 push (Shadow-Trigger)"
        )

    # ================================================================
    # WHISPER RUDENESS DETECTION — Tension-Spike bei Beleidigungen
    # ================================================================

    def on_whisper_result(self, event: Dict[str, Any]):
        """Event-Handler fuer whisper.result Events.

        Prueft transkribierten Text auf Beleidigungen und erhoeht Tension.
        Rate-Limited: max 1 Spike pro _RUDENESS_COOLDOWN_S Sekunden.
        """
        if not self._core_integrator:
            return

        payload = event.get("payload", {})
        text = payload.get("text", "")
        if not text or len(text) < 3:
            return

        # Besaenftigung pruefen (unabhaengig von Rudeness)
        self._check_appeasement(text)

        boost = self._detect_rudeness(text)
        if boost <= 0.0:
            return

        now = time.time()
        with self._lock:
            # Rate-Limiting — kein Dauerfeuer
            if now - self._last_rudeness_ts < _RUDENESS_COOLDOWN_S:
                logger.debug(f"[TENSION] Rudeness cooldown aktiv, ignoriere ({boost:.2f})")
                return
            self._last_rudeness_ts = now
            self._last_rudeness_boost = boost

        # Tension-Spike via CoreIntegrator — conflict_input erhoeht Tension
        self._core_integrator.update_input("voice", "conflict_input", boost)
        logger.info(f"[TENSION] Rudeness erkannt! Boost={boost:.2f} Text='{text[:50]}'")

        # Character Journal: Beleidigung als charakter-formenden Event protokollieren
        try:
            from core.memory.character_journal import get_journal
            get_journal().write_event(
                type="tension",
                interpretation="Beleidigung erkannt",
                tension_delta=boost,
                context=f"text='{text[:60]}'",
                tags=["rudeness"],
            )
        except Exception as e:
            logger.debug(f"[TENSION] Journal rudeness-hook: {e}")

        # Anger-Floor starten — Tension darf nicht sofort wieder sinken
        with self._lock:
            self._anger_floor_until = now + _ANGER_FLOOR_DURATION_S
            self._anger_floor_value = boost * 0.8
            # Nur einen Sustain-Thread gleichzeitig
            if self._anger_sustain_thread is None or not self._anger_sustain_thread.is_alive():
                t = threading.Thread(target=self._sustain_anger_floor, daemon=True,
                                     name="AngerFloor")
                self._anger_sustain_thread = t
                t.start()
        return  # fruehe Rueckkehr — Appeasement wird separat geprueft

    def _sustain_anger_floor(self):
        """Background-Thread: re-injiziert conflict_input waehrend Anger-Floor aktiv."""
        while True:
            time.sleep(_ANGER_SUSTAIN_INTERVAL_S)
            with self._lock:
                remaining = self._anger_floor_until - time.time()
                if remaining <= 0:
                    logger.info("[TENSION] Anger-Floor abgelaufen")
                    return
                # Linear fallend: 100% am Start → 0% am Ende
                ratio = remaining / _ANGER_FLOOR_DURATION_S
                inject = self._anger_floor_value * ratio
            if self._core_integrator and inject > 0.01:
                self._core_integrator.update_input("voice", "conflict_input", inject)
                logger.debug(f"[TENSION] Anger-Floor inject={inject:.3f} remaining={remaining:.1f}s")

    # ================================================================
    # BESAENFTIGUNG (Appeasement) — nette Worte senken Tension
    # ================================================================

    def _check_appeasement(self, text: str):
        """Prueft Text auf Besaenftigung und senkt Tension entsprechend."""
        boost = self._detect_appeasement(text)
        if boost >= 0.0:
            return
        # Bei aktivem Anger-Floor: Wirkung stark reduziert
        with self._lock:
            anger_active = time.time() < self._anger_floor_until
        if anger_active:
            effective = boost * _APPEASEMENT_RESISTANCE_DURING_ANGER
            logger.info(f"[TENSION] Besaenftigung bei aktivem Zorn: {boost:.2f} → {effective:.2f} (25%)")
            # Starke Besaenftigung kann Anger-Floor vorzeitig beenden
            if boost <= -0.3:
                with self._lock:
                    self._anger_floor_until = min(self._anger_floor_until,
                                                   time.time() + 10.0)
                logger.info("[TENSION] Starke Besaenftigung — Anger-Floor verkuerzt auf 10s")
            boost = effective
        with self._lock:
            self._last_appeasement_boost = boost
        # Negativer Wert → respect_score erhoeht → Tension sinkt
        self._core_integrator.update_input("voice", "respect_score", abs(boost))
        logger.info(f"[TENSION] Besaenftigung erkannt: boost={boost:.2f} Text='{text[:50]}'")

        # Character Journal: Besaenftigung als charakter-formenden Event protokollieren
        try:
            from core.memory.character_journal import get_journal
            get_journal().write_event(
                type="tension",
                interpretation="Besaenftigung erkannt",
                tension_delta=boost,
                context=f"text='{text[:60]}'",
                tags=["appeasement"],
            )
        except Exception as e:
            logger.debug(f"[TENSION] Journal appeasement-hook: {e}")

    def _detect_appeasement(self, text: str) -> float:
        """Gibt negativen Boost zurueck (-0.2 bis -0.3) bei Besaenftigung, sonst 0.0."""
        text_lower = text.lower()
        hits = sum(1 for kw in _APPEASEMENT_KEYWORDS if kw in text_lower)
        if hits == 0:
            return 0.0
        elif hits == 1:
            return -0.2
        else:
            return max(-0.3, -0.2 - (hits - 1) * 0.05)

    def _detect_rudeness(self, text: str) -> float:
        """Gibt Tension-Boost zurueck: 0.0 (keine Beleidigung) bis 0.8 (massive Beleidigung)."""
        text_lower = text.lower()
        hits = sum(1 for kw in _RUDENESS_KEYWORDS if kw in text_lower)
        if hits == 0:
            return 0.0
        elif hits == 1:
            return 0.3
        else:
            return min(0.5 + (hits - 2) * 0.1, 0.8)

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer Debugging/IPC."""
        with self._lock:
            now = time.time()
            anger_remaining = max(0.0, self._anger_floor_until - now)
            return {
                "context_score": round(self._last_context_score, 3),
                "alertness": round(self._last_alertness, 3),
                "activity": self._last_activity,
                "motion": self._last_motion,
                "last_rudeness_boost": round(self._last_rudeness_boost, 3),
                "anger_floor_active": anger_remaining > 0,
                "anger_floor_remaining_s": round(anger_remaining, 1),
                "last_appeasement_boost": round(self._last_appeasement_boost, 3),
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[TensionIntegrator] = None
_instance_lock = threading.Lock()


def get_tension_integrator() -> TensionIntegrator:
    """Singleton-Zugriff auf Tension Integrator."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TensionIntegrator()
    return _instance
