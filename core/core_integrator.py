#!/usr/bin/env python3
"""
M.O.L.O.C.H. Core Integrator — Zentrales Zustandsmodell
=========================================================

3 State-Achsen (alle 0.0 - 1.0):
  1. TENSION   — Anspannung (steuert Personality Zone, Voice Intensity)
  2. ATTENTION — Aufmerksamkeit (steuert Kamera-Stabilitaet, LED Feedback)
  3. PRESENCE  — Praesenz (steuert spontane Kommentare, Ambient-Verhalten)

REGEL: Module beeinflussen NUR den Core State.
       Module loesen NIEMALS direkt Aktionen aus.
       Der Integrator berechnet Effekte, Consumer lesen sie ab.

Tick-Rate: 1 Hz (1x pro Sekunde State neu berechnen)
Thread-safe: Lock fuer jeden State-Zugriff.
Logging: State-Changes > 0.1 Delta werden geloggt.
"""

import time
import json
import os
import threading
import logging
from datetime import datetime
from typing import Dict, Optional, List

_logger = logging.getLogger("CoreIntegrator")

# Status-Datei in Shared Memory (Panel IPC)
_STATUS_PATH = "/dev/shm/moloch_status.json"


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Wert auf [lo, hi] begrenzen."""
    return max(lo, min(hi, val))


class CoreIntegrator:
    """
    Zentrales Zustandsmodell mit 3 Achsen.

    Module fuettern Inputs via update_input().
    Consumer lesen State via get_state() / get_effects() / get_personality_zone().
    Der Integrator-Thread berechnet 1x/s den neuen State.
    """

    # --- Decay-Raten pro Tick (1 Hz) ---
    DECAY_TENSION = 0.95
    DECAY_ATTENTION = 0.98
    DECAY_PRESENCE = 0.99

    # --- Personality-Zone-Schwellen ---
    TENSION_GUARDIAN_MAX = 0.4      # < 0.4 = Guardian
    TENSION_SHADOW_MAX = 0.75       # 0.4 - 0.75 = Shadow
    # > 0.75 = Berserker (auto-decay zurueck)

    # --- Input-Gewichte fuer jede Achse ---
    # Tension-Inputs
    TENSION_WEIGHTS = {
        "respect_score": -0.3,          # Hoher Respekt -> WENIGER Tension
        "disrespect_spike": 0.8,        # Respektlosigkeit -> Berserker-Spike
        "system_load": 0.15,            # CPU/RAM Auslastung
        "conflict_input": 0.5,          # Unbekannte Person, Alarm etc.
        "environmental_stress": 0.2,    # Laerm, Temperatur etc.
        "unknown_person": 0.4,          # Unbekannter erkannt
        "alarm_active": 0.9,            # Alarm -> maximale Tension
    }

    # Attention-Inputs
    ATTENTION_WEIGHTS = {
        "teach_mode": 0.7,              # Lern-Modus aktiv
        "proximity": 0.3,               # Naehere Person -> mehr Attention
        "voice_activity": 0.6,          # Jemand spricht
        "face_confidence": 0.5,         # Gesicht klar erkannt
        "face_detected": 0.4,           # Ueberhaupt ein Gesicht da
        "person_detected": 0.2,         # Person im Bild
        "markus_recognized": 0.6,       # Markus erkannt -> volle Aufmerksamkeit
    }

    # Presence-Inputs
    PRESENCE_WEIGHTS = {
        "time_since_interaction": -0.3,  # Lange nichts passiert -> weniger praesent
        "user_proximity": 0.4,           # Jemand in der Naehe
        "festival_mode": 0.8,            # WGT/Festival -> maximale Praesenz
        "manual_activation": 1.0,        # Manuell aktiviert
        "voice_activity": 0.3,           # Stimme gehoert -> mehr Praesenz
    }

    def __init__(self):
        # --- State-Achsen ---
        self._tension = 0.0
        self._attention = 0.0
        self._presence = 0.0

        # --- Vorheriger State (fuer Delta-Logging) ---
        self._prev_tension = 0.0
        self._prev_attention = 0.0
        self._prev_presence = 0.0

        # --- Input-Puffer: {source: {key: value}} ---
        # Jede Quelle (Modul) hat eigenen Namespace
        self._inputs: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        # --- Effekt-Cache (wird pro Tick berechnet) ---
        self._effects: Dict[str, float] = {}

        # --- Thread-Steuerung ---
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

        # --- Phase 3: Cross-Model Correlation ---
        # Perception Buffer Referenz (lazy, vermeidet zirkulaere Imports)
        self._perception_buffer = None
        # Letzte Trend-Daten (1x/s aktualisiert)
        self._last_trends: Dict = {}

        _logger.info("[CORE] CoreIntegrator initialisiert")

    # =========================================================================
    # Public API
    # =========================================================================

    def update_input(self, source: str, key: str, value: float):
        """Module fuettern Inputs hierueber.

        Args:
            source: Modul-Name (z.B. "perception", "voice", "panel")
            key: Input-Key (z.B. "face_detected", "voice_activity")
            value: Wert (0.0 - 1.0, wird geclamped)
        """
        with self._lock:
            if source not in self._inputs:
                self._inputs[source] = {}
            self._inputs[source][key] = _clamp(value)

    def update_inputs(self, source: str, data: Dict[str, float]):
        """Mehrere Inputs auf einmal (Batch)."""
        with self._lock:
            if source not in self._inputs:
                self._inputs[source] = {}
            for key, value in data.items():
                self._inputs[source][key] = _clamp(value)

    def get_state(self) -> Dict[str, float]:
        """Aktueller State aller 3 Achsen."""
        with self._lock:
            return {
                "tension": round(self._tension, 4),
                "attention": round(self._attention, 4),
                "presence": round(self._presence, 4),
            }

    def get_personality_zone(self) -> str:
        """Aktuelle Personality-Zone basierend auf Tension.

        Returns:
            "guardian" | "shadow" | "berserker"
        """
        with self._lock:
            t = self._tension
        if t < self.TENSION_GUARDIAN_MAX:
            return "guardian"
        elif t < self.TENSION_SHADOW_MAX:
            return "shadow"
        else:
            return "berserker"

    def get_effects(self) -> Dict[str, float]:
        """Alle aktuellen Effekt-Werte (abgeleitet aus State).

        Returns:
            Dict mit Effekten wie:
              - voice_intensity (0.0-1.0)
              - response_latency (0.0-1.0, hoeher = schneller antworten)
              - micro_ptz_movement (0.0-1.0)
              - language_sharpness (0.0-1.0)
              - camera_stability (0.0-1.0, hoeher = ruhiger)
              - led_feedback_frequency (0.0-1.0)
              - speech_focus (0.0-1.0)
              - snapshot_probability (0.0-1.0)
              - spontaneous_comments (0.0-1.0)
              - ambient_ptz_behavior (0.0-1.0)
              - manifestation_intensity (0.0-1.0)
        """
        with self._lock:
            return dict(self._effects)

    def get_tension(self) -> float:
        """Direkt-Zugriff auf Tension (fuer PersonalityEngine Kompatibilitaet)."""
        with self._lock:
            return self._tension

    def get_attention(self) -> float:
        """Direkt-Zugriff auf Attention."""
        with self._lock:
            return self._attention

    def get_presence(self) -> float:
        """Direkt-Zugriff auf Presence."""
        with self._lock:
            return self._presence

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Integrator-Thread starten (1 Hz tick). Laedt persistenten State."""
        if self._running:
            return

        # Persistenten State laden (Langzeitgedaechtnis)
        try:
            from core.longterm_memory import get_memory
            saved = get_memory().load_core_state()
            if saved and saved.get("last_updated"):
                self._tension = float(saved.get("tension", 0.0))
                self._attention = float(saved.get("attention", 0.0))
                self._presence = float(saved.get("presence", 0.0))
                _logger.info(f"[CORE] State aus Langzeitgedaechtnis geladen: "
                             f"T={self._tension:.2f} A={self._attention:.2f} P={self._presence:.2f} "
                             f"(gespeichert: {saved.get('last_updated', '?')})")
        except Exception as e:
            _logger.warning(f"[CORE] Persistenter State nicht verfuegbar: {e}")

        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="CoreIntegrator"
        )
        self._thread.start()
        _logger.info("[CORE] Integrator-Thread gestartet (1 Hz)")

    def stop(self):
        """Integrator-Thread stoppen + State persistent sichern."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        # Letzten State auf Disk sichern
        self._persist_state()
        _logger.info("[CORE] Integrator-Thread gestoppt (State persistent gesichert)")

    # =========================================================================
    # Tick-Loop (1 Hz)
    # =========================================================================

    def _tick_loop(self):
        """Hauptschleife: 1x pro Sekunde State neu berechnen."""
        _persist_counter = 0
        while self._running:
            try:
                self._tick()
                # Alle 60 Ticks (~60 Sekunden): State persistent speichern
                _persist_counter += 1
                if _persist_counter >= 60:
                    _persist_counter = 0
                    self._persist_state()
            except Exception as e:
                _logger.error(f"[CORE] Tick-Fehler: {e}")
            time.sleep(1.0)

    def _get_time_period(self) -> str:
        """Aktuelle Tageszeit-Periode bestimmen."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morgens"
        elif 12 <= hour < 17:
            return "mittags"
        elif 17 <= hour < 22:
            return "abends"
        else:
            return "nachts"

    def _tick(self):
        """Ein Tick: Inputs sammeln, Achsen berechnen, Effekte ableiten."""
        with self._lock:
            # Tageszeit-Input automatisch einspeisen
            hour = datetime.now().hour
            period = self._get_time_period()

            # Zeit-basierte Modifikatoren
            if "time" not in self._inputs:
                self._inputs["time"] = {}

            if period == "nachts":
                # Nachts: Presence auf 0, Tension sinkt
                self._inputs["time"]["time_since_interaction"] = 1.0  # Presence soll sinken
                self._inputs["time"]["user_proximity"] = 0.0
            elif period == "morgens":
                # Morgens: Langsam hochfahren, Guardian bevorzugt
                self._inputs["time"]["environmental_stress"] = 0.0
            elif period == "abends":
                # Abends: Ruhiger, Tension sinkt natuerlich
                self._inputs["time"]["environmental_stress"] = 0.0

            # Snapshot der Inputs (Thread-safe)
            all_inputs = {}
            for source, data in self._inputs.items():
                all_inputs.update(data)

            # --- 1. TENSION berechnen ---
            tension_impulse = 0.0
            for key, weight in self.TENSION_WEIGHTS.items():
                val = all_inputs.get(key, 0.0)
                tension_impulse += val * weight

            # Decay + Impuls
            self._tension = _clamp(
                self._tension * self.DECAY_TENSION + tension_impulse * 0.3
            )

            # Berserker Auto-Decay: schnellerer Zerfall ueber 0.75
            if self._tension > self.TENSION_SHADOW_MAX:
                self._tension *= 0.90  # Zusaetzlicher 10% Decay

            # --- 2. ATTENTION berechnen ---
            attention_impulse = 0.0
            for key, weight in self.ATTENTION_WEIGHTS.items():
                val = all_inputs.get(key, 0.0)
                attention_impulse += val * weight

            self._attention = _clamp(
                self._attention * self.DECAY_ATTENTION + attention_impulse * 0.3
            )

            # --- 3. PRESENCE berechnen ---
            presence_impulse = 0.0
            for key, weight in self.PRESENCE_WEIGHTS.items():
                val = all_inputs.get(key, 0.0)
                presence_impulse += val * weight

            self._presence = _clamp(
                self._presence * self.DECAY_PRESENCE + presence_impulse * 0.3
            )

            # --- Cross-Model Correlation (Phase 3) ---
            self._apply_cross_model_patterns(all_inputs)

            # --- Nachtsperre: Presence auf 0 druecken (22:00-06:00) ---
            if period == "nachts":
                self._presence *= 0.8  # Schneller Decay nachts
                if self._presence < 0.05:
                    self._presence = 0.0

            # --- Abends: Tension sinkt natuerlich schneller ---
            if period == "abends":
                self._tension *= 0.97  # Zusaetzlicher Decay abends

            # --- Effekte ableiten ---
            self._effects = self._compute_effects()

            # --- Delta-Logging ---
            self._tick_count += 1
            d_t = abs(self._tension - self._prev_tension)
            d_a = abs(self._attention - self._prev_attention)
            d_p = abs(self._presence - self._prev_presence)

            if d_t > 0.1 or d_a > 0.1 or d_p > 0.1:
                zone = self._zone_unlocked()
                _logger.info(
                    f"[CORE] State: T={self._tension:.3f} A={self._attention:.3f} "
                    f"P={self._presence:.3f} zone={zone} "
                    f"(dT={d_t:+.3f} dA={d_a:+.3f} dP={d_p:+.3f})"
                )

            # Alle 60 Ticks (1 Minute) State loggen, auch ohne grosse Aenderung
            if self._tick_count % 60 == 0:
                zone = self._zone_unlocked()
                _logger.info(
                    f"[CORE] Heartbeat #{self._tick_count}: "
                    f"T={self._tension:.3f} A={self._attention:.3f} "
                    f"P={self._presence:.3f} zone={zone} "
                    f"inputs={len(all_inputs)}"
                )

            self._prev_tension = self._tension
            self._prev_attention = self._attention
            self._prev_presence = self._presence

    def _zone_unlocked(self) -> str:
        """Personality Zone OHNE Lock (intern, nur innerhalb Lock aufrufen)."""
        if self._tension < self.TENSION_GUARDIAN_MAX:
            return "guardian"
        elif self._tension < self.TENSION_SHADOW_MAX:
            return "shadow"
        return "berserker"

    def _get_buffer_trends(self) -> Dict:
        """Trends aus dem Perception Buffer holen (lazy init)."""
        if self._perception_buffer is None:
            try:
                from core.perception.perception_buffer import get_perception_buffer
                self._perception_buffer = get_perception_buffer()
            except Exception:
                return {}
        try:
            return self._perception_buffer.get_trends()
        except Exception:
            return {}

    def _apply_cross_model_patterns(self, inputs: Dict):
        """Cross-Model Correlation: Muster aus kombinierten Inputs erkennen.

        Nicht mehr einzelne Inputs, sondern Muster:
        - person + unknown_face + close → Tension STEIGT schnell
        - markus + calm_pose + neutral → Tension SINKT
        - unknown_face + high_pose_energy → Tension STEIGT stark
        - markus + happy_emotion → Presence STEIGT
        - niemand + lange_zeit → Attention SINKT, Idle-Mode

        WICHTIG: Wird innerhalb Lock aufgerufen.
        """
        # Trends aus Buffer (1x/s reicht)
        if self._tick_count % 2 == 0:  # Alle 2 Ticks aktualisieren
            self._last_trends = self._get_buffer_trends()
        trends = self._last_trends

        person = inputs.get("person_detected", 0.0) > 0.5
        face = inputs.get("face_detected", 0.0) > 0.5
        markus = inputs.get("markus_recognized", 0.0) > 0.5
        unknown = inputs.get("unknown_person", 0.0) > 0.5
        proximity = inputs.get("proximity", 0.0)
        voice = inputs.get("voice_activity", 0.0) > 0.5

        smoothed_emotion = trends.get("smoothed_emotion")
        pose_energy = trends.get("avg_pose_energy", 0.0)
        approaching = trends.get("approaching", False)
        leaving = trends.get("leaving", False)
        absence = trends.get("absence_duration", 0.0)

        # === Pattern 1: Unbekannter nah dran → Tension BOOST ===
        if person and unknown and proximity > 0.1:
            self._tension = _clamp(self._tension + 0.05)

        # === Pattern 2: Unbekannter + hohe Bewegung → Tension STARK ===
        if unknown and pose_energy > 0.5:
            self._tension = _clamp(self._tension + 0.08)

        # === Pattern 3: Markus + ruhig + neutral → Tension SINKT ===
        if markus and pose_energy < 0.2 and smoothed_emotion in (None, "neutral", "happy"):
            self._tension = _clamp(self._tension - 0.02)

        # === Pattern 4: Markus + happy → Presence BOOST ===
        if markus and smoothed_emotion == "happy":
            self._presence = _clamp(self._presence + 0.03)

        # === Pattern 5: Niemand da + lange Absence → Attention SINKT ===
        if not person and not face and absence > 10:
            self._attention *= 0.95  # Schnellerer Attention-Decay

        # === Pattern 6: Person naehert sich → Attention STEIGT ===
        if approaching and person:
            self._attention = _clamp(self._attention + 0.04)

        # === Pattern 7: Person entfernt sich → Attention sinkt langsam ===
        if leaving:
            self._attention *= 0.97

        # === Pattern 8: Voice + Face → maximale Attention + Presence ===
        if voice and face:
            self._attention = _clamp(self._attention + 0.05)
            self._presence = _clamp(self._presence + 0.03)

    def _compute_effects(self) -> Dict[str, float]:
        """Effekte aus den 3 Achsen ableiten (intern, nur innerhalb Lock aufrufen)."""
        t = self._tension
        a = self._attention
        p = self._presence

        return {
            # --- Tension-Effekte ---
            "voice_intensity": _clamp(0.3 + t * 0.7),
            "response_latency": _clamp(1.0 - t * 0.5),    # Hoehere Tension -> schnellere Antwort
            "micro_ptz_movement": _clamp(t * 0.4),          # Nervoeser bei Tension
            "language_sharpness": _clamp(t * 0.8),          # Schaerferer Ton

            # --- Attention-Effekte ---
            "camera_stability": _clamp(1.0 - a * 0.3),     # Mehr Attention -> weniger Wackeln
            "led_feedback_frequency": _clamp(0.1 + a * 0.9),
            "speech_focus": _clamp(a * 0.8),
            "snapshot_probability": _clamp(a * 0.6),

            # --- Presence-Effekte ---
            "spontaneous_comments": _clamp(p * 0.5),
            "ambient_ptz_behavior": _clamp(p * 0.4),
            "manifestation_intensity": _clamp(0.2 + p * 0.8),
        }

    # =========================================================================
    # Status-Export (fuer /dev/shm/moloch_status.json)
    # =========================================================================

    def get_time_period(self) -> str:
        """Aktuelle Tageszeit als Public API (morgens/mittags/abends/nachts)."""
        return self._get_time_period()

    def get_status_dict(self) -> Dict:
        """Kompaktes Status-Dict fuer SHM-Export."""
        with self._lock:
            result = {
                "tension": round(self._tension, 4),
                "attention": round(self._attention, 4),
                "presence": round(self._presence, 4),
                "zone": self._zone_unlocked(),
                "time_period": self._get_time_period(),
                "effects": {k: round(v, 3) for k, v in self._effects.items()},
                "tick": self._tick_count,
            }
        # Trends ausserhalb des Locks (Buffer hat eigenen Lock)
        if self._last_trends:
            result["trends"] = self._last_trends
        return result

    def _persist_state(self):
        """State auf SSD2 persistent speichern (Langzeitgedaechtnis)."""
        try:
            from core.longterm_memory import get_memory
            state = self.get_state()
            state["personality_zone"] = self.get_personality_zone()
            state["uptime_seconds"] = self._tick_count
            get_memory().save_core_state(state)
        except Exception as e:
            _logger.error(f"[CORE] State-Persistenz fehlgeschlagen: {e}")


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[CoreIntegrator] = None
_instance_lock = threading.Lock()


def get_core_integrator() -> CoreIntegrator:
    """Singleton-Zugriff auf den CoreIntegrator."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CoreIntegrator()
    return _instance
