#!/usr/bin/env python3
"""
M.O.L.O.C.H. Core Integrator v2 — Blueprint 5.2 + 5.3
=========================================================

2 State-Variablen:
  1. TENSION    (0.0 - 1.0)   — Anspannung/Arousal. Spike bei Trigger,
                                 exponentieller Decay (tau=300s).
  2. DOMINANCE  (-1.0 - +1.0) — Persoenlichkeits-Achse.
                                 -1 = Shadow, +1 = Guardian.

Physiologisches Feedback (Blueprint 5.3):
  - CPU-Temperatur: Daempft Tension-Spikes, verlangsamt PTZ bei Hitze.
  - NPU-Last: Leichte Latenz-Variation bei hoher Auslastung.

Zonen-Logik:
  - Guardian:  dominance > +0.15 (mit Hysterese)
  - Shadow:    dominance < -0.15 (mit Hysterese)
  - Berserker: tension > 0.95 UND externer Impulse-Flag. Auto-Reset 10s.

Homoeostatischer Drift: dominance -> +0.5 (0.01/min).
Anti-Complexity: Max 2 State-Variablen, max ±3% Expression-Randomness.

REGEL: Module beeinflussen NUR den Core State.
       Module loesen NIEMALS direkt Aktionen aus.
       Der Integrator berechnet Effekte, Consumer lesen sie ab.

Tick-Rate: 1 Hz (1x pro Sekunde State neu berechnen)
Thread-safe: Lock fuer jeden State-Zugriff.
"""

import math
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Optional

_logger = logging.getLogger("CoreIntegrator")

# Status-Datei in Shared Memory (Panel IPC)
_STATUS_PATH = "/dev/shm/moloch_status.json"
# CPU-Temperatur Sensor
_CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Wert auf [lo, hi] begrenzen."""
    return max(lo, min(hi, val))


class CoreIntegrator:
    """
    Core Integrator v2 — 2-Achsen-Modell mit physiologischem Feedback.

    Module fuettern Inputs via update_input().
    Consumer lesen State via get_state() / get_effects() / get_personality_zone().
    Der Integrator-Thread berechnet 1x/s den neuen State.
    """

    # === Tension Decay ===
    TENSION_TAU = 300.0  # Sekunden, exponentieller Zerfall
    _TENSION_DECAY_FACTOR = math.exp(-1.0 / 300.0)  # ~0.99667 pro Tick

    # === Dominance Drift ===
    DOMINANCE_DRIFT_TARGET = 0.5    # Homoeostatisches Ziel (leicht Guardian)
    DOMINANCE_DRIFT_RATE = 0.05 / 60.0  # 0.05 pro Minute, aufgeloest in 1-Hz-Ticks

    # === Hysterese ===
    ZONE_HYSTERESIS = 0.15  # Mindest-Delta fuer Zone-Wechsel (Dominance-basiert)

    # === Tension-basierte Zone-Hysterese (Gate0 Phase 5) ===
    TENSION_HIGH_THRESHOLD = 0.6   # Ueber diesem Wert -> Shadow
    TENSION_LOW_THRESHOLD = 0.3    # Unter diesem Wert -> Guardian
    TENSION_HYSTERESIS_TIME = 10.0 # Sekunden stabil bevor Zone-Wechsel

    # === Berserker ===
    BERSERKER_TENSION_THRESHOLD = 0.95
    BERSERKER_DURATION = 10.0  # Sekunden bis Auto-Reset
    BERSERKER_DOMINANCE_RESET = 0.2  # Dominance nach Berserker

    # === CPU-Temperatur Normalisierung ===
    CPU_TEMP_MIN = 40.0   # °C -> 0.0
    CPU_TEMP_MAX = 85.0   # °C -> 1.0

    # === Thermal Damping (konfigurierbar via settings.json mpo.thermal_damping_start) ===
    THERMAL_DAMPING_START = 70.0  # °C ab der Tension gedaempft wird

    # === Tension-Inputs (treiben Tension hoch) ===
    TENSION_WEIGHTS = {
        "respect_score": -0.3,          # Hoher Respekt senkt Tension
        "disrespect_spike": 0.8,        # Respektlosigkeit -> Spike
        "conflict_input": 0.5,          # Unbekannte Person, Alarm etc.
        "unknown_person": 0.4,          # Unbekannter erkannt -> Tension steigt
        "person_detected": 0.1,         # Jemand sichtbar -> leichte Wachsamkeit
        "markus_recognized": -0.4,      # Markus erkannt -> Tension faellt aktiv
        "alarm_active": 0.9,            # Alarm -> maximale Tension
        "environmental_stress": 0.2,    # Laerm, Temperatur etc.
        "hardware_pain": 0.7,           # Watchdog: akuter Schmerz (Pipeline, Mic, Netz)
        "system_stress": 0.3,           # Watchdog: chronischer Stress (Temp, RAM, Disk)
        # system_load ENTFERNT: CPU-Last ist keine Bedrohung, hat Tension bei 1.0 fixiert
    }

    # === Dominance-Inputs (positiv=Guardian, negativ=Shadow) ===
    DOMINANCE_WEIGHTS = {
        "markus_recognized": 0.3,       # Markus -> Guardian
        "face_confidence": 0.1,         # Klares Gesicht -> leicht Guardian
        "voice_activity": 0.15,         # Sprachkontakt -> Guardian
        "teach_mode": 0.2,              # Lern-Modus -> Guardian
        "unknown_person": -0.3,         # Unbekannter -> Shadow
        "conflict_input": -0.4,         # Konflikt -> Shadow
        "alarm_active": -0.5,           # Alarm -> Shadow
        "hardware_pain": -0.3,          # Watchdog-Schmerz -> Shadow
        "disrespect_spike": -0.6,       # Respektlosigkeit -> Shadow
        "festival_mode": -0.4,          # WGT -> Shadow
    }

    def __init__(self):
        # === State-Achsen ===
        self._tension = 0.0
        self._dominance = 0.5  # Start: leicht Guardian

        # === Vorheriger State (Delta-Logging) ===
        self._prev_tension = 0.0
        self._prev_dominance = 0.5

        # === Zone mit Hysterese ===
        self._current_zone = "guardian"

        # === Berserker State ===
        self._impulse_flag = False
        self._berserker_active = False
        self._berserker_until = 0.0

        # === Owner Override (Chat/Voice Identifikation) ===
        self._owner_confirmed = False
        self._owner_confirmed_until = 0.0  # monotonic timestamp

        # === Presence/Wohlbefinden — waechst bei dauerhafter Owner-Praesenz ===
        self._presence = 0.0       # 0.0 = niemand da, 1.0 = Owner seit langem hier
        self._presence_grow_rate = 0.005   # +0.005/Tick (~0.3/Min bei Owner-Erkennung)
        self._presence_decay_rate = 0.01   # -0.01/Tick (~0.6/Min ohne Owner)

        # === Tension-basierte Zone-Hysterese (Gate0 Phase 5) ===
        self._tension_high_since: Optional[float] = None  # monotonic, wann tension > 0.6
        self._tension_low_since: Optional[float] = None   # monotonic, wann tension < 0.3

        # === Physiologisches Feedback ===
        self._cpu_temp_raw = 0.0        # Celsius
        self._cpu_temp_normalized = 0.0  # 0.0-1.0
        self._npu_load = 0.0            # 0.0-1.0

        # === Input-Puffer: {source: {key: value}} ===
        self._inputs: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

        # === Effekt-Cache ===
        self._effects: Dict[str, float] = {}

        # === Thread-Steuerung ===
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0

        # === Perception Buffer (lazy init) ===
        self._perception_buffer = None
        self._last_trends: Dict = {}

        _logger.info("[CORE] CoreIntegrator v2 initialisiert (tension + dominance)")

    # =========================================================================
    # Public API
    # =========================================================================

    def feed_event(self, event_type: str, weight: float = 0.1):
        """Event als Tension/Dominance-Input einspeisen.

        Convenience-Alias fuer update_input(). Mappt event_type direkt
        auf den passenden Input-Key (z.B. "markus_recognized", "unknown_person").

        Args:
            event_type: Key aus TENSION_WEIGHTS oder DOMINANCE_WEIGHTS
            weight: Staerke des Inputs (0.0-1.0)
        """
        self.update_input("event", event_type, weight)

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
            prev = self._inputs[source].get(key)
            self._inputs[source][key] = _clamp(value)

        # Character Journal: Alarm-Edge ('protective'-Event)
        if key == "alarm_active":
            try:
                curr_alarm = _clamp(value) > 0.5
                prev_alarm = (prev or 0.0) > 0.5
                if curr_alarm != prev_alarm:
                    from core.memory.character_journal import get_journal
                    if curr_alarm:
                        _interp = "Alarm aktiviert — Eindringling moeglich"
                        _td, _tags = +0.9, ["alarm", "alert"]
                    else:
                        _interp = "Alarm aufgehoben"
                        _td, _tags = -0.4, ["alarm", "cleared"]
                    get_journal().write_event(
                        type="protective",
                        interpretation=_interp,
                        tension_delta=_td,
                        context=f"source={source}",
                        tags=_tags,
                    )
            except Exception as e:
                _logger.debug(f"Journal protective-hook (alarm): {e}")

    def update_inputs(self, source: str, data: Dict[str, float]):
        """Mehrere Inputs auf einmal (Batch)."""
        with self._lock:
            if source not in self._inputs:
                self._inputs[source] = {}
            for key, value in data.items():
                self._inputs[source][key] = _clamp(value)

    def set_impulse_flag(self):
        """Externen Impulse setzen — Voraussetzung fuer Berserker-Aktivierung.

        Muss von aussen gesetzt werden (z.B. Alarm, schwere Respektlosigkeit).
        Berserker triggert NUR wenn impulse UND tension > 0.95.
        """
        with self._lock:
            self._impulse_flag = True
            _logger.info("[CORE] Impulse-Flag gesetzt")

    # === Owner Override (Chat/Voice -> Core State) ===

    OWNER_OVERRIDE_DURATION = 600.0  # 10 Minuten
    OWNER_OVERRIDE_TENSION_DROP = 0.3
    OWNER_OVERRIDE_DOMINANCE_BOOST = 0.3

    def owner_override(self):
        """Owner hat sich per Chat/Voice identifiziert.

        Effekt: Tension sofort -0.3, Dominance Richtung Guardian +0.3.
        Gilt fuer 10 Minuten oder bis naechste positive Face-ID (clear_owner_override).
        """
        with self._lock:
            was_confirmed = self._owner_confirmed
            if self._owner_confirmed:
                _logger.info("[CORE] Owner-Override bereits aktiv, erneuert")
            self._owner_confirmed = True
            self._owner_confirmed_until = time.monotonic() + self.OWNER_OVERRIDE_DURATION
            # Sofortiger Effekt auf State
            self._tension = _clamp(self._tension - self.OWNER_OVERRIDE_TENSION_DROP, lo=-1.0, hi=1.0)
            self._dominance = _clamp(
                self._dominance + self.OWNER_OVERRIDE_DOMINANCE_BOOST, -1.0, 1.0
            )
            _logger.info(
                f"[CORE] Owner-Override AKTIV: T={self._tension:.3f} "
                f"D={self._dominance:+.3f} (gilt {self.OWNER_OVERRIDE_DURATION:.0f}s)"
            )

        # Character Journal: Owner-Confirmation als 'protective'-Event (nur Edge)
        if not was_confirmed:
            try:
                from core.memory.character_journal import get_journal
                get_journal().write_event(
                    type="protective",
                    interpretation="Owner zurueck, Schutz aktiv",
                    tension_delta=-self.OWNER_OVERRIDE_TENSION_DROP,
                    context="owner_override",
                    tags=["guardian", "owner"],
                )
            except Exception as e:
                _logger.debug(f"Journal protective-hook (owner): {e}")

    def clear_owner_override(self):
        """Override loeschen (z.B. nach positiver Face-ID durch Vision)."""
        with self._lock:
            if self._owner_confirmed:
                self._owner_confirmed = False
                self._owner_confirmed_until = 0.0
                _logger.info("[CORE] Owner-Override geloescht (Face-ID bestaetigt)")

    def is_owner_confirmed(self) -> bool:
        """Ist Owner-Override gerade aktiv?"""
        with self._lock:
            return self._owner_confirmed

    # === Calm Down (Beruhigung per Sprache/Text) ===

    CALM_DOWN_TENSION_DROP = 0.3

    def calm_down(self):
        """Beruhigung per Sprache/Text — Tension sofort senken.

        Effekt: Tension -0.3. Kein Dominance-Shift.
        """
        with self._lock:
            old_t = self._tension
            self._tension = _clamp(self._tension - self.CALM_DOWN_TENSION_DROP, lo=-1.0, hi=1.0)
            _logger.info(
                f"[CORE] Calm-Down: T={old_t:.3f} -> {self._tension:.3f} "
                f"(delta={self._tension - old_t:+.3f})"
            )

    def set_npu_load(self, load: float):
        """NPU-Auslastung setzen (0.0-1.0). Aus Model Health Monitoring."""
        with self._lock:
            self._npu_load = _clamp(load)

    def get_state(self) -> Dict[str, float]:
        """Aktueller State der 2 Achsen + CPU-Temp."""
        with self._lock:
            return {
                "tension": round(self._tension, 4),
                "dominance": round(self._dominance, 4),
                "cpu_temp": round(self._cpu_temp_normalized, 4),
            }

    def get_personality_zone(self) -> str:
        """Aktuelle Personality-Zone (mit Hysterese).

        Returns:
            "guardian" | "shadow" | "berserker"
        """
        with self._lock:
            return self._current_zone

    def get_effects(self) -> Dict[str, float]:
        """Alle aktuellen Effekt-Werte (abgeleitet aus State).

        Returns:
            Dict mit Effekten:
              - voice_intensity, response_latency, language_sharpness
              - micro_ptz_movement, camera_stability
              - led_feedback_frequency, snapshot_probability
              - guardian_influence, shadow_influence
              - speech_focus, spontaneous_comments
              - manifestation_intensity, ambient_ptz_behavior
              - jitter_damping, ptz_speed_factor (CPU-Temp)
              - cpu_temp, npu_load
        """
        with self._lock:
            return dict(self._effects)

    def get_tension(self) -> float:
        """Direkt-Zugriff auf Tension (0.0-1.0)."""
        with self._lock:
            return self._tension

    def get_dominance(self) -> float:
        """Direkt-Zugriff auf Dominance (-1.0 bis +1.0)."""
        with self._lock:
            return self._dominance

    def get_cpu_temp(self) -> float:
        """CPU-Temperatur normalisiert (0.0-1.0). 40°C=0.0, 85°C=1.0."""
        with self._lock:
            return self._cpu_temp_normalized

    def get_cpu_temp_celsius(self) -> float:
        """CPU-Temperatur in Celsius."""
        with self._lock:
            return self._cpu_temp_raw

    # --- Kompatibilitaets-Shims (Legacy-Module) ---

    def get_attention(self) -> float:
        """DEPRECATED: Abgeleitet aus tension + dominance."""
        with self._lock:
            return _clamp(self._tension * 0.7 + 0.3 * abs(self._dominance))

    def get_presence(self) -> float:
        """DEPRECATED: Abgeleitet aus dominance + tension."""
        with self._lock:
            return _clamp(abs(self._dominance) * 0.6 + self._tension * 0.3)

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
                # Migration: alte States hatten attention/presence statt dominance
                if "dominance" in saved:
                    self._dominance = float(saved.get("dominance", 0.5))
                else:
                    self._dominance = 0.5
                    _logger.info("[CORE] Altes State-Format erkannt, dominance auf 0.5 gesetzt")
                zone = saved.get("personality_zone", "guardian")
                if zone in ("guardian", "shadow"):
                    self._current_zone = zone
                else:
                    self._current_zone = "guardian"
                _logger.info(f"[CORE] State geladen: T={self._tension:.2f} "
                             f"D={self._dominance:+.2f} zone={self._current_zone} "
                             f"(gespeichert: {saved.get('last_updated', '?')})")
        except Exception as e:
            _logger.warning(f"[CORE] Persistenter State nicht verfuegbar: {e}")

        self._running = True
        self._thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="CoreIntegrator"
        )
        self._thread.start()
        _logger.info("[CORE] Integrator v2 gestartet (1 Hz)")

    def stop(self):
        """Integrator-Thread stoppen + State persistent sichern."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._persist_state()
        _logger.info("[CORE] Integrator gestoppt (State gesichert)")

    # =========================================================================
    # Tick-Loop (1 Hz)
    # =========================================================================

    def _tick_loop(self):
        """Hauptschleife: 1x pro Sekunde State neu berechnen."""
        _persist_counter = 0
        while self._running:
            try:
                self._tick()
                _persist_counter += 1
                if _persist_counter >= 60:
                    _persist_counter = 0
                    self._persist_state()
            except Exception as e:
                _logger.error(f"[CORE] Tick-Fehler: {e}")
            time.sleep(1.0)

    def _read_cpu_temp(self) -> tuple:
        """CPU-Temperatur lesen. Returns (celsius, normalized 0-1)."""
        try:
            with open(_CPU_TEMP_PATH) as f:
                raw = int(f.read().strip())
            celsius = raw / 1000.0
            normalized = _clamp(
                (celsius - self.CPU_TEMP_MIN) / (self.CPU_TEMP_MAX - self.CPU_TEMP_MIN)
            )
            return celsius, normalized
        except Exception:
            return 0.0, 0.0

    def _get_time_period(self) -> str:
        """Aktuelle Tageszeit-Periode bestimmen."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morgens"
        elif 12 <= hour < 17:
            return "mittags"
        elif 17 <= hour < 22:
            return "abends"
        return "nachts"

    def _tick(self):
        """Ein Tick: Inputs sammeln, Achsen berechnen, Effekte ableiten."""
        with self._lock:
            now = time.monotonic()
            period = self._get_time_period()

            # === CPU-Temperatur lesen (alle 5 Ticks = 5s) ===
            if self._tick_count % 5 == 0:
                self._cpu_temp_raw, self._cpu_temp_normalized = self._read_cpu_temp()

            # === Tageszeit-Input automatisch einspeisen ===
            if "time" not in self._inputs:
                self._inputs["time"] = {}
            if period == "nachts":
                self._inputs["time"]["environmental_stress"] = 0.0

            # Alle Inputs zusammenfuehren
            all_inputs = {}
            for source, data in self._inputs.items():
                all_inputs.update(data)

            # =============================================================
            # 1. TENSION berechnen (exponentieller Decay + Impulse)
            # =============================================================
            tension_impulse = 0.0
            _debug_parts = []
            for key, weight in self.TENSION_WEIGHTS.items():
                val = all_inputs.get(key, 0.0)
                contrib = val * weight
                tension_impulse += contrib
                if abs(contrib) > 0.001:
                    _debug_parts.append(f"{key}={val:.2f}*{weight:+.1f}={contrib:+.3f}")
            # Debug: alle 60 Ticks (1 Min) die Tension-Inputs loggen
            if self._tick_count % 60 == 0 and _debug_parts:
                _logger.info(f"[TENSION-DEBUG] Impulse={tension_impulse:+.3f} | {' | '.join(_debug_parts)}")

            # CPU-Temperatur Daempfung: Spikes -20% ab THERMAL_DAMPING_START
            _thermal_norm = (self.THERMAL_DAMPING_START - self.CPU_TEMP_MIN) / (self.CPU_TEMP_MAX - self.CPU_TEMP_MIN)
            if self._cpu_temp_normalized > _thermal_norm:
                tension_impulse *= 0.8

            # Exponentieller Decay (tau=300s)
            self._tension *= self._TENSION_DECAY_FACTOR

            # Impuls addieren — Tension darf negativ werden (Wohlbefinden)
            # -1.0 = maximales Wohlbefinden, 0.0 = neutral, +1.0 = maximaler Stress
            # Range [-1.0, +1.0] gilt fuer ALLE _clamp(self._tension ...)-Calls in dieser Datei.
            # Owner-Detection (Zeile 261) + Calm-Down (308) + Pattern-Matching (750ff)
            # MUESSEN explizit lo=-1.0 angeben — sonst rasiert _clamp-Default (lo=0.0)
            # vorhandenes Wohlbefinden auf 0 ab.
            self._tension = _clamp(self._tension + tension_impulse * 0.3, lo=-1.0, hi=1.0)

            # CPU Selbstschutz: Tension deckeln bei Ueberhitzung
            if self._cpu_temp_normalized > 0.9:
                self._tension = min(self._tension, 0.8)

            # Nacht: zusaetzlicher Decay
            if period == "nachts":
                self._tension *= 0.98

            # Abends: leicht schnellerer Decay
            if period == "abends":
                self._tension *= 0.995

            # =============================================================
            # 2. DOMINANCE berechnen (Drift + Impulse)
            # =============================================================
            dominance_impulse = 0.0
            for key, weight in self.DOMINANCE_WEIGHTS.items():
                val = all_inputs.get(key, 0.0)
                dominance_impulse += val * weight

            # Homoeostatischer Drift Richtung +0.5
            drift = self.DOMINANCE_DRIFT_RATE
            if self._dominance < self.DOMINANCE_DRIFT_TARGET:
                self._dominance += drift
            elif self._dominance > self.DOMINANCE_DRIFT_TARGET:
                self._dominance -= drift

            # Impuls addieren (gedaempft — langsame Persoenlichkeits-Shifts)
            self._dominance += dominance_impulse * 0.05
            self._dominance = _clamp(self._dominance, -1.0, 1.0)

            # Nacht: leichter Shadow-Drift
            if period == "nachts":
                self._dominance -= 0.0003  # ~-0.018/min
                self._dominance = max(-1.0, self._dominance)

            # =============================================================
            # 2b. PRESENCE — waechst bei Owner, sinkt ohne
            # =============================================================
            markus_val = all_inputs.get("markus_recognized", 0.0)
            if markus_val > 0.1:
                # Owner erkannt → Presence waechst
                self._presence = min(1.0, self._presence + self._presence_grow_rate)
            else:
                # Kein Owner → Presence sinkt
                self._presence = max(0.0, self._presence - self._presence_decay_rate)

            # =============================================================
            # 3. Cross-Model Correlation
            # =============================================================
            self._apply_cross_model_patterns(all_inputs)

            # =============================================================
            # 4. BERSERKER-Logik
            # =============================================================
            if self._berserker_active:
                # Berserker laeuft — pruefen ob Auto-Reset
                if now > self._berserker_until:
                    self._berserker_active = False
                    self._dominance = max(self._dominance, self.BERSERKER_DOMINANCE_RESET)
                    _logger.info("[CORE] Berserker Auto-Reset: D -> %.2f", self._dominance)
                else:
                    # Waehrend Berserker: dominance geclampt auf min 0.2
                    self._dominance = max(self._dominance, self.BERSERKER_DOMINANCE_RESET)
            elif (self._impulse_flag
                  and self._tension > self.BERSERKER_TENSION_THRESHOLD
                  and self._cpu_temp_normalized < 0.9):
                # Berserker aktivieren
                self._berserker_active = True
                self._berserker_until = now + self.BERSERKER_DURATION
                self._impulse_flag = False
                _logger.warning("[CORE] === BERSERKER AKTIVIERT === T=%.3f", self._tension)

            # =============================================================
            # 4b. Owner-Override Timer pruefen
            # =============================================================
            if self._owner_confirmed:
                if now > self._owner_confirmed_until:
                    self._owner_confirmed = False
                    self._owner_confirmed_until = 0.0
                    _logger.info("[CORE] Owner-Override abgelaufen (Timer)")
                else:
                    # Waehrend Override: leichter Guardian-Drift (wie Markus erkannt)
                    self._dominance = _clamp(self._dominance + 0.005, -1.0, 1.0)

            # =============================================================
            # 5. Zone mit Hysterese bestimmen (Gate0 Phase 5)
            #    Tension-basiert mit 10s Stabilitaetsfenster.
            #    Tension > 0.6 stabil 10s -> Shadow
            #    Tension < 0.3 stabil 10s -> Guardian
            #    Dazwischen: Zone bleibt (kein Flackern).
            # =============================================================

            # Tension-Schwellen-Timer aktualisieren
            if self._tension > self.TENSION_HIGH_THRESHOLD:
                if self._tension_high_since is None:
                    self._tension_high_since = now
                self._tension_low_since = None
            elif self._tension < self.TENSION_LOW_THRESHOLD:
                if self._tension_low_since is None:
                    self._tension_low_since = now
                self._tension_high_since = None
            else:
                # Zwischen den Schwellen: beide Timer resetten
                self._tension_high_since = None
                self._tension_low_since = None

            # Zone-Tracking fuer zone_changed-Event-Emission (2026-05-03)
            zone_before_tick = self._current_zone

            if self._berserker_active:
                self._current_zone = "berserker"
            elif self._current_zone == "berserker" and not self._berserker_active:
                # Berserker endet -> Zone basierend auf tension
                self._current_zone = "guardian" if self._tension < 0.5 else "shadow"
                _logger.info(f"[CORE] Zone: BERSERKER -> {self._current_zone.upper()}")
            elif (self._tension_high_since is not None
                  and (now - self._tension_high_since) >= self.TENSION_HYSTERESIS_TIME
                  and self._current_zone != "shadow"):
                # Tension stabil ueber 0.6 seit 10s -> Shadow
                old_zone = self._current_zone
                self._current_zone = "shadow"
                _logger.info(
                    f"[WECHSLE] {old_zone}→shadow "
                    f"weil=tension_stabil_ueber_0.6_seit_10s "
                    f"T={self._tension:.3f}"
                )
                # ArbitrationEngine Identity-Override aufheben
                # (hohe Tension bedeutet Szene hat sich geaendert)
                try:
                    from core.arbitration import get_arbitration
                    arbi = get_arbitration()
                    if arbi.is_override_active():
                        info = arbi.get_override_info()
                        if info.get("source") == "identity":
                            arbi.clear_identity()
                            _logger.info(
                                "[CORE] Identity-Override aufgehoben "
                                "(Tension-Hysterese erzwingt Shadow)"
                            )
                except Exception:
                    pass
            elif (self._tension_low_since is not None
                  and (now - self._tension_low_since) >= self.TENSION_HYSTERESIS_TIME
                  and self._current_zone != "guardian"):
                # Tension stabil unter 0.3 seit 10s -> Guardian
                old_zone = self._current_zone
                self._current_zone = "guardian"
                _logger.info(
                    f"[WECHSLE] {old_zone}→guardian "
                    f"weil=tension_stabil_unter_0.3_seit_10s "
                    f"T={self._tension:.3f}"
                )

            # =============================================================
            # 6. Effekte ableiten
            # =============================================================
            self._effects = self._compute_effects()

            # =============================================================
            # 6b. zone_changed Event publishen wenn Zone-Wechsel passierte
            # =============================================================
            # Bug-Fix 2026-05-03: cam_led_to_state, zone_to_led, zone_to_ptz
            # subscribten auf "zone_changed", aber NIEMAND publishte das
            # Event. Markus' Beschwerde: Cam-LED hat nie aufgeblitzt.
            if self._current_zone != zone_before_tick:
                try:
                    from core.moloch_event_bus import get_event_bus
                    get_event_bus().publish(
                        "zone_changed",
                        {
                            "zone": self._current_zone,
                            "old_zone": zone_before_tick,
                            "tension": self._tension,
                            "source": "core_integrator_tick",
                        },
                    )
                except Exception as _e:
                    _logger.debug(f"[CORE] zone_changed publish skipped: {_e}")

            # =============================================================
            # 7. Logging
            # =============================================================
            self._tick_count += 1
            d_t = abs(self._tension - self._prev_tension)
            d_d = abs(self._dominance - self._prev_dominance)

            if d_t > 0.1 or d_d > 0.1:
                _logger.info(
                    f"[CORE] T={self._tension:.3f} D={self._dominance:+.3f} "
                    f"zone={self._current_zone} CPU={self._cpu_temp_raw:.0f}°C "
                    f"(dT={d_t:+.3f} dD={d_d:+.3f})"
                )

            # Heartbeat alle 60 Ticks
            if self._tick_count % 60 == 0:
                _logger.info(
                    f"[CORE] Heartbeat #{self._tick_count}: "
                    f"T={self._tension:.3f} D={self._dominance:+.3f} "
                    f"P={self._presence:.2f} "
                    f"zone={self._current_zone} CPU={self._cpu_temp_raw:.0f}°C "
                    f"NPU={self._npu_load:.2f}"
                )

            self._prev_tension = self._tension
            self._prev_dominance = self._dominance

    # =========================================================================
    # Cross-Model Patterns (angepasst auf tension + dominance)
    # =========================================================================

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

        Patterns treiben tension UND dominance basierend auf Sensormustern.
        WICHTIG: Wird innerhalb Lock aufgerufen.
        """
        # Trends alle 2 Ticks aktualisieren
        if self._tick_count % 2 == 0:
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

        # === Pattern 1: Unbekannter nah -> Tension + Shadow ===
        if person and unknown and proximity > 0.1:
            self._tension = _clamp(self._tension + 0.05, lo=-1.0, hi=1.0)
            self._dominance = _clamp(self._dominance - 0.02, -1.0, 1.0)

        # === Pattern 2: Unbekannter + hohe Bewegung -> Tension stark + Shadow ===
        if unknown and pose_energy > 0.5:
            self._tension = _clamp(self._tension + 0.08, lo=-1.0, hi=1.0)
            self._dominance = _clamp(self._dominance - 0.03, -1.0, 1.0)

        # === Pattern 3: Markus + ruhig + neutral/happy -> Tension sinkt, Guardian ===
        if markus and pose_energy < 0.2 and smoothed_emotion in (None, "neutral", "happy"):
            self._tension = _clamp(self._tension - 0.02, lo=-1.0, hi=1.0)
            self._dominance = _clamp(self._dominance + 0.01, -1.0, 1.0)

        # === Pattern 4: Markus + happy -> Guardian Boost ===
        if markus and smoothed_emotion == "happy":
            self._dominance = _clamp(self._dominance + 0.02, -1.0, 1.0)

        # === Pattern 5: Niemand + lange Absence -> Tension sinkt ===
        if not person and not face and absence > 10:
            self._tension *= 0.95

        # === Pattern 6: Person naehert sich -> Tension leicht hoch ===
        if approaching and person:
            self._tension = _clamp(self._tension + 0.03, lo=-1.0, hi=1.0)

        # === Pattern 7: Person entfernt sich -> Tension sinkt ===
        if leaving:
            self._tension *= 0.97

        # === Pattern 8: Voice + Face -> Guardian Interaktion ===
        if voice and face:
            self._dominance = _clamp(self._dominance + 0.02, -1.0, 1.0)
            self._tension = _clamp(self._tension + 0.01)

    # =========================================================================
    # Effekt-Berechnung
    # =========================================================================

    def _compute_effects(self) -> Dict[str, float]:
        """Effekte aus tension + dominance + CPU-Temp ableiten.

        Intern, nur innerhalb Lock aufrufen.
        """
        t = self._tension
        d = self._dominance
        cpu = self._cpu_temp_normalized
        npu = self._npu_load

        # Guardian-Anteil (0-1): -1->0, 0->0.5, +1->1
        guardian_influence = _clamp((d + 1.0) / 2.0)
        shadow_influence = 1.0 - guardian_influence

        # CPU-Temperatur Modifikatoren
        jitter_damping = 1.0
        ptz_speed_factor = 1.0
        _thermal_norm = (self.THERMAL_DAMPING_START - self.CPU_TEMP_MIN) / (self.CPU_TEMP_MAX - self.CPU_TEMP_MIN)
        if cpu > _thermal_norm:
            jitter_damping = 0.5      # -50% Jitter
            ptz_speed_factor = 0.7    # PTZ langsamer
        if cpu > 0.9:
            jitter_damping = 0.2
            ptz_speed_factor = 0.4

        # NPU-Last: Latenz-Variation
        latency_variation = 1.0
        if npu > 0.8:
            latency_variation = 1.05  # +5%

        p = self._presence

        return {
            # --- Tension-basierte Effekte ---
            "voice_intensity": _clamp(0.3 + t * 0.7),
            "response_latency": _clamp((1.0 - t * 0.5) * latency_variation),
            "micro_ptz_movement": _clamp(t * 0.4 * jitter_damping),
            "language_sharpness": _clamp(t * 0.8),

            # --- Dominance-basierte Effekte ---
            "guardian_influence": guardian_influence,
            "shadow_influence": shadow_influence,

            # --- Presence-basierte Effekte (waechst bei Owner-Praesenz) ---
            "presence": p,
            "voice_warmth": _clamp(0.3 + p * 0.7),      # Waermere Stimme bei hoher Presence
            "patience": _clamp(0.4 + p * 0.6),           # Mehr Geduld
            "engagement": _clamp(0.2 + p * 0.5),         # Mehr Eigeninitiative

            # --- Kombinierte Effekte ---
            "camera_stability": _clamp((1.0 - t * 0.4) * ptz_speed_factor),
            "led_feedback_frequency": _clamp(0.2 + t * 0.8),
            "speech_focus": _clamp(0.3 + abs(d) * 0.5),
            "snapshot_probability": _clamp(t * 0.6),
            "spontaneous_comments": _clamp(max(t, abs(d)) * 0.5),
            "ambient_ptz_behavior": _clamp(t * 0.3 + abs(d) * 0.2),
            "manifestation_intensity": _clamp(0.3 + t * 0.4 + abs(d) * 0.3),

            # --- Physiologische Modifikatoren ---
            "jitter_damping": jitter_damping,
            "ptz_speed_factor": ptz_speed_factor,
            "cpu_temp": cpu,
            "npu_load": npu,
        }

    # =========================================================================
    # Status-Export
    # =========================================================================

    def get_time_period(self) -> str:
        """Aktuelle Tageszeit als Public API."""
        return self._get_time_period()

    def get_status_dict(self) -> Dict:
        """Kompaktes Status-Dict fuer SHM-Export und Panel IPC."""
        with self._lock:
            result = {
                "tension": round(self._tension, 4),
                "dominance": round(self._dominance, 4),
                "presence": round(self._presence, 3),
                "zone": self._current_zone,
                "time_period": self._get_time_period(),
                "cpu_temp": round(self._cpu_temp_raw, 1),
                "cpu_temp_norm": round(self._cpu_temp_normalized, 3),
                "npu_load": round(self._npu_load, 3),
                "berserker_active": self._berserker_active,
                "owner_confirmed": self._owner_confirmed,
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
