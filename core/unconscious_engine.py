#!/usr/bin/env python3
"""
M.O.L.O.C.H. Unconscious Engine — Unterbewusstsein
====================================================

Tick-basierter Background-Loop (alle 10 Sekunden).
Liest Systemzustand aus moloch_status.json und sendet
Impulse an die MoodEngine via /dev/shm/moloch_impulse.json.

Zwei Schichten:
  Schicht 1 (Mood):     Tension/Face/Zone → Persoenlichkeits-Impulse
  Schicht 2 (Pipeline): FPS/RAM/Tracking/TTS → Self-Tune Impulse

Wu-Wei: Kein Trigger = kein Impuls. Nichtstun ist auch eine Entscheidung.

Laeuft als daemon=True Thread in moloch_service.py.
"""

import json
import logging
import os
import threading
import time

logger = logging.getLogger("UnconsciousEngine")

# Pfade
STATUS_PATH = "/dev/shm/moloch_status.json"
IMPULSE_PATH = "/dev/shm/moloch_impulse.json"
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")
HANDSHAKE_PATH = os.path.expanduser("~/moloch/ipc/handshake.json")
SELF_TUNE_LOG = os.path.expanduser("~/moloch/logs/self_tune.log")

# Tick-Intervall in Sekunden
TICK_INTERVAL = 10.0

# === Schicht 1: Mood-Schwellen ===
TENSION_HIGH = 0.7
TENSION_LOW = 0.3
FACE_TIMEOUT_S = 60.0

# === Schicht 2: Pipeline-Schwellen ===
TEMP_CRITICAL_C = 70.0
TEMP_WARNING_C = 65.0
FPS_MINIMUM = 10.0
FPS_WARNING = 15.0
RAM_WARNING_MB = 3200
RAM_CRITICAL_MB = 3500
TRACKING_JITTER_LIMIT = 60
FACE_RECOGNITION_MIN_SIM = 0.50


class UnconsciousEngine:
    """Unterbewusstsein — bewertet intern Zustaende und sendet Impulse."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._last_impulse = None
        self._last_impulse_time = 0.0
        # Cooldown pro Impuls-Typ: gleicher Impuls nicht oefter als X Sekunden
        self._impulse_cooldown = 30.0
        self._cooldowns = {}  # {impuls_key: last_time}
        # Trend-Tracking: RAM und FPS ueber Zeit beobachten
        self._ram_history = []  # [(timestamp, ram_mb)]
        self._fps_history = []  # [(timestamp, fps)]
        self._history_max = 30  # 30 Ticks = 5 Minuten bei 10s Intervall
        # Self-Tune Zaehler: max 3 pro Stunde
        self._tune_count_hour = 0
        self._tune_hour_start = time.time()
        logger.info("[UNCONSCIOUS] Initialisiert (Mood + Pipeline)")

    def start(self):
        """Startet den Tick-Loop als Daemon-Thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="UnconsciousEngine",
            daemon=True,
        )
        self._thread.start()
        logger.info("[UNCONSCIOUS] Gestartet (Tick alle %.0fs)", TICK_INTERVAL)

    def stop(self):
        """Stoppt den Tick-Loop."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("[UNCONSCIOUS] Gestoppt")

    def _tick_loop(self):
        """Hauptschleife — alle TICK_INTERVAL Sekunden einen Tick."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error("[UNCONSCIOUS] Tick-Fehler: %s", e)
            self._stop_event.wait(TICK_INTERVAL)

    def _tick(self):
        """Ein Bewertungszyklus — zwei Schichten."""
        status = self._read_status()
        if not status:
            return

        now = time.time()

        # Werte extrahieren
        tension = float(status.get("tension", 0.0))
        last_face_ts = float(status.get("last_seen_face", 0.0))
        temp = float(status.get("system_temp", 0.0))
        fps = float(status.get("fps", 20.0))
        ram_mb = float(status.get("ram_mb", 0.0))
        tracking_moves = float(status.get("tracking_moves_per_minute", 0.0))
        tracking_state = str(status.get("tracking_state", ""))
        face_sim = float(status.get("face_similarity", 0.0))
        face_id = str(status.get("face_id", ""))
        npu_scenario = str(status.get("npu_scenario", ""))

        face_age = now - last_face_ts if last_face_ts > 0 else 9999.0
        face_active = face_age < FACE_TIMEOUT_S

        # Trend-Tracking aktualisieren
        self._ram_history.append((now, ram_mb))
        self._fps_history.append((now, fps))
        if len(self._ram_history) > self._history_max:
            self._ram_history.pop(0)
        if len(self._fps_history) > self._history_max:
            self._fps_history.pop(0)

        # ================================================================
        # SCHICHT 1: MOOD — Persoenlichkeits-Impulse
        # ================================================================

        # Regel 1: Hohe Tension + kein Gesicht → Shadow-Impuls
        if tension > TENSION_HIGH and not face_active:
            self._mood_push("shadow", "Tension hoch, niemand da")

        # Regel 2: Niedrige Tension + Gesicht aktiv → Guardian-Impuls
        elif tension < TENSION_LOW and face_active:
            self._mood_push("guardian", "Ruhig, Markus ist da")

        # ================================================================
        # SCHICHT 2: PIPELINE — System-Gesundheit und Self-Tune
        # ================================================================

        # Regel 3: Temperatur kritisch → reduce + self_tune Fan
        if temp > TEMP_CRITICAL_C:
            self._mood_push("reduce", f"Temp {temp:.0f}C kritisch")
            self._self_tune_push("fan", "noctua_base_pct", 0.05,
                                 f"CPU {temp:.0f}C > {TEMP_CRITICAL_C}C")

        elif temp > TEMP_WARNING_C:
            self._self_tune_push("fan", "noctua_base_pct", 0.03,
                                 f"CPU {temp:.0f}C > {TEMP_WARNING_C}C (Warnung)")

        # Regel 4: FPS zu niedrig → reduce + self_tune Confidence hoch
        if fps < FPS_MINIMUM:
            self._mood_push("reduce", f"FPS {fps:.1f} kritisch")
            self._self_tune_push("thresholds", "yolo_conf", 0.05,
                                 f"FPS {fps:.1f} < {FPS_MINIMUM}")

        elif fps < FPS_WARNING:
            self._self_tune_push("thresholds", "yolo_conf", 0.03,
                                 f"FPS {fps:.1f} < {FPS_WARNING} (Warnung)")

        # Regel 5: RAM steigt stetig → Warnung loggen
        if ram_mb > RAM_CRITICAL_MB:
            self._mood_push("reduce", f"RAM {ram_mb:.0f}MB kritisch")
            self._log_concern("ram_critical", f"RAM {ram_mb:.0f}MB > {RAM_CRITICAL_MB}MB")

        elif ram_mb > RAM_WARNING_MB:
            ram_trend = self._check_ram_trend()
            if ram_trend > 0:
                self._log_concern("ram_rising",
                                  f"RAM {ram_mb:.0f}MB steigend (+{ram_trend:.0f}MB/5min)")

        # Regel 6: Tracking zu hektisch → self_tune smooth_alpha hoch
        if tracking_state == "tracking" and tracking_moves > TRACKING_JITTER_LIMIT:
            self._self_tune_push("tracker", "smooth_alpha", 0.05,
                                 f"Tracking {tracking_moves:.0f} moves/min > {TRACKING_JITTER_LIMIT}")

        # Regel 7: Gesicht erkannt aber Similarity niedrig → Warnung
        if face_active and face_id and face_sim > 0 and face_sim < FACE_RECOGNITION_MIN_SIM:
            self._log_concern("face_sim_low",
                              f"Face '{face_id}' sim={face_sim:.2f} < {FACE_RECOGNITION_MIN_SIM}")

        # Regel 8: FPS-Trend faellt → frueh warnen bevor es kritisch wird
        fps_trend = self._check_fps_trend()
        if fps_trend < -3.0:  # FPS sinkt um mehr als 3/5min
            self._log_concern("fps_dropping",
                              f"FPS-Trend faellt: {fps_trend:+.1f}/5min (aktuell {fps:.1f})")

        # Regel 9: Wu-Wei — Nichtstun ist auch eine Entscheidung
        # Kein Impuls. Stille. Das Unterbewusstsein ruht.

    # ================================================================
    # IMPULS-AUSGABE
    # ================================================================

    def _mood_push(self, impulse: str, reason: str = ""):
        """Schreibt Mood-Impuls in /dev/shm/moloch_impulse.json mit Cooldown."""
        if not self._check_cooldown(f"mood_{impulse}"):
            return

        payload = {
            "source": "unconscious",
            "type": "mood",
            "impulse": impulse,
            "reason": reason,
            "timestamp": time.time(),
        }

        try:
            with open(IMPULSE_PATH, "w") as f:
                json.dump(payload, f)
            logger.info("[UNCONSCIOUS] Mood: %s (%s)", impulse, reason)
        except Exception as e:
            logger.error("[UNCONSCIOUS] Mood-Write fehlgeschlagen: %s", e)

    def _self_tune_push(self, section: str, key: str, step: float, reason: str):
        """Schreibt Self-Tune Impuls — Parameter soll geaendert werden.
        Liest aktuellen Wert und Registry-Limits, berechnet neuen Wert."""
        if not self._check_cooldown(f"tune_{section}_{key}"):
            return
        if not self._check_tune_rate_limit():
            return

        registry = self._read_registry()
        if not registry:
            return

        # Parameter in Registry finden
        param = None
        for p in registry.get("parameters", []):
            if p.get("section") == section and p.get("key") == key:
                param = p
                break
        if not param:
            logger.warning("[UNCONSCIOUS] Parameter %s.%s nicht in Registry", section, key)
            return

        # Aktuellen Wert aus settings.json lesen
        current = self._read_setting(section, key)
        if current is None:
            current = param.get("default", 0)

        # Neuen Wert berechnen (immer erhoehen fuer Self-Tune)
        new_val = current + step
        param_max = param.get("max")
        if param_max is not None and new_val > float(param_max):
            new_val = float(param_max)
            if new_val == current:
                return  # Schon am Limit

        new_val = round(new_val, 3)

        payload = {
            "source": "unconscious",
            "type": "self_tune",
            "section": section,
            "key": key,
            "old_value": current,
            "new_value": new_val,
            "reason": reason,
            "timestamp": time.time(),
        }

        try:
            with open(IMPULSE_PATH, "w") as f:
                json.dump(payload, f)
            self._tune_count_hour += 1
            logger.info("[UNCONSCIOUS] Tune: %s.%s %s → %s (%s)",
                        section, key, current, new_val, reason)
        except Exception as e:
            logger.error("[UNCONSCIOUS] Tune-Write fehlgeschlagen: %s", e)

    def _log_concern(self, concern_id: str, message: str):
        """Loggt eine Beobachtung ohne sofortigen Impuls (fuer HANDSHAKE spaeter)."""
        if not self._check_cooldown(f"concern_{concern_id}"):
            return
        try:
            log_line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {concern_id}: {message}\n"
            with open(SELF_TUNE_LOG, "a") as f:
                f.write(log_line)
            logger.info("[UNCONSCIOUS] Concern: %s", message)
        except Exception:
            pass

    # ================================================================
    # COOLDOWN + RATE LIMIT
    # ================================================================

    def _check_cooldown(self, key: str) -> bool:
        """Prueft ob ein Impuls-Typ seinen Cooldown ueberschritten hat."""
        now = time.time()
        last = self._cooldowns.get(key, 0.0)
        if now - last < self._impulse_cooldown:
            return False
        self._cooldowns[key] = now
        return True

    def _check_tune_rate_limit(self) -> bool:
        """Max 3 Self-Tune Impulse pro Stunde."""
        now = time.time()
        if now - self._tune_hour_start > 3600:
            self._tune_count_hour = 0
            self._tune_hour_start = now
        return self._tune_count_hour < 3

    # ================================================================
    # TREND-ANALYSE
    # ================================================================

    def _check_ram_trend(self) -> float:
        """RAM-Trend: Differenz zwischen aeltestem und neuestem Wert in MB."""
        if len(self._ram_history) < 5:
            return 0.0
        oldest = self._ram_history[0][1]
        newest = self._ram_history[-1][1]
        return newest - oldest

    def _check_fps_trend(self) -> float:
        """FPS-Trend: Differenz zwischen aeltestem und neuestem Wert."""
        if len(self._fps_history) < 5:
            return 0.0
        oldest = self._fps_history[0][1]
        newest = self._fps_history[-1][1]
        return newest - oldest

    # ================================================================
    # DATEI-LESER
    # ================================================================

    def _read_status(self):
        """Liest moloch_status.json. Gibt dict oder None zurueck."""
        try:
            if not os.path.exists(STATUS_PATH):
                return None
            with open(STATUS_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _read_setting(self, section: str, key: str):
        """Liest einen Wert aus settings.json."""
        try:
            with open(SETTINGS_PATH, "r") as f:
                data = json.load(f)
            return data.get(section, {}).get(key)
        except Exception:
            return None

    def _read_registry(self):
        """Liest self_tune_registry.json. Gibt dict oder None zurueck."""
        try:
            registry_path = os.path.expanduser("~/moloch/config/self_tune_registry.json")
            with open(registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return None


# Singleton-Accessor
_engine = None
_engine_lock = threading.Lock()


def get_unconscious_engine() -> UnconsciousEngine:
    """Singleton-Zugriff auf UnconsciousEngine."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = UnconsciousEngine()
    return _engine
