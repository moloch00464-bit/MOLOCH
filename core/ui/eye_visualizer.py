#!/usr/bin/env python3
"""
M.O.L.O.C.H. Eye Visualizer — Organisches Auge mit Musik-Seele
===============================================================

DAS IST DER KERN — hier liegt die Seele!

Abonniert music.beat und music.frequency_bands Events.
Berechnet die 5 Animations-Parameter mit Lerp-Interpolation:

  1. iris_radius     — bass-getrieben (Iris pulsiert)
  2. pupil_scale     — beat-getrieben (Pupille zieht sich zusammen)
  3. brightness      — overall_energy (Helligkeit in Grundfarbe)
  4. ray_jitter      — high-getrieben (Iris-Textur Knistern)
  5. glow_alpha      — mid-getrieben (aeusserer Leuchtring)

Design-Philosophie:
  Grundfarbe bleibt IMMER die Guardian-State-Farbe.
  Musik fliesst IN die Farbe — sie zuckt nicht, sie atmet.
  Lerp ueberall — kein hartes Springen!

Guardian State → Grundfarben:
  IDLE:     (  0,  80, 200) — Tiefes Blau
  ALERT:    (200, 100,   0) — Amber
  SHADOW:   ( 80,   0, 120) — Dunkles Violett
  GUARDIAN: (  0, 180, 100) — Gruen-Tuerki
  SPEAKING: (  0, 150, 220) — Helles Cyan

Update-Rate: 30 FPS (33ms Timer) — nicht schneller!
Wenn keine Musik: Organisches Atmen (±5% Radius, 3-4 Sekunden Periode).

Singleton: get_eye_visualizer()
"""

import logging
import math
import threading
import time
from typing import Optional, Tuple

from core.moloch_event_bus import get_event_bus, PRIO_INFO

logger = logging.getLogger("EyeVisualizer")

# Guardian State → Grundfarben (RGB)
BASE_COLORS = {
    "IDLE":      (  0,  80, 200),
    "ALERT":     (200, 100,   0),
    "SHADOW":    ( 80,   0, 120),
    "GUARDIAN":  (  0, 180, 100),
    "SPEAKING":  (  0, 150, 220),
    # Zusatz-States (Kompatibilitaet mit bestehendem System)
    "guardian":  (  0, 102, 255),
    "shadow":    (204,   0,   0),
    "berserker": (136,   0,   0),
}

# Basis-Geometrie
BASE_IRIS_RADIUS   = 90    # Pixel
BASE_PUPIL_RADIUS  = 30    # Pixel

# Musik-Einfluss-Staerken
IRIS_PULSE_STRENGTH   = 0.25   # Bass → Iris +25%
PUPIL_CONTRACT_MIN    = 0.6    # Beat → Pupille auf 60%
GLOW_MAX_ALPHA        = 80     # Mid → Glow max 80/255
RAY_JITTER_MAX        = 3.0    # High → ±3px Textur-Stoerung

# Lerp-Alphas
LERP_SLOW   = 0.30    # Iris/Helligkeit (~300ms bis 90%)
LERP_FAST   = 0.65    # Beat-Reaktion Pupille (~130ms bis 90%)

# Atem-Modus (keine Musik)
BREATHE_SPEED  = 0.5   # Radiant/Sek (ergibt ~3.5s Periode)
BREATHE_AMP    = 0.05  # ±5% Iris-Radius


def lerp(current: float, target: float, alpha: float) -> float:
    """Lineare Interpolation. alpha=0 kein Wechsel, alpha=1 sofort."""
    return current + (target - current) * alpha


class EyeVisualizer:
    """
    Event-getriebener Animations-State fuer das MOLOCH-Auge.

    Abonniert music.beat + music.frequency_bands.
    Stellt get_render_state() bereit fuer GUI-Panel.
    Thread-safe.
    """

    def __init__(self):
        self._bus = get_event_bus()
        self._lock = threading.Lock()

        # Guardian State
        self._state = "IDLE"

        # Musik-Eingangswerte (roh, aus Events)
        self._in_bass    = 0.0
        self._in_mid     = 0.0
        self._in_high    = 0.0
        self._in_energy  = 0.0
        self._in_beat    = False
        self._music_active = False
        self._last_band_time = 0.0

        # Animierte Ausgangswerte (Lerp-interpoliert)
        self._iris_radius  = float(BASE_IRIS_RADIUS)
        self._pupil_scale  = 1.0
        self._brightness   = 0.7
        self._ray_jitter   = 0.0
        self._glow_alpha   = 0

        # Beat-Decay State
        self._beat_energy  = 0.0   # decayed 0→1→0

        # Letzter Render-Zeitpunkt
        self._last_render = time.monotonic()

    # =========================================================================
    # Start / Stop
    # =========================================================================

    def start(self):
        """Event-Subscriptions registrieren."""
        self._bus.subscribe("music.frequency_bands", self._on_bands, priority=5)
        self._bus.subscribe("music.beat", self._on_beat, priority=5)
        self._bus.subscribe("music.stopped", self._on_music_stopped, priority=5)
        self._bus.subscribe("music.playing", self._on_music_playing, priority=5)
        logger.info("[EYE-VIZ] Gestartet")

    def stop(self):
        """Subscriptions entfernen."""
        self._bus.unsubscribe("music.frequency_bands", self._on_bands)
        self._bus.unsubscribe("music.beat", self._on_beat)
        self._bus.unsubscribe("music.stopped", self._on_music_stopped)
        self._bus.unsubscribe("music.playing", self._on_music_playing)
        logger.info("[EYE-VIZ] Gestoppt")

    # =========================================================================
    # State setzen
    # =========================================================================

    def set_guardian_state(self, state: str):
        """Guardian State aktualisieren (bestimmt Grundfarbe)."""
        with self._lock:
            self._state = state

    # =========================================================================
    # Event Handler (werden in Event-Bus Thread aufgerufen)
    # =========================================================================

    def _on_bands(self, event):
        """music.frequency_bands → Eingangswerte aktualisieren."""
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        with self._lock:
            self._in_bass    = float(payload.get("bass",           0.0))
            self._in_mid     = float(payload.get("mid",            0.0))
            self._in_high    = float(payload.get("high",           0.0))
            self._in_energy  = float(payload.get("overall_energy", 0.0))
            self._music_active = True
            self._last_band_time = time.monotonic()

    def _on_beat(self, event):
        """music.beat → Beat-Energy triggern."""
        with self._lock:
            self._beat_energy = 1.0  # Decay laeuft in tick()
            self._in_beat = True

    def _on_music_stopped(self, event):
        with self._lock:
            self._music_active = False
            self._in_beat = False

    def _on_music_playing(self, event):
        with self._lock:
            self._music_active = True

    # =========================================================================
    # Render-Tick (soll alle 33ms vom GUI-Thread aufgerufen werden)
    # =========================================================================

    def tick(self) -> None:
        """
        Animierten State vorwaerts berechnen (33ms = 30 FPS).
        Muss regelmaessig aufgerufen werden — aus GUI-after() Loop.
        """
        with self._lock:
            now = time.monotonic()
            music_active = self._music_active

            # Musik-Timeout: wenn kein Band-Event seit 2s → inaktiv
            if music_active and (now - self._last_band_time) > 2.0:
                music_active = False
                self._music_active = False

            if music_active:
                self._tick_music(now)
            else:
                self._tick_breathe(now)

            self._last_render = now

    def _tick_music(self, now: float):
        """Musik-reaktive Animation."""
        bass   = self._in_bass
        mid    = self._in_mid
        high   = self._in_high
        energy = self._in_energy

        # 1. Iris-Radius: Bass → Puls
        target_iris = BASE_IRIS_RADIUS * (1.0 + bass * IRIS_PULSE_STRENGTH)
        self._iris_radius = lerp(self._iris_radius, target_iris, LERP_SLOW)

        # 2. Pupillen-Kontraktion: Beat-Decay
        self._beat_energy = max(0.0, self._beat_energy * 0.82)  # Decay ~200ms bei 30FPS
        pupil_target = PUPIL_CONTRACT_MIN + (1.0 - self._beat_energy) * (1.0 - PUPIL_CONTRACT_MIN)
        self._pupil_scale = lerp(self._pupil_scale, pupil_target, LERP_FAST)

        # 3. Helligkeit: Overall Energy
        brightness_target = 0.7 + energy * 0.4
        self._brightness = lerp(self._brightness, brightness_target, LERP_SLOW)

        # 4. Ray Jitter: High-Frequenz
        jitter_target = high * RAY_JITTER_MAX
        self._ray_jitter = lerp(self._ray_jitter, jitter_target, LERP_SLOW)

        # 5. Glow Alpha: Mid-Band
        glow_target = int(mid * GLOW_MAX_ALPHA)
        self._glow_alpha = int(lerp(float(self._glow_alpha), float(glow_target), LERP_SLOW))

        self._in_beat = False

    def _tick_breathe(self, now: float):
        """Organisches Atmen wenn keine Musik (schlafendes Tier)."""
        breathe_offset = math.sin(now * BREATHE_SPEED) * BREATHE_AMP
        target_iris = BASE_IRIS_RADIUS * (1.0 + breathe_offset)
        self._iris_radius  = lerp(self._iris_radius,  target_iris, LERP_SLOW)
        self._pupil_scale  = lerp(self._pupil_scale,  1.0,         LERP_SLOW)
        self._brightness   = lerp(self._brightness,   0.7,         LERP_SLOW * 0.5)
        self._ray_jitter   = lerp(self._ray_jitter,   0.0,         LERP_SLOW)
        self._glow_alpha   = int(lerp(float(self._glow_alpha), 0.0, LERP_SLOW))
        self._beat_energy *= 0.9

    # =========================================================================
    # Ausgabe fuer GUI
    # =========================================================================

    def get_render_state(self) -> dict:
        """
        Thread-safe Snapshot des aktuellen Animations-States.

        Rueckgabe-Dict:
          iris_radius:  float (Pixel)
          pupil_radius: float (Pixel)
          brightness:   float (0.7-1.1)
          ray_jitter:   float (Pixel, 0-3)
          glow_alpha:   int (0-80)
          base_color:   tuple (R,G,B)
          music_active: bool
        """
        with self._lock:
            state = self._state
            base_color = BASE_COLORS.get(state, BASE_COLORS["IDLE"])
            brightness = self._brightness

            # Helligkeit auf Grundfarbe anwenden — NIEMALS Farbe wechseln!
            r = min(255, max(0, int(base_color[0] * brightness)))
            g = min(255, max(0, int(base_color[1] * brightness)))
            b = min(255, max(0, int(base_color[2] * brightness)))

            return {
                "iris_radius":  round(self._iris_radius, 1),
                "pupil_radius": round(BASE_PUPIL_RADIUS * self._pupil_scale, 1),
                "brightness":   round(self._brightness, 3),
                "ray_jitter":   round(self._ray_jitter, 2),
                "glow_alpha":   self._glow_alpha,
                "base_color":   base_color,
                "active_color": (r, g, b),
                "beat":         self._beat_energy > 0.5,
                "music_active": self._music_active,
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[EyeVisualizer] = None
_instance_lock = threading.Lock()


def get_eye_visualizer() -> EyeVisualizer:
    """Singleton-Zugriff auf den EyeVisualizer."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EyeVisualizer()
    return _instance
