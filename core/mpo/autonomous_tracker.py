#!/usr/bin/env python3
"""
M.O.L.O.C.H. Autonomous Tracker v2 - AbsoluteMove
===================================================

Dedicated 5 Hz tracking thread with AbsoluteMove position control.
Implements proportional tracking and search behavior using real camera
position feedback (closed-loop control).

Upgrade from v1 (ContinuousMove):
- AbsoluteMove replaces ContinuousMove (no 90-degree-per-call limit)
- Real camera position via get_position() replaces virtual position tracking
- track_target(error_x, error_y) for proportional position-based tracking
- move_absolute() for search/patrol movements
- Full 342.8 degree pan range utilization

Features:
- 5 Hz tracking loop (200ms cycle)
- AbsoluteMove with proportional position control
- Search mode: patrol sweep when target lost
- Largest bounding box selection with scoring
- Configurable deadzone and gain
- State machine: IDLE, TRACKING, SEARCHING, LOCKED, DWELL, FROZEN

Author: M.O.L.O.C.H. System
Date: 2026-02-08
"""

import json
import time
import math
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# Motor-Learner: adaptiver Gain aus Bewegungsfeedback
try:
    from core.mpo.motor_learner import get_motor_learner as _get_motor_learner
    _MOTOR_LEARNER_AVAILABLE = True
except ImportError:
    _MOTOR_LEARNER_AVAILABLE = False

# Persistente letzte Face-Position (ueberlebt Reboot)
import os as _os
_LAST_FACE_POS_FILE = _os.path.expanduser("~/moloch/config/last_face_position.json")
_LEARNED_POSITIONS_FILE = _os.path.expanduser("~/moloch/config/learned_patrol_positions.json")

def _save_last_face_pos(pan: float, tilt: float):
    """Letzte Face-Position auf Disk speichern (max alle 10s)."""
    try:
        with open(_LAST_FACE_POS_FILE, "w") as f:
            json.dump({"pan": round(pan, 1), "tilt": round(tilt, 1),
                       "ts": time.time()}, f)
    except Exception:
        pass

def _load_last_face_pos() -> tuple:
    """Letzte Face-Position laden. Returns (pan, tilt) oder (0.0, 0.0)."""
    try:
        with open(_LAST_FACE_POS_FILE) as f:
            d = json.load(f)
        # Nur gueltig wenn < 24h alt
        if time.time() - d.get("ts", 0) < 86400:
            return d.get("pan", 0.0), d.get("tilt", 0.0)
    except Exception:
        pass
    return 0.0, 0.0

def _save_learned_positions(positions: list):
    """Gelernte Patrol-Positionen persistent speichern."""
    try:
        with open(_LEARNED_POSITIONS_FILE, "w") as f:
            json.dump({"positions": positions, "ts": time.time()}, f)
    except Exception:
        pass

def _load_learned_positions() -> list:
    """Gelernte Patrol-Positionen laden. Returns Liste oder []."""
    try:
        with open(_LEARNED_POSITIONS_FILE) as f:
            d = json.load(f)
        # Nur gueltig wenn < 7 Tage alt
        if time.time() - d.get("ts", 0) < 7 * 86400:
            return d.get("positions", [])
    except Exception:
        pass
    return []


class STMovementLearner:
    """Lernt aus Kamera-Smart-Tracking Bewegungen.

    Zeichnet Positionen auf waehrend Kamera-ST aktiv ist.
    Wenn Kamera stoppt/verlangsamt = Person erkannt → "Hot-Spot".
    Clustert Hot-Spots zu Patrol-Positionen (max 8).
    """

    def __init__(self):
        self._positions = []       # [(pan, tilt, ts), ...] — Rohaufzeichnung
        self._hot_spots = []       # [(pan, tilt, count), ...] — Wo Kamera stoppt
        self._prev_pan = None
        self._prev_tilt = None
        self._prev_ts = 0.0
        self._still_since = 0.0    # Seit wann Kamera stillsteht
        self._STILL_THRESHOLD = 2.0  # Grad — weniger Bewegung = "still"
        self._STILL_TIME = 1.5      # Sekunden still = Hot-Spot
        self._MAX_HOTSPOTS = 50     # Ringbuffer-Groesse
        self._MAX_POSITIONS = 500   # Rohaufzeichnung Ringbuffer

        # Bewegungsdynamik: gelernte Kamera-Motor-Eigenschaften
        self._velocities_pan = []   # deg/s — Pan-Geschwindigkeiten
        self._velocities_tilt = []  # deg/s — Tilt-Geschwindigkeiten
        self._MAX_VELOCITIES = 200  # Ringbuffer

    def record(self, pan: float, tilt: float):
        """Aufrufen bei jedem Position-Read waehrend Kamera-ST aktiv."""
        now = time.time()

        # Rohaufzeichnung (Ringbuffer)
        self._positions.append((pan, tilt, now))
        if len(self._positions) > self._MAX_POSITIONS:
            self._positions = self._positions[-self._MAX_POSITIONS:]

        if self._prev_pan is not None:
            dt = now - self._prev_ts
            delta_pan = pan - self._prev_pan
            delta_tilt = tilt - self._prev_tilt
            delta = abs(delta_pan) + abs(delta_tilt)

            # Bewegungsdynamik: Geschwindigkeit aufzeichnen (nur bei echter Bewegung)
            if dt > 0.05 and delta > 0.5:
                vel_pan = abs(delta_pan) / dt
                vel_tilt = abs(delta_tilt) / dt
                self._velocities_pan.append(vel_pan)
                self._velocities_tilt.append(vel_tilt)
                if len(self._velocities_pan) > self._MAX_VELOCITIES:
                    self._velocities_pan = self._velocities_pan[-self._MAX_VELOCITIES:]
                    self._velocities_tilt = self._velocities_tilt[-self._MAX_VELOCITIES:]

            if delta < self._STILL_THRESHOLD:
                # Kamera steht (fast) still — Sensor hat was erkannt
                if self._still_since == 0.0:
                    self._still_since = now
                elif now - self._still_since >= self._STILL_TIME:
                    # Genuegend lang still → Hot-Spot registrieren
                    self._add_hot_spot(pan, tilt)
                    self._still_since = 0.0  # Reset, erst wieder nach Bewegung
            else:
                # Kamera bewegt sich — Reset
                self._still_since = 0.0

        self._prev_pan = pan
        self._prev_tilt = tilt
        self._prev_ts = now

    def _add_hot_spot(self, pan: float, tilt: float):
        """Hot-Spot registrieren (Ringbuffer, max _MAX_HOTSPOTS)."""
        # Existierenden Hot-Spot in der Naehe mergen (±15°)
        for i, (hp, ht, count) in enumerate(self._hot_spots):
            if abs(hp - pan) < 15.0 and abs(ht - tilt) < 10.0:
                # Gleitender Durchschnitt
                new_pan = (hp * count + pan) / (count + 1)
                new_tilt = (ht * count + tilt) / (count + 1)
                self._hot_spots[i] = (round(new_pan, 1), round(new_tilt, 1), count + 1)
                return

        # Neuer Hot-Spot
        if len(self._hot_spots) >= self._MAX_HOTSPOTS:
            # Aeltesten/seltensten entfernen
            self._hot_spots.sort(key=lambda x: x[2])
            self._hot_spots.pop(0)
        self._hot_spots.append((round(pan, 1), round(tilt, 1), 1))

    def get_patrol_positions(self, max_positions: int = 8) -> list:
        """Top Hot-Spots als Patrol-Positionen (sortiert nach Haeufigkeit).

        Returns: [(pan, tilt), ...] — max max_positions Eintraege.
        """
        if not self._hot_spots:
            return []
        # Nach Haeufigkeit sortieren, Top N
        sorted_spots = sorted(self._hot_spots, key=lambda x: x[2], reverse=True)
        positions = [(p, t) for p, t, _c in sorted_spots[:max_positions]]
        return positions

    def get_learned_dynamics(self) -> dict:
        """Gelernte Kamera-Motor-Dynamik fuer MOLOCH Tracking.

        Returns dict mit:
          avg_vel_pan/tilt:  Durchschnittliche Geschwindigkeit (deg/s)
          max_vel_pan/tilt:  Maximale beobachtete Geschwindigkeit
          median_vel_pan/tilt: Median-Geschwindigkeit (robust)
          samples: Anzahl Messungen
        """
        if not self._velocities_pan:
            return {"samples": 0}

        sorted_pan = sorted(self._velocities_pan)
        sorted_tilt = sorted(self._velocities_tilt)
        n = len(sorted_pan)
        median_pan = sorted_pan[n // 2]
        median_tilt = sorted_tilt[n // 2]

        return {
            "avg_vel_pan": round(sum(self._velocities_pan) / n, 1),
            "avg_vel_tilt": round(sum(self._velocities_tilt) / n, 1),
            "max_vel_pan": round(max(self._velocities_pan), 1),
            "max_vel_tilt": round(max(self._velocities_tilt), 1),
            "median_vel_pan": round(median_pan, 1),
            "median_vel_tilt": round(median_tilt, 1),
            "samples": n,
            "positions_recorded": len(self._positions),
        }

    def get_stats(self) -> dict:
        """Statistiken fuer Status/Debug."""
        dynamics = self.get_learned_dynamics()
        return {
            "hot_spots": len(self._hot_spots),
            "top_positions": self.get_patrol_positions(4),
            "dynamics": dynamics,
        }

# PTZ Debug Logger - schreibt in ~/moloch/logs/ptz_debug.log
_ptz_log_path = _os.path.expanduser("~/moloch/logs/ptz_debug.log")
_os.makedirs(_os.path.dirname(_ptz_log_path), exist_ok=True)
ptz_debug = logging.getLogger("ptz_debug")
ptz_debug.setLevel(logging.DEBUG)
_ptz_fh = logging.FileHandler(_ptz_log_path, mode="w")
_ptz_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
ptz_debug.addHandler(_ptz_fh)
ptz_debug.propagate = False

# Import perception state for user visibility check
try:
    from context.perception_state import get_perception_state, is_user_visible
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    logger.warning("perception_state not available - tracker will use raw detections")


class TrackerState(Enum):
    """Tracker state machine."""
    IDLE = "idle"           # Tracking disabled
    TRACKING = "tracking"   # Following target with AbsoluteMove
    SEARCHING = "searching" # Lost target, sweeping
    LOCKED = "locked"       # Target centered in deadzone
    DWELL = "dwell"         # Target acquired, waiting before movement
    FROZEN = "frozen"       # Target perfectly centered, no movement needed
    COAST = "coast"         # Ziel stabil seit 2s, Kamera komplett eingefroren
    PARKED = "parked"       # Search-Timeout: Kamera geparkt auf Home, NPU IDLE


class TargetType(Enum):
    """Type of tracking target - adaptive selection."""
    NONE = "none"           # No valid target
    FACE = "face"           # Tracking face (preferred)
    BODY = "body"           # Tracking full body (fallback)


@dataclass
class TrackingConfig:
    """Tracking parameters."""
    # === LOCK/FROZEN State Parameters ===
    lock_threshold_pixels: int = 8
    unlock_threshold_pixels: int = 15
    frozen_threshold_pixels: int = 5

    # === Dwell Timer ===
    dwell_time_sec: float = 0.0  # SOFORT tracken, kein Warten (war 0.5)

    # === AbsoluteMove Tracking Parameters ===
    # Kamera Motor-Speed: ~30 deg/s (Kalibrierung: 342deg in ~12s)
    # -----------------------------------------------------------------------
    # SMOOTHING-TUNING (Markus: alle Parameter hier anpassen!)
    # smooth_alpha: EMA-Glaettung 0.0=eingefroren / 1.0=kein Glaetten
    # min_step_deg: Mikro-Zittern unterdruecken (< 2 Grad = kein Befehl)
    # max_step_pan/tilt: Max-Sprung pro Update (Grad) — kein Ruckeln bei grossen Fehlern
    # move_cooldown_ms: Min. Abstand zwischen zwei Befehlen (kein Command-Flood)
    # tracking_speed: Basis-Speed (0.0-1.0), skaliert proportional mit Fehlergroesse
    # -----------------------------------------------------------------------
    fov_horizontal: float = 110.0
    fov_vertical: float = 65.0
    pan_gain: float = 0.45          # Reduziert (war 0.65) — weniger Ueberschwinger
    tilt_gain: float = 0.40         # Reduziert (war 0.50)
    max_step_pan: float = 15.0      # Kleinere Schritte (war 25.0) — Kamera laeuft nicht drüber
    max_step_tilt: float = 12.0     # Tilt auch reduziert (war 18.0)
    face_settle_time: float = 0.35  # Sekunden einfrieren wenn Gesicht frisch erkannt
    min_step_deg: float = 0.2       # Feinste Restkorrektur
    tracking_speed: float = 1.0     # Motoren Vollgas
    move_cooldown_ms: float = 50.0   # 50ms — 20 Updates/s maximal
    smooth_alpha: float = 0.70      # Fast direkte Reaktion, minimaler EMA-Filter

    # Kamera Hardware-Limits (SonoffCameraController clampt intern,
    # aber Tracker muss gecachte Position AUCH clampen!)
    pan_limit_min: float = -168.4
    pan_limit_max: float = 170.0
    tilt_limit_min: float = -15.0   # Boden bringt nix, Personen stehen/sitzen
    tilt_limit_max: float = 30.0    # Decke auch nicht (war 78.8)

    # Search mode parameters
    search_speed_min: float = 0.08      # Minimum bei kurzen Distanzen
    search_speed_max: float = 0.30      # Maximum bei weiten Sprüngen (>120 Grad)
    search_speed: float = 0.15          # Fallback/Default (wird dynamisch ueberschrieben)
    search_direction_interval: float = 6.0  # 6s pro Position, mehr Zeit zum Scannen (war 4.0)
    search_reset_to_center: bool = False
    search_patrol_positions: list = field(default_factory=lambda: [
        (0.0, 0.0),        # Markus' Sitzplatz
        (-60.0, 0.0),      # Leicht links
        (-120.0, 0.0),     # Tuer (Park-Position)
        (0.0, 20.0),       # Mitte hoch
        (60.0, 0.0),       # Leicht rechts
        (120.0, 0.0),      # Weiter rechts
    ])
    # G1-T06: Park-Position = Tuer (links ~-120 Grad)
    park_pan: float = -120.0
    park_tilt: float = 0.0
    search_home_timeout: float = 120.0  # 120s ohne Fund -> Home
    search_park_timeout: float = 180.0  # 3 Min ohne Detection -> Park-Modus (keine Bewegung, NPU IDLE)

    # Verlust-Logik (3-Phasen):
    #   Phase 1: 0-5s nach Verlust -> STEHEN BLEIBEN (halt_wait_timeout)
    #   Phase 2: 5-30s -> langsam Home fahren
    #   Phase 3: >30s -> Idle-Suche (langsames Patrol)
    halt_wait_timeout: float = 5.0     # 5s an letzter Position warten
    home_return_timeout: float = 30.0  # Nach 30s ohne Detection -> Home + Suche starten
    target_lost_timeout: float = 5.0   # Ab 5s -> Home (war 10s, jetzt = halt_wait)
    frame_width: int = 640
    frame_height: int = 640

    # === Detection filtering ===
    min_bbox_height_ratio: float = 0.10   # war 0.40 — filterte ALLE Personen auf Distanz (13% Height)
    max_bbox_center_y_ratio: float = 0.92   # Gate 0 Phase 3: war 0.75, filterte ALLE Gesichter raus
    min_bbox_area_ratio: float = 0.08
    min_confidence: float = 0.30   # war 0.50 — filterte Faces mit 0.45-0.49 Confidence
    min_aspect_ratio: float = 0.35

    # === Target persistence ===
    confidence_hysteresis: float = 0.15
    stability_frames: int = 7
    center_priority_weight: float = 0.4

    # === ADAPTIVE TARGET STRATEGY ===
    face_min_confidence: float = 0.55
    face_min_stability: int = 4
    face_max_bbox_height: float = 0.65
    body_min_confidence: float = 0.45
    body_min_bbox_height: float = 0.30
    body_min_stability: int = 5
    switch_cooldown_sec: float = 1.0
    prefer_current_target: bool = True

    # === SMOOTHING: Face/Person Wechsel-Daempfung ===
    source_hysteresis_frames: int = 1   # Body->Face sofort, kein Warten (war 3)
    center_ring_buffer_size: int = 5    # Kleinerer Buffer fuer schnellere Reaktion (war 10)
    min_frames_before_move: int = 1     # SOFORT bewegen nach erstem Frame (war 3)

    # === DEAD ZONE + COAST MODE (Tracker-Beruhigung) ===
    dead_zone_pct: float = 0.10        # ±10% Deadzone (Fallback, wird nicht mehr fuer frozen genutzt)
    track_start_pct: float = 0.13      # 13% - Hysterese-Obergrenze (war 18%)
    coast_stable_time: float = 3.0     # 3s stabil fuer COAST (war 1.5s — zu schnell)
    coast_resume_pct: float = 0.10     # Fallback (wird nicht mehr genutzt)
    min_move_speed: float = 0.15       # Minimale ONVIF-Speed bei kleinen Korrekturen

    # === PIXEL-BASIERTE SCHWELLWERTE (Markus: hier anpassen!) ===
    # Alle Schwellwerte in absoluten Pixeln (Frame 640x640)
    # frozen_threshold_px: Kein Tracking unter diesem Error — kein Micro-Ruckeln
    # coast_threshold_px:  COAST nur wenn AUCH tilt_error < diesem Wert (kein Einfrieren bei Tilt-Drift)
    # coast_resume_px:     COAST verlassen wenn error wieder groesser
    # tilt_boost_threshold_px: Tilt-Verstaerkung ab diesem Pixel-Error
    # tilt_boost_factor:   Multiplikator fuer Tilt-Delta bei grossem Error
    frozen_threshold_px: float = 20.0      # < 20px -> FROZEN (war 30 → zu frueh eingefroren)
    coast_threshold_px: float = 25.0       # COAST bei < 25px (war 40)
    coast_resume_px: float = 50.0          # COAST aufwachen bei > 50px (war 35 → Ping-Pong mit frozen)
    tilt_boost_threshold_px: float = 80.0  # Tilt-Boost ab 80px tilt-Error
    tilt_boost_factor: float = 2.0         # Tilt-Delta verdoppeln bei grossem Error


@dataclass
class DetectionData:
    """Detection data for tracking."""
    detected: bool = False
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    center_x: float = 0.5
    center_y: float = 0.5
    confidence: float = 0.0
    target_id: int = 0
    timestamp: float = field(default_factory=time.time)
    is_pose_detection: bool = False
    has_face: bool = False
    has_torso: bool = False
    head_center_x: float = 0.5
    head_center_y: float = 0.5
    validation_reason: str = ""
    target_type: str = "none"


class AutonomousTracker:
    """
    Autonomous person tracking with 5 Hz control loop.

    Uses AbsoluteMove for position-based tracking with real camera feedback.
    Replaces ContinuousMove (which was limited to 90 degrees per call).
    """

    LOOP_RATE_HZ = 5  # 200ms cycle time

    def __init__(self, camera_controller=None, config: TrackingConfig = None):
        self.camera = camera_controller
        self.config = config or TrackingConfig()

        # State
        self.state = TrackerState.IDLE
        self._prev_state = TrackerState.IDLE
        self.tracking_active = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Detection data (updated by vision system)
        self.latest_detection = DetectionData()
        self.last_detection_time = 0.0

        # Target persistence state
        self.current_target_id = 0
        self.current_target_bbox = [0, 0, 0, 0]
        self.current_target_confidence = 0.0
        self.candidate_target_id = 0
        self.candidate_stability_count = 0
        self._next_target_id = 1

        # === ADAPTIVE TARGET STATE ===
        self.current_target_type = TargetType.NONE
        self.candidate_target_type = TargetType.NONE
        self.target_type_stability = 0
        self.last_target_switch_time = 0.0

        # === SMOOTHING: Source-Typ Hysterese + Ring-Buffer ===
        self._current_source = "none"       # "face" oder "body" - aktiver Source-Typ
        self._candidate_source = "none"     # Anwaerter-Source
        self._source_stability = 0          # Frames stabil beim neuen Source
        self._center_ring = []              # Ring-Buffer: [(cx, cy), ...] letzte N Frames

        # Search mode state
        self.search_direction = 1
        self.last_direction_switch = 0.0
        self.search_patrol_index = 0
        self.search_move_time = 0.0
        self._returning_home = False  # Phase-2 Flag: langsam Home fahren
        self._visited_positions: set = set()  # Bereits abgefahrene Patrol-Positionen
        self._camera_smart_tracking_on = False  # Kamera-eigenes Smart-Tracking aktiv?
        self._last_face_save_time = 0.0  # Throttle: max alle 10s speichern

        # ST-Bewegungslernen: Kamera-Positionen aufzeichnen
        self._st_learner = STMovementLearner()

        # Motor-Learner: adaptiver Gain (beobachtet, korrigiert Basis-Gains)
        self._motor_learner = _get_motor_learner() if _MOTOR_LEARNER_AVAILABLE else None
        self._motor_learner_cycle = 0   # Zaehler fuer periodischen Gain-Update
        # Gelernte Positionen laden (persistent)
        learned = _load_learned_positions()
        if learned:
            logger.info(f"[TRACKER] {len(learned)} gelernte Patrol-Positionen geladen")
            self.config.search_patrol_positions = learned

        # G1-T04: Letzte Face-Position laden (persistent, ueberlebt Reboot)
        saved_pan, saved_tilt = _load_last_face_pos()
        self._last_tracking_pan = saved_pan
        self._last_tracking_tilt = saved_tilt
        if saved_pan != 0.0 or saved_tilt != 0.0:
            logger.info(f"[TRACKER] Letzte Face-Position geladen: "
                       f"pan={saved_pan:+.1f} tilt={saved_tilt:+.1f}")

        # === Real Camera Position (replaces virtual position) ===
        self.last_known_pan = 0.0
        self.last_known_tilt = 0.0
        self.last_position_time = 0.0
        self.last_move_time = 0.0
        # Anti-Overshoot: letztes Ziel tracken
        self._target_pan = None
        self._target_tilt = None
        self._target_arrival_thresh = 10.0  # Grad — groesserer Puffer (war 5.0), Kamera muss wirklich stehen
        # EMA Glaettung fuer smooth tracking
        self._smooth_x = None
        self._smooth_y = None
        # PD-Regler: vorherigen Fehler speichern fuer Derivative (Bremse)
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_error_time = 0.0
        # Face-Settle: kurz einfrieren wenn Gesicht frisch im Bild erscheint
        self._prev_had_face = False
        self._face_settle_start = None
        # Motor-Learner: vorherigen Fehler + Delta merken (fuer record_step naechster Cycle)
        self._ml_prev_error_x = 0.0
        self._ml_prev_error_y = 0.0
        self._ml_prev_delta_pan = 0.0
        self._ml_prev_delta_tilt = 0.0

        # === COAST MODE: Kamera einfrieren wenn Ziel stabil ===
        self._stable_start_time = None    # Wann wurde Ziel zuletzt stabil (fuer Coast-Timer)

        # === Dwell Timer State ===
        self.dwell_start_time = 0.0
        self.dwell_target_acquired = False

        # Statistics
        self.stats = {
            "cycles": 0,
            "tracking_moves": 0,
            "search_moves": 0,
            "state_changes": 0,
            "detections_filtered": 0,
            "target_switches": 0,
            "position_reads": 0
        }

        # Callbacks
        self.on_state_change: Optional[Callable[[TrackerState], None]] = None
        # Park-Modus Callback: wird mit True (parken) / False (aufwachen) aufgerufen
        # Extern setzen fuer NPU-IDLE Steuerung (nur YOLO im Park-Modus)
        self.on_park_change: Optional[Callable[[bool], None]] = None
        self._park_time: float = 0.0  # Zeitpunkt des Park-Eintritts

        # Core Integrator Referenz (fuer adaptive Tracking-Parameter)
        self._core_integrator = None
        try:
            from core.core_integrator import get_core_integrator
            self._core_integrator = get_core_integrator()
            logger.info("[TRACKER] CoreIntegrator angebunden")
        except Exception as e:
            logger.warning(f"[TRACKER] CoreIntegrator nicht verfuegbar: {e}")

        # Basis-Parameter speichern (fuer dynamische Anpassung)
        # Motor-Learner: gespeicherte Gains laden falls vorhanden
        if self._motor_learner:
            self._base_pan_gain  = self._motor_learner.get_base_pan_gain()
            self._base_tilt_gain = self._motor_learner.get_base_tilt_gain()
            logger.info(
                f"[TRACKER] Motor-Learner Gains: "
                f"pan={self._base_pan_gain:.3f} tilt={self._base_tilt_gain:.3f}"
            )
        else:
            self._base_pan_gain = self.config.pan_gain
            self._base_tilt_gain = self.config.tilt_gain
        self._base_max_step_pan = self.config.max_step_pan
        self._base_max_step_tilt = self.config.max_step_tilt
        self._base_move_cooldown = self.config.move_cooldown_ms
        self._base_tracking_speed = self.config.tracking_speed
        self._base_target_lost_timeout = self.config.target_lost_timeout

        logger.info(f"AutonomousTracker v2 (AbsoluteMove) initialized (rate={self.LOOP_RATE_HZ}Hz)")

    def set_camera(self, camera_controller):
        """Set camera controller."""
        self.camera = camera_controller
        logger.info(f"Camera controller connected to AutonomousTracker")
        if camera_controller:
            logger.info(f"  Controller id: {id(camera_controller)}")
            logger.info(f"  is_connected: {camera_controller.is_connected}")

    def start(self) -> bool:
        """Start the tracking thread."""
        if self._running:
            logger.warning("Tracker already running")
            return True

        if not self.camera:
            logger.error("No camera controller - cannot start tracker")
            return False

        # === VERIFY CONTROLLER INSTANCE ===
        logger.info("=" * 60)
        logger.info("=== TRACKER v2 START: CONTROLLER VERIFICATION ===")
        logger.info(f"self.camera:          {self.camera}")
        logger.info(f"self.camera id:       {id(self.camera)}")
        logger.info(f"is_connected:         {self.camera.is_connected}")
        logger.info(f"mode:                 {self.camera.mode}")
        logger.info("=" * 60)

        # === DIAGNOSTIC: Read initial position ===
        logger.info("=== READING INITIAL CAMERA POSITION ===")
        try:
            pos = self.camera.get_position()
            self.last_known_pan = pos.pan
            self.last_known_tilt = pos.tilt
            self.last_position_time = time.time()
            logger.info(f"Initial position: pan={pos.pan:.1f} deg, tilt={pos.tilt:.1f} deg")
        except Exception as e:
            logger.error(f"Failed to read initial position: {e}")

        # Grace Period: 5s bevor SEARCH (Modelle brauchen Zeit fuer erste Detection)
        self.last_detection_time = time.time()
        self._running = True
        self.tracking_active = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

        logger.info("AutonomousTracker v2 started")
        return True

    def stop(self):
        """Stop the tracking thread."""
        self._running = False
        self.tracking_active = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.camera:
            self.camera.stop()

        self._set_state(TrackerState.IDLE)
        logger.info(f"AutonomousTracker stopped (cycles={self.stats['cycles']})")

    def update_detection(self, detections: List[Dict], frame_width: int = 640, frame_height: int = 640):
        """
        Update with new detection data from vision system.

        Implements stable target selection:
        - Face hat IMMER Prioritaet vor Person-Box
        - Source-Typ Hysterese: Wechsel erst nach N stabilen Frames
        - Ring-Buffer Smoothing: Tracking-Zentrum ueber letzte 5 Frames gemittelt
        - Filters out hands, partial bodies, low-confidence detections
        - Maintains current target if still valid
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            frame_area = frame_width * frame_height

            if not detections:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            # === Face-Prioritaet: Face und Person getrennt behandeln ===
            face_dets = [d for d in detections if d.get("class", "") == "face"]
            person_dets = [d for d in detections if d.get("class", "") == "person"]
            incoming_source = "none"

            # Face hat IMMER Vorrang
            if face_dets:
                work_dets = face_dets
                incoming_source = "face"
            elif person_dets:
                work_dets = person_dets
                incoming_source = "body"
            else:
                work_dets = detections
                incoming_source = "body"

            # === Source-Typ Hysterese ===
            if incoming_source != self._current_source:
                if incoming_source == self._candidate_source:
                    self._source_stability += 1
                else:
                    self._candidate_source = incoming_source
                    self._source_stability = 1

                # Wechsel erst nach N stabilen Frames (AUSNAHME: face -> sofort)
                if incoming_source == "face":
                    # Face hat sofort Prioritaet - kein Warten
                    self._current_source = "face"
                    self._source_stability = 0
                    self._candidate_source = "none"
                elif self._source_stability >= self.config.source_hysteresis_frames:
                    logger.info(f"[SMOOTH] Source-Wechsel: {self._current_source} -> {incoming_source} "
                               f"(stabil seit {self._source_stability} Frames)")
                    self._current_source = incoming_source
                    self._source_stability = 0
                    self._candidate_source = "none"
                else:
                    # Noch nicht stabil genug - aktuelle Quelle beibehalten, KEIN Update
                    return
            else:
                self._source_stability = 0
                self._candidate_source = "none"

            # === STAGE 1: Filter out invalid detections ===
            valid_dets = []
            for d in work_dets:
                bbox = d.get("bbox", [0, 0, 0, 0])
                conf = d.get("confidence", 0)
                det_class = d.get("class", "person")
                is_face = (det_class == "face")

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height

                if conf < self.config.min_confidence:
                    self.stats["detections_filtered"] += 1
                    continue

                # Gate 0 Phase 3: Face-Filter stark relaxt - SCRFD Confidence reicht
                min_height = 0.03 if is_face else self.config.min_bbox_height_ratio
                min_area = 0.0005 if is_face else self.config.min_bbox_area_ratio  # war 0.002 — filterte kleine Gesichter

                height_ratio = height / frame_height
                if height_ratio < min_height:
                    self.stats["detections_filtered"] += 1
                    continue

                area_ratio = area / frame_area
                if area_ratio < min_area:
                    self.stats["detections_filtered"] += 1
                    continue

                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                # center_y Filter NUR fuer Person-BBoxen
                # Face am Bildrand = Kamera muss sich bewegen (Gate 0 Phase 3)
                if not is_face:
                    center_y_ratio = ((y1 + y2) / 2) / frame_height
                    if center_y_ratio > self.config.max_bbox_center_y_ratio:
                        self.stats["detections_filtered"] += 1
                        continue

                valid_dets.append(d)

            if not valid_dets:
                self.latest_detection = DetectionData(detected=False)
                self.candidate_stability_count = 0
                return

            # === STAGE 2: Score detections ===
            def score_detection(d):
                bbox = d.get("bbox", [0, 0, 0, 0])
                conf = d.get("confidence", 0)
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                area_score = area / frame_area
                center_x = (x1 + x2) / 2 / frame_width
                center_y = (y1 + y2) / 2 / frame_height
                dist_from_center = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
                center_score = 1.0 - min(1.0, dist_from_center * 2)
                return area_score + (center_score * self.config.center_priority_weight) + (conf * 0.2)

            scored_dets = sorted(valid_dets, key=score_detection, reverse=True)
            best_candidate = scored_dets[0]
            best_bbox = best_candidate.get("bbox", [0, 0, 0, 0])
            best_conf = best_candidate.get("confidence", 0)

            # === STAGE 3: Target persistence with hysteresis ===
            x1, y1, x2, y2 = best_bbox
            center_x = (x1 + x2) / 2 / frame_width
            center_y = (y1 + y2) / 2 / frame_height

            current_target_still_valid = False
            current_target_detection = None
            if self.current_target_id > 0 and self.current_target_confidence > 0:
                for d in valid_dets:
                    bbox = d.get("bbox", [0, 0, 0, 0])
                    if self._bbox_iou(bbox, self.current_target_bbox) > 0.3:
                        current_target_still_valid = True
                        current_target_detection = d
                        self.current_target_bbox = bbox
                        self.current_target_confidence = d.get("confidence", 0)
                        break

            best_alternative = None
            best_alternative_conf = 0.0
            for d in valid_dets:
                conf = d.get("confidence", 0)
                bbox = d.get("bbox", [0, 0, 0, 0])
                if self._bbox_iou(bbox, self.current_target_bbox) < 0.3:
                    if conf > best_alternative_conf:
                        best_alternative_conf = conf
                        best_alternative = d

            should_switch = False
            switch_to_bbox = best_bbox
            switch_to_conf = best_conf

            if not current_target_still_valid:
                should_switch = True
                self.candidate_stability_count = 0
            elif best_alternative and best_alternative_conf > self.current_target_confidence + self.config.confidence_hysteresis:
                self.candidate_stability_count += 1
                logger.debug(f"Stability count: {self.candidate_stability_count}/{self.config.stability_frames} "
                           f"(alt_conf={best_alternative_conf:.2f} vs cur={self.current_target_confidence:.2f})")
                if self.candidate_stability_count >= self.config.stability_frames:
                    should_switch = True
                    switch_to_bbox = best_alternative.get("bbox", [0, 0, 0, 0])
                    switch_to_conf = best_alternative_conf
                    logger.info(f"TARGET SWITCH: id={self.current_target_id} -> new (conf {self.current_target_confidence:.2f} -> {switch_to_conf:.2f})")
            else:
                self.candidate_stability_count = 0

            if should_switch:
                self.current_target_id = self._next_target_id
                self._next_target_id += 1
                self.current_target_bbox = switch_to_bbox
                self.current_target_confidence = switch_to_conf
                self.stats["target_switches"] += 1

            if self.current_target_id > 0:
                tx1, ty1, tx2, ty2 = self.current_target_bbox
                center_x = (tx1 + tx2) / 2 / frame_width
                center_y = (ty1 + ty2) / 2 / frame_height

            # === STAGE 4: Ring-Buffer Smoothing (letzte N Frames mitteln) ===
            buf_size = self.config.center_ring_buffer_size
            self._center_ring.append((center_x, center_y))
            if len(self._center_ring) > buf_size:
                self._center_ring = self._center_ring[-buf_size:]

            # Gewichteter Mittelwert: neuere Frames zaehlen mehr
            if len(self._center_ring) > 1:
                weights = list(range(1, len(self._center_ring) + 1))
                w_sum = sum(weights)
                smooth_cx = sum(w * c[0] for w, c in zip(weights, self._center_ring)) / w_sum
                smooth_cy = sum(w * c[1] for w, c in zip(weights, self._center_ring)) / w_sum
            else:
                smooth_cx = center_x
                smooth_cy = center_y

            self.latest_detection = DetectionData(
                detected=True,
                bbox=self.current_target_bbox,
                center_x=smooth_cx,
                center_y=smooth_cy,
                confidence=self.current_target_confidence,
                target_id=self.current_target_id,
                timestamp=time.time(),
                has_face=(self._current_source == "face"),
                target_type=self._current_source
            )
            self.last_detection_time = time.time()

            # CoreIntegrator: Target gefunden -> Presence/Proximity steigt
            if self._core_integrator:
                try:
                    self._core_integrator.update_inputs("tracker", {
                        "user_proximity": self.current_target_confidence,
                        "time_since_interaction": 0.0,  # Reset: Ziel gerade gesehen
                    })
                except Exception:
                    pass

    def _bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bboxes."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update_pose_detection(self, poses: List[Dict], frame_width: int = 640, frame_height: int = 640):
        """
        ADAPTIVE pose detection - decides between FACE and BODY tracking.

        Priority: FACE > BODY > NONE
        """
        with self._lock:
            self.config.frame_width = frame_width
            self.config.frame_height = frame_height
            now = time.time()

            if not poses:
                self.latest_detection = DetectionData(detected=False, is_pose_detection=True)
                self.candidate_stability_count = 0
                self.target_type_stability = 0
                return

            face_candidates = []
            body_candidates = []

            for p in poses:
                bbox = p.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue

                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                height_ratio = height / frame_height
                area_ratio = (width * height) / (frame_width * frame_height)
                aspect_ratio = width / height if height > 0 else 0

                if area_ratio < self.config.min_bbox_area_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                if aspect_ratio < self.config.min_aspect_ratio:
                    self.stats["detections_filtered"] += 1
                    continue
                center_y = (bbox[1] + bbox[3]) / 2 / frame_height
                if center_y > self.config.max_bbox_center_y_ratio:
                    self.stats["detections_filtered"] += 1
                    continue

                has_face = p.get("has_face", False)
                face_conf = p.get("face_confidence", 0)
                face_center = p.get("face_center")
                has_torso = p.get("has_torso", False)

                if has_face and face_conf >= self.config.face_min_confidence and face_center:
                    if height_ratio <= self.config.face_max_bbox_height:
                        face_candidates.append(p)
                    else:
                        body_candidates.append(p)
                elif has_torso and height_ratio >= self.config.body_min_bbox_height:
                    body_candidates.append(p)
                else:
                    self.stats["detections_filtered"] += 1

            selected_pose = None
            selected_type = TargetType.NONE
            track_x, track_y = 0.5, 0.5

            def score_pose(p, for_face: bool):
                face_conf = p.get("face_confidence", 0)
                det_conf = p.get("confidence", 0)
                fc = p.get("face_center", (0.5, 0.5))
                dist = math.sqrt((fc[0] - 0.5)**2 + (fc[1] - 0.5)**2)
                center_bonus = 1.0 - min(1.0, dist * 2)
                if for_face:
                    return face_conf * 0.5 + center_bonus * 0.3 + det_conf * 0.2
                else:
                    bbox = p.get("bbox", [0, 0, 0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_score = area / (frame_width * frame_height)
                    return area_score * 0.4 + center_bonus * 0.3 + det_conf * 0.3

            if self.config.prefer_current_target and self.current_target_type != TargetType.NONE:
                cooldown_ok = (now - self.last_target_switch_time) > self.config.switch_cooldown_sec
                if self.current_target_type == TargetType.FACE and face_candidates:
                    face_candidates.sort(key=lambda p: score_pose(p, True), reverse=True)
                    selected_pose = face_candidates[0]
                    selected_type = TargetType.FACE
                elif self.current_target_type == TargetType.BODY and body_candidates:
                    body_candidates.sort(key=lambda p: score_pose(p, False), reverse=True)
                    selected_pose = body_candidates[0]
                    selected_type = TargetType.BODY
                elif cooldown_ok:
                    pass

            if selected_pose is None:
                if face_candidates:
                    self.candidate_target_type = TargetType.FACE
                    self.target_type_stability += 1
                    if self.target_type_stability >= self.config.face_min_stability:
                        face_candidates.sort(key=lambda p: score_pose(p, True), reverse=True)
                        selected_pose = face_candidates[0]
                        selected_type = TargetType.FACE
                        if self.current_target_type != TargetType.FACE:
                            logger.info(f"[ADAPTIVE] Switching to FACE tracking (stability={self.target_type_stability})")
                            self.last_target_switch_time = now
                elif body_candidates:
                    self.candidate_target_type = TargetType.BODY
                    self.target_type_stability += 1
                    if self.target_type_stability >= self.config.body_min_stability:
                        body_candidates.sort(key=lambda p: score_pose(p, False), reverse=True)
                        selected_pose = body_candidates[0]
                        selected_type = TargetType.BODY
                        if self.current_target_type != TargetType.BODY:
                            logger.info(f"[ADAPTIVE] Switching to BODY tracking (stability={self.target_type_stability})")
                            self.last_target_switch_time = now
                else:
                    self.target_type_stability = 0

            if selected_pose is None:
                self.latest_detection = DetectionData(detected=False, is_pose_detection=True)
                return

            self.current_target_type = selected_type

            bbox = selected_pose.get("bbox", [0, 0, 0, 0])
            if selected_type == TargetType.FACE:
                # Prioritaet 1: Nose-Keypoint, Prioritaet 2: Face-Center
                nose = selected_pose.get("nose_center")
                if nose and nose[1] > 0:
                    track_x, track_y = nose
                else:
                    face_center = selected_pose.get("face_center", (0.5, 0.5))
                    track_x, track_y = face_center
            else:
                # Body-Tracking: Kopf zentrieren
                # Prioritaet 1: Nose-Keypoint (wenn sichtbar)
                nose = selected_pose.get("nose_center")
                if nose and nose[1] > 0:
                    track_x, track_y = nose
                else:
                    # Prioritaet 3: Kopfbereich der Person-Box (~8% von oben)
                    bbox_center_x = (bbox[0] + bbox[2]) / 2 / frame_width
                    bbox_top_y = bbox[1] / frame_height
                    bbox_bottom_y = bbox[3] / frame_height
                    bbox_height = bbox_bottom_y - bbox_top_y
                    track_x = bbox_center_x
                    track_y = bbox_top_y + bbox_height * 0.08

            if self.current_target_id == 0:
                self.current_target_id = self._next_target_id
                self._next_target_id += 1

            self.current_target_bbox = bbox
            self.current_target_confidence = selected_pose.get("face_confidence", 0) if selected_type == TargetType.FACE else selected_pose.get("confidence", 0)

            # Ring-Buffer Smoothing (gleiche Logik wie update_detection Stage 4)
            buf_size = self.config.center_ring_buffer_size
            self._center_ring.append((track_x, track_y))
            if len(self._center_ring) > buf_size:
                self._center_ring = self._center_ring[-buf_size:]

            if len(self._center_ring) > 1:
                weights = list(range(1, len(self._center_ring) + 1))
                w_sum = sum(weights)
                smooth_tx = sum(w * c[0] for w, c in zip(weights, self._center_ring)) / w_sum
                smooth_ty = sum(w * c[1] for w, c in zip(weights, self._center_ring)) / w_sum
            else:
                smooth_tx = track_x
                smooth_ty = track_y

            self.latest_detection = DetectionData(
                detected=True,
                bbox=bbox,
                center_x=smooth_tx,
                center_y=smooth_ty,
                confidence=self.current_target_confidence,
                target_id=self.current_target_id,
                timestamp=time.time(),
                is_pose_detection=True,
                has_face=selected_pose.get("has_face", False),
                has_torso=selected_pose.get("has_torso", False),
                head_center_x=track_x,
                head_center_y=track_y,
                validation_reason=selected_pose.get("validation_reason", ""),
                target_type=selected_type.value
            )
            self.last_detection_time = time.time()

            if self.stats["cycles"] % 30 == 0:
                logger.info(f"[ADAPTIVE] {selected_type.value.upper()} at ({track_x:.2f},{track_y:.2f}) "
                           f"conf={self.current_target_confidence:.2f} "
                           f"faces={len(face_candidates)} bodies={len(body_candidates)}")

    # =========================================================================
    # Camera Position Reading (ONVIF)
    # =========================================================================

    def _read_camera_position(self):
        """Liest echte Kameraposition via ONVIF GET.

        Wird alle 2 Cycles (~400ms) aufgerufen. Ohne echte Position
        wuerde der Anti-Overshoot-Check nie feuern und Befehle stapeln sich.
        """
        if not self.camera or not self.camera.is_connected:
            return
        try:
            pos = self.camera.get_position()
            old_pan = self.last_known_pan
            old_tilt = self.last_known_tilt
            self.last_known_pan = pos.pan
            self.last_known_tilt = pos.tilt
            self.last_position_time = time.time()
            self.stats["position_reads"] += 1

            # ST-Learner: Kamera-Bewegungen aufzeichnen wenn ST aktiv
            if self._camera_smart_tracking_on:
                self._st_learner.record(pos.pan, pos.tilt)

            # Drift erkennen und Target-Cache invalidieren wenn zu gross
            drift = abs(old_pan - pos.pan) + abs(old_tilt - pos.tilt)
            if drift > 5.0:
                # Grosser Drift: Target-Cache invalidieren, neue Befehle erlauben
                self._target_pan = None
                self._target_tilt = None
                self._target_wait_start = None
                ptz_debug.warning(
                    f"POS_DRIFT pan={pos.pan:+.1f} tilt={pos.tilt:+.1f} "
                    f"drift={drift:.1f} > 5.0 — cache invalidiert"
                )
            elif drift > 2.0 and self.stats["position_reads"] % 5 == 0:
                ptz_debug.info(
                    f"POS_READ pan={pos.pan:+.1f} tilt={pos.tilt:+.1f} "
                    f"(drift={drift:.1f} von cached ({old_pan:+.1f},{old_tilt:+.1f}))"
                )
        except Exception as e:
            if self.stats["cycles"] % 50 == 0:
                logger.warning(f"[TRACKER] Position read failed: {e}")

    # =========================================================================
    # Core Integrator Adaption
    # =========================================================================

    def _adapt_from_integrator(self):
        """Tracking-Parameter dynamisch an CoreIntegrator State anpassen.

        Hohe Attention (> 0.8): Kamera ruhig, praezises Tracking, wenig Bewegung
        Niedrige Attention (< 0.3): Kamera unruhiger, mehr Scan-Bewegungen
        Hohe Tension: Schnellere Reaktion, aggressiveres Tracking
        Niedrige Tension: Sanftere Bewegungen, mehr Coast-Zeit
        """
        if not self._core_integrator:
            return

        try:
            effects = self._core_integrator.get_effects()
            attention = self._core_integrator.get_attention()
            tension = self._core_integrator.get_tension()
        except Exception:
            return

        camera_stability = effects.get("camera_stability", 0.5)
        micro_ptz = effects.get("micro_ptz_movement", 0.2)

        # --- Attention-basierte Anpassung ---
        # Hohe Attention -> ruhigere Kamera (kleinere Schritte, laengere Cooldowns)
        # Niedrige Attention -> unruhigere Kamera (groessere Schritte, kuerzere Cooldowns)
        stability_factor = camera_stability  # 0.0 (unruhig) - 1.0 (stabil)

        # Pan/Tilt Gain: Bei hoher Stabilitaet weniger aggressiv
        self.config.pan_gain = self._base_pan_gain * (0.5 + (1.0 - stability_factor) * 0.8)
        self.config.tilt_gain = self._base_tilt_gain * (0.5 + (1.0 - stability_factor) * 0.8)

        # Max Step: Bei hoher Stabilitaet kleinere Schritte
        self.config.max_step_pan = self._base_max_step_pan * (0.4 + (1.0 - stability_factor) * 0.8)
        self.config.max_step_tilt = self._base_max_step_tilt * (0.4 + (1.0 - stability_factor) * 0.8)

        # --- Tension-basierte Anpassung ---
        # Hohe Tension -> schnellere Reaktion
        tension_speed = 0.7 + tension * 0.6  # 0.7 (ruhig) bis 1.3 (hektisch)

        # Move Cooldown: Bei hoher Tension kuerzere Pausen
        self.config.move_cooldown_ms = self._base_move_cooldown / tension_speed

        # Tracking Speed: Bei hoher Tension schneller
        self.config.tracking_speed = min(1.0, self._base_tracking_speed * tension_speed)

        # Target Lost Timeout: Bei hoher Tension schneller in Search
        self.config.target_lost_timeout = self._base_target_lost_timeout / tension_speed

    # =========================================================================
    # Tracking Loop
    # =========================================================================

    def _tracking_loop(self):
        """Main 5 Hz tracking loop."""
        cycle_time = 1.0 / self.LOOP_RATE_HZ
        logger.info(f"Tracking loop STARTED (rate={self.LOOP_RATE_HZ}Hz)")

        while self._running:
            loop_start = time.time()

            try:
                if self.tracking_active:
                    # Alle 5 Zyklen (~1x/s): Parameter vom CoreIntegrator anpassen
                    if self.stats["cycles"] % 5 == 0:
                        self._adapt_from_integrator()

                    self._process_tracking_cycle()
                    self.stats["cycles"] += 1

                    if self.stats["cycles"] % 15 == 0:
                        logger.info(f"Tracker loop: cycles={self.stats['cycles']} state={self.state.value} "
                                  f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg "
                                  f"search={self.stats['search_moves']} track={self.stats['tracking_moves']}")

            except Exception as e:
                logger.error(f"Tracking cycle error: {e}")
                import traceback
                logger.error(traceback.format_exc())

            elapsed = time.time() - loop_start
            sleep_time = cycle_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_tracking_cycle(self):
        """Process one tracking cycle.

        3-Phasen Verlust-Logik:
          Phase 0: Detection aktiv     -> SOFORT tracken (Person/Gesicht)
          Phase 1: 0-5s ohne Detection -> STEHEN BLEIBEN an letzter Position
          Phase 2: 5-30s              -> langsam Home fahren, dort warten
          Phase 3: >30s               -> Idle-Suche (langsames Patrol)

        ABSOLUTE REGEL: Wenn Person im Bild -> NIEMALS Search starten.
        """
        # Echte Kameraposition alle 2 Cycles (~400ms) via ONVIF lesen
        if self.stats["cycles"] % 2 == 0:
            self._read_camera_position()

        with self._lock:
            detection = self.latest_detection

        now = time.time()
        time_since_detection = now - self.last_detection_time

        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[CYCLE] detected={detection.detected} "
                       f"time_since={time_since_detection:.2f}s "
                       f"conf={detection.confidence:.2f} "
                       f"state={self.state.value} "
                       f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

        # === PARKED: Keine Bewegung, nur auf Detection warten ===
        if self.state == TrackerState.PARKED:
            if detection.detected and time_since_detection < 0.5:
                # YOLO hat Person gemeldet -> Aufwachen!
                park_duration = now - self._park_time if self._park_time > 0 else 0
                logger.info(f"[PARK] AUFGEWACHT! Person erkannt nach {park_duration:.0f}s Park-Modus")
                self._set_state(TrackerState.TRACKING)
                # NPU zurueck auf Vollbetrieb
                if self.on_park_change:
                    try:
                        self.on_park_change(False)
                    except Exception as e:
                        logger.error(f"[PARK] on_park_change(False) Fehler: {e}")
                self._do_tracking(detection)
            return

        # === PHASE 0: Detection aktiv -> Smart-Handover entscheiden ===
        if detection.detected and time_since_detection < 0.5:
            # Search/Patrol SOFORT abbrechen wenn Person im Bild
            if self.state == TrackerState.SEARCHING:
                logger.info("[CYCLE] Person erkannt waehrend Search -> SOFORT tracken!")
                if self.camera:
                    self.camera.stop()

            # Smart-Handover: Moloch uebernimmt NUR wenn noetig
            moloch_should_track = self._should_moloch_track(detection)
            if moloch_should_track:
                self._do_tracking(detection)

                # Auto-ST: Wenn MOLOCH die BBox nicht zentriert kriegt → ST einschalten
                off_center = max(abs(detection.center_x - 0.5), abs(detection.center_y - 0.5))
                # Cooldown: nach ST-AUS hat Moloch _ST_COOLDOWN_S Zeit sich einzupendeln
                st_off_time = getattr(self, '_st_deactivate_time', 0.0)
                in_cooldown = (time.time() - st_off_time) < self._ST_COOLDOWN_S
                if off_center > self._ST_AUTO_ERROR_THRESHOLD and not self._camera_smart_tracking_on and not in_cooldown:
                    self._st_auto_fail_count = getattr(self, '_st_auto_fail_count', 0) + 1
                    if self._st_auto_fail_count >= self._ST_AUTO_CYCLES:
                        logger.info(f"[AUTO-ST] BBox {self._st_auto_fail_count}x off-center "
                                   f"({off_center:.2f} > {self._ST_AUTO_ERROR_THRESHOLD}) "
                                   f"→ Kamera-ST einschalten (Sensoren schneller)")
                        self._enable_camera_smart_tracking(True)
                        self._st_auto_fail_count = 0
                else:
                    # BBox zentriert oder ST schon an → Counter reset
                    self._st_auto_fail_count = 0
            else:
                # Kamera-ST laeuft gut, Moloch beobachtet nur
                if not self._camera_smart_tracking_on:
                    self._enable_camera_smart_tracking(True)
                    logger.info("[HANDOVER] Kamera-ST uebernimmt (Person stabil mittig)")
                self._returning_home = False
                self.last_detection_time = time.time()
            return

        # === Kein Target: 3-Phasen Verlust-Logik ===

        # Phase 1: 0-5s -> STEHEN BLEIBEN (halt_wait_timeout)
        if time_since_detection <= self.config.halt_wait_timeout:
            if self.state == TrackerState.SEARCHING:
                # Laufende Search stoppen — Person war gerade noch da
                if self.camera:
                    self.camera.stop()
                self._set_state(TrackerState.COAST)
            if debug_log:
                logger.info(f"[CYCLE] Phase 1: HALT ({time_since_detection:.1f}s < {self.config.halt_wait_timeout}s)")
            self._do_coast()
            return

        # Phase 2: 5s+ -> Kamera-Smart-Tracking AN + zur letzten Face-Position
        # Sonoff-Sensoren scannen den Raum, Moloch wartet auf YOLO-Detection
        if not getattr(self, '_returning_home', False):
            self._returning_home = True
            # Letzte Tracking-Position speichern BEVOR Kamera-ST uebernimmt
            self._last_tracking_pan = self.last_known_pan
            self._last_tracking_tilt = self.last_known_tilt

            # Kamera-ST aktivieren: Sonoff-Sensoren + interner Motor uebernehmen
            self._enable_camera_smart_tracking(True)

            # Kamera zur letzten Face-Position fahren (dort sind Sensoren am relevantesten)
            target_pan = self._last_tracking_pan
            target_tilt = max(self.config.tilt_limit_min,
                            self._last_tracking_tilt)  # Nicht unter Tilt-Limit
            logger.info(f"[CYCLE] Phase 2: Kamera-ST AN + fahre zu letzter Position "
                       f"({target_pan:+.1f},{target_tilt:+.1f}) "
                       f"nach {time_since_detection:.1f}s ohne Detection")
            if self.camera and self.camera.is_connected:
                self.camera.move_absolute(target_pan, target_tilt, speed=0.15)

        # Phase 3: >30s -> PARK (NPU IDLE), aber Kamera-ST bleibt AN
        if time_since_detection > self.config.home_return_timeout:
            if self.state != TrackerState.PARKED:
                logger.info(f"[PARK] Geparkt — Kamera-ST scannt weiter, "
                           f"Moloch wartet auf YOLO-Detection")
                self._park_time = time.time()
                self._set_state(TrackerState.PARKED)
                if self.on_park_change:
                    try:
                        self.on_park_change(True)
                    except Exception as e:
                        logger.error(f"[PARK] on_park_change(True) Fehler: {e}")
                if self._core_integrator:
                    try:
                        self._core_integrator.update_input("tracker", "user_proximity", 0.0)
                        self._core_integrator.update_input("tracker", "time_since_interaction", 1.0)
                    except Exception:
                        pass
        elif debug_log:
            logger.info(f"[CYCLE] Phase 2: Kamera-ST aktiv, warte auf YOLO ({time_since_detection:.1f}s)")

    # =========================================================================
    # Tracking (AbsoluteMove-based)
    # =========================================================================

    def _do_tracking(self, detection: DetectionData):
        """Execute tracking: SOFORT auf Person locken, proportional + PD-Regler.

        ABSOLUTE REGEL: Person im Bild -> Kamera folgt. Sofort. Kein Dwell.
        YOLO Person -> Kamera folgt. SCRFD Gesicht -> praeziser zentrieren.
        """
        now = time.time()

        # Home-Return Flag zuruecksetzen — Person gefunden
        self._returning_home = False
        # Kamera-Smart-Tracking aus — Moloch uebernimmt wieder
        if self._camera_smart_tracking_on:
            self._enable_camera_smart_tracking(False)
            # Gelernte Positionen aktualisieren und speichern
            learned = self._st_learner.get_patrol_positions()
            if learned:
                self.config.search_patrol_positions = learned
                _save_learned_positions(learned)
                logger.info(f"[ST-LEARN] {len(learned)} Patrol-Positionen gelernt: "
                           f"{[(p,t) for p,t in learned[:3]]}...")
        # G1-T04: Tracking-Position laufend aktualisieren (fuer Suchrichtung bei Verlust)
        self._last_tracking_pan = self.last_known_pan
        self._last_tracking_tilt = self.last_known_tilt

        # Face erkannt? Position persistent speichern (max alle 10s, SSD-schonend)
        if detection.has_face and now - self._last_face_save_time > 10.0:
            self._last_face_save_time = now
            _save_last_face_pos(self.last_known_pan, self.last_known_tilt)

        # EMA Glaettung: smooth detection center (kein Ruckeln/Springen)
        alpha = self.config.smooth_alpha
        if self._smooth_x is None:
            self._smooth_x = detection.center_x
            self._smooth_y = detection.center_y
        else:
            self._smooth_x = (1 - alpha) * self._smooth_x + alpha * detection.center_x
            self._smooth_y = (1 - alpha) * self._smooth_y + alpha * detection.center_y

        # Calculate error from frame center (pixels) - mit geglaetteten Werten
        center_x_px = self._smooth_x * self.config.frame_width
        center_y_px = self._smooth_y * self.config.frame_height
        frame_center_x = self.config.frame_width / 2
        # Gesicht naeher an Bildmitte (40% statt 33% — war zu hoch, Person nicht zentriert)
        frame_center_y = self.config.frame_height * 0.40

        error_x = center_x_px - frame_center_x  # Positive = target RIGHT of center
        error_y = center_y_px - frame_center_y  # Positive = target BELOW center
        error_magnitude = math.sqrt(error_x**2 + error_y**2)

        # Normalized error - geglaettet
        # WICHTIG: error_y_norm muss GLEICHE Referenz wie frame_center_y nutzen (0.40)!
        error_x_norm = self._smooth_x - 0.5
        error_y_norm = self._smooth_y - 0.40  # Gesicht zur Bildmitte (war 0.33)

        # PTZ Debug: raw + smooth Position + Error bei jedem Cycle
        ptz_debug.debug(
            f"DETECT raw=({detection.center_x:.3f},{detection.center_y:.3f}) "
            f"smooth=({self._smooth_x:.3f},{self._smooth_y:.3f}) "
            f"err=({error_x_norm:+.3f},{error_y_norm:+.3f}) "
            f"cam=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg"
        )

        debug_log = (self.stats["cycles"] % 15 == 0)
        if debug_log:
            logger.info(f"[TRACK] error=({error_x:+.0f},{error_y:+.0f})px mag={error_magnitude:.0f}px "
                       f"state={self.state.value} pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

        # === FACE-SETTLE: Wenn Gesicht frisch im Person-BBox erscheint → kurz einfrieren ===
        # Gibt der Face-BBox Zeit sich zu stabilisieren bevor wir auf sie korrigieren.
        just_got_face = detection.has_face and not self._prev_had_face
        self._prev_had_face = detection.has_face
        if just_got_face:
            self._face_settle_start = now
            logger.debug("[FACE-SETTLE] Gesicht frisch erkannt — Settle-Timer gestartet")
        if self._face_settle_start is not None:
            if now - self._face_settle_start < self.config.face_settle_time:
                ptz_debug.debug(
                    f"FACE_SETTLE {now - self._face_settle_start:.2f}s "
                    f"< {self.config.face_settle_time:.2f}s — kein Move"
                )
                return  # BBox stabilisiert sich, keine Kamerabewegung
            else:
                self._face_settle_start = None  # Settle abgeschlossen

        # SOFORT in TRACKING State — kein Dwell, kein Warten
        if self.state != TrackerState.TRACKING and self.state != TrackerState.FROZEN and self.state != TrackerState.COAST:
            self._set_state(TrackerState.TRACKING)
            self.dwell_target_acquired = True

        # === Error-Magnitude als Prozent vom Bild (fuer Dead Zone / Coast) ===
        error_magnitude_pct = math.sqrt(error_x_norm**2 + error_y_norm**2)

        # === COAST MODE: Kamera komplett eingefroren wenn Ziel stabil ===
        if self.state == TrackerState.COAST:
            if error_magnitude > self.config.coast_resume_px:
                # Ziel hat sich signifikant bewegt -> Tracking aufnehmen
                self._set_state(TrackerState.TRACKING)
                self._stable_start_time = None
                logger.info(f"[COAST] Aufgewacht! error={error_magnitude:.0f}px > {self.config.coast_resume_px:.0f}px")
            else:
                # Stabil -> nichts tun
                if debug_log:
                    ptz_debug.debug(f"COAST still error={error_magnitude:.0f}px < {self.config.coast_resume_px:.0f}px")
                return

        # === DEAD ZONE: < frozen_threshold_px -> keine Kamerabewegung ===
        if error_magnitude < self.config.frozen_threshold_px:
            if self.state not in (TrackerState.FROZEN, TrackerState.COAST):
                self._set_state(TrackerState.FROZEN)
                ptz_debug.debug(
                    f"FROZEN error={error_magnitude:.0f}px < {self.config.frozen_threshold_px:.0f}px"
                )

            # Coast nur aktivieren wenn GESAMTER Error klein (nicht nur Tilt!)
            if error_magnitude < self.config.coast_threshold_px:
                if self._stable_start_time is None:
                    self._stable_start_time = now
                elif (now - self._stable_start_time) >= self.config.coast_stable_time:
                    self._set_state(TrackerState.COAST)
                    logger.info(
                        f"[COAST] Aktiviert - stabil {self.config.coast_stable_time:.0f}s, "
                        f"error={error_magnitude:.0f}px < {self.config.coast_threshold_px:.0f}px"
                    )
            else:
                # Grosser Error -> kein Coast, Timer zurueck
                self._stable_start_time = None
                ptz_debug.debug(
                    f"COAST_BLOCKED: error={error_magnitude:.0f}px > {self.config.coast_threshold_px:.0f}px"
                )
            return

        # === HYSTERESE ZONE: 3-5% -> nur bewegen wenn bereits TRACKING ===
        if error_magnitude_pct < self.config.track_start_pct:
            if self.state in (TrackerState.FROZEN, TrackerState.LOCKED, TrackerState.COAST):
                # In der Hysteresezone bleiben wir still
                if debug_log:
                    ptz_debug.debug(
                        f"HYSTERESIS error={error_magnitude_pct:.3f} "
                        f"zone=[{self.config.dead_zone_pct},{self.config.track_start_pct}]"
                    )
                return
            # Wenn bereits TRACKING -> weitermachen (weiche Abbremsung)

        # Aus Dead Zone raus -> Timer zuruecksetzen
        self._stable_start_time = None

        # === TRACKING MODE: AbsoluteMove ===
        self._set_state(TrackerState.TRACKING)

        # Cooldown check
        time_since_move = (now - self.last_move_time) * 1000  # ms
        if time_since_move < self.config.move_cooldown_ms:
            return

        # === STUCK-AT-LIMIT Erkennung: Kamera am mechanischen Anschlag?  ===
        # Wenn Kamera > 8s am Pan/Tilt-Limit UND Error treibt weiter in die Grenze
        # → wahrscheinlich Artefakt-Detection, kein echtes Ziel → SEARCH starten
        pan_at_min = self.last_known_pan <= self.config.pan_limit_min + 3.0
        pan_at_max = self.last_known_pan >= self.config.pan_limit_max - 3.0
        tilt_at_max = self.last_known_tilt >= self.config.tilt_limit_max + 20.0  # Physiklimit > SW-Limit
        error_drives_into_pan_limit = (pan_at_min and error_x_norm > 0.15) or (pan_at_max and error_x_norm < -0.15)
        error_drives_into_tilt_limit = tilt_at_max and error_y_norm < -0.10
        stuck_at_limit = error_drives_into_pan_limit or error_drives_into_tilt_limit

        if stuck_at_limit:
            if not getattr(self, '_stuck_limit_start', None):
                self._stuck_limit_start = now
            elif now - self._stuck_limit_start > 8.0:
                logger.warning(
                    f"[STUCK-LIMIT] >8s am Anschlag pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) "
                    f"err_x={error_x_norm:+.3f} err_y={error_y_norm:+.3f} → SEARCH starten"
                )
                self._stuck_limit_start = None
                self._smooth_x = None  # EMA-Filter zuruecksetzen
                self._smooth_y = None
                self._set_state(TrackerState.SEARCHING)
                return
        else:
            self._stuck_limit_start = None

        # === POSITIONS-FROZEN: Kamera physisch blockiert (Anschlag unter SW-Limit)? ===
        # Prüft ob last_known_pan sich >8s nicht um mehr als 1° ändert bei grossem Error.
        # Greift wenn physikalischer Anschlag (z.B. +108°) unter SW-Limit (170°) liegt.
        if error_magnitude > 150:
            prev_pan = getattr(self, '_posfrozen_last_pan', None)
            posfrozen_start = getattr(self, '_posfrozen_start', None)
            if prev_pan is None or abs(self.last_known_pan - prev_pan) > 1.0:
                self._posfrozen_last_pan = self.last_known_pan
                self._posfrozen_start = now
            elif posfrozen_start is not None and now - posfrozen_start > 8.0:
                logger.warning(
                    f"[POS-FROZEN] >8s keine Bewegung bei error={error_magnitude:.0f}px "
                    f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) → SEARCH"
                )
                self._posfrozen_last_pan = None
                self._posfrozen_start = None
                self._smooth_x = None
                self._smooth_y = None
                self._set_state(TrackerState.SEARCHING)
                return
        else:
            self._posfrozen_last_pan = None
            self._posfrozen_start = None

        # Anti-Overshoot: warte bis Kamera am letzten Ziel angekommen ist
        if self._target_pan is not None:
            dist = abs(self.last_known_pan - self._target_pan) + abs(self.last_known_tilt - self._target_tilt)
            if dist > self._target_arrival_thresh:
                if not hasattr(self, '_target_wait_start') or self._target_wait_start is None:
                    self._target_wait_start = time.time()
                elif time.time() - self._target_wait_start > 5.0:
                    ptz_debug.warning(
                        f"WAIT TIMEOUT target=({self._target_pan:+.1f},{self._target_tilt:+.1f}) "
                        f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) dist={dist:.1f} - clearing"
                    )
                    self._target_pan = None
                    self._target_tilt = None
                    self._target_wait_start = None
                else:
                    ptz_debug.debug(
                        f"WAIT target=({self._target_pan:+.1f},{self._target_tilt:+.1f}) "
                        f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) dist={dist:.1f}"
                    )
                    return
            else:
                self._target_wait_start = None

        # === TILT-KORREKTUR: Body ohne Face -> nach oben (Kopf suchen) ===
        # Wenn nur Body/Hand erkannt wird, liegt der BBox-Center zu tief.
        # Kopf ist ca. 15% der BBox-Hoehe ueber dem Center -> Tilt-Bias nach oben
        tilt_up_bias = 0.0
        if not detection.has_face and self._current_source == "body":
            tilt_up_bias = 0.08  # ~8% vom Bild nach oben = Kopf-Richtung

        # === PD-REGLER: Proportional + Derivative (Bremse) ===
        # P-Term: proportional zum Fehler
        # VORZEICHEN: positiver error_x_norm = Gesicht RECHTS → Kamera muss RECHTS (Pan negativ)
        # Daher: -error_x_norm (analog zu Tilt, wo -error_y_norm korrekt war)
        pan_p = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
        tilt_p = -(error_y_norm - tilt_up_bias) * self.config.fov_vertical * self.config.tilt_gain

        # D-Term: Bremse wenn Fehler ABNIMMT (Kamera naehert sich dem Ziel)
        d_gain = 0.4  # Daempfungsfaktor (0 = kein Bremsen, 1 = starkes Bremsen)
        dt = now - self._prev_error_time if self._prev_error_time > 0 else 0.2
        if dt > 0 and dt < 2.0:  # Nur bei plausiblem Zeitintervall
            d_error_x = (error_x_norm - self._prev_error_x) / dt
            d_error_y = (error_y_norm - self._prev_error_y) / dt
            # D-Term: Aenderungsrate * Gain -> bremst ab wenn Fehler schrumpft
            # VORZEICHEN: gleich wie P-Term (negiert, analog zu Tilt)
            pan_d = -d_error_x * self.config.fov_horizontal * d_gain * self.config.pan_gain
            tilt_d = -d_error_y * self.config.fov_vertical * d_gain * self.config.tilt_gain
        else:
            pan_d = 0.0
            tilt_d = 0.0

        # Fehler-Historie aktualisieren
        self._prev_error_x = error_x_norm
        self._prev_error_y = error_y_norm
        self._prev_error_time = now

        # PD-Summe: P bringt uns zum Ziel, D bremst beim Ankommen
        pan_delta = pan_p + pan_d
        tilt_delta = tilt_p + tilt_d
        pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
        tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        # Tilt-Boost: Wenn Pixel-Tilt-Error dauerhaft gross -> Delta verstaerken
        # Sicherheitsnetz: greift auch wenn error_y_norm zu klein skaliert
        if abs(error_y) > self.config.tilt_boost_threshold_px:
            tilt_delta *= self.config.tilt_boost_factor
            tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))
            ptz_debug.debug(
                f"TILT_BOOST: error_y={error_y:+.0f}px > {self.config.tilt_boost_threshold_px:.0f}px "
                f"→ tilt_delta={tilt_delta:+.2f}deg"
            )

        if abs(pan_delta) < self.config.min_step_deg and abs(tilt_delta) < self.config.min_step_deg:
            if debug_log:
                logger.info(f"[TRACK] delta below threshold (pan={pan_delta:+.2f}, tilt={tilt_delta:+.2f})")
            return

        # === ADAPTIVE SPEED: proportional zum Error ===
        # Kleine Korrekturen (15-20%) -> langsam, grosse (>30%) -> volle Speed
        speed_range = self.config.tracking_speed - self.config.min_move_speed
        move_speed = self.config.min_move_speed + speed_range * min(1.0, error_magnitude_pct / 0.30)

        # Execute AbsoluteMove tracking (mit bereits berechnetem PD-Delta)
        result = self._track_target(error_x_norm, error_y_norm, speed=move_speed,
                                     pd_pan_delta=pan_delta, pd_tilt_delta=tilt_delta)

        if result:
            self.stats["tracking_moves"] += 1
            # TESTER-Log: pan_delta, tilt_delta, smoothed_delta bei jedem Move
            ptz_debug.info(
                f"MOVE_SENT smooth=({self._smooth_x:.3f},{self._smooth_y:.3f}) "
                f"pan_delta={pan_delta:+.2f}deg tilt_delta={tilt_delta:+.2f}deg "
                f"speed={move_speed:.2f} err_pct={error_magnitude_pct:.3f}"
            )

            # Motor-Learner: vorherigen Cycle auswerten (post_error = aktueller error)
            if self._motor_learner and self._ml_prev_delta_pan != 0.0:
                self._motor_learner.record_step(
                    self._ml_prev_error_x, self._ml_prev_error_y,
                    self._ml_prev_delta_pan, self._ml_prev_delta_tilt,
                    error_x_norm, error_y_norm
                )
                # Alle 100 Moves: Basis-Gains aus Motor-Learner uebernehmen
                self._motor_learner_cycle += 1
                if self._motor_learner_cycle >= 100:
                    self._motor_learner_cycle = 0
                    self._base_pan_gain  = self._motor_learner.get_base_pan_gain()
                    self._base_tilt_gain = self._motor_learner.get_base_tilt_gain()

            # Aktuellen Fehler + Delta fuer naechsten Cycle merken
            self._ml_prev_error_x  = error_x_norm
            self._ml_prev_error_y  = error_y_norm
            self._ml_prev_delta_pan  = pan_delta
            self._ml_prev_delta_tilt = tilt_delta

        if self.stats["tracking_moves"] % 15 == 0:
            logger.info(f"TRACK: err=({error_x:+.0f},{error_y:+.0f})px err_pct={error_magnitude_pct:.3f} "
                       f"speed={move_speed:.2f} delta=({pan_delta:+.1f},{tilt_delta:+.1f})deg "
                       f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f})deg")

    def _track_target(self, error_x_norm: float, error_y_norm: float, speed: float = None,
                       pd_pan_delta: float = None, pd_tilt_delta: float = None) -> bool:
        """
        Track target using AbsoluteMove with real position feedback.

        Args:
            error_x_norm: Normalized horizontal error (-0.5 to +0.5), positive = target right
            error_y_norm: Normalized vertical error (-0.5 to +0.5), positive = target below
            speed: ONVIF move speed (0.0-1.0), None = config.tracking_speed
            pd_pan_delta: Vorberechnetes PD-Delta fuer Pan (optional)
            pd_tilt_delta: Vorberechnetes PD-Delta fuer Tilt (optional)

        Returns:
            True if move command sent successfully
        """
        if not self.camera or not self.camera.is_connected:
            return False

        # Check exclusive PTZ lock
        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        # PTZ Arbiter: Darf MOLOCH jetzt einen Befehl senden?
        try:
            from core.ptz_arbiter import get_ptz_arbiter
            arbiter = get_ptz_arbiter()
            if not arbiter.may_send_ptz():
                return False
            arbiter.record_takeover_reason()
        except Exception:
            pass

        # PD-Delta nutzen wenn vorhanden, sonst reiner P-Term als Fallback
        if pd_pan_delta is not None and pd_tilt_delta is not None:
            pan_delta = pd_pan_delta
            tilt_delta = pd_tilt_delta
        else:
            pan_delta = -error_x_norm * self.config.fov_horizontal * self.config.pan_gain
            tilt_delta = -error_y_norm * self.config.fov_vertical * self.config.tilt_gain
            pan_delta = max(-self.config.max_step_pan, min(self.config.max_step_pan, pan_delta))
            tilt_delta = max(-self.config.max_step_tilt, min(self.config.max_step_tilt, tilt_delta))

        # Calculate target position + Soft-Limit Clamping
        # 2 Grad INNERHALB der Hardware-Limits (war 10 = zu viel Range-Verlust)
        LIMIT_BUFFER = 2.0
        soft_pan_min = self.config.pan_limit_min + LIMIT_BUFFER
        soft_pan_max = self.config.pan_limit_max - LIMIT_BUFFER
        soft_tilt_min = self.config.tilt_limit_min + LIMIT_BUFFER
        soft_tilt_max = self.config.tilt_limit_max - LIMIT_BUFFER
        target_pan = max(soft_pan_min, min(soft_pan_max, self.last_known_pan + pan_delta))
        target_tilt = max(soft_tilt_min, min(soft_tilt_max, self.last_known_tilt + tilt_delta))

        # Wenn schon am Soft-Limit und Move wuerde weiter in Richtung Limit gehen -> abbrechen
        if (self.last_known_pan <= soft_pan_min and pan_delta < 0) or \
           (self.last_known_pan >= soft_pan_max and pan_delta > 0):
            ptz_debug.warning(
                f"LIMIT_BLOCK pan={self.last_known_pan:+.1f} delta={pan_delta:+.1f} "
                f"soft_range=[{soft_pan_min:+.1f},{soft_pan_max:+.1f}]"
            )
            return False

        # PTZ Debug: Vollstaendige Berechnung loggen
        face_side = "LINKS" if error_x_norm < 0 else "RECHTS"
        face_vert = "OBEN" if error_y_norm < 0 else "UNTEN"
        cam_pan_dir = "LINKS(+)" if pan_delta > 0 else "RECHTS(-)"
        cam_tilt_dir = "HOCH(+)" if tilt_delta > 0 else "RUNTER(-)"
        ptz_debug.info(
            f"MOVE err_norm=({error_x_norm:+.3f},{error_y_norm:+.3f}) "
            f"Gesicht={face_side}/{face_vert} | "
            f"pan_delta={pan_delta:+.1f} ({cam_pan_dir}) tilt_delta={tilt_delta:+.1f} ({cam_tilt_dir}) | "
            f"pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) -> "
            f"target=({target_pan:+.1f},{target_tilt:+.1f})deg"
        )

        # Adaptive Speed: uebergeben oder Default
        move_speed = speed if speed is not None else self.config.tracking_speed

        # SonoffCameraController.move_absolute() clamps to calibrated limits internally
        result = self.camera.move_absolute(target_pan, target_tilt, speed=move_speed)

        if result:
            self.last_move_time = time.time()
            self._target_pan = target_pan
            self._target_tilt = target_tilt
            # KEIN sofortiges Position-Caching! Kamera ist noch unterwegs.
            # last_known_pan/tilt wird NUR durch _read_camera_position() aktualisiert.
            # Anti-Overshoot-Wait (Z.998) blockiert neue Befehle bis Kamera ankommt.

            total_moves = self.stats["tracking_moves"] + self.stats["search_moves"]
            if total_moves % 15 == 0:
                logger.info(f"[TRACKER] AbsoluteMove: pos=({self.last_known_pan:+.1f},{self.last_known_tilt:+.1f}) "
                           f"-> target=({target_pan:+.1f},{target_tilt:+.1f})deg")

        return result

    # =========================================================================
    # Search Mode (AbsoluteMove patrol)
    # =========================================================================

    def _do_search(self):
        """Idle-Suche: Langsames Patrol nach >30s ohne Detection.

        ABSOLUTE REGEL: Wenn waehrend Search eine Detection reinkommt,
        wird Search in _process_tracking_cycle() SOFORT abgebrochen.
        Hier nur die Patrol-Logik fuer Phase 3.

        Search Speed ist LANGSAM (0.15), Positionswechsel alle 6s.
        """
        # Smoothing + Coast-Timer + PD-State zuruecksetzen wenn Ziel verloren
        self._smooth_x = None
        self._smooth_y = None
        self._stable_start_time = None
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._prev_error_time = 0.0

        # Sofort-Check: Person sichtbar laut Perception -> KEIN Search
        if PERCEPTION_AVAILABLE:
            try:
                if is_user_visible():
                    if self.state == TrackerState.SEARCHING:
                        logger.info("[SEARCH] Abbruch: user_visible=True in perception")
                    self._do_coast()
                    return
            except Exception:
                pass

        now = time.time()

        # === START SEARCH: Reset and begin patrol ===
        if self.state != TrackerState.SEARCHING:
            self._set_state(TrackerState.SEARCHING)
            self.search_move_time = now
            self._search_start_time = now
            self._visited_positions.clear()  # Neue Suche, alles reset

            # G1-T04: Suchrichtung = Richtung wo Person ZULETZT GESEHEN wurde
            # _last_tracking_pan wird in Phase 2 gespeichert BEVOR Home faehrt
            last_pan = self._last_tracking_pan
            positions = self.config.search_patrol_positions
            nearest_idx = 0
            min_dist = float('inf')
            for i, (pp, _pt) in enumerate(positions):
                dist = abs(pp - last_pan)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            self.search_patrol_index = nearest_idx

            logger.info(f"[SEARCH] Suche gestartet bei letzter Tracking-Pan={last_pan:+.1f} "
                       f"(start=Pos[{nearest_idx}]={positions[nearest_idx][0]:+.1f}, "
                       f"{len(positions)} Positionen)")

            # Reset dwell state for next target acquisition
            self.dwell_target_acquired = False
            self.dwell_start_time = 0.0

            # CoreIntegrator: Presence sinkt beim Suchen
            if self._core_integrator:
                try:
                    self._core_integrator.update_input("tracker", "time_since_interaction", 0.5)
                except Exception:
                    pass
            return

        # === PATROL: Positionen abfahren, besuchte ueberspringen ===
        search_duration = now - getattr(self, '_search_start_time', now)
        patrol_positions = self.config.search_patrol_positions

        # === ALLE POSITIONEN ABGEFAHREN -> Home + Park ===
        if len(self._visited_positions) >= len(patrol_positions):
            if self.state != TrackerState.PARKED:
                logger.info(f"[SEARCH] Alle {len(patrol_positions)} Positionen abgefahren, "
                           f"nichts gefunden -> Park bei Tuer "
                           f"(Dauer: {search_duration:.0f}s)")
                # G1-T06: Park-Position = Tuer
                if self.camera and self.camera.is_connected:
                    self.camera.move_absolute(self.config.park_pan, self.config.park_tilt, speed=0.15)
                self._park_time = now
                self._set_state(TrackerState.PARKED)
                # NPU auf IDLE-Stufe
                if self.on_park_change:
                    try:
                        self.on_park_change(True)
                    except Exception as e:
                        logger.error(f"[PARK] on_park_change(True) Fehler: {e}")
                # CoreIntegrator: Presence komplett auf 0
                if self._core_integrator:
                    try:
                        self._core_integrator.update_input("tracker", "user_proximity", 0.0)
                        self._core_integrator.update_input("tracker", "time_since_interaction", 1.0)
                    except Exception:
                        pass
            return

        # === PARK-MODUS: Fallback nach search_park_timeout (180s) ===
        if search_duration > self.config.search_park_timeout:
            if self.state != TrackerState.PARKED:
                logger.info(f"[PARK] {self.config.search_park_timeout:.0f}s ohne Detection "
                           f"-> Park bei Tuer (NPU IDLE)")
                if self.camera and self.camera.is_connected:
                    self.camera.move_absolute(self.config.park_pan, self.config.park_tilt, speed=0.15)
                self._park_time = now
                self._set_state(TrackerState.PARKED)
                if self.on_park_change:
                    try:
                        self.on_park_change(True)
                    except Exception as e:
                        logger.error(f"[PARK] on_park_change(True) Fehler: {e}")
                if self._core_integrator:
                    try:
                        self._core_integrator.update_input("tracker", "user_proximity", 0.0)
                        self._core_integrator.update_input("tracker", "time_since_interaction", 1.0)
                    except Exception:
                        pass
            return

        # Patrol-Position wechseln alle search_direction_interval Sekunden
        time_at_position = now - self.search_move_time
        if time_at_position >= self.config.search_direction_interval:
            # Aktuelle Position als besucht markieren
            self._visited_positions.add(self.search_patrol_index)

            # G1-T04: Naechste UNBESUCHTE Position — sortiert nach Naehe
            # zur letzten Tracking-Pan (Fluchtrichtung zuerst)
            ref_pan = self._last_tracking_pan
            candidates = [
                (i, abs(patrol_positions[i][0] - ref_pan))
                for i in range(len(patrol_positions))
                if i not in self._visited_positions
            ]
            candidates.sort(key=lambda x: x[1])  # Naechste zuerst
            next_idx = candidates[0][0] if candidates else None

            if next_idx is None:
                # Alle besucht — naechster Cycle-Aufruf geht in den Park-Block oben
                return

            self.search_patrol_index = next_idx
            target_pan, target_tilt = patrol_positions[self.search_patrol_index]

            remaining = len(patrol_positions) - len(self._visited_positions)
            logger.info(f"[SEARCH] Position [{self.search_patrol_index}/{len(patrol_positions)}] "
                       f"-> ({target_pan:+.1f},{target_tilt:+.1f}) "
                       f"(noch {remaining} uebrig, seit {search_duration:.0f}s)")

            self._send_search_move(target_pan, target_tilt)

            # CoreIntegrator: Presence sinkt langsam beim Suchen
            if self._core_integrator:
                try:
                    decay = min(1.0, search_duration / 60.0)
                    self._core_integrator.update_input("tracker", "time_since_interaction", decay)
                except Exception:
                    pass

    def _calc_search_speed(self, target_pan: float, target_tilt: float) -> float:
        """Distanzabhaengige Speed: kurze Wege langsam, weite schneller.

        Lineare Interpolation zwischen search_speed_min und search_speed_max
        basierend auf der Winkeldistanz (0-240 Grad mapped auf min-max).
        """
        delta_pan = abs(target_pan - self.last_known_pan)
        delta_tilt = abs(target_tilt - self.last_known_tilt)
        distance = (delta_pan**2 + delta_tilt**2) ** 0.5

        # 0 Grad -> min speed, 240 Grad (max moegliche Distanz) -> max speed
        t = min(1.0, distance / 240.0)
        speed = self.config.search_speed_min + t * (self.config.search_speed_max - self.config.search_speed_min)
        return round(speed, 3)

    def _send_search_move(self, pan_deg: float, tilt_deg: float) -> bool:
        """Send AbsoluteMove for search/patrol mit distanzabhaengiger Speed."""
        if not self.camera or not self.camera.is_connected:
            return False

        if hasattr(self.camera, '_exclusive_owner') and self.camera._exclusive_owner is not None:
            return False

        # PTZ Arbiter: Darf MOLOCH jetzt einen Befehl senden?
        try:
            from core.ptz_arbiter import get_ptz_arbiter
            if not get_ptz_arbiter().may_send_ptz():
                return False
        except Exception:
            pass

        speed = self._calc_search_speed(pan_deg, tilt_deg)
        result = self.camera.move_absolute(pan_deg, tilt_deg, speed=speed)

        if result:
            self.search_move_time = time.time()
            self.last_move_time = time.time()
            self.stats["search_moves"] += 1
            logger.debug(f"[SEARCH] Move -> ({pan_deg:+.1f},{tilt_deg:+.1f}) speed={speed:.3f}")

        return result

    # Smart-Tracking Handover: Kamera-Bewegung beobachten
    _ST_SETTLE_THRESHOLD = 2.0   # Grad — weniger Bewegung = Kamera hat sich beruhigt
    _ST_SETTLE_FRAMES = 3        # N aufeinanderfolgende stabile Reads = "settled"
    _ST_MIN_TIME = 1.5           # Absolute Mindestzeit (Sicherheit)

    # Auto-ST-Aktivierung: Wenn MOLOCH-Tracking BBox nicht zentriert bekommt
    _ST_AUTO_ERROR_THRESHOLD = 0.35   # 35% off-center = wirklich weit weg (war 0.25 → zu schnell)
    _ST_AUTO_CYCLES = 20              # 20 Cycles = ~4s (war 10 → griff zu frueh ein)
    _ST_COOLDOWN_S = 10.0             # Cooldown nach ST-AUS: Moloch hat 10s Zeit sich einzupendeln

    # Hysterese: ST erst nach N aufeinanderfolgenden face-losen Frames aktivieren
    _ST_NO_FACE_THRESHOLD = 8         # 8 Frames ohne Face (~0.4s) bevor ST uebernimmt

    def _should_moloch_track(self, detection: DetectionData) -> bool:
        """Entscheidet ob Moloch selbst tracken soll oder Kamera-ST laufen laesst.

        KERNREGEL: Kein Gesicht erkannt → ST bleibt an, Moloch beobachtet.
        Erst bei Face-Detection uebernimmt Moloch fuer Praezision.

        Kamera-ST ist schneller (Hardware-Sensoren, interner Motor).
        Moloch ist praeziser (BBox-Zentrierung, Face-Tracking).
        """
        # === NEUE LOGIK: Kein Face → ST soll laufen (mit Hysterese) ===
        if not detection.has_face:
            # Hysterese: erst nach N aufeinanderfolgenden face-losen Frames ST einschalten
            self._no_face_count = getattr(self, '_no_face_count', 0) + 1
            if self._no_face_count >= self._ST_NO_FACE_THRESHOLD:
                if not self._camera_smart_tracking_on:
                    logger.info(
                        f"[HANDOVER] {self._no_face_count}x kein Face → ST einschalten"
                    )
                    self._enable_camera_smart_tracking(True)
            # ST laeuft (oder wartet noch), Moloch beobachtet
            return False

        # === Ab hier: Face erkannt → Moloch-Praezision gefragt ===
        # Hysterese-Counter zuruecksetzen
        self._no_face_count = 0

        # ST war nie an → Moloch trackt direkt
        if not self._camera_smart_tracking_on:
            return True

        # ST ist an + Face erkannt → Uebergang zu Moloch
        now = time.time()
        st_duration = now - getattr(self, '_st_activate_time', 0.0)

        # Mindestzeit — Sonoff braucht kurz
        if st_duration < self._ST_MIN_TIME:
            return False

        # Kamera-Bewegung pruefen: hat sie sich beruhigt?
        pan_delta = abs(self.last_known_pan - getattr(self, '_st_prev_pan', self.last_known_pan))
        tilt_delta = abs(self.last_known_tilt - getattr(self, '_st_prev_tilt', self.last_known_tilt))
        movement = pan_delta + tilt_delta

        self._st_prev_pan = self.last_known_pan
        self._st_prev_tilt = self.last_known_tilt

        if movement < self._ST_SETTLE_THRESHOLD:
            self._st_settle_count = getattr(self, '_st_settle_count', 0) + 1
        else:
            self._st_settle_count = 0
            return False

        camera_settled = self._st_settle_count >= self._ST_SETTLE_FRAMES
        if not camera_settled:
            return False

        # Kamera settled + Face erkannt → Moloch uebernimmt
        logger.info(f"[HANDOVER] Face erkannt + Kamera settled ({st_duration:.1f}s) → Moloch uebernimmt")
        self._enable_camera_smart_tracking(False)
        return True

    def _enable_camera_smart_tracking(self, on: bool):
        """Sonoff-eigenes Smart-Tracking — DEAKTIVIERT.

        Smart Tracking kaempft mit Moloch-Tracking und verursacht:
        - Nervöses Kamera-Hin-und-Her (Toggle-Schleife)
        - Mögliche RTSP-Stream-Störungen bei jedem Umschalten
        - Szenario-Wechsel die Valve-Transitions triggern
        Bleibt AUS bis Moloch-Tracking stabil genug ist (2026-03-30).
        """
        # ST komplett deaktiviert — Moloch trackt allein
        if self._camera_smart_tracking_on:
            try:
                if self.camera and hasattr(self.camera, 'cloud_bridge') and self.camera.cloud_bridge:
                    self.camera.cloud_bridge.set_smart_tracking(False)
                    logger.info("[SMART-TRACK] Kamera Smart-Tracking AUS (permanent deaktiviert)")
            except Exception:
                pass
            self._camera_smart_tracking_on = False
        return

    def _do_coast(self):
        """Coast - stop movement when target briefly lost."""
        # With AbsoluteMove, camera naturally stops at the last commanded position.
        # Only send explicit stop if transitioning from a search/patrol state.
        if self.state == TrackerState.SEARCHING:
            if self.camera:
                self.camera.stop()

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, new_state: TrackerState):
        """Update state with logging."""
        if new_state != self.state:
            old_state = self.state
            self._prev_state = old_state
            self.state = new_state
            self.stats["state_changes"] += 1

            perception_info = ""
            if PERCEPTION_AVAILABLE:
                try:
                    ps = get_perception_state()
                    snap = ps.get_snapshot()
                    perception_info = f" | perception: user={snap.user_visible}, face={snap.face_visible}, gesture={snap.gesture_type}"
                except Exception:
                    pass

            logger.info(f"[TRACKER STATE] {old_state.value} -> {new_state.value}{perception_info}")

            if self.on_state_change:
                self.on_state_change(new_state)

    def enable(self):
        """Enable tracking."""
        self.tracking_active = True
        logger.info("Tracking ENABLED")

    def disable(self):
        """Disable tracking and stop movement."""
        self.tracking_active = False
        if self.camera:
            self.camera.stop()
        self.dwell_target_acquired = False
        self.dwell_start_time = 0.0
        self._set_state(TrackerState.IDLE)
        logger.info("Tracking DISABLED")

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        with self._lock:
            detection = self.latest_detection

        return {
            "state": self.state.value,
            "tracking_active": self.tracking_active,
            "running": self._running,
            "current_target": {
                "id": self.current_target_id,
                "confidence": self.current_target_confidence,
                "bbox": self.current_target_bbox
            },
            "latest_detection": {
                "detected": detection.detected,
                "center": (detection.center_x, detection.center_y),
                "confidence": detection.confidence,
                "target_id": detection.target_id,
                "age_ms": int((time.time() - detection.timestamp) * 1000)
            },
            "camera_position": {
                "pan_deg": self.last_known_pan,
                "tilt_deg": self.last_known_tilt,
                "position_age_ms": int((time.time() - self.last_position_time) * 1000) if self.last_position_time > 0 else -1
            },
            "dwell": {
                "target_acquired": self.dwell_target_acquired,
                "elapsed_sec": time.time() - self.dwell_start_time if self.dwell_target_acquired else 0
            },
            "park": {
                "parked": self.state == TrackerState.PARKED,
                "parked_since_sec": int(time.time() - self._park_time) if self.state == TrackerState.PARKED and self._park_time > 0 else 0,
                "park_timeout_sec": self.config.search_park_timeout
            },
            "camera_smart_tracking": self._camera_smart_tracking_on,
            "st_learner": self._st_learner.get_stats(),
            "stats": self.stats.copy(),
            "config": {
                "lock_threshold_px": self.config.lock_threshold_pixels,
                "frozen_threshold_px": self.config.frozen_threshold_pixels,
                "dwell_time_sec": self.config.dwell_time_sec,
                "pan_gain": self.config.pan_gain,
                "tilt_gain": self.config.tilt_gain,
                "fov_h": self.config.fov_horizontal,
                "fov_v": self.config.fov_vertical,
                "max_step_pan": self.config.max_step_pan,
                "max_step_tilt": self.config.max_step_tilt,
                "tracking_speed": self.config.tracking_speed,
                "min_bbox_height": self.config.min_bbox_height_ratio
            }
        }


# Singleton instance
_tracker: Optional[AutonomousTracker] = None
_tracker_lock = threading.Lock()


def get_autonomous_tracker() -> AutonomousTracker:
    """Get or create singleton tracker instance."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = AutonomousTracker()
    return _tracker
