#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Router — Model Scheduler
==================================================
Situationsbasiertes Modell-Scheduling fuer Hailo-10H NPU.

7 Szenarien bestimmen welche Modelle aktiv sind:
- IDLE:     Keine Person >30s → nur YOLO
- FERN:     Person bbox_height < 20% → YOLO + ReID + Pose
- MITTEL:   Person bbox_height 20-45% → YOLO + SCRFD + ArcFace + FaceAttr + Pose
- NAH:      Person bbox_height > 45% → SCRFD + ArcFace + FaceAttr + Hand
- RUECKEN:  Person aber kein Face >3s → YOLO + ReID + Pose (SCRFD-Probe)
- MULTI:    >1 Person → YOLO + SCRFD + ArcFace + ReID + Pose
- NACHT:    IDLE >30min UND 23:00-07:00 → alles aus

Alle Modelle bleiben permanent im 8GB NPU-RAM.
Valve-Elemente schalten Branches ein/aus.
Wu-Wei: Nur was noetig ist, feuert.

Autor: Claude Opus (Architekt)
Stand: 2026-03-27
"""

import time
import logging
from datetime import datetime

logger = logging.getLogger("moloch.scheduler")

# --- Szenarien ---
SCENARIO_IDLE = "IDLE"
SCENARIO_FERN = "FERN"
SCENARIO_MITTEL = "MITTEL"
SCENARIO_NAH = "NAH"
SCENARIO_RUECKEN = "RUECKEN"
SCENARIO_MULTI = "MULTI"
SCENARIO_NACHT = "NACHT"

# --- Aktive Modelle pro Szenario ---
SCENARIO_MODELS = {
    SCENARIO_IDLE:     frozenset({"yolo"}),
    SCENARIO_FERN:     frozenset({"yolo", "reid", "pose"}),
    SCENARIO_MITTEL:   frozenset({"yolo", "scrfd", "arcface", "faceattr", "pose"}),
    SCENARIO_NAH:      frozenset({"scrfd", "arcface", "faceattr", "hand"}),
    SCENARIO_RUECKEN:  frozenset({"yolo", "reid", "pose"}),
    SCENARIO_MULTI:    frozenset({"yolo", "scrfd", "arcface", "reid", "pose"}),
    SCENARIO_NACHT:    frozenset(),
}

# --- Szenario-Prioritaet (hoeher = mehr Modelle aktiv) ---
_SCENARIO_PRIORITY = {
    SCENARIO_NACHT: 0,
    SCENARIO_IDLE: 1,
    SCENARIO_FERN: 2,
    SCENARIO_RUECKEN: 3,
    SCENARIO_MITTEL: 4,
    SCENARIO_NAH: 5,
    SCENARIO_MULTI: 6,
}

# --- Schwellwerte ---
IDLE_TIMEOUT_S = 30.0        # Keine Person → IDLE nach 30s
NACHT_TIMEOUT_S = 1800.0     # IDLE → NACHT nach 30min
NACHT_START_H = 23           # Nachtmodus ab 23:00
NACHT_END_H = 7              # Nachtmodus bis 07:00
RUECKEN_NO_FACE_S = 3.0      # Kein Gesicht → RUECKEN nach 3s
HYSTERESE_DOWN_S = 3.0       # Downgrade-Verzoegerung
BBOX_HEIGHT_FERN = 0.20      # < 20% = FERN
BBOX_HEIGHT_NAH = 0.45       # > 45% = NAH
SCRFD_PROBE_CYCLE_S = 2.0    # RUECKEN: SCRFD-Probe alle 2s
SCRFD_PROBE_DURATION_S = 0.5 # RUECKEN: SCRFD-Probe fuer 0.5s


class ModelScheduler:
    """Situationsbasierter NPU Model Scheduler.

    Wird 1x pro Sekunde (oder pro Frame) mit aktuellen Perception-Daten
    gefuettert und gibt das aktuelle Szenario + aktive Modelle zurueck.
    """

    def __init__(self):
        now = time.time()
        self._current_scenario = SCENARIO_IDLE
        self._last_person_seen = 0.0
        self._last_face_seen = 0.0
        self._last_upgrade_time = now
        self._last_tick_time = now
        self._tick_count = 0
        self._scrfd_probe_active = False
        logger.info("[SCHED] ModelScheduler initialisiert — Start: IDLE")

    def tick(self, person_count: int, face_detected: bool,
             bbox_height_pct: float, time_of_day: int = -1) -> str:
        """Szenario basierend auf aktuellen Perception-Daten bestimmen.

        Args:
            person_count: Anzahl erkannter Personen (YOLO)
            face_detected: Mindestens ein Gesicht erkannt (SCRFD)
            bbox_height_pct: Groesste Person-BBox Hoehe als Anteil (0.0-1.0)
            time_of_day: Aktuelle Stunde (0-23), -1 = auto

        Returns:
            Aktuelles Szenario (IDLE/FERN/MITTEL/NAH/RUECKEN/MULTI/NACHT)
        """
        now = time.time()
        self._tick_count += 1

        if time_of_day < 0:
            time_of_day = datetime.now().hour

        # Fallback: Gesicht erkannt = Person vorhanden (fuer NAH-Modus ohne YOLO)
        # Verhindert NAH→IDLE Oszillation wenn YOLO aus ist aber SCRFD noch feuert
        if person_count == 0 and face_detected:
            person_count = 1

        # Timer aktualisieren
        if person_count > 0:
            self._last_person_seen = now
        if face_detected:
            self._last_face_seen = now

        # Szenario bestimmen
        idle_duration = now - self._last_person_seen if self._last_person_seen > 0 else 999.0
        no_face_duration = now - self._last_face_seen if self._last_face_seen > 0 else 999.0
        is_night = (time_of_day >= NACHT_START_H or time_of_day < NACHT_END_H)

        # NACHT: IDLE >30min UND Nachtzeit
        if idle_duration > NACHT_TIMEOUT_S and is_night:
            new_scenario = SCENARIO_NACHT
        # IDLE: Keine Person >30s
        elif idle_duration > IDLE_TIMEOUT_S:
            new_scenario = SCENARIO_IDLE
        # MULTI: >1 Person
        elif person_count > 1:
            new_scenario = SCENARIO_MULTI
        # Einzelperson-Szenarien
        elif person_count >= 1:
            # RUECKEN: Person aber kein Face >3s
            if not face_detected and no_face_duration > RUECKEN_NO_FACE_S:
                new_scenario = SCENARIO_RUECKEN
            # NAH: Person nah dran
            elif bbox_height_pct > BBOX_HEIGHT_NAH:
                new_scenario = SCENARIO_NAH
            # MITTEL: Person in mittlerer Distanz
            elif bbox_height_pct > BBOX_HEIGHT_FERN:
                new_scenario = SCENARIO_MITTEL
            # FERN: Person weit weg
            else:
                new_scenario = SCENARIO_FERN
        else:
            new_scenario = SCENARIO_IDLE

        # Hysterese: Downgrade nur nach 3s Stabilitaet
        if self._is_downgrade(new_scenario):
            if (now - self._last_upgrade_time) < HYSTERESE_DOWN_S:
                new_scenario = self._current_scenario

        # Szenario-Wechsel
        if new_scenario != self._current_scenario:
            old = self._current_scenario
            self._current_scenario = new_scenario
            if not self._is_downgrade_from_to(old, new_scenario):
                self._last_upgrade_time = now
            logger.info(
                f"[SCHED] {old} → {new_scenario} "
                f"(persons={person_count}, face={face_detected}, "
                f"height={bbox_height_pct:.2f}, hour={time_of_day})"
            )

        # RUECKEN: SCRFD-Probe Logik (alle 2s fuer 0.5s)
        if self._current_scenario == SCENARIO_RUECKEN:
            cycle_pos = now % SCRFD_PROBE_CYCLE_S
            self._scrfd_probe_active = (cycle_pos < SCRFD_PROBE_DURATION_S)
        else:
            self._scrfd_probe_active = False

        self._last_tick_time = now
        return self._current_scenario

    def get_scenario(self) -> str:
        """Aktuelles Szenario zurueckgeben."""
        return self._current_scenario

    def get_active_models(self) -> frozenset:
        """Aktive Modelle fuer aktuelles Szenario."""
        return SCENARIO_MODELS.get(self._current_scenario, frozenset({"yolo"}))

    def is_model_active(self, model_name: str) -> bool:
        """Pruefen ob ein bestimmtes Modell im aktuellen Szenario aktiv ist."""
        return model_name in self.get_active_models()

    def get_scrfd_probe_needed(self) -> bool:
        """RUECKEN-Szenario: Soll SCRFD kurz aktiviert werden?"""
        return self._scrfd_probe_active

    def _is_downgrade(self, new_scenario: str) -> bool:
        """Ist der Wechsel zum neuen Szenario ein Downgrade?"""
        return self._is_downgrade_from_to(self._current_scenario, new_scenario)

    @staticmethod
    def _is_downgrade_from_to(old: str, new: str) -> bool:
        """Ist der Wechsel von old nach new ein Downgrade (weniger Modelle)?"""
        old_prio = _SCENARIO_PRIORITY.get(old, 1)
        new_prio = _SCENARIO_PRIORITY.get(new, 1)
        return new_prio < old_prio

    def get_status_dict(self) -> dict:
        """Status-Dict fuer IPC/JSON."""
        return {
            "scenario": self._current_scenario,
            "active_models": sorted(self.get_active_models()),
            "scrfd_probe": self._scrfd_probe_active,
            "tick_count": self._tick_count,
        }
