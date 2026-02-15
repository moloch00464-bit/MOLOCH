#!/usr/bin/env python3
"""Patch: PerceptionEngine einbauen - Scoring-basierte NPU Slot-Rotation."""
import sys
import os

# =====================================================
# STEP 1: Create core/perception_engine.py
# =====================================================
perception_module = '''\
#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Engine
================================
Scoring-basierte NPU Modell-Rotation.

Hardware: Hailo-10H NPU (40 TOPS), max 2 Modelle gleichzeitig.
Slot 1 = Basis (SCRFD oder YOLO), bleibt stabil.
Slot 2 = Rotiert nach Scoring-Logik.

Personality Engine (Guardian/Shadow/Emergentis) steuert Gewichtung.
Tension aus dem bestehenden System als Modulator.
"""
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger("PerceptionEngine")


class PerceptionEngine:
    """NPU Slot-Management mit Personality-gesteuertem Scoring."""

    # Modelle die in Slot 2 rotieren koennen
    SLOT2_CANDIDATES = ["arcface", "yolov8m", "pose"]

    # Statische Basis-Scores (Ausgangswert ohne Kontext)
    BASE_SCORES = {
        "arcface": 0.5,     # Gesichtserkennung - mittlere Prioritaet
        "yolov8m": 0.4,     # Person Detection - wenn nicht als Basis
        "pose": 0.3,        # Pose Estimation - niedrigere Prioritaet
    }

    def __init__(self, personality_engine=None):
        self._personality = personality_engine
        self.slot_1 = "scrfd"           # Default Basis-Modell
        self.slot_2 = None              # Wird beim ersten tick gesetzt
        self._scores = {}               # Aktuelle Scores pro Modell
        self._last_rotation = 0.0       # Zeitpunkt letzter Wechsel
        self._last_active = {}          # model -> timestamp (Anti-Starvation)
        self._min_interval = 10.0       # Min. Sekunden zwischen Rotationen
        self._hysteresis = 0.15         # Score-Differenz fuer Wechsel
        self._forced_slot2 = None       # Manueller Override
        self._enabled = True            # Engine aktiv?

        logger.info(
            f"[PERCEPTION] Init: Slot 1={self.slot_1}, "
            f"Personality={'ja' if personality_engine else 'nein'}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def tick(self, context: Dict) -> Optional[str]:
        """Pro Inference-Zyklus aufrufen. Gibt neues Slot-2-Modell zurueck oder None.

        Args:
            context: {
                "face_detected": bool,
                "person_detected": bool,
                "unknown_person": bool,
                "motion_level": float (0.0-1.0),
            }

        Returns:
            Modellname fuer Slot 2 wenn Wechsel empfohlen, sonst None.
        """
        if not self._enabled:
            return None

        # Manueller Override hat Vorrang
        if self._forced_slot2:
            if self._forced_slot2 != self.slot_2:
                old = self.slot_2
                self.slot_2 = self._forced_slot2
                logger.info(
                    f"[PERCEPTION] Force: {old} -> {self.slot_2}"
                )
                return self.slot_2
            return None

        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores

        # Kandidaten (alles ausser Slot 1)
        candidates = {k: v for k, v in scores.items() if k != self.slot_1}
        if not candidates:
            return None

        best = max(candidates, key=candidates.get)
        best_score = candidates[best]

        # Schon aktiv?
        if best == self.slot_2:
            return None

        # Hysterese: Wechsel nur bei deutlichem Unterschied
        current_score = candidates.get(self.slot_2, 0.0)
        if best_score - current_score < self._hysteresis:
            return None

        # Timing: Nicht zu haeufig wechseln
        now = time.time()
        if now - self._last_rotation < self._min_interval:
            return None

        # Wechsel empfehlen
        old = self.slot_2
        self._last_rotation = now
        if self.slot_2:
            self._last_active[self.slot_2] = now
        self.slot_2 = best

        logger.info(
            f"[PERCEPTION] Rotation: {old} -> {best} "
            f"(score={best_score:.2f}, old={current_score:.2f})"
        )
        return best

    def set_base_model(self, name: str):
        """Slot 1 (Basis) setzen."""
        if name != self.slot_1:
            logger.info(f"[PERCEPTION] Basis: {self.slot_1} -> {name}")
            self.slot_1 = name

    def force_slot2(self, name: Optional[str]):
        """Manueller Override fuer Slot 2. None = zurueck zu Scoring."""
        self._forced_slot2 = name
        if name:
            logger.info(f"[PERCEPTION] Force Slot 2: {name}")
        else:
            logger.info("[PERCEPTION] Force aufgehoben -> Scoring")

    def set_enabled(self, enabled: bool):
        """Engine aktivieren/deaktivieren."""
        self._enabled = enabled

    def get_state(self) -> Dict:
        """Status fuer GUI/Debug."""
        tension = 0.0
        mode = "standalone"
        if self._personality:
            tension = getattr(self._personality, "_last_tension", 0.0)
            mode = getattr(self._personality, "mode", None)
            if mode:
                mode = mode.value

        return {
            "enabled": self._enabled,
            "slot_1": self.slot_1,
            "slot_2": self.slot_2,
            "forced": self._forced_slot2,
            "scores": {k: round(v, 3) for k, v in self._scores.items()},
            "tension": round(tension, 3),
            "personality_mode": mode,
            "min_interval": round(self._min_interval, 1),
        }

    # =========================================================================
    # Scoring
    # =========================================================================

    def _compute_scores(self, ctx: Dict) -> Dict[str, float]:
        """Berechne Scores fuer alle Slot-2-Kandidaten."""
        scores = dict(self.BASE_SCORES)

        # --- Kontext-Boosts ---

        # Gesicht erkannt -> ArcFace hoch (identifizieren!)
        if ctx.get("face_detected"):
            scores["arcface"] += 0.4

        # Person erkannt -> Pose interessant
        if ctx.get("person_detected"):
            scores["pose"] += 0.3

        # Unbekannte Person -> ArcFace dringend (wer ist das?)
        if ctx.get("unknown_person"):
            scores["arcface"] += 0.3

        # Viel Bewegung -> YOLOv8m (was passiert da?)
        motion = ctx.get("motion_level", 0.0)
        if motion > 0.5:
            scores["yolov8m"] += 0.2

        # --- Personality-Gewichtung ---

        if self._personality:
            if self._personality.is_guardian:
                # Guardian: Sicherheit > Spielerei
                scores["arcface"] *= 1.3
                scores["pose"] *= 0.7
            elif self._personality.is_shadow:
                # Shadow: Neugier > Kontrolle
                scores["pose"] *= 1.3
                scores["arcface"] *= 0.9

            # Tension-Modulator
            tension = getattr(self._personality, "_last_tension", 0.0)

            if tension > 0.6:
                # Hohe Spannung -> Gesichter pruefen!
                scores["arcface"] += 0.3

            if tension > 0.3:
                # Mittlere Spannung -> haeufiger rotieren
                self._min_interval = max(5.0, 10.0 - tension * 8)
            else:
                # Niedrige Spannung -> ruhig bleiben
                self._min_interval = 10.0

        # --- Anti-Starvation ---

        now = time.time()
        for model in self.SLOT2_CANDIDATES:
            last = self._last_active.get(model, now)
            idle_mins = (now - last) / 60.0
            scores[model] += min(idle_mins * 0.1, 0.3)

        return scores
'''

perception_path = "/home/molochzuhause/moloch/core/perception_engine.py"
with open(perception_path, "w") as f:
    f.write(perception_module)
print(f"STEP 1: Created {perception_path}")

# =====================================================
# STEP 2: Patch moloch_service.py - PerceptionEngine Init
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    code = f.read()

changes = 0

# PATCH 2a: Add PerceptionEngine init after Age+Gender init
old_age_gender_init = """        # Age + Gender Detection (CPU, kein NPU)
        self._age_gender_detector = None
        try:
            from core.vision.age_gender_detector import get_age_gender_detector
            self._age_gender_detector = get_age_gender_detector()
            if self._age_gender_detector and self._age_gender_detector.available:
                logger.info("[INIT] Age+Gender Detection bereit (Caffe CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Age+Gender Detection nicht verfuegbar: {e}")"""

new_age_gender_init = """        # Age + Gender Detection (CPU, kein NPU)
        self._age_gender_detector = None
        try:
            from core.vision.age_gender_detector import get_age_gender_detector
            self._age_gender_detector = get_age_gender_detector()
            if self._age_gender_detector and self._age_gender_detector.available:
                logger.info("[INIT] Age+Gender Detection bereit (Caffe CPU)")
        except Exception as e:
            logger.warning(f"[INIT] Age+Gender Detection nicht verfuegbar: {e}")

        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None
        try:
            from core.perception_engine import PerceptionEngine
            self._perception = PerceptionEngine()
            logger.info("[INIT] Perception Engine bereit")
        except Exception as e:
            logger.warning(f"[INIT] Perception Engine nicht verfuegbar: {e}")"""

if old_age_gender_init in code:
    code = code.replace(old_age_gender_init, new_age_gender_init, 1)
    changes += 1
    print("PATCH 2a: PerceptionEngine init added")
else:
    print("ERROR: Could not find age_gender init block")
    sys.exit(1)

# PATCH 2b: Add perception tick in inference loop (after all detections, before frame output)
# Find the point AFTER emotion+age/gender detection but BEFORE the TTS announce
old_announce = """                            # TTS Ansage (60s Cooldown pro Person)
                            if name != "Unbekannt" and name != "Keine DB":"""

new_announce = """                            # Perception Engine: Slot-2-Rotation
                            if self._perception:
                                _perc_ctx = {
                                    "face_detected": len(face_boxes) > 0,
                                    "person_detected": len(person_boxes) > 0 if 'person_boxes' in dir() else False,
                                    "unknown_person": name == "Unbekannt",
                                    "motion_level": 0.0,
                                }
                                _recommended = self._perception.tick(_perc_ctx)
                                if _recommended:
                                    # Slot 2 wechseln (im Hintergrund)
                                    def _do_rotation(old_model, new_model):
                                        try:
                                            if old_model and old_model in self._active_ctx:
                                                self._unconfigure_model(old_model)
                                                time.sleep(0.2)
                                            if new_model not in self._active_ctx:
                                                self._configure_model(new_model)
                                        except Exception as e:
                                            logger.error(f"[PERCEPTION] Rotation failed: {e}")
                                    _old_slot2 = [m for m in self._active_ctx if m != "scrfd"]
                                    _old_name = _old_slot2[0] if _old_slot2 else None
                                    if _old_name != _recommended:
                                        threading.Thread(
                                            target=_do_rotation,
                                            args=(_old_name, _recommended),
                                            daemon=True
                                        ).start()

                            # TTS Ansage (60s Cooldown pro Person)
                            if name != "Unbekannt" and name != "Keine DB":"""

if old_announce in code:
    code = code.replace(old_announce, new_announce, 1)
    changes += 1
    print("PATCH 2b: Perception tick added in inference loop")
else:
    print("ERROR: Could not find TTS announce block")
    sys.exit(1)

with open(svc_path, "w") as f:
    f.write(code)
print(f"Service patched: {changes} changes")

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Perception Engine eingebaut:")
print(f"  1. core/perception_engine.py (NEU, ~160 LOC)")
print(f"  2. moloch_service.py: {changes} patches")
print(f"\nSlot 1 (Basis): SCRFD (Face Detection)")
print(f"Slot 2 (rotiert): arcface / yolov8m / pose nach Score")
print(f"Personality: Guardian=arcface*1.3, Shadow=pose*1.3")
print(f"Tension: >0.6 = arcface boost, >0.3 = schnellere Rotation")
