#!/usr/bin/env python3
"""Perception Engine v2: Sauber nach Spec, keine privaten Felder."""
import sys

# =====================================================
# STEP 0: Add get_tension() to PersonalityEngine
# =====================================================
pe_path = "/home/molochzuhause/moloch/core/personality/personality_engine.py"
with open(pe_path, "r") as f:
    pe_code = f.read()

pe_changes = 0

old_update_drift = """    def update_drift_factors(self, factors: Dict[str, float]):
        \"\"\"Update drift factors from real sensor values. Called externally.\"\"\"
        self._drift_factors.update(factors)"""

new_update_drift = """    def get_tension(self) -> float:
        \"\"\"Public API: Aktuelle Tension (0.0-1.0).\"\"\"
        return self._compute_tension()

    def update_drift_factors(self, factors: Dict[str, float]):
        \"\"\"Update drift factors from real sensor values. Called externally.\"\"\"
        self._drift_factors.update(factors)"""

if old_update_drift in pe_code:
    pe_code = pe_code.replace(old_update_drift, new_update_drift, 1)
    pe_changes += 1
    print("STEP 0: get_tension() added to PersonalityEngine")
else:
    print("WARNING: Could not find update_drift_factors (get_tension may already exist)")

if pe_changes > 0:
    with open(pe_path, "w") as f:
        f.write(pe_code)

# =====================================================
# STEP 1: Create core/perception_engine.py (v2, clean)
# =====================================================
perception_module = '''\
#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Engine
================================
Scoring-basierte NPU Slot-Rotation.

Hardware: Hailo-10H NPU (40 TOPS), max 2 Modelle gleichzeitig.
Slot 1 = Basis (SCRFD oder YOLO), stabil, nicht von Engine gewechselt.
Slot 2 = Dynamisch, per Scoring gewaehlt, Hysterese + Intervall.

Reines Beratungsmodul. Kein Hardware-Zugriff. Kein Threading. Kein Random.
"""
import time
from typing import Dict, Optional


class PerceptionEngine:
    """NPU Slot-Management mit Personality-gesteuertem Scoring."""

    SLOT2_CANDIDATES = ["arcface", "yolov8m", "pose"]

    BASE_SCORES = {
        "arcface": 0.5,
        "yolov8m": 0.4,
        "pose": 0.3,
    }

    def __init__(self, personality_engine=None):
        self._personality = personality_engine
        self.slot_1 = "scrfd"
        self.slot_2: Optional[str] = None
        self._scores: Dict[str, float] = {}
        self._last_rotation = 0.0
        self._last_active: Dict[str, float] = {}
        self._min_interval = 10.0
        self._hysteresis = 0.15
        self._forced_slot2: Optional[str] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def tick(self, context: Dict) -> Optional[str]:
        """Pro Inference-Zyklus aufrufen.

        Args:
            context: {
                "face_detected": bool,
                "person_detected": bool,
                "unknown_person": bool,
                "motion_level": float (0.0-1.0),
            }

        Returns:
            Modellname wenn Slot 2 wechseln soll, sonst None.
        """
        # Manueller Override
        if self._forced_slot2:
            if self._forced_slot2 != self.slot_2:
                self.slot_2 = self._forced_slot2
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

        # Erster tick: sofort setzen
        if self.slot_2 is None:
            self.slot_2 = best
            self._last_rotation = time.time()
            return best

        # Schon aktiv
        if best == self.slot_2:
            return None

        # Hysterese
        current_score = candidates.get(self.slot_2, 0.0)
        if best_score - current_score < self._hysteresis:
            return None

        # Mindest-Intervall
        now = time.time()
        if now - self._last_rotation < self._min_interval:
            return None

        # Wechsel
        if self.slot_2:
            self._last_active[self.slot_2] = now
        self._last_rotation = now
        self.slot_2 = best
        return best

    def set_base_model(self, name: str):
        """Slot 1 (Basis) setzen."""
        self.slot_1 = name

    def force_slot2(self, name: Optional[str]):
        """Manueller Override fuer Slot 2. None = zurueck zu Scoring."""
        self._forced_slot2 = name

    def get_state(self) -> Dict:
        """Status fuer GUI/Debug."""
        tension = 0.0
        mode = "standalone"
        if self._personality:
            tension = self._personality.get_tension() if hasattr(self._personality, "get_tension") else 0.0
            mode = self._personality.mode.value if hasattr(self._personality, "mode") else "unknown"

        return {
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

        # Kontext-Boosts
        if ctx.get("face_detected"):
            scores["arcface"] += 0.4
        if ctx.get("person_detected"):
            scores["pose"] += 0.3
        if ctx.get("unknown_person"):
            scores["arcface"] += 0.3
        if ctx.get("motion_level", 0.0) > 0.5:
            scores["yolov8m"] += 0.2

        # Personality-Gewichtung
        if self._personality:
            if self._personality.is_guardian:
                scores["arcface"] *= 1.3
                scores["pose"] *= 0.7
            elif self._personality.is_shadow:
                scores["pose"] *= 1.3
                scores["arcface"] *= 0.9

            # Tension-Modulator
            tension = self._personality.get_tension() if hasattr(self._personality, "get_tension") else 0.0
            if tension > 0.6:
                scores["arcface"] += 0.3
            if tension > 0.3:
                self._min_interval = max(5.0, 10.0 - tension * 8)
            else:
                self._min_interval = 10.0

        # Anti-Starvation
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
# STEP 2: Fix moloch_service.py integration (clean)
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    code = f.read()

svc_changes = 0

# PATCH 2a: Replace verbose perception tick with clean version
old_perception_tick = """                            # Perception Engine: Slot-2-Rotation
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

                            # TTS Ansage (60s Cooldown pro Person)"""

new_perception_tick = """                            # Perception Engine: Slot-2-Empfehlung
                            if self._perception:
                                _perc_ctx = {
                                    "face_detected": len(face_boxes) > 0,
                                    "person_detected": bool(getattr(self, '_last_person_boxes', [])),
                                    "unknown_person": name == "Unbekannt",
                                    "motion_level": 0.0,
                                }
                                _recommended = self._perception.tick(_perc_ctx)
                                if _recommended:
                                    _current = [m for m in self._active_ctx if m != self._perception.slot_1]
                                    _old = _current[0] if _current else None
                                    if _old != _recommended:
                                        logger.info(f"[PERCEPTION] Rotation: {_old} -> {_recommended}")
                                        if _old and _old in self._active_ctx:
                                            self._unconfigure_model(_old)
                                            time.sleep(0.2)
                                        if _recommended not in self._active_ctx:
                                            self._configure_model(_recommended)

                            # TTS Ansage (60s Cooldown pro Person)"""

if old_perception_tick in code:
    code = code.replace(old_perception_tick, new_perception_tick, 1)
    svc_changes += 1
    print("PATCH 2a: Perception tick cleaned up")
else:
    print("WARNING: Old perception tick not found (may already be v2)")

if svc_changes > 0:
    with open(svc_path, "w") as f:
        f.write(code)
    print(f"Service patched: {svc_changes} changes")
else:
    print("Service: no changes needed")

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Perception Engine v2:")
print(f"  0. get_tension() in PersonalityEngine: {pe_changes} patch")
print(f"  1. core/perception_engine.py: NEU (~140 LOC, clean)")
print(f"  2. moloch_service.py: {svc_changes} patch")
print(f"\nKeine privaten Personality-Felder. Kein Logging. Kein Sleep. Kein Random.")
print(f"Nur Entscheidung: tick(context) -> Optional[str]")
