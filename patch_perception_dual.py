#!/usr/bin/env python3
"""Patch: Perception Engine - Beide Slots dynamisch.

NEU: Top 2 aus Scoring gewinnen. Kein fester Slot 1 mehr.
tick() -> Optional[List[str]] (max 2 Modelle, oder None wenn kein Swap)

Beispiele:
- Face sichtbar      -> scrfd + arcface
- Face bekannt       -> scrfd + pose
- Hand Occlusion     -> scrfd + pose
- Person weit weg    -> yolov8m + pose
- Nichts los         -> yolov8m + scrfd
"""
import sys

# =====================================================
# STEP 1: Neue perception_engine.py - Dual Dynamic Slots
# =====================================================
perception_module = '''\
#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Engine
================================
Dual-Slot NPU Management. Beide Slots dynamisch per Scoring.

Hardware: Hailo-10H NPU (40 TOPS), max 2 Modelle gleichzeitig.
Top 2 Scores gewinnen. ArcFace braucht SCRFD (Dependency).

tick(context) -> Optional[List[str]]: Neue Modell-Kombination oder None.
Nur Entscheidung. Kein Hardware-Zugriff. Kein Threading. Kein Random.
"""
import time
from typing import Dict, List, Optional, Tuple


class PerceptionEngine:
    """Dual-Slot NPU Management mit Scoring."""

    ALL_MODELS = ["scrfd", "arcface", "yolov8m", "pose"]

    BASE_SCORES = {
        "scrfd": 0.6,
        "arcface": 0.5,
        "yolov8m": 0.4,
        "pose": 0.3,
    }

    # ArcFace ohne SCRFD ist nutzlos (braucht Face-Crops)
    DEPENDENCIES = {"arcface": "scrfd"}

    def __init__(self, personality_engine=None):
        self._personality = personality_engine
        self.slots: List[str] = []  # Aktuelle 2 Modelle (leer = noch nicht gesetzt)
        self._scores: Dict[str, float] = {}
        self._last_rotation = 0.0
        self._last_active: Dict[str, float] = {}
        self._min_interval = 10.0
        self._hysteresis = 0.15
        self._forced: Optional[List[str]] = None

        # Face State Tracking (Hand-Occlusion)
        self._last_face_bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_face_time = 0.0
        self._face_streak = 0

        # Hand Occlusion
        self._hand_occlusion = False
        self._hand_occlusion_start = 0.0
        self._HAND_TIMEOUT = 5.0
        self._FACE_RECENCY = 2.0
        self._MIN_FACE_STREAK = 3

    # =========================================================================
    # Public API
    # =========================================================================

    def tick(self, context: Dict) -> Optional[List[str]]:
        """Pro Inference-Zyklus aufrufen.

        Args:
            context: {
                "face_detected": bool,
                "face_bbox": Optional[Tuple],
                "person_detected": bool,
                "unknown_person": bool,
                "motion_level": float (0.0-1.0),
                "camera_moving": bool,
            }

        Returns:
            [model1, model2] wenn Swap noetig, None wenn kein Wechsel.
        """
        self._update_face_tracking(context)
        self._update_hand_occlusion(context)

        # Manueller Override
        if self._forced:
            if set(self._forced) != set(self.slots):
                self.slots = list(self._forced)
                return list(self.slots)
            return None

        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores

        # Top 2 waehlen
        new_slots = self._select_top2(scores)

        # Erster tick: sofort setzen
        if not self.slots:
            self.slots = new_slots
            self._last_rotation = time.time()
            return list(new_slots)

        # Gleiche Modelle? Kein Swap.
        if set(new_slots) == set(self.slots):
            return None

        # Hysterese: Eintretende muessen deutlich besser sein als Gehende
        leaving = set(self.slots) - set(new_slots)
        entering = set(new_slots) - set(self.slots)

        if leaving and entering:
            leaving_best = max(scores.get(m, 0) for m in leaving)
            entering_worst = min(scores.get(m, 0) for m in entering)
            if entering_worst - leaving_best < self._hysteresis:
                return None

        # Cooldown
        now = time.time()
        if now - self._last_rotation < self._min_interval:
            return None

        # Swap!
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        return list(new_slots)

    def force_models(self, models: Optional[List[str]]):
        """Manueller Override. None = zurueck zu Scoring."""
        self._forced = models

    def get_state(self) -> Dict:
        """Status fuer GUI/Debug."""
        tension = 0.0
        mode = "standalone"
        if self._personality:
            tension = self._personality.get_tension() if hasattr(self._personality, "get_tension") else 0.0
            mode = self._personality.mode.value if hasattr(self._personality, "mode") else "unknown"

        return {
            "slots": list(self.slots),
            "forced": self._forced,
            "scores": {k: round(v, 3) for k, v in self._scores.items()},
            "tension": round(tension, 3),
            "personality_mode": mode,
            "min_interval": round(self._min_interval, 1),
            "hand_occlusion": self._hand_occlusion,
            "face_streak": self._face_streak,
        }

    # =========================================================================
    # Top-2 Selection
    # =========================================================================

    def _select_top2(self, scores: Dict[str, float]) -> List[str]:
        """Top 2 Modelle waehlen, Dependencies beachten."""
        ranked = sorted(self.ALL_MODELS, key=lambda m: scores.get(m, 0), reverse=True)
        s1, s2 = ranked[0], ranked[1]

        # Dependency: arcface braucht scrfd
        if "arcface" in (s1, s2) and "scrfd" not in (s1, s2):
            if s1 == "arcface":
                s2 = "scrfd"
            else:
                s1 = "scrfd"

        return [s1, s2]

    # =========================================================================
    # Face Tracking
    # =========================================================================

    def _update_face_tracking(self, ctx: Dict):
        """Face-Streak und BBox aktualisieren."""
        if ctx.get("face_detected", False):
            self._face_streak += 1
            self._last_face_time = time.time()
            bbox = ctx.get("face_bbox")
            if bbox:
                self._last_face_bbox = bbox

    # =========================================================================
    # Hand Occlusion
    # =========================================================================

    def _update_hand_occlusion(self, ctx: Dict):
        """Hand-Occlusion State Machine aktualisieren."""
        now = time.time()
        face_detected = ctx.get("face_detected", False)
        person_detected = ctx.get("person_detected", False)
        camera_moving = ctx.get("camera_moving", False)

        # Face zurueck -> Occlusion vorbei
        if self._hand_occlusion and face_detected:
            self._hand_occlusion = False
            return

        # Timeout -> Occlusion aufgeben
        if self._hand_occlusion:
            if now - self._hand_occlusion_start > self._HAND_TIMEOUT:
                self._hand_occlusion = False
                self._face_streak = 0
            return

        # Face da -> nichts zu tun
        if face_detected:
            return

        # Face JETZT weg - Occlusion pruefen
        streak_before = self._face_streak
        self._face_streak = 0

        # Alle Bedingungen gleichzeitig
        if camera_moving:
            return
        if not person_detected and (now - self._last_face_time > 1.0):
            return
        if now - self._last_face_time > self._FACE_RECENCY:
            return
        if streak_before < self._MIN_FACE_STREAK:
            return

        # Hand-Occlusion!
        self._hand_occlusion = True
        self._hand_occlusion_start = now

    # =========================================================================
    # Scoring
    # =========================================================================

    def _compute_scores(self, ctx: Dict) -> Dict[str, float]:
        """Scores fuer alle Modelle berechnen."""
        scores = dict(self.BASE_SCORES)

        face = ctx.get("face_detected", False)
        person = ctx.get("person_detected", False)
        unknown = ctx.get("unknown_person", False)
        motion = ctx.get("motion_level", 0.0)

        # --- Kontext-Boosts ---

        if face:
            scores["scrfd"] += 0.3
            scores["arcface"] += 0.4

        if person:
            scores["pose"] += 0.3
            scores["yolov8m"] += 0.1

        if unknown:
            scores["arcface"] += 0.3

        if motion > 0.5:
            scores["yolov8m"] += 0.2

        # Face bekannt -> ArcFace weniger dringend, Pose interessanter
        if face and not unknown:
            scores["arcface"] -= 0.15
            scores["pose"] += 0.15

        # Nichts erkannt -> Waechter-Modus (scannen)
        if not face and not person:
            scores["yolov8m"] += 0.3
            scores["scrfd"] += 0.2

        # Hand Occlusion -> Pose massiv boosten
        if self._hand_occlusion:
            scores["pose"] += 0.8
            scores["scrfd"] += 0.2

        # --- Personality-Gewichtung ---

        if self._personality:
            if self._personality.is_guardian:
                scores["scrfd"] *= 1.1
                scores["arcface"] *= 1.3
                scores["pose"] *= 0.7
            elif self._personality.is_shadow:
                scores["pose"] *= 1.3
                scores["arcface"] *= 0.9

            tension = self._personality.get_tension() if hasattr(self._personality, "get_tension") else 0.0
            if tension > 0.6:
                scores["arcface"] += 0.3
                scores["scrfd"] += 0.2
            if tension > 0.3:
                self._min_interval = max(5.0, 10.0 - tension * 8)
            else:
                self._min_interval = 10.0

        # --- Anti-Starvation ---

        now = time.time()
        for model in self.ALL_MODELS:
            last = self._last_active.get(model, now)
            idle_mins = (now - last) / 60.0
            scores[model] += min(idle_mins * 0.1, 0.3)

        return scores
'''

perception_path = "/home/molochzuhause/moloch/core/perception_engine.py"
with open(perception_path, "w") as f:
    f.write(perception_module)
print(f"STEP 1: {perception_path} geschrieben")

# =====================================================
# STEP 2: Service-Handler fuer Dual-Slot aktualisieren
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    code = f.read()

svc_changes = 0

old_handler = """            # ===== Perception Engine: Slot-2-Empfehlung (nach allen Detektionen) =====
            if self._perception:
                _perc_face_bbox = None
                if face_boxes:
                    _fb = face_boxes[0][0]  # (box, score, landmarks)[0] = box
                    _perc_face_bbox = (float(_fb[0]), float(_fb[1]), float(_fb[2]), float(_fb[3]))
                _perc_camera_moving = False
                if self._tracker and hasattr(self._tracker, '_camera') and self._tracker._camera:
                    _cam_pos = getattr(self._tracker._camera, 'current_position', None)
                    if _cam_pos:
                        _perc_camera_moving = getattr(_cam_pos, 'moving', False)
                _perc_person = False
                if self.yolo_active and 'persons' in dir() and persons:
                    _perc_person = True
                elif getattr(self, '_last_person_boxes', []):
                    _perc_person = True
                _perc_ctx = {
                    "face_detected": face_detected,
                    "face_bbox": _perc_face_bbox,
                    "person_detected": _perc_person,
                    "unknown_person": face_detected and 'name' in dir() and name == "Unbekannt",
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                }
                _recommended = self._perception.tick(_perc_ctx)
                if _recommended:
                    _current = [m for m in self._active_ctx if m != self._perception.slot_1]
                    _old = _current[0] if _current else None
                    if _old != _recommended:
                        logger.info(f"[PERCEPTION] Rotation: {_old} -> {_recommended} (occlusion={self._perception._hand_occlusion})")
                        if _old and _old in self._active_ctx:
                            self._unconfigure_model(_old)
                            time.sleep(0.2)
                        if _recommended not in self._active_ctx:
                            self._configure_model(_recommended)"""

new_handler = """            # ===== Perception Engine: Dual-Slot Empfehlung (nach allen Detektionen) =====
            if self._perception:
                _perc_face_bbox = None
                if face_boxes:
                    _fb = face_boxes[0][0]
                    _perc_face_bbox = (float(_fb[0]), float(_fb[1]), float(_fb[2]), float(_fb[3]))
                _perc_camera_moving = False
                if self._tracker and hasattr(self._tracker, '_camera') and self._tracker._camera:
                    _cam_pos = getattr(self._tracker._camera, 'current_position', None)
                    if _cam_pos:
                        _perc_camera_moving = getattr(_cam_pos, 'moving', False)
                _perc_person = False
                if self.yolo_active and 'persons' in dir() and persons:
                    _perc_person = True
                elif getattr(self, '_last_person_boxes', []):
                    _perc_person = True
                _perc_ctx = {
                    "face_detected": face_detected,
                    "face_bbox": _perc_face_bbox,
                    "person_detected": _perc_person,
                    "unknown_person": face_detected and 'name' in dir() and name == "Unbekannt",
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                }
                _new_slots = self._perception.tick(_perc_ctx)
                if _new_slots:
                    _want = set(_new_slots)
                    _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")
                        for _m in _to_remove:
                            self._unconfigure_model(_m)
                            time.sleep(0.2)
                        for _m in _to_add:
                            if _m not in self._active_ctx:
                                self._configure_model(_m)"""

if old_handler in code:
    code = code.replace(old_handler, new_handler, 1)
    svc_changes += 1
    print("STEP 2: Service-Handler auf Dual-Slot umgestellt")
else:
    print("ERROR: Alter Perception-Handler nicht gefunden")
    sys.exit(1)

if svc_changes > 0:
    with open(svc_path, "w") as f:
        f.write(code)

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Perception Engine Dual-Slot:")
print(f"  1. perception_engine.py: NEU (~250 LOC)")
print(f"     - ALL_MODELS = [scrfd, arcface, yolov8m, pose]")
print(f"     - Top 2 aus Scoring gewinnen")
print(f"     - Dependency: arcface braucht scrfd")
print(f"     - tick() -> Optional[List[str]]")
print(f"  2. moloch_service.py: Handler auf set-diff Swap")
print(f"\nScoring-Beispiele:")
print(f"  Face sichtbar      -> scrfd(1.1) + arcface(1.3) = scrfd+arcface")
print(f"  Face bekannt       -> scrfd(1.1) + pose(0.75)   = scrfd+pose")
print(f"  Hand Occlusion     -> scrfd(1.0) + pose(1.4)    = scrfd+pose")
print(f"  Nichts los         -> yolov8m(1.0) + scrfd(1.0) = yolov8m+scrfd")
print(f"  Person weit weg    -> pose(0.6) + yolov8m(0.5)  = yolov8m+pose")
