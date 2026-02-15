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

    ALL_MODELS = ["scrfd", "arcface", "yolov8m", "pose", "hand_landmark"]

    BASE_SCORES = {
        "scrfd": 0.6,
        "arcface": 0.5,
        "yolov8m": 0.4,
        "pose": 0.3,
        "hand_landmark": 0.2,
    }

    # ArcFace ohne SCRFD ist nutzlos (braucht Face-Crops)
    DEPENDENCIES = {"arcface": "scrfd", "hand_landmark": "pose"}

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
        self._last_scores = scores

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
        # Bei Hand-Occlusion: Skip (Event-basierter Swap, nicht graduell)
        if not self._hand_occlusion:
            leaving = set(self.slots) - set(new_slots)
            entering = set(new_slots) - set(self.slots)

            if leaving and entering:
                leaving_best = max(scores.get(m, 0) for m in leaving)
                entering_worst = min(scores.get(m, 0) for m in entering)
                if entering_worst - leaving_best < self._hysteresis:
                    return None

        # Cooldown (verkuerzt bei Hand-Occlusion)
        now = time.time()
        cooldown = 3.0 if self._hand_occlusion else self._min_interval
        if now - self._last_rotation < cooldown:
            return None

        # Swap!
        leaving = set(self.slots) - set(new_slots)
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        import logging
        logging.getLogger("PerceptionEngine").info(
            f"[SWAP] {list(leaving)} -> {list(set(new_slots) - set(self.slots) if hasattr(self, '_prev_slots') else new_slots)} "
            f"occlusion={self._hand_occlusion} scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}}")
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
            "hand_timeout": self._HAND_TIMEOUT,
            "hand_streak_min": self._MIN_FACE_STREAK,
            "hand_recency": self._FACE_RECENCY,
        }

    # =========================================================================
    # Top-2 Selection
    # =========================================================================

    def _select_top2(self, scores: Dict[str, float]) -> List[str]:
        """Top 2 Modelle waehlen, Dependencies beachten."""
        ranked = sorted(self.ALL_MODELS, key=lambda m: scores.get(m, 0), reverse=True)
        s1, s2 = ranked[0], ranked[1]

        # Dependencies erzwingen (arcface->scrfd, hand_landmark->pose)
        for _dep, _req in self.DEPENDENCIES.items():
            if _dep in (s1, s2) and _req not in (s1, s2):
                if s1 == _dep:
                    s2 = _req
                else:
                    s1 = _req
                break

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
            scores["yolov8m"] += 0.2
            scores["pose"] += 0.15

        if unknown:
            scores["arcface"] += 0.5

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

        # person_count_jump: YOLO 2+ Personen aber <=1 Gesicht -> Anomalie
        _person_count = ctx.get("person_count", 0)
        _face_count = ctx.get("face_count", 0)
        if _person_count >= 2 and _face_count <= 1:
            scores["hand_landmark"] += 0.5
            scores["pose"] += 0.4

        # Hand Occlusion -> pose + hand_landmark boosten
        if self._hand_occlusion:
            scores["pose"] += 1.2
            scores["hand_landmark"] += 1.5
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
