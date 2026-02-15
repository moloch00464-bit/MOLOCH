#!/usr/bin/env python3
"""Patch: Intelligente Hand-Erkennung in Perception Engine.

Erweitert PerceptionEngine um:
- Face-State Tracking (History, Streak, BBox)
- Hand-Occlusion Erkennung (4 Bedingungen gleichzeitig)
- Pose-Modell forcen bei Occlusion, Timeout 5s

Verschiebt Perception-Tick im Service:
- RAUS aus ArcFace-Loop (lief nur bei Face+ArcFace!)
- REIN nach allen Detektionen (laeuft immer, pro Zyklus einmal)
- Neue Context-Felder: face_bbox, camera_moving
"""
import sys

# =====================================================
# STEP 1: Neue perception_engine.py mit Hand-Occlusion
# =====================================================
perception_module = '''\
#!/usr/bin/env python3
"""
M.O.L.O.C.H. Perception Engine
================================
Scoring-basierte NPU Slot-Rotation mit Hand-Occlusion Erkennung.

Hardware: Hailo-10H NPU (40 TOPS), max 2 Modelle gleichzeitig.
Slot 1 = Basis (SCRFD oder YOLO), stabil, nicht von Engine gewechselt.
Slot 2 = Dynamisch, per Scoring gewaehlt, Hysterese + Intervall.

Hand-Occlusion: Erkennt wenn Hand Gesicht verdeckt.
Trigger NUR wenn ALLE Bedingungen gleichzeitig:
  1. Person noch da (YOLO)
  2. Gesicht war gerade da (<2s)
  3. Gesicht ploetzlich weg (stabil -> weg, nicht langsam)
  4. Kamera bewegt sich NICHT

Reines Beratungsmodul. Kein Hardware-Zugriff. Kein Threading. Kein Random.
"""
import time
from typing import Dict, List, Optional, Tuple


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

        # Face State Tracking (fuer Hand-Occlusion)
        self._last_face_bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_face_time = 0.0
        self._face_streak = 0  # Consecutive ticks with face

        # Hand Occlusion State
        self._hand_occlusion = False
        self._hand_occlusion_start = 0.0
        self._HAND_TIMEOUT = 5.0       # Timeout: Face-Modell zurueck
        self._FACE_RECENCY = 2.0       # Face muss <2s her sein
        self._MIN_FACE_STREAK = 3      # Min Frames stabil bevor "ploetzlich weg"

    # =========================================================================
    # Public API
    # =========================================================================

    def tick(self, context: Dict) -> Optional[str]:
        """Pro Inference-Zyklus aufrufen.

        Args:
            context: {
                "face_detected": bool,
                "face_bbox": Optional[Tuple[float,float,float,float]],
                "person_detected": bool,
                "unknown_person": bool,
                "motion_level": float (0.0-1.0),
                "camera_moving": bool,
            }

        Returns:
            Modellname wenn Slot 2 wechseln soll, sonst None.
        """
        # Face-Tracking aktualisieren (VOR Occlusion-Check)
        self._update_face_tracking(context)

        # Hand-Occlusion pruefen (hat Vorrang vor allem)
        occlusion = self._check_hand_occlusion(context)
        if occlusion:
            if self.slot_2 != "pose":
                old = self.slot_2
                self.slot_2 = "pose"
                self._last_rotation = time.time()
                if old:
                    self._last_active[old] = time.time()
                return "pose"
            return None

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
            "hand_occlusion": self._hand_occlusion,
            "face_streak": self._face_streak,
        }

    # =========================================================================
    # Face Tracking
    # =========================================================================

    def _update_face_tracking(self, ctx: Dict):
        """Face-Streak und letzte BBox aktualisieren."""
        face_detected = ctx.get("face_detected", False)
        face_bbox = ctx.get("face_bbox")

        if face_detected:
            self._face_streak += 1
            self._last_face_time = time.time()
            if face_bbox:
                self._last_face_bbox = face_bbox
        # Streak reset passiert in _check_hand_occlusion NACH Auswertung

    # =========================================================================
    # Hand Occlusion Detection
    # =========================================================================

    def _check_hand_occlusion(self, ctx: Dict) -> bool:
        """Pruefen ob Hand wahrscheinlich Gesicht verdeckt.

        Bedingungen (ALLE muessen erfuellt sein):
        1. Person noch sichtbar
        2. Gesicht war gerade da (<2s)
        3. Gesicht ploetzlich weg (Streak war >= MIN_FACE_STREAK)
        4. Kamera bewegt sich NICHT

        Returns:
            True wenn Occlusion aktiv (Pose-Modell forcen).
        """
        now = time.time()
        face_detected = ctx.get("face_detected", False)
        person_detected = ctx.get("person_detected", False)
        camera_moving = ctx.get("camera_moving", False)

        # --- Occlusion AUFHEBEN ---

        # Face ist zurueck -> Occlusion vorbei
        if self._hand_occlusion and face_detected:
            self._hand_occlusion = False
            return False

        # Timeout 5s -> Occlusion aufgeben
        if self._hand_occlusion:
            if now - self._hand_occlusion_start > self._HAND_TIMEOUT:
                self._hand_occlusion = False
                self._face_streak = 0
                return False
            return True  # Noch in Occlusion, Pose bleibt

        # --- Face da -> keine neue Occlusion ---
        if face_detected:
            return False

        # --- NEUE Occlusion pruefen (Face ist JETZT weg) ---

        # Streak merken VOR Reset
        streak_before = self._face_streak
        self._face_streak = 0  # Reset: Face ist weg

        # 1. Person muss noch sichtbar sein
        if not person_detected:
            return False

        # 2. Kamera darf sich nicht bewegen
        if camera_moving:
            return False

        # 3. Gesicht war gerade eben noch da (<2s)
        if now - self._last_face_time > self._FACE_RECENCY:
            return False

        # 4. Gesicht war stabil, dann ploetzlich weg
        if streak_before < self._MIN_FACE_STREAK:
            return False

        # ALLE Bedingungen erfuellt: Hand-Occlusion!
        self._hand_occlusion = True
        self._hand_occlusion_start = now
        return True

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
# STEP 2: Patch moloch_service.py
# =====================================================
svc_path = "/home/molochzuhause/moloch/core/moloch_service.py"
with open(svc_path, "r") as f:
    code = f.read()

svc_changes = 0

# PATCH 2a: Perception tick aus ArcFace-Loop ENTFERNEN
old_perception_in_arcface = """                            # Perception Engine: Slot-2-Empfehlung
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

new_no_perception_in_arcface = """                            # TTS Ansage (60s Cooldown pro Person)"""

if old_perception_in_arcface in code:
    code = code.replace(old_perception_in_arcface, new_no_perception_in_arcface, 1)
    svc_changes += 1
    print("PATCH 2a: Perception tick aus ArcFace-Loop entfernt")
else:
    print("ERROR: Perception tick in ArcFace-Loop nicht gefunden")
    sys.exit(1)

# PATCH 2b: Perception tick NACH Pose Detection, VOR Total FPS einfuegen
old_total_fps = """            # Total FPS
            dt_total = time.perf_counter() - t_total"""

new_perception_then_fps = """            # ===== Perception Engine: Slot-2-Empfehlung (nach allen Detektionen) =====
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
                            self._configure_model(_recommended)

            # Total FPS
            dt_total = time.perf_counter() - t_total"""

if old_total_fps in code:
    code = code.replace(old_total_fps, new_perception_then_fps, 1)
    svc_changes += 1
    print("PATCH 2b: Perception tick nach allen Detektionen, vor Total FPS")
else:
    print("ERROR: '# Total FPS' Block nicht gefunden")
    sys.exit(1)

if svc_changes > 0:
    with open(svc_path, "w") as f:
        f.write(code)
    print(f"Service patched: {svc_changes} changes")
else:
    print("Service: no changes needed")

# =====================================================
# Summary
# =====================================================
print(f"\nDone! Hand-Occlusion Detection:")
print(f"  1. core/perception_engine.py: NEU (~220 LOC)")
print(f"     - Face State Tracking (Streak, BBox, Timestamp)")
print(f"     - _check_hand_occlusion(): 4 Bedingungen gleichzeitig")
print(f"     - Pose forcen bei Occlusion, 5s Timeout")
print(f"  2. moloch_service.py: {svc_changes} patches")
print(f"     - Perception tick NACH allen Detektionen (nicht mehr in ArcFace-Loop)")
print(f"     - face_bbox + camera_moving im Context")
print(f"\nTrigger: Person da + Face war da (<2s) + Face ploetzlich weg + Kamera steht")
print(f"Release: Face zurueck ODER 5s Timeout")
print(f"NICHT bei: Kopf drehen (Streak zu kurz), Person geht, Kamera bewegt sich")
