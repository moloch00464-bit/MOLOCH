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
import json
import os
import logging
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger("PerceptionEngine")
_HISTORY_PATH = os.path.expanduser("~/moloch/data/perception_history.json")
_WEIGHTS_PATH = os.path.expanduser("~/moloch/config/perception_weights.json")
_LEARN_EVERY = 100  # Alle 100 Entscheidungen lernen
_MAX_ADJUST = 0.10  # Max 10% Aenderung pro Lernzyklus


class PerceptionEngine:
    """Dual-Slot NPU Management mit Scoring."""

    ALL_MODELS = ["scrfd", "arcface", "yolov8m", "hand_landmark"]

    BASE_SCORES = {
        "scrfd": 0.6,
        "arcface": 0.5,
        "yolov8m": 0.4,
        "hand_landmark": 0.2,
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

        # Lernfaehigkeit
        self._history: list = []
        self._learned_weights: Dict[str, float] = {}
        self._last_context: Dict = {}
        self._last_chosen: List[str] = []
        self._decision_count = 0
        self._log_skip_counter = 0
        self._LOG_SAMPLE_RATE = 15  # Nur jeden 15. Frame loggen (1/s bei 15fps)
        self._load_weights()

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

        # Log previous decision utility
        if self._last_chosen and self.slots:
            utility = self._check_utility(self._last_chosen, context)
            self._log_decision(self._last_chosen, context, utility)

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

        # Context merken fuer log_result()
        self._last_context = dict(context)

        # Top 2 waehlen (Context fuer Hard Rules)
        new_slots = self._select_top2(scores, context)

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
        entering = set(new_slots) - set(self.slots)
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        self._last_chosen = list(new_slots)
        _logger.info(
            f"[SWAP] -{list(leaving)} +{list(entering)} "
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
            "learned_weights": dict(self._learned_weights),
            "decision_count": self._decision_count,
        }

    # =========================================================================
    # Top-2 Selection
    # =========================================================================

    def _select_top2(self, scores: Dict[str, float], context: Dict = None) -> List[str]:
        """Top 2 Modelle waehlen, Dependencies + Hard Rules beachten."""
        # HARD RULE: Face erkannt -> SCRFD + ArcFace, IMMER.
        # Einzige Ausnahme: Hand-Occlusion (Face gerade verdeckt)
        if context and context.get("face_detected", False) and not self._hand_occlusion:
            return ["scrfd", "arcface"]

        ranked = sorted(self.ALL_MODELS, key=lambda m: scores.get(m, 0), reverse=True)
        s1, s2 = ranked[0], ranked[1]

        # Dependencies erzwingen (arcface->scrfd)
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

        # Gelernte Gewichte anwenden (addiert auf Base)
        for model, adj in self._learned_weights.items():
            if model in scores:
                scores[model] += adj

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

        if unknown:
            scores["arcface"] += 0.5

        if motion > 0.5:
            scores["yolov8m"] += 0.2

        # Face bekannt -> ArcFace weniger dringend
        if face and not unknown:
            scores["arcface"] -= 0.15

        # Nichts erkannt -> Waechter-Modus (scannen)
        if not face and not person:
            scores["yolov8m"] += 0.3
            scores["scrfd"] += 0.2

        # Person da aber kein Gesicht -> Hand pruefen (Gesten-Modus)
        if person and not face:
            scores["hand_landmark"] += 0.4

        # person_count_jump: YOLO 2+ Personen aber <=1 Gesicht -> Anomalie
        _person_count = ctx.get("person_count", 0)
        _face_count = ctx.get("face_count", 0)
        if _person_count >= 2 and _face_count <= 1:
            scores["hand_landmark"] += 0.5

        # Hand Occlusion -> hand_landmark boosten
        if self._hand_occlusion:
            scores["hand_landmark"] += 1.5
            scores["scrfd"] += 0.2

        # --- Personality-Gewichtung ---

        if self._personality:
            if self._personality.is_guardian:
                scores["scrfd"] *= 1.1
                scores["arcface"] *= 1.3
            elif self._personality.is_shadow:
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

    # =========================================================================
    # Lernfaehigkeit
    # =========================================================================

    def log_result(self, results: Dict):
        """Ergebnis der letzten Inference loggen (1x pro Sekunde).

        Args:
            results: {
                "face_identified": bool,  # ArcFace hat Name geliefert
                "person_count": int,      # YOLO Person-Count
                "hand_detected": bool,    # Hand Landmark erkannt
                "face_detected": bool,    # SCRFD hat Face gefunden
            }
        """
        # Throttle: nur jeden N-ten Frame loggen
        self._log_skip_counter += 1
        if self._log_skip_counter < self._LOG_SAMPLE_RATE:
            return
        self._log_skip_counter = 0

        if not self._last_chosen:
            self._last_chosen = list(self.slots)

        entry = {
            "ts": round(time.time(), 1),
            "models": list(self._last_chosen),
            "ctx": {
                "face": self._last_context.get("face_detected", False),
                "person": self._last_context.get("person_detected", False),
                "unknown": self._last_context.get("unknown_person", False),
                "occlusion": self._hand_occlusion,
            },
            "results": results,
        }

        self._history.append(entry)
        self._decision_count += 1

        # Alle _LEARN_EVERY Entscheidungen: lernen + speichern
        if len(self._history) >= _LEARN_EVERY:
            self._learn_from_history()

    def _learn_from_history(self):
        """Aus History lernen: Gewichte anpassen (max 10% pro Zyklus)."""
        if not self._history:
            return

        # Pro Modell zaehlen: aktiv, nuetzlich
        model_active = {m: 0 for m in self.ALL_MODELS}
        model_useful = {m: 0 for m in self.ALL_MODELS}

        for entry in self._history:
            models = entry.get("models", [])
            results = entry.get("results", {})
            ctx = entry.get("ctx", {})

            for m in models:
                if m in model_active:
                    model_active[m] += 1

            # Nuetzlichkeit bewerten
            if "scrfd" in models and results.get("face_detected", False):
                model_useful["scrfd"] += 1
            if "arcface" in models and results.get("face_identified", False):
                model_useful["arcface"] += 1
            if "yolov8m" in models and results.get("person_count", 0) > 0:
                model_useful["yolov8m"] += 1
            if "hand_landmark" in models and results.get("hand_detected", False):
                model_useful["hand_landmark"] += 1

        # Gewichtung anpassen
        adjustments = {}
        for model in self.ALL_MODELS:
            active = model_active[model]
            if active < 5:
                continue  # Zu wenig Daten

            useful = model_useful[model]
            ratio = useful / active  # 0.0 - 1.0

            # Ziel: 50% Nuetzlichkeit = neutral. Darueber = Score rauf, darunter = runter.
            delta = (ratio - 0.5) * _MAX_ADJUST  # Max +/- 0.05 pro Zyklus
            adjustments[model] = round(delta, 4)

        # Auf bestehende Gewichte addieren (kumulativ, max +/- 0.3 gesamt)
        for model, delta in adjustments.items():
            current = self._learned_weights.get(model, 0.0)
            new_val = max(-0.3, min(0.3, current + delta))
            self._learned_weights[model] = round(new_val, 4)

        _logger.info(
            f"[LEARN] {len(self._history)} Entscheidungen analysiert. "
            f"Active: {model_active}, Useful: {model_useful}, "
            f"Adjustments: {adjustments}, Weights: {self._learned_weights}")

        # Speichern
        self._save_weights()
        self._save_history()

        # History zuruecksetzen
        self._history = []

    def _load_weights(self):
        """Gelernte Gewichte aus perception_weights.json laden."""
        if os.path.exists(_WEIGHTS_PATH):
            try:
                with open(_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._learned_weights = data.get("weights", {})
                self._decision_count = data.get("total_decisions", 0)
                _logger.info(f"[LEARN] Weights geladen: {self._learned_weights} "
                             f"(nach {self._decision_count} Entscheidungen)")
            except Exception as e:
                _logger.warning(f"[LEARN] Weights laden fehlgeschlagen: {e}")

    def _save_weights(self):
        """Gelernte Gewichte in perception_weights.json speichern."""
        try:
            data = {
                "version": 1,
                "total_decisions": self._decision_count,
                "weights": self._learned_weights,
                "base_scores": dict(self.BASE_SCORES),
                "effective_scores": {
                    m: round(self.BASE_SCORES.get(m, 0) + self._learned_weights.get(m, 0), 4)
                    for m in self.ALL_MODELS
                },
            }
            os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
            _tmp = _WEIGHTS_PATH + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(_tmp, _WEIGHTS_PATH)
        except Exception as e:
            _logger.warning(f"[LEARN] Weights speichern fehlgeschlagen: {e}")

    def _save_history(self):
        """History in perception_history.json speichern (letzte 200 Eintraege)."""
        try:
            existing = []
            if os.path.exists(_HISTORY_PATH):
                try:
                    with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []

            # Neue Eintraege anhaengen, max 200 behalten
            combined = existing + self._history
            combined = combined[-200:]

            os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
            _tmp = _HISTORY_PATH + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=1, ensure_ascii=False)
            os.replace(_tmp, _HISTORY_PATH)
        except Exception as e:
            _logger.warning(f"[LEARN] History speichern fehlgeschlagen: {e}")

    def _check_utility(self, chosen_models: List[str], context: Dict) -> Dict[str, bool]:
        """Check if chosen models were useful based on what was detected."""
        utility = {}
        face_detected = context.get("face_detected", False)
        person_detected = context.get("person_detected", False)
        unknown_person = context.get("unknown_person", False)

        for model in chosen_models:
            if model == "scrfd":
                utility[model] = face_detected
            elif model == "arcface":
                # Nuetzlich wenn Face UND bekannte Person
                utility[model] = face_detected and not unknown_person
            elif model == "yolov8m":
                utility[model] = person_detected
            elif model == "hand_landmark":
                # Nuetzlich wenn Person detected
                utility[model] = person_detected
            else:
                utility[model] = False

        return utility

    def _log_decision(self, models: List[str], context: Dict, utility: Dict[str, bool]):
        """Log decision to history."""
        self._log_skip_counter += 1
        if self._log_skip_counter < self._LOG_SAMPLE_RATE:
            return  # Only log every Nth frame
        self._log_skip_counter = 0

        entry = {
            "timestamp": time.time(),
            "models": models,
            "detected": {
                "face": context.get("face_detected", False),
                "person": context.get("person_detected", False),
                "unknown": context.get("unknown_person", False),
            },
            "utility": utility,
            "useful_count": sum(1 for u in utility.values() if u),
        }
        self._history.append(entry)
        self._decision_count += 1

        # Save every 20 entries
        if len(self._history) >= 20:
            self._save_history()
            self._history = []

        # Auto-adjust after 100 decisions
        if self._decision_count % 100 == 0:
            self._auto_adjust_weights()

    def _auto_adjust_weights(self):
        """Auto-adjust weights based on utility stats from last 100 decisions."""
        try:
            # Load full history
            history = []
            if os.path.exists(_HISTORY_PATH):
                with open(_HISTORY_PATH, "r") as f:
                    history = json.load(f)

            # Take last 100 entries
            recent = history[-100:]
            if len(recent) < 50:
                return  # Not enough data

            # Calculate utility rate per model
            stats = {m: {"used": 0, "useful": 0} for m in self.ALL_MODELS}
            for entry in recent:
                util = entry.get("utility", {})
                for model, was_useful in util.items():
                    if model in stats:
                        stats[model]["used"] += 1
                        if was_useful:
                            stats[model]["useful"] += 1

            # Adjust weights
            MINIMUMS = {"scrfd": 0.5, "arcface": 0.4}
            DEFAULT_MIN = 0.2
            adjustments = {}

            for model, data in stats.items():
                if data["used"] < 10:
                    continue  # Not enough samples

                rate = data["useful"] / data["used"]
                base = self.BASE_SCORES.get(model, 0.3)
                current_adj = self._learned_weights.get(model, 0.0)
                new_adj = current_adj

                if rate > 0.7:  # >70% useful
                    new_adj = min(current_adj + 0.05, 0.10)  # Max +10%
                elif rate < 0.4:  # <40% useful
                    new_adj = max(current_adj - 0.05, -0.10)  # Max -10%

                # Apply minimum
                min_score = MINIMUMS.get(model, DEFAULT_MIN)
                effective = base + new_adj
                if effective < min_score:
                    new_adj = min_score - base

                if abs(new_adj - current_adj) > 0.001:
                    adjustments[model] = new_adj

            # Apply adjustments
            if adjustments:
                self._learned_weights.update(adjustments)
                self._save_weights()
                _logger.info(f"[LEARN] Auto-adjust: {adjustments} (nach {self._decision_count} Entscheidungen)")

        except Exception as e:
            _logger.warning(f"[LEARN] Auto-adjust failed: {e}")
