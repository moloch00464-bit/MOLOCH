#!/usr/bin/env python3
"""Perception Engine: ArcFace Zwang + Lernfaehigkeit.

1. SOFORT-FIX: face_detected -> scrfd+arcface IMMER (Hard Rule)
2. LERNFAEHIGKEIT: History logging + adaptive Gewichtung

Aendert:
- core/perception_engine.py (Hauptlogik)
- core/moloch_service.py (log_result Aufruf)
"""
import sys
import os
import json

# ============================================================
# TEIL 1: perception_engine.py
# ============================================================
pe_path = '/home/molochzuhause/moloch/core/perception_engine.py'
with open(pe_path) as f:
    code = f.read()

fixes = 0

# --- FIX 1A: Imports erweitern (json, os, logging) ---
old_imports = """import time
from typing import Dict, List, Optional, Tuple"""

new_imports = """import time
import json
import os
import logging
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger("PerceptionEngine")
_HISTORY_PATH = os.path.expanduser("~/moloch/data/perception_history.json")
_WEIGHTS_PATH = os.path.expanduser("~/moloch/config/perception_weights.json")
_LEARN_EVERY = 100  # Alle 100 Entscheidungen lernen
_MAX_ADJUST = 0.10  # Max 10% Aenderung pro Lernzyklus"""

if old_imports in code:
    code = code.replace(old_imports, new_imports)
    print('FIX 1A: Imports + Konstanten - OK')
    fixes += 1
else:
    print('FIX 1A: ANCHOR NOT FOUND!')

# --- FIX 1B: __init__ erweitern (History + Weights) ---
old_init_end = """        # Hand Occlusion
        self._hand_occlusion = False
        self._hand_occlusion_start = 0.0
        self._HAND_TIMEOUT = 5.0
        self._FACE_RECENCY = 2.0
        self._MIN_FACE_STREAK = 3"""

new_init_end = """        # Hand Occlusion
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
        self._load_weights()"""

if old_init_end in code:
    code = code.replace(old_init_end, new_init_end)
    print('FIX 1B: __init__ History + Weights - OK')
    fixes += 1
else:
    print('FIX 1B: ANCHOR NOT FOUND!')

# --- FIX 1C: tick() - Context speichern fuer spaeteres log_result ---
old_tick_scores = """        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores
        self._last_scores = scores

        # Top 2 waehlen
        new_slots = self._select_top2(scores)"""

new_tick_scores = """        # Scores berechnen
        scores = self._compute_scores(context)
        self._scores = scores
        self._last_scores = scores

        # Context merken fuer log_result()
        self._last_context = dict(context)

        # Top 2 waehlen (Context fuer Hard Rules)
        new_slots = self._select_top2(scores, context)"""

if old_tick_scores in code:
    code = code.replace(old_tick_scores, new_tick_scores)
    print('FIX 1C: tick() Context speichern - OK')
    fixes += 1
else:
    print('FIX 1C: ANCHOR NOT FOUND!')

# --- FIX 1D: tick() - Chosen models merken nach Swap ---
old_tick_swap_log = """        # Swap!
        leaving = set(self.slots) - set(new_slots)
        for m in leaving:
            self._last_active[m] = now
        self._last_rotation = now
        self.slots = new_slots
        import logging
        logging.getLogger("PerceptionEngine").info(
            f"[SWAP] {list(leaving)} -> {list(set(new_slots) - set(self.slots) if hasattr(self, '_prev_slots') else new_slots)} "
            f"occlusion={self._hand_occlusion} scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}}")
        return list(new_slots)"""

new_tick_swap_log = """        # Swap!
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
        return list(new_slots)"""

if old_tick_swap_log in code:
    code = code.replace(old_tick_swap_log, new_tick_swap_log)
    print('FIX 1D: tick() Swap Logging - OK')
    fixes += 1
else:
    print('FIX 1D: ANCHOR NOT FOUND!')

# --- FIX 1E: _select_top2 - Hard Rule face_detected -> scrfd+arcface ---
old_select = """    def _select_top2(self, scores: Dict[str, float]) -> List[str]:
        \"\"\"Top 2 Modelle waehlen, Dependencies beachten.\"\"\"
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

        return [s1, s2]"""

new_select = """    def _select_top2(self, scores: Dict[str, float], context: Dict = None) -> List[str]:
        \"\"\"Top 2 Modelle waehlen, Dependencies + Hard Rules beachten.\"\"\"
        # HARD RULE: Face erkannt -> SCRFD + ArcFace, IMMER.
        # Einzige Ausnahme: Hand-Occlusion (Face gerade verdeckt)
        if context and context.get("face_detected", False) and not self._hand_occlusion:
            return ["scrfd", "arcface"]

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

        return [s1, s2]"""

if old_select in code:
    code = code.replace(old_select, new_select)
    print('FIX 1E: _select_top2 Hard Rule face->arcface - OK')
    fixes += 1
else:
    print('FIX 1E: ANCHOR NOT FOUND!')

# --- FIX 1F: _compute_scores - Learned Weights anwenden ---
old_scores_start = """    def _compute_scores(self, ctx: Dict) -> Dict[str, float]:
        \"\"\"Scores fuer alle Modelle berechnen.\"\"\"
        scores = dict(self.BASE_SCORES)"""

new_scores_start = """    def _compute_scores(self, ctx: Dict) -> Dict[str, float]:
        \"\"\"Scores fuer alle Modelle berechnen.\"\"\"
        scores = dict(self.BASE_SCORES)

        # Gelernte Gewichte anwenden (addiert auf Base)
        for model, adj in self._learned_weights.items():
            if model in scores:
                scores[model] += adj"""

if old_scores_start in code:
    code = code.replace(old_scores_start, new_scores_start)
    print('FIX 1F: _compute_scores Learned Weights - OK')
    fixes += 1
else:
    print('FIX 1F: ANCHOR NOT FOUND!')

# --- FIX 1G: Lernfaehigkeit - Neue Methoden am Ende der Datei ---
# Vor dem letzten Return/Ende der Klasse einfuegen
# Finden wir das Ende der _compute_scores Methode

old_anti_starvation = """        # --- Anti-Starvation ---

        now = time.time()
        for model in self.ALL_MODELS:
            last = self._last_active.get(model, now)
            idle_mins = (now - last) / 60.0
            scores[model] += min(idle_mins * 0.1, 0.3)

        return scores"""

new_anti_starvation = """        # --- Anti-Starvation ---

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
        \"\"\"Ergebnis der letzten Inference loggen.

        Args:
            results: {
                "face_identified": bool,  # ArcFace hat Name geliefert
                "person_count": int,      # YOLO Person-Count
                "pose_useful": bool,      # Pose hat Keypoints erkannt
                "hand_detected": bool,    # Hand Landmark erkannt
                "face_detected": bool,    # SCRFD hat Face gefunden
            }
        \"\"\"
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
        \"\"\"Aus History lernen: Gewichte anpassen (max 10% pro Zyklus).\"\"\"
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
            if "pose" in models and results.get("pose_useful", False):
                model_useful["pose"] += 1
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
        \"\"\"Gelernte Gewichte aus perception_weights.json laden.\"\"\"
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
        \"\"\"Gelernte Gewichte in perception_weights.json speichern.\"\"\"
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
        \"\"\"History in perception_history.json speichern (letzte 200 Eintraege).\"\"\"
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
            _logger.warning(f"[LEARN] History speichern fehlgeschlagen: {e}")"""

if old_anti_starvation in code:
    code = code.replace(old_anti_starvation, new_anti_starvation)
    print('FIX 1G: Lernfaehigkeit Methoden - OK')
    fixes += 1
else:
    print('FIX 1G: ANCHOR NOT FOUND!')

# --- FIX 1H: get_state() erweitern ---
old_get_state_end = """            "hand_recency": self._FACE_RECENCY,
        }"""

new_get_state_end = """            "hand_recency": self._FACE_RECENCY,
            "learned_weights": dict(self._learned_weights),
            "decision_count": self._decision_count,
        }"""

if old_get_state_end in code:
    code = code.replace(old_get_state_end, new_get_state_end)
    print('FIX 1H: get_state() Weights + Count - OK')
    fixes += 1
else:
    print('FIX 1H: ANCHOR NOT FOUND!')

with open(pe_path, 'w') as f:
    f.write(code)

print(f'\nPerception Engine: {fixes}/8 Fixes.')
if fixes < 8:
    print('PERCEPTION ENGINE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 2: moloch_service.py - log_result() Aufruf
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

svc_fixes = 0

# --- FIX 2A: Nach Perception tick(), Ergebnis loggen ---
# Wir fuegen den log_result Aufruf NACH dem Perception-Swap-Block ein
old_auto_switch = """            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:"""

new_auto_switch = """            # Perception Engine: Ergebnis loggen (fuer Lernfaehigkeit)
            if hasattr(self, "_perception") and self._perception:
                _perc_results = {
                    "face_detected": face_detected,
                    "face_identified": face_detected and 'name' in dir() and name not in ("Unbekannt", None, ""),
                    "person_count": len(persons) if self.yolo_active and 'persons' in dir() and persons else 0,
                    "pose_useful": self.pose_active and 'poses' in dir() and bool(poses),
                    "hand_detected": getattr(self, '_last_hand_detected', False),
                }
                try:
                    self._perception.log_result(_perc_results)
                except Exception:
                    pass

            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:"""

if old_auto_switch in code:
    code = code.replace(old_auto_switch, new_auto_switch)
    print('FIX 2A: log_result() Aufruf - OK')
    svc_fixes += 1
else:
    print('FIX 2A: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {svc_fixes}/1 Fixes.')
if svc_fixes < 1:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 3: Initiale perception_weights.json (leer)
# ============================================================
weights_path = '/home/molochzuhause/moloch/config/perception_weights.json'
if not os.path.exists(weights_path):
    weights_data = {
        "version": 1,
        "total_decisions": 0,
        "weights": {},
        "base_scores": {
            "scrfd": 0.6,
            "arcface": 0.5,
            "yolov8m": 0.4,
            "pose": 0.3,
            "hand_landmark": 0.2
        },
        "effective_scores": {
            "scrfd": 0.6,
            "arcface": 0.5,
            "yolov8m": 0.4,
            "pose": 0.3,
            "hand_landmark": 0.2
        }
    }
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f, indent=2)
    print(f'\n3: perception_weights.json erstellt')
else:
    print(f'\n3: perception_weights.json existiert bereits')

# Sicherstellen dass data/ Ordner existiert
os.makedirs('/home/molochzuhause/moloch/data', exist_ok=True)

print('\n=== PERCEPTION LEARNING KOMPLETT ===')
