#!/usr/bin/env python3
"""Fix: log_result nur jeden 15. Frame (1/s statt 15/s).

100 Entscheidungen = ~100 Sekunden statt 7 Sekunden.
Und Weights zuruecksetzen (waren durch zu schnelles Lernen verzerrt).
"""
import sys
import json

# ============================================================
# TEIL 1: perception_engine.py - Throttle log_result
# ============================================================
pe_path = '/home/molochzuhause/moloch/core/perception_engine.py'
with open(pe_path) as f:
    code = f.read()

fixes = 0

# --- FIX 1A: _log_skip_counter in __init__ ---
old_init = """        # Lernfaehigkeit
        self._history: list = []
        self._learned_weights: Dict[str, float] = {}
        self._last_context: Dict = {}
        self._last_chosen: List[str] = []
        self._decision_count = 0
        self._load_weights()"""

new_init = """        # Lernfaehigkeit
        self._history: list = []
        self._learned_weights: Dict[str, float] = {}
        self._last_context: Dict = {}
        self._last_chosen: List[str] = []
        self._decision_count = 0
        self._log_skip_counter = 0
        self._LOG_SAMPLE_RATE = 15  # Nur jeden 15. Frame loggen (1/s bei 15fps)
        self._load_weights()"""

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 1A: _log_skip_counter - OK')
    fixes += 1
else:
    print('FIX 1A: ANCHOR NOT FOUND!')

# --- FIX 1B: log_result() Throttle ---
old_log_start = """    def log_result(self, results: Dict):
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
            self._last_chosen = list(self.slots)"""

new_log_start = """    def log_result(self, results: Dict):
        \"\"\"Ergebnis der letzten Inference loggen (1x pro Sekunde).

        Args:
            results: {
                "face_identified": bool,  # ArcFace hat Name geliefert
                "person_count": int,      # YOLO Person-Count
                "pose_useful": bool,      # Pose hat Keypoints erkannt
                "hand_detected": bool,    # Hand Landmark erkannt
                "face_detected": bool,    # SCRFD hat Face gefunden
            }
        \"\"\"
        # Throttle: nur jeden N-ten Frame loggen
        self._log_skip_counter += 1
        if self._log_skip_counter < self._LOG_SAMPLE_RATE:
            return
        self._log_skip_counter = 0

        if not self._last_chosen:
            self._last_chosen = list(self.slots)"""

if old_log_start in code:
    code = code.replace(old_log_start, new_log_start)
    print('FIX 1B: log_result() Throttle - OK')
    fixes += 1
else:
    print('FIX 1B: ANCHOR NOT FOUND!')

with open(pe_path, 'w') as f:
    f.write(code)

print(f'\nPerception Engine: {fixes}/2 Fixes.')

# ============================================================
# TEIL 2: Weights zuruecksetzen (waren verzerrt)
# ============================================================
weights_path = '/home/molochzuhause/moloch/config/perception_weights.json'
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
with open(weights_path, 'w', encoding='utf-8') as f:
    json.dump(weights_data, f, indent=2)
print('\nWeights zurueckgesetzt (clean start)')

if fixes < 2:
    print('INCOMPLETE!')
    sys.exit(1)

print('\n=== THROTTLE FIX KOMPLETT ===')
