#!/usr/bin/env python3
"""Perception Config: JSON erstellen + Service-Integration.

1. config/perception.json mit Defaults anlegen
2. moloch_service.py: Pfad-Konstante + _load_perception_config + mtime Polling
"""
import sys
import json
import os

# ============================================================
# TEIL 1: config/perception.json erstellen
# ============================================================
perc_path = '/home/molochzuhause/moloch/config/perception.json'

perc_data = {
    "version": 1,
    "face": {
        "confidence_threshold": 0.40,
        "nms_threshold": 0.40,
        "recognition_threshold": 0.60,
        "min_size": 0.08,
        "max_size": 0.65,
        "position_smoothing": 0.5,
        "landmarks_enabled": True,
        "head_pose_enabled": True
    },
    "hand": {
        "confidence_threshold": 0.30,
        "position_smoothing": 0.5,
        "min_crop_size": 140,
        "max_crop_size": 300,
        "max_hands": 1,
        "landmarks_enabled": True,
        "occlusion_timeout": 5.0,
        "occlusion_streak": 3,
        "occlusion_recency": 2.0
    },
    "pose": {
        "confidence_threshold": 0.50,
        "nms_threshold": 0.70,
        "position_smoothing": 0.5,
        "motion_sensitivity": 0.5,
        "max_detections": 10,
        "landmarks_enabled": True
    },
    "global": {
        "min_confidence": 0.50,
        "position_smoothing": 0.5,
        "max_tracked_objects": 3,
        "min_bbox_area": 0.08,
        "filter_jitter": True,
        "filter_small_objects": True,
        "filter_outliers": True
    },
    "npu": {
        "max_fps": 15,
        "power_mode": "balanced",
        "perception_enabled": True,
        "swap_interval": 10.0,
        "swap_hysteresis": 0.15,
        "base_scrfd": 0.6,
        "base_arcface": 0.5,
        "base_yolov8m": 0.4,
        "base_pose": 0.3,
        "base_hand": 0.2
    },
    "debug": {
        "show_bboxes": True,
        "show_landmarks": True,
        "show_names": True,
        "show_fps": True,
        "show_confidence": True,
        "show_head_pose": True,
        "show_skeleton": True,
        "show_hand_crop": False,
        "display_flip_h": False,
        "display_flip_v": False,
        "display_rotation": 0
    }
}

with open(perc_path, 'w', encoding='utf-8') as f:
    json.dump(perc_data, f, indent=2, ensure_ascii=False)
print(f'1: perception.json erstellt ({perc_path})')

# ============================================================
# TEIL 2: moloch_service.py Patches
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# --- 2A: PERCEPTION_CONFIG_PATH nach SETTINGS_PATH ---
old_settings_path = 'SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")'

new_settings_path = '''SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.json")
PERCEPTION_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "perception.json")'''

if old_settings_path in code:
    code = code.replace(old_settings_path, new_settings_path)
    print('2A: PERCEPTION_CONFIG_PATH - OK')
    fixes += 1
else:
    print('2A: ANCHOR NOT FOUND!')

# --- 2B: _load_perception_config() Aufruf nach _load_settings() ---
old_load_call = """        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # Perception Engine (NPU Slot-Rotation mit Personality)"""

new_load_call = """        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # Perception Config laden (ueberschreibt Thresholds aus settings.json)
        self._perc_config_mtime = 0.0
        self._load_perception_config()

        # Perception Engine (NPU Slot-Rotation mit Personality)"""

if old_load_call in code:
    code = code.replace(old_load_call, new_load_call)
    print('2B: _load_perception_config() Aufruf - OK')
    fixes += 1
else:
    print('2B: ANCHOR NOT FOUND!')

# --- 2C: mtime Polling in Inference Loop (am Anfang jeder Iteration) ---
old_loop_start = """            t_total = time.perf_counter()
            annotated = frame.copy()
            fh, fw = frame.shape[:2]"""

new_loop_start = """            t_total = time.perf_counter()

            # Perception Config Hot-Reload (alle ~3s = ~45 Frames bei 15fps)
            self._perc_poll_counter = getattr(self, '_perc_poll_counter', 0) + 1
            if self._perc_poll_counter >= 45:
                self._perc_poll_counter = 0
                try:
                    _mt = os.path.getmtime(PERCEPTION_CONFIG_PATH)
                    if _mt > self._perc_config_mtime:
                        self._load_perception_config()
                except FileNotFoundError:
                    pass

            annotated = frame.copy()
            fh, fw = frame.shape[:2]"""

if old_loop_start in code:
    code = code.replace(old_loop_start, new_loop_start)
    print('2C: mtime Polling in Inference Loop - OK')
    fixes += 1
else:
    print('2C: ANCHOR NOT FOUND!')

# --- 2D: _load_perception_config() Methode vor _load_settings() ---
old_settings_method = """    # ----------------------------------------------------------------
    # Settings Persistence
    # ----------------------------------------------------------------
    def _load_settings(self):"""

new_settings_method = """    # ----------------------------------------------------------------
    # Perception Config (config/perception.json)
    # ----------------------------------------------------------------
    def _load_perception_config(self):
        \"\"\"Lade Perception-Parameter aus config/perception.json (Hot-Reload).\"\"\"
        if not os.path.exists(PERCEPTION_CONFIG_PATH):
            return
        try:
            with open(PERCEPTION_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._perc_config_mtime = os.path.getmtime(PERCEPTION_CONFIG_PATH)
        except Exception as e:
            logger.warning(f"[PERC-CFG] perception.json Fehler: {e}")
            return

        logger.info(f"[PERC-CFG] Lade perception.json (version={data.get('version', '?')})")

        # Face
        face = data.get("face", {})
        self.scrfd_conf_val = float(face.get("confidence_threshold", self.scrfd_conf_val))
        self.scrfd_nms_val = float(face.get("nms_threshold", self.scrfd_nms_val))
        self.arcface_thresh_val = float(face.get("recognition_threshold", self.arcface_thresh_val))
        self._perc_face_min_size = float(face.get("min_size", 0.08))
        self._perc_face_max_size = float(face.get("max_size", 0.65))
        self._perc_face_smoothing = float(face.get("position_smoothing", 0.5))
        self._perc_face_landmarks = bool(face.get("landmarks_enabled", True))
        self._perc_head_pose = bool(face.get("head_pose_enabled", True))

        # Hand
        hand = data.get("hand", {})
        self._perc_hand_conf = float(hand.get("confidence_threshold", 0.30))
        self._perc_hand_smoothing = float(hand.get("position_smoothing", 0.5))
        self._perc_hand_min_crop = int(hand.get("min_crop_size", 140))
        self._perc_hand_max_crop = int(hand.get("max_crop_size", 300))
        self._perc_hand_max_hands = int(hand.get("max_hands", 1))
        self._perc_hand_landmarks = bool(hand.get("landmarks_enabled", True))
        _ho_timeout = float(hand.get("occlusion_timeout", 5.0))
        _ho_streak = int(hand.get("occlusion_streak", 3))
        _ho_recency = float(hand.get("occlusion_recency", 2.0))

        # Pose
        pose = data.get("pose", {})
        self.pose_conf_val = float(pose.get("confidence_threshold", self.pose_conf_val))
        self.pose_nms_val = float(pose.get("nms_threshold", self.pose_nms_val))
        self._perc_pose_smoothing = float(pose.get("position_smoothing", 0.5))
        self._perc_pose_motion = float(pose.get("motion_sensitivity", 0.5))
        self._perc_pose_max_det = int(pose.get("max_detections", 10))
        self._perc_pose_landmarks = bool(pose.get("landmarks_enabled", True))

        # Global
        glob = data.get("global", {})
        self._perc_global_conf = float(glob.get("min_confidence", 0.50))
        self._perc_global_smooth = float(glob.get("position_smoothing", 0.5))
        self._perc_global_max_obj = int(glob.get("max_tracked_objects", 3))
        self._perc_global_min_area = float(glob.get("min_bbox_area", 0.08))
        self._perc_filter_jitter = bool(glob.get("filter_jitter", True))
        self._perc_filter_small = bool(glob.get("filter_small_objects", True))
        self._perc_filter_outliers = bool(glob.get("filter_outliers", True))

        # NPU / Perception Engine
        npu = data.get("npu", {})
        self._perc_max_fps = int(npu.get("max_fps", 15))
        self._perc_power_mode = str(npu.get("power_mode", "balanced"))
        self._perc_enabled = bool(npu.get("perception_enabled", True))

        # Debug
        dbg = data.get("debug", {})
        self._dbg_bboxes = bool(dbg.get("show_bboxes", True))
        self._dbg_landmarks = bool(dbg.get("show_landmarks", True))
        self._dbg_names = bool(dbg.get("show_names", True))
        self._dbg_fps = bool(dbg.get("show_fps", True))
        self._dbg_confidence = bool(dbg.get("show_confidence", True))
        self._dbg_head_pose = bool(dbg.get("show_head_pose", True))
        self._dbg_skeleton = bool(dbg.get("show_skeleton", True))
        self._dbg_hand_crop = bool(dbg.get("show_hand_crop", False))
        self._dbg_flip_h = bool(dbg.get("display_flip_h", False))
        self._dbg_flip_v = bool(dbg.get("display_flip_v", False))
        self._dbg_rotation = int(dbg.get("display_rotation", 0))

        # Perception Engine Params anwenden (wenn vorhanden)
        if self._perception:
            _swap = float(npu.get("swap_interval", 10.0))
            _hyst = float(npu.get("swap_hysteresis", 0.15))
            self._perception._min_interval = _swap
            self._perception._hysteresis = _hyst
            self._perception._HAND_TIMEOUT = _ho_timeout
            self._perception._MIN_FACE_STREAK = _ho_streak
            self._perception._FACE_RECENCY = _ho_recency
            # Base-Scores
            self._perception.BASE_SCORES["scrfd"] = float(npu.get("base_scrfd", 0.6))
            self._perception.BASE_SCORES["arcface"] = float(npu.get("base_arcface", 0.5))
            self._perception.BASE_SCORES["yolov8m"] = float(npu.get("base_yolov8m", 0.4))
            self._perception.BASE_SCORES["pose"] = float(npu.get("base_pose", 0.3))
            self._perception.BASE_SCORES["hand_landmark"] = float(npu.get("base_hand", 0.2))

        # Tracker Params anwenden (wenn vorhanden)
        if hasattr(self, '_tracker') and self._tracker:
            try:
                self._tracker.config.smooth_alpha = self._perc_face_smoothing
                self._tracker.config.min_confidence = self._perc_global_conf
                self._tracker.config.min_bbox_area_ratio = self._perc_global_min_area
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Settings Persistence
    # ----------------------------------------------------------------
    def _load_settings(self):"""

if old_settings_method in code:
    code = code.replace(old_settings_method, new_settings_method, 1)
    print('2D: _load_perception_config() Methode - OK')
    fixes += 1
else:
    print('2D: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/4 Fixes.')
if fixes < 4:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

print('\n=== PERCEPTION CONFIG SERVICE KOMPLETT ===')
