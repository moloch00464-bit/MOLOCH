#!/usr/bin/env python3
"""Patch: Hand-Landmark als eigene Checkbox, getrennt von Pose.

Aenderungen:
1. Neue Checkbox "Hand LM" mit FPS-Anzeige
2. In _on_model_toggle, _sync_model_toggles, _update_fps eingetragen
3. _apply_status synct hand_active Status
4. Hand-Occlusion Section umbenannt fuer Klarheit
"""
import sys

panel_path = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel_path, 'r') as f:
    panel = f.read()

fixes_ok = 0

# === Fix 1: hand_lm_var + Checkbox nach Pose ===
old_pose_section = """        # Pose
        self._pose_fps = self._build_model_section(
            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

        # --- Hand-Occlusion Controls ---"""

new_pose_section = """        # Pose
        self._pose_fps = self._build_model_section(
            model_frame, "Pose", self.pose_var, "pose",
            [("Conf", self.pose_conf_var, 0.1, 0.9),
             ("NMS", self.pose_nms_var, 0.1, 0.9)])

        # Hand Landmark (braucht Pose als Dependency)
        self.hand_lm_var = tk.BooleanVar(value=False)
        self._hand_lm_fps = self._build_model_section(
            model_frame, "Hand LM", self.hand_lm_var, "hand_landmark", [])

        # --- Hand-Occlusion (Auto-Erkennung) ---"""

if old_pose_section in panel:
    panel = panel.replace(old_pose_section, new_pose_section)
    print('FIX 1: Hand LM Checkbox + Section - OK')
    fixes_ok += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# === Fix 2: _on_model_toggle - hand_landmark in var_map ===
old_var_map = """        var_map = {
            "scrfd": self.scrfd_var,
            "arcface": self.arcface_var,
            "yolov8m": self.yolo_var,
            "pose": self.pose_var,
        }"""

new_var_map = """        var_map = {
            "scrfd": self.scrfd_var,
            "arcface": self.arcface_var,
            "yolov8m": self.yolo_var,
            "pose": self.pose_var,
            "hand_landmark": self.hand_lm_var,
        }"""

if old_var_map in panel:
    panel = panel.replace(old_var_map, new_var_map)
    print('FIX 2: var_map mit hand_landmark - OK')
    fixes_ok += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# === Fix 3: _sync_model_toggles - hand_landmark in toggle_map ===
old_toggle_map = """            toggle_map = {
                "scrfd": self.scrfd_var,
                "arcface": self.arcface_var,
                "yolov8m": self.yolo_var,
                "pose": self.pose_var,
            }"""

new_toggle_map = """            toggle_map = {
                "scrfd": self.scrfd_var,
                "arcface": self.arcface_var,
                "yolov8m": self.yolo_var,
                "pose": self.pose_var,
                "hand_landmark": self.hand_lm_var,
            }"""

if old_toggle_map in panel:
    panel = panel.replace(old_toggle_map, new_toggle_map)
    print('FIX 3: toggle_map mit hand_landmark - OK')
    fixes_ok += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# === Fix 4: _update_fps - hand_landmark FPS ===
old_fps = """            for key, label in [("scrfd", self._scrfd_fps),
                                ("arcface", self._arcface_fps),
                                ("yolov8m", self._yolov8m_fps),
                                ("pose", self._pose_fps)]:"""

new_fps = """            for key, label in [("scrfd", self._scrfd_fps),
                                ("arcface", self._arcface_fps),
                                ("yolov8m", self._yolov8m_fps),
                                ("pose", self._pose_fps),
                                ("hand_landmark", self._hand_lm_fps)]:"""

if old_fps in panel:
    panel = panel.replace(old_fps, new_fps)
    print('FIX 4: FPS-Update mit hand_landmark - OK')
    fixes_ok += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# === Fix 5: _apply_status - hand_active synchen ===
old_apply = """        self.pose_active = s.get('pose_active', False)
        # Bei Aenderung -> Checkboxen synchronisieren
        _curr = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
        }"""

new_apply = """        self.pose_active = s.get('pose_active', False)
        self.hand_active = s.get('hand_active', False)
        # Bei Aenderung -> Checkboxen synchronisieren
        _curr = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
            "hand_landmark": self.hand_active,
        }"""

if old_apply in panel:
    panel = panel.replace(old_apply, new_apply)
    print('FIX 5: _apply_status mit hand_active - OK')
    fixes_ok += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

# === Fix 5b: _prev dict auch erweitern ===
old_prev = """        _prev = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
        }"""

new_prev = """        _prev = {
            "scrfd": self.scrfd_active,
            "arcface": self.arcface_active,
            "yolov8m": self.yolo_active,
            "pose": self.pose_active,
            "hand_landmark": getattr(self, 'hand_active', False),
        }"""

if old_prev in panel:
    panel = panel.replace(old_prev, new_prev)
    print('FIX 5b: _prev dict mit hand_landmark - OK')
    fixes_ok += 1
else:
    print('FIX 5b: ANCHOR NOT FOUND!')

# === Fix 6: ServiceProxy braucht hand_active Attribut ===
old_proxy_models = """        # Model states
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.pose_active = False"""

new_proxy_models = """        # Model states
        self.scrfd_active = False
        self.arcface_active = False
        self.yolo_active = False
        self.pose_active = False
        self.hand_active = False"""

if new_proxy_models in panel:
    print('FIX 6: ServiceProxy hand_active - bereits vorhanden, SKIP')
    fixes_ok += 1
elif old_proxy_models in panel:
    panel = panel.replace(old_proxy_models, new_proxy_models)
    print('FIX 6: ServiceProxy hand_active - OK')
    fixes_ok += 1
else:
    print('FIX 6: ANCHOR NOT FOUND!')

# === Fix 7: Seeding - hand_lm_var setzen ===
old_seed = """        self.pose_var.set(self.service.pose_active)
        self._syncing = False"""

new_seed = """        self.pose_var.set(self.service.pose_active)
        self.hand_lm_var.set(getattr(self.service, 'hand_active', False))
        self._syncing = False"""

if old_seed in panel:
    panel = panel.replace(old_seed, new_seed, 1)  # Nur erste Occurrence
    print('FIX 7: Seeding hand_lm_var - OK')
    fixes_ok += 1
else:
    print('FIX 7: ANCHOR NOT FOUND!')

# === Fix 8: Hand-Occlusion Section Label klarer ===
old_label = """        self.hand_var = tk.BooleanVar(value=True)
        hand_cb = tk.Checkbutton(hand_header, text="Hand-Erkennung","""

new_label = """        self.hand_var = tk.BooleanVar(value=True)
        hand_cb = tk.Checkbutton(hand_header, text="Auto-Occlusion","""

if old_label in panel:
    panel = panel.replace(old_label, new_label)
    print('FIX 8: Label "Auto-Occlusion" - OK')
    fixes_ok += 1
else:
    print('FIX 8: ANCHOR NOT FOUND!')

with open(panel_path, 'w') as f:
    f.write(panel)

print(f'\n{fixes_ok}/8 Fixes erfolgreich.')
if fixes_ok < 8:
    sys.exit(1)
