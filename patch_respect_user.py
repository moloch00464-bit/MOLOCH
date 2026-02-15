#!/usr/bin/env python3
"""Fix: Takeover respektiert User-Modell-Wahl.

Problem: Takeover cleared forced_models und lud scrfd+yolov8m.
Fix: Wenn User Modelle manuell gewaehlt hat (forced != None),
     Takeover NICHT NPU umkonfigurieren.
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# === Fix 1: Takeover Skip wenn User Modelle forced hat ===
old_takeover = """        def do_takeover():
            try:
                # 1. NPU Modelle aktivieren (ST bleibt AN!)
                models_cached = "scrfd" in self._active_ctx and "yolov8m" in self._active_ctx"""

new_takeover = """        def do_takeover():
            try:
                # User hat Modelle manuell gewaehlt? -> NPU nicht antasten!
                if self._perception and self._perception._forced:
                    logger.info(f"[TENTAKEL] User forced_models={self._perception._forced} - NPU bleibt!")
                    # Takeover-Flags setzen fuer Kamera-Kontrolle
                    self._sync_flags_from_npu()
                    self._first_detection_event.set()  # Skip Detection-Wait
                    self._transitioning = False
                    return

                # 1. NPU Modelle aktivieren (ST bleibt AN!)
                models_cached = "scrfd" in self._active_ctx and "yolov8m" in self._active_ctx"""

if old_takeover in code:
    code = code.replace(old_takeover, new_takeover)
    print('FIX 1: Takeover respektiert forced_models - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# === Fix 2: Takeover NICHT forced_models clearen (Revert Fix D) ===
old_clear = """                # 2. Inference starten - ALLE Flags sauber setzen
                self.scrfd_active = True
                self.arcface_active = False
                self.yolo_active = True
                self.pose_active = False
                self.hand_active = False
                # User-forced Models aufheben (Takeover hat Vorrang)
                if self._perception and self._perception._forced:
                    logger.info(f"[TENTAKEL] Cleared forced_models={self._perception._forced}")
                    self._perception.force_models(None)
                self._notify("model_toggle", {
                    "scrfd": True, "arcface": False, "yolov8m": True,
                    "pose": False, "hand_landmark": False})"""

new_clear = """                # 2. Inference starten - Flags aus NPU-Realitaet
                self._sync_flags_from_npu()
                self._notify("model_toggle", {
                    "scrfd": self.scrfd_active, "arcface": self.arcface_active,
                    "yolov8m": self.yolo_active, "pose": self.pose_active,
                    "hand_landmark": self.hand_active})"""

if old_clear in code:
    code = code.replace(old_clear, new_clear)
    print('FIX 2: Takeover Flag-Sync statt forced clear - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# === Fix 3: Release NICHT forced_models clearen (Revert Fix E) ===
old_release = """            self.hand_active = False
            if self._perception and self._perception._forced:
                self._perception.force_models(None)"""

new_release = """            self.hand_active = False"""

if old_release in code:
    code = code.replace(old_release, new_release)
    print('FIX 3: Release behaelt forced_models - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\n{fixes}/3 Fixes angewendet.')
if fixes < 3:
    sys.exit(1)
