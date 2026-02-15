#!/usr/bin/env python3
"""Fix: Auto-Switch Counter nach AUSSEN verschieben.

Problem 1: Counter nur in 'if poses:' Block -> nie inkrementiert ohne Person
Problem 2: Presence fast immer >0.5 -> Hand immer "erkannt"
Fix: Counter am Ende der gesamten Inference, default=False pro Frame.
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# === Fix 1: _last_hand_detected am Anfang der Inference-Iteration setzen ===
# VOR dem any_active Check, nach dem Watchdog
old_watchdog = """            # === NPU WATCHDOG: Max-2 + Anti-Oszillation ===
            self._npu_watchdog()

            # Kein Modell konfiguriert ODER Inference pausiert -> Raw-Frame"""

new_watchdog = """            # === NPU WATCHDOG: Max-2 + Anti-Oszillation ===
            self._npu_watchdog()
            self._last_hand_detected = False  # Default: keine Hand pro Frame

            # Kein Modell konfiguriert ODER Inference pausiert -> Raw-Frame"""

if old_watchdog in code:
    code = code.replace(old_watchdog, new_watchdog)
    print('FIX 1: _last_hand_detected Default am Loop-Anfang - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# === Fix 2: Entferne den alten _last_hand_detected = False im Hand-Block ===
old_hand_init = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            self._last_hand_detected = False
                            for _pose in poses[:1]:"""

new_hand_init = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:"""

if old_hand_init in code:
    code = code.replace(old_hand_init, new_hand_init)
    print('FIX 2: Redundantes init im Hand-Block entfernt - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# === Fix 3: Auto-Switch Counter NACH gesamtem Pose-Block (nicht drin) ===
# Alten Counter-Block im Pose-Block entfernen
old_counter = """                        # Auto-Switch: Hand erkannt? Counter updaten
                        if self.hand_active and self._perception and self._perception._forced:
                            _any_hand = False
                            if "hand_landmark" in self._active_ctx:
                                # Hand-Block lief -> check ob _hand_res jemals gesetzt
                                _any_hand = getattr(self, '_last_hand_detected', False)
                            if _any_hand:
                                self._hand_no_detect = 0
                            else:
                                self._hand_no_detect += 1
                                if self._hand_no_detect >= self._HAND_RELEASE_FRAMES:
                                    logger.info(f"[AUTO-SWITCH] {self._HAND_RELEASE_FRAMES} Frames keine Hand -> zurueck zu Auto-Scoring")
                                    self._perception.force_models(None)
                                    self._hand_no_detect = 0

                        # Gesten-Erkennung aus Pose-Keypoints"""

new_counter = """                        # Gesten-Erkennung aus Pose-Keypoints"""

if old_counter in code:
    code = code.replace(old_counter, new_counter)
    print('FIX 3: Alter Counter aus Pose-Block entfernt - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# === Fix 4: Neuer Counter NACH dem gesamten Perception-Swap Block ===
# Am Ende der Inference-Iteration, nach dem SHM-Write
old_shm_end = """            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0"""

new_shm_end = """            # Auto-Switch: Hand-Forced zurueck zu Auto wenn keine Hand
            if self.hand_active and self._perception and self._perception._forced:
                if self._last_hand_detected:
                    self._hand_no_detect = 0
                else:
                    self._hand_no_detect += 1
                    if self._hand_no_detect >= self._HAND_RELEASE_FRAMES:
                        logger.info(f"[AUTO-SWITCH] {self._HAND_RELEASE_FRAMES} Frames keine Hand -> Auto-Scoring")
                        self._perception.force_models(None)
                        self._hand_no_detect = 0

            # Total FPS
            dt_total = time.perf_counter() - t_total
            with self._fps_lock:
                self._fps["total"] = 1.0 / dt_total if dt_total > 0 else 0"""

if old_shm_end in code:
    code = code.replace(old_shm_end, new_shm_end)
    print('FIX 4: Neuer Counter am Ende der Inference - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\n{fixes}/4 Fixes angewendet.')
if fixes < 4:
    sys.exit(1)
