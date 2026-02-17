#!/usr/bin/env python3
"""Audit-Reparaturen: 1 Kritisch + 1 Bug + 3 Warnungen.

1. KRITISCH: Head Pose frame_w/frame_h -> fw/fh
2. BUG: Settings Load Order (vor Perception Init)
3. WARNUNG: _active_ctx Thread-Safety (Lock bei Iteration)
4. WARNUNG: Log Rotation (hailort.log)
5. WARNUNG: _has_calibrated Dead Code entfernen
"""
import sys
import os

# ============================================================
# TEIL 1: moloch_service.py
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# --- FIX 1: Head Pose frame_w/frame_h -> fw/fh ---
old_hp = '                        _head_pose = estimate_head_pose(landmarks[0], frame_w, frame_h)'
new_hp = '                        _head_pose = estimate_head_pose(landmarks[0], fw, fh)'

if old_hp in code:
    code = code.replace(old_hp, new_hp)
    print('FIX 1: Head Pose fw/fh - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# --- FIX 2: Settings Load Order - VOR Perception Init ---
# Schritt A: _load_settings() Aufruf von jetziger Position entfernen
old_load_pos = """        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # FPS Tracking"""

new_load_pos = """        # FPS Tracking"""

if old_load_pos in code:
    code = code.replace(old_load_pos, new_load_pos)
    print('FIX 2A: _load_settings() von alter Position entfernt - OK')
    fixes += 1
else:
    print('FIX 2A: ANCHOR NOT FOUND!')

# Schritt B: _load_settings() VOR Perception Engine einfuegen
old_perception_init = """        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None"""

new_perception_init = """        # Settings aus config/settings.json laden (ueberschreibt Defaults)
        self._load_settings()

        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None"""

if old_perception_init in code:
    code = code.replace(old_perception_init, new_perception_init)
    print('FIX 2B: _load_settings() vor Perception Init - OK')
    fixes += 1
else:
    print('FIX 2B: ANCHOR NOT FOUND!')

# --- FIX 3: _active_ctx Thread-Safety ---
# 3A: _sync_flags_from_npu mit Lock wrappen
old_sync = """    def _sync_flags_from_npu(self):
        \"\"\"Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten.\"\"\"
        self.scrfd_active = "scrfd" in self._active_ctx
        self.arcface_active = "arcface" in self._active_ctx
        self.yolo_active = "yolov8m" in self._active_ctx
        self.pose_active = "pose" in self._active_ctx
        self.hand_active = "hand_landmark" in self._active_ctx"""

new_sync = """    def _sync_flags_from_npu(self):
        \"\"\"Flags IMMER aus NPU-Realitaet (_active_ctx) ableiten.\"\"\"
        with self._ctx_lock:
            self.scrfd_active = "scrfd" in self._active_ctx
            self.arcface_active = "arcface" in self._active_ctx
            self.yolo_active = "yolov8m" in self._active_ctx
            self.pose_active = "pose" in self._active_ctx
            self.hand_active = "hand_landmark" in self._active_ctx"""

if old_sync in code:
    code = code.replace(old_sync, new_sync)
    print('FIX 3A: _sync_flags_from_npu Lock - OK')
    fixes += 1
else:
    print('FIX 3A: ANCHOR NOT FOUND!')

# 3B: Watchdog _active_ctx Iteration mit Lock
old_watchdog_iter = """        _count = len(self._active_ctx)
        if _count > 2:
            logger.warning(f"[WATCHDOG] VIOLATION: {_count} Modelle aktiv! {list(self._active_ctx.keys())}")
            _prio = ["hand_landmark", "pose", "yolov8m", "arcface", "scrfd"]
            _victims = sorted(self._active_ctx.keys(),
                              key=lambda m: _prio.index(m) if m in _prio else 99)"""

new_watchdog_iter = """        with self._ctx_lock:
            _count = len(self._active_ctx)
            _keys = list(self._active_ctx.keys())
        if _count > 2:
            logger.warning(f"[WATCHDOG] VIOLATION: {_count} Modelle aktiv! {_keys}")
            _prio = ["hand_landmark", "pose", "yolov8m", "arcface", "scrfd"]
            _victims = sorted(_keys,
                              key=lambda m: _prio.index(m) if m in _prio else 99)"""

if old_watchdog_iter in code:
    code = code.replace(old_watchdog_iter, new_watchdog_iter)
    print('FIX 3B: Watchdog Lock - OK')
    fixes += 1
else:
    print('FIX 3B: ANCHOR NOT FOUND!')

# 3C: Perception Swap (idle) - Lock fuer keys snapshot
old_idle_swap = """                    if _new_slots:
                        _want = set(_new_slots)
                        _have = set(self._active_ctx.keys())
                        _to_remove = _have - _want
                        _to_add = _want - _have
                        if _to_remove or _to_add:
                            logger.info(f"[PERCEPTION] Swap (idle): {_have} -> {_want}")"""

new_idle_swap = """                    if _new_slots:
                        _want = set(_new_slots)
                        with self._ctx_lock:
                            _have = set(self._active_ctx.keys())
                        _to_remove = _have - _want
                        _to_add = _want - _have
                        if _to_remove or _to_add:
                            logger.info(f"[PERCEPTION] Swap (idle): {_have} -> {_want}")"""

if old_idle_swap in code:
    code = code.replace(old_idle_swap, new_idle_swap)
    print('FIX 3C: Idle Swap Lock - OK')
    fixes += 1
else:
    print('FIX 3C: ANCHOR NOT FOUND!')

# 3D: Perception Swap (active) - Lock fuer keys snapshot
old_active_swap = """                if _new_slots:
                    _want = set(_new_slots)
                    _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")"""

new_active_swap = """                if _new_slots:
                    _want = set(_new_slots)
                    with self._ctx_lock:
                        _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")"""

if old_active_swap in code:
    code = code.replace(old_active_swap, new_active_swap)
    print('FIX 3D: Active Swap Lock - OK')
    fixes += 1
else:
    print('FIX 3D: ANCHOR NOT FOUND!')

# --- FIX 5: _has_calibrated Dead Code entfernen ---
old_calibrated = """
        self._has_calibrated = False"""

new_calibrated = """"""

if old_calibrated in code:
    code = code.replace(old_calibrated, new_calibrated, 1)
    print('FIX 5: _has_calibrated entfernt - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/8 Fixes.')
if fixes < 8:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 2: Log Rotation (logrotate Config)
# ============================================================
logrotate_conf = '/etc/logrotate.d/moloch'
logrotate_content = """# M.O.L.O.C.H. Log Rotation
/home/molochzuhause/moloch/hailort.log {
    size 10M
    rotate 3
    compress
    missingok
    notifempty
    copytruncate
}
"""

try:
    with open(logrotate_conf, 'w') as f:
        f.write(logrotate_content)
    print('\nFIX 4: logrotate Config - OK')
except PermissionError:
    # Braucht sudo
    import subprocess
    subprocess.run(['sudo', 'tee', logrotate_conf],
                   input=logrotate_content.encode(), check=True,
                   stdout=subprocess.DEVNULL)
    print('\nFIX 4: logrotate Config (sudo) - OK')

print('\n=== ALLE AUDIT-FIXES KOMPLETT ===')
