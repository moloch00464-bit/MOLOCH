#!/usr/bin/env python3
"""Fix: Perception Engine waehrend Kalibrierung pausieren.

Problem: Waehrend Gender/Age/Emotions Kalibrierung (nur CPU) laedt
die Perception Engine unnoetig NPU-Modelle (Pose, Hand, YOLO).

Loesung: Perception ticks skippen wenn _calibration_active=True.
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Idle perception tick skippen
old_idle_tick = '''            if not any_active:
                # Perception tick auch ohne aktive Modelle (forced/initial swap)
                if hasattr(self, "_perception") and self._perception:'''

new_idle_tick = '''            if not any_active:
                # Perception tick auch ohne aktive Modelle (forced/initial swap)
                # SKIP waehrend Kalibrierung (braucht keine NPU-Modelle)
                if hasattr(self, "_perception") and self._perception and not self._calibration_active:'''

if old_idle_tick in code:
    code = code.replace(old_idle_tick, new_idle_tick)
    print('FIX 1: Idle perception tick skip - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Active perception tick skippen
old_active_tick = '''                _new_slots = self._perception.tick(_perc_ctx)
                if _new_slots:
                    _want = set(_new_slots)
                    with self._ctx_lock:
                        _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")'''

new_active_tick = '''                _new_slots = self._perception.tick(_perc_ctx) if not self._calibration_active else None
                if _new_slots:
                    _want = set(_new_slots)
                    with self._ctx_lock:
                        _have = set(self._active_ctx.keys())
                    _to_remove = _have - _want
                    _to_add = _want - _have
                    if _to_remove or _to_add:
                        logger.info(f"[PERCEPTION] Swap: {_have} -> {_want} (occlusion={self._perception._hand_occlusion})")'''

if old_active_tick in code:
    code = code.replace(old_active_tick, new_active_tick)
    print('FIX 2: Active perception tick skip - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: Tentakel idle pre-load skippen
old_preload = '''            # Idle Pre-Load: NPU Modelle vorladen wenn Kamera still
            if hasattr(self, "_perception") and self._perception:'''

new_preload = '''            # Idle Pre-Load: NPU Modelle vorladen wenn Kamera still
            # SKIP waehrend Kalibrierung
            if hasattr(self, "_perception") and self._perception and not self._calibration_active:'''

if old_preload in code:
    code = code.replace(old_preload, new_preload, 1)
    print('FIX 3: Tentakel Pre-Load skip - OK')
    fixes += 1
else:
    # Versuche alternative Formulierung
    old_preload2 = '''                # Pre-Load Modelle wenn Kamera still steht'''
    if old_preload2 in code:
        print('FIX 3: Alternative Anchor, manuell pruefen')
    else:
        print('FIX 3: ANCHOR NOT FOUND (Pre-Load nicht gefunden, evtl anders formuliert)')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\nService: {fixes} Fixes. Syntax OK.')

if fixes < 2:
    print('WARNUNG: Mindestens Idle + Active Tick muessen gefixt sein!')
    sys.exit(1)

print('\n=== PERCEPTION PAUSE WAEHREND KALIBRIERUNG KOMPLETT ===')
