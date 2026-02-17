#!/usr/bin/env python3
"""Fix: Perception Engine darf waehrend Takeover-Transition keine Modelle swappen.

Problem: Waehrend Takeover wird yolov8m durch arcface ersetzt (Perception Swap).
Dann hat der Tracker kein YOLO fuer Person-Detection und findet nichts.

Fix: Perception tick wird NICHT aufgerufen waehrend:
- _transitioning = True (Takeover/Release laeuft)
- _waiting_for_first_detection = True (MOLOCH wartet auf erste Detection)
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Guard fuer Perception Swap waehrend Transition (aktive Inference-Pfad)
old = '_new_slots = self._perception.tick(_perc_ctx) if not self._calibration_active else None'
new = '_new_slots = self._perception.tick(_perc_ctx) if not self._calibration_active and not self._transitioning and not getattr(self, "_waiting_for_first_detection", False) else None'
count = code.count(old)
if count > 0:
    code = code.replace(old, new)
    print(f'FIX 1: Perception swap guard (transitioning + waiting) - OK ({count}x)')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Auch im idle-Pfad (kein Modell aktiv) den Guard einbauen
old_idle = '''                if hasattr(self, "_perception") and self._perception and not self._calibration_active:'''
new_idle = '''                if hasattr(self, "_perception") and self._perception and not self._calibration_active and not self._transitioning:'''
if old_idle in code:
    code = code.replace(old_idle, new_idle)
    print('FIX 2: Idle perception swap guard - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
