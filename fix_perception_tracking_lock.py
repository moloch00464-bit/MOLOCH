#!/usr/bin/env python3
"""Fix: Perception darf waehrend MOLOCH-Tracking KEINE Modelle swappen.

Root Cause: Perception Engine swapped yolov8m -> hand_landmark/pose waehrend
der Tracker aktiv trackt. Tracker verliert damit seine Detection-Quelle und
findet nichts mehr -> Search timeout -> Release.

Fix: Perception tick blockiert wenn _moloch_has_control UND _autonomous_mode.
Perception darf nur swappen wenn:
- Tentakel idle (Smart Tracking aktiv, kein Tracking)
- Manueller Modus
- Nach Release
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Aktive Inference-Pfad - striktes Lock bei MOLOCH-Kontrolle
old = '_new_slots = self._perception.tick(_perc_ctx) if not self._calibration_active and not self._transitioning and not getattr(self, "_waiting_for_first_detection", False) else None'
new = '_new_slots = self._perception.tick(_perc_ctx) if not self._calibration_active and not self._transitioning and not getattr(self, "_waiting_for_first_detection", False) and not (self._moloch_has_control and self._autonomous_mode) else None'
if old in code:
    code = code.replace(old, new)
    print('FIX 1: Perception lock bei MOLOCH-Tracking (aktiver Pfad) - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Idle-Pfad - auch dort Lock bei MOLOCH-Kontrolle
old_idle = 'if hasattr(self, "_perception") and self._perception and not self._calibration_active and not self._transitioning:'
new_idle = 'if hasattr(self, "_perception") and self._perception and not self._calibration_active and not self._transitioning and not (self._moloch_has_control and self._autonomous_mode):'
if old_idle in code:
    code = code.replace(old_idle, new_idle)
    print('FIX 2: Perception lock bei MOLOCH-Tracking (idle Pfad) - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
