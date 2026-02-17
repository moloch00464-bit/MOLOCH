#!/usr/bin/env python3
"""Fix: MOLOCH verliert Tracking - bessere Recovery.

Probleme:
1. Nach Release zeigt Kamera evtl. an Decke -> ST findet niemanden
2. Cooldown 60s+ bei fehlgeschlagenem Takeover zu lang
3. Search-Timeout 20s zu lang wenn nichts da ist

Fixes:
1. _moloch_release: Kamera zur Mitte bewegen nach fehlgeschlagenem Takeover
2. RELEASE_COOLDOWN: 60 -> 30s, MAX_COOLDOWN: 180 -> 120s
3. SEARCH_TIMEOUT: 20 -> 12s
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Kuerzerer RELEASE_COOLDOWN (60 -> 30)
old = 'self.RELEASE_COOLDOWN = 60'
new = 'self.RELEASE_COOLDOWN = 30'
if old in code:
    code = code.replace(old, new)
    print('FIX 1: RELEASE_COOLDOWN 60 -> 30 - OK')
    fixes += 1
else:
    print('FIX 1: RELEASE_COOLDOWN ANCHOR NOT FOUND!')

# FIX 2: Kuerzerer MAX_COOLDOWN (180 -> 120)
old = 'self.MAX_COOLDOWN = 180'
new = 'self.MAX_COOLDOWN = 120'
if old in code:
    code = code.replace(old, new)
    print('FIX 2: MAX_COOLDOWN 180 -> 120 - OK')
    fixes += 1
else:
    print('FIX 2: MAX_COOLDOWN ANCHOR NOT FOUND!')

# FIX 3: Kuerzerer SEARCH_TIMEOUT (20 -> 12)
old = 'self.SEARCH_TIMEOUT = 20'
new = 'self.SEARCH_TIMEOUT = 12'
if old in code:
    code = code.replace(old, new)
    print('FIX 3: SEARCH_TIMEOUT 20 -> 12 - OK')
    fixes += 1
else:
    print('FIX 3: SEARCH_TIMEOUT ANCHOR NOT FOUND!')

# FIX 4: Nach fehlgeschlagenem Takeover Kamera zur Mitte bewegen
# VORHER: Nur Position-Tracking zuruecksetzen
# NACHHER: Kamera aktiv zur Mitte bewegen wenn nichts gefunden
old_block = '''            # Position-Tracking zuruecksetzen
            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0

            # Progressive Backoff (1.5x, max 180s)
            if self._takeover_found_something:'''

new_block = '''            # Position-Tracking zuruecksetzen
            self._guardian_last_pan = None
            self._guardian_last_tilt = None
            self._guardian_move_count = 0

            # Kamera zur Mitte wenn nichts gefunden (verhindert "Decke starren")
            if not self._takeover_found_something:
                try:
                    cam = self.get_camera_controller()
                    if cam and cam.is_connected:
                        cam.move_absolute(0, 10)  # Mitte, leicht nach oben
                        logger.info("[TENTAKEL] Kamera zur Mitte bewegt (nichts gefunden)")
                except Exception as e:
                    logger.warning(f"[TENTAKEL] Kamera-Reset fehlgeschlagen: {e}")

            # Progressive Backoff (1.5x, max 120s)
            if self._takeover_found_something:'''

if old_block in code:
    code = code.replace(old_block, new_block)
    print('FIX 4: Kamera-Reset zur Mitte nach Failed Takeover - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
