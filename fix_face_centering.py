#!/usr/bin/env python3
"""Fix: Tracker zentriert Gesicht statt Torso bei Person-Detection.

Problem: YOLO Person-BBox hat die Mitte am Torso, nicht am Gesicht.
Tracker zentriert auf BBox-Mitte -> Gesicht ist im oberen Drittel, nicht zentriert.

Fix: Person-BBox um 25% der Hoehe nach oben shiften, bevor sie an den Tracker
geht. Gleiche Hoehe (Filter passiert), aber Zentrum am Kopf/Gesicht.
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# Finde den YOLO-Person -> Tracker Feed Code
old = '''                        if self._autonomous_mode and self._tracker and not face_fed_to_tracker:
                            try:
                                pixel_dets = []
                                for p in persons:
                                    bx = p["bbox"]
                                    pixel_dets.append({
                                        "bbox": [bx[0] * 640, bx[1] * 640, bx[2] * 640, bx[3] * 640],
                                        "confidence": p["confidence"],
                                        "class": "person"
                                    })'''

new = '''                        if self._autonomous_mode and self._tracker and not face_fed_to_tracker:
                            try:
                                pixel_dets = []
                                for p in persons:
                                    bx = p["bbox"]
                                    # BBox nach oben shiften: Tracker zielt auf Kopf statt Torso
                                    y1 = bx[1]
                                    y2 = bx[3]
                                    shift = (y2 - y1) * 0.25
                                    y1s = max(0, y1 - shift)
                                    y2s = max(0, y2 - shift)
                                    pixel_dets.append({
                                        "bbox": [bx[0] * 640, y1s * 640, bx[2] * 640, y2s * 640],
                                        "confidence": p["confidence"],
                                        "class": "person"
                                    })'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Person-BBox Shift fuer Gesicht-Zentrierung - OK')
    fixes += 1
else:
    print('FIX: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
