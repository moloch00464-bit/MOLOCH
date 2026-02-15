#!/usr/bin/env python3
"""Fix: Hand-Crop in Fingerrichtung verschieben + groesserer Crop.

Problem: Crop zentriert auf Wrist -> Finger ausserhalb.
Fix: Elbow->Wrist Vektor berechnen, Crop-Zentrum 50% in Fingerrichtung verschieben.
     Crop-Max von 200 auf 300 erhoehen.
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old_crop = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:
                                _kpts = _pose["keypoints"]  # (17, 3) in 640-Space
                                for _wi in (9, 10):  # left/right wrist
                                    _wx = _kpts[_wi, 0]
                                    _wy = _kpts[_wi, 1]
                                    _wvis = _kpts[_wi, 2]
                                    if _wvis < 0.3:
                                        continue
                                    # Crop-Groesse: ~25% des Pose-BBox oder fix 140px
                                    _pbx = _pose["bbox"]
                                    _pw = _pbx[2] - _pbx[0]
                                    _ph = _pbx[3] - _pbx[1]
                                    _csz = max(int(max(_pw, _ph) * 0.4), 100)
                                    _csz = min(_csz, 200)
                                    # Crop-Region (640x640 Space)
                                    _cx1 = max(0, int(_wx - _csz // 2))
                                    _cy1 = max(0, int(_wy - _csz // 2))
                                    _cx2 = min(640, _cx1 + _csz)
                                    _cy2 = min(640, _cy1 + _csz)"""

new_crop = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:
                                _kpts = _pose["keypoints"]  # (17, 3) in 640-Space
                                for _wi in (9, 10):  # left/right wrist
                                    _wx = _kpts[_wi, 0]
                                    _wy = _kpts[_wi, 1]
                                    _wvis = _kpts[_wi, 2]
                                    if _wvis < 0.3:
                                        continue

                                    # Elbow-Index: wrist 9->elbow 7, wrist 10->elbow 8
                                    _ei = _wi - 2
                                    _ex = _kpts[_ei, 0]
                                    _ey = _kpts[_ei, 1]
                                    _evis = _kpts[_ei, 2]

                                    # Crop-Groesse: groesser fuer gespreizte Finger
                                    _pbx = _pose["bbox"]
                                    _pw = _pbx[2] - _pbx[0]
                                    _ph = _pbx[3] - _pbx[1]
                                    _csz = max(int(max(_pw, _ph) * 0.5), 140)
                                    _csz = min(_csz, 300)

                                    # Crop-Zentrum: Wrist + 50% Offset in Fingerrichtung
                                    _ccx = _wx
                                    _ccy = _wy
                                    if _evis > 0.2:
                                        _dx = _wx - _ex
                                        _dy = _wy - _ey
                                        _dist = max((_dx**2 + _dy**2)**0.5, 1.0)
                                        # Normalisiert * 50% der Crop-Groesse
                                        _off = _csz * 0.45
                                        _ccx = _wx + (_dx / _dist) * _off
                                        _ccy = _wy + (_dy / _dist) * _off

                                    # Crop-Region (640x640 Space)
                                    _cx1 = max(0, int(_ccx - _csz // 2))
                                    _cy1 = max(0, int(_ccy - _csz // 2))
                                    _cx2 = min(640, _cx1 + _csz)
                                    _cy2 = min(640, _cy1 + _csz)"""

if old_crop in code:
    code = code.replace(old_crop, new_crop)
    with open(svc, 'w') as f:
        f.write(code)
    print('Hand-Crop Offset + Groesse Fix - OK')
else:
    print('ANCHOR NOT FOUND!')
    sys.exit(1)
