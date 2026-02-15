#!/usr/bin/env python3
"""Finetuning: Hand-Crop kleiner + Auto-Switch zurueck.

Teil A: Crop 300->220, Offset 45%->25%, Kopf nicht abschneiden
Teil B: 5s keine Hand erkannt -> forced_models loeschen -> Auto-Scoring
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# === TEIL A: Crop-Skalierung reduzieren ===
old_crop = """                                    # Crop-Groesse: groesser fuer gespreizte Finger
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
                                        _ccy = _wy + (_dy / _dist) * _off"""

new_crop = """                                    # Crop-Groesse: kompakt um Hand (nicht zu gross -> Kopf!)
                                    _pbx = _pose["bbox"]
                                    _pw = _pbx[2] - _pbx[0]
                                    _ph = _pbx[3] - _pbx[1]
                                    _csz = max(int(max(_pw, _ph) * 0.35), 120)
                                    _csz = min(_csz, 220)

                                    # Crop-Zentrum: Wrist + 25% Offset in Fingerrichtung
                                    _ccx = _wx
                                    _ccy = _wy
                                    if _evis > 0.2:
                                        _dx = _wx - _ex
                                        _dy = _wy - _ey
                                        _dist = max((_dx**2 + _dy**2)**0.5, 1.0)
                                        _off = _csz * 0.25
                                        _ccx = _wx + (_dx / _dist) * _off
                                        _ccy = _wy + (_dy / _dist) * _off"""

if old_crop in code:
    code = code.replace(old_crop, new_crop)
    print('TEIL A: Crop 220px, Offset 25% - OK')
    fixes += 1
else:
    print('TEIL A: ANCHOR NOT FOUND!')

# === TEIL B1: _hand_no_detect Counter init ===
old_init = """        # Watchdog: Anti-Oszillation Swap-Log
        self._swap_log = []"""

new_init = """        # Watchdog: Anti-Oszillation Swap-Log
        self._swap_log = []
        # Auto-Switch: Zaehlt Frames ohne Hand-Detection
        self._hand_no_detect = 0
        self._HAND_RELEASE_FRAMES = 75  # ~5s bei 15fps"""

if old_init in code:
    code = code.replace(old_init, new_init)
    print('TEIL B1: _hand_no_detect Counter init - OK')
    fixes += 1
else:
    print('TEIL B1: ANCHOR NOT FOUND!')

# === TEIL B2: Counter in Hand-Inference-Block ===
# Nach dem Hand-Block: wenn Hand erkannt -> Counter reset, sonst increment
old_after_hand = """                        # Gesten-Erkennung aus Pose-Keypoints"""

new_after_hand = """                        # Auto-Switch: Hand erkannt? Counter updaten
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

if old_after_hand in code:
    code = code.replace(old_after_hand, new_after_hand, 1)
    print('TEIL B2: Auto-Switch Counter nach Hand-Block - OK')
    fixes += 1
else:
    print('TEIL B2: ANCHOR NOT FOUND!')

# === TEIL B3: _last_hand_detected Flag im Hand-Block setzen ===
old_hand_entry = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            for _pose in poses[:1]:"""

new_hand_entry = """                        if self.hand_active and "hand_landmark" in self._active_ctx:
                            self._last_hand_detected = False
                            for _pose in poses[:1]:"""

if old_hand_entry in code:
    code = code.replace(old_hand_entry, new_hand_entry)
    print('TEIL B3: _last_hand_detected init - OK')
    fixes += 1
else:
    print('TEIL B3: ANCHOR NOT FOUND!')

# === TEIL B4: _last_hand_detected auf True wenn decode erfolgreich ===
old_draw = """                                    if _hand_res and "hand" in _allowed_draws:
                                        draw_hand_landmarks("""

new_draw = """                                    if _hand_res:
                                        self._last_hand_detected = True
                                    if _hand_res and "hand" in _allowed_draws:
                                        draw_hand_landmarks("""

if old_draw in code:
    code = code.replace(old_draw, new_draw)
    print('TEIL B4: _last_hand_detected = True bei Erkennung - OK')
    fixes += 1
else:
    print('TEIL B4: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\n{fixes}/5 Fixes angewendet.')
if fixes < 5:
    sys.exit(1)
