#!/usr/bin/env python3
"""Fix: Calibration Frames ueber /dev/shm ans Panel schicken.

1. Service: _calibration_active Flag, Inference Loop checkt es
2. CalibrationEngine: _write_shm() direkt aufrufen + Flag setzen
"""
import sys

# ============================================================
# TEIL 1: moloch_service.py - _calibration_active Flag
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1A: _calibration_active Attribut in __init__ (neben _calibration)
old_cal_init = """        # Calibration Engine
        self._calibration = None"""

new_cal_init = """        # Calibration Engine
        self._calibration = None
        self._calibration_active = False"""

if old_cal_init in code:
    code = code.replace(old_cal_init, new_cal_init)
    print('FIX 1A: _calibration_active Attribut - OK')
    fixes += 1
else:
    print('FIX 1A: ANCHOR NOT FOUND!')

# FIX 1B: Inference Loop - Skip Frame Write wenn Calibration aktiv
# Im "kein Modell aktiv" Pfad
old_no_model = """                with self._annotated_lock:
                    self._annotated_frame = frame.copy()
                self._write_shm(frame)
                time.sleep(0.03)
                continue"""

new_no_model = """                if not self._calibration_active:
                    with self._annotated_lock:
                        self._annotated_frame = frame.copy()
                    self._write_shm(frame)
                time.sleep(0.03)
                continue"""

if old_no_model in code:
    code = code.replace(old_no_model, new_no_model)
    print('FIX 1B: Skip frame write (no model) - OK')
    fixes += 1
else:
    print('FIX 1B: ANCHOR NOT FOUND!')

# FIX 1C: Inference Loop - Skip Frame Write nach Inference
old_post_inference = """            with self._annotated_lock:
                self._annotated_frame = annotated

            # Panel IPC: Frame + Status nach /dev/shm
            self._write_shm(annotated)"""

new_post_inference = """            if not self._calibration_active:
                with self._annotated_lock:
                    self._annotated_frame = annotated
                # Panel IPC: Frame + Status nach /dev/shm
                self._write_shm(annotated)"""

if old_post_inference in code:
    code = code.replace(old_post_inference, new_post_inference)
    print('FIX 1C: Skip frame write (post inference) - OK')
    fixes += 1
else:
    print('FIX 1C: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/3 Fixes.')

# ============================================================
# TEIL 2: calibration_engine.py - _write_shm() direkt nutzen
# ============================================================
cal = '/home/molochzuhause/moloch/core/calibration_engine.py'
with open(cal) as f:
    code2 = f.read()

fixes2 = 0

# FIX 2A: start() setzt _calibration_active = True
old_start = '''        logger.info(f"[CAL] Start: phase={phase}, speed={speed}")
        self.service._notify("calibration_status", {
            "status": "running", "phase": phase})'''

new_start = '''        logger.info(f"[CAL] Start: phase={phase}, speed={speed}")
        self.service._calibration_active = True
        self.service._notify("calibration_status", {
            "status": "running", "phase": phase})'''

if old_start in code2:
    code2 = code2.replace(old_start, new_start)
    print('FIX 2A: _calibration_active = True bei Start - OK')
    fixes2 += 1
else:
    print('FIX 2A: ANCHOR NOT FOUND!')

# FIX 2B: _finish() setzt _calibration_active = False
old_finish = '''    def _finish(self):
        """Kalibrierung abschliessen, Ergebnisse speichern."""
        self._running = False'''

new_finish = '''    def _finish(self):
        """Kalibrierung abschliessen, Ergebnisse speichern."""
        self._running = False
        self.service._calibration_active = False'''

if old_finish in code2:
    code2 = code2.replace(old_finish, new_finish)
    print('FIX 2B: _calibration_active = False bei Finish - OK')
    fixes2 += 1
else:
    print('FIX 2B: ANCHOR NOT FOUND!')

# FIX 2C: Emotions - Frame via _write_shm statt nur _annotated_frame
old_emo_frame = '''            # Frame injizieren
            with self.service._frame_lock:
                self.service._annotated_frame = annotated'''

new_emo_frame = '''            # Frame via shm ans Panel senden
            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)'''

if old_emo_frame in code2:
    code2 = code2.replace(old_emo_frame, new_emo_frame, 1)
    print('FIX 2C: Emotions _write_shm - OK')
    fixes2 += 1
else:
    print('FIX 2C: ANCHOR NOT FOUND (emotions)!')

# FIX 2D: Gesten - Frame via _write_shm statt nur _annotated_frame
old_gest_frame = '''            # Frame injizieren
            with self.service._frame_lock:
                self.service._annotated_frame = annotated'''

new_gest_frame = '''            # Frame via shm ans Panel senden
            with self.service._annotated_lock:
                self.service._annotated_frame = annotated
            self.service._write_shm(annotated)'''

if old_gest_frame in code2:
    code2 = code2.replace(old_gest_frame, new_gest_frame, 1)
    print('FIX 2D: Gesten _write_shm - OK')
    fixes2 += 1
else:
    print('FIX 2D: ANCHOR NOT FOUND (gesten)!')

with open(cal, 'w') as f:
    f.write(code2)

print(f'\nCalibration Engine: {fixes2}/4 Fixes.')

# Syntax check
total = fixes + fixes2
if total < 7:
    print(f'\n!!! INCOMPLETE: {total}/7 Fixes !!!')
    sys.exit(1)

compile(open(svc).read(), svc, 'exec')
compile(open(cal).read(), cal, 'exec')
print('\nSyntax OK (beide Dateien)')
print('\n=== CALIBRATION FRAME FIX KOMPLETT ===')
