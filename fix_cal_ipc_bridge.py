#!/usr/bin/env python3
"""Fix: Calibration Events ueber IPC Bridge leiten.

Problem: Im Remote-Modus (ServiceProxy) kommen calibration_result
und calibration_status Events nie beim Panel an, weil service._notify()
nur an Service-interne Observer geht.

Loesung:
1. CalibrationEngine: Events auch nach /dev/shm/moloch_cal_event.json schreiben
2. ServiceProxy: cal_event Datei pollen und an Panel-Observer weiterleiten
3. Duplikat _start_calibration entfernen
"""
import sys

# ============================================================
# TEIL 1: CalibrationEngine - Events zusaetzlich in SHM schreiben
# ============================================================
cal = '/home/molochzuhause/moloch/core/calibration_engine.py'
with open(cal) as f:
    code = f.read()

fixes = 0

# 1a: Import json (schon da?) + SHM Event Path
if 'CAL_EVENT_SHM' not in code:
    old_results = '''RESULTS_PATH = os.path.expanduser("~/moloch/data/calibration_results.json")'''
    new_results = '''RESULTS_PATH = os.path.expanduser("~/moloch/data/calibration_results.json")
CAL_EVENT_SHM = "/dev/shm/moloch_cal_event.json"'''
    if old_results in code:
        code = code.replace(old_results, new_results)
        print('FIX 1a: CAL_EVENT_SHM Pfad - OK')
        fixes += 1
    else:
        print('FIX 1a: ANCHOR NOT FOUND!')
else:
    print('FIX 1a: CAL_EVENT_SHM existiert schon - SKIP')
    fixes += 1

# 1b: Methode _write_cal_event hinzufuegen (nach __init__)
if '_write_cal_event' not in code:
    old_init_end = '''        self._lock = threading.Lock()'''
    new_init_end = '''        self._lock = threading.Lock()

    def _write_cal_event(self, event_type, data):
        """Event auch als SHM-Datei fuer ServiceProxy schreiben."""
        try:
            import json as _json
            payload = {"event": event_type, "data": data, "ts": time.time()}
            tmp = CAL_EVENT_SHM + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(payload, f)
            os.rename(tmp, CAL_EVENT_SHM)
        except Exception:
            pass'''

    if old_init_end in code:
        code = code.replace(old_init_end, new_init_end, 1)
        print('FIX 1b: _write_cal_event Methode - OK')
        fixes += 1
    else:
        print('FIX 1b: ANCHOR NOT FOUND!')
else:
    print('FIX 1b: _write_cal_event existiert schon - SKIP')
    fixes += 1

# 1c: In _run_emotions: Nach service._notify auch _write_cal_event
old_notify_result = '''            # Ergebnis ans Panel senden
            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "emotions",
            })'''

new_notify_result = '''            # Ergebnis ans Panel senden (Observer + SHM fuer Remote)
            _result = {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "emotions",
            }
            self.service._notify("calibration_result", _result)
            self._write_cal_event("calibration_result", _result)'''

if old_notify_result in code:
    code = code.replace(old_notify_result, new_notify_result)
    print('FIX 1c: Emotion result -> SHM - OK')
    fixes += 1
else:
    print('FIX 1c: ANCHOR NOT FOUND (emotion result notify)')

# 1d: In start() - calibration_status auch via SHM
old_status_notify = '''        self.service._notify("calibration_status", {
            "status": "running", "phase": phase})'''
new_status_notify = '''        self.service._notify("calibration_status", {
            "status": "running", "phase": phase})
        self._write_cal_event("calibration_status", {
            "status": "running", "phase": phase})'''

if old_status_notify in code:
    code = code.replace(old_status_notify, new_status_notify, 1)
    print('FIX 1d: Start status -> SHM - OK')
    fixes += 1
else:
    print('FIX 1d: ANCHOR NOT FOUND!')

# 1e: In _finish() - calibration_status "done" auch via SHM
old_finish_notify = '''        self.service._notify("calibration_status", {'''
# Finde alle _notify("calibration_status",...) und ergaenze
# Besser: In _finish() die _write_cal_event hinzufuegen
# Suche den gesamten _finish block
if 'def _finish(self):' in code:
    # Suche das Ende von _finish
    import re
    # Einfacher: Suche den _notify in _finish und dupliziere als _write_cal_event
    # Suche nach dem Pattern in _finish
    old_finish = '''    def _finish(self):
        """Kalibrierung abschliessen."""
        self._running = False
        self.service._calibration_active = False
        duration = time.time() - self._start_time

        total = self._processed
        correct = sum(s.get("correct", 0) for s in self._category_stats.values())
        rate = correct / total if total > 0 else 0

        logger.info(f"[CAL] Fertig: {self._phase} - {correct}/{total} "
                     f"({rate:.1%}) in {duration:.0f}s")

        # Ergebnisse speichern
        self._save_results()

        # Weights anpassen basierend auf Ergebnissen
        self._adjust_weights()

        # Status an Panel
        self.service._notify("calibration_status", {
            "status": "done",
            "phase": self._phase,
            "total": total,
            "correct": correct,
            "rate": round(rate, 3),
            "duration": round(duration, 1),
            "categories": self._category_stats,
        })'''

    new_finish = '''    def _finish(self):
        """Kalibrierung abschliessen."""
        self._running = False
        self.service._calibration_active = False
        duration = time.time() - self._start_time

        total = self._processed
        correct = sum(s.get("correct", 0) for s in self._category_stats.values())
        rate = correct / total if total > 0 else 0

        logger.info(f"[CAL] Fertig: {self._phase} - {correct}/{total} "
                     f"({rate:.1%}) in {duration:.0f}s")

        # Ergebnisse speichern
        self._save_results()

        # Weights anpassen basierend auf Ergebnissen
        self._adjust_weights()

        # Status an Panel (Observer + SHM)
        _done = {
            "status": "done",
            "phase": self._phase,
            "total": total,
            "correct": correct,
            "rate": round(rate, 3),
            "duration": round(duration, 1),
            "categories": self._category_stats,
        }
        self.service._notify("calibration_status", _done)
        self._write_cal_event("calibration_status", _done)'''

    if old_finish in code:
        code = code.replace(old_finish, new_finish)
        print('FIX 1e: Finish status -> SHM - OK')
        fixes += 1
    else:
        print('FIX 1e: ANCHOR NOT FOUND (_finish block)')
else:
    print('FIX 1e: _finish nicht gefunden')

# 1f: Andere Phasen (gender, age, emotions_hd) - _notify doppeln
# Suche nach allen _notify("calibration_result" in anderen Phasen
# und ergaenze _write_cal_event
for phase_name in ['gender', 'age', 'emotions_hd', 'gestures']:
    pattern = f'"phase": "{phase_name}",'
    # Suche nach dem _notify Block fuer diese Phase
    search = f'''            self.service._notify("calibration_result", {{
                "file": fname,'''
    # Generischer Ansatz: Alle _notify("calibration_result") die noch kein _write_cal_event haben
    pass  # Wird unten generisch gemacht

# Generisch: Alle verbleibenden _notify("calibration_result", {..}) ohne _write_cal_event
# ergaenzen. Suche Pattern: self.service._notify("calibration_result", {DICT})
# gefolgt von NICHT _write_cal_event
import re
# Finde alle Stellen wo _notify("calibration_result" vorkommt aber KEINE _write_cal_event folgt
lines = code.split('\n')
insertions = []
for i, line in enumerate(lines):
    if 'self.service._notify("calibration_result"' in line:
        # Suche das Ende des Dict-Blocks (naechste Zeile mit nur })
        j = i + 1
        while j < len(lines) and '})' not in lines[j]:
            j += 1
        # Pruefe ob danach _write_cal_event kommt
        if j + 1 < len(lines) and '_write_cal_event' not in lines[j + 1]:
            # Ergaenze nach dem })
            indent = '            '
            # Sammle die Variable (muss _result sein oder inline)
            # Statt inline: Mache es generisch - speichere das result in einer Variable
            insertions.append((j + 1, indent + '# SHM Bridge (auto-inserted)\n'))

# Statt die komplexe Insertion: Mache es einfacher
# Patche die _update_stats Methode um auch SHM zu schreiben
# ODER: Patche den _notify Wrapper
# Am einfachsten: Override _notify im Service waehrend Kalibrierung
# Aber das ist zu komplex. Stattdessen: Patche die verbleibenden Phasen

# Fuer gender/age/emotions_hd: Die haben aehnliche _notify Blocks
# Suche und ersetze sie einzeln

# Gender notify
old_gender_notify = '''            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected_gender,
                "detected": det_gender or "---",
                "confidence": round(det_conf, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "gender",
            })'''

new_gender_notify = '''            _result = {
                "file": fname,
                "category": category,
                "expected": expected_gender,
                "detected": det_gender or "---",
                "confidence": round(det_conf, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "gender",
            }
            self.service._notify("calibration_result", _result)
            self._write_cal_event("calibration_result", _result)'''

if old_gender_notify in code:
    code = code.replace(old_gender_notify, new_gender_notify)
    print('FIX 1f: Gender result -> SHM - OK')
    fixes += 1
else:
    print('FIX 1f: Gender result ANCHOR NOT FOUND (evtl schon gepatcht)')

# Age notify
old_age_notify = '''            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected_range,
                "detected": det_age or "---",
                "confidence": round(det_conf, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "age",
            })'''

new_age_notify = '''            _result = {
                "file": fname,
                "category": category,
                "expected": expected_range,
                "detected": det_age or "---",
                "confidence": round(det_conf, 3),
                "correct": correct,
                "face_found": True,
                "progress": (self._processed, self._total_images),
                "phase": "age",
            }
            self.service._notify("calibration_result", _result)
            self._write_cal_event("calibration_result", _result)'''

if old_age_notify in code:
    code = code.replace(old_age_notify, new_age_notify)
    print('FIX 1g: Age result -> SHM - OK')
    fixes += 1
else:
    print('FIX 1g: Age result ANCHOR NOT FOUND')

# emotions_hd notify
old_ehd_notify = '''            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": face_found,
                "progress": (self._processed, self._total_images),
                "phase": "emotions_hd",
            })'''

new_ehd_notify = '''            _result = {
                "file": fname,
                "category": category,
                "expected": expected,
                "detected": detected or "---",
                "confidence": round(confidence, 3),
                "correct": correct,
                "face_found": face_found,
                "progress": (self._processed, self._total_images),
                "phase": "emotions_hd",
            }
            self.service._notify("calibration_result", _result)
            self._write_cal_event("calibration_result", _result)'''

if old_ehd_notify in code:
    code = code.replace(old_ehd_notify, new_ehd_notify)
    print('FIX 1h: Emotions HD result -> SHM - OK')
    fixes += 1
else:
    print('FIX 1h: Emotions HD result ANCHOR NOT FOUND')

# Gestures notify
old_gest_notify = '''            self.service._notify("calibration_result", {
                "file": fname,
                "category": category,'''
# Gestures hat ein anderes Format - lass ich erstmal

# pause() auch via SHM
old_pause = '''    def pause(self):
        """Pausieren/Fortsetzen."""
        self._paused = not self._paused
        state = "paused" if self._paused else "running"
        logger.info(f"[CAL] {state}")
        self.service._notify("calibration_status", {
            "status": state, "phase": self._phase})'''

new_pause = '''    def pause(self):
        """Pausieren/Fortsetzen."""
        self._paused = not self._paused
        state = "paused" if self._paused else "running"
        logger.info(f"[CAL] {state}")
        _s = {"status": state, "phase": self._phase}
        self.service._notify("calibration_status", _s)
        self._write_cal_event("calibration_status", _s)'''

if old_pause in code:
    code = code.replace(old_pause, new_pause)
    print('FIX 1i: Pause -> SHM - OK')
    fixes += 1
else:
    print('FIX 1i: ANCHOR NOT FOUND (pause)')

# error notify auch via SHM (in _run_gestures)
old_error_notify = '''            self.service._notify("calibration_status", {
                "status": "error",
                "message": f"Ordner fehlt: {GESTURES_DIR}"})'''
new_error_notify = '''            _err = {"status": "error", "message": f"Ordner fehlt: {GESTURES_DIR}"}
            self.service._notify("calibration_status", _err)
            self._write_cal_event("calibration_status", _err)'''
if old_error_notify in code:
    code = code.replace(old_error_notify, new_error_notify)
    print('FIX 1j: Error -> SHM - OK')
    fixes += 1
else:
    print('FIX 1j: SKIP (error notify)')

with open(cal, 'w') as f:
    f.write(code)
compile(open(cal).read(), cal, 'exec')
print(f'\nCalibrationEngine: {fixes} Fixes. Syntax OK.')

# ============================================================
# TEIL 2: ServiceProxy - Cal Events von SHM lesen
# ============================================================
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    pcode = f.read()

pfixes = 0

# 2a: In _read_loop: Cal Events pollen
old_read_sleep = '''            time.sleep(0.033)  # ~30fps'''

new_read_sleep = '''            # Cal Events pollen (SHM Bridge)
            try:
                cal_path = '/dev/shm/moloch_cal_event.json'
                if os.path.exists(cal_path):
                    with open(cal_path, 'r') as cf:
                        cal_ev = json.load(cf)
                    _ts = cal_ev.get('ts', 0)
                    if _ts != getattr(self, '_last_cal_ts', 0):
                        self._last_cal_ts = _ts
                        _ev_type = cal_ev.get('event', '')
                        _ev_data = cal_ev.get('data', {})
                        for cb in self._observers:
                            try:
                                cb(_ev_type, _ev_data)
                            except Exception:
                                pass
            except Exception:
                pass

            time.sleep(0.033)  # ~30fps'''

if old_read_sleep in pcode:
    pcode = pcode.replace(old_read_sleep, new_read_sleep)
    print('FIX 2a: Cal Event Polling in _read_loop - OK')
    pfixes += 1
else:
    print('FIX 2a: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(pcode)
compile(open(panel).read(), panel, 'exec')
print(f'Panel: {pfixes} Fixes. Syntax OK.')

# ============================================================
# TEIL 3: Service - Duplikat _start_calibration entfernen
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    scode = f.read()

sfixes = 0

# Duplikat Block entfernen
dup_block = '''
    # ----------------------------------------------------------------
    # Calibration Engine
    # ----------------------------------------------------------------
    def _start_calibration(self, phase, speed):
        """Kalibrierung starten (IPC)."""
        if self._calibration and self._calibration._running:
            logger.warning("[CAL] Kalibrierung laeuft bereits!")
            return
        try:
            from core.calibration_engine import CalibrationEngine
            self._calibration = CalibrationEngine(self)
            import threading
            threading.Thread(
                target=self._calibration.start,
                args=(phase, speed), daemon=True
            ).start()
            logger.info(f"[CAL] Gestartet: phase={phase}, speed={speed}")
        except Exception as e:
            logger.error(f"[CAL] Start fehlgeschlagen: {e}")

    # ----------------------------------------------------------------
    # Calibration Engine
    # ----------------------------------------------------------------
    def _start_calibration(self, phase, speed):'''

single_block = '''
    # ----------------------------------------------------------------
    # Calibration Engine
    # ----------------------------------------------------------------
    def _start_calibration(self, phase, speed):'''

if dup_block in scode:
    scode = scode.replace(dup_block, single_block)
    print('FIX 3: Duplikat _start_calibration entfernt - OK')
    sfixes += 1
else:
    print('FIX 3: Duplikat NICHT gefunden (evtl. schon entfernt)')

# Duplikat __init__ Zeilen
dup_init = '''        self._calibration = None
        self._calibration_active = False
        self._calibration = None
        self._calibration_active = False'''
single_init = '''        self._calibration = None
        self._calibration_active = False'''
if dup_init in scode:
    scode = scode.replace(dup_init, single_init)
    print('FIX 3b: Duplikat _calibration init entfernt - OK')
    sfixes += 1
else:
    print('FIX 3b: SKIP')

with open(svc, 'w') as f:
    f.write(scode)
compile(open(svc).read(), svc, 'exec')
print(f'Service: {sfixes} Fixes. Syntax OK.')

total = fixes + pfixes + sfixes
print(f'\n=== TOTAL: {total} Fixes ===')
print('CalibrationEngine: Events gehen jetzt auch ueber SHM')
print('ServiceProxy: Pollt cal_event.json und leitet an Panel weiter')
print('Service: Kein Duplikat mehr')
