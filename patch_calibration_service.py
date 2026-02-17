#!/usr/bin/env python3
"""Calibration Engine: Service-Integration.

1. _calibration Attribut in __init__
2. IPC Handler (calibration_start/pause/stop)
3. _start_calibration() Methode
"""
import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# --- FIX 1: _calibration Attribut in __init__ ---
old_perception_init = """        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None"""

new_perception_init = """        # Calibration Engine
        self._calibration = None

        # Perception Engine (NPU Slot-Rotation mit Personality)
        self._perception = None"""

if old_perception_init in code:
    code = code.replace(old_perception_init, new_perception_init)
    print('FIX 1: _calibration Attribut - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# --- FIX 2: IPC Handler ---
# Finde den letzten elif in _execute_panel_cmd fuer hand_occlusion
old_ipc_end = """        elif action == 'save_settings':
            # Audio + Camera Werte aus Panel uebernehmen"""

new_ipc_end = """        elif action == 'calibration_start':
            phase = cmd.get('phase', 'emotions')
            speed = int(cmd.get('speed', 3))
            self._start_calibration(phase, speed)
        elif action == 'calibration_pause':
            if self._calibration:
                self._calibration.pause()
        elif action == 'calibration_stop':
            if self._calibration:
                self._calibration.stop()
        elif action == 'save_settings':
            # Audio + Camera Werte aus Panel uebernehmen"""

if old_ipc_end in code:
    code = code.replace(old_ipc_end, new_ipc_end)
    print('FIX 2: IPC Handler - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# --- FIX 3: _start_calibration() Methode vor _load_perception_config ---
old_perc_config = """    # ----------------------------------------------------------------
    # Perception Config (config/perception.json)
    # ----------------------------------------------------------------
    def _load_perception_config(self):"""

new_perc_config = """    # ----------------------------------------------------------------
    # Calibration Engine
    # ----------------------------------------------------------------
    def _start_calibration(self, phase, speed):
        \"\"\"Kalibrierung starten (IPC).\"\"\"
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
    # Perception Config (config/perception.json)
    # ----------------------------------------------------------------
    def _load_perception_config(self):"""

if old_perc_config in code:
    code = code.replace(old_perc_config, new_perc_config)
    print('FIX 3: _start_calibration() Methode - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {fixes}/3 Fixes.')
if fixes < 3:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

print('\n=== CALIBRATION SERVICE KOMPLETT ===')
