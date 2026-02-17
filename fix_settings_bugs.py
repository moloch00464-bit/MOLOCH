#!/usr/bin/env python3
"""Fix: Settings Bugs

BUG 1: Audio/Camera Settings werden NICHT im Status-JSON mitgesendet
       -> Panel kann gespeicherte Werte beim Neustart nicht laden

BUG 2: ALLTAG Button Race Condition
       -> Optimistic Update wird sofort vom Status-Sync ueberschrieben

BUG 3: Camera Settings fehlen auch im Status-JSON

FIXES:
1. Service: Audio + Camera Settings in Status-JSON hinzufuegen
2. Panel: Debounce-Flag fuer ALLTAG Button (500ms keine Sync-Ueberschreibung)
"""

import os

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'

fixes = 0

# ========== FIX 1: Audio/Camera Settings in Status-JSON ==========

with open(svc) as f:
    svc_code = f.read()

# Audio + Camera nach thresholds hinzufuegen (vor perception)
old_status = '''                "thresholds": {
                    "scrfd_conf": self.scrfd_conf_val,
                    "scrfd_nms": self.scrfd_nms_val,
                    "arcface_thresh": self.arcface_thresh_val,
                    "yolo_conf": self.yolo_conf_val,
                    "pose_conf": self.pose_conf_val,
                    "pose_nms": self.pose_nms_val,
                },
            }
            if self._perception:'''

new_status = '''                "thresholds": {
                    "scrfd_conf": self.scrfd_conf_val,
                    "scrfd_nms": self.scrfd_nms_val,
                    "arcface_thresh": self.arcface_thresh_val,
                    "yolo_conf": self.yolo_conf_val,
                    "pose_conf": self.pose_conf_val,
                    "pose_nms": self.pose_nms_val,
                },
                "audio": {
                    "mic_gain": getattr(self, '_saved_mic_gain', 1.0),
                    "agc_enabled": getattr(self, '_saved_agc', False),
                    "noise_gate_db": getattr(self, '_saved_noise_gate', -60.0),
                },
                "camera": {
                    "ptz_speed": getattr(self, '_saved_ptz_speed', 25.0),
                    "led_enabled": getattr(self, '_saved_led', False),
                    "ir_mode": getattr(self, '_saved_ir', "Aus"),
                },
            }
            if self._perception:'''

if old_status in svc_code:
    svc_code = svc_code.replace(old_status, new_status)
    print('FIX 1: Audio + Camera Settings in Status-JSON - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(svc_code)

# ========== FIX 2: ALLTAG Button Debounce ==========

with open(panel) as f:
    panel_code = f.read()

# Init debounce flag nach _daily_learner_enabled
old_init = '''        # Daily Learner (remote state)
        self._daily_learner_enabled = False

        # Cloud bridge (eigene Instanz fuer LED/IR/Alarm)'''

new_init = '''        # Daily Learner (remote state)
        self._daily_learner_enabled = False
        self._daily_btn_debounce_until = 0  # Timestamp bis wann Status-Sync geblockt ist

        # Cloud bridge (eigene Instanz fuer LED/IR/Alarm)'''

if old_init in panel_code:
    panel_code = panel_code.replace(old_init, new_init)
    print('FIX 2a: Debounce Flag initialisiert - OK')
    fixes += 1
else:
    print('FIX 2a: ANCHOR NOT FOUND!')

# _apply_status: Skip Button-Update wenn Debounce aktiv
old_apply = '''        self._daily_learner_enabled = s.get('daily_learner_enabled', False)

        # Update Daily Learner Button
        if hasattr(self, "daily_btn"):
            if self._daily_learner_enabled:
                self.daily_btn.config(bg="#006622", text="ALLTAG AN")
            else:
                self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

new_apply = '''        self._daily_learner_enabled = s.get('daily_learner_enabled', False)

        # Update Daily Learner Button (nur wenn nicht debounced)
        if hasattr(self, "daily_btn"):
            import time
            now = time.time()
            if now >= self._daily_btn_debounce_until:
                if self._daily_learner_enabled:
                    self.daily_btn.config(bg="#006622", text="ALLTAG AN")
                else:
                    self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

if old_apply in panel_code:
    panel_code = panel_code.replace(old_apply, new_apply)
    print('FIX 2b: Debounce Check in _apply_status - OK')
    fixes += 1
else:
    print('FIX 2b: ANCHOR NOT FOUND!')

# _toggle_daily_learner: Setze Debounce Flag
old_toggle = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus) via IPC."""
        if not self.service:
            return

        # Send IPC command
        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_daily_learner"})
            # Optimistically update button (status update will correct if needed)
            enabled = not getattr(self.service, '_daily_learner_enabled', False)
            self.service._daily_learner_enabled = enabled
        else:
            # Direct mode
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                enabled = self.service._daily_learner.toggle()
            else:
                return

        # Update button
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

new_toggle = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus) via IPC."""
        import time
        if not self.service:
            return

        # Send IPC command
        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_daily_learner"})
            # Optimistically update button + debounce
            enabled = not getattr(self.service, '_daily_learner_enabled', False)
            self.service._daily_learner_enabled = enabled
            self._daily_btn_debounce_until = time.time() + 0.5  # 500ms debounce
        else:
            # Direct mode
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                enabled = self.service._daily_learner.toggle()
                self._daily_btn_debounce_until = time.time() + 0.5
            else:
                return

        # Update button
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

if old_toggle in panel_code:
    panel_code = panel_code.replace(old_toggle, new_toggle)
    print('FIX 2c: Debounce in _toggle_daily_learner - OK')
    fixes += 1
else:
    print('FIX 2c: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(panel_code)

# Syntax Check
try:
    compile(open(svc).read(), svc, 'exec')
    print(f'\n{fixes}/4 Fixes. Service Syntax OK.')
except SyntaxError as e:
    print(f'\nSERVICE SYNTAX ERROR: {e}')
    import subprocess
    subprocess.run(['git', 'checkout', svc], cwd='/home/molochzuhause/moloch')

try:
    compile(open(panel).read(), panel, 'exec')
    print('Panel Syntax OK.')
except SyntaxError as e:
    print(f'PANEL SYNTAX ERROR: {e}')
    import subprocess
    subprocess.run(['git', 'checkout', panel], cwd='/home/molochzuhause/moloch')

if fixes == 4:
    print('\n=== ALLE BUGS GEFIXT ===')
    print('\nJETZT:')
    print('1. sudo systemctl restart moloch.service')
    print('2. Panel neu starten')
    print('3. Audio-Settings aendern -> SAVE SETTINGS -> Service Neustart -> Settings noch da?')
    print('4. ALLTAG druecken -> AN. Nochmal druecken -> AUS. Bleibt stabil?')
