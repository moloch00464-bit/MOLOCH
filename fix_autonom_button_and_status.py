#!/usr/bin/env python3
"""Fix: AUTONOM/MANUELL Button + Status-Anzeige

BUG: AUTONOM/MANUELL Button updated nicht
- manual_mode fehlt im Status-JSON
- Button-Update fehlt im Panel

FIX:
1. Service: manual_mode in Status-JSON hinzufuegen
2. Panel: AUTONOM Button Update in _update_fps() (wie ST Button)
3. Panel: Bessere Status-Anzeige (Modus-Label mit Farben)
"""

import sys

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'

fixes = 0

# ========== FIX 1: manual_mode in Status-JSON ==========

with open(svc) as f:
    svc_code = f.read()

# Nach autonomous_mode hinzufuegen
old_status = '''                "autonomous_mode": self._autonomous_mode,
                "moloch_has_control": self._moloch_has_control,'''

new_status = '''                "autonomous_mode": self._autonomous_mode,
                "manual_mode": self._manual_mode,
                "moloch_has_control": self._moloch_has_control,'''

if old_status in svc_code:
    svc_code = svc_code.replace(old_status, new_status)
    print('FIX 1: manual_mode in Status-JSON - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(svc_code)

# ========== FIX 2: AUTONOM Button Update im Panel ==========

with open(panel) as f:
    panel_code = f.read()

# Nach ST Button Update hinzufuegen (in _update_fps)
old_update = '''        # Update ST Button from service state
        if hasattr(self.service, '_smart_tracking_on'):
            st_on = self.service._smart_tracking_on
            self.st_btn.config(
                text=f"ST:{"AN" if st_on else "AUS"}",
                bg="#884400" if st_on else "#2a2a4e"
            )

        self.after(66, self._update_fps)'''

new_update = '''        # Update ST Button from service state
        if hasattr(self.service, '_smart_tracking_on'):
            st_on = self.service._smart_tracking_on
            self.st_btn.config(
                text=f"ST:{"AN" if st_on else "AUS"}",
                bg="#884400" if st_on else "#2a2a4e"
            )

        # Update AUTONOM Button from service state
        if hasattr(self.service, '_manual_mode'):
            manual = self.service._manual_mode
            self.auto_btn.config(
                text="MANUELL" if manual else "AUTONOM",
                bg="#228822" if manual else "#aa2222"
            )

        self.after(66, self._update_fps)'''

if old_update in panel_code:
    panel_code = panel_code.replace(old_update, new_update)
    print('FIX 2: AUTONOM Button Update - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# ========== FIX 3: Status-Label fuer Modus-Anzeige ==========

# Nach self._manual_mode = False initialisieren
old_init = '''        # Smart Tracking (remote state)
        self._smart_tracking_on = False

        # Cloud bridge (eigene Instanz fuer LED/IR/Alarm)'''

new_init = '''        # Smart Tracking (remote state)
        self._smart_tracking_on = False

        # Status label for mode display
        self._mode_label = None

        # Cloud bridge (eigene Instanz fuer LED/IR/Alarm)'''

if old_init in panel_code:
    panel_code = panel_code.replace(old_init, new_init)
    print('FIX 3a: _mode_label initialisiert - OK')
    fixes += 1
else:
    print('FIX 3a: ANCHOR NOT FOUND!')

# Status-Label in GUI erstellen (nach FPS label)
old_fps_label = '''        self.fps_label = tk.Label(status_bar, text="FPS: -- | Kamera: -- | scrfd: --",
                                  bg="#1a1a1a", fg="#00ff00", font=("Consolas", 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)'''

new_fps_label = '''        self.fps_label = tk.Label(status_bar, text="FPS: -- | Kamera: -- | scrfd: --",
                                  bg="#1a1a1a", fg="#00ff00", font=("Consolas", 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)

        # Mode status label
        self._mode_label = tk.Label(status_bar, text="MODUS: --",
                                     bg="#1a1a1a", fg="#ffffff", font=("Consolas", 10, "bold"))
        self._mode_label.pack(side=tk.LEFT, padx=10)'''

if old_fps_label in panel_code:
    panel_code = panel_code.replace(old_fps_label, new_fps_label)
    print('FIX 3b: Status-Label GUI erstellt - OK')
    fixes += 1
else:
    print('FIX 3b: ANCHOR NOT FOUND!')

# Modus-Label Update in _update_fps (nach AUTONOM button update)
old_after = '''        # Update AUTONOM Button from service state
        if hasattr(self.service, '_manual_mode'):
            manual = self.service._manual_mode
            self.auto_btn.config(
                text="MANUELL" if manual else "AUTONOM",
                bg="#228822" if manual else "#aa2222"
            )

        self.after(66, self._update_fps)'''

new_after = '''        # Update AUTONOM Button from service state
        if hasattr(self.service, '_manual_mode'):
            manual = self.service._manual_mode
            self.auto_btn.config(
                text="MANUELL" if manual else "AUTONOM",
                bg="#228822" if manual else "#aa2222"
            )

        # Update Mode Label (Tentakel/MOLOCH/Manuell)
        if self._mode_label and hasattr(self.service, '_smart_tracking_on'):
            st_on = getattr(self.service, '_smart_tracking_on', False)
            manual = getattr(self.service, '_manual_mode', False)
            moloch_ctrl = getattr(self.service, '_moloch_has_control', False)

            if manual:
                mode_text = "MODUS: MANUELL"
                mode_color = "#00ff00"  # Gruen
            elif moloch_ctrl:
                mode_text = "MODUS: MOLOCH AKTIV"
                mode_color = "#ff4444"  # Rot
            elif st_on:
                mode_text = "MODUS: TENTAKEL SCANNT"
                mode_color = "#00ffff"  # Cyan
            else:
                mode_text = "MODUS: BEREIT"
                mode_color = "#ffaa00"  # Orange

            self._mode_label.config(text=mode_text, fg=mode_color)

        self.after(66, self._update_fps)'''

if old_after in panel_code:
    panel_code = panel_code.replace(old_after, new_after)
    print('FIX 3c: Modus-Label Update - OK')
    fixes += 1
else:
    print('FIX 3c: ANCHOR NOT FOUND!')

# ========== FIX 4: _manual_mode aus Status lesen ==========

# In _apply_status nach _smart_tracking_on
old_apply = '''        self._smart_tracking_on = s.get('smart_tracking_on', False)
        self._daily_learner_enabled = s.get('daily_learner_enabled', False)'''

new_apply = '''        self._smart_tracking_on = s.get('smart_tracking_on', False)
        self._manual_mode = s.get('manual_mode', False)
        self._moloch_has_control = s.get('moloch_has_control', False)
        self._daily_learner_enabled = s.get('daily_learner_enabled', False)'''

if old_apply in panel_code:
    panel_code = panel_code.replace(old_apply, new_apply)
    print('FIX 4: Status-Reading fuer manual_mode - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(panel_code)

# Syntax Check
try:
    compile(open(svc).read(), svc, 'exec')
    print(f'\n{fixes}/6 Fixes. Service Syntax OK.')
except SyntaxError as e:
    print(f'\nSERVICE SYNTAX ERROR: {e}')
    sys.exit(1)

try:
    compile(open(panel).read(), panel, 'exec')
    print('Panel Syntax OK.')
except SyntaxError as e:
    print(f'PANEL SYNTAX ERROR: {e}')
    sys.exit(1)

if fixes == 6:
    print('\n=== ALLE FIXES ANGEWENDET ===')
    print('\nJETZT:')
    print('1. sudo systemctl restart moloch.service')
    print('2. Panel neu starten')
    print('3. AUTONOM Button testen -> Farbe togglet (Gruen/Rot)?')
    print('4. Status-Label zeigt: TENTAKEL/MOLOCH/MANUELL/BEREIT?')
else:
    print(f'\nWARNING: Nur {fixes}/6 Fixes angewendet!')
