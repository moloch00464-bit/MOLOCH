#!/usr/bin/env python3
"""Fix: Daily Learner Button im Panel.

Fügt "ALLTAG" Button hinzu neben CAL Button.
"""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: ALLTAG Button nach CAL Button
old_buttons = '''        tk.Button(brow, text="CAL", bg="#ff8800", fg="white", width=5,
                  font=("Helvetica", 11, "bold"),
                  command=self._trigger_calibration).pack(side=tk.LEFT, padx=1)

        # --- eWeLink controls (right) ---'''

new_buttons = '''        tk.Button(brow, text="CAL", bg="#ff8800", fg="white", width=5,
                  font=("Helvetica", 11, "bold"),
                  command=self._trigger_calibration).pack(side=tk.LEFT, padx=1)
        self.daily_btn = tk.Button(brow, text="ALLTAG", bg="#1a1a3e", fg="white", width=7,
                                   font=("Helvetica", 11, "bold"),
                                   command=self._toggle_daily_learner)
        self.daily_btn.pack(side=tk.LEFT, padx=1)

        # --- eWeLink controls (right) ---'''

if old_buttons in code:
    code = code.replace(old_buttons, new_buttons)
    print('FIX 1: ALLTAG Button hinzugefügt - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: _toggle_daily_learner Methode am Ende der Button-Callbacks
# Finde einen guten Platz - nach _toggle_autonomous
old_toggle = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL via service."""
        if not self.service:
            return
        self.service.toggle_autonomous()'''

new_toggle = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL via service."""
        if not self.service:
            return
        self.service.toggle_autonomous()

    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus)."""
        if not self.service or not self.service._daily_learner:
            return
        enabled = self.service._daily_learner.toggle()
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

if old_toggle in code:
    code = code.replace(old_toggle, new_toggle)
    print('FIX 2: _toggle_daily_learner Methode - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: Status-Update in _update_fps (optional - zeigt Status im Button)
# Nach FPS-Update, check daily learner status
old_fps = '''        # FPS + Checkbox Sync
        if self.service:
            fps = getattr(self.service, "_fps", {})'''

new_fps = '''        # FPS + Checkbox Sync
        if self.service:
            fps = getattr(self.service, "_fps", {})

            # Daily Learner Status
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                dl = self.service._daily_learner
                if dl.enabled:
                    self.daily_btn.config(bg="#006622", text="ALLTAG AN")
                else:
                    self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

if old_fps in code:
    code = code.replace(old_fps, new_fps)
    print('FIX 3: Status-Update in _update_fps - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
