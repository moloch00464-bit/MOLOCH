#!/usr/bin/env python3
"""Fix: Daily Learner Panel Method."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: Add _toggle_daily_learner after _toggle_autonomous
old = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL via service."""
        if self.service:
            self.service.toggle_autonomous_manual()

    def _set_status_led(self):'''

new = '''    def _toggle_autonomous(self):
        """Toggle AUTONOM/MANUELL via service."""
        if self.service:
            self.service.toggle_autonomous_manual()

    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus)."""
        if not self.service or not hasattr(self.service, '_daily_learner') or not self.service._daily_learner:
            return
        enabled = self.service._daily_learner.toggle()
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")

    def _set_status_led(self):'''

if old in code:
    code = code.replace(old, new)
    print('FIX 1: _toggle_daily_learner method - OK')
    fixes += 1
else:
    print('ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print('Syntax OK.')
