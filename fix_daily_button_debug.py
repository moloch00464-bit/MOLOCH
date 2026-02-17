#!/usr/bin/env python3
"""Fix: Daily Learner Button - Add Logging + Force Update."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

# Add logging to toggle method
old = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus)."""
        if not self.service or not hasattr(self.service, '_daily_learner') or not self.service._daily_learner:
            return
        enabled = self.service._daily_learner.toggle()
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")'''

new = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus)."""
        print("[DEBUG] Toggle daily learner called")
        if not self.service:
            print("[DEBUG] No service")
            return
        if not hasattr(self.service, '_daily_learner'):
            print("[DEBUG] Service has no _daily_learner attribute")
            return
        if not self.service._daily_learner:
            print("[DEBUG] _daily_learner is None")
            return
        
        enabled = self.service._daily_learner.toggle()
        print(f"[DEBUG] Daily learner toggled to: {enabled}")
        
        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
            print("[DEBUG] Button set to green")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")
            print("[DEBUG] Button set to gray")
        
        # Force immediate visual update
        self.daily_btn.update()'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Added debug logging to toggle - OK')
    with open(panel, 'w') as f:
        f.write(code)
    compile(open(panel).read(), panel, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
