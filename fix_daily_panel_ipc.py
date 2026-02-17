#!/usr/bin/env python3
"""Fix: Panel nutzt IPC für Daily Learner Toggle."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: ServiceProxy liest daily_learner_enabled
old_proxy = '''        # Autonomy
        self._autonomous_mode = False
        self._moloch_has_control = False
        self._tentakel_enabled = False'''

new_proxy = '''        # Autonomy
        self._autonomous_mode = False
        self._moloch_has_control = False
        self._tentakel_enabled = False
        
        # Daily Learner (remote state)
        self._daily_learner_enabled = False'''

if old_proxy in code:
    code = code.replace(old_proxy, new_proxy)
    print('FIX 1: ServiceProxy daily_learner_enabled - OK')
    fixes += 1

# FIX 2: ServiceProxy liest Status aus JSON
old_read = '''                self._moloch_has_control = status.get('moloch_has_control', False)
                self._tentakel_enabled = status.get('tentakel_enabled', False)'''

new_read = '''                self._moloch_has_control = status.get('moloch_has_control', False)
                self._tentakel_enabled = status.get('tentakel_enabled', False)
                self._daily_learner_enabled = status.get('daily_learner_enabled', False)'''

if old_read in code:
    code = code.replace(old_read, new_read)
    print('FIX 2: ServiceProxy read daily_learner from JSON - OK')
    fixes += 1

# FIX 3: Panel toggle sendet IPC Command
old_toggle = '''    def _toggle_daily_learner(self):
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

new_toggle = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner (Alltag-Modus) via IPC."""
        if not self.service:
            return
        
        # Send IPC command
        if isinstance(self.service, ServiceProxy):
            self.service.send_command({"action": "toggle_daily_learner"})
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

if old_toggle in code:
    code = code.replace(old_toggle, new_toggle)
    print('FIX 3: Panel toggle via IPC - OK')
    fixes += 1

# FIX 4: Status update liest von service attribute (works für beide Modi)
old_status = '''        # Daily Learner Status
        try:
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                dl = self.service._daily_learner
                if dl.enabled:
                    self.daily_btn.config(bg="#006622", text="ALLTAG AN")
                else:
                    self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")
        except Exception:
            pass'''

new_status = '''        # Daily Learner Status
        try:
            # Works für both direct and remote mode
            if hasattr(self.service, '_daily_learner_enabled'):
                enabled = self.service._daily_learner_enabled
            elif hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                enabled = self.service._daily_learner.enabled
            else:
                enabled = False
            
            if enabled:
                self.daily_btn.config(bg="#006622", text="ALLTAG AN")
            else:
                self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")
        except Exception:
            pass'''

if old_status in code:
    code = code.replace(old_status, new_status)
    print('FIX 4: Status update unified - OK')
    fixes += 1

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
