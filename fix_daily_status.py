#!/usr/bin/env python3
"""Fix: Daily Learner Status Update."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

# Add status update before root.after
old = '''        except Exception:
            pass

        self.root.after(500, self._update_fps)'''

new = '''        except Exception:
            pass

        # Daily Learner Status
        try:
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                dl = self.service._daily_learner
                if dl.enabled:
                    self.daily_btn.config(bg="#006622", text="ALLTAG AN")
                else:
                    self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")
        except Exception:
            pass

        self.root.after(500, self._update_fps)'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Daily Learner Status Update - OK')
    with open(panel, 'w') as f:
        f.write(code)
    compile(open(panel).read(), panel, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)
