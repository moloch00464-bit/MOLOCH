#!/usr/bin/env python3
# Add _toggle_daily_learner method

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    lines = f.readlines()

# Find _toggle_autonomous and add _toggle_daily_learner after it
for i, line in enumerate(lines):
    if '    def _toggle_autonomous' in line:
        # Find end of method
        j = i + 1
        while j < len(lines) and not (lines[j].startswith('    def ') and j != i):
            j += 1

        # Insert method
        method = '''    def _toggle_daily_learner(self):
        """Toggle Daily Learner via IPC."""
        import time
        if not self.service:
            return

        if isinstance(self.service, ServiceProxy):
            self.service._send_cmd({"action": "toggle_daily_learner"})
            enabled = not getattr(self.service, '_daily_learner_enabled', False)
            self.service._daily_learner_enabled = enabled
        else:
            if hasattr(self.service, '_daily_learner') and self.service._daily_learner:
                enabled = self.service._daily_learner.toggle()
            else:
                return

        if enabled:
            self.daily_btn.config(bg="#006622", text="ALLTAG AN")
        else:
            self.daily_btn.config(bg="#1a1a3e", text="ALLTAG")

'''
        lines.insert(j, method)
        print(f'Method added at line {j}')
        break

with open(panel, 'w') as f:
    f.writelines(lines)

import subprocess
subprocess.run(['python3', '-m', 'py_compile', panel], check=True)
print('Syntax OK')
