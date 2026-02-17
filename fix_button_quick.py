#!/usr/bin/env python3
"""Quick fix: Button send_command -> _send_cmd"""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

# Fix: Use _send_cmd instead of send_command
code = code.replace(
    'self.service.send_command({"action": "toggle_daily_learner"})',
    'self.service._send_cmd({"action": "toggle_daily_learner"})'
)

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print('FIX: send_command -> _send_cmd - OK')
