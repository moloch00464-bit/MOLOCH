#!/usr/bin/env python3
"""Fix: Add toggle_daily_learner IPC command."""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old = '''        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
        elif action == 'reload_face_db':'''

new = '''        elif action == 'toggle_autonomous':
            self.toggle_autonomous_manual()
        elif action == 'toggle_daily_learner':
            if self._daily_learner:
                self._daily_learner.toggle()
                logger.info(f"[IPC] toggle_daily_learner -> {self._daily_learner.enabled}")
        elif action == 'reload_face_db':'''

if old in code:
    code = code.replace(old, new)
    print('FIX: toggle_daily_learner IPC command - OK')
    with open(svc, 'w') as f:
        f.write(code)
    compile(open(svc).read(), svc, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
