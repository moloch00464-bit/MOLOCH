#!/usr/bin/env python3
"""Fix: Read daily_learner_enabled from status JSON."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

old = '''        self._npu_paused = s.get('npu_paused', False)
        self._autonomous_mode = s.get('autonomous_mode', False)
        self._moloch_has_control = s.get('moloch_has_control', False)
        self._tentakel_enabled = s.get('tentakel_enabled', False)

        self._active_ctx = {m: True for m in s.get('active_models', [])}'''

new = '''        self._npu_paused = s.get('npu_paused', False)
        self._autonomous_mode = s.get('autonomous_mode', False)
        self._moloch_has_control = s.get('moloch_has_control', False)
        self._tentakel_enabled = s.get('tentakel_enabled', False)
        self._daily_learner_enabled = s.get('daily_learner_enabled', False)

        self._active_ctx = {m: True for m in s.get('active_models', [])}'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Read daily_learner_enabled from status - OK')
    with open(panel, 'w') as f:
        f.write(code)
    compile(open(panel).read(), panel, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
