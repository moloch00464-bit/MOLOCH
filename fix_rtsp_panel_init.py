#!/usr/bin/env python3
"""Fix: RTSP Panel Init-Variablen."""

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

old = '''        # Display
        self._photo = None
        self._display_after_id = None
        self._canvas_image_id = None

        # --- Style Setup ---'''

new = '''        # Display
        self._photo = None
        self._display_after_id = None
        self._canvas_image_id = None

        # RTSP Stream Indikator
        self._stream_indicator_id = None
        self._stream_offline_text_id = None

        # --- Style Setup ---'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Panel Init-Variablen - OK')
else:
    print('FIX: ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print('Syntax OK.')
