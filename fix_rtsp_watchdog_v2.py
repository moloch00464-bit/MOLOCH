#!/usr/bin/env python3
"""Fix: RTSP Watchdog Init-Variablen."""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

# FIX: Init-Variablen fuer RTSP Watchdog NACH _annotated_lock
old = '''        # Frame Locks
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        # Model enable flags (plain bools, NOT tk.BooleanVar)'''

new = '''        # Frame Locks
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        # RTSP Watchdog
        self._rtsp_last_frame_time = 0
        self._rtsp_stream_alive = False
        self._rtsp_reconnecting = False
        self._rtsp_cap = None

        # Model enable flags (plain bools, NOT tk.BooleanVar)'''

if old in code:
    code = code.replace(old, new)
    print('FIX: RTSP Watchdog Init-Variablen - OK')
else:
    print('FIX: ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print('Syntax OK.')
