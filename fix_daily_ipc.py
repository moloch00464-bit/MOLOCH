#!/usr/bin/env python3
"""Fix: Daily Learner über IPC + Status JSON.

1. Daily Learner Status in Status-JSON schreiben
2. ServiceProxy liest Status und erstellt Proxy-Objekt
3. IPC-Kommando für Toggle
"""

# FIX 1: Daily Learner Status in Status-JSON
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old = '''                "rtsp_alive": self._rtsp_stream_alive,
                "rtsp_reconnecting": self._rtsp_reconnecting,
                "fps": {k: round(v, 1) for k, v in self._fps.items()},'''

new = '''                "rtsp_alive": self._rtsp_stream_alive,
                "rtsp_reconnecting": self._rtsp_reconnecting,
                "daily_learner_enabled": self._daily_learner.enabled if self._daily_learner else False,
                "fps": {k: round(v, 1) for k, v in self._fps.items()},'''

if old in code:
    code = code.replace(old, new)
    print('FIX 1: Daily Learner Status in JSON - OK')
    with open(svc, 'w') as f:
        f.write(code)
else:
    print('FIX 1: ANCHOR NOT FOUND!')

compile(open(svc).read(), svc, 'exec')

# FIX 2: IPC Command Handler für toggle_daily_learner
with open(svc) as f:
    code = f.read()

# Find where other toggle commands are handled
if 'toggle_daily_learner' not in code:
    old_cmd = '''            elif cmd == "toggle_autonomous_manual":
                logger.info("[IPC] toggle_autonomous_manual")
                self.toggle_autonomous_manual()'''
    
    new_cmd = '''            elif cmd == "toggle_autonomous_manual":
                logger.info("[IPC] toggle_autonomous_manual")
                self.toggle_autonomous_manual()
            elif cmd == "toggle_daily_learner":
                logger.info("[IPC] toggle_daily_learner")
                if self._daily_learner:
                    self._daily_learner.toggle()'''
    
    if old_cmd in code:
        code = code.replace(old_cmd, new_cmd)
        print('FIX 2: IPC Command toggle_daily_learner - OK')
        with open(svc, 'w') as f:
            f.write(code)
    else:
        print('FIX 2: Could not find IPC command handler')

compile(open(svc).read(), svc, 'exec')
print('Service fixes applied.')
