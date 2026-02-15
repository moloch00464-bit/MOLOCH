#!/usr/bin/env python3
"""Entferne verbliebene _hlm FPS Counter Artefakte."""
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old = """                            if _hlm_ran:
                                _hlm_dt = time.perf_counter() - _hlm_t0
                                with self._fps_lock:
                                    self._fps[hand_landmark] = 1.0 / _hlm_dt if _hlm_dt > 0 else 0

                        # Gesten-Erkennung aus Pose-Keypoints"""

new = """                        # Gesten-Erkennung aus Pose-Keypoints"""

if old in code:
    code = code.replace(old, new)
    print('ENTFERNT')
else:
    print('NOT FOUND')

with open(svc, 'w') as f:
    f.write(code)
