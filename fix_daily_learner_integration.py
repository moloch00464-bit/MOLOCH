#!/usr/bin/env python3
"""Fix: Daily Learner Integration in Service.

1. Import DailyLearner
2. Init in __init__
3. Call maybe_snapshot() nach ArcFace Recognition
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Import DailyLearner
if 'from core.daily_learner import get_daily_learner' not in code:
    old = 'from core.perception.spatial_learning import get_spatial_learning'
    new = '''from core.perception.spatial_learning import get_spatial_learning
from core.daily_learner import get_daily_learner'''
    if old in code:
        code = code.replace(old, new)
        print('FIX 1: Import DailyLearner - OK')
        fixes += 1
    else:
        print('FIX 1: ANCHOR NOT FOUND!')
else:
    print('FIX 1: Already applied')

# FIX 2: Init DailyLearner nach SpatialLearning
if 'self._daily_learner = None' not in code:
    old = 'self._spatial_learning = None  # Spatial Learning'
    new = '''self._spatial_learning = None  # Spatial Learning
        self._daily_learner = None  # Daily Learner'''
    if old in code:
        code = code.replace(old, new)
        print('FIX 2: Init DailyLearner var - OK')
        fixes += 1
    else:
        print('FIX 2: ANCHOR NOT FOUND!')
else:
    print('FIX 2: Already applied')

# FIX 3: Lazy-Load DailyLearner
if 'self._daily_learner = get_daily_learner()' not in code:
    old = '''            self._spatial_learning = get_spatial_learning()  # Init Spatial Learning'''
    new = '''            self._spatial_learning = get_spatial_learning()  # Init Spatial Learning
            self._daily_learner = get_daily_learner()  # Init Daily Learner'''
    if old in code:
        code = code.replace(old, new)
        print('FIX 3: Lazy-Load DailyLearner - OK')
        fixes += 1
    else:
        print('FIX 3: ANCHOR NOT FOUND!')
else:
    print('FIX 3: Already applied')

# FIX 4: Call maybe_snapshot nach ArcFace Recognition
# Nach draw_name, vor Spatial Learning log
old_snapshot = '''                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range,
                                                   head_pose=_head_pose if '_head_pose' in dir() else None)

                            # Spatial Learning: Log "Unbekannt" mit Kamera-Position'''

new_snapshot = '''                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range,
                                                   head_pose=_head_pose if '_head_pose' in dir() else None)

                            # Daily Learner: Maybe Snapshot fuer Markus/Unbekannt
                            if self._daily_learner and crop is not None:
                                try:
                                    self._daily_learner.maybe_snapshot(
                                        face_crop=crop,
                                        name=name,
                                        confidence=sim,
                                        bbox=box,
                                        frame_height=fh,
                                        head_pose=_head_pose if '_head_pose' in dir() else None
                                    )
                                except Exception as e:
                                    logger.debug(f"Daily learner snapshot failed: {e}")

                            # Spatial Learning: Log "Unbekannt" mit Kamera-Position'''

if old_snapshot in code:
    code = code.replace(old_snapshot, new_snapshot)
    print('FIX 4: Call maybe_snapshot nach ArcFace - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
