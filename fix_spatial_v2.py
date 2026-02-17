#!/usr/bin/env python3
"""Fix: Spatial Learning - Manual Integration."""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Add import after hailo_postprocess import
old = '''from core.hardware.hailo_manager import get_hailo_manager'''
new = '''from core.hardware.hailo_manager import get_hailo_manager
from core.perception.spatial_learning import get_spatial_learning'''

if old in code and 'from core.perception.spatial_learning' not in code:
    code = code.replace(old, new)
    print('FIX 1: Import SpatialLearning - OK')
    fixes += 1
else:
    print('FIX 1: Already applied or anchor not found')

# FIX 2: Init in __init__ after Perception Engine init
# Find line with self._perception = None
lines = code.split('\n')
for i, line in enumerate(lines):
    if '        self._perception = None' in line and 'self._spatial_learning' not in lines[i+1]:
        lines.insert(i+1, '        self._spatial_learning = None  # Spatial Learning')
        code = '\n'.join(lines)
        print('FIX 2: Init SpatialLearning var - OK')
        fixes += 1
        break

# FIX 3: Lazy load after Perception lazy-load
if 'if not self._perception:' in code and 'if not self._spatial_learning:' not in code:
    old = '''        # Perception Engine (lazy-init)
        if not self._perception:
            from core.perception_engine import PerceptionEngine
            self._perception = PerceptionEngine()'''
    new = '''        # Perception Engine (lazy-init)
        if not self._perception:
            from core.perception_engine import PerceptionEngine
            self._perception = PerceptionEngine()
        if not self._spatial_learning:
            self._spatial_learning = get_spatial_learning()'''
    if old in code:
        code = code.replace(old, new)
        print('FIX 3: Lazy-load SpatialLearning - OK')
        fixes += 1

# FIX 5: Apply penalty at SCRFD (find the postprocess line)
old = '''                    boxes, scores, landmarks = decode_scrfd(
                        outputs, (fw, fh), conf_thresh=self.scrfd_conf_val,
                        nms_thresh=self.scrfd_nms_val
                    )

                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

new = '''                    boxes, scores, landmarks = decode_scrfd(
                        outputs, (fw, fh), conf_thresh=self.scrfd_conf_val,
                        nms_thresh=self.scrfd_nms_val
                    )

                    # Spatial Learning: Penalty fuer False-Detection Zones
                    if self._spatial_learning and len(boxes) > 0:
                        try:
                            cam = self.get_camera_controller()
                            if cam and cam.is_connected:
                                pos = cam.get_position()
                                if pos:
                                    penalty = self._spatial_learning.get_penalty_factor(pos.pan, pos.tilt)
                                    if penalty < 1.0:
                                        scores = [s * penalty for s in scores]
                                        keep = [i for i, s in enumerate(scores) if s >= self.scrfd_conf_val]
                                        if len(keep) < len(boxes):
                                            logger.info(f"[SpatialLearning] Filtered {len(boxes)-len(keep)} faces in penalty zone")
                                        boxes = [boxes[i] for i in keep]
                                        scores = [scores[i] for i in keep]
                                        landmarks = [landmarks[i] for i in keep]
                        except Exception:
                            pass

                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

if old in code:
    code = code.replace(old, new)
    print('FIX 5: Apply SCRFD Penalty - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND (decode_scrfd)')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes applied. Syntax OK.')
