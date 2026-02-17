#!/usr/bin/env python3
"""Fix: Spatial Learning Integration in Service.

1. Import SpatialLearning
2. Bei "Unbekannt" -> log_unknown_face(pan, tilt)
3. Bei SCRFD Detection -> check penalty zone, senke Score
"""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

fixes = 0

# FIX 1: Import SpatialLearning
old_import = '''from core.perception.hailo_postprocess import (
    postprocess_scrfd, normalize_arcface, match_face,
    yolov8m_postprocess, yolov8s_pose_postprocess
)'''

new_import = '''from core.perception.hailo_postprocess import (
    postprocess_scrfd, normalize_arcface, match_face,
    yolov8m_postprocess, yolov8s_pose_postprocess
)
from core.perception.spatial_learning import get_spatial_learning'''

if old_import in code:
    code = code.replace(old_import, new_import)
    print('FIX 1: Import SpatialLearning - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Init SpatialLearning im Service __init__
old_init = '''        # Perception Engine
        self._perception = None'''

new_init = '''        # Perception Engine
        self._perception = None

        # Spatial Learning (False-Detection Zones)
        self._spatial_learning = None'''

if old_init in code:
    code = code.replace(old_init, new_init)
    print('FIX 2: Init SpatialLearning - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: Lazy-Load SpatialLearning
old_lazy = '''        # Perception Engine (lazy-init)
        if not self._perception:
            from core.perception_engine import PerceptionEngine
            self._perception = PerceptionEngine()'''

new_lazy = '''        # Perception Engine (lazy-init)
        if not self._perception:
            from core.perception_engine import PerceptionEngine
            self._perception = PerceptionEngine()

        # Spatial Learning (lazy-init)
        if not self._spatial_learning:
            self._spatial_learning = get_spatial_learning()'''

if old_lazy in code:
    code = code.replace(old_lazy, new_lazy)
    print('FIX 3: Lazy-Load SpatialLearning - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# FIX 4: Log "Unbekannt" mit Kamera-Position
# Nach ArcFace Recognition, wenn name=="Unbekannt"
old_unknown = '''                            # TTS Ansage (60s Cooldown pro Person)
                            if name != "Unbekannt" and name != "Keine DB":
                                now = time.time()
                                if now - self._last_announce.get(name, 0) > 60:
                                    self._last_announce[name] = now
                                    threading.Thread(
                                        target=self._announce_person,
                                        args=(name,), daemon=True
                                    ).start()'''

new_unknown = '''                            # Spatial Learning: Log "Unbekannt" mit Kamera-Position
                            if name == "Unbekannt":
                                try:
                                    cam = self.get_camera_controller()
                                    if cam and cam.is_connected:
                                        pos = cam.get_position()
                                        if pos and self._spatial_learning:
                                            self._spatial_learning.log_unknown_face(pos.pan, pos.tilt)
                                except Exception as e:
                                    logger.debug(f"Spatial learning log failed: {e}")

                            # TTS Ansage (60s Cooldown pro Person)
                            if name != "Unbekannt" and name != "Keine DB":
                                now = time.time()
                                if now - self._last_announce.get(name, 0) > 60:
                                    self._last_announce[name] = now
                                    threading.Thread(
                                        target=self._announce_person,
                                        args=(name,), daemon=True
                                    ).start()'''

if old_unknown in code:
    code = code.replace(old_unknown, new_unknown)
    print('FIX 4: Log Unbekannt mit Kamera-Position - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# FIX 5: Apply Penalty bei SCRFD Detection
# Nach postprocess_scrfd, vor face_boxes genutzt wird
old_scrfd = '''                    boxes, scores, landmarks = postprocess_scrfd(
                        outputs, (fw, fh), conf_thresh=self.scrfd_conf_val,
                        nms_thresh=self.scrfd_nms_val
                    )

                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

new_scrfd = '''                    boxes, scores, landmarks = postprocess_scrfd(
                        outputs, (fw, fh), conf_thresh=self.scrfd_conf_val,
                        nms_thresh=self.scrfd_nms_val
                    )

                    # Spatial Learning: Apply Penalty fuer False-Detection Zones
                    if self._spatial_learning and len(boxes) > 0:
                        try:
                            cam = self.get_camera_controller()
                            if cam and cam.is_connected:
                                pos = cam.get_position()
                                if pos:
                                    penalty = self._spatial_learning.get_penalty_factor(pos.pan, pos.tilt)
                                    if penalty < 1.0:
                                        # Score senken fuer Detections in Penalty-Zone
                                        scores = [s * penalty for s in scores]
                                        # Filter out low-score detections
                                        keep_idx = [i for i, s in enumerate(scores) if s >= self.scrfd_conf_val]
                                        boxes = [boxes[i] for i in keep_idx]
                                        scores = [scores[i] for i in keep_idx]
                                        landmarks = [landmarks[i] for i in keep_idx]
                                        if penalty < 1.0 and len(keep_idx) < len(boxes):
                                            logger.info(f"[SpatialLearning] Penalty zone - filtered {len(boxes)-len(keep_idx)} faces")
                        except Exception as e:
                            logger.debug(f"Spatial learning penalty failed: {e}")

                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

if old_scrfd in code:
    code = code.replace(old_scrfd, new_scrfd)
    print('FIX 5: Apply Penalty bei SCRFD Detection - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

compile(open(svc).read(), svc, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
