#!/usr/bin/env python3
"""Fix: Spatial Learning Penalty Application."""

svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

old = '''                    boxes, scores, landmarks = decode_scrfd(
                        outputs, img_size=640,
                        conf_thresh=self.scrfd_conf_val,
                        iou_thresh=self.scrfd_nms_val
                    )
                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

new = '''                    boxes, scores, landmarks = decode_scrfd(
                        outputs, img_size=640,
                        conf_thresh=self.scrfd_conf_val,
                        iou_thresh=self.scrfd_nms_val
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
                                            logger.info(f"[SpatialLearning] Penalty zone - filtered {len(boxes)-len(keep)} faces")
                                        boxes = [boxes[i] for i in keep]
                                        scores = [scores[i] for i in keep]
                                        landmarks = [landmarks[i] for i in keep]
                        except Exception:
                            pass

                    dt = time.perf_counter() - t0
                    with self._fps_lock:
                        self._fps["scrfd"] = 1.0 / dt if dt > 0 else 0

                    if len(boxes) > 0:'''

if old in code:
    code = code.replace(old, new)
    print('FIX: Apply SCRFD Penalty - OK')
    with open(svc, 'w') as f:
        f.write(code)
    compile(open(svc).read(), svc, 'exec')
    print('Syntax OK.')
else:
    print('ANCHOR NOT FOUND!')
    import sys
    sys.exit(1)
