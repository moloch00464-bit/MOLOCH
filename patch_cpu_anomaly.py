#!/usr/bin/env python3
"""CPU-NPU Spec: 3 Features zusammen.

1. Head Pose: PnP Solve aus SCRFD 5-Point Landmarks -> face_state.json + Video
2. person_count_jump: YOLO 2+ Personen + SCRFD <=1 Face -> pose/hand Boost
3. unknown_person: ArcFace Boost +0.3 -> +0.5
"""
import sys

# ============================================================
# TEIL 1: hailo_postprocess.py - Head Pose Funktion
# ============================================================
pp = '/home/molochzuhause/moloch/core/perception/hailo_postprocess.py'
with open(pp) as f:
    pcode = f.read()

fixes = 0

# --- 1A: estimate_head_pose Funktion nach draw_faces, vor draw_name ---
old_draw_name = '''def draw_name(frame: np.ndarray, box: np.ndarray, name: str,
              similarity: float, h: int, w: int, emotion: str = None,
              gender: str = None, age_range: str = None):'''

new_draw_name = '''def estimate_head_pose(landmarks_5: np.ndarray, frame_w: int, frame_h: int):
    """Schaetze Kopf-Pose (Pitch/Yaw/Roll) aus SCRFD 5-Point Landmarks.

    Args:
        landmarks_5: (10,) = 5 Punkte x (x,y) normalisiert [0,1]
        frame_w, frame_h: Frame-Dimensionen fuer Kamera-Matrix
    Returns:
        (pitch, yaw, roll) in Grad oder None bei Fehler
    """
    # 2D Punkte: left_eye, right_eye, nose, left_mouth, right_mouth
    pts_2d = np.array([
        [landmarks_5[0] * frame_w, landmarks_5[1] * frame_h],
        [landmarks_5[2] * frame_w, landmarks_5[3] * frame_h],
        [landmarks_5[4] * frame_w, landmarks_5[5] * frame_h],
        [landmarks_5[6] * frame_w, landmarks_5[7] * frame_h],
        [landmarks_5[8] * frame_w, landmarks_5[9] * frame_h],
    ], dtype=np.float64)

    # 3D Modell-Punkte (generisches Gesicht, mm)
    pts_3d = np.array([
        [-30.0, -30.0, -30.0],   # left eye
        [ 30.0, -30.0, -30.0],   # right eye
        [  0.0,   0.0,   0.0],   # nose tip
        [-25.0,  30.0, -20.0],   # left mouth
        [ 25.0,  30.0, -20.0],   # right mouth
    ], dtype=np.float64)

    # Kamera-Matrix (Naeherung: focal_length ~ frame_width)
    focal = float(frame_w)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    cam_matrix = np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1.0],
    ], dtype=np.float64)

    try:
        import cv2 as _cv2
        success, rvec, tvec = _cv2.solvePnP(
            pts_3d, pts_2d, cam_matrix, None,
            flags=_cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        rmat, _ = _cv2.Rodrigues(rvec)
        # Euler-Winkel aus Rotationsmatrix
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        if sy > 1e-6:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = 0.0
        return (round(pitch, 1), round(yaw, 1), round(roll, 1))
    except Exception:
        return None


def draw_name(frame: np.ndarray, box: np.ndarray, name: str,
              similarity: float, h: int, w: int, emotion: str = None,
              gender: str = None, age_range: str = None,
              head_pose: tuple = None):'''

if old_draw_name in pcode:
    pcode = pcode.replace(old_draw_name, new_draw_name)
    print('1A: estimate_head_pose Funktion - OK')
    fixes += 1
else:
    print('1A: ANCHOR NOT FOUND!')

# --- 1B: draw_name: Head Pose anzeigen ---
old_label = '''    label = f"{name} ({similarity:.0%})" if name != "Unbekannt" else "Unbekannt"
    if emotion:
        label += f" [{emotion}]"
    if gender and age_range:
        label += f" {gender}/{age_range}"'''

new_label = '''    label = f"{name} ({similarity:.0%})" if name != "Unbekannt" else "Unbekannt"
    if emotion:
        label += f" [{emotion}]"
    if gender and age_range:
        label += f" {gender}/{age_range}"
    if head_pose:
        _p, _y, _r = head_pose
        label += f" P{_p:.0f}/Y{_y:.0f}/R{_r:.0f}"'''

if old_label in pcode:
    pcode = pcode.replace(old_label, new_label)
    print('1B: Head Pose in draw_name - OK')
    fixes += 1
else:
    print('1B: ANCHOR NOT FOUND!')

with open(pp, 'w') as f:
    f.write(pcode)

print(f'\nPostprocess: {fixes}/2 Fixes.')
if fixes < 2:
    print('POSTPROCESS INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 2: moloch_service.py - Head Pose + Kontext-Felder
# ============================================================
svc = '/home/molochzuhause/moloch/core/moloch_service.py'
with open(svc) as f:
    code = f.read()

sfixes = 0

# --- 2A: Import estimate_head_pose ---
old_import = '''from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms, decode_yolov8_pose,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_poses, draw_hands, enforce_draw_priority,
    decode_hand_landmark, draw_hand_landmarks,
)'''

new_import = '''from core.perception.hailo_postprocess import (
    decode_scrfd, decode_yolov8_nms, decode_yolov8_pose,
    normalize_arcface, match_face,
    draw_faces, draw_name, draw_persons, draw_poses, draw_hands, enforce_draw_priority,
    decode_hand_landmark, draw_hand_landmarks,
    estimate_head_pose,
)'''

if old_import in code:
    code = code.replace(old_import, new_import)
    print('2A: Import estimate_head_pose - OK')
    sfixes += 1
else:
    print('2A: ANCHOR NOT FOUND!')

# --- 2B: Head Pose berechnen nach SCRFD ---
# Nach face_boxes Zuweisung, Head Pose fuer erstes Gesicht berechnen
old_face_boxes = '''                    if len(boxes) > 0:
                        if "face" in _allowed_draws:
                            draw_faces(annotated, boxes, scores, landmarks, scale_x, scale_y)
                        face_boxes = list(zip(boxes, scores, landmarks))
                        face_detected = True'''

new_face_boxes = '''                    if len(boxes) > 0:
                        if "face" in _allowed_draws:
                            draw_faces(annotated, boxes, scores, landmarks, scale_x, scale_y)
                        face_boxes = list(zip(boxes, scores, landmarks))
                        face_detected = True
                        # Head Pose fuer erstes Gesicht (CPU, ~5ms)
                        _head_pose = estimate_head_pose(landmarks[0], frame_w, frame_h)'''

if old_face_boxes in code:
    code = code.replace(old_face_boxes, new_face_boxes)
    print('2B: Head Pose nach SCRFD - OK')
    sfixes += 1
else:
    print('2B: ANCHOR NOT FOUND!')

# --- 2C: Head Pose an draw_name uebergeben ---
old_draw_name_call = '''                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range)'''

new_draw_name_call = '''                            draw_name(annotated, box, name, sim, fh, fw,
                                      emotion=emotion, gender=gender, age_range=age_range,
                                      head_pose=_head_pose if '_head_pose' in dir() else None)
                            self._write_face_state(name, sim, len(face_boxes),
                                                   emotion=emotion, gender=gender, age_range=age_range,
                                                   head_pose=_head_pose if '_head_pose' in dir() else None)'''

if old_draw_name_call in code:
    code = code.replace(old_draw_name_call, new_draw_name_call)
    print('2C: Head Pose an draw_name + face_state - OK')
    sfixes += 1
else:
    print('2C: ANCHOR NOT FOUND!')

# --- 2D: _write_face_state um head_pose erweitern ---
old_write_state = '''    def _write_face_state(self, name, similarity, person_count, emotion=None, gender=None, age_range=None):
        """Schreibe Face-Recognition-State fuer IPC mit push_to_talk."""
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "gender": gender,
                "age_range": age_range,
                "timestamp": time.time(),
                "source": "moloch_service"
            }'''

new_write_state = '''    def _write_face_state(self, name, similarity, person_count, emotion=None, gender=None, age_range=None, head_pose=None):
        """Schreibe Face-Recognition-State fuer IPC mit push_to_talk."""
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "gender": gender,
                "age_range": age_range,
                "head_pose": {"pitch": head_pose[0], "yaw": head_pose[1], "roll": head_pose[2]} if head_pose else None,
                "timestamp": time.time(),
                "source": "moloch_service"
            }'''

if old_write_state in code:
    code = code.replace(old_write_state, new_write_state)
    print('2D: head_pose in face_state.json - OK')
    sfixes += 1
else:
    print('2D: ANCHOR NOT FOUND!')

# --- 2E: person_count + face_count im Perception-Kontext ---
old_perc_ctx = '''                _perc_ctx = {
                    "face_detected": face_detected,
                    "face_bbox": _perc_face_bbox,
                    "person_detected": _perc_person,
                    "unknown_person": face_detected and 'name' in dir() and name == "Unbekannt",
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                    "gesture": self._current_gesture.type.value if self._current_gesture else "none",
                }'''

new_perc_ctx = '''                _person_count = len(persons) if self.yolo_active and 'persons' in dir() and persons else 0
                _face_count = len(face_boxes)
                _perc_ctx = {
                    "face_detected": face_detected,
                    "face_bbox": _perc_face_bbox,
                    "person_detected": _perc_person,
                    "unknown_person": face_detected and 'name' in dir() and name == "Unbekannt",
                    "person_count": _person_count,
                    "face_count": _face_count,
                    "motion_level": 0.0,
                    "camera_moving": _perc_camera_moving,
                    "gesture": self._current_gesture.type.value if self._current_gesture else "none",
                }'''

if old_perc_ctx in code:
    code = code.replace(old_perc_ctx, new_perc_ctx)
    print('2E: person_count + face_count im Kontext - OK')
    sfixes += 1
else:
    print('2E: ANCHOR NOT FOUND!')

with open(svc, 'w') as f:
    f.write(code)

print(f'\nService: {sfixes}/5 Fixes.')
if sfixes < 5:
    print('SERVICE INCOMPLETE!')
    sys.exit(1)

# ============================================================
# TEIL 3: perception_engine.py - person_count_jump + unknown boost
# ============================================================
pe = '/home/molochzuhause/moloch/core/perception_engine.py'
with open(pe) as f:
    ecode = f.read()

efixes = 0

# --- 3A: unknown_person Boost erhoehen: +0.3 -> +0.5 ---
old_unknown = '''        if unknown:
            scores["arcface"] += 0.3'''

new_unknown = '''        if unknown:
            scores["arcface"] += 0.5'''

if old_unknown in ecode:
    ecode = ecode.replace(old_unknown, new_unknown)
    print('3A: unknown_person Boost +0.3 -> +0.5 - OK')
    efixes += 1
else:
    print('3A: ANCHOR NOT FOUND!')

# --- 3B: person_count_jump Anomalie ---
# Nach dem "Nichts erkannt" Block und vor "Hand Occlusion" Block
old_hand_occ = '''        # Hand Occlusion -> pose + hand_landmark boosten
        if self._hand_occlusion:'''

new_hand_occ = '''        # person_count_jump: YOLO 2+ Personen aber <=1 Gesicht -> Anomalie
        _person_count = ctx.get("person_count", 0)
        _face_count = ctx.get("face_count", 0)
        if _person_count >= 2 and _face_count <= 1:
            scores["hand_landmark"] += 0.5
            scores["pose"] += 0.4

        # Hand Occlusion -> pose + hand_landmark boosten
        if self._hand_occlusion:'''

if old_hand_occ in ecode:
    ecode = ecode.replace(old_hand_occ, new_hand_occ)
    print('3B: person_count_jump Anomalie - OK')
    efixes += 1
else:
    print('3B: ANCHOR NOT FOUND!')

with open(pe, 'w') as f:
    f.write(ecode)

print(f'\nPerception: {efixes}/2 Fixes.')
if efixes < 2:
    print('PERCEPTION INCOMPLETE!')
    sys.exit(1)

print(f'\n=== ALLE {fixes + sfixes + efixes}/9 FIXES KOMPLETT ===')
