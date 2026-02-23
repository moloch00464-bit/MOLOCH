# NPU-Output Inventar — M.O.L.O.C.H. v3.x

> Stand: 2026-02-23 | Nur Diagnose, nichts geaendert

## Zusammenfassung

6 Modelle aktiv auf Hailo-10H (40 TOPS), ~51 MB HEF total, alle permanent geladen.

| Modell | HEF | Berechnet | Genutzt | Weggeworfen |
|--------|-----|-----------|---------|-------------|
| SCRFD | 5.8 MB | Boxes+Scores+Landmarks(N) | 1 Face: bbox, conf, pitch, yaw | Faces 2+, roll, Landmarks 2+ |
| ArcFace | 2.6 MB | 512-dim Embedding | name, similarity | Roher Embedding-Vektor |
| YOLOv8m | 21 MB | 80 Klassen, bis 100/Klasse | person_count, distance, objects[] | Person-BBox-Koordinaten |
| Pose | 14 MB | 17 Keypoints x (x,y,vis) | pose_count, pose_energy | Alle 17 KP-Koordinaten |
| Hand Landmark | 1.2 MB | 21 KP + 3D-World + Handedness | hand_detected, hand_gesture | World-3D, Z-Werte, Presence |
| Face Attr | 6.9 MB | 40 CelebA-Attribute | gender, age_range, emotion | 35 von 40 Attributen |

---

## Modell 1: SCRFD — Face Detection

- **HEF**: `scrfd_10g.hef` (5.8 MB, ~40 FPS live)
- **Input**: 640x640 RGB
- **Postprocess**: `hailo_postprocess.py::decode_scrfd()`

### Outputs (9 NPU-Layer, 3 Strides x 3 Heads)

| Layer | Shape | Inhalt |
|-------|-------|--------|
| conv41/49/56 | (H,W,2) | Scores (Confidence pro Anchor) |
| conv42/50/57 | (H,W,8) | Bounding Boxes (4 Werte pro Anchor) |
| conv43/51/58 | (H,W,20) | Landmarks (5 Punkte x 2 Koord. pro Anchor) |

Nach Anchor-Decode + NMS:
```
boxes:     (N, 4)   xyxy normalisiert [0,1]
scores:    (N,)     confidence float
landmarks: (N, 10)  5 Punkte: left_eye, right_eye, nose, left_mouth, right_mouth
```

### Genutzt

| Daten | Wohin | Datei:Zeile |
|-------|-------|-------------|
| face_detected (bool) | PerceptionFrame.face_detected | moloch_service.py:2358 |
| face_count (int) | PerceptionFrame.face_count | moloch_service.py:2359 |
| face_confidence (float, nur Face #1) | PerceptionFrame.face_confidence | moloch_service.py:2360 |
| face_bbox (x1,y1,x2,y2, nur Face #1) | PerceptionFrame.face_bbox | moloch_service.py:2361 |
| head_pitch (Grad, nur Face #1) | PerceptionFrame.head_pitch | moloch_service.py:2363 |
| head_yaw (Grad, nur Face #1) | PerceptionFrame.head_yaw | moloch_service.py:2364 |

### Ignoriert

| Daten | Status | Kommentar |
|-------|--------|-----------|
| **head_roll** | Berechnet in estimate_head_pose(), NICHT in PerceptionFrame | Feld fehlt im Dataclass |
| **Landmarks Face 2+** | Komplett verworfen | Nur landmarks[0] geht in Head-Pose |
| **Boxes/Scores Face 2+** | Nur fuer draw_faces() gezeichnet | Keine Analyse |
| **face_bbox** | Im Dataclass, aber NICHT in to_dict() | Wird nicht via IPC exportiert |

### Potential

- **head_roll**: Trivial hinzufuegbar — ist bereits berechnet, nur kein pf-Feld
- **Multi-Face-Tracking**: Landmarks aller Gesichter fuer Aufmerksamkeits-Analyse
- **Face-Groesse als Distanz-Proxy**: Genauer als YOLOv8m Person-BBox bei Gesichtserkennung
- **Landmark-Qualitaet als Verdeckungs-Indikator**: Teilweise verdeckte Gesichter erkennen

---

## Modell 2: ArcFace — Face Recognition

- **HEF**: `arcface_mobilefacenet.hef` (2.6 MB, ~81 FPS live)
- **Input**: 112x112 RGB (Crop aus SCRFD-BBox + 20% Margin)
- **Postprocess**: `hailo_postprocess.py::normalize_arcface()` + `match_face()`

### Outputs (1 NPU-Layer)

| Layer | Shape | Inhalt |
|-------|-------|--------|
| (dynamisch) | (512,) | L2-normierter Embedding-Vektor |

Nach Matching gegen `face_embeddings.json`:
```
name:       str ("markus"|"lilly"|...|"unknown")
similarity: float 0.0-1.0 (Cosine-Similarity)
```

### Genutzt

| Daten | Wohin | Datei:Zeile |
|-------|-------|-------------|
| face_id (name.lower()) | PerceptionFrame.face_id | moloch_service.py:2366 |
| face_similarity | PerceptionFrame.face_similarity | moloch_service.py:2367 |
| name+sim+emotion+gender+age | /tmp/moloch_face_state.json | moloch_service.py:2111-2129 |
| Embedding (temporaer) | DailyLearner fuer Re-Learning | moloch_service.py:963 |

### Ignoriert

| Daten | Status | Kommentar |
|-------|--------|-----------|
| **512-dim Embedding** | Temporaer benutzt, nicht persistent | Wird nach match_face() verworfen |
| **Sub-Threshold Similarity** | Auf 0.0 gesetzt wenn unter Threshold | Genauer Wert verloren |
| **Embeddings Face 2+** | Alle berechnet, letztes ueberschreibt | for-Schleife, letztes gewinnt |

### Potential

- **Embedding-History**: Clustering ueber Zeit fuer bessere Erkennung
- **Sub-Threshold Tracking**: "Fast erkannt"-Metriken fuer adaptive Thresholds
- **Multi-Face Recognition**: Alle Gesichter gleichzeitig identifizieren
- **Embedding-Drift-Erkennung**: Aenderung der Embeddings einer Person ueber Tage

---

## Modell 3: YOLOv8m — Object Detection

- **HEF**: `yolov8m_h10.hef` (21 MB, ~32 FPS live)
- **Input**: 640x640 RGB
- **Postprocess**: `hailo_postprocess.py::decode_yolov8_nms()`
- **WICHTIG**: On-Chip NMS (Hailo macht NMS intern), kein CPU-NMS noetig
- **WICHTIG**: Laeuft NUR wenn face_detected=False (SCRFD hat Prioritaet)

### Outputs (1 NPU-Layer, On-Chip NMS Format)

```
80 Klassen-Bloecke, je: float count + count x (y1, x1, y2, x2, score)
Max 100 Detections pro Klasse, normalisiert [0,1]
```

### 80 COCO-Klassen (alle dekodiert)

```
 0: person       1: bicycle      2: car          3: motorcycle
 4: airplane     5: bus          6: train        7: truck
 8: boat         9: traffic_light 10: fire_hydrant 11: stop_sign
12: parking_meter 13: bench      14: bird        15: cat
16: dog         17: horse       18: sheep       19: cow
20: elephant    21: bear        22: zebra       23: giraffe
24: backpack    25: umbrella    26: handbag     27: tie
28: suitcase    29: frisbee     30: skis        31: snowboard
32: sports_ball 33: kite        34: baseball_bat 35: baseball_glove
36: skateboard  37: surfboard   38: tennis_racket 39: bottle
40: wine_glass  41: cup         42: fork        43: knife
44: spoon       45: bowl        46: banana      47: apple
48: sandwich    49: orange      50: broccoli    51: cake
52: chair       53: couch       54: potted_plant 55: bed
56: dining_table 57: toilet     58: tv          59: laptop
60: mouse       61: remote      62: keyboard    63: cell_phone
64: microwave   65: oven        66: toaster     67: sink
68: refrigerator 69: book       70: clock       71: vase
72: scissors    73: teddy_bear  74: hair_drier  75: toothbrush
76-79: (reserviert)
```

### Genutzt

| Daten | Wohin | Datei:Zeile |
|-------|-------|-------------|
| person_detected (bool) | PerceptionFrame.person_detected | moloch_service.py:2340 |
| person_count (int) | PerceptionFrame.person_count | moloch_service.py:2341 |
| distance ("close"/"medium"/"far") | PerceptionFrame.distance | moloch_service.py:2344 |
| distance_ratio (float, BBox-Flaeche) | PerceptionFrame.distance_ratio | moloch_service.py:2343 |
| objects[] (Nicht-Person-Klassen) | PerceptionFrame.objects | moloch_service.py:2356 |

Distanz-Schwellen (nur groesste Person):
- close: area > 0.15
- medium: area 0.05-0.15
- far: area < 0.05

### Ignoriert

| Daten | Status | Kommentar |
|-------|--------|-----------|
| **Person-BBox-Koordinaten** | Nur fuer Distanz-Berechnung, dann weg | Nicht in PerceptionFrame.to_dict() |
| **distance_ratio** | Im Dataclass, aber NICHT in to_dict() | Wird nicht via IPC exportiert |
| **Person-Positionen 2+** | Nur groesste fuer Distanz | Rest ignoriert |
| **class_id in objects** | Nur class-String behalten | ID geht verloren |

### Potential

- **Raumkontext**: Objekt-Kombination ergibt Raumtyp (Couch+TV = Wohnzimmer)
- **Objekt-Persistenz**: Tracking welche Objekte normalerweise da sind vs. neu
- **Person-Positionen**: Wo im Raum stehen Personen (nicht nur "wie gross")
- **Objekt-Person-Relationen**: Wer haelt was, wer sitzt wo

---

## Modell 4: YOLOv8s Pose — Skeleton Detection

- **HEF**: `yolov8s_pose_h10.hef` (14 MB, ~17 FPS live)
- **Input**: 640x640 RGB
- **Postprocess**: `hailo_postprocess.py::decode_yolov8_pose()`

### Outputs (9 NPU-Layer, 3 Strides x 3 Heads)

| Layer | Shape | Inhalt |
|-------|-------|--------|
| conv43/57/70 | (H,W,64) | BBox Regression (DFL, 4x16 bins) |
| conv44/58/71 | (H,W,1) | Person Score |
| conv45/59/72 | (H,W,51) | 17 Keypoints x 3 (x,y,vis) |

Nach DFL-Decode + NMS:
```
List[Dict]:
  "bbox":      [x1,y1,x2,y2]  Model-Pixel (640x640)
  "score":     float (Person-Confidence)
  "keypoints": np.ndarray (17, 3) = [x, y, visibility]
```

### 17 COCO-Keypoints

```
 0: nose           1: left_eye       2: right_eye
 3: left_ear       4: right_ear      5: left_shoulder
 6: right_shoulder 7: left_elbow     8: right_elbow
 9: left_wrist    10: right_wrist   11: left_hip
12: right_hip     13: left_knee     14: right_knee
15: left_ankle    16: right_ankle
```

### Genutzt

| Daten | Wohin | Datei:Zeile |
|-------|-------|-------------|
| pose_count (int) | PerceptionFrame.pose_count | moloch_service.py:2377 |
| pose_energy (float 0-1, Keypoint-Delta) | PerceptionFrame.pose_energy | moloch_service.py:2378 |
| face_center (Nase+Augen+Ohren) | AutonomousTracker.update_pose_detection() | moloch_service.py:1139-1171 |
| Wrist-KPs (9,10) | Hand-Landmark Crop-Region | moloch_service.py (Hand-Pipeline) |

Pose-Energy = Mean-Norm aller sichtbaren Keypoint-Deltas / 50px, clamped [0,1].

### Ignoriert

| Daten | Status | Kommentar |
|-------|--------|-----------|
| **Alle 17 KP-Koordinaten** | Nur fuer Energy + Tracker, dann weg | Nicht in PerceptionFrame |
| **KP-Visibility pro Punkt** | Intern fuer Energy-Filter (vis>0.3) | Nicht exponiert |
| **Skeleton-Verbindungen** | Nur gezeichnet (draw_poses) | Keine Analyse |
| **Personen 2+** | Nur Person mit hoechstem Score fuer Energy | Rest nur gezeichnet |

### Potential

- **Aktivitaets-Erkennung**: Sitzen/Stehen/Liegen aus Keypoint-Geometrie
- **Gesten aus Body-Pose**: Arme hoch, winken, zeigen (aktuell nur GestureDetector, der aber falsch mit Hand-Landmarks gefuettert wird)
- **Koerpersprache**: Verschraenkte Arme, Kopf-Neigung, Schulter-Asymmetrie
- **Person Re-ID**: Skelett-Proportionen als biometrisches Merkmal

---

## Modell 5: Hand Landmark Lite — 21 Finger-Keypoints

- **HEF**: `hand_landmark_lite.hef` (1.2 MB, ~124 FPS live)
- **Input**: 224x224 RGB (Crop aus Pose-Wrist oder Detection)
- **Postprocess**: `hailo_postprocess.py::decode_hand_landmark()`

### Outputs (4 NPU-Layer)

| Layer | Shape | Inhalt |
|-------|-------|--------|
| fc1 (HAND_LM_SCREEN) | (63,) | 21 Landmarks x (x,y,z) Screen-Space |
| fc2 (HAND_LM_PRESENCE) | (1,) | Presence Logit |
| fc3 (HAND_LM_WORLD) | (63,) | 21 Landmarks x (x,y,z) World-Space 3D |
| fc4 (HAND_LM_HANDEDNESS) | (1,) | Handedness Logit (Links/Rechts) |

Nach Decode:
```
Dict:
  "landmarks":  np.ndarray (21, 3)  x,y norm [0,1], z unnormalisiert
  "handedness": "L" oder "R"
  "presence":   float [0,1]
```

### 21 MediaPipe Hand-Landmarks

```
 0: WRIST
 1: THUMB_CMC        2: THUMB_MCP       3: THUMB_IP        4: THUMB_TIP
 5: INDEX_MCP        6: INDEX_PIP       7: INDEX_DIP       8: INDEX_TIP
 9: MIDDLE_MCP      10: MIDDLE_PIP     11: MIDDLE_DIP     12: MIDDLE_TIP
13: RING_MCP        14: RING_PIP       15: RING_DIP       16: RING_TIP
17: PINKY_MCP       18: PINKY_PIP      19: PINKY_DIP      20: PINKY_TIP
```

### Genutzt

| Daten | Wohin | Datei:Zeile |
|-------|-------|-------------|
| hand_detected (bool) | PerceptionFrame.hand_detected | moloch_service.py:2382 |
| hand_gesture (str) | PerceptionFrame.hand_gesture | moloch_service.py:2383 |

### BUG: GestureDetector Mismatch

`GestureDetector.detect()` erwartet **COCO-17 Body-Keypoints** (gesture_detector.py:117: `if len(keypoints) < 17`), bekommt aber **21 MediaPipe Hand-Landmarks**. Anatomie-Mapping ist komplett falsch — Finger werden als Schultern/Knie interpretiert. Gesten-Erkennung funktioniert dadurch NICHT korrekt.

### Ignoriert

| Daten | Status | Kommentar |
|-------|--------|-----------|
| **fc3 (World-Space 3D)** | Komplett ignoriert | Nicht in decode_hand_landmark() |
| **Z-Koordinaten (Tiefe)** | Dekodiert, verworfen | Nur x,y verwendet |
| **Presence Score** | Nur bool-Entscheidung (>0.65) | Genauer Wert weg |
| **Handedness** | Dekodiert, nur gezeichnet | Nicht in PerceptionFrame |
| **21 KP-Koordinaten** | Nur an (falschen) GestureDetector | Nicht in PerceptionFrame |

### Potential

- **Echte Finger-Gesten**: Daumen hoch, Zeigefinger, Peace, OK-Zeichen — direkt aus Finger-Geometrie
- **3D-Handpose**: World-Space-Daten fuer raeumliche Interaktion (fc3 liegt brach!)
- **Greiferkennung**: Welche Finger sind gebeugt/gestreckt
- **Handedness nutzen**: Links/Rechts-Hand fuer UI-Interaktion
- **GestureDetector fixen**: Entweder Pose-Keypoints verwenden ODER eigenen Hand-Gesture-Detektor

---

## Modell 6: Face Attributes ResNet — CelebA 40 Attribute

- **HEF**: `face_attr_resnet_v1_18.hef` (6.9 MB, ~2926 FPS theoretisch)
- **Input**: 178x218 RGB Face-Crop
- **Postprocess**: `core/perception/face_attr_npu.py::parse_face_attributes()` + `analyze_face()`
- **Lazy-Loading**: Erst beim ersten erkannten Gesicht geladen
- **Caching**: Pro erkanntem Namen gecacht, nicht jedes Frame neu

### Outputs (1 NPU-Layer)

80 Werte (40 Attribute x 2: neg/pos-Paar) oder 40 Werte direkt.
Score = pos - neg Differenz. Positiv = Attribut vorhanden.

### 40 CelebA-Attribute (alle berechnet)

```
 0: 5_o_Clock_Shadow    1: Arched_Eyebrows     2: Attractive
 3: Bags_Under_Eyes     4: Bald                 5: Bangs
 6: Big_Lips            7: Big_Nose             8: Black_Hair
 9: Blond_Hair         10: Blurry              11: Brown_Hair
12: Bushy_Eyebrows     13: Chubby              14: Double_Chin
15: Eyeglasses         16: Goatee              17: Gray_Hair
18: Heavy_Makeup       19: High_Cheekbones     20: Male
21: Mouth_Slightly_Open 22: Mustache           23: Narrow_Eyes
24: No_Beard           25: Oval_Face           26: Pale_Skin
27: Pointy_Nose        28: Receding_Hairline   29: Rosy_Cheeks
30: Sideburns          31: Smiling             32: Straight_Hair
33: Wavy_Hair          34: Wearing_Earrings    35: Wearing_Hat
36: Wearing_Lipstick   37: Wearing_Necklace    38: Wearing_Necktie
39: Young
```

### Genutzt (5 von 40 Attributen)

| Attribut | Ableitung | PerceptionFrame-Feld |
|----------|-----------|---------------------|
| Male (#20) | `"M" if attrs["Male"] > 0 else "F"` | pf.gender |
| Young (#39) | Age-Range Primaer-Indikator | pf.age_range |
| Gray_Hair (#17) | Age-Range Sekundaer | pf.age_range |
| Smiling (#31) | Emotion: "Happy" | pf.emotion |
| Mouth_Slightly_Open (#21) | Emotion: "Surprised" | pf.emotion |

Age-Range Logik: `Young` + `Gray_Hair` + `Bald` + `Receding_Hairline` + `Bags_Under_Eyes` + `Double_Chin` → "15-20" | "25-32" | "38-43" | "48-53" | "60+"

Emotion Logik: `Smiling` + `Mouth_Slightly_Open` + `Narrow_Eyes` → "Happy" | "Surprised" | "Angry" | "Neutral"

### Ignoriert (35 von 40 Attributen)

| Attribut | Potential |
|----------|-----------|
| Eyeglasses | Brille erkannt — Personenbeschreibung, Sicherheit |
| Wearing_Hat | Kopfbedeckung — Indoor/Outdoor-Kontext |
| Heavy_Makeup | Erscheinungs-Kontext |
| Attractive | Subjektiv, aber: Gesichtssymmetrie-Proxy |
| 5_o_Clock_Shadow, Goatee, Mustache, No_Beard, Sideburns | Bartstil — Personen-Beschreibung |
| Black/Blond/Brown/Gray_Hair, Straight/Wavy_Hair, Bald, Bangs, Receding_Hairline | Haar — Personen-Beschreibung (teilweise in Age genutzt) |
| Big_Lips, Big_Nose, Pointy_Nose, Oval_Face, High_Cheekbones, Arched/Bushy_Eyebrows | Gesichtsmerkmale — koennten Face-ID ergaenzen |
| Wearing_Earrings, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie | Accessoires — Anlass/Kontext |
| Chubby, Double_Chin, Bags_Under_Eyes, Pale_Skin, Rosy_Cheeks | Koerpermerkmale (teilweise in Age genutzt) |
| Blurry | Bildqualitaets-Indikator |

---

## Datenfluesse — Was geht wohin

### PerceptionFrame.to_dict() — IPC Export

```python
{
  "timestamp":       float,
  "person_detected": bool,          # SCRFD oder YOLOv8m
  "person_count":    int,           # YOLOv8m oder 1 bei Face
  "distance":        str,           # YOLOv8m BBox-Flaeche
  "face_detected":   bool,          # SCRFD
  "face_count":      int,           # SCRFD
  "face_id":         str|None,      # ArcFace
  "face_confidence": float,         # SCRFD
  "face_similarity": float,         # ArcFace
  "gender":          str|None,      # Face Attr (Male)
  "age_range":       str|None,      # Face Attr (Young+...)
  "emotion":         str|None,      # Face Attr (Smiling+...)
  "pose_count":      int,           # Pose
  "pose_energy":     float,         # Pose (KP-Delta)
  "hand_detected":   bool,          # Hand Landmark
  "hand_gesture":    str|None,      # Hand Landmark → (broken) GestureDetector
  "head_pitch":      float|None,    # SCRFD Landmarks → Head Pose
  "head_yaw":        float|None,    # SCRFD Landmarks → Head Pose
  "objects":         List[Dict],    # YOLOv8m Nicht-Person-Klassen
  "active_models":   List[str],
  "inference_ms":    float
}
```

### Felder im Dataclass aber NICHT in to_dict()

| Feld | Typ | Kommentar |
|------|-----|-----------|
| face_bbox | tuple | BBox des ersten Gesichts — nicht exportiert |
| distance_ratio | float | Rohe BBox-Flaeche — nicht exportiert |

### Separate IPC-Kanaele

| Datei | Inhalt | Schreiber |
|-------|--------|-----------|
| /dev/shm/moloch_status.json | Service-Status, FPS, Thresholds, Perception-Scores | moloch_service._write_status_json() |
| /tmp/moloch_face_state.json | Name, Similarity, Emotion, Gender, Age, Head-Pose, Objects | moloch_service._write_face_state() |

---

## Bekannte Bugs / Design-Probleme

### 1. GestureDetector Anatomy Mismatch (KRITISCH)
- `GestureDetector.detect()` erwartet 17 COCO Body-Keypoints
- Bekommt 21 MediaPipe Hand-Landmarks
- Finger werden als Schultern/Knie interpretiert
- **Ergebnis**: Gesten-Erkennung funktioniert nicht korrekt

### 2. head_roll berechnet aber nicht gespeichert
- `estimate_head_pose()` liefert (pitch, yaw, roll)
- roll wird in draw_name() angezeigt
- PerceptionFrame hat kein head_roll Feld
- **Ergebnis**: Roll-Daten gehen verloren

### 3. face_bbox und distance_ratio nicht in to_dict()
- Beide Felder existieren im Dataclass
- Werden in _build_perception_frame() befuellt
- Fehlen in to_dict() → nicht ueber IPC verfuegbar
- **Ergebnis**: Downstream-Konsumenten haben keinen Zugriff

### 4. Multi-Face Last-Wins
- ArcFace laeuft fuer alle erkannten Gesichter
- _write_face_state() ueberschreibt bei jedem Face
- Letztes Gesicht in der Schleife "gewinnt"
- **Ergebnis**: Bei mehreren Personen unzuverlaessig

---

## Ungenutztes Potential — Top 5

### 1. Echte Finger-Gesten (Hand Landmark)
21 Finger-Keypoints liegen vor, werden aber an den falschen Detektor gefuettert.
Daumen-hoch, Peace, Zeigen — alles machbar ohne neues Modell.

### 2. 35 CelebA-Attribute (Face Attr)
Brille, Hut, Bart, Haarfarbe — alles schon berechnet und weggeworfen.
Personen-Beschreibung fuer Claude-Kontext waere trivial.

### 3. Body-Pose Aktivitaeten (Pose)
17 Keypoints werden nur zu einer Zahl (Energy) reduziert.
Sitzen/Stehen/Liegen, Arm-Gesten — ohne neues Modell machbar.

### 4. World-Space 3D Hande (Hand Landmark)
fc3-Layer mit 3D-Koordinaten wird komplett ignoriert.
Raeumliche Hand-Interaktion ohne Zusatz-Hardware moeglich.

### 5. Objekt-Raumkontext (YOLOv8m)
80 COCO-Klassen werden erkannt aber nur als Liste weitergereicht.
Semantischer Raumkontext ("Markus sitzt auf der Couch vor dem TV") waere ableitbar.

---

## Nicht geladene Modelle auf SSD2

| Modell | Groesse | Potential |
|--------|---------|-----------|
| yolov11m_h10.hef | 28 MB | Neuere YOLO-Generation, evtl. bessere Accuracy |
| yolov8m_pose_h10.hef | 29 MB | Groesseres Pose-Modell (m statt s) |
| yolov5n_seg_h10.hef | 3.5 MB | Instanz-Segmentierung (Pixel-Masken statt Boxen) |
| resnet_v1_50_h10.hef | 23 MB | Image Classification (ImageNet 1000 Klassen) |
