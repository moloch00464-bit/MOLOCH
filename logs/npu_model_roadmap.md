# M.O.L.O.C.H. — NPU Modell-Roadmap (Hailo-10H)
# Stand: 2026-04-01 | Recherchiert von Claude Sonnet 4.6
# Alle HEFs: hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/
# Basis-URL: https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/

---

## LEGENDE

Status: ✅ Integriert | 🔄 In Arbeit | ⬜ Ausstehend | ❌ Nicht relevant
Aufwand: ★☆☆ Einfach (1 Worker) | ★★☆ Mittel | ★★★ Komplex

---

## BEREITS INTEGRIERT

| Modell | Datei | Aufgabe | Status |
|--------|-------|---------|--------|
| yolov11m_h10.hef | tappas_pipeline.py | Person Detection 20 FPS | ✅ |
| scrfd_10g.hef | face_pipeline.py | Face Detection 640x640 | ✅ |
| arcface_mobilefacenet.hef | face_pipeline.py | Face Recognition | ✅ |
| face_attr_resnet_v1_18.hef | face_pipeline.py | Face Attributes 218x178 | ✅ |
| yolov8s_pose_h10.hef | pose_worker.py | Pose Estimation | ✅ |
| repvgg_a0_person_reid_512.hef | pose_worker.py | Person ReID | ✅ |
| hand_landmark_lite.hef | pose_worker.py | Hand Landmarks | ✅ |
| real_esrgan_x2.hef | super_res_worker.py | Super Resolution 2x | ✅ |
| CLIP (mehrere) | npu_extras.py | Text-Image Retrieval | ✅ |
| PaddleOCR | npu_extras.py | Text Erkennung | ✅ |
| Qwen2-VL-2B | npu_extras.py | Visual Language Model | ✅ |

---

## SOFORT INTEGRIERBAR (einfach, direkt relevant)

### 1. Low Light Enhancement — zero_dce ✅ INTEGRIERT (2026-04-01)
- **HEF**: zero_dce.hef
- **Input**: 400x600x3 (H=400, W=600) uint8
- **Output**: 400x600x3 float32 [0,1] aufgehellt
- **FPS**: 200 FPS (Batch=1) — kein Performance-Problem
- **Aufwand**: ★☆☆
- **Nutzen**: Nacht / schlechtes Licht → YOLO + FaceWorker sehen mehr
- **Integration**: tappas_pipeline._on_appsink_sample(), vor SHM-Write
- **Aktivierung**: Nur wenn mittlere Helligkeit < 80/255

### 2. Person Attribute Detection — person_attr_resnet_v1_18
- **HEF**: person_attr_resnet_v1_18.hef
- **Input**: 224x224x3
- **FPS**: 2451 FPS (trivial schnell)
- **Aufwand**: ★☆☆
- **Nutzen**: Kleidungsfarbe, Alter, Rucksack, Hut, Geschlecht pro Person
- **Integration**: PersonAttrWorker in pose_worker.py, wird auf Person-Crop angewendet
- **Trigger**: Gleicher Frame-Takt wie PoseWorker (every_n_frames=3)
- **Anwendung**: CoreIntegrator-Kontext ("Markus kommt — rotes Shirt")
- **Download**: person_attr_resnet_v1_18.hef

### 3. Low Light Enhancement v2 — zero_dce_pp
- **HEF**: zero_dce_pp.hef
- **Input**: 400x600x3
- **FPS**: 96 FPS (kleineres Modell)
- **Aufwand**: ★☆☆
- **Nutzen**: Alternative zu zero_dce, kompakter (0.02M statt 0.21M Parameter)
- **Status**: Erst zero_dce testen, dann bei Bedarf wechseln

---

## MITTELFRISTIG (nächste Gates)

### 4. Aktivitätserkennung — r3d_18
- **HEF**: r3d_18.hef
- **Input**: Video-Clips (mehrere Frames), Kinetics-400
- **FPS**: 55 FPS (Batch=1)
- **Aufwand**: ★★☆
- **Nutzen**: Erkennt ob Person sitzt, steht, geht, läuft, tanzt…
- **Anwendung**: Tension-System ("Markus sitzt seit 3h → Spannung sinkt")
- **Besonderheit**: Braucht Frame-Buffer (5-16 aufeinanderfolgende Frames)

### 5. Zero-Shot Objekterkennung — yolo_world_v2s
- **HEF**: yolo_world_v2s.hef
- **Input**: 640x640x3
- **FPS**: 45 FPS
- **Aufwand**: ★★☆
- **Nutzen**: Suche nach beliebigen Objekten ohne Training ("suche Flasche")
- **Besonderheit**: Braucht Text-Encoder (Prompt → Embedding → YOLO World)
- **Anwendung**: Autonome Suche, Voice-Command "Wo ist mein X?"

### 6. Tiefenschätzung (Monocular) — scdepthv3
- **HEF**: scdepthv3.hef
- **Input**: 256x320x3
- **FPS**: ~30
- **Aufwand**: ★★☆
- **Nutzen**: Wie weit ist Person weg? → Präzisere PTZ-Steuerung
- **Anwendung**: Tracker-Entscheidungen ("Person 2m entfernt → nah-Modus")

### 7. Semantische Segmentierung — deeplab_v3_mobilenet_v2_wo_dilation
- **HEF**: deeplab_v3_mobilenet_v2_wo_dilation.hef
- **Input**: 513x513x3
- **FPS**: 375 FPS
- **Aufwand**: ★★☆
- **Nutzen**: Raum-Verständnis (Boden, Wand, Person, Stuhl, Sofa…)
- **Anwendung**: Night Cycle Szenebeschreibung, Kontext-Awareness

---

## LANGFRISTIG (Gate 5+)

### 8. SigLIP2 Text-Encoder — siglip2_b_16_text_encoder
- **HEF**: siglip2_b_16_text_encoder.hef
- **FPS**: 34 FPS
- **Retrieval@10**: 97.1% (besser als CLIP ~86%)
- **Aufwand**: ★★☆
- **Nutzen**: CLIP-Ersatz in npu_extras.py — bessere Bild-Text-Ähnlichkeit
- **Anwendung**: Qwen2-VL Unterstützung, Scene Understanding

### 9. unet_mobilenet_v2 (Schnelle Maske)
- **HEF**: unet_mobilenet_v2.hef
- **Input**: 256x256x3
- **FPS**: 693 FPS
- **Aufwand**: ★★☆
- **Nutzen**: Person-Maske für Hintergrundentfernung, Fokus-Effekte

### 10. Stereo Depth — stereonet (NICHT RELEVANT)
- Braucht 2 synchrone Kameras (MOLOCH hat nur eine)

### 11. fast_depth (Monocular, leicht)
- **HEF**: fast_depth.hef
- **Input**: 224x224x3
- **Aufwand**: ★☆☆
- **Nutzen**: Einfachere Alternative zu scdepthv3

---

## NICHT RELEVANT FÜR MOLOCH

| Modell | Grund |
|--------|-------|
| stereonet | Braucht 2 Kameras |
| fcn8_resnet_v1_18 | 1920x1024 Input — zu groß für Pi5 |
| stdc1 | 1920x1024 — zu groß |
| Video classification r3d_18 | Interessant aber Gates 5+ |
| Alle Classification-Modelle | Kein Use-Case |
| Image Denoising (diverse) | zero_dce macht das besser |

---

## DOWNLOAD-URLS (alle v5.2.0, hailo10h)

```
BASE=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h

Sofort:
${BASE}/zero_dce.hef
${BASE}/zero_dce_pp.hef
${BASE}/person_attr_resnet_v1_18.hef

Mittelfristig:
${BASE}/r3d_18.hef
${BASE}/yolo_world_v2s.hef
${BASE}/scdepthv3.hef
${BASE}/fast_depth.hef
${BASE}/deeplab_v3_mobilenet_v2_wo_dilation.hef

Langfristig:
${BASE}/siglip2_b_16_text_encoder.hef
${BASE}/unet_mobilenet_v2.hef
```
