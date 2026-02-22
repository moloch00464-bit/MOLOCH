# M.O.L.O.C.H. Modell-Auswahl Diagnose

**Datum:** 2026-02-22
**Hardware:** Hailo-10H (8 GB NPU-RAM, 40 TOPS)
**HailoRT:** 5.1.1
**Zoo:** 56 HEF-Modelle, 7.1 GB gesamt auf SSD2

---

## 1. IST-ZUSTAND

### Aktiv im Service (MODEL_PATHS in moloch_service.py)

| Modell | Datei | Groesse | FPS | Funktion |
|--------|-------|---------|-----|----------|
| SCRFD 10G | scrfd_10g.hef | 5.8 MB | ~47 | Face Detection |
| ArcFace MobileFaceNet | arcface_mobilefacenet.hef | 2.6 MB | ~498 | Face Recognition |
| YOLOv8m | yolov8m_h10.hef | 21 MB | ~39 | Person Detection |
| Hand Landmark Lite | hand_landmark_lite.hef | 1.3 MB | — | Hand Gesture |
| **Summe aktiv** | | **30.7 MB** | | |

### Auf Disk aber NICHT geladen

| Modell | Datei | Groesse | Status |
|--------|-------|---------|--------|
| YOLOv8s Pose | yolov8s_pose_h10.hef | 14 MB | Code existiert, nicht verdrahtet |
| YOLOv8m Pose | yolov8m_pose_h10.hef | 28 MB | Code existiert, nicht verdrahtet |
| YOLOv5n Seg | yolov5n_seg_h10.hef | 3.4 MB | Kein Code vorhanden |
| YOLOv11m | yolov11m_h10.hef | 27 MB | Nicht verwendet |
| ResNet-50 | resnet_v1_50_h10.hef | 23 MB | Nicht verwendet |

### Whisper (STT auf NPU)

| Modell | Pfad | Groesse | Status |
|--------|------|---------|--------|
| Whisper-Base | /usr/local/hailo/resources/models/hailo10h/Whisper-Base.hef | 130.5 MB | AKTIV, lazy-load |

---

## 2. ANALYSE PRO GEWUENSCHTEM MODELL

### 2.1 WHISPER — Upgrade auf Small/Medium?

**Ergebnis: NICHT MOEGLICH.**

- `hailo-download-resources --list-models` zeigt NUR **Whisper-Base** fuer Hailo-10H
- Kein Whisper-Small, kein Whisper-Medium als HEF verfuegbar
- Hailo hat nur Whisper-Base fuer die NPU kompiliert (Stand hailo-apps 25.12.0)
- Der Zoo enthaelt ebenfalls nur Whisper-Base.hef (130.5 MB)

**Optionen:**
1. **Whisper-Base auf NPU beibehalten** (aktueller Zustand) — schnell, aber deutsch mittelmässig
2. **Whisper-Small auf CPU** via faster-whisper/whisper.cpp — besser fuer Deutsch, aber ~2-3s Latenz auf Pi5 ARM
3. **Warten** auf Hailo Model Zoo Update mit Whisper-Small HEF

**Empfehlung:** Whisper-Base NPU beibehalten. Die Latenz-Vorteile der NPU ueberwiegen die etwas bessere Deutsch-Qualitaet von Small auf CPU. Wenn Deutsch-Qualitaet kritisch wird: Hybrid-Ansatz (NPU fuer schnelle Erkennung, CPU-Fallback fuer lange Texte).

---

### 2.2 POSE ESTIMATION — Koerperhaltung erkennen

**Ergebnis: SOFORT AKTIVIERBAR!**

Alles ist da:
- **Modell:** yolov8s_pose_h10.hef (14 MB) auf Disk ✓
- **Postprocessing:** `decode_yolov8_pose()` in hailo_postprocess.py ✓
- **Zeichenfunktion:** `draw_poses()` in hailo_postprocess.py ✓
- **17 COCO Keypoints:** Nase, Augen, Ohren, Schultern, Ellbogen, Handgelenke, Hueften, Knie, Knoechel ✓
- **Skeleton-Rendering:** Verbindungslinien zwischen Keypoints ✓

**Was fehlt:**
1. Eintrag in `MODEL_PATHS` in moloch_service.py
2. Inference-Aufruf in der Hauptschleife
3. Eintrag in `PerceptionEngine.ALL_MODELS`
4. Interpretation der Pose (stehen/sitzen/liegen) — muss geschrieben werden

**Modell-Wahl:**
| Variante | Groesse | Geschaetzte FPS | Empfehlung |
|----------|---------|-----------------|------------|
| yolov8s_pose | 14 MB | ~36 FPS | **JA — schnell genug** |
| yolov8m_pose | 28 MB | ~25 FPS | Genauer, aber langsamer |

**Empfehlung:** yolov8s_pose_h10.hef — 14 MB, schnell genug fuer Echtzeit.

---

### 2.3 OBJEKT-ERKENNUNG (erweitert) — Moebel, TV, Couch

**Ergebnis: BEREITS AKTIV — nur Label-Map fehlt!**

YOLOv8m erkennt ALLE 80 COCO-Klassen. Das Modell sieht bereits Couch, TV, Stuhl etc.
Aber: Der Code filtert auf `class_id=0` (Person) und ignoriert alles andere.

**COCO-Klassen die fuer MOLOCH relevant sind (von 80):**

| ID | Klasse | Relevanz |
|----|--------|----------|
| 0 | person | ✓ Aktiv |
| 56 | chair | Markus sitzt |
| 57 | couch | Markus auf Couch |
| 58 | potted plant | Deko |
| 59 | bed | Markus liegt |
| 60 | dining table | Esstisch |
| 62 | tv | Fernseher an/aus |
| 63 | laptop | Arbeitsplatz |
| 64 | mouse | Arbeitsplatz |
| 65 | remote | Fernbedienung |
| 67 | cell phone | Handy |
| 73 | book | Liest |
| 41 | cup | Trinkt |
| 39 | bottle | Flasche |
| 15 | cat | Haustier? |
| 16 | dog | Haustier? |

**Was fehlt:**
1. Vollstaendige COCO-Label-Map (80 Klassen) in hailo_postprocess.py
2. Filter auf relevante Klassen statt nur class_id=0
3. Annotation mit richtigem Namen statt "class_X"
4. Optional: Nur relevante Klassen anzeigen (nicht alle 80)

**Empfehlung:** Kein neues Modell noetig! Nur Code-Aenderung in `decode_yolov8_nms()` und `draw_persons()` → `draw_detections()`.

---

### 2.4 SEGMENTATION — Pixel-genaue Erkennung

**Ergebnis: MODELL VORHANDEN, ABER KEIN CODE.**

Verfuegbare Seg-Modelle:

| Modell | Groesse | Klassen | Im aktiven Verzeichnis? |
|--------|---------|---------|-------------------------|
| yolov5n_seg_h10.hef | 3.4 MB | 80 COCO | ✓ Bereits auf Disk! |
| yolov8n_seg (Zoo) | 5.8 MB | 80 COCO | Nur im Zoo |
| yolov8s_seg (Zoo) | 15.5 MB | 80 COCO | Nur im Zoo |
| yolov8m_seg (Zoo) | 27.8 MB | 80 COCO | Nur im Zoo |

**Was fehlt — VIEL Arbeit:**
1. Postprocessing-Funktion `decode_yolov8_seg()` oder `decode_yolov5_seg()` — muss geschrieben werden
2. Mask-Rendering (halbtransparente Overlays auf Frame) — muss geschrieben werden
3. Integration in Inference-Loop
4. Performance-Tuning (Masks sind teuer zu rendern)

**Nutzen vs. Aufwand:**
- Bounding Boxes (YOLO Detection) reichen fuer 90% der Anwendungsfaelle
- Segmentation braucht man wenn: Objekte ueberlappen, exakte Kontur wichtig, oder fuer Szenen-Verstaendnis
- Fuer MOLOCH aktuell: **Bounding Boxes reichen**

**Empfehlung:** Niedrige Prioritaet. Erst Pose + erweiterte COCO-Labels aktivieren. Segmentation spaeter wenn Szenen-Verstaendnis gebraucht wird.

---

## 3. NPU-RAM BUDGET (Hailo-10H, 8 GB)

### Szenario: Alle gewuenschten Modelle gleichzeitig

| Modell | RAM (HEF-Groesse ≈ NPU-RAM) | Status |
|--------|------------------------------|--------|
| SCRFD 10G | 5.8 MB | Aktiv |
| ArcFace MobileFaceNet | 2.6 MB | Aktiv |
| YOLOv8m (80 COCO) | 21 MB | Aktiv |
| Hand Landmark Lite | 1.3 MB | Aktiv |
| YOLOv8s Pose | 14 MB | **NEU** |
| YOLOv5n Seg | 3.4 MB | **NEU (optional)** |
| **Summe Vision** | **48.1 MB** | |
| Whisper-Base (on-demand) | 130.5 MB | Lazy-load |
| **Summe ALLES** | **178.6 MB** | |

**8192 MB verfuegbar → 178.6 MB belegt → 2.2% Auslastung**

**Ergebnis: LOCKER. Faktor 45x Headroom.** Selbst wenn alle Modelle gleichzeitig geladen sind, wird nicht mal 3% des NPU-RAM benoetigt.

### Pi5 System-RAM (4 GB)

| Ressource | Aktuell |
|-----------|---------|
| Total | 3992 MB |
| Used | 1878 MB |
| Available | 2114 MB |
| Swap used | 377 MB von 2048 MB |

System-RAM ist knapper (53% genutzt), aber fuer die HEF-Lade-Operationen reicht es.

---

## 4. EMPFOHLENE REIHENFOLGE

### Phase 1: Quick Wins (nur Config/Code, kein neues Modell)

**A) COCO-Label-Map aktivieren (YOLOv8m)**
- Aufwand: ~30 Min Code
- Nutzen: Erkennt sofort Couch, TV, Stuhl, Bett, etc.
- Aenderungen: hailo_postprocess.py (Label-Map + Filter), moloch_service.py (Annotation)
- Risiko: Sehr gering

### Phase 2: Pose Estimation aktivieren

**B) YOLOv8s Pose laden und verdrahten**
- Aufwand: ~1-2h Code
- Nutzen: Erkennt Koerperhaltung (stehen, sitzen, liegen, Arme heben)
- Aenderungen:
  - moloch_service.py: MODEL_PATHS + Inference-Aufruf
  - perception_engine.py: ALL_MODELS + Scoring
  - Neue Logik: Pose-Interpretation (Keypoint-Analyse → Zustand)
- Risiko: Gering (additives Feature)

### Phase 3: Spaeter / Optional

**C) Segmentation — nur bei Bedarf**
- Aufwand: ~4-6h Code (Postprocessing + Rendering komplett neu)
- Nutzen: Marginal gegenueber Detection + Pose
- Empfehlung: Erstmal weglassen

**D) Whisper-Upgrade — warten auf Hailo**
- Aktuell keine bessere Option als Whisper-Base auf NPU
- Bei schlechter Deutsch-Erkennung: faster-whisper-small auf CPU als Hybrid

---

## 5. BONUS: Weitere interessante Zoo-Modelle

| Modell | Groesse | Potential |
|--------|---------|-----------|
| OCR (PaddleOCR) | 14 MB | Text auf Bildschirm/Buechern lesen |
| Depth (SCDepthV3) | 14.4 MB | Abstand zu Objekten schaetzen |
| CLIP ResNet-50x4 | 62.6 MB | "Beschreibe was du siehst" (Vision-Language) |
| Qwen2-VL 2B | 2.2 GB | Vision-Language Model (NPU-LLM!) |
| YOLOv11m | 25.8 MB | Neuere YOLO-Generation (evtl. besser als v8m) |

---

## 6. ZUSAMMENFASSUNG

```
WHISPER:       Kein Upgrade moeglich (nur Base als HEF)     → Beibehalten
POSE:          Sofort aktivierbar (Code + Modell da)         → Phase 2
COCO-LABELS:   Sofort aktivierbar (nur Code-Aenderung)       → Phase 1
SEGMENTATION:  Modell da, aber viel Code-Arbeit              → Spaeter
NPU-RAM:       178 MB von 8192 MB = 2.2% — KEIN Problem     → Alles passt
```
