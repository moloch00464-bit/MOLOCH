# Hailo-10H 8GB Audit — M.O.L.O.C.H.

**Datum:** 2026-02-22
**System:** Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)
**HailoRT:** 5.1.1 (h10-hailort)

---

## 1. Hardware-Identifikation

```
Chip:         Hailo-10H AI Processor (PCI 1e60:45c4)
PCIe:         0001:01:00.0 (Gen 2 x1 via FFC)
Firmware:     5.1.1 (release, app)
Architektur:  HAILO10H
Treiber:      hailo1x 5.1.1 (out-of-tree)
RAM:          8 GB LPDDR4X (on-board, NICHT Pi-RAM)
TOPS:         40 (INT4)
Max Power:    3W
```

### AI HAT+ 2 vs AI HAT+ (alt)

| Feature          | AI HAT+ v1 (alt)        | AI HAT+ 2 (Moloch)       |
|------------------|--------------------------|---------------------------|
| Chip             | Hailo-8L (13T) / 8 (26T)| Hailo-10H (40 TOPS)      |
| On-Board RAM     | KEINER (nutzt Pi-RAM)    | 8 GB LPDDR4X              |
| LLM Support      | Nein                     | Ja (Qwen, DeepSeek, etc.) |
| Whisper Support  | Nur Hailo-8 HEF          | Ja (H10 HEF, 131 MB)     |
| VLM Support      | Nein                     | Ja (Qwen2-VL-2B)         |

**Der entscheidende Vorteil**: Modelle liegen im 8 GB On-Board RAM, NICHT im Pi-RAM.
Kein PCIe-Bandbreiten-Engpass, kein Pi-RAM-Verbrauch fuer AI-Modelle.

---

## 2. HailoRT Versionen — DUAL INSTALL PROBLEM

```
h10-hailort              5.1.1    (fuer Hailo-10H)
hailort                  4.23.0   (fuer Hailo-8/8L, ALT)
python3-h10-hailort      5.1.1-1
hailo-tappas-core        5.1.0
hailo-apps               25.12.0  (editable, SYMLINK BROKEN)
```

**WARNUNG**: Zwei HailoRT Versionen parallel installiert (5.1.1 + 4.23.0).
Kann zu Konflikten fuehren. `hailort 4.23.0` sollte deinstalliert werden.

---

## 3. KRITISCH: NTFS SSD2 NICHT GEMOUNTET

### Problem
```
# dmesg:
ntfs3(sda2): volume is dirty and "force" flag is not set!

# Ergebnis:
/mnt/moloch-data/ ist LEER
```

### Auswirkungen (Kaskade)
```
NTFS dirty → Mount fail → /mnt/moloch-data leer
  → Symlink ~/hailo-apps → /mnt/moloch-data/hailo/repos/hailo-apps BROKEN
  → "No module named 'hailo_apps.python.core.common'"
  → unified_pipeline kann nicht laden
  → NPU Vision-Modelle NICHT ladbar ueber hailo-apps Pipeline

  → Symlink ~/.hailo → /mnt/moloch-data/hailo/config BROKEN
  → "Failed to create directory /home/molochzuhause/.hailo/hailort"
  → hailortcli Warnungen bei jedem Aufruf
```

### Fix
```bash
# Option A: Force-Mount (schnell, liest dirty FS)
sudo mount -t ntfs3 -o uid=1000,gid=1000,force /dev/sda2 /mnt/moloch-data

# Option B: ntfsfix (repariert dirty flag, sicherer)
sudo ntfsfix /dev/sda2
sudo mount /mnt/moloch-data

# Option C: fstab anpassen (permanent)
# Alt:  UUID=F4BE3BC4BE3B7E64 /mnt/moloch-data ntfs3 uid=1000,gid=1000,nofail 0 0
# Neu:  UUID=F4BE3BC4BE3B7E64 /mnt/moloch-data ntfs3 uid=1000,gid=1000,nofail,force 0 0
```

**EMPFEHLUNG**: Option B (ntfsfix) + fstab mit `force` Flag als Absicherung.

---

## 4. HEF-Modell-Inventar

### Auf SSD1 (immer verfuegbar — /usr/local/hailo/resources/models/hailo10h/)

| Modell                         | Disk   | Typ                | Status      |
|--------------------------------|--------|--------------------|-------------|
| scrfd_10g.hef                  | 5.8 MB | Face Detection     | AKTIV       |
| arcface_mobilefacenet.hef      | 2.6 MB | Face Recognition   | AKTIV       |
| Whisper-Base.hef               | 131 MB | Speech-to-Text     | VORHANDEN   |
| Qwen2.5-1.5B-Instruct.hef     | 2.2 GB | LLM Text           | VORHANDEN   |

### Auf SSD1 (/usr/share/hailo-models/ — H10-kompatibel)

| Modell                         | Disk   | Typ                | Status      |
|--------------------------------|--------|--------------------|-------------|
| yolov8m_h10.hef                | 21 MB  | Person Detection   | AKTIV       |
| yolov8s_pose_h10.hef           | 14 MB  | Pose/Keypoints     | verfuegbar  |
| yolov8m_pose_h10.hef           | 28 MB  | Pose/Keypoints     | verfuegbar  |
| yolov11m_h10.hef               | 27 MB  | Object Detection   | verfuegbar  |
| yolov5n_seg_h10.hef            | 3.4 MB | Segmentation       | verfuegbar  |
| resnet_v1_50_h10.hef           | 23 MB  | Classification     | verfuegbar  |

### Gesamt HEF-Groesse (alle H10): ~2.5 GB Disk

---

## 5. NPU-RAM Kapazitaetsanalyse (8 GB)

### Aktuelle Nutzung (Dual-Slot System)
```
Slot 1: scrfd_10g (5.8 MB) + arcface (2.6 MB)  = ~8.4 MB
Slot 2: yolov8m (21 MB)                          = ~21 MB
Swap:   yolov8s_pose (14 MB) — wird bei Bedarf getauscht
                                          Gesamt: ~43 MB
```

### Theoretische Kapazitaet
```
8 GB NPU RAM verfuegbar
- Alle 4 Vision-Modelle gleichzeitig:            ~43 MB  (0.5%)
- + Whisper-Base:                                 ~174 MB (2.1%)
- + Qwen2.5-1.5B:                               ~2.4 GB (29%)
- Verbleibend:                                   ~5.6 GB (70%)
```

**Ergebnis: Die 8 GB sind MASSIV unterausgelastet.**
Aktuell werden <1% des NPU-RAM genutzt.

### ABER: Limitierung ist NICHT der RAM

Das Limit ist der **VDevice Scheduler**, nicht der RAM:

| Kombination              | Status          | Methode                    |
|--------------------------|-----------------|----------------------------|
| Vision + Vision          | FUNKTIONIERT    | ROUND_ROBIN VDevice        |
| Vision + Whisper         | SEQUENZIELL     | acquire/release Handoff    |
| Whisper + LLM            | NICHT MOEGLICH  | Physical Device Lock       |
| LLM + Vision             | PROBLEMATISCH   | Segfaults berichtet        |

**Fazit**: Alle Vision-Modelle koennen gleichzeitig geladen bleiben.
GenAI (Whisper/LLM) braucht weiterhin Handoff, aber KEIN Swap noetig.

---

## 6. Whisper auf NPU — Status

### Ist-Zustand
- Whisper-Base.hef (131 MB) VORHANDEN auf SSD1
- `hailo_platform.genai.Speech2Text` API VERFUEGBAR
- Code in `core/speech/hailo_whisper.py` IMPLEMENTIERT
- ABER: `hailo_apps` Import BROKEN (SSD2 nicht gemountet)
- Aktuell: CPU Fallback (faster-whisper tiny, int8)

### Was fehlt fuer NPU-Whisper
1. SSD2 mounten (ntfsfix) → hailo-apps Symlink reparieren
2. ODER: hailo_whisper.py so anpassen, dass es OHNE hailo-apps funktioniert
   (die HEF-Pfade direkt setzen statt ueber resolve_hef_path)
3. VDevice Handoff funktioniert bereits via HailoManager

### NPU-Whisper Vorteile
- CPU-Entlastung: 0% CPU statt ~30% fuer Whisper-tiny
- Schneller: NPU-Whisper-Base ~2-3x schneller als CPU-tiny
- Bessere Qualitaet: Base > Tiny (groesseres Modell, besseres Deutsch)
- RAM-Sparsamkeit: Modell liegt im NPU-RAM, nicht im Pi-RAM

---

## 7. LLM auf NPU — Bewertung

### Qwen2.5-1.5B-Instruct
- HEF vorhanden: 2.2 GB
- Geschwindigkeit: ~6.5 tok/s auf NPU
- Vergleich: Pi 5 CPU schafft ~9 tok/s (schneller!)
- API: `hailo_platform.genai.LLM`

### Empfehlung
**NICHT sinnvoll als Ersatz fuer Claude API.**
- Claude API: Bessere Qualitaet, schnellere Antworten (Netzwerk < 2s)
- Qwen NPU: Schlechte Qualitaet bei 1.5B, nur 6.5 tok/s
- ABER: Offline-Fallback wenn Internet ausfaellt → bedingt sinnvoll

---

## 8. Umsetzungsplan

### Phase 1: NTFS fixen (SOFORT, 5 Minuten)
```bash
sudo ntfsfix /dev/sda2
sudo mount /mnt/moloch-data
# fstab: force Flag hinzufuegen
```
**Ergebnis**: hailo-apps funktioniert wieder, NPU Vision-Pipeline laeuft.

### Phase 2: Vision ohne Swap (nach Phase 1)
- Alle 4 Vision-Modelle permanent in NPU laden
- Dual-Slot-Limit aufheben → kein Swap-Overhead mehr
- `perception_engine.py` anpassen: force_models() vereinfachen

### Phase 3: Whisper auf NPU (nach Phase 1)
- `hailo_whisper.py` reparieren: HEF-Pfad direkt setzen
  `/usr/local/hailo/resources/models/hailo10h/Whisper-Base.hef`
- hailo_apps Abhaengigkeit optional machen
- VDevice Handoff beibehalten (acquire/release)

### Phase 4: Qwen Offline-Fallback (optional, niedrige Prio)
- Nur aktivieren wenn Internet/Claude nicht erreichbar
- `hailo_platform.genai.LLM` mit Qwen2.5-1.5B
- Schlechtere Qualitaet aber funktioniert offline

### Phase 5: Aufraeumen
- `hailort 4.23.0` (alt, Hailo-8) deinstallieren
- hailo-apps als richtiges Package installieren (nicht editable Symlink)

---

## 9. Offene Fragen

1. **Warum ist SSD2 dirty?** Unsauber abgeschaltet? Windows hat drauf geschrieben?
2. **hailort 4.23.0 Konflikte?** Koennte die parallele Installation den H10 stoeren?
3. **VDevice HAILO_OUT_OF_PHYSICAL_DEVICES**: Tritt das nur auf weil hailo-apps fehlt,
   oder gibt es ein echtes Device-Lock Problem?

---

## TL;DR

| Punkt                     | Status                          | Aktion                        |
|---------------------------|---------------------------------|-------------------------------|
| Hardware                  | Hailo-10H, 8 GB, 40 TOPS       | OK                            |
| NTFS SSD2                 | NICHT GEMOUNTET (dirty)         | ntfsfix + force mount         |
| hailo-apps                | BROKEN (Symlink tot)            | SSD2 mounten                  |
| NPU RAM Nutzung           | <1% von 8 GB                   | Alle Modelle laden            |
| Whisper HEF               | VORHANDEN, nicht aktiviert      | HEF-Pfad direkt setzen        |
| Qwen LLM HEF             | VORHANDEN, 6.5 tok/s            | Nur als Offline-Fallback      |
| Vision Dual-Slot          | Unnoetig mit 8 GB               | Alle 4 permanent laden        |
| HailoRT Versionskonflikt  | 5.1.1 + 4.23.0 parallel         | 4.23.0 deinstallieren         |
