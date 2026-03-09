# NPU Drei-Schichten-Architektur
**Stand: 2026-03-09 | M.O.L.O.C.H. Gate 1+**

---

## Überblick

```
┌─────────────────────────────────────────────────────┐
│  SCHICHT 3: Kreativ/Expression    (LLM Reasoning)   │
├─────────────────────────────────────────────────────┤
│  SCHICHT 2: Traum/Assoziation     (Memory/Context)  │
├─────────────────────────────────────────────────────┤
│  SCHICHT 1: Wach/Perzeption       (Vision 20 FPS)   │
└─────────────────────────────────────────────────────┘
```

---

## Schicht 1 — Wach/Perzeption (Höchste Priorität)
**Hardware: Hailo-10H NPU (exklusiv)**

- YOLO v8m → Person-Detection, 640×640, 20 FPS
- SCRFD 10g → Face-Detection, 640×640, Letterbox
- ArcFace MobileFaceNet → Face-ID, 112×112
- **Permanent geladen, keine Unterbrechung**
- Exklusiver VDevice-Zugriff über GStreamer/TAPPAS
- RAM: ~200 MB Pi5-RAM, ~30 MB NPU-RAM

---

## Schicht 2 — Traum/Assoziation (CPU, kein NPU)
**Hardware: Pi5 CPU + SSD2**

- Qdrant Vektordatenbank (Langzeit-Gedächtnis)
- LongTermMemory (identity.json, facts.json, conversations/)
- Kontext-Aggregation, Personen-Matching, Event-History
- **BEWUSST auf CPU** → kein VDevice-Konflikt mit Schicht 1
- RAM: ~150 MB für Qdrant + Memory-Cache
- Pfad: `/mnt/moloch-data/memory/`

---

## Schicht 3 — Kreativ/Expression (LLM Reasoning)
**Hardware: Situationsabhängig**

| Modus | LLM | NPU-Status | Latenz |
|-------|-----|------------|--------|
| Online (normal) | DeepSeek API | Vision läuft weiter | ~500ms |
| Fallback 1 | Claude API | Vision läuft weiter | ~800ms |
| Fallback 2 | Qwen2.5 lokal (hailo-ollama) | Vision pausiert 5-8s | ~2s |
| Nacht-Zyklus | Qwen2.5 lokal | Vision AUS, volle NPU | max throughput |

### Fallback-Kette
```
DeepSeek → Claude → Qwen2.5 lokal → Stille
```
- Stille = kein Crash, kein Fehler, nur keine Antwort
- Prüfung über `MOLOCH_OFFLINE` Flag + `/moloch/health`

---

## VDevice-Konflikt: Warum Schicht 2 auf CPU
- H10 erlaubt **nur EINEN VDevice gleichzeitig**
- Schicht 1 hält VDevice permanent (TAPPAS Pipeline)
- Embedding-Berechnungen (Qdrant) auf CPU sind schnell genug (~10ms)
- Qwen2.5 für Schicht 3 **erfordert VDevice** → Vision muss kurz pausieren
- **Nie Schicht 1 + Schicht 3 gleichzeitig auf NPU!**

---

## Nacht-Zyklus (Gate 5, geplant)
- Trigger: 02:00 Uhr oder manuell
- Vision Pipeline stoppt (TAPPAS teardown)
- Qwen2.5 bekommt volle NPU (8GB LPDDR4X)
- Tages-Verarbeitung: neue Embeddings, Memory-Konsolidierung
- Morgens: Vision Pipeline startet neu, Qwen2.5 entladen
