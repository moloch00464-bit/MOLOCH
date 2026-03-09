# Nanobot Flow Tracer — Spezifikation
**Stand: 2026-03-09 | M.O.L.O.C.H. Gate 1**

---

## Konzept

Ein Tracer-Event fließt wie **Kontrastmittel** einmalig durch die gesamte Pipeline und misst Latenz pro Station. Ergebnis: Engpass-Karte mit Millisekunden pro Stage.

```
RTSP → GStreamer → TAPPAS → YOLO → SCRFD → ArcFace → EventBus → Bridge → Tracker → PTZ
  ↑         ↑         ↑       ↑       ↑        ↑          ↑        ↑        ↑       ↑
 t0        t1        t2      t3      t4       t5         t6       t7       t8      t9
```

---

## 9 Stationen

| # | Station | Messung | Gesund |
|---|---------|---------|--------|
| 0 | RTSP Eingang | Frame-Empfang → GStreamer | < 5ms |
| 1 | GStreamer Decode | Decode + Resize | < 10ms |
| 2 | TAPPAS Preprocessing | Letterbox + Normalisierung | < 5ms |
| 3 | YOLO Inference | Person-Detection auf NPU | < 20ms |
| 4 | SCRFD Inference | Face-Detection auf NPU | < 15ms |
| 5 | ArcFace Inference | Embedding auf NPU | < 10ms |
| 6 | EventBus Dispatch | Event → Subscriber | < 2ms |
| 7 | Action Bridge | FSM-Zustandswechsel | < 5ms |
| 8 | Tracker Update | Position + Kalman | < 5ms |
| 9 | PTZ Command | ONVIF → Kamera ACK | < 100ms |

**Total gesund: < 170ms (≈ 6 FPS Latenz-Budget)**

---

## Output-Format

```json
{
  "trace_id": "nano_20260309_142301",
  "trigger": "manual",
  "total_ms": 143,
  "bottleneck": "PTZ Command",
  "stations": {
    "rtsp": {"ms": 4, "status": "OK"},
    "gstreamer": {"ms": 8, "status": "OK"},
    "tappas": {"ms": 3, "status": "OK"},
    "yolo": {"ms": 18, "status": "OK"},
    "scrfd": {"ms": 12, "status": "OK"},
    "arcface": {"ms": 9, "status": "OK"},
    "eventbus": {"ms": 1, "status": "OK"},
    "bridge": {"ms": 4, "status": "OK"},
    "tracker": {"ms": 3, "status": "OK"},
    "ptz": {"ms": 81, "status": "SLOW"}
  },
  "broken_paths": []
}
```

---

## Integration

### Diagnostics API
```
GET /moloch/flow_trace
→ Startet einmaligen Trace, gibt JSON zurück (sync, ~200ms)

GET /moloch/flow_trace?async=1
→ Trace ID zurück, Ergebnis per GET /moloch/flow_trace/{id}
```

Bereits vorhanden: `scripts/nanobot_trace.py` + Route in Flask-API

### Supervisor-Popup
- Button "Nanobot Trace" → ruft `/moloch/flow_trace` auf
- Ergebnis: Tabelle mit Farben (grün/gelb/rot per Station)
- Rot = > 2× Sollwert, Gelb = > 1.5× Sollwert
- Engpass wird fett markiert

---

## AGENT_TOOLBOX Eintrag

```json
"nanobot": {
  "name": "Nanobot Flow Tracer Agent",
  "prompt": "Du injizierst ein Tracer-Event in die MOLOCH Pipeline und misst Latenz pro Station (RTSP→GStreamer→TAPPAS→YOLO→SCRFD→ArcFace→EventBus→Bridge→Tracker→PTZ). Identifiziere den Engpass. Melde unterbrochene Pfade. Kein Code ändern, nur messen und berichten.",
  "territorium": ["scripts/nanobot_trace.py", "diagnostics API /moloch/flow_trace"],
  "outputs": ["latency_per_station_ms", "bottleneck", "broken_paths", "total_ms"]
}
```

---

## Implementierungs-Status

| Komponente | Status |
|-----------|--------|
| `scripts/nanobot_trace.py` | ✅ Vorhanden |
| `GET /moloch/flow_trace` | ✅ Vorhanden |
| Supervisor-Popup Integration | 📋 TODO (Gate 1 T09/T10) |
| AGENT_TOOLBOX Eintrag | 📋 TODO (diese Spec umsetzen) |
