---
name: moloch-status
description: Schneller MOLOCH System-Status (FPS, Temp, Face-ID, Tracking, NPU, Worker). Nutze wenn Du wissen willst ob alles laeuft.
---

# LOKOMOTIVE aktiv — System-Status

Rufe BEIDE Tools auf (Pflicht-Startprotokoll):

```
moloch_status()        — loest Session-Lock + zeigt System
moloch_npu_workers()   — Worker-Health + Fehler
```

Zeige kompakt:

| Was | Wert |
|-----|------|
| Service | laeuft / gestoppt |
| FPS | total (Soll: >18) |
| CPU Temp | °C (Warnung: >75°C) |
| RAM | % (Warnung: >90%) |
| Face-ID | wer erkannt + Confidence |
| Zone | Guardian / Shadow / Berserker |
| NPU Stage | face / person / idle |
| Tracker | tracking / searching / idle |
| Worker | X/7 running, Y errors |

**Bei Problemen:**
- FPS < 18 → `moloch_logs(filter_str="ERROR")`
- Worker errors > 0 → `moloch_npu_workers()` + `moloch_dmesg()`
- RAM > 90% → `moloch_service(action="restart")`
- CPU > 75°C → Fan-Check, Thermal-Manager pruefen
