---
name: moloch-status
description: Schneller MOLOCH System-Status (FPS, Temp, Face-ID, Tracking, NPU). Nutze wenn Du wissen willst ob alles laeuft.
---

Lies `/dev/shm/moloch_status.json` und zeige kompakt:
- Service: laeuft/gestoppt
- FPS (total + pro Modell)
- CPU Temp + RAM
- Face-ID (wer erkannt?)
- Tracking-State (coast/tracking/searching)
- NPU Szenario (IDLE/FERN/MITTEL/NAH)
- Letzter FACE-MATCH aus journalctl (1 Zeile)

Alles in einer kompakten Tabelle. Kein langer Text.
