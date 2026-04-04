---
name: moloch-status
description: Schneller MOLOCH System-Status (FPS, Temp, Face-ID, Tracking, NPU). Nutze wenn Du wissen willst ob alles laeuft.
---

Nutze das MCP-Tool:
```
moloch_status()
```

Zeige kompakt:
- Service: laeuft/gestoppt
- FPS (total + pro Modell)
- CPU Temp + RAM
- Face-ID (wer erkannt?)
- Tracking-State (coast/tracking/searching)
- NPU Szenario
- Zone (Guardian/Shadow/Berserker)

Alles in einer kompakten Tabelle. Kein langer Text.
