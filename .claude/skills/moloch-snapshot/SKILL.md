---
name: moloch-snapshot
description: Kamera-Snapshot von MOLOCH holen und analysieren. Zeigt was MOLOCH gerade sieht. Real-ESRGAN Upscaling + Low-Light Enhancement.
---

# MOLOCH Kamera-Snapshot

```
moloch_snapshot()
```

## Was analysieren?

**BBoxen (panel_preview.py zeichnet via PIL):**
- Cyan = Person erkannt mit Face-ID
- Gelb = unbekannte Person
- Gruen = Person ohne Gesicht

**Qualitaet pruefen:**
- BBoxen sitzen korrekt ueber Person/Gesicht (kein Offset)?
- Landmarks (Pose-Punkte) auf richtigen Koerperteilen?
- Keine Artefakt-BBoxen bei leerer Szene?
- Bild klar oder noch dunkel (Low-Light aktiv)?

**NPU-Verarbeitung im Snapshot:**
- Real-ESRGAN x2: 640x360 → 1024x1024 (schaerfer)
- Low-Light (zero_dce): automatisch wenn Helligkeit < 80/255

**Bei Problemen:**
- BBox falsch positioniert → vision-Agent (BBox-Inferenz)
- BBox falsch gezeichnet → gui-Agent (panel_preview.py)
- Letterbox-Versatz → NIEMALS manuell rescalen (TAPPAS korrigiert automatisch)
