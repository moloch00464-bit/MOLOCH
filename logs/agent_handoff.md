# M.O.L.O.C.H. Übergabeprotokoll
**Datum:** 2026-03-28, 14:30 CET
**Von:** Claude Sonnet 4.6 — MCP Server + GitHub + Panel-Analyse Session
**Service-Status:** LAEUFT (20 FPS, 39/39 Audit PASS)
**USE_TAPPAS:** 1 (aktiv)
**GitHub:** Verbunden (moloch00464-bit/MOLOCH), sauber gepusht

---

## NEU SEIT LETZTER SESSION

### 1. MOLOCH MCP Server
- Datei: mcp/moloch_mcp_server.py
- Config: .mcp.json im Projektverzeichnis
- 8 Tools: moloch_status, moloch_logs, moloch_snapshot, moloch_service, moloch_audit, moloch_read, moloch_git_log, moloch_dmesg
- Aktivierung: Beim naechsten Session-Start fragt Claude Code nach Bestaetigung
- MCP Package: mcp==1.26.0 system-wide installiert

### 2. Claude Code Skills
- /moloch-status  -> Live-Status kompakt
- /moloch-snapshot -> Kamera-Frame holen + analysieren
- /moloch-audit   -> 39-Test Audit starten

### 3. GitHub
- gh CLI verbunden als moloch00464-bit
- Repo: github.com/moloch00464-bit/MOLOCH
- api_keys.json aus GESAMTER Git-History entfernt (git-filter-repo)
- Stand 2026-03-28 gepusht

---

## KRITISCH: SEGV Regel (IMMER GUELTIG!)
- bbox.ymin()/xmin() auf Pose-Detections -> SEGV nach ~50s
- Gilt fuer _on_buffer UND _on_pre_overlay
- NIEMALS bbox.*() auf Detections mit HAILO_LANDMARKS
- Sicher: get_label(), get_confidence(), get_objects_typed()

---

## AUFGABEN NAECHSTE SESSION

### PRIO 1: panel_models.py aufraumen
Problem: Gruen = TAPPAS (immer aktiv), Weiss = togglebar
FEHLT: face_attr laeuft in Pipeline aber nicht im Panel!

Fix in core/gui/panel_models.py:
1. ("FaceAttr", "faceattr") zu TAPPAS_MODELS hinzufuegen
2. FPS-Detail: auch pose FPS anzeigen
3. status_key_map: "faceattr": "faceattr_active" ergaenzen
4. EXTRA_MODELS: Hand LM ausblenden wenn deaktiviert

### PRIO 2: Tracking-Glaettung (Kamera ruckelt)
Datei: core/hardware/camera.py ~Zeile 721
- TRACKING_GAIN_PAN: 0.7 -> 0.4
- MAX_STEP_PAN: 30 -> 15
- Dead-Zone: erst bewegen wenn Abweichung > 3%
- Smooth: Durchschnitt letzte 3 BBox-Positionen
ACHTUNG: pan_delta = -error_x (MINUS korrekt, nicht aendern!)

### PRIO 3: Neue Modelle aktivieren (NPU RAM: <1% genutzt!)
- face_attr: Laeuft schon, nur Panel-Fix (PRIO 1)
- Person-ReID (repvgg_a0_person_reid_512.hef): Valve pruefen
  Test: journalctl -u moloch | grep -i "reid\|cv2\|crash"
  Falls kein Crash -> Valve in tappas_pipeline.py aktivieren
- Segmentierung (yolov5n_seg_h10.hef): TAPPAS Pipeline erweitern

### PRIO 4: Stabilitaets-Waechter
CronCreate: Alle 5 Min frame_age pruefen, bei >120s Restart

---

## SYSTEM-ZUSTAND
FPS:            20.1 (scrfd/arcface/yolov8m je 20.1)
NPU RAM:        ~55MB / 8192MB (<1%)
Aktive Modelle: arcface, faceattr, hand, scrfd
Person:         erkannt (face_id: markus, Szenario: NAH)
Power:          93% Akku, Netzteil, 10.2W
GitHub:         sauber gepusht 03-28
Audit:          39/39 PASS

## BEKANNTE BUGS (CLAUDE.md)
1. Hot-Plug: Stecker raus -> nur Reboot
2. ArcFace Threshold 0.45 zu niedrig
3. Suchrichtung asymmetrisch
4. Tracking Gains zu hoch -> Ueberschwinger (PRIO 2)
5. Tension-Popup Kontrast schlecht

## GEAENDERTE DATEIEN DIESE SESSION
- mcp/moloch_mcp_server.py (NEU)
- .mcp.json (NEU)
- config/perception_weights.json
- config/system_capabilities.json
- .claude/skills/moloch-audit.md (NEU)
- .claude/skills/moloch-snapshot.md (NEU)
- .claude/skills/moloch-status.md (NEU)
