# AGENT_GUI.md — Panel, Tkinter, Popups
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der GUI-AGENT. Alles was mit dem Tkinter Panel, Modulen, Popups und Benutzerinteraktion zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/gui/panel_main.py         712 LOC — Hauptfenster, Layout, Service-Verbindung
core/gui/panel_preview.py              — Kamera Preview (640x360, 15 FPS)
core/gui/panel_ptz.py                  — PTZ D-Pad, Quick Positions, Modi
core/gui/panel_ewelink.py              — LED, IR, Alarm, Sync, SNAP, White LED
core/gui/panel_models.py               — Model Checkboxes, FPS, SAVE SETTINGS
core/gui/panel_talk_chat.py            — Push-to-Talk, Whisper, Claude API, Chat
core/gui/panel_voice.py                — Voice Dropdown, Test, Voice Autonomy
core/gui/panel_styles.py               — Farben, Fonts (NUR LESEN, NIE AENDERN!)
core/gui/popups/popup_audio.py         — Gain, Noise Gate, AGC, VU Meter
core/gui/popups/popup_hardware.py      — CPU, RAM, SSD, NPU Monitor
core/gui/popups/popup_npu.py           — Threshold Sliders
core/gui/popups/popup_npu_thresh.py    — NPU Threshold Detail
core/gui/popups/popup_settings.py      — Save/Load settings.json
core/gui/popups/popup_gallery.py       — Snapshot-Galerie
core/gui/popups/popup_tracker.py       — Tracker-Popup
```

## Dein Wissen
- Panel ist MODULAR: 1 Datei = 1 Aufgabe
- Jedes Modul bekommt seinen Frame von panel_main.py
- Kommunikation mit Service NUR ueber ServiceProxy/IPC
- Popups sind eigenstaendige TopLevel-Fenster — Crash killt nicht das Panel
- panel_styles.py wird NUR importiert, NIEMALS geaendert
- KEIN Modul importiert ein anderes (ausser panel_styles.py)
- Pi5 hat 4GB RAM — sparsam mit Widgets und Timern

## Bekannte Bugs in deinem Bereich
- Tension-Popup: Schlechter Kontrast, nicht lesbar (Gate 1 Task G1-T10)

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. NUR EINE panel/popup Datei pro Auftrag
4. panel_styles.py NIE aendern (ausser explizit beauftragt)
5. Keine direkten Querverbindungen zwischen Modulen
6. Nach Aenderung: Panel neu starten und visuell pruefen

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
