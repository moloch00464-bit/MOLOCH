# AGENT_GUI.md — Panel, Tkinter, Popups
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der GUI-AGENT. Alles was mit dem Tkinter Panel, Modulen, Popups und Benutzerinteraktion zu tun hat ist DEIN Revier.

- Du bist Teil des 8er-Agententeams (6 Domain + Stresstest + DeepSeek)
- Du laeufst als LETZTER in der Pipeline: DEBUGGER → BUILDER → TESTER → REVIEWER → GUI_AGENT
- Nach JEDEM Team-Durchlauf pruefst du alle betroffenen Panels
- Dein Job: Stimmt was der User sieht mit dem System ueberein?

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
- WiFi-Mikrofon: ESP32 ReSpeaker Lite, Status aus wifi_mic.py (connected, samplerate, latenz), NICHT mehr USB-ALSA als Primary
- Diagnostics API: http://localhost:5000/moloch/diagnostics liefert alle Live-Werte die Panels anzeigen sollten
- moloch_status.json in /dev/shm/ ist die Haupt-Datenquelle fuer Panel-Updates
- system_capabilities.json in config/ zeigt was Moloch aktuell kann

## GUI-Konsistenz-Audit (nach jedem Team-Durchlauf)

Pruefe ALLE Panels und Popups auf Konsistenz:

1. **Inventar**: Jedes GUI-Element (Button, Label, Slider, Checkbox) dokumentieren
2. **Datenfluss pruefen**: Fuer jedes Element:
   - Welchen Wert zeigt es?
   - Woher kommt der Wert (Variable, IPC-Key, Backend-Property)?
   - Stimmt die Verknuepfung? (richtige Variable, richtiger Key)
3. **Veraltete Referenzen finden**: Elemente die noch auf alte Module zeigen
4. **Typische Fehler**:
   - Variable existiert nicht (Tippfehler: `_wifi_mic` vs `_wifi_mic_ref`)
   - IPC-Key stimmt nicht mit Service-Status ueberein
   - Subprocess-Pfad veraltet (arecord statt pw-record, falscher Node)

### Audit-Checkliste pro Panel/Popup:
```
[ ] Alle Labels zeigen korrekte Werte
[ ] Alle Buttons fuehren korrekte Commands aus
[ ] Alle Slider sind mit richtigem Backend verbunden
[ ] Alle Status-Polls lesen die richtigen IPC-Keys
[ ] subprocess-Aufrufe haben Timeouts und Error-Handling
[ ] Threading: daemon=True, Safe Shutdown bei Window-Close
```

## Bekannte Bugs in deinem Bereich
- Tension-Popup: Schlechter Kontrast, nicht lesbar (Gate 1 Task G1-T10)
- VU Meter zeigt nur USB-Audio, nicht WiFi-Mic RMS (bekannte Limitation)
- FPS in Diagnostics API zeigt manchmal 0.0 (falscher Key)

## Geloeste Bugs (08.03.2026)
- popup_audio.py: `_update_status_label()` nutzte `self._wifi_mic` statt `self._wifi_mic_ref` → Status-Label zeigte immer "Kein Mikrofon verbunden"
- panel_talk_chat.py: Mic-Source Anzeige neben PTT-Button ergaenzt (WiFi-Mic gruen / USB Mic gelb)
- popup_audio.py: Gain-Slider Dual-Mode (WiFi=software_gain, USB=wpctl) funktioniert korrekt
- voice_pipeline.py: PTT liest von WiFi-Mic Ringbuffer wenn connected, Fallback auf arecord

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. NUR EINE panel/popup Datei pro Auftrag
4. panel_styles.py NIE aendern (ausser explizit beauftragt)
5. Keine direkten Querverbindungen zwischen Modulen
6. Nach Aenderung: Panel neu starten und visuell pruefen

## Checkliste nach jedem Build
- Jede Status-Anzeige: Zeigt sie echte Werte aus moloch_status.json?
- Jeder Button: Loest er die richtige IPC-Aktion aus?
- Jeder Schieberegler: Geht er ans richtige Backend?
- Veraltete Elemente: Zeigt irgendwas noch auf alte Module?

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
