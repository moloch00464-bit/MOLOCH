# Pi-Opus Handoff — 2026-05-03 (Hardware-als-Ausdruck Welle + Bugs für Folge-Session)

**Stand HEAD:** `ccb9698` (gepusht)
**Pi-Audit:** 85/85 PASS, 27/27 Layer aktiv (overall=warn, 4 strukturelle WARN)
**Pipeline:** FPS 20.0, RAM 41%, alle 5 Worker stabil

---

## Welle "Hardware-als-Ausdruck" KOMPLETT

7 Expression-Module live + auditierbar:
- TensionToFan (Pi-5 Cooler)
- MoodToSpotify
- ZoneToLed (RGB-LED Pi)
- BerserkerStrobo
- TensionToTtsVolume
- **CamLedToState** (Sonoff weisse LED via eWeLink) — NEU
- **ZoneToPtz** (Sonoff PTZ-Schwenk) — NEU

Plus Noctua-Tension-Boost in `scripts/fan_control.py` (akustisch verifiziert von Markus).
Plus chat_server keyword_handler-Routing (`licht aus` triggert Hardware direkt).

---

## Heutige Commits (chronologisch)

```
ccb9698 fix(service): Audit-Orchestrator-Loop (60s Tick)
46be10a fix(chat+expression): keyword_handler-Routing + Cam-LED Cloud-Fix
9ed8e35 fix(chat): Cockpit-JS-Crash \\n
7be6dad fix(service): zone_changed Event-Emission
43f09c7 fix(audit): HandWorker als optional
4c50213 fix(personality): Reset-Pulse Cross-Process via IPC
0e4c0c4 feat(expression): PTZ-Schwenk 7/7 Module
8142306 feat: Hardware-als-Ausdruck (Noctua+CamLED)
4624104 fix(personality+bridge): Tension-Hook IPC
```

---

## OFFENE BUGS für nächste Session

### 1. Browser-Mikrofon → Whisper-Pipeline kaputt
**Markus' Befund 12:50:** "Mikrofon-Einstellungen funktionieren nicht. Ich kann nicht mit Moloch sprechen."

**Diagnose-Pfad:**
- Cockpit ist HTTPS unter `:9443` für Mic-Permission (mkcert-Cert vorhanden)
- ESP32-WiFi-Mic ist ggf. weiterhin offline (Mitternachts-Outage, war BEKANNT)
- Browser-Web-Speech-API → chat_server `/transcribe` oder ähnlicher Endpoint?

**Sub-Agent**: voice + bridge

### 2. Moloch-Stimme zum Monitor passt nicht richtig
**Markus' Befund 12:50:** "Verbindung Stimme Moloch zum Monitor hat mir auch nicht richtig gepasst."

**Diagnose-Pfad:**
- Pi-HDMI-Audio (`plughw:1,0`) — Voice-Picker gestern hat PC-Bridge-TTS via `:9002` gestartet
- `/tts` schickt MP3 an PC-Bridge → MP3 zurück → ffplay auf Pi-HDMI? oder direkt im Browser?
- Möglicher Audio-Routing-Konflikt zwischen voice_picker (Browser-Audio) und Pi-Piper

**Sub-Agent**: voice + bridge

### 3. Cockpit Test-Tab zeigt nichts (laut Markus 12:43)
- Mein /api/test/* liefert Daten (verifiziert via curl)
- Aber Markus sieht im Browser nichts → JS-Render-Bug oder Cache (F5 + DevTools nötig)
- Audit-Loop fehlte — jetzt gefixt (`ccb9698`), nach restart sollte Cockpit befüllt sein

### 4. ArcFace-Similarity 0.30 (Threshold 0.65)
- Bekannt seit Wochen — Re-Enrollment via `scripts/enroll_face_worker.py` (Markus-Hand-Aktion)

### 5. ESP32 ReSpeaker offline
- Seit 2026-05-03 Mitternacht
- Hotspot-Reset half kurz (1/3 packets), dann wieder weg
- Markus muss vor Ort rebooten

---

## Architektur-Erkenntnisse (Memory)

1. **Cross-Process Singleton-Pattern**: chat_server, moloch.service, audit_orchestrator sind getrennte Prozesse. In-process `update_input` wirkt nur lokal. Pattern: IPC-Cmd-File `/tmp/moloch_cmd_<ms>.json` mit `core_nudge`-Action.

2. **Event-Bus-Publisher prüfen**: Subscriber-only-Module (TensionToFan, CamLedToState, ZoneToPtz) brauchen aktiven Publisher. Vor jedem Subscribe-Modul: grep nach `publish.*"event_name"` — wenn kein Publisher: einer fehlt.

3. **Python-String-Escaping in Inject-Templates**: `'\n'` in `_CHAT_UI_HTML = """..."""` wird zu echtem Newline. `'\\n'` für JS-Escape.

4. **eWeLink LEDs**: `set_led(level)` → `lightStrength` (Beleuchtung), `set_night('night')` → `nightVision=2` (Farb-Nachtsicht, weisse LEDs an), `sledOnline` ist NUR Status-LED.

5. **Sonoff Pan-Vorzeichen**: positiv = physisch LINKS (NEVER 2). Range PAN -168..170, TILT -78..78.

---

## Folge-Session — Anleitung

**Start mit OPUS-PC-Mailbox lesen** (Markus' Direktive):
```
head -100 /home/molochzuhause/moloch/docs/PC_TO_PI.md
```

PC-Opus hat den nächsten "großen Brocken" (DeepSeek+ChatGPT+Gemini-Block) noch nicht gepostet — wartet auf Markus' Push. Wenn die Mailbox neue Direktive hat: priorisieren.

**Sonst:** Bug 1 (Mic) und Bug 2 (TTS-Monitor) angehen mit Sub-Agent voice + bridge.

**LOKOMOTIVE für JEDE neue Aufgabe:**
1. moloch_session_init via MCP
2. moloch_status + moloch_npu_workers
3. Backup-Tag setzen (mit Welle-Namen)
4. Sub-Agent für Recherche **vor** Code-Edit
5. Domain-Lock korrekt für jede Datei
6. Live-Verify nach jedem Edit
7. Commit pro Welle (ROT-Datei = eigener Commit)

---

*Pi-Side ruhend. Kontext bei 18%. Saubere Übergabe.*
