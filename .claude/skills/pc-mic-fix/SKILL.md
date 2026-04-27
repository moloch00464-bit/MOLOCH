---
name: pc-mic-fix
description: Chrome-Mikrofon-Permission auf Markus-PC reparieren. Nutze wenn Markus sagt "Mic kaputt", "kein Mic-Berechtigung", "Sprachnachricht geht nicht" oder das Cockpit-UI keine Mic-Erlaubnis hat. 3-Schritt-Recipe (Chrome zu, Prefs editieren, Chrome auf).
user-invocable: true
---

# PC-Mic-Fix — Chrome-Mikrofon-Permission Reset

Standard-Procedure wenn Mic im Cockpit-UI streikt. Symptome: kein Mic-Icon in Adressleiste, Click auf Mic-Button macht nichts, Browser fragt Permission und Markus klickt "Allow" aber speichert nicht persistent.

## Voraussetzung

`pc/fix_chrome_mic_prefs.py` existiert + lauffaehig. `MOLOCH.lnk` auf Desktop existiert.

## Schritte

### 1. Diagnose: ist Mic wirklich das Problem?
- Markus' offene URL? (typisch `localhost:9000`, manchmal raw-IP-HTTPS-Variante)
- Chrome-Adressleiste: Mic-Icon mit Schloss-X oder mit Allow-Symbol?
- Pi `:9000/history` lesen — gibt's neue user-Messages? (wenn ja: Mic geht eigentlich, Issue ist anderes)
- iframe-Permission im parent-HTML (Avatar `:11800` embedded Cockpit)?

### 2. Chrome komplett schliessen
Wenn Markus Tabs offen hat: ihn bitten zu schliessen. Mit OK auch:
```bash
taskkill /F /IM chrome.exe
```
Verify:
```bash
tasklist /fi "IMAGENAME eq chrome.exe"
# muss leer sein
```

### 3. Prefs-Edit
```bash
cd C:\Users\49179\moloch_repo
"%USERPROFILE%\moloch_pc_env\Scripts\python.exe" pc/fix_chrome_mic_prefs.py
```
Output: `[mic] N URLs neu auf Allow gesetzt` ODER `keine Aenderung noetig`.

### 4. Chrome wieder oeffnen via MOLOCH.lnk
Markus klickt Desktop-Shortcut `MOLOCH.lnk`. Das Bat:
- prueft alle Services
- prueft Mic-Prefs (jetzt OK)
- oeffnet Chrome zu `http://localhost:9000/`

### 5. Verify
Markus testet: Sprachnachricht aufnehmen. Wenn klappt → done. Wenn nicht → siehe Edge-Cases.

## Edge-Cases

### A) URL-Drift
Markus auf URL die NICHT in Allow-Liste ist. Lookup `pc/fix_chrome_mic_prefs.py URLS` — 8 URLs sind drin. Wenn Markus eine andere nutzt: URL adden, Skript erweitern, neu laufen lassen.

### B) iframe blocked durch Permission-Policy
Wenn Avatar `:11800` ein Cockpit-iframe einbettet und der iframe `allow=microphone` Header fehlt: Pi muss das fixen (`allow="microphone; camera; autoplay"` im iframe-Tag). Pi hat das 2026-04-27 08:14 defensiv gepushed — sollte bereits drin sein.

### C) Permission-Policy Server-Header
`curl -I http://localhost:9000/` — wenn `Permission-Policy: microphone=()` im Response-Header: Server blockt komplett. Das ist Pi-Side `chat_server.py` — Mailbox-Anfrage an Pi.

### D) Chrome Profile-Switch
Markus hat eventuell mehrere Chrome-Profile. Skript editiert nur `Default`. Falls anderes Profile aktiv: in Chrome `chrome://version/` checken welches Profile-Pfad. Skript anpassen.

## Reboot-Persistence

Die Mic-Permission-Aenderung in Chrome-Prefs ueberlebt Reboot — sie ist im Chrome-User-Profile gespeichert. Ein einmaliger Fix reicht. Nur wenn Chrome-Update die Prefs umschreibt oder Markus manuell auf "Block" klickt: nochmal noetig.

## NEVER

- NIE Skript laufen lassen wenn Chrome offen ist (Prefs werden ueberschrieben)
- NIE andere Settings als `media_stream_mic` editieren
- NIE Backups loeschen — sind bei Skript-Ueberschreibungen die einzige Recovery
