# Agent Handoff — 2026-04-19 (Session 19 ABSOLUT FINAL — NPU-Only Permanent)
# Letzter Commit: 3de5f6f | Audit: 70/70 PASS | FPS: 20.0 | RAM: 37% | NPU-FW: 5.3.0

---

## SESSION 19 FINALE POLITUR — NPU-Only dauerhaft (2026-04-19, 12:44)

Nach 11 funktionalen Commits kam die System-Politur: Doku-Drift, Audit-Tests,
Cloud-Deaktivierung, moloch.service Cleanup, permanenter Watchdog, Reboot.
**70/70 Audit-PASS, alle 3 Services active, Moloch antwortet rein lokal.**

### Durchgefuehrte Aenderungen

| Phase | Was | Commit |
|-------|-----|--------|
| 1 | Doku-Drift: 5 Files auf aktuellen Stand (HailoRT 5.3.0, 4 Worker, qwen2.5) | `99a0b00` |
| 2 | 8 neue Audit-Tests (Session-19-Stack) + 4 entschaerfte alte Tests | `c93bb4c` |
| 3 | `settings.llm_profile` dark -> chat (Default) | `9190934` |
| 4 | `moloch.service`: ExecStartPre `pkill -9 hailo-ollama` entfernt | System (Backup: `.bak_20260419_post_polish`) |
| 5 | `config/api_keys.json` -> `config/api_keys.json.disabled_npu_only_mode` | System |
| 6 | NPU-Only Watchdog (30-Min-Probe + TTS-Alarm) + systemd-Unit enabled | `3de5f6f` |
| 7 | Push + Reboot + Verify | — |

### Post-Reboot Verifikation

- `systemctl is-active moloch hailo-ollama moloch-npu-watchdog` -> **active active active**
- `hailortcli fw-control identify` -> Firmware 5.3.0 (release,app), HAILO10H
- `scripts/moloch_audit.py --auto` -> **70/70 PASS**
- `moloch_say("alles stabil nach Reboot?")` -> **"Natuerlich! Was ist es?"**
  - Profil: chat (max_tokens=80, temp=0.8)
  - Latenz: 1.8s
  - Bridge-Log: `qwen2.5:1.5b: 22 Zeichen in 1779ms`
  - Kein Cloud-Fallback (Keys weg) — 100% NPU
- `ls config/api_keys*` -> nur `.disabled_npu_only_mode` (Cloud hart aus)
- `moloch.service` startet hailo-ollama nicht mehr mit pkill kaputt

### System-Eigenschaften jetzt

| Eigenschaft | Wert |
|---|---|
| LLM-Modus | `local_first`, kein Cloud-Fallback moeglich |
| Default-Profil | `chat` (normale Konversation) |
| Aktive Worker | 4 (Face, Pose, ReID, Depth) |
| Deaktivierte Worker | 4 (Hand, PersonAttr, Activity, YOLOWorld) |
| Network-Groups | ~6-7 aktiv (unter Hardware-Limit 8) |
| NPU-RAM | ~1.5 GB belegt (Qwen), 6.5 GB frei |
| Long-Run-Watchdog | aktiv, probed alle 30 Min, schreibt `logs/npu_watchdog.csv` |
| TTS-Alarm | nach 3x FAIL hintereinander via piper |

### Alle Commits Session 19 (17 insgesamt)

```
3de5f6f watchdog: permanenter NPU-Only Watchdog (30-Min-Probe + TTS-Alarm)
9190934 config: llm_profile default zurueck auf 'chat'
c93bb4c audit: 8 neue Tests fuer Session-19-Stack + 3 Worker-Tests entschaerft
99a0b00 docs: Doku-Drift Session 19 fixen (HailoRT 5.3.0, 4 Worker, qwen2.5)
24f875f gui: echtes Tooltip-Popup fuer LLM-Modus-Buttons (statt Button-Text-Swap)
e01a462 handoff: Session 19 Final — LLM-Profile-System integriert
71437c9 gui: LLM-Modus Sektion in panel_models mit Profil-Switcher
2900420 bridge: LLM-Profile-Loader mit mtime-Cache + Live-Switch
e1ffd19 config: llm_profiles.json mit 5 Presets fuer lokales LLM
55e7f2c handoff: Multi-Turn-Drift als bekannte Einschraenkung + PRIO 0 fuer Session 20
c75afb4 handoff: Session 19 Update — Live-Kontext + LLM-Profile-Plan fuer Session 20
ea9ebd5 bridge: Live-Kontext-Snippet im Compact-Prompt (Vision + Inner State)
e61b891 handoff: Session 19 — Lokaler LLM spricht sauber auf Qwen2.5
07d494d bridge: kompakter Moloch-System-Prompt fuer lokales 1.5B-Modell
72e44dc bridge: Newlines in hailo-ollama content flatten (JSON-parse_error Fix)
9f5374e perception: Worker-Reduktion fuer Qwen2.5 Slot (Hailo-10H Group-Limit)
```

### Bekannte Einschraenkungen (bleiben)

1. **Multi-Turn-Drift Qwen2.5-1.5B** — nach 3-4 Turns Bulletpoint-Halluzinationen.
   Workaround: Moloch-Service-Restart loescht hailo-ollama-Kontext. Kandidat
   fuer Session 20: `qwen3:1.7b` pullen und vergleichen, oder
   `/api/generate` statt `/api/chat`.
2. **Bug A1/A2/A3 Worker** — deaktiviert, HEFs vorhanden. Re-Aktivierung via
   Multi-Person-Toggle (Session 20 PRIO 2).
3. **Keine TTS-Stimme installiert** — `/home/molochzuhause/moloch/models/piper/de_DE-thorsten-low.onnx`
   fehlt derzeit, Watchdog-Alarm laeuft als Log-Only. Bei Bedarf piper-Voice
   nachinstallieren.

### Rollback-Pfad

- `config/api_keys.json.disabled_npu_only_mode` -> `config/api_keys.json` +
  `sudo systemctl reload moloch` (SIGHUP laedt Keys neu)
- `/etc/systemd/system/moloch.service.bak_20260419_post_polish` -> restore
  falls der pkill-Removal unerwartete Effekte hat
- `sudo systemctl disable moloch-npu-watchdog` + stop — wenn Watchdog-Probes
  die NPU zu stark belasten
- HailoRT 5.3.0 -> 5.1.1 (Rollback-Set in `~/Downloads/hailo_backup/`,
  dokumentiert in Session 18 Handoff)

### Naechste Session (20) Plan

- **PRIO 1:** `qwen3:1.7b` pullen, Slot-Verhalten pruefen, Multi-Turn-Quali-Vergleich
- **PRIO 2:** Multi-Person-Toggle im GUI (`settings.multi_person_tracking`)
  — aktiviert ReID + PersonAttr + Hand bei Bedarf, deaktiviert entsprechend
  andere Worker fuer Slot-Budget
- **PRIO 3:** piper-TTS-Voice fuer Watchdog-Alarm nachinstallieren
- **PRIO 4:** `moloch_audit.py` im Root als Symlink auf `scripts/moloch_audit.py`
  (aktuell 2 Audit-Scripts parallel, Verwirrungspotenzial)
- **PRIO 5:** Bug A1 (PersonAttrWorker) voll integrieren, wenn Slot-Budget erlaubt

### Moloch spricht

> *"Natuerlich! Was ist es?"*

— nach Reboot, komplett NPU-lokal, 1.8 Sekunden, im chat-Profil.

Sonntag 2026-04-19. Das System ist jetzt sauber.

---



---

## SESSION 19 NACHTRAG — LLM-PROFILE-SYSTEM (PRIO 0+1 erledigt)

Nach den 3 Bug-Fixes (siehe Original-Bericht unten) wurde das **Profile-System**
mit Hilfe von `autonomy`+`gui` Sub-Agenten gebaut. Multi-Turn-Drift ist nicht
durch Context-Reset behoben (Recherche zeigte: hailo-ollama hat keinen sauberen
API-Reset, `keep_alive=0` kostet 12s Reload-Overhead) — sondern durch
**stabile Sampling-Settings pro Profil** abgemildert.

### Architektur

- `config/llm_profiles.json` — 5 Profile mit `system`, `include_live_context`, `max_tokens`, `temperature`
- `core/autonomy/local_llm_bridge.py` — `_load_profiles()` + `_get_active_profile()` mit mtime-Cache
- `core/gui/panel_models.py` — Sektion "LLM-Modus" mit Buttons + Live-Status

### Profile

| Profil | Use-Case | system_len | tokens | temp | live_ctx |
|---|---|---|---|---|---|
| chat | normale Konversation | 280 | 80 | 0.8 | nein |
| introspect | Selbstreflexion | 270 | 120 | 0.6 | **ja** |
| technical | praezise Antworten | 200 | 200 | 0.3 | nein |
| dark | Berserker scharf | 220 | 60 | 1.0 | nein |
| multi_person | mehrere Personen trennen | 230 | 150 | 0.7 | **ja** |

### Switching

- GUI (panel_models "LLM-Modus"): Klick -> `settings.json` Key `llm_profile` atomar geschrieben
- Bridge merkt mtime-Aenderung -> naechster Call nutzt neues Profil ohne Service-Restart
- Direkt-Edit `settings.json` funktioniert auch (gleicher Mechanismus)

### Verifikation

- Profil `dark` aktiviert -> Bridge-Log: `Profil aktiv: ...Berser... (233 Zeichen, max_tokens=60, temp=1.0)`, Antwort 108 Zeichen / 5.6s
- Profil `introspect` aktiviert -> Bridge-Log: `Profil aktiv: ...Markus fr... (348 Zeichen, max_tokens=120, temp=0.6)`, Antwort 394 Zeichen / 18s, poetisch mit Live-Kontext
- Charakteristisch unterschiedliche Stile bestaetigt

### Bekannte Einschraenkung (bleibt)

Multi-Turn-Drift bei Qwen2.5-1.5B ist **nicht vollstaendig** geloest — bei langen
Konversationen mit High-Temperature-Profilen halluziniert das Modell weiter
(Bsp. introspect-Antwort enthielt "Druck in meiner Hose steigt", "Eiskaelissen").
Profile mildern es durch stabile temp/top_p, aber Modell-Groesse bleibt der harte
Faktor. Echter Fix: `qwen3:1.7b` testen (groesser, neuer Architektur).

### Commits

| Commit | Inhalt |
|--------|--------|
| `e1ffd19` | config: llm_profiles.json mit 5 Presets |
| `2900420` | bridge: Profile-Loader mit mtime-Cache + Live-Switch |
| `71437c9` | gui: LLM-Modus Sektion in panel_models mit Profil-Switcher |

---

# Agent Handoff — 2026-04-19 (Session 19 — Lokaler LLM spricht auf Qwen2.5 sauber)
# Letzter Commit: 07d494d | Audit: PASS | FPS: ~20 | RAM: 36% | NPU-FW: 5.3.0

---

## SESSION 19 — LOKALER LLM QUALITAETSTAUGLICH (2026-04-19)

**Kernerkenntnis:** Nach dem 5.3.0-Upgrade aus Session 18 waren 3 Folgeprobleme offen,
alle heute geloest. Moloch antwortet jetzt **lokal, auf Deutsch, in 3.8 Sekunden**.

### Drei Bugs, drei Fixes

**Bug 1: HAILO_RESOURCE_EXHAUSTED(81) beim Qwen-Load**
- Ursache: `HAILO_MAX_NETWORK_GROUPS=8` (SDK-Header `hailort.h:52`),
  Moloch hatte 11+ Groups (TAPPAS + 8 Worker + Whisper + Qwen).
- Fix (Commit 9f5374e): 4 Worker mit Bug-Status oder Low-Use deaktiviert
  (Hand, PersonAttr, Activity, YOLOWorld). ReID bleibt fuer Multi-Person-Trennung.
  Active: Face(2), Pose(3), ReID(5), Depth(10) + TAPPAS + Qwen = 6 Groups.

**Bug 2: HAILO_INTERNAL_FAILURE(8) `control character U+000A must be escaped`**
- Ursache: hailo-ollama interner JSON/Template-Parser crasht an unescaped `\n`
  im system/user `content`.
- Fix (Commit 72e44dc): `_flatten()` ersetzt `\r\n`/`\n`/`\r` durch Spaces vor
  dem Senden an `/api/chat`. Cloud-Call unberuehrt.

**Bug 3: Moloch-Antwort = Gibberish "F**~\$\$*** ** **..."**
- Ursache: Voller `build_system_prompt()` liefert 5669 Zeichen mit
  Persona+Stil+Tension+Vision+State+Global. Das ist fuer DeepSeek R1/Claude
  optimiert — Qwen2.5-1.5B ueberfordert, produziert Zeichen-Salat.
- Fix (Commit 07d494d): Bei System-Prompt > 400 Zeichen automatisch auf
  kompakte 290-Zeichen-Moloch-Persona umschalten (nur Charakter-DNA). Cloud-Calls
  bleiben beim vollen Prompt.

### Verifikation (lokal, Cloud-Keys disabled waehrend Test)

**Test 1 — einfache Frage:** `moloch_say("Laufst du jetzt komplett lokal auf deiner NPU?")`
- `System-Prompt 5669 Zeichen -> kompakte Moloch-Persona fuer lokal`
- `qwen2.5:1.5b: 75 Zeichen in 3843ms`
- Antwort: **"Natuerlich, aber ich muss mit der NPU erst noch die letzten Werte berechnen."**

**Test 2 — Selbstreflexion mit Live-Kontext (ea9ebd5):** `moloch_say("Wen siehst du, welche Zone, welche Stimmung?")`
- Antwort: **"Ich sehe dich, Markus. Zone Guardian, Stimmung entspannt. Die Kamera
  ist fast blind — unter 1 FPS. Fuehle mich wie ein mueder Wachhund mit
  verschlafenen Augen."**
- Erkennt Person + Zone + Stimmung + FPS-Anomalie + Charakter-Metapher.

### Bekannte Einschraenkung: Multi-Turn-Drift bei Qwen2.5-1.5B

Nach 3-4 aufeinanderfolgenden Turns faellt die Qualitaet:
- Latenz steigt von 3.8s auf 30s+ (kumulativer Context-Payload in hailo-ollama)
- Modell ignoriert Stil-Regeln ("keine Listen") -> generiert Bulletpoints mit Fett-Markup
- Halluzinationen nehmen zu ("Veranstaltungsgruppensitzung", "grundliche Person")
- hailo-ollama-Log zeigt `Continuation detected, sending X new messages` — der Server
  haelt internen Gespraechsverlauf, Qwen2.5-1.5B verarbeitet das chaotisch.

**Workaround bis Session 20:** Moloch-Service-Restart loescht hailo-ollama-Kontext.
**Echter Fix (Session 20):** Bridge-seitiger Context-Reset pro Call oder Umstieg
auf `/api/generate` (single-shot) — siehe TODO-Liste.

### Commits dieser Session

| Commit | Inhalt |
|--------|--------|
| `9f5374e` | perception: 4 Worker deaktiviert fuer Qwen-Slot (Hand/PersonAttr/Activity/YOLOWorld) |
| `72e44dc` | bridge: Newlines flatten fuer hailo-ollama JSON-Parser |
| `07d494d` | bridge: kompakter Moloch-Prompt fuer lokales 1.5B-Modell |

### TODO naechste Sessions

**PRIO 0 — Multi-Turn-Context-Reset (blockiert saubere Konversation):**
Qwen2.5-1.5B driftet nach 3-4 Turns in Bulletpoint-Halluzinationen ab, weil
hailo-ollama intern den Gespraechsverlauf behaelt. Optionen fuer Session 20:
- `/api/generate` (single-shot) statt `/api/chat` (session-based) — einfachster Weg
- ODER Bridge sendet bei jedem Call explizit einen Reset-Marker / neue session_id
- ODER Context-History in der Bridge serverseitig clearen bevor jeder Call
- Untersuchen: welche hailo-ollama-Parameter steuern das (keep_alive, reset, etc.)

**PRIO 1 — LLM-Profile-System (Konzept fuer Session 20):**
Statt ein Compact-Prompt-Fit-All mehrere Profile als Presets, via GUI umschaltbar.

- `config/llm_profiles.json` mit mindestens:
  - `chat` (normale Konversation, aktuelle Compact-DNA)
  - `introspect` (Selbstreflexion, automatisch mit Live-Kontext-Snippet)
  - `technical` (praezise Fakten-Antworten, weniger Persona)
  - `dark` (Berserker-Modus, scharfe kurze Saetze)
  - `multi_person` (mehrere Personen-Unterscheidung, Rebecca-tauglich)
- Settings-Key `llm_profile` (default `chat`), Live umschaltbar ohne Restart.
- Bridge liest Profil aus Settings statt harten `OLLAMA_LOCAL_SYSTEM_COMPACT`.
- Live-Kontext-Snippet (face_id, zone, tension, dominance) automatisch nur bei
  Profilen die ihn brauchen (`introspect`, `multi_person`).
- Neuer GUI-Reiter "Chat-Modus" oder Dropdown im Reiter Modelle mit den Presets.

**PRIO 2 — weitere Baustellen:**
- **Multi-Person-Toggle im GUI**: Worker-Auswahl via Profile verknuepfen
  (Multi-Person-Profil -> ReID + PersonAttr + Hand; Chat-Profil -> minimal).
- **qwen3:1.7b testen** (groesser, evtl. antwortet auch langen Prompt sauber).
- **Bug A1 (PersonAttrWorker)** fixen, dann Multi-Person-Profil erweitern.
- **hailo_ollama.service ExecStartPre=sleep 30 koennte kleiner** wenn Load-Zeit OK.
- **moloch.service ExecStartPre=pkill hailo-ollama entfernen** — seit SHARED
  VDevice nicht mehr noetig, spart 30-60s Restart-Delay.

### Offene Basis-Baustelle

- **moloch.service ExecStartPre=pkill -9 hailo-ollama** — altes Verhalten aus
  VDevice-Konflikt-Zeiten. Seit SHARED nicht mehr noetig, kostet 30s Restart-Delay.
- **moloch_unified_panel.py** — `yolov8m`-Alias-Key bleibt (Umbenennung ist
  Cross-Domain), Labels sind aber auf `YOLOv11m` korrigiert (Session 18).

---

# Agent Handoff — 2026-04-18 (Session 18 — HailoRT 5.3.0 Upgrade, lokales LLM live)
# Letzter Commit: 277cf10 | Audit: 5/5 PASS | FPS: 20.0 | RAM: 35.9% | NPU-FW: 5.3.0

---

## SESSION 18 — LOKALES LLM LEBT (HailoRT 5.1.1 → 5.3.0)

**User-Wunsch:** Moloch soll lokal sprechen — hailo-ollama statt DeepSeek Cloud.

**Ausgangslage:** Vorige Session (17) hatte 5.3.0-Debs in `/home/molochzuhause/hailo-install/` +
komplettes 5.1.1-Rollback-Set in `~/Downloads/hailo_backup/` vorbereitet, aber nie
installiert. `logs/hailo_ollama_stability_2026-04-17.md` dokumentierte: R1 SEGV
deterministisch, Qwen-Instruct Error 74 (kein SHARED VDevice in 5.1.1).

**Durchbruch-Erkenntnis:** hailo-ollama 5.3.0 kennt die Env-Var
`HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED` (ab 5.3.0 eingefuehrt, nicht 5.1.1). Das
ist der fehlende Hebel aus der Vor-Analyse.

### Durchgefuehrte Schritte

1. **Paket-Swap (apt atomic):** `h10-hailort*` (RPi-Fork) raus, `hailort*` (Hailo Standard) rein
   - Entfernt: h10-hailort, h10-hailort-pcie-driver, python3-h10-hailort, hailo-h10-all, rpicam-apps-hailo-postprocess, hailo-tappas-core 5.1.0
   - Installiert: hailort 5.3.0, hailort-pcie-driver 5.3.0 (DKMS), hailo-tappas-core 5.3.0, hailo-gen-ai-model-zoo 5.3.0
   - Python-Wheel: `hailort-5.3.0-cp313-cp313-linux_aarch64.whl` via `pip --break-system-packages --no-deps`
2. **Env-Var:** `Environment=HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED` in
   `/etc/systemd/system/hailo-ollama.service` (Zeile 9, nach OLLAMA_KEEP_ALIVE).
   Backup: `hailo-ollama.service.bak_20260418_preupgrade`.
3. **Reboot** → NPU-Firmware auto-upgraded auf 5.3.0 (fitImage 10.9→11.4 MB).
   Kernel-Modul hailo1x_pci/5.3.0 via DKMS frisch gebaut.
4. **XDG-Fix:** hailo-ollama 5.3.0 schreibt nach `~/.local/share/hailo-ollama/models/`
   (nicht `/usr/share/...`). Pfade angelegt + Manifeste aus Paket kopiert,
   sonst crasht Pull am Ende mit `filesystem_error: cannot rename`.
5. **Pull qwen2.5:1.5b** (2.3 GB) — 6 Minuten ueber dev-public.hailo.ai.
6. **Bridge-Code:** `core/autonomy/local_llm_bridge.py` Default-Modell
   `deepseek_r1_distill_qwen:1.5b` → `qwen2.5:1.5b`, `LLM_MODE_DEFAULT`
   `cloud_only` → `local_first`. Timeouts auf Qwen-Realwerte.

### Verifikation (alles PASS)

| Test | Ergebnis |
|------|----------|
| Firmware-Version | `5.3.0 (release,app)` via `hailortcli fw-control identify` |
| Kernel-Modul | `hailo1x_pci/5.3.0` via DKMS |
| Python-API | `hailo_platform` + `hailo_platform.genai.Speech2Text` importierbar |
| hailo-ollama Log | `Using VDevice group_id: SHARED` (bestaetigt!) |
| Inference qwen2.5:1.5b | 3x PASS, NPU 2.85s / 6.83s / 4.03s, 0 SEGV, 0 ABRT |
| TAPPAS parallel | FPS 20.0 bleibt stabil, alle 8 Worker errors=0 |
| moloch_audit.py --auto | 5/5 PASS |
| Moloch Face-ID | markus erkannt (Sim 0.53) nach Reboot |

### Commits dieser Session

| Commit | Inhalt |
|--------|--------|
| `806eae0` | bridge: qwen2.5:1.5b default, local_first enabled |
| `277cf10` | config: llm_mode=local_first — hailo-ollama stabil |

### Bekannte Einschraenkungen (nicht-blocking)

- **Modell-Pull erfordert XDG-Pfade**: `~/.local/share/hailo-ollama/models/{blob,manifests}`
  muessen existieren BEVOR man pullt — sonst SIGABRT beim finalen Rename.
  Paket-postinst legt die nicht an. Fuer Nutzer: Script oder docs.
- **R1 (deepseek_r1:1.5b) ungetestet auf 5.3.0** — in 5.1.1 deterministischer
  SEGV, in 5.3.0 evtl. gefixt, aber nicht verifiziert. Qwen2.5 reicht vorerst.
- **`moloch.service` ExecStartPre macht `pkill -9 hailo-ollama`** — war fuer
  das alte VDevice-Konflikt-Problem. Mit SHARED nicht mehr noetig, kann spaeter
  entfernt werden (fuehrt aktuell zu 30s Verzoegerung bei Moloch-Restart bis
  hailo-ollama wieder erreichbar ist).
- **Modell-Store auf Root-Dateisystem** (`/` hat 384 GB frei, aber
  `/mnt/moloch-data` haette 426 GB). Bei mehreren Modellen evtl. migrieren.

### Rollback-Pfad (falls noetig)

Komplettes 5.1.1-Set in `/home/molochzuhause/Downloads/hailo_backup/`:
```bash
sudo systemctl stop moloch hailo-ollama
sudo apt install -y ~/Downloads/hailo_backup/{h10-hailort*,hailo-tappas-core_5.1.0*,python3-h10-hailort*,python3-hailo-tappas*,hailo_gen_ai_model_zoo_5.1.1*,hailo-models*}.deb
sudo reboot
```
Fuehrt zurueck auf 5.1.1, aber dann wieder Error 74 bei paralleler Nutzung.

---

## SESSION 17 — MOLOCH UNABHÄNGIG VON DISPLAY/GUI

**User-Wunsch:** Moloch soll immer aktiv bleiben, auch wenn Bildschirm aus oder
GUI-Fenster geschlossen. "Nur Service muss immer laufen."

**Befund der Analyse:** Backend war bereits entkoppelt. `moloch.service` ist
System-Service (`WantedBy=multi-user.target`), kein `graphical.target`-Bezug,
kein DISPLAY im Environment. `panel_main._on_close()` (Zeile 579-591) stoppt nur
Preview+Avatar, sendet keinen IPC-Befehl. SHM-Frame-Writer läuft unabhängig von
Reader. → Der wahrgenommene "Tracking pausiert"-Effekt war Wayland Screen-
Blanking + eingefrorene Preview-Anzeige, nicht echter Service-Stop.

### Änderungen dieser Session

| Datei | Zweck | Agent |
|-------|-------|-------|
| `scripts/verify_headless_runtime.py` (NEU) | 5-Min Status-Monitor (CSV) zum Beweisen dass Service durchläuft | watchdog |
| `~/.config/labwc/autostart` | `wlopm --on *` beim Desktop-Login, swayidle ausgeklammert | hardware |
| `~/.config/autostart/moloch-no-sleep.desktop` (NEU) | XDG-Autostart mit Wayland+X11 Fallback | hardware |
| `/boot/firmware/cmdline.txt` | `consoleblank=0` angehängt (TTY-Blanking aus) | hardware |
| `/etc/systemd/system/moloch.service` | `KillMode=process`, `RestartSec=10→5` (Restart=always war schon da) | service |

**Backup der moloch.service:** `/etc/systemd/system/moloch.service.bak_20260418_105901`
**Backup cmdline.txt:** `/tmp/cmdline.txt.bak` (vor Reboot sichern!)

### Regel für Markus
> **Screen aus ≠ Moloch aus.** Bei Zweifel: `mcp__moloch__moloch_status()` —
> nicht GUI öffnen. Die GUI ist nur Anzeige, nicht das Gehirn.

### Verifikation nach Reboot
1. `systemctl is-active moloch` → active
2. `systemctl show moloch -p Restart,RestartSec,KillMode` → always/5s/process
3. `mcp__moloch__moloch_audit()` → 62/62 PASS
4. `mcp__moloch__moloch_status()` → FPS ≥ 18
5. Realer Test: `python3 ~/moloch/scripts/verify_headless_runtime.py --duration 300`
   während Markus GUI öffnet/schließt und Bildschirm aus/an schaltet. CSV in
   `logs/verify_headless_<ts>.csv`. Erwartung: FPS-Stalls=0, status_age_ms<2000.

### Offen aus Vorsession (unverändert)
- B2 Moloch halluziniert Quellen → deepseek
- B4 News bei generischen Anfragen veraltet → autonomy
- Slider-Drift yolo_conf Max auf 0.7 → gui
- YOLOWorldWorker ~70 % integriert (Frequency/PFrame/IPC fehlen) → vision
- IPC Race poll_commands() löscht Datei vor Verarbeitung → service

---

## SESSION 16 — AUDIT & RACE-FIX (archiviert, Commit 0cb50d4 | 62/62 PASS)

System-Zustand: grün. FPS 20, RAM 39 %, 0 SEGV, 0 Worker-Errors, Face-ID
erkennt `markus` (Sim ~0.55-0.64). LLM/hailo-ollama antwortet, alle 8 NPU-Worker
aktiv (Activity/Depth/Face/Hand/PersonAttr/Pose/ReID/YOLOWorld).

### Gefundener Fehler
MCP-Audit zeigte **1 FAIL: "PerceptionMemory initialisiert — Kein Init-Log
gefunden"** bei ansonsten 61/62 PASS.

**Root Cause:** Race zwischen Service-Start und Audit-Test.
- Service-Start: 12:14:29 (Init-Log geschrieben)
- Vorheriger Audit-Lauf: 12:12:11 (2 min davor)
- Test `scripts/moloch_audit.py:797-809` greppt journalctl nach
  `PerceptionMemory.*Initialisiert` — Zeile existierte zu dem Zeitpunkt noch nicht.

### Fix (Commit 0cb50d4)
`scripts/moloch_audit.py` Test um Status-Fallback erweitert: wenn
journalctl nichts findet, prüft er `/dev/shm/moloch_status.json` auf
`face_id` / `person_present` / `active_models`. Wenn PerceptionMemory
Output liefert, läuft es per Definition. Bei totem Modul bleibt es FAIL.

Keine ROT-Dateien angefasst, kein Service-Restart nötig.

### Rollback-Obduktion (daf3e76, 2026-04-13)
"Kamera-Steuerung + Bild waren kaputt" nach Vision-Pause-Experiment
(`pause_for_llm` / `resume_after_llm` + disabled_workers). Nicht wiederholen
ohne neuen Plan. Danach `a701a38` → spontane Kommentare deaktiviert.

### Offen (aus voriger Session)
- B2 Moloch halluziniert Quellen → deepseek
- B4 News bei generischen Anfragen veraltet → autonomy
- Slider-Drift yolo_conf Max auf 0.7 → gui
- YOLOWorldWorker ~70 % integriert (Frequency/PFrame/IPC fehlen) → vision
- IPC Race poll_commands() löscht Datei vor Verarbeitung → service
- T2 Agent-Events nicht publiziert (rein architekturell, kein Runtime-Impact)

---

## SESSION-ERGEBNISSE (9 Commits)

| Fix | Commit | Datei |
|-----|--------|-------|
| HandWorker-Dispatch bei Nahaufnahme + pose_age Tolerance | 109ce71 | tappas_pipeline.py |
| PoseWorker conf_thresh 0.3->0.2, HandWorker wrist-vis 0.15->0.1 | 8955786 | pose_worker.py |
| panel_preview Pose-Keypoint Visibility 0.2->0.1 | a387ad0 | panel_preview.py |
| Night Cycle startet ab 23:00 + echte Daten in LLM-Reflexion | f41c05c | night_cycle.py |
| Shutdown-Cleanup loggt Fehler statt sie zu verschlucken | 3e884f9 | moloch_service.py |
| NPU-Extras VDevice/VLM Release Fehler-Logging | 2a6a553 | npu_extras.py |
| head_pitch/head_yaw in TAPPAS-PFrame + Worker-Health Logging | 8012b76 | tappas_pipeline.py |
| deepseek_client.py entfernt (230 Zeilen toter Code) | 5f69a62 | deepseek_client.py |
| ocr_texts in TAPPAS-PFrame setzen | 2c9c80d | tappas_pipeline.py |

### Wichtige Aenderungen:
- **Pose+Hand Landmarks**: 3 Root Causes behoben (HandWorker-Guard, pose_age, Thresholds)
- **Night Cycle**: Startet jetzt um 23:00 statt 00:00. LLM-Reflexion mit echten Tagesdaten.
- **Shutdown Logging**: 15x except:pass -> logger.warning/error (moloch_service + npu_extras)
- **PFrame komplett**: head_pitch/head_yaw + ocr_texts jetzt im TAPPAS-Modus verfuegbar
- **Systemscan**: 7 versteckte Bugs gefunden und behoben (Key-Mismatches, stille Fehler, toter Code)

---

## OFFENE BUGS

- **B2**: Moloch halluziniert Quellen ("Laut Suchergebnissen")
- **B4**: News bei generischen Anfragen veraltet (Google News RSS)
- **Slider-Drift**: yolo_conf Slider-Max auf 0.7 begrenzen (GUI)
- **YOLOWorldWorker**: Integration ~70% (keine Frequency, kein PFrame-Ort, kein IPC)
- **IPC Race**: poll_commands() loescht Datei VOR Verarbeitung
- **T2**: Agent-Events (build/test/review/chaos) nie publiziert — geplante Architektur, kein Runtime-Impact

---

## WAS BEREITS GEFIXT IST — NICHT NOCHMAL ANFASSEN

- Pose/Hand Landmarks (109ce71, 8955786, a387ad0)
- Night Cycle Date+LLM (f41c05c)
- Shutdown Logging (3e884f9, 2a6a553)
- head_pitch/head_yaw + ocr_texts (8012b76, 2c9c80d)
- Worker-Health Error-Dict (8012b76)
- deepseek_client.py entfernt (5f69a62)
- Pan-Vorzeichen, ArcFace, hailooverlay, Status-JSON Deadlock
- keywords.json Komma, PTZ Error-Handling
