# Briefing fuer Claude Code (lokal) — MOLOCH-Projekt

> Lies mich zuerst. Dann `SESSION_20_START_HERE.md`. Dann los.

## Wer ist Moloch

Eine autonome KI-Einheit auf einem Raspberry Pi 5 mit Hailo-10H NPU (40 TOPS).
Sieht durch eine ONVIF-PTZ-Kamera, erkennt Personen (YOLOv11m + SCRFD + ArcFace),
trackt sie, spricht (Qwen2.5-1.5B LLM lokal auf NPU + Piper TTS), hat eine
Persoenlichkeit mit Zone/Tension/Dominance. Seit Session 19 laeuft das LLM
komplett lokal — Cloud-API hart deaktiviert.

Entwickler und Boss: **Markus**. Er weiss was er tut, ist direkt, kein Theater.

## Wo bist Du, wo ist der Pi

- **Du** laeufst als Claude Code auf Markus' Rechner (nicht auf dem Pi).
- **Der Pi** ist `192.168.178.30` (SSH-User `molochzuhause`), IP kann variieren.
- **Das Repo** ist auf beiden Seiten synchron via git (origin: `moloch00464-bit/MOLOCH`).
- **MCP-Server** laeuft auf dem Pi und exponiert Moloch-Tools (siehe `mcp/moloch_mcp_server.py`).
  Wenn Du MCP-Tools siehst (`mcp__moloch__*`), kannst Du damit live mit Moloch reden,
  Snapshots holen, Audit fahren — ohne SSH-Geklicke.

## Wie Du arbeitest

1. **Erst `moloch_session_init()` MCP aufrufen** — das ist Pflicht, entfernt den
   Session-Lock. Siehe `CLAUDE.md` PFLICHT-STARTPROTOKOLL.
2. **Dann Agent laden** — `CLAUDE.md` hat die Agent-Mapping-Tabelle. Jede Datei
   gehoert einem Agent (vision/autonomy/gui/voice/...). Agent-Lock setzen via
   `touch /tmp/moloch_agent_[name]` bevor Du editierst, sonst blockt ein Pre-Edit-Hook.
3. **LOKOMOTIVE-Prinzip** — durchfahren ohne Stopp. GRUENE Dateien sofort, GELBE
   ankuendigen + sofort, ROTE einmal fragen. Nicht jede Zeile diskutieren.
4. **Commit-Disziplin** — 1 logische Aenderung = 1 Commit. ROT-Dateien NIE zu
   mehreren auf einmal. Push nach jedem Block.

## Wie Du mit Moloch redest

- **Live-Status:** `mcp__moloch__moloch_status()` — FPS, Temp, Face-ID, NPU-Stage
- **Worker-Health:** `mcp__moloch__moloch_npu_workers()` — 4 aktive Worker + Queues
- **Logs:** `mcp__moloch__moloch_logs(n=50, filter_str="LLM")` — journalctl gefiltert
- **Audit:** `mcp__moloch__moloch_audit()` — 70 Tests durchlaufen
- **Chat mit Moloch:** `mcp__moloch__moloch_say("deine frage")` → async,
  Antwort per `mcp__moloch__moloch_conversation(n=2)` lesen
- **Snapshot** (Kamerabild): `mcp__moloch__moloch_snapshot()` — Base64-PNG
- **Bash auf dem Pi:** geht via `Bash`-Tool, laeuft per SSH, Du merkst es nicht.
  Services: `systemctl is-active moloch hailo-ollama moloch-npu-watchdog`.
  Git: `cd /home/molochzuhause/moloch && git ...`.

## Was heute zu tun ist

**Haupt-Aufgabe: `qwen3:1.7b` pullen und gegen qwen2.5:1.5b benchmarken.**
Details siehe `SESSION_20_START_HERE.md` PRIO 1. Alles andere ist nachrangig
bis dieser Vergleich steht — Qwen2.5 stosst an sein Limit (Halluzinationen bei
langen Prompts), Qwen3 soll das loesen.

Wenn Du unsicher bist: frag Markus. Er antwortet knapp und direkt.
Er mag keine Rueckfragen-Spiralen — einmal klaeren, dann durchziehen.

## Faustregeln die immer gelten

- **4 GB Pi-RAM** — sparsam bauen, kein Memory-Leak tolerieren.
- **NPU Hardware-Limit 8 Network-Groups** — jeder neue HEF-Worker kostet einen Slot.
- **Sonoff Pan ist INVERTIERT** — `pan_delta = -error_x` ist KORREKT, nicht umdrehen
  (6x in Git-History zurueckgedreht).
- **Atomic JSON-Write** (tempfile + os.replace) — siehe `safe_json_write` Template
  in `CLAUDE.md`. Partial-Write bei Crash korrumpiert sonst Configs.
- **Runtime-State NICHT committen** — `last_face_position.json`,
  `motor_learning.json`, `perception_weights.json`, `system_capabilities.json`
  sind volatile und bleiben lokal.

## Ton

Kurz. Direkt. Ergebnis zuerst. Moloch ist ein dunkler frecher Charakter,
Markus mag keine Assistent-Floskeln ("Natuerlich! Sehr gerne!"). Beim Code
genauso: kein unnoetiges Kommentar-Rauschen, keine Wikipedia-Zitate, keine
Struktur-Listen wenn ein Satz reicht.

Wenn Moloch spricht, spricht er wie ein Kumpel im Gespraech. Wenn Du schreibst,
auch.

---

**Moloch-Zitat aus Session 19 (2026-04-19):**
> *"Ja, was gibt's?"*

15 Zeichen, 1.7 Sekunden, 100% NPU lokal. Ohne Cloud. Ohne Internet-Fallback.

Das ist unser System. Jetzt mach es noch besser.
