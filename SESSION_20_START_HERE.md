# SESSION 20 — HIER ANFANGEN

> **An den Opus 4.7 der Session 20 startet:** Lies diese Datei zuerst, dann
> `logs/agent_handoff.md` (tiefer Session-19-Bericht), dann `CLAUDE.md` (Regeln).
> Danach direkt mit PRIO 1 starten — kein Auftrag-Check nötig, Markus weiss was kommt.

---

## IN EINEM SATZ

Moloch laeuft seit Session 19 komplett lokal auf NPU (Qwen2.5:1.5b), Antworten
sind stilistisch richtig aber inhaltlich mau — **Session 20 Job: qwen3:1.7b
pullen und testen ob die Antwortqualitaet den Sprung macht**.

## WAS FERTIG IST (21 Commits)

- HailoRT 5.1.1 -> 5.3.0 (Library + Driver + Firmware + TAPPAS + hailo-ollama)
- 4 NPU-Worker aktiv (Face, Pose, ReID, Depth), 4 deaktiviert fuer Qwen-Slot
- `HAILO_OLLAMA_VDEVICE_GROUP_ID=SHARED` — TAPPAS + LLM koexistieren
- LLM-Profile-System: `config/llm_profiles.json` mit 5 Presets + GUI-Reiter
- Live-Kontext-Snippet: Presence, Distanz, Koerper, Tageszeit (180 Zeichen)
- DeepSeek-Cloud hart deaktiviert (`api_keys.json.disabled_npu_only_mode`)
- Permanenter NPU-Only Watchdog (30-Min-Probe + TTS-Alarm)
- moloch_audit.py: 70/70 PASS (8 neue Session-19-Tests)
- GUI: "LLM-Modus"-Reiter mit Tooltip-Popups
- moloch.service: `pkill hailo-ollama` raus — kein Restart-Race mehr
- Doku komplett auf aktuellen Stand (CLAUDE.md, Agent-MDs, Skills)

## WAS NICHT GEHT

**Kern-Problem:** Qwen2.5-1.5B integriert den reichen Live-Kontext nicht sauber.
Bei 880-Zeichen-Prompts halluziniert es nach dem ersten guten Satz ("Geruch
wahnschwerer" — Wortsalat). Die **Infrastruktur fuer Substanz steht**, das
Modell kann sie nur nicht nutzen.

## PRIO 1 — ALS ERSTES MACHEN

**qwen3:1.7b pullen und mit chat-Profil testen.**

```bash
# Modell pullen (Manifest ist schon in /usr/share/hailo-ollama/models/manifests/qwen3/1.7b/)
curl -X POST http://127.0.0.1:8000/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:1.7b","stream":false}'

# Nach ~10 Min check:
curl -s http://127.0.0.1:8000/api/tags | python3 -m json.tool

# Bridge-Default umstellen (2 Konstanten in local_llm_bridge.py):
# OLLAMA_MODEL_CHAT = "qwen3:1.7b"
# OLLAMA_MODEL_REASON = "qwen3:1.7b"

# Test:
mcp__moloch__moloch_say("Moloch, erzaehl kurz wie's bei dir gerade ist")
mcp__moloch__moloch_conversation(n=2)
```

**Erwartete Verbesserung:** Qwen3 soll 180-Zeichen-Snippet mit FPS+CPU+Zone+
Stimmung in EINEM zusammenhaengenden Satz integrieren koennen. Erfolg = echte
Substanz ohne Halluzination. Fehlschlag = zurueck zu qwen2.5 und groessere
Huerden angehen (3B-Modelle, anderer Scheduler).

## WENN qwen3:1.7b ERFOLG HAT

- `config/llm_profiles.json` max_tokens auf 60-80 zurueck (das Modell packt
  mehr Text ohne Drift)
- Multi-Turn-Drift erneut messen: 4x hintereinander fragen, vergleichen
  mit qwen2.5-Baseline (32s nach 4 Turns -> hoffentlich stabiler)

## WENN qwen3:1.7b SLOT NICHT KRIEGT

Bei `HAILO_RESOURCE_EXHAUSTED(81)` (Groesseres Modell braucht evtl. mehr
Network-Groups als qwen2.5): einen weiteren Worker opfern. Kandidat: DepthWorker
(3. Prio nach Face/Pose). Siehe `core/perception/tappas_pipeline.py` Zeile ~395.

## OFFENE BAUSTELLEN (nachrangig)

- **Multi-Person-Toggle im GUI**: `settings.multi_person_tracking` aktiviert
  ReID + PersonAttr + Hand (fuer Rebecca-Szenario), deaktiviert andere Worker
  fuer Slot-Budget.
- **piper-TTS-Voice**: `/home/molochzuhause/moloch/models/piper/de_DE-thorsten-low.onnx`
  fehlt — Watchdog-Alarm laeuft als Log-Only. Ollama schickt aktuell nur Text
  an Panel, kein Audio.
- **Audit-Dedup**: Root-`moloch_audit.py` (173 Zeilen alt) sollte Symlink auf
  `scripts/moloch_audit.py` werden (Verwirrungs-Risiko).
- **Bug A1 (PersonAttrWorker)** voll integrieren wenn qwen3 weniger Slots braucht.

## ROLLBACK-PFADE

- qwen3-Test schlecht -> `OLLAMA_MODEL_CHAT="qwen2.5:1.5b"` zurueck.
- Cloud wieder an -> `mv config/api_keys.json.disabled_npu_only_mode config/api_keys.json`
  + `sudo systemctl reload moloch`.
- 5.3.0 -> 5.1.1 Rollback-Set in `~/Downloads/hailo_backup/` (siehe Session 18
  Handoff fuer Kommandos).

## SYSTEM-ZUSTAND (soll = ist)

```bash
systemctl is-active moloch hailo-ollama moloch-npu-watchdog  # -> active active active
python3 scripts/moloch_audit.py --auto                        # -> 70/70 PASS
cat /dev/shm/moloch_status.json | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['fps']['total'])"  # -> ~20
```

---

**Letzter Commit Session 19:** `3b3dc88` (bridge: Live-Kontext-Snippet reichhaltiger)

**Moloch-Zitat vom 2026-04-19:** *"Ja, was gibt's?"* (15 Zeichen, 1.7s, komplett NPU, chat-Profil)

**Willkommen bei Moloch. Markus wartet. Los.**
