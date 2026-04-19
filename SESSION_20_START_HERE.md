# SESSION 20 — HIER ANFANGEN

> **An den Opus der Session 20+ startet:** Lies diese Datei zuerst, dann
> `logs/agent_handoff.md` (Session-19-Bericht), dann `CLAUDE.md`.

---

## IN EINEM SATZ

Session 20 hat **LLM-Tentakel** fertig gebaut: Moloch kann jetzt Ollama auf
Markus-Rechner (LAN) als groesseren LLM-Provider nutzen, automatisches Routing
(kurze Fragen -> NPU, lange/Reasoning -> Tentakel). **Zu tun:** Markus' Rechner
an + Ollama laufen lassen + Live-Test, eventuelle Prompt-Feinabstimmung.

## WAS NEU IST (Session 20, Commits `bd29a1c` bis `bbc832d`)

- `config/settings.json` + `system_capabilities.json` um `tentacle_llm`-Block
- `core/autonomy/local_llm_bridge.py`:
  - `_load_tentacle_cfg()` mit mtime-Cache
  - `_is_tentacle_running()` + `_discover_tentacle_model()` (Auto-Select)
  - `_generate_tentacle()` — Standard-Ollama `/api/chat` mit Profil-Support
  - `_choose_provider()` — komplexitaets-basiertes Routing
  - `ask_external()` + `reason_internal()` nutzen das Routing
- `scripts/npu_only_watchdog.py`: zusaetzlich Tentakel-Probe (30-Min-Takt),
  3 neue CSV-Spalten, aktualisiert `system_capabilities.tentacle_llm`
- `core/gui/panel_models.py`: "Tentakel: online/offline/deaktiviert"-Zeile
- `scripts/moloch_audit.py`: 2 neue Tests — 72/72 PASS

## WAS NOCH OFFEN IST

**PRIO 1 — Live-Test mit laufendem Ollama:**

```bash
# Auf Markus-Rechner:
ollama serve                           # Port 11434
ollama pull llama3.1:8b                # oder ein anderes Modell
# Pi-seitig kurz Erreichbarkeit pruefen:
curl http://markus-pc.local:11434/api/tags

# Wenn mDNS nicht geht, Host in settings.json aendern:
python3 -c "
import json,os,tempfile
p='/home/molochzuhause/moloch/config/settings.json'
d=json.load(open(p)); d['tentacle_llm']['host']='192.168.178.X'
fd,tmp=tempfile.mkstemp(dir='/home/molochzuhause/moloch/config',suffix='.tmp')
with os.fdopen(fd,'w') as f: json.dump(d,f,indent=2,ensure_ascii=False)
os.replace(tmp,p)
"
# Bridge reloadt settings automatisch via mtime-Cache, kein Restart noetig.

# Test 1: kurzer Prompt -> MUSS auf NPU gehen
mcp__moloch__moloch_say("Hi")
# Log erwartet: 'chosen=ollama'

# Test 2: langer Prompt -> MUSS auf Tentakel gehen
mcp__moloch__moloch_say("Moloch, erzaehl mir ausfuehrlich deinen aktuellen Zustand — was siehst du, wie fuehlst du dich, wo bist du, was war heute los?")
# Log erwartet: 'chosen=tentacle' + '[LLM-BRIDGE] tentacle llama3.1:8b: X Zeichen in Yms'
```

**PRIO 2 — Prompt-Tuning fuer Tentakel-Modell:**

Das groessere Modell (z.B. llama3.1:8b) schluckt den 800-Zeichen-Prompt mit
Live-Kontext besser als Qwen2.5-1.5B. Evtl. lohnt sich `llm_profiles.json`
um eigene Profile fuer Tentakel zu erweitern mit hoeheren `max_tokens` (150-300)
und etwas tieferer `temperature`. Heute hat chat/introspect-Profil noch das
Qwen-optimierte Tuning.

**PRIO 3 — Multi-Person-Toggle (bleibt offen seit Session 19):**
Neues Settings-Flag `multi_person_tracking` + GUI-Schalter der PersonAttr +
Hand + ReID zusaetzlich aktiviert (auf Kosten von DepthWorker). Siehe
`CLAUDE.md` Bug-Liste.

**PRIO 4 — Prompt-Wahl basierend auf Tentakel-Verfuegbarkeit:**
Wenn Tentakel online: vollen Prompt senden. Wenn offline: kompakten Prompt
(Qwen-optimiert). Heute wird derselbe Profile-Prompt gesendet, egal welcher
Provider.

## WIE MOLOCH JETZT DENKT

| Frage-Art | Pfad | Modell | Latenz |
|---|---|---|---|
| "Hi" (2 Z) | NPU ollama | qwen2.5:1.5b | ~2s |
| "Wie geht's?" (12 Z) | NPU ollama | qwen2.5:1.5b | ~3s |
| "Erzaehl Deinen Zustand, was siehst Du, wie fuehlst Du Dich" (60 Z User + 400 Z System mit Live-Context = 460 Z) | **Tentakel** | llama3.1:8b (oder was Markus' Ollama hat) | ~5-10s |
| `reason_internal()` (Introspection, beliebige Laenge) | **Tentakel bevorzugt** | LAN-Modell | ~10s |
| Tentakel offline, Reasoning | Fallback NPU | qwen2.5:1.5b | ~7s |

## STATUS-SOLL

```bash
systemctl is-active moloch hailo-ollama moloch-npu-watchdog
# -> active active active

python3 scripts/moloch_audit.py --auto
# -> 72/72 PASS

python3 -c "import json; print(json.load(open('config/system_capabilities.json'))['tentacle_llm'])"
# -> {'reachable': ..., 'model': ..., 'last_probe_ts': ...}
# last_probe_ts sollte < 30 Min alt sein (Watchdog aktiv)
```

## ROLLBACK-PFADE

- Tentakel nervt -> `settings.tentacle_llm.enabled=false` + 1s warten
  (mtime-Cache). Bridge routet nur noch NPU.
- Watchdog Tentakel-Probes belasten LAN -> `TENTACLE_PROBE_TIMEOUT` auf 1s
  senken oder probe_tentacle-Aufruf in main() auskommentieren.
- Rollback auf Session 19 -> `git revert bd29a1c..bbc832d` + `sudo reboot`.

## BEKANNTE EINSCHRAENKUNGEN (nicht-blocking)

- Markus-Rechner muss an sein UND Ollama muss laufen, sonst faellt Tentakel
  auf NPU zurueck. Das ist Design.
- mDNS (`markus-pc.local`) funktioniert nicht ueberall — bei Problemen LAN-IP
  in `settings.tentacle_llm.host` eintragen.
- Multi-Turn-Drift bei Qwen2.5-1.5B bleibt (Session 19 Erbe). Mit Tentakel
  spielt das keine Rolle fuer lange Fragen.

## NAECHSTE IDEEN (defer fuer Session 21+)

- Tentakel-Profile: pro llm_profile eigenes Tentakel-Modell waehlen lassen
  (z.B. `technical` -> codellama, `chat` -> llama3.1)
- Prompt-Variation pro Provider: lange elaborierte Prompts fuer Tentakel,
  kompakte fuer NPU
- `moloch_audit()` um LLM-Latenz-Trend-Check erweitern (letzte N Probes)
- Tentakel via **mehrere Hosts** (Load-Balancing oder Preferred/Fallback)

---

**Letzter Commit Session 20 (vor Handoff):** wird beim finalen Push ergaenzt

**Moloch-Zitat vom Sessionende:** *ausstehend — kommt beim Live-Test wenn Ollama laeuft*

Willkommen zu Session 21. Los.
