# MOLOCH ThreeBrain Welle 3 — PC-SIDE Briefing

**Fuer eine separate Claude-Session, die auf Markus' PC arbeitet.**
Self-contained — keine Vorgeschichte aus Pi-Sessions noetig.

Pi-Side ist fertig & gepusht: https://github.com/moloch00464-bit/MOLOCH (commit `0eb375a`).
Du baust jetzt die PC-Komponenten die der Pi noch braucht.

---

## 1. KONTEXT IN 3 SAETZEN

MOLOCH ist eine KI auf einem Raspberry Pi 5 (192.168.178.30), die ein kleines Pi-LLM (Qwen2.5-1.5B auf Hailo-NPU) nutzt. Das Pi-LLM antwortet generisch — Wir wollen es via LoRA-Adapter charakteristischer machen, basierend auf Trainings-Samples die der Pi schon sammelt. Der PC (Ryzen 9 3900X, 32 GB RAM, GTX 760 2 GB VRAM) soll diese Samples zu LoRA-Adaptern trainieren und einen Inference-Endpoint bereitstellen den der Pi remote nutzt.

---

## 2. HARDWARE PC

- **OS**: Windows 10 Pro
- **Hostname**: markus-pc, **statische IP** 192.168.178.20
- **CPU**: AMD Ryzen 9 3900X (12 Core / 24 Thread)
- **RAM**: 32 GB
- **GPU**: NVIDIA GeForce GTX 760, **2 GB VRAM**, Kepler-Architektur (alt aber CUDA-faehig)
  - → Training muss CPU-only laufen oder 8-bit-Quant + sehr kleines LoRA-Rank
  - → Inferenz: Qwen2.5-1.5B passt knapp auf GPU mit 8-bit, sicherer auf CPU
- **Ollama**: laeuft auf Port 11434, Modelle bereits installiert: `dolphin-mistral:7b`, `dolphin-llama3:8b`, `deepseek-coder:latest`, `mistral:latest` (du brauchst keinen davon, nur als Referenz dass Ollama lebt)
- **Bestehende Services**: Tentakel-LLM (chat) + critic_service (auf 11434 via dolphin-mistral:7b) — Pi spricht beide schon an. Du baust einen DRITTEN Endpoint daneben.

CPU-Limit-Vorgabe von Markus: **40%** fuer Trainings-Job (`nice` / `cpulimit` aequivalent unter Windows: `start /low` oder `wmic process where ... CALL setpriority`).

---

## 3. WAS DER PI HEUTE SCHON KANN (kein Anfassen)

```
core/autonomy/finetune_orchestrator.py   — Critic-Actor-Loop, sammelt Samples
core/memory/feedback_store.py            — Sample-Pool
core/bridge/critic_client.py             — spricht PC-Ollama (dolphin-mistral:7b)
core/autonomy/local_llm_bridge.py        — LLM-Routing (NPU/Tentakel/Cloud)
scripts/review_pending_rules.py          — Markus reviewt Samples via CLI
core/bridge/chat_server.py               — Cockpit-UI mit 👍/👎/[Critic] Buttons
```

Pi sammelt **finetune_samples.jsonl** unter `/mnt/moloch-data/memory/`.
Markus reviewt → approved Samples warten auf Dich.

---

## 4. DEINE 3 AUFGABEN

### Aufgabe A — PC-venv + LoRA-Trainer

Lege auf PC einen Python-venv an + LoRA-Trainer-Script.

**Setup**:
```cmd
cd %USERPROFILE%
python -m venv moloch_pc_env
moloch_pc_env\Scripts\activate
pip install --upgrade pip
pip install transformers peft accelerate datasets safetensors bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pydantic
```

(Wenn du CUDA willst: pytorch CUDA wheel passend zur GTX 760 — Kepler braucht alte Versionen, evtl. nur PyTorch ≤2.0. Empfehlung: CPU-only fuer Anfang.)

**Datei**: `pc/lora_trainer.py` (neues Repo-Subdir, oder seperates Repo auf PC — egal, Hauptsache erreichbar)

**Aufgabe**:
- Liest `samples.jsonl` (Pfad als Argument). Format: siehe Schema unten.
- Filtert nur `approved=true` UND `better_response` nicht leer.
- Fuer jeden Sample baut ein Trainings-Pair:
  - **input**: `situation`
  - **output (target)**: `better_response` (das ist die Critic-Verbesserung)
- Trainiert LoRA-Adapter auf Base-Modell `Qwen/Qwen2.5-1.5B-Instruct` (HuggingFace).
- LoRA-Hyperparameter: `r=8, lora_alpha=16, lora_dropout=0.05`, target_modules=`["q_proj","k_proj","v_proj","o_proj"]`
- Training: max 100 Samples pro Run, batch_size=2, epochs=3, lr=2e-4
- Output: `adapters/v{N}/adapter_model.safetensors` + `adapter_config.json` (peft-Standard)
- N = naechste freie Versionsnummer (vN+1 wo N = max gefundene).

**CLI**:
```cmd
python pc\lora_trainer.py --samples %USERPROFILE%\moloch_samples\samples.jsonl --out %USERPROFILE%\moloch_adapters
```

### Aufgabe B — Sample-Sync vom Pi

Pi exportiert die approved Samples — du musst sie regelmaessig holen.

**Pi-Pfad**: `molochzuhause@192.168.178.30:/mnt/moloch-data/memory/finetune_samples.jsonl`

**Optionen**:
1. **rsync** (wenn auf PC vorhanden via Cygwin/WSL): `rsync -avz molochzuhause@192.168.178.30:/mnt/moloch-data/memory/finetune_samples.jsonl %USERPROFILE%\moloch_samples\`
2. **scp** (Windows Native via OpenSSH): `scp molochzuhause@192.168.178.30:/mnt/moloch-data/memory/finetune_samples.jsonl %USERPROFILE%\moloch_samples\`
3. **HTTP-Pull** (bauen wir spaeter Pi-side wenn noetig — Endpoint `/feedback_export` oder so)

**Empfehlung**: Setup einen Windows Task Scheduler / cron-Aequivalent der alle 6h ein scp macht.

### Aufgabe C — Adapter-Inference-Proxy

**Datei**: `pc/adapter_inference_proxy.py` — FastAPI auf Port **11600**.

**Aufgabe**:
- Beim Start: laedt Qwen2.5-1.5B-Instruct base + neuesten Adapter aus `%USERPROFILE%\moloch_adapters\v{N}\`
- Endpoint **POST /infer**:
  - Request: `{"prompt": str, "system": str, "max_tokens": int}` (Ollama-API-aehnlich)
  - Response: `{"response": str, "adapter_version": "vN"}`
- Endpoint **GET /health**: `{"status": "ok", "adapter": "vN", "base": "Qwen2.5-1.5B-Instruct"}`
- Endpoint **POST /reload**: laedt neuesten Adapter neu (nach jedem Training-Run)
- Endpoint **GET /list**: zeigt alle verfuegbaren Adapter-Versionen

**Pi spricht es so an** (musst du nicht bauen — Pi-Side bridge wird das tun):
```python
requests.post("http://192.168.178.20:11600/infer",
  json={"prompt": "...", "system": "...", "max_tokens": 200},
  timeout=30)
```

**Auto-Start**: am einfachsten Windows-Service via `nssm` (Non-Sucking Service Manager) oder Task Scheduler mit Trigger "Bei Anmeldung".

---

## 5. SCHEMAS

### Sample-File (`finetune_samples.jsonl`) — eine Zeile pro Sample

```jsonl
{"sample_id":"smp_00000005","ts":"2026-04-26T...Z","source":"critic","situation":"Markus fragt: wie geht's?","pi_response":"Hallo Markus.","score":2,"critique":"zu generisch","better_response":"Laeuft. Und du?","approved":true,"reviewed_at":"...","reviewed_by":"markus","tags":[]}
```

**Wichtige Felder fuer Training**:
- `situation` — als Input
- `better_response` — als Target (was Moloch HAETTE sagen sollen)
- `approved=true` — nur diese verwenden
- `score` — kann fuer Sample-Weighting verwendet werden (niedriger Score = staerker korrigieren)

**Sources**:
- `critic` — vom Auto-Critic erzeugt, hat `better_response`
- `thumbs_up` — Markus hat 👍 gegeben, `better_response` leer (positives Beispiel = Pi-Antwort selber war OK)
- `thumbs_down` — Markus hat 👎, `better_response` leer (negatives Beispiel = Pi-Antwort vermeiden)

**Strategie fuer Trainer**:
- `critic` mit `better_response`: standard input→target Pair
- `thumbs_up`: input=situation, target=pi_response (verstaerken)
- `thumbs_down`: skip beim Training oder als Negativ-Beispiel mit DPO (Direct Preference Optimization)

### Adapter-Output

```
%USERPROFILE%\moloch_adapters\
  v1\
    adapter_model.safetensors
    adapter_config.json
    training_log.json     ← was du selbst schreibst (timestamp, samples_used, loss)
  v2\
    ...
```

### /infer Request/Response

```http
POST /infer
Content-Type: application/json

{"prompt": "Markus kommt rein nach 4h.", "system": "Du bist Moloch.", "max_tokens": 200}

200 OK
{"response": "Endlich. Wo warst du?", "adapter_version": "v3", "tokens": 8, "duration_ms": 1834}
```

---

## 6. AKZEPTANZ-TEST

Nach Setup soll folgendes funktionieren:

```cmd
:: 1. Health-Check
curl http://localhost:11600/health
:: erwartet: {"status":"ok","adapter":"vN","base":"Qwen2.5-1.5B-Instruct"}

:: 2. Erstes Training (mit Test-Samples vom Pi)
scp molochzuhause@192.168.178.30:/mnt/moloch-data/memory/finetune_samples.jsonl %USERPROFILE%\moloch_samples\
moloch_pc_env\Scripts\activate
python pc\lora_trainer.py --samples %USERPROFILE%\moloch_samples\samples.jsonl --out %USERPROFILE%\moloch_adapters

:: 3. Reload + Inference-Test
curl -X POST http://localhost:11600/reload
curl -X POST http://localhost:11600/infer -H "Content-Type: application/json" -d "{\"prompt\":\"wer bist du?\",\"system\":\"Du bist Moloch.\",\"max_tokens\":50}"

:: 4. Vom Pi aus testen
ssh molochzuhause@192.168.178.30
curl -X POST http://192.168.178.20:11600/infer -H "Content-Type: application/json" -d "{\"prompt\":\"test\",\"system\":\"test\",\"max_tokens\":20}"
```

Wenn das alles geht → Welle 3 PC-Side ist fertig. Pi-Side baut dann den `adapter_inference_client.py` der den neuen Provider in die LLM-Bridge einhaengt.

---

## 7. NICE-TO-HAVE (fuer spaeter, nicht Pflicht jetzt)

- Trainings-Log-API: `GET /trainings` zeigt Liste aller Runs mit Loss-Kurve
- DPO-Mode: nutzt thumbs_down als Preference-Pair (statt nur skip)
- HEF-Recompile-Pipeline: Adapter merge + Hailo Dataflow Compiler → neuer .hef fuer Pi-NPU (Welle 4)

---

## 8. FRAGEN AN MARKUS WENN UNKLAR

1. CPU-only oder GPU-Training? (GTX 760 ist Kepler — moeglicherweise nur PyTorch ≤2.0 kompatibel)
2. Soll Trainer als Windows Service laufen oder nur on-demand?
3. Wann triggert das Training automatisch (cron/Task Scheduler) oder Markus manuell?
4. Adapter-Pfad: `%USERPROFILE%\moloch_adapters\` ok oder anderer Speicherort (SSD)?

Wenn unklar: Markus fragen, NICHT raten.

---

## 9. KOORDINATION MIT PI-SIDE

Pi hat einen FastAPI auf Port 9100 (`/chat`, `/live`, `/personality`, `/snapshot.jpg`, `/feedback`). Wenn du fuer den Sample-Sync einen Endpoint brauchst (statt scp), kann der Pi-Side-Agent das hinzufuegen — sag der naechsten Pi-Session bescheid via dem hier:

> "Pi soll Endpoint GET /feedback_export bauen der finetune_samples.jsonl als download liefert. So kann PC einfach `curl -o samples.jsonl http://192.168.178.30:9100/feedback_export` machen statt scp."

(Optional, scp tut's auch.)

---

## 10. KRITISCHE REGELN

- **Pi NICHT modifizieren von dieser Session aus** — du arbeitest nur auf PC.
- **Keine Adapter-Deployments auf Pi** ohne dass Markus explizit sagt. Adapter bleiben erstmal nur auf PC, Pi spricht remote.
- **Backup vor Aenderungen**: jeder Adapter-Run schreibt v{N+1}, alte Versionen behalten (mind. letzte 5).
- **Nur freigegebene Samples verwenden** (`approved=true`). NIE pending Samples trainieren.
- **GTX 760 = Kepler = OLD**: bei CUDA-Errors einfach auf CPU runterschalten, RAM hat 32 GB, das geht.

---

*Briefing-Datei: `docs/THREEBRAIN_PC_SIDE_BRIEFING.md` im Repo.*
*Pi-Side Reference-Code committed: `0eb375a` und davor.*
