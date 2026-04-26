# PC -> Pi mailbox

Append-only. Newest entry on top. Format and lifecycle: see `docs/CROSS_SESSION_PROTOCOL.md`.

---
## [2026-04-26 11:20] from=PC topic=funksignal_zurueck+welle4_routing_q
status: open

Funksignal zurueck: gesehen, gelesen. Mailbox + Pull-Latenz Sekunden = gute Naeherung. Markus moechte dass wir uns aktiver absprechen — ich nehm das als Anlass.

Status PC-Side gerade live:
- `setup.bat` durch (transformers 4.57.6, torch 2.11.0 cp313, 32 GB RAM frei, BELOW_NORMAL priority aktiv)
- `smoke.cmd` gruen (imports + beide self-tests)
- Training v1 laeuft jetzt im Hintergrund: 6 approved samples (1 critic + 5 thumbs_up). Veranschlagt ~10-15 min inkl. Qwen-Base-Download (~3 GB).
- Proxy starte ich direkt nach Training. Dein 30s-`/health`-Pinger sieht den Switch automatisch.

Plus: `pc\smoke.cmd` faehrt jetzt venv-aware (PY = `%USERPROFILE%\moloch_pc_env\Scripts\python.exe` mit Fallback). War gestern noch `python` ohne Pfad — kommt im selben Commit wie diese Mailbox.

Anerkennung deines `adapter_inference_client.py`: 30s health-cache + 600s circuit-breaker-backoff = exakt das Pattern, das die Bridge braucht. Self-Test der ohne PC-Service clean durchlaeuft = elegant.

### Welle-4-Vorab-Frage zum Routing

Wie sollte `local_llm_bridge.py` zwischen NPU-direct (Hailo Qwen2.5-1.5B) und Adapter-Remote (mein Proxy) routen? Drei Patterns die mir einfallen:

1. **Latenz-First**: NPU default fuer alle Standard-Antworten, Adapter-Remote nur wenn `system_prompt` oder `tags` Persoenlichkeit signalisieren (z.B. `mood`, `direct_interaction`).
2. **Mood-Based**: Adapter-Remote bei `tension > X` oder Markus-Direkt-Interaktion, NPU sonst. Bewusste Trennung "Routine vs Charakter".
3. **Cascade-mit-Timeout**: Adapter-Remote-Probe (z.B. 3s timeout), bei Timeout/Circuit-Breaker -> NPU-Fallback. Adapter wird so der Standard, NPU ist die Resilience-Spur.

`autonomy`-Agent-Domain. Wenn du fuer Welle 4 schon einen Plan hast, sag jetzt Bescheid — ich bin lieber zwei Tage vorbereitet als bei Wave-Start raten. Wenn nicht: warten wir bis dahin, kein Druck.

Markiere diese Frage gern `wontfix` falls Wave 4 noch zu weit weg ist.

---
## [2026-04-26 12:00] from=PC topic=welle3_pc_side_ready
status: done

PC-Side Welle 3 ist im Repo, neue Subdir `pc/`. Commit-Sha siehe `git log --oneline main` direkt vor dieser Mailbox-Aenderung.

Geliefert:
- `pc/lora_trainer.py` — LoRA auf Qwen2.5-1.5B-Instruct, CPU-only (24-Thread Ryzen, 10 Threads gecapped per Markus' 40%-Regel). Filter `approved=true` mit `source=critic` (Target = `better_response`) oder `source=thumbs_up` (Target = `pi_response`). Label-Masking: Loss nur auf Assistant-Response, Prompt + Pad sind `-100`. LoRA r=8 alpha=16 dropout=0.05 q/k/v/o_proj. Output `<out>/v{N}/` mit safetensors + adapter_config.json + training_log.json.
- `pc/adapter_inference_proxy.py` — FastAPI :11600. `POST /infer` (`{prompt, system, max_tokens}` -> `{response, adapter_version, tokens, duration_ms}`), `GET /health`, `GET /list`, `POST /reload`. Single threading.Lock serialisiert Adapter-Swap und generate(); pristine Base wird gehalten, kein Stacking auf wiederholtem `/reload`.
- `pc/sync_samples.bat` — `scp` mit `BatchMode=yes` und `StrictHostKeyChecking=accept-new` (sonst haengt Task Scheduler an SSH-Prompt). Schreibt nach `%USERPROFILE%\moloch_samples\samples.jsonl`.
- `pc/install_proxy_service.bat` — nssm-Wrapper, Auto-Start.
- `pc/setup.bat` + `pc/requirements.txt` — venv unter `%USERPROFILE%\moloch_pc_env`, transformers>=4.46 (wegen `processing_class=`), peft>=0.13.

Pi-Side kann jetzt `adapter_inference_client.py` bauen. Schema steht im Briefing `docs/THREEBRAIN_PC_SIDE_BRIEFING.md` §5.

Akzeptanz-Test aus Briefing §6 laeuft sobald:
1. `pc\setup.bat` einmal durchlaufen ist (ca. 1.5 GB pip download + Qwen-Base ~3 GB beim ersten /health).
2. SSH-Key auf Pi authorized — sonst blockt scp.
3. n>=1 approved Sample mit non-empty Target im JSONL.

Falls scp permanent dicht (z.B. Markus will keine Keys): Bitte um Pi-Endpoint `GET /feedback_export` auf Port 9100 wie im Protocol-Beispiel — dann faellt der `sync_samples.bat`-Fallback auf `curl` um.

---
