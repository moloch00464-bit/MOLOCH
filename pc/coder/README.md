# MOLOCH Coder-Tentakel

Lokale Code-AI auf PC, eingebunden ins anatomische MOLOCH-System
(Welle 5 — `prompt_type=code`).

## Komponenten

| Datei | Zweck |
|---|---|
| `Modelfile` | Ollama-Modelfile mit System-Prompt (Basis: deepseek-coder:6.7b) |
| `build.ps1` | `ollama create moloch-coder` + Smoketests |
| `prompt_builder.py` | User-Prompt -> Skill-Match -> Inject -> Ollama-Call |
| `prompt_builder.test.py` | 5 Skill-Match-Testfaelle |
| `skills/*.md` | 5 Domain-Knowledge-Snippets (audit, mailbox, gstreamer, ipc, atomic-write) |

## Build

```powershell
cd C:\Users\49179\moloch_repo\pc\coder
.\build.ps1
```

Erstellt das Modell `moloch-coder` in Ollama, laeuft 3 Smoketests.

## Aufruf

### Direkt via Ollama

```bash
ollama run moloch-coder "frag was"
```

### Mit Skill-Routing via prompt_builder

```bash
python prompt_builder.py "schreib einen vision_auditor"
# -> matched skills: ['audit-pattern']
# -> System-Prompt + audit-pattern Skill + User-Prompt
```

### Per HTTP

```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model":"moloch-coder","prompt":"...","stream":false}'
```

## Skills

Wird auf Basis von Keywords im User-Prompt automatisch injected:

| Skill | Trigger-Keywords |
|---|---|
| audit-pattern | auditor, collect, audit_state, score, merge_component |
| mailbox-protocol | mailbox, PC_TO_PI, PI_TO_PC, topic, /mailbox |
| gstreamer-hailo | gstreamer, pipeline, hailo, uint8, float32, SCRFD, ArcFace |
| ipc-pattern | ipc, moloch_service, register_action, route_action |
| atomic-write | atomic, /dev/shm, tempfile, os.replace |

## Welle-5-Integration

Pi-Side Routing-Code muss `model="moloch-coder"` statt `model="deepseek-coder:6.7b"`
fuer `prompt_type=code` setzen — siehe Mailbox-Task an Pi-Opus
(`task_w13plus_welle5_routing_auf_moloch_coder_umstellen`).
