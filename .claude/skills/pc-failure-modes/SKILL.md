---
name: pc-failure-modes
description: Decision-Tree fuer Failure-Modes im Pi <-> PC <-> Cloud Stack. Was tun wenn Pi/PC/Ollama/DeepSeek/Search-Proxy nicht antwortet. Circuit-Breaker, Backoff, User-Notification.
user-invocable: true
---

# Failure-Modes — Pi <-> PC <-> Cloud

Was tun wenn etwas im Bridge-Stack tot ist. Aktiv ab W19+W20a+W21.

## Failure-Matrix

| Wer ist tot | Wer merkts | Was passiert | Fallback |
|---|---|---|---|
| **Pi-Brain** (Service moloch.service) | PC-cross_session_monitor (30s probe) | Cockpit zeigt nichts, kein chat_server | Markus muss Pi neu booten / `sudo systemctl restart moloch` |
| **Pi-chat_server** (:9100) | PC-Auditoren (5min) | Audit-POSTs HTTP 502/timeout | `moloch_service(action=restart)` (W20a-A3 alle 3 Units) |
| **PC-Ollama** (:11434) | Pi-Specialist-Router | tentacle_fail_count++, nach 3 -> 300s Backoff -> NPU faengt ab | Antwort: kurzer NPU-Output (`lokal_qwen2.5`) |
| **PC-Adapter-Proxy** (:11600) | Pi-adapter_inference_client | Circuit-Breaker (3 fails -> 600s) | LoRA-Inferenz aus -> Tentakel oder Cloud |
| **PC-Search-Proxy** (:11650) | Pi-Specialist-Router web-Branch | Web-Augmentation faellt aus | fail-soft: Original-Prompt ohne Web-Context an Cloud-LLM |
| **DeepSeek-Cloud** | Pi-LLM-Bridge | HTTP 5xx oder timeout | tentacle_llm.web_model fallback auf dolphin-mistral:7b lokal (Hallu-Risk!) |
| **Pi-Tool-Dispatcher** (:9100/api/agent) | PC-Orchestrator (HttpBridge) | get_bridge() probiert HTTP, faellt auf MockBridge | Orchestrator laeuft mit Mock weiter, Markus sieht Mock-Markers in Antwort |
| **Mailbox-API** (:9100/mailbox) | beide Sessions | POST gibt 5xx, GET timeout | warten bis chat_server wieder up |

## Decision-Tree

### Pi reagiert nicht

```
ping 192.168.178.30 ?
  -> NEIN -> Hardware-Down (Pi-Brain hart)
  -> JA   -> ssh oder mDNS check
              -> SSH OK + moloch.service down -> systemctl restart moloch
              -> SSH OK + chat_server down    -> moloch_service(restart) alle 3 Units
              -> SSH down                      -> Pi-Reboot (Markus physisch)
```

### PC reagiert nicht

```
Pi probed http://192.168.178.20:11434/api/tags ?
  -> 503/timeout -> Ollama-Service down (Markus muss Ollama-Tray-App pruefen)
  -> 401/403     -> Firewall-Block (selten)
  -> Connection refused -> PC schlaeft / WLAN aus
  -> 200 langsam (>10s) -> CPU-Load hoch (Markus arbeitet parallel)
                            -> Backoff 300s, NPU faengt ab
```

### DeepSeek-Cloud nicht erreichbar

```
PC pingt api.deepseek.com ?
  -> NEIN -> Internet-Ausfall (Markus' Router)
  -> JA   -> POST /v1/chat/completions ?
              -> 401 -> API-Key abgelaufen / falsch
              -> 429 -> Rate-Limit -> Backoff 60s + retry
              -> 5xx -> DeepSeek-Outage -> fallback auf Claude-API (wenn key da)
                                        -> sonst lokal dolphin-mistral
```

### Tool-Dispatch fail

```
Orchestrator ruft Tool X, error != null ?
  -> Tool unknown -> Catalog re-laden, vielleicht hat Pi neu gebaut
  -> Tool timeout -> Pi-Tool-Implementation broken
                      -> dem LLM melden, LLM versucht andere Tools
                      -> wenn alle Tools fail -> finale text-Antwort mit Apologie
  -> Tool error message -> dem LLM in tool_response weitergeben (LLM kann adaptieren)
```

## Circuit-Breaker-Konvention

| Komponente | Fails bis Open | Backoff | Reset |
|---|---|---|---|
| Tentakel-Ollama | 3 | 300s | Erste erfolgreiche Probe |
| Adapter-Proxy | 3 | 600s | Erste erfolgreiche Probe |
| DeepSeek-Cloud | 5 | 120s | Erste erfolgreiche Probe |
| Search-Proxy | 2 | 60s | Erste erfolgreiche Probe |
| Tool-Dispatcher | 2 (per Tool) | 30s | Erste erfolgreiche Probe |

## User-Notification

Bei Fail dem User klar sagen — keine Halluzination!

| Fail | TTS-Output |
|---|---|
| Cloud-LLM down + Web-Frage | "DeepSeek antwortet grad nicht. Lokales Modell hat keinen Web-Stack — frag in 5 Min nochmal." |
| Pi-Cam down + Sehen-Frage | "Kamera ist gerade aus. Hardware oben checken." |
| Spotify-API down | "Spotify ist offline — entweder du bist nicht eingeloggt oder API hat Schluckauf." |
| Tool fehlt | "Das Tool gibt's noch nicht (Welle X+Y). Bauen wir noch." |
| Halluzination-Detector triggert | "Ich hab das nicht verifizieren koennen — pruef bitte selber nach: <link>" |

## Audit-Sichtbarkeit

Alle Fails landen in `/dev/shm/audit_state.json:layers.<component>.detail.error`. Cockpit zeigt rot. Closed-Loop-Verifier (W15) bestaetigt End-to-End.

## NEVER

- NIE Fail still verschlucken — Markus muss es wissen
- NIE Hallucinated-Antwort statt "weiss nicht"
- NIE Cloud-Retry ohne Backoff (Cost-Explosion)
- NIE Bridge ohne Timeout
- NIE Fallback in Loop ohne Maximal-Iterations
