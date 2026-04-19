---
name: pc-bridge
description: Cross-Platform-Setup und Debug fuer Pi <-> Markus-PC Bruecken (LLM-Tentakel, STT, TTS, Chat-UI). Nutze bei Bridge-Aufgaben oder PC-Erreichbarkeits-Problemen.
user-invocable: true
---

# PC-Bridge — Pi <-> Markus-PC

**Topologie:**
```
Pi (192.168.178.30, Brain)
   |
   +-- LAN --+ Markus-PC (192.168.178.20, Co-Worker)
                |
                +-- Ollama   :11434  (LLM-Tentakel, LIVE)
                +-- Whisper  :9001   (STT-Bridge, GEPLANT)
                +-- Piper    :9002   (TTS-Bridge, GEPLANT)
                +-- Chat-UI  :9000   (Web/Desktop, GEPLANT)
```

## PC-Hardware (2026-04-19)

- **CPU:** AMD Ryzen 9 3900X (12C/24T) — stark, CPU-Whisper-medium machbar
- **RAM:** 32 GB
- **GPU:** NVIDIA GTX 760 (**2 GB VRAM**, Kepler) — Whisper-medium ja, large-v3 zu wenig VRAM
- **Audio:** USB-Audiogeraet + HD Audio + NVIDIA HDMI
- **OS:** Windows 10 Pro, statische IP 192.168.178.20

## Aktive Bridge: LLM-Tentakel

**Setup (Windows, einmalig, Admin):**
```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'Machine')
New-NetFirewallRule -DisplayName 'Ollama LAN (MOLOCH)' -Direction Inbound `
  -Protocol TCP -LocalPort 11434 -Action Allow -RemoteAddress 192.168.178.0/24
```

**Pi-Config (`~/moloch/config/settings.json`):**
```json
"tentacle_llm": {
  "enabled": true,
  "host": "192.168.178.20",
  "port": 11434,
  "model": "",
  "complexity_threshold": 120,
  "timeout_sec": 30,
  "backoff_sec": 300
}
```

**Routing (in `local_llm_bridge.py`):**
- Prompt+System >= 120 Zeichen oder caller="reason" -> Tentakel (mistral 7B)
- Sonst -> NPU (qwen2.5:1.5b)
- Bei 3 Tentakel-Fails -> 300s Backoff -> NPU faengt ab
- Auto-Discovery wenn `model: ""` -> erstes /api/tags-Result wird gecached

## Geplante Bridges

| Bridge | Modell-Vorschlag | Endpoint | Aufwand |
|---|---|---|---|
| STT-Bridge | faster-whisper medium (CPU) oder large-v3 (GPU wenn moeglich) | POST :9001/transcribe | mittel — Bash-Wrapper + FastAPI |
| TTS-Bridge | Piper-Windows (gleich wie Pi) oder Edge-TTS (besser) | POST :9002/speak | klein — fertige Tools |
| Chat-UI | Browser-UI mit WebSocket zu Pi-IPC | http://192.168.178.20:9000 | gross — eigenes Frontend |

## Debug-Befehle (vom Pi)

```bash
# Erreichbarkeit
ping -c 3 192.168.178.20
curl -sS --max-time 5 http://192.168.178.20:11434/api/tags

# Bridge-Logs
journalctl -u moloch -n 100 | grep -iE 'BRIDGE|tentacle'

# Live-Test (Reasoning -> Tentakel)
cd ~/moloch && python3 -c "
from core.autonomy.local_llm_bridge import get_llm_bridge
b = get_llm_bridge()
out = b.reason_internal('Erklaere kurz wie ein Ringpuffer funktioniert.')
print('Provider:', b._last_provider)
print('Out:', out[:200])
"

# Bridge-State im Bridge-Singleton
python3 -c "
from core.autonomy.local_llm_bridge import get_llm_bridge, _load_tentacle_cfg
b = get_llm_bridge()
print('mode:', b._llm_mode)
print('cfg:', _load_tentacle_cfg())
print('fail:', b._tentacle_fail_count, 'backoff_until:', b._tentacle_backoff_until)
"
```

## Troubleshooting

| Symptom | Erste Schritte |
|---|---|
| curl /api/tags hangt | Windows: Ollama Tray-App laeuft? netstat findstr 11434 zeigt 0.0.0.0? Firewall-Regel da? |
| Pi sagt 'PROVIDER=lokal_qwen2.5' obwohl Tentakel erwartet | Prompt war kurz (<120 Zeichen) -> Routing ist korrekt. Oder backoff_until > 0 -> warte 5 Min. |
| tentacle_fail_count > 0 | Letzte journalctl-Zeile mit '[LLM-BRIDGE]' lesen — Connection Error vs Timeout vs Schema-Mismatch |
| Tentakel laeuft, aber Antwort ist Mist | model='' nutzt erstes /api/tags-Result. Setze model: 'mistral:latest' explizit. |

## Watchdog-Probe

`core/system_watchdog.py` probed alle 30 Min die Bridges.
Status-Pfad: `config/system_capabilities.json.bridges` (nach Erweiterung).

## Wichtige Regeln

- LAN-Calls IMMER mit Timeout
- Firewall-Scope STRIKT 192.168.178.0/24
- KEINE Auth-Header (LAN ist unsecured) — nicht ins Internet exponieren
- Failover-Kette IMMER bis Stille (kein Crash)
