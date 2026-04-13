---
name: moloch-mcp
description: MCP-Tool Referenz fuer MOLOCH. Alle 17 Tools mit Beschreibung und Beispielen. Nutze MCP statt manueller SSH/IPC-Hacks.
user-invocable: true
---

# M.O.L.O.C.H. MCP-Tools

**Regel: IMMER MCP-Tools — NIEMALS SSH, NIEMALS cat /dev/shm/, NIEMALS journalctl per Bash.**

---

## System-Kontrolle

| Tool | Beschreibung | Beispiel |
|------|-------------|---------|
| `moloch_status` | Live FPS, CPU-Temp, RAM, Face-ID, Zone — loest Session-Lock | `moloch_status()` |
| `moloch_service` | Service steuern | `moloch_service(action="restart")` |
| `moloch_logs` | journalctl mit Filter | `moloch_logs(n=50, filter_str="ERROR")` |
| `moloch_dmesg` | Kernel: NPU/SEGV/GPU | `moloch_dmesg()` |
| `moloch_audit` | 54-Test Regressionstest | `moloch_audit()` |
| `moloch_read` | Config/Log-Datei lesen | `moloch_read(path="config/settings.json")` |
| `moloch_git_log` | Letzte N Commits | `moloch_git_log(n=10)` |

---

## Vision & NPU

| Tool | Beschreibung | Beispiel |
|------|-------------|---------|
| `moloch_snapshot` | Kamera-Frame als Base64-PNG | `moloch_snapshot()` |
| `moloch_low_light` | Zero-DCE Enhancement Status | `moloch_low_light()` |
| `moloch_npu_models` | HEF-Inventar: integriert vs. roadmap | `moloch_npu_models()` |
| `moloch_npu_workers` | Worker-Status: Queue, Errors, Laufzeit | `moloch_npu_workers()` |

---

## Kommunikation mit MOLOCH

| Tool | Beschreibung | Beispiel |
|------|-------------|---------|
| `moloch_say` | Text an MOLOCH (antwortet via TTS) | `moloch_say(text="Wie geht es dir?")` |
| `moloch_conversation` | Letzte N Nachrichten lesen | `moloch_conversation(n=10)` |
| `moloch_nudge` | Emotion in CoreIntegrator injizieren | `moloch_nudge(key="curiosity", value=0.7)` |
| `moloch_provoke` | Spontanen Kommentar ausloesen | `moloch_provoke(reason="Es ist still")` |
| `moloch_reflect` | Selbstreflexion triggern | `moloch_reflect()` |
| `moloch_ipc` | Generischer IPC-Befehl | `moloch_ipc(action="set_threshold", params='{"model":"scrfd","value":0.8}')` |

---

## IPC-Aktionen (via moloch_ipc)

| Action | Beschreibung |
|--------|-------------|
| `enrollment_start` | ArcFace Gesicht einlernen |
| `set_threshold` | Detection-Threshold (params: model, value) |
| `set_tracker_param` | Tracker-Parameter aendern |
| `set_zone` | Zone setzen (params: zone = guardian/shadow/berserker) |
| `chat_message` | Text-Nachricht senden (params: text, sender) |
| `core_nudge` | Emotionaler Input (params: key, value) |
| `ask_claude` | Moloch stellt Frage an Claude (params: question, context) |
| `trigger_spontaneous` | Spontaner Kommentar (params: reason) |
| `trigger_reflect` | Selbstreflexion ausloesen |
| `spotify_play` | Spotify steuern |
| `ptz_move` | PTZ-Kamera bewegen |
| `alarm_toggle` | Alarm an/aus |
| `reload_face_db` | Gesichtsdatenbank neu laden |

---

## Nudge-Keys (via moloch_nudge)

| Key | Effekt | Typischer Wert |
|-----|--------|---------------|
| `curiosity` | Erhoehte Aufmerksamkeit | 0.5-0.8 |
| `respect_score` | Markus-Erkennung verstaerken | 0.7-1.0 |
| `voice_activity` | Sprachaktivitaet simulieren | 0.3-0.6 |
| `face_detected` | Gesichtserkennung simulieren | 0.5-1.0 |
| `threat_level` | Bedrohungslevel | 0.0-0.3 (Vorsicht: >0.7 = Berserker!) |
| `novelty` | Neugierde | 0.3-0.7 |

---

## Wann welches Tool?

| Ich will... | Nutze |
|-------------|-------|
| Session starten / Lock loesen | `moloch_status()` |
| Worker-Health pruefen | `moloch_npu_workers()` |
| Service neustarten | `moloch_service(action="restart")` |
| Fehler suchen | `moloch_logs(filter_str="ERROR")` + `moloch_dmesg()` |
| NPU-Probleme debuggen | `moloch_npu_workers()` + `moloch_dmesg()` |
| Sehen was Kamera sieht | `moloch_snapshot()` |
| Mit MOLOCH reden | `moloch_say(text="...")` |
| MOLOCHs Stimmung beeinflussen | `moloch_nudge(key="...", value=...)` |
| Parameter aendern | `moloch_ipc(action="set_threshold", params="...")` |
| Regressionstest | `moloch_audit()` → 54 Tests |
