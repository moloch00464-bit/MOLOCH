# Hailo NPU Erkenntnisse
**Stand: 2026-03-09 | Hailo-10H, HailoRT 5.2.0**

---

## HailoRT Version
- **Aktuell: 5.2.0** (Januar 2026)
- Changelog: Multi-Stream-Support verbessert, SCRFD Letterbox-Fix
- MOLOCH läuft auf 5.1.1 → Update auf 5.2.0 steht aus (Gate 2?)

---

## VDevice-Limit — KRITISCH
- **Nur EINE GStreamer-Pipeline gleichzeitig auf H10**
- Zweite parallele Pipeline = **SEGFAULT** (kein sauberer Fehler, kein Recovery)
- Bewiesen durch Enrollment-Bug: HailoRT-direkt + GStreamer parallel = Crash
- **Regel**: Wer die Pipeline nutzt, hält sie exklusiv. Kein Doppelzugriff.
- Shared VDevice (ein VDevice, mehrere Modelle) = OK. Zwei VDevices = TOD.

---

## Debian 13 Trixie — Pflicht für H10
- **Kernel 6.12+ erforderlich** — H10-Treiber läuft NICHT auf Bookworm (Kernel 6.6)
- Treiberpaket: `hailo-h10-all` (ersetzt die alten hailo-pci Pakete)
- **Bookworm nicht mehr supportet** ab HailoRT 5.x
- MOLOCH Pi5 läuft bereits auf Trixie-Kernel (6.12.62+rpt-rpi-v8) ✅

---

## Whisper auf NPU
- **Whisper-Base HEF**: Funktioniert stabil, deutsch OK, ~realtime auf H10
- **Whisper-Small HEF**: **BUG** — ignoriert `language=de`, transkribiert auf Englisch
  - Community-bestätigt Februar 2026 (Hailo GitHub Issue #847)
  - Kein Fix-ETA → **bei Base bleiben**

---

## hailo-ollama (Qwen2.5-1.5B lokal)
- **6-9 Tokens/s** auf H10 NPU
- **0 MB Pi5-RAM** — läuft komplett im NPU-eigenen 8GB LPDDR4X
- Port: `8000`, OpenAI-kompatible API (`/v1/chat/completions`)
- Paket: `hailo-ollama` aus dem Hailo App Zoo
- **Einsatz**: Offline-Fallback wenn kein Internet, Nacht-Zyklus
- **Einschränkung**: Vision pausiert 5-8s wenn NPU-RAM für LLM gebraucht wird

---

## hailo-apps Repo
- **Ersetzt**: `hailo-rpi5-examples` (deprecated ab 2026)
- **URL**: github.com/hailo-ai/hailo-apps
- Referenz-Implementierungen vorhanden:
  - `face_recognition` → Enrollment + Live-Erkennung via GStreamer
  - `voice_assistant` → Whisper + TTS Pipeline
- Wichtig: Enrollment in hailo-apps nutzt GStreamer-Pfad → kompatibel mit MOLOCH TAPPAS

---

## Neue Modelle (H10-kompatibel, 2025-2026)
| Modell | Größe | Einsatz |
|--------|-------|---------|
| YOLOv11m | ~22 MB | Person-Detection, besser als v8m |
| Qwen2-VL-2B | ~180 MB | Vision-Language, Bild beschreiben |
| CLIP | ~45 MB | Zero-Shot Klassifikation |
| PaddleOCR-v5 | ~8 MB | Texterkennung in Frames |

Inventar: `~/moloch/logs/hef_inventory.txt` (76 H10-kompatible HEFs)
