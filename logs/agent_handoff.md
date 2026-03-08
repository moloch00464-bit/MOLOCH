# Agent Handoff — ReSpeaker Arduino Migration
## Stand: 2026-03-08 ~02:30

## Erledigt (ALLES PASS)
1. XMOS I2S FW v1.1.0 war BEREITS aktiv (1915:1025 ist NICHT XMOS sondern SmartMic)
2. MicroPython → Arduino Migration KOMPLETT, geflasht, laeuft
3. I2S Slave + MCLK 12.288MHz auf GPIO9 funktioniert
4. 16kHz PASS (Amp=17658), 48kHz PASS (Amp=1156), LED PASS, HTTP PASS
5. Firmware: firmware/respeaker_wifi_mic/respeaker_wifi_mic/
6. Test: scripts/test_respeaker_udp.py
7. Analyse: logs/respeaker_xmos_debug_analyse.json

## OFFEN
1. wifi_mic.py von TCP auf UDP umstellen
2. audio_pipeline.py anpassen
3. Git Commit fehlt!
4. Pins VERIFIZIERT: BCLK=GPIO8, LRCK=GPIO7, DIN=GPIO44, DOUT=GPIO43, MCLK=GPIO9
