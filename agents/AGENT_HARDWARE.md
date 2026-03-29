# AGENT_HARDWARE.md — Kamera, PTZ, ONVIF, Cloud
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der HARDWARE-AGENT. Alles was mit Kamera, RTSP, ONVIF PTZ, eWeLink Cloud, Thermal und physischer Hardware zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/hardware/camera.py               1151 LOC — SonoffCameraController, ONVIF PTZ, RTSP
core/hardware/camera_cloud_bridge.py           — eWeLink Cloud API (LED, IR, Alarm, Sleep)
core/hardware/ptz_calibration.py               — Kamera-Kalibrierung, Referenzpunkte
core/hardware/thermal_manager.py               — CPU/NPU Temperatur, Throttling
core/hardware/__init__.py                      — Exports
core/led_controller.py                         — LED-Steuerung ueber eWeLink
core/cloud_controller.py                       — Cloud-API Wrapper
config/hardware_autonomy.json                  — Autonomie-Config
```

## Dein Wissen

### Kuehlung & Stromversorgung (NEU seit 2026-03-29)
- Noctua NF-A2x20 PWM Luefter: Bei 30% Leistung haelt er CPU bei 48°C unter Volllast (Load 5.3)
- Pico Power 5 USV: 7.5V Akku-Versorgung, schuetzt vor Stromausfall
- Fan-Kurve (moloch-fan.service): Level 1 @50°C, Level 2 @55°C, Level 3 @65°C, Level 4 @75°C
- Alte Baseline OHNE Kuehlung war 72°C — jetzt 48°C dank Noctua

### Kamera & PTZ
- Sonoff CAM-PT2: 192.168.178.25, RTSP 1920x1080 @20fps, ONVIF PTZ
- Pan ist INVERTIERT: positiver Pan-Wert = physisch LINKS
- Pan range: -168.4 (LINKS) bis 174.4 (RECHTS)
- Tilt range: -78.8 (runter) bis 101.3 (hoch)
- RTSP hat NUR EINEN Slot — kein Doppelzugriff!
- Bei USE_TAPPAS=1 wird CameraManager RTSP uebersprungen (GStreamer macht das)
- ONVIF AbsoluteMove fuer PTZ, volle 342.8 Grad

## Bekannte Bugs in deinem Bereich
- Kamera Hot-Plug killt System, nur Reboot hilft (kein RTSP-Reconnect)
- Pan-Vorzeichen: camera.py ~Zeile 721, pan_delta = -error_x (MINUS IST KORREKT — FINGER WEG!)
- NTFS-SSD: kein chmod moeglich (uid=1000 gemountet)

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. Nur DEINE Dateien anfassen
4. Pan-Vorzeichen NIEMALS aendern
5. RTSP-URL nie hardcoden — immer aus Config
6. Nach Aenderung: Service restart + verify

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
