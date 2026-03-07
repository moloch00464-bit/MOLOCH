# AGENT HANDOFF — ESP32 WiFi-Mic + RGB-LED Session
# Geschrieben: 2026-03-07 ~15:30
# Naechste Instanz: Lies CLAUDE.md, dann diese Datei

## AKTUELLER STAND

### Erledigt
1. ESP32-S3 geflasht mit MicroPython v1.27.0 (SPIRAM_OCT)
2. WiFi Direkt-AP: Pi wlan0 = `MOLOCH_DIRECT` (10.42.0.1), ESP = 10.42.0.2
3. RGB-LED (WS2812 GPIO 1) funktioniert via UDP Port 8888
4. Firmware deployed als `:main.py` auf ESP32
5. Pi-seitige Module erstellt: wifi_mic.py, audio_pipeline.py, rgb_led_controller.py
6. Alle Import-Tests + USB-Fallback PASS

### Neue Dateien (UNCOMMITTED!)
- `docs/respeaker_esp32s3_firmware.py` — ESP32 MicroPython Firmware
- `core/audio/wifi_mic.py` — TCP-Client fuer Audio-Streams
- `core/hardware/audio_pipeline.py` — Source-Router (WiFi>USB)
- `core/hardware/rgb_led_controller.py` — LED-Steuerung via UDP

### OFFEN — Naechste Schritte

1. **I2S PINS VERIFIZIEREN** (BLOCKER fuer Audio!)
   - Annahme SCK=5, WS=6, SDI=4, SDO=7 → Init OK aber nur Nullen
   - Seeed Wiki 404, Schaltplan noetig
   - Markus soll Leitungen am Board pruefen
   - Alternative: Pins systematisch durchprobieren

2. **Audio E2E testen** (erst nach I2S-Fix)
   - TCP Ports 12345/12346/12347 laufen auf ESP32

3. **RGB-LED in moloch_service.py integrieren**
   ```python
   from core.hardware.rgb_led_controller import get_rgb_led
   self._rgb_led = get_rgb_led(event_bus=self._event_bus)
   self._rgb_led.start()
   ```

4. **Git Commit** — alles noch uncommitted

## Netzwerk
- eth0: 192.168.178.30 (NICHT .24 wie CLAUDE.md sagt!)
- wlan0: 10.42.0.1/24 (AP MOLOCH_DIRECT, autoconnect=yes)
- ESP32: 10.42.0.2 (statisch)

## Wichtig
- `neopixel.write()` NICHT `show()` in MicroPython
- `sudo` noetig fuer nmcli hotspot
- mpremote: `exec` statt `eval` fuer mehrzeiligen Code
- Letzter Git Commit: `1e99cd0 BACKUP vor ReSpeaker ESP32S3 WiFi-Mic Integration`
