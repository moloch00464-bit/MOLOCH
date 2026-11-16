# ANIMA Laufzeit-Inventar – 2026-11-16 07:03

Erzeugt von `scripts/anima_inventory.py` (rein lesend). `unbekannt` = nicht messbar, keine Schaetzung.

## System

| Was | Wert | Quelle |
|---|---|---|
| Modell | Raspberry Pi 5 Model B Rev 1.1 | /proc/device-tree/model |
| RAM gesamt (GB) | 3.9 | /proc/meminfo |
| RAM verfuegbar (GB) | 2.32 | /proc/meminfo |
| Python | 3.13.5 | sys.version |
| Kernel | 6.12.62+rpt-rpi-v8 | uname -r |
| Laufzeit | up 21 hours, 14 minutes | uptime -p |
| CPU-Temperatur | temp=49.4'C | vcgencmd measure_temp |
| unterspannung_jetzt | nein | vcgencmd get_throttled |
| gedrosselt_jetzt | nein | vcgencmd get_throttled |
| unterspannung_seit_start | nein | vcgencmd get_throttled |
| gedrosselt_seit_start | nein | vcgencmd get_throttled |
| Luefter-Drehzahl (U/min) | 0 | hwmon fan1_input (nur mit Tacho-Signal) |

## Massenspeicher

| Was | Wert | Quelle |
|---|---|---|
| Datentraeger | sda    476.9G      <br>└─sda1 476.9G ext4 /mnt/moloch-data<br>sdb    465.8G      <br>├─sdb1   512M vfat /boot/firmware<br>└─sdb2 465.3G ext4 /<br>zram0      0B | lsblk |
| / | ext4, 412.8 GB frei von 491.6 GB (12 % belegt) | findmnt / disk_usage |
| /mnt/moloch-data | ext4, 443.2 GB frei von 502.9 GB (7 % belegt) | findmnt / disk_usage |

## NPU (Hailo)

| Was | Wert | Quelle |
|---|---|---|
| Geraeteknoten /dev/hailo0 | fehlt | Dateisystem (belegt keine Inferenz) |
| Kernelmodul | hailo1x_pci           110592  16 | lsmod |
| Identitaet | Executing on device: pci/0001:01:00.0<br>Identifying board<br>Control Protocol Version: 2<br>Firmware Version: 5.3.0 (release,app)<br>Logger Version: 0<br>Device Architecture: HAILO10H | hailortcli fw-control identify |
| Service-Status | 0.0 s alt | /dev/shm/moloch_status.json |
| FPS | {"scrfd": 17.1, "arcface": 17.1, "yolov8m": 17.1, "total": 17.1} | /dev/shm/moloch_status.json |
| Aktive Modelle | arcface, faceattr, hand, pose, reid, scrfd, yolo | /dev/shm/moloch_status.json |
| Worker DepthWorker | laeuft=True inferenzen=365 fehler=0 letzte_ms=41.5 | /dev/shm/moloch_status.json |
| Worker FaceWorker | laeuft=True inferenzen=1789 fehler=0 letzte_ms=122.9 | /dev/shm/moloch_status.json |
| Worker HandWorker | laeuft=True inferenzen=912 fehler=0 letzte_ms=0.0 | /dev/shm/moloch_status.json |
| Worker PoseWorker | laeuft=True inferenzen=1214 fehler=0 letzte_ms=0.0 | /dev/shm/moloch_status.json |
| Worker ReIDWorker | laeuft=True inferenzen=730 fehler=0 letzte_ms=0.0 | /dev/shm/moloch_status.json |
| Worker _dispatcher | laeuft=None inferenzen=None fehler=None letzte_ms=None | /dev/shm/moloch_status.json |

## Dienste und Ports

| Was | Wert | Quelle |
|---|---|---|
| Dienst moloch | active / enabled | systemctl is-active / is-enabled |
| Dienst moloch-chat | active / enabled | systemctl is-active / is-enabled |
| Dienst moloch-chat-https | active / enabled | systemctl is-active / is-enabled |
| Dienst moloch-cross-monitor | active / enabled | systemctl is-active / is-enabled |
| Dienst qdrant | inactive / not-found | systemctl is-active / is-enabled |
| Dienst hailo-ollama | active / disabled | systemctl is-active / is-enabled |
| Dienst lightdm | active / enabled | systemctl is-active / is-enabled |
| Port 9100 (Chat-Server) | offen | ss -ltn |
| Port 9443 (Chat HTTPS) | offen | ss -ltn |
| Port 6333 (Qdrant) | offen | ss -ltn |
| Port 8000 (hailo-ollama) | offen | ss -ltn |

## Sprachmodelle

| Was | Wert | Quelle |
|---|---|---|
| llm_mode | kaskade | settings.json |
| llm_profile | technical | settings.json |
| tentacle_llm.model (konfiguriert) | qwen2.5:7b-instruct | settings.json |
| embedding_bridge | nicht gesetzt (Vektorsuche aus) | settings.json |
| PC-Ollama erreichbar | ja | 192.168.178.20:11434/api/tags |
| PC-Ollama geladen (antwortet gerade) | qwen2.5:7b-instruct | /api/ps |
| PC-Ollama vorhandene Modelle | moondream:latest, qwen2.5:7b-instruct, moloch-coder:latest, deepseek-coder:6.7b, dolphin-llama3:8b, dolphin-mistral:7b | /api/tags |
| NPU-Modell (hailo-ollama) | qwen2.5:1.5b | localhost:8000/api/tags |

## Gedaechtnis

| Was | Wert | Quelle |
|---|---|---|
| facts.json | 23736 Bytes, 156 Eintraege | /mnt/moloch-data/memory/facts.json |
| identity.json | 519 Bytes, 8 Eintraege | /mnt/moloch-data/memory/identity.json |
| core_state.json | 176 Bytes, 6 Eintraege | /mnt/moloch-data/memory/core_state.json |
| Gespraechstage | 221 | /mnt/moloch-data/memory/conversations |
| Qdrant-Collections | moloch_muster, moloch_emergentis, moloch_memory, moloch_gedanken | localhost:6333/collections |
| GET /memory/recall | 1 Treffer, Quellen ['selbstwissen'] | localhost:9100 |

## Peripherie

| Was | Wert | Quelle |
|---|---|---|
| Soundkarten | 0 [RP1AudioOut    ]: simple-card - RP1-Audio-Out<br>                      RP1-Audio-Out<br> 1 [Device         ]: USB-Audio - USB Composite Device<br>                      USB Composite Device USB Composite Device at usb-xhci-hcd.1-2, full speed<br> 2 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0<br>                      vc4-hdmi-0<br> 3 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1<br>                      vc4-hdmi-1 | /proc/asound/cards |
| audio_source | auto | settings.json |
| Kamera-Position | pan=12.6 tilt=14.0 tracker=frozen | /live |
| Erkannt | person=True gesicht=markus | /live |

