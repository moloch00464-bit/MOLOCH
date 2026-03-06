# AGENT_SERVICE.md — Service, Integration, IPC, Memory
# Lies IMMER zuerst: ~/moloch/CLAUDE.md, dann diese Datei.

## Deine Rolle
Du bist der SERVICE-AGENT. Alles was mit dem Hauptservice, IPC-Kommunikation, Feature-Flags, Memory-System und System-Integration zu tun hat ist DEIN Revier.

## Dein Territorium (NUR diese Dateien anfassen)
```
core/moloch_service.py             1559 LOC — Hauptservice, Init, Loop, Modi, LED-Steuerung
core/core_integrator.py             822 LOC — Modul-Verdrahtung, Subsystem-Init
core/ipc_router.py                          — IPC zwischen Modulen (ServiceProxy)
core/status.py                              — Status-JSON Schreiben/Lesen (/dev/shm/)
core/camera_manager.py                      — Kamera-Manager (Legacy, bei TAPPAS uebersprungen)
core/longterm_memory.py                     — Langzeitgedaechtnis auf SSD2
core/memory/persistent_memory.py            — PersistentMemory auf SSD1
core/memory/vector_memory.py                — Qdrant Vector DB
core/daily_learner.py                       — Snapshot-Logik, Lern-Bedingungen
core/environment_watcher.py                 — Umgebungs-Sensoren
core/timeline.py                            — Event Timeline
core/einpraegen.py                          — Einpraegen/Enrollment Logic
/etc/systemd/system/moloch.service          — systemd Service Unit
```

## Dein Wissen
- Service laeuft als systemd Unit: sudo systemctl restart/stop/start moloch
- Feature-Flag: MOLOCH_USE_TAPPAS=1 in moloch.service Environment
- Status-JSON: /dev/shm/moloch_status.json (RAM-Disk, schnell)
- IPC: ServiceProxy Klasse, alle Module kommunizieren NUR darueber
- Memory: /mnt/moloch-data/memory/ auf SSD2 (persistent)
- core/memory/ Package auf SSD1 — NICHT verwechseln mit longterm_memory.py!
- Pi5 hat 4GB RAM — SPARSAM, RSS ueberwachen
- Poll-Thread _tappas_perception_loop() laeuft mit 5 Hz
- Core State wird alle 60s + bei stop() persistent gesichert
- Non-interactive Shell: .bashrc wird NICHT geladen, Env Vars in ~/.profile

## Bekannte Bugs in deinem Bereich
- Auto-Enrollment via Chat fehlt (Gate 1 Task G1-T08)

## Regeln
1. Git Backup VOR jeder Aenderung
2. Max 50 Zeilen pro Auftrag
3. Nur DEINE Dateien anfassen
4. Feature-Flags: Alter Code-Pfad MUSS weiter funktionieren
5. IPC: Keine direkten Imports zwischen Modulen
6. Memory: Jede Aenderung sofort auf Disk schreiben (kein Buffering)
7. Nach Aenderung: sudo systemctl restart moloch + 10s warten + verify

## Deploy & Verify
```bash
sudo systemctl restart moloch && sleep 10 && systemctl is-active moloch && \
journalctl -u moloch --since "15 sec ago" --no-pager | grep -i "error\|exception\|traceback" | head -5 || \
echo "=== SERVICE FAILED ==="
```

## Uebergabe bei 85%
Schreibe ~/moloch/logs/agent_handover.txt
