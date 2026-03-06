# AGENT_STRESSTEST.md — Chaos-Team
# Lies IMMER zuerst: CLAUDE.md, dann GATE_05_KONTEXT.md, dann diese Datei.

## Deine Rolle
Du bist das CHAOS-TEAM. Dein Job ist es M.O.L.O.C.H. zu STRESSEN — nicht kaputtzumachen, sondern Schwachstellen zu finden bevor sie im Echtbetrieb zuschlagen.

## Du darfst
- Scripts in scripts/ erstellen die das System stressen
- Den Service stoppen und starten
- Status-JSON, Logs, RAM, CPU, Temperatur überwachen
- RTSP-Stream öffnen und schließen
- PTZ-Befehle senden (über ONVIF)
- Mehrere Dinge gleichzeitig tun

## Du darfst NICHT
- Code in core/ ändern
- Dateien in config/ ändern
- Face-DB löschen oder ändern
- Den Pi rebooten
- Die NPU direkt ansprechen (nur über den Service)

## Stress-Szenarien die du testen sollst

### 1. PTZ-Stress
Sende 50 PTZ-Befehle in 10 Sekunden. Crasht der Arbiter?
```python
# scripts/stress_ptz.py
# Schnelle abwechselnde links/rechts Befehle
```

### 2. RAM-Überwachung unter Last
Starte tracking_diagnose.py 5x parallel. Steigt der RAM? Memory Leak?
```python
# scripts/stress_ram.py
# RSS alle 10 Sekunden loggen über 30 Minuten
```

### 3. RTSP-Reconnect
Kann der Service einen RTSP-Disconnect überleben? Sonoff kurz offline simulieren.
```python
# scripts/stress_rtsp.py
# RTSP Stream öffnen von extern → Konflikt mit Service?
```

### 4. NPU-Auslastung
Wie reagiert das System wenn die NPU dauerhaft unter Volllast steht?
```python
# scripts/stress_npu.py
# Überwache FPS-Einbrüche, Thermal Throttling, NPU Errors
```

### 5. Status-JSON Korruption
Was passiert wenn moloch_status.json kurz gelöscht oder geleert wird?
```python
# scripts/stress_status.py
# Status-JSON manipulieren, prüfen ob Service crashed
```

### 6. Langzeit-Drift
Über 6 Stunden: Steigen Threads? Steigt RAM? Sinken FPS? Steigt Temperatur?
```python
# scripts/stress_longrun.py
# Alle 60 Sekunden: RAM, Threads, FPS, Temp, NPU-Stage loggen
```

### 7. Schneller Modus-Wechsel
Manuell→Autonom→Manuell→Autonom in schneller Folge. Crasht der Tracker?
```python
# scripts/stress_mode_switch.py
# 20x umschalten in 30 Sekunden
```

### 8. Face-DB Hot-Reload
Während der Service läuft: Face-DB überschreiben. Crashed ArcFace?
```python
# scripts/stress_facedb.py
# face_embeddings.json überschreiben während Inference läuft
```

## Output
Alle Ergebnisse in ~/moloch/logs/stress_results.txt
Format pro Test:
```
=== TEST: [Name] ===
Dauer: Xs
Ergebnis: PASS/FAIL
Details: ...
RAM vorher/nachher: X/Y MB
Crashes: 0/N
```

## Bewertung
- 8/8 PASS → System ist stressresistent
- 6-7/8 PASS → Bekannte Schwächen, aber stabil
- <6/8 PASS → Kritische Schwachstellen, Gate 0.5 nicht bestanden

## Übergabe bei 85%
Schreibe Ergebnisse in ~/moloch/logs/stress_results.txt
