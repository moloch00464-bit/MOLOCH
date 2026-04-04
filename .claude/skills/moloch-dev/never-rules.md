# NEVER-DO Regeln — Vollstaendige Liste

Aus der Analyse von 1200 Commits, 377 davon BACKUP/Fix/Revert:

## NEVER 1: GStreamer-String blind aendern
**Datei**: `tappas_pipeline.py`, Methode `_build_pipeline_string()`
**Warum**: Gst.parse_launch() crashed bei jedem Typo mit SEGV.
**Stattdessen**: Element-Properties einzeln via `pipeline.get_by_name("x").set_property()` aendern.

## NEVER 2: Pan-Vorzeichen aendern
**Datei**: `camera.py`, Zeile 732: `pan_delta = -error_x`
**Warum**: MINUS ist KORREKT. Sonoff Pan ist physisch invertiert. 6x in Git-Historie "gefixt" und zurueckgedreht.
**Regel**: Diese Zeile ist TABU.

## NEVER 3: ArcFace-Threshold aendern
**Warum**: Root Cause ist GStreamer/HailoRT Embedding-Inkompatibilitaet.
**Loesung**: Enrollment muss durch denselben GStreamer-Pfad laufen wie Live-Inference.

## NEVER 4: Mehrere ROT-Dateien in einem Commit
**Warum**: Rollback unmoeglich bei Multi-File Shotgun Surgery.
**Regel**: 1 Commit = 1 Datei (bei ROT-Dateien).

## NEVER 5: subprocess ohne timeout
**Warum**: Zombie-Prozesse auf 4GB Pi.
**Pattern**: `subprocess.run([...], capture_output=True, timeout=30, text=True)`

## NEVER 6: JSON direkt schreiben
**Warum**: Partial-Write bei Crash korrumpiert Datei.
**Pattern**: tempfile + os.replace() (atomic)

## NEVER 7: Runtime-State in Git committen
**Dateien**: `last_face_position.json`, `learned_patrol_positions.json`
**Regel**: Vor git add pruefen.

## NEVER 8: shell=True in subprocess
**Warum**: Command Injection Risiko.
**Regel**: Immer Liste statt String.

## NEVER 9: HailoRT falscher dtype
**Warum**: Buffer-Size-Mismatch (4x) bei float32 statt uint8.
**Regel**: VOR Implementierung Input-Format pruefen.

## NEVER 10: np.ndarray Type-Hint in moloch_service.py
**Warum**: numpy nur lokal importiert → NameError beim Parsen.

## NEVER 11: __pycache__ ignorieren
**Warum**: Service laeuft alten Bytecode.
**Regel**: Nach Aenderung: `find ~/moloch/core -name __pycache__ -exec rm -rf {} +`

## NEVER 12: Im Worktree Service testen
**Warum**: Service laeuft von ~/moloch/, nicht vom Worktree.
