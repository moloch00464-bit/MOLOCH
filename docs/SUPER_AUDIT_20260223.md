# M.O.L.O.C.H. SUPER AUDIT — 2026-02-23

> Kompletter Systemcheck nach allen Umbauten (Blueprint 5.2/5.3, Refactoring, Spotify, Memory)

---

## GESAMTBEWERTUNG: 8/10

| Kategorie | Status | Score |
|-----------|--------|-------|
| Code-Qualitaet | OK | 9/10 |
| Integration (5.2/5.3) | OK | 10/10 |
| Thread Safety | WARNUNG | 8/10 |
| Settings Konsistenz | WARNUNG | 6/10 |
| Memory/Persistence | OK | 9/10 |
| NPU | OK | 9/10 |
| Encoding | OK | 10/10 |
| Spotify | WARNUNG | 7/10 |
| Git Hygiene | WARNUNG | 5/10 |

---

## 1. CODE AUDIT

### Codebase-Groesse

| Metrik | Wert |
|--------|------|
| Python-Dateien in core/ | **95** |
| Gesamtzeilen Python | **44.988** |
| Groesste Datei | moloch_unified_panel.py (2.500 Zeilen) |

### Top 10 groesste Dateien

| Zeilen | Datei | Bewertung |
|-------:|-------|-----------|
| 2.500 | core/gui/moloch_unified_panel.py | WARNUNG — Kandidat fuer Aufspaltung |
| 1.358 | core/console/moloch_console.py | OK |
| 1.357 | core/mpo/autonomous_tracker.py | OK — komplex aber fokussiert |
| 1.209 | core/hardware/camera_cloud_bridge.py | OK |
| 1.149 | core/hardware/camera.py | OK |
| 1.116 | core/voice_pipeline.py | OK |
| 1.061 | core/vision/gst_hailo_pose_detector.py | OK |
| 1.044 | core/inference_engine.py | OK |
| 1.007 | core/personality/personality_engine.py | OK |
| 991 | core/spotify_controller.py | OK |

### Bare Excepts

**Gefunden: 2** (beide in `core/audio/audio_manager.py`)

| Zeile | Code | Problem |
|------:|------|---------|
| 188 | `except: pass` | Bluetooth Battery-Parsing — faengt SystemExit/KeyboardInterrupt |
| 211 | `except: pass` | Device-ID-Parsing — faengt SystemExit/KeyboardInterrupt |

**Typed Excepts: ~664** — Ratio: 0.3% bare (EXZELLENT)

**Empfehlung**: Beide auf `except (ValueError, IndexError):` aendern.

### Tote Imports

Stichproben der Hauptdateien (moloch_service.py, core_integrator.py, voice_pipeline.py, inference_engine.py): **Keine toten Imports gefunden.**

Import-Hygiene: GUT.

### Dead Code

Keine offensichtlichen Dead-Code-Bloecke gefunden. Deprecated-Methoden (`get_attention()`, `get_presence()` in core_integrator.py) sind als Backward-Compat-Shims markiert und werden noch von model_orchestrator.py genutzt.

---

## 2. INTEGRATION CHECK (Blueprint 5.2/5.3)

### Variablen-Migration: tension + dominance

| Modul | Variablen | Status |
|-------|-----------|--------|
| core/core_integrator.py | tension, dominance, cpu_temp, npu_load | OK — Primaerquelle |
| core/personality/personality_engine.py | Liest zone via get_personality_zone() | OK |
| core/led_controller.py | Liest effects.led_feedback_frequency | OK |
| core/gui/panel_avatar.py | tension, dominance, zone, cpu_temp_norm | OK |
| core/gui/avatar_pygame.py | tension, dominance (Farbe+Pupille) | OK |
| core/model_orchestrator.py | get_attention() SHIM (= tension*0.7 + 0.3*|dom|) | OK — Legacy-Adapter |
| core/inference_engine.py | update_inputs() API | OK |
| core/camera_manager.py | Kein CoreIntegrator-Zugriff | OK — neutral |
| core/ipc_router.py | Kein CoreIntegrator-Zugriff | OK — neutral |
| core/voice_pipeline.py | get_zone_system_prompt_addon() | OK |
| core/spotify_controller.py | get_personality_zone() | OK |
| core/mpo/autonomous_tracker.py | get_tension() fuer adaptive Speed | OK |
| core/perception_engine.py | Liest tension fuer Model-Score-Boost | OK |
| core/longterm_memory.py | Speichert tension+dominance persistent | OK |

### Alte Variablen (attention/presence/energy)

| Referenz | Datei | Status |
|----------|-------|--------|
| get_attention() | core_integrator.py:242 | DEPRECATED-Shim, wird von model_orchestrator genutzt |
| get_presence() | core_integrator.py:247 | DEPRECATED-Shim, kein aktiver Nutzer |
| presence_duration | perception_buffer.py:157 | Trending-Variable, read-only |
| pose_energy | perception_frame.py:49 | NEUES Feld (nicht das alte "energy") |
| energy_level | tts/voice_selector.py | LEGITIM — Voice-Klassifikation, kein State |

**Verdict**: Migration 100% komplett. Alle "alten" Referenzen sind entweder Shims oder andere Bedeutung.

---

## 3. THREAD SAFETY

### Score: 8.5/10

### Lock-Uebersicht

| Modul | Locks | Bewertung |
|-------|-------|-----------|
| core_integrator.py | `_lock` (Lock) | OK — alle State-Zugriffe geschuetzt |
| model_orchestrator.py | `_ctx_lock`, `_attention_level_lock`, `_configuring` (Event) | OK |
| camera_manager.py | `_frame_lock`, `_annotated_lock`, `_st_lock`, `_transition_lock` | WARNUNG |
| inference_engine.py | `_fps_lock` | OK |
| voice_pipeline.py | `_lock`, `_msg_lock` | WARNUNG |
| spotify_controller.py | `_lock`, `_auto_dj_lock` | OK |
| perception_buffer.py | `_lock` | OK |
| hailo_manager.py | `_lock` (RLock), `_available` (Condition) | VORBILDLICH |
| longterm_memory.py | `_conv_lock`, `_facts_lock`, `_state_lock` | OK |
| led_controller.py | Keine Locks | OK (single-threaded Kontext) |

### Gefundene Race Conditions

| Prio | Datei | Problem | Risiko |
|------|-------|---------|--------|
| WARNUNG | camera_manager.py | `_moloch_has_control` Property ohne Lock gelesen | NIEDRIG — Boolean, max 1 Frame inkonsistent |
| WARNUNG | voice_pipeline.py | `_processing`, `_recording`, `_speaking` Flags ungeschuetzt | MITTEL — Doppel-TTS moeglich bei schlechtem Timing |
| INFO | spotify_controller.py | `_auto_dj_active` in Loop ohne Lock | NIEDRIG — Stop-Signal max 0.5s verzoegert |
| INFO | led_controller.py | Keine Locks | NIEDRIG — Inference ist single-threaded |

### Deadlock-Analyse

Keine Deadlocks gefunden. Lock-Ordering ist konsistent (frame_lock → annotated_lock). Keine verschachtelten Lock-Zyklen.

---

## 4. SETTINGS KONSISTENZ

### Score: 6/10

### settings.json Struktur

| Key | Im JSON | Geladen | Gespeichert | Status |
|-----|---------|---------|-------------|--------|
| version | Ja | Ja | Ja | OK |
| thresholds.scrfd_conf | Ja | Ja | Ja | OK |
| thresholds.scrfd_nms | Ja | Ja | Ja | OK |
| thresholds.arcface_thresh | Ja | Ja | Ja | OK |
| thresholds.yolo_conf | Ja | Ja | Ja | OK |
| thresholds.pose_conf | Ja | **NEIN** | **NEIN** | KRITISCH |
| thresholds.hand_conf | Ja | **NEIN** | **NEIN** | KRITISCH |
| hand_occlusion.* | Ja | Ja | Ja | OK |
| audio.* | Ja | Ja | Ja | OK |
| camera.* | Ja | Ja | Ja | OK |
| learner.* | Ja | Ja | Ja | OK |
| orchestration_mode | Ja | Ja | Ja | OK |
| active_models | Ja | Ja | Ja | OK |
| mpo.* | Ja | **NEIN** | **NEIN** | KRITISCH |
| gestures.* | Ja | **NEIN** | **NEIN** | KRITISCH |

### Kritische Settings-Probleme

1. **`mpo` Block verwaist**: tension_tau, dominance_drift, zone_hysteresis, berserker_threshold, thermal_damping_start — definiert in JSON aber NIE geladen/gespeichert in `_load_settings()`/`_save_settings()`
2. **`gestures` Block verwaist**: wave_enabled, thumbs_up_enabled etc. — gleich
3. **`pose_conf` + `hand_conf`**: Im JSON aber nicht im Load/Save-Zyklus
4. **perception_weights.json**: Floating-Point-Artefakt (`pose` weight: `-0.09999999999999998`)

### Settings-Laden (4 Stellen)

| Datei | Methode | Was |
|-------|---------|-----|
| moloch_service.py:711 | `_load_settings()` | Hauptlader bei init() |
| moloch_unified_panel.py:1166 | `_load_settings()` | Manueller Reload |
| popup_npu_thresh.py:376 | Direkt | Liest Thresholds fuer Slider |
| popup_tracker.py:400 | Direkt | Liest Tracker-Settings |

---

## 5. MEMORY/PERSISTENCE

### Score: 9/10

### Dateistruktur auf SSD2

```
/mnt/moloch-data/memory/
├── identity.json          519 B    — Identitaet (Name, Personalities, Owner)
├── facts.json             9.2 KB   — 99 Fakten (aktuell 2026-02-23)
├── core_state.json        170 B    — Letzter State (tension, dominance, zone)
├── conversations/
│   ├── 2026-02-22.json    7.9 KB   — Gespraeche Tag 1
│   └── 2026-02-23.json    45.8 KB  — Gespraeche Tag 2 (~254 Messages)
└── spotify/
    ├── raw_stats.json     7.7 KB   — Rohdaten
    └── spotify_profile.json 22 KB  — Aufbereitetes Profil
```

### Persistence-Validierung

| Check | Status |
|-------|--------|
| identity.json vollstaendig | OK — Name, Personalities, Owner, Crew |
| facts.json wird geschrieben | OK — 99 Eintraege, aktiv |
| conversations/ werden gespeichert | OK — Sofort auf Disk, kein Buffering |
| core_state.json bei Shutdown | OK — Atomare Writes (.tmp + os.replace) |
| core_state.json bei Start geladen | OK — v2 Format (tension+dominance) |
| Thread-Safety Memory | OK — Separate Locks fuer facts/state/conv |

### core_state.json (aktuell)

```json
{
  "tension": 1.0,
  "dominance": 1.0,
  "cpu_temp": 0.4822,
  "personality_zone": "guardian",
  "last_updated": "2026-02-23T16:51:50",
  "uptime_seconds": 1680
}
```

### Bekanntes Problem

**Dual Memory System**: `core/memory/persistent_memory.py` (SSD1) und `core/longterm_memory.py` (SSD2) existieren parallel. Sync nur SSD1→SSD2, nie rueckwaerts.

---

## 6. SPOTIFY

### Score: 7/10

### Funktionsstatus

| Feature | Status |
|---------|--------|
| Auth (SpotifyOAuth) | OK — Token cached |
| Playlists gecacht | OK — spotify_profile.json (22 KB) |
| Genre-Lock | OK — 20 erlaubte, 14 gebannte Genres |
| Auto-DJ (Zone-basiert) | OK — Monitort Zone alle 5s |
| Time-based Modulation | OK — morgen/tag/abend/nacht |
| Voice Commands | OK — search_and_play(), play_artist() etc. |

### Zone-Mapping

```
guardian:  VNV Nation, Depeche Mode, KANGA, Apoptygma Berzerk ...
shadow:   Suicide Commando, Vomito Negro, ESA, Phase Fatale ...
berserker: Ministry, 16Volt, Prong, Terrorfakt ...
```

### Spotify-Probleme

| Prio | Problem | Beschreibung |
|------|---------|-------------|
| WARNUNG | Credentials in Git | `.env.spotify` mit Client-ID/Secret im Repo |
| INFO | Profile-Cache stale | Kein Auto-Refresh von spotify_profile.json |
| INFO | Emergentis fehlt | Kein Zone-Mapping fuer dritte Personality |

---

## 7. NPU

### Score: 9/10

### HEF-Modelle auf SSD2

| Modell | Groesse | Zweck | Status |
|--------|---------|-------|--------|
| scrfd_10g.hef | 5.8 MB | Face Detection | AKTIV |
| arcface_mobilefacenet.hef | 2.6 MB | Face Recognition | AKTIV |
| yolov8m_h10.hef | 21 MB | Person Detection | AKTIV |
| yolov8s_pose_h10.hef | 14 MB | Pose/Keypoints | AKTIV |
| yolov8m_pose_h10.hef | 29 MB | Pose Medium | Reserve |
| yolov11m_h10.hef | 28 MB | YOLOv11 Detection | Reserve |
| resnet_v1_50_h10.hef | 24 MB | Classification | Reserve |
| face_attr_resnet_v1_18.hef | 6.9 MB | Face Attributes | Reserve |
| yolov5n_seg_h10.hef | 3.5 MB | Segmentation | Reserve |
| hand_landmark_lite.hef | 1.3 MB | Hand Landmarks | Reserve |

**Gesamt-RAM**: ~180 MB von 8 GB (2.3%) — HERVORRAGEND

### Whisper

Kein separates Whisper-HEF gefunden. Whisper nutzt shared VDevice (CPU+NPU Hybrid, dokumentiert in Memory).

### Service-Status

```
moloch.service: ACTIVE (running)
```

---

## 8. ENCODING

### Score: 10/10

| Check | Ergebnis |
|-------|----------|
| Kaputte UTF-8 Sequenzen (Ã¶, Ã¼ etc.) | **KEINE** gefunden |
| Explizite Encoding-Deklarationen | 10 Dateien mit `# -*- coding: utf-8 -*-` |
| JSON-Encoding | Alle `ensure_ascii=False` — Umlaute korrekt |
| Memory-Dateien | UTF-8 verifiziert |

**Verdict**: Sauber. Keine Mixed-Encoding-Probleme.

---

## 9. GIT STATUS

### Score: 5/10

### Ueberblick

| Metrik | Wert |
|--------|------|
| Branch | main |
| Unpushed Commits | **135** |
| Uncommitted Changes | 7 Dateien (5 modified, 2 new) |

### Uncommitted Changes

```
M  config/perception_weights.json
M  config/settings.json
M  core/gui/panel_models.py
M  core/gui/popups/popup_hardware.py
M  core/gui/popups/popup_settings.py
?? core/gui/popups/popup_npu_thresh.py   (NEU)
?? core/gui/popups/popup_tracker.py      (NEU)
```

**Netto-Aenderung**: +301 / -150 Zeilen

### Letzte 10 Commits

```
b75f063 BACKUP vor Popup-Fenster Ueberarbeitung
628777a BACKUP vor Chat Auto-Scroll Fix
cf737cf BACKUP vor PTZ Tracking Ueberarbeitung
64307bf BACKUP vor Avatar-Auge Voice-Reaktion
18dd3f6 BACKUP vor Spotify Controller Ausbau
ada6f63 BACKUP vor Tracker Dead Zone + Smoothing Fix
41f79bc BACKUP vor Avatar-Auge PyGame Panel-Einbettung
111cefd BACKUP vor PyGame Avatar-Auge Implementation
829118c BACKUP vor Core Integrator v2 Blueprint 5.2+5.3
05eb263 BACKUP vor Avatar-Auge visuell Overhaul
```

### Git-Probleme

| Prio | Problem |
|------|---------|
| KRITISCH | **135 Commits nicht gepusht** — lokale Arbeit ungesichert gegen Hardware-Ausfall |
| WARNUNG | 7 uncommitted Changes — Popup-Ueberarbeitung in Progress |

---

## ZUSAMMENFASSUNG ALLER FINDINGS

### KRITISCH (Sofort fixen)

| # | Problem | Datei | Fix |
|---|---------|-------|-----|
| K1 | 135 unpushed Commits | git | `git push origin main` |
| K2 | `mpo` Settings verwaist | moloch_service.py | In _load/_save einbauen |
| K3 | `gestures` Settings verwaist | moloch_service.py | In _load/_save einbauen |
| K4 | `pose_conf`+`hand_conf` nicht persistiert | moloch_service.py | In _load/_save einbauen |

### WARNUNG (Bald fixen)

| # | Problem | Datei | Fix |
|---|---------|-------|-----|
| W1 | 2 bare excepts | audio_manager.py:188,211 | `except (ValueError, IndexError):` |
| W2 | camera_manager Property ohne Lock | camera_manager.py | `_transition_lock` in Property |
| W3 | VoicePipeline Boolean-Flags ungeschuetzt | voice_pipeline.py | `_lock` oder `threading.Event` |
| W4 | Spotify Credentials in Git | .env.spotify | In .gitignore aufnehmen |
| W5 | perception_weights Float-Artefakt | perception_weights.json | Auf `-0.1` runden |

### INFO (Nice to have)

| # | Problem | Datei | Empfehlung |
|---|---------|-------|------------|
| I1 | spotify_profile.json kein Auto-Refresh | spotify_controller.py | Bei Service-Start refreshen |
| I2 | "emergentis" Zone fehlt in Spotify | spotify_controller.py | Zone-Mapping ergaenzen |
| I3 | Dual Memory System (SSD1+SSD2) | persistent_memory.py | Langfristig konsolidieren |
| I4 | Deprecated get_attention/get_presence Shims | core_integrator.py | Deprecation-Warning loggen |
| I5 | moloch_unified_panel.py 2500 Zeilen | moloch_unified_panel.py | Weitere Aufspaltung |

### OK (Funktioniert)

| Bereich | Details |
|---------|---------|
| Blueprint 5.2 Migration | 100% komplett, tension+dominance ueberall |
| Blueprint 5.3 Physiological | cpu_temp + npu_load integriert |
| Exception Handling | 664 typed vs 2 bare (99.7% sauber) |
| UTF-8 Encoding | Makellos, keine Probleme |
| Memory Persistence | Atomic writes, Thread-safe, sofortige Disk-Writes |
| NPU RAM | 180MB von 8GB (2.3%) |
| Service | Laeuft stabil |
| Hailo Manager | Vorbildliche Thread-Safety (RLock + Condition) |
| Lock-Ordering | Keine Deadlocks, konsistente Reihenfolge |
| Import-Hygiene | Keine toten Imports gefunden |

---

## MODUL-UEBERSICHT

| Modul | Zeilen | Status | Hauptproblem |
|-------|-------:|--------|-------------|
| core/moloch_service.py | 881 | OK | Settings Load/Save unvollstaendig |
| core/core_integrator.py | 683 | OK | Deprecated Shims aufzuloesen |
| core/inference_engine.py | 1.044 | OK | — |
| core/model_orchestrator.py | ~800 | OK | Nutzt noch get_attention() Shim |
| core/voice_pipeline.py | 1.116 | WARNUNG | Boolean-Flag Race Condition |
| core/camera_manager.py | ~600 | WARNUNG | Property ohne Lock |
| core/perception_engine.py | ~500 | OK | — |
| core/led_controller.py | ~300 | OK | Keine Locks, aber safe by design |
| core/spotify_controller.py | 991 | WARNUNG | Credentials in Git |
| core/longterm_memory.py | 491 | OK | Dual-System mit SSD1 |
| core/audio/audio_manager.py | ~300 | WARNUNG | 2 bare excepts |
| core/personality/personality_engine.py | 1.007 | OK | — |
| core/console/moloch_console.py | 1.358 | OK | — |
| core/mpo/autonomous_tracker.py | 1.357 | OK | — |
| core/hardware/camera.py | 1.149 | OK | — |
| core/hardware/camera_cloud_bridge.py | 1.209 | OK | — |
| core/gui/moloch_unified_panel.py | 2.500 | INFO | Gross, Aufspaltung empfohlen |
| core/gui/panel_avatar.py | ~500 | OK | — |
| core/gui/avatar_pygame.py | ~400 | OK | — |

---

*Audit durchgefuehrt am 2026-02-23 von Claude Opus 4.6*
*95 Python-Dateien, 44.988 Zeilen analysiert*
*Systemgesundheit: **8/10***
