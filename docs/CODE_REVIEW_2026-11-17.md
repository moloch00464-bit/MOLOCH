# Code-Review 2026-11-17 — fertige Patches

Erstellt in einer Cloud-Session (kein Pi, MCP nicht verbunden). Der Session-Lock liess sich dort nicht aufheben, deshalb wurde **kein Code direkt geaendert**. Stattdessen liegen hier fertige Patches. Jeder einzelne ist geprueft mit `git apply --check` gegen `a5bf8cb`, `py_compile`, ruff und — wo moeglich — einem isolierten Vorher/Nachher-Test.

Sicherheits- und Datenschutz-Befunde stehen bewusst **nicht** in diesem Dokument, solange das Repo oeffentlich ist. Siehe Session-Chat vom 2026-11-17.

## Anwenden (auf dem Pi, mit normalem Startprotokoll)

```bash
cd ~/moloch
# 1. Patches aus diesem Dokument extrahieren -> /tmp/moloch_patch_*.diff
awk '/^<!-- PATCH /{name=$3} /^```diff$/{if(name){f="/tmp/moloch_patch_" name ".diff"; next}} /^```$/{f=""; name=""} f{print > f}' docs/CODE_REVIEW_2026-11-17.md
ls /tmp/moloch_patch_*.diff

# 2. Pro Patch: pruefen, anwenden, Syntax, Restart, Audit, einzeln committen
git apply --check /tmp/moloch_patch_01_service_threading_import.diff
git apply /tmp/moloch_patch_01_service_threading_import.diff
python3 -c "import core.moloch_service"
find ~/moloch/core -name __pycache__ -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch
python3 ~/moloch/moloch_audit.py --auto
git add core/moloch_service.py && git commit -m "fix(service): lokales import threading entfernt (UnboundLocalError)"
```

Reihenfolge einhalten: 02 setzt 01 voraus, 05 setzt 04 voraus. Alle anderen sind unabhaengig.

Die awk-Extraktion ist byte-genau. Patches 03, 06 und 08 enthalten CRLF-Zeilenenden, weil die Zieldateien CRLF haben. Wer stattdessen aus der GitHub-Weboberflaeche kopiert, verliert die CRs: dann `git apply --ignore-whitespace` nutzen (funktioniert, ergibt aber gemischte Zeilenenden in diesen Dateien).

## Uebersicht

| # | Datei | Ampel | Agent | Danach |
|---|---|---|---|---|
| 01 | `core/moloch_service.py` | ROT | service | Service-Restart |
| 02 | `core/moloch_service.py` | ROT | service | Service-Restart |
| 03 | `core/camera_manager.py` | – | hardware | Service-Restart |
| 04 | `core/gui/moloch_unified_panel.py` | GELB | gui | Panel neu starten |
| 05 | `core/gui/moloch_unified_panel.py` | GELB | gui | Panel neu starten |
| 06 | `core/gui/popups/popup_whisper.py, popup_settings.py` | GELB | gui | Panel neu starten |
| 07 | `core/tts.py` | – | voice | Service-Restart |
| 08 | `moloch.service` | – | service | Unit nach /etc kopieren + **Reboot** (CLAUDE.md 0c), danach `rm -f /tmp/moloch_env` |

## Patch 01 — service threading import

**Was:** Lokales `import threading` in `_execute_panel_cmd` entfernt. Es machte `threading` fuer die ganze Methode lokal → `UnboundLocalError` in `mic_test` (WiFi), `teach_trigger`, `npu_clip`, `npu_ocr`, `npu_vlm_describe`. Sichtbar z. B. am Mic-Test-Button im Audio-Popup.

**Beleg:** Isoliert ausgefuehrt: `npu_ocr/npu_clip/npu_vlm_describe` vorher `UnboundLocalError`, nachher ok.

<!-- PATCH 01_service_threading_import -->
```diff
--- a/core/moloch_service.py
+++ b/core/moloch_service.py
@@ -3079,7 +3079,9 @@
                         logger.info(f"[IPC] Reflect gesprochen: {comment[:60]}")
                     logger.info(f"[IPC] Reflect fertig: {result}")
 
-                import threading
+                # Kein lokales 'import threading' hier: das macht 'threading' fuer
+                # die ganze Methode lokal -> UnboundLocalError in mic_test,
+                # teach_trigger, npu_clip/ocr/vlm. Modul-Import (Kopf) reicht.
                 threading.Thread(target=_do_reflect_ipc, daemon=True,
                                  name="IPC-Reflect").start()
                 logger.info("[IPC] Reflect gestartet")
```

## Patch 02 — service save settings kein rumpf

**Was:** `_save_settings`: Wenn `settings.json` nicht lesbar ist, wird abgebrochen statt mit `{"version": 1}` weiterzuschreiben. Vorher gingen dabei alle Keys anderer Prozesse verloren.

**Beleg:** Kaputte Datei mit `tentacle_llm` + `llm_profile`: vorher ueberschrieben (nur noch Service-Keys), nachher unveraendert. Normalfall: fremde Keys bleiben erhalten.

<!-- PATCH 02_service_save_settings_kein_rumpf -->
```diff
--- a/core/moloch_service.py
+++ b/core/moloch_service.py
@@ -3524,13 +3524,19 @@
         """
         # Bestehende Datei lesen (Merge statt Overwrite)
         data = {"version": 1}
-        try:
-            if os.path.exists(SETTINGS_PATH):
+        if os.path.exists(SETTINGS_PATH):
+            try:
                 with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                     data = json.load(f)
-                data["version"] = 1
-        except Exception:
-            data = {"version": 1}
+            except Exception as e:
+                # NICHT mit Rumpf-Dict weiterschreiben: sonst gehen Keys anderer
+                # Prozesse verloren (tentacle_llm, llm_profile, hand_detection_enabled ...)
+                logger.error(f"[SETTINGS] settings.json nicht lesbar ({e}) — Speichern abgebrochen")
+                return
+            if not isinstance(data, dict):
+                logger.error("[SETTINGS] settings.json ist kein JSON-Objekt — Speichern abgebrochen")
+                return
+            data["version"] = 1
 
         _inf = self._inference
 
```

## Patch 03 — hardware ptz goto calibrate

**Was:** `ptz_goto` uebergab `pan=`/`tilt=`, `move_absolute` erwartet `pan_deg`/`tilt_deg` → `TypeError`. `ptz_calibrate` baute `CameraCloudBridgeSync()` ohne `config` → `TypeError`; nutzt jetzt den bestehenden CloudController wie `cloud_set_night_mode()`.

**Beleg:** Mit echter Signatur von `move_absolute`: vorher `TypeError`, nachher `move_absolute(75.0, 10.0)`. Kalibrierung: vorher `TypeError`, nachher ok.

**Achtung:** Die Positionen in `ptz_goto` (`schreibtisch` 75/10, `wohnzimmer` −90/0 …) wurden wegen des Bugs **nie real angefahren**. Nach dem Fix faehrt auch die DecisionEngine (`ptz_move` → `tuer`/`schreibtisch`) die Kamera wirklich. Erst im PTZ-Panel jede Position einmal von Hand pruefen. Pan ist invertiert — falls eine Position falsch liegt, den Wert korrigieren, nicht die Vorzeichen-Logik (NEVER #2).

<!-- PATCH 03_hardware_ptz_goto_calibrate -->
```diff
--- a/core/camera_manager.py
+++ b/core/camera_manager.py
@@ -956,15 +956,23 @@
         if not cam.is_connected:
             cam.connect()
         pan, tilt = positions[position]
-        cam.move_absolute(pan=pan, tilt=tilt)
-        logger.info(f"[PTZ] goto {position} ({pan}, {tilt})")
+        # Signatur: move_absolute(pan_deg, tilt_deg, speed) — 'pan='/'tilt=' gab TypeError
+        if cam.move_absolute(pan_deg=pan, tilt_deg=tilt):
+            logger.info(f"[PTZ] goto {position} ({pan}, {tilt})")
+        else:
+            logger.warning(f"[PTZ] goto {position} fehlgeschlagen (Kamera nicht verbunden?)")
 
     def ptz_calibrate(self):
-        """PTZ Kalibrierung triggern."""
-        from core.hardware.camera_cloud_bridge import CameraCloudBridgeSync
-        bridge = CameraCloudBridgeSync()
-        bridge.trigger_ptz_calibration()
-        logger.info("[PTZ] Kalibrierung getriggert")
+        """PTZ Kalibrierung triggern (ueber bestehende Cloud-Verbindung)."""
+        # Vorher: CameraCloudBridgeSync() ohne config -> TypeError.
+        # Bestehenden CloudController nutzen wie cloud_set_night_mode().
+        if not self._cloud or not self._cloud.connected:
+            logger.warning("[PTZ] Kalibrierung nicht moeglich: Cloud nicht verbunden")
+            return
+        if self._cloud.run(self._cloud.bridge.trigger_ptz_calibration()):
+            logger.info("[PTZ] Kalibrierung getriggert")
+        else:
+            logger.warning("[PTZ] Kalibrierung von Cloud abgelehnt")
 
     def cloud_set_night_mode(self, level):
         """Weisse LEDs / Night Vision Mode setzen."""
```

## Patch 04 — gui lambda closure

**Was:** 6× `except Exception as e: root.after(0, lambda: …{e})` → `lambda err=e:`. Python loescht `e` am Ende des except-Blocks; die Lambda lief spaeter in einen `NameError`, die Fehlermeldung kam nie an.

**Beleg:** ruff F821: 6 → 0.

<!-- PATCH 04_gui_lambda_closure -->
```diff
--- a/core/gui/moloch_unified_panel.py
+++ b/core/gui/moloch_unified_panel.py
@@ -905,8 +905,8 @@
 
             except Exception as e:
                 logger.error(f"Service init failed: {e}\n{traceback.format_exc()}")
-                self.root.after(0, lambda: self.status_label.config(
-                    text=f"Service FEHLER: {e}", fg="#ff4444"))
+                self.root.after(0, lambda err=e: self.status_label.config(
+                    text=f"Service FEHLER: {err}", fg="#ff4444"))
 
         threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()
 
@@ -2094,8 +2094,8 @@
                     proc.send_signal(signal.SIGINT)
                     proc.wait(timeout=3)
                 except Exception as e:
-                    self.root.after(0, lambda: self._append_chat(
-                        f"[Mic Test] Aufnahme fehlgeschlagen: {e}", "system"))
+                    self.root.after(0, lambda err=e: self._append_chat(
+                        f"[Mic Test] Aufnahme fehlgeschlagen: {err}", "system"))
                     return
 
                 if not os.path.exists(test_path) or os.path.getsize(test_path) < 1000:
@@ -2134,8 +2134,8 @@
                             self.root.after(0, lambda: self._append_chat(
                                 "[Mic Test] Nichts erkannt", "system"))
                     except Exception as e:
-                        self.root.after(0, lambda: self._append_chat(
-                            f"[Mic Test] Whisper Fehler: {e}", "system"))
+                        self.root.after(0, lambda err=e: self._append_chat(
+                            f"[Mic Test] Whisper Fehler: {err}", "system"))
 
                     # Whisper VDevice freigeben (sonst blockiert naechste Spracheingabe!)
                     try:
@@ -2258,8 +2258,8 @@
                     boxes, landmarks, _lb_px, _lb_py, _lb_rw, _lb_rh)
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] SCRFD Fehler: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] SCRFD Fehler: {err}", "system"))
                 return
 
             # Groesstes Gesicht finden
@@ -2303,8 +2303,8 @@
                 embedding = normalize_arcface(embedding)
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] ArcFace Fehler: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] ArcFace Fehler: {err}", "system"))
                 return
 
             # Save to face_embeddings.json
@@ -2344,8 +2344,8 @@
                 logger.info(f"[SNAPSHOT] Markus embedding saved to {db_path}")
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] Speichern fehlgeschlagen: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] Speichern fehlgeschlagen: {err}", "system"))
 
         threading.Thread(target=do_snapshot, daemon=True).start()
 
```

## Patch 05 — gui deepseek history limit

**Was:** `conversation_history` auf 10 Wortwechsel + aktuelle Frage begrenzt (vorher unbegrenzt, komplett bei jeder DeepSeek-Anfrage). Bei Fehler wird die unbeantwortete Frage wieder entfernt.

**Beleg:** py_compile ok.

<!-- PATCH 05_gui_deepseek_history_limit -->
```diff
--- a/core/gui/moloch_unified_panel.py
+++ b/core/gui/moloch_unified_panel.py
@@ -2425,6 +2425,10 @@
             self.conversation_history.append({
                 "role": "user", "content": message_content
             })
+            # History begrenzen (10 Wortwechsel + aktuelle Frage) — sonst waechst
+            # jede Anfrage unbegrenzt (Kosten, Kontextlimit, 15s-Timeout)
+            if len(self.conversation_history) > 21:
+                del self.conversation_history[:-21]
 
             import requests
             api_msgs = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
@@ -2463,6 +2467,9 @@
 
         except Exception as e:
             logger.error(f"DeepSeek error: {e}")
+            # Unbeantwortete Frage nicht in der History stehen lassen
+            if self.conversation_history and self.conversation_history[-1].get("role") == "user":
+                self.conversation_history.pop()
             return f"Fehler: {e}"
 
     def _append_chat(self, text, tag=None):
```

## Patch 06 — gui settings atomar

**Was:** Beide schrieben `settings.json` nicht atomar (NEVER #6). Jetzt tempfile + `os.replace`, gleiches Muster wie `popup_audio.py`.

**Beleg:** py_compile ok.

<!-- PATCH 06_gui_settings_atomar -->
```diff
--- a/core/gui/popups/popup_whisper.py
+++ b/core/gui/popups/popup_whisper.py
@@ -17,6 +17,7 @@
 import json
 import logging
 import os
+import tempfile
 import time
 import tkinter as tk
 from tkinter import scrolledtext
@@ -424,8 +425,19 @@
                 },
             }
 
-            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
-                json.dump(settings, f, indent=2, ensure_ascii=False)
+            # Atomar schreiben: tmp + rename (NEVER #6) — sonst kann der Service
+            # einen halb geschriebenen Stand lesen
+            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(SETTINGS_PATH), suffix=".tmp")
+            try:
+                with os.fdopen(fd, "w", encoding="utf-8") as f:
+                    json.dump(settings, f, indent=2, ensure_ascii=False)
+                os.replace(tmp_path, SETTINGS_PATH)
+            except Exception:
+                try:
+                    os.unlink(tmp_path)
+                except OSError:
+                    pass
+                raise
 
             self._lbl_save_status.config(text="Gespeichert!", fg=ACCENT_GREEN)
             self.win.after(2000, lambda: self._lbl_save_status.config(text=""))
--- a/core/gui/popups/popup_settings.py
+++ b/core/gui/popups/popup_settings.py
@@ -20,6 +20,7 @@
 import logging
 import os
 import shutil
+import tempfile
 import time
 import tkinter as tk
 from tkinter import messagebox
@@ -286,9 +287,19 @@
             return
 
         try:
-            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
-                json.dump(DEFAULTS, f, indent=2, ensure_ascii=False)
-                f.write("\n")
+            # Atomar schreiben: tmp + rename (NEVER #6)
+            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(SETTINGS_PATH), suffix=".tmp")
+            try:
+                with os.fdopen(fd, "w", encoding="utf-8") as f:
+                    json.dump(DEFAULTS, f, indent=2, ensure_ascii=False)
+                    f.write("\n")
+                os.replace(tmp_path, SETTINGS_PATH)
+            except OSError:
+                try:
+                    os.unlink(tmp_path)
+                except OSError:
+                    pass
+                raise
             self._load_and_display()
             self._show_feedback("Defaults wiederhergestellt.", color=ACCENT_GREEN)
             logger.info("Settings auf Defaults zurueckgesetzt")
```

## Patch 07 — voice tts timeouts

**Was:** 6 `subprocess.run` ohne Timeout (NEVER #5). Synthese/sox: 30 s. Wiedergabe (pw-cat, mpv): Audiodauer + 10 s — schneidet nichts ab, aber ein haengendes pw-cat blockiert den Sprech-Thread nicht mehr ewig.

**Beleg:** 25 s Sprache → Timeout 35 s. Alle Timeouts landen in bestehenden `except Exception`-Zweigen.

<!-- PATCH 07_voice_tts_timeouts -->
```diff
--- a/core/tts.py
+++ b/core/tts.py
@@ -51,6 +51,16 @@
 RESPEAKER_PORT = 12347        # UDP_SPEAKER_PORT in config.h
 RESPEAKER_ENABLED = True      # False = nur HDMI
 
+# Subprocess-Timeouts (NEVER #5): Synthese fest, Wiedergabe = Audiodauer + Reserve
+SYNTH_TIMEOUT_S = 30
+PLAY_TIMEOUT_RESERVE_S = 10.0
+
+
+def _play_timeout(raw_audio: bytes, sample_rate: int) -> float:
+    """Wiedergabe-Timeout: Dauer des PCM (s16 mono) + Reserve — schneidet nichts ab,
+    verhindert aber, dass ein haengendes pw-cat/mpv den Sprech-Thread ewig blockiert."""
+    return len(raw_audio) / (2.0 * max(int(sample_rate), 1)) + PLAY_TIMEOUT_RESERVE_S
+
 
 class VoiceModel:
     """Represents a Piper voice model."""
@@ -158,7 +168,7 @@
         ]
         result = subprocess.run(
             cmd, input=text.encode('utf-8'),
-            capture_output=True, check=True)
+            capture_output=True, check=True, timeout=SYNTH_TIMEOUT_S)
         return result.stdout
 
     def _build_pw_cmd(self, sample_rate: int) -> list:
@@ -215,7 +225,8 @@
     def _play_raw(self, raw_audio: bytes, sample_rate: int):
         """Play raw PCM via HDMI (pw-cat) + parallel an ReSpeaker senden."""
         self._send_to_respeaker(raw_audio)
-        subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True)
+        subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True,
+                       timeout=_play_timeout(raw_audio, sample_rate))
 
     def set_speed(self, speed: float):
         """TTS Geschwindigkeit aendern. 0.8=schnell, 1.0=normal, 1.2=langsam."""
@@ -329,7 +340,8 @@
 
             # HDMI via pw-cat (blockierend bis fertig)
             subprocess.run(self._build_pw_cmd(sample_rate),
-                           input=raw_audio, stderr=subprocess.DEVNULL)
+                           input=raw_audio, stderr=subprocess.DEVNULL,
+                           timeout=_play_timeout(raw_audio, sample_rate))
 
             logger.info("[TTS] Ausgabe fertig")
             return True
@@ -379,7 +391,8 @@
                     str(pitched_file),
                     "pitch", str(PITCH_SHIFT)
                 ]
-                subprocess.run(sox_cmd, check=True, capture_output=True)
+                subprocess.run(sox_cmd, check=True, capture_output=True,
+                               timeout=SYNTH_TIMEOUT_S)
                 play_file = pitched_file
             else:
                 play_file = wav_file
@@ -392,13 +405,15 @@
                 "--really-quiet",
                 str(play_file)
             ]
-            subprocess.run(mpv_cmd, check=True)
+            subprocess.run(mpv_cmd, check=True,
+                           timeout=_play_timeout(raw_audio, sample_rate))
 
         except subprocess.CalledProcessError as e:
             logger.error(f"Audio playback failed: {e}")
             # Fallback zu pw-cat wenn mpv/sox fehlschlaegt
             try:
-                subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True)
+                subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True,
+                               timeout=_play_timeout(raw_audio, sample_rate))
             except Exception:
                 raise
         finally:
```

## Patch 08 — service unit kein env dump

**Was:** `ExecStartPre` schrieb Umgebungsvariablen (u. a. Zugangsdaten) bei jedem Start nach `/tmp/moloch_env`. Kein Code liest die Datei.

**Beleg:** Grep: einzige Referenz ist die Unit selbst.

<!-- PATCH 08_service_unit_kein_env_dump -->
```diff
--- a/moloch.service
+++ b/moloch.service
@@ -8,7 +8,6 @@
 User=molochzuhause
 WorkingDirectory=/home/molochzuhause/moloch
 Environment=HOME=/home/molochzuhause
-ExecStartPre=/bin/bash -c 'source /home/molochzuhause/.profile && env | grep -E "^EWELINK_|^MOLOCH_" > /tmp/moloch_env'
 ExecStart=/bin/bash -c 'set -a; source /home/molochzuhause/.profile; set +a; exec /usr/bin/python3 -c "from core.moloch_service import MolochService; s = MolochService(); s.init(); s.start(blocking=True)"'
 Restart=always
 RestartSec=10
```

## Aufraeumen (auf dem Pi per `git rm`, jeweils ohne Referenz im Repo)

| Datei | Grund |
|---|---|
| `simple_ptz_fix.sh` | Einmal-Migration: loescht Zeilen 579–582 im Panel per Zeilennummer und startet den Service neu. Erneut ausgefuehrt = kaputtes Panel. |
| `core/audio/audio_manager.py` | Nirgends importiert (Bluetooth-SmartMic-Altlast), enthaelt `shell=True` mit eingesetztem Text. |
| `core/gui/eye_control_panel.py` | Nirgends importiert. |
| `scripts/face_capture.py` | Importiert `core.vision.xiao_vision`, das es nicht gibt — laeuft nicht. |

```bash
git rm simple_ptz_fix.sh core/audio/audio_manager.py core/gui/eye_control_panel.py scripts/face_capture.py
git commit -m "chore: tote/gefaehrliche Altlasten entfernt (Review 2026-11-17)"
```

## Neue Befunde beim Patchen

1. **Repo ist kein vollstaendiges Backup.** `core/vision/__init__.py` importiert `unified_pipeline` und `hybrid_vision` — beide Dateien waren nie im Repo. `core/inference_engine.py` importiert `core.vision.*` auf Modulebene, `moloch_service.py` haengt daran. Ein frischer Clone kann den Service nicht starten. Auf dem Pi pruefen: `git status --short | grep '^??'` und benoetigte Code-Dateien committen.
2. **Alter Enroll-Pfad kaputt.** `moloch_unified_panel.py` (Snapshot-Enroll), `scripts/enroll_arcface.py` und `scripts/calibrate_tracking.py` importieren `letterbox_resize`/`_unletterbox_scrfd` aus `core.inference_engine` — dort gibt es sie nicht mehr (jetzt `core/perception/face_pipeline.py`). Laut NEVER #3 laeuft Enrollment ueber den GStreamer-Pfad (`scripts/tappas_enroll.py`). Vorschlag: Legacy-Pfade stilllegen statt reparieren — Entscheidung Markus.

## Offen — braucht eine Entscheidung

| Thema | Stand | Vorschlag |
|---|---|---|
| Beat-Daten in `/api/state/current` | Commit `da5489f` nur halb committet: `chat_server.py:2936` importiert weiter das geloeschte `core.music.music_listener`; `get_last_beat()` existiert nicht; `chat_server` ist ein eigener Prozess. `music_bpm`/`music_beat_phase` sind immer 0. | `music_listener` schreibt BPM + letzten Beat zusaetzlich nach `/dev/shm/moloch_music_beat.json`, `chat_server` liest das. Betrifft voice + bridge. |
| Doppelte Musik-Analyse | Service startet `MusicVisualizer` **und** `MusicListener`; beide FFT + Beat, beide schreiben `/dev/shm/moloch_music.bin`. | Einen abschalten (vermutlich Visualizer). |
| Pflicht-Audit | CLAUDE.md: `~/moloch/moloch_audit.py --auto` „39 Tests“. Die Datei prueft 5 Dinge. Der grosse Audit (~90 Checks) ist `scripts/moloch_audit.py`. | CLAUDE.md und Hooks auf das gewollte Skript ausrichten. |
| Nicht erreichbare Module | ~80 Module in `core/` (~22.000 Zeilen) sind ueber Imports nicht erreichbar; einige laufen evtl. per Cron. | Crontab ins Repo (`docs/`), dann aufraeumen. |
| Namenskollision | `core/tts.py` neben Paket `core/tts/`; `import core.tts` laedt immer das Paket, deshalb der importlib-Workaround in `personality_engine.py`. | Datei umbenennen (z. B. `core/tts_engine.py`). |

## Pruef-Methode

Statische Analyse ueber alle 404 Python-Dateien (ruff F/E9/PLE/B/S/ASYNC, mypy call-arg/name-defined, eigene AST-Checks: subprocess ohne timeout, nicht-atomare JSON-Writes, Imports nicht existierender Namen, Import-Graph ab den systemd-Einstiegspunkten). Gezieltes Lesen der Kernpfade. Kein Laufzeittest (kein Pi).
