# Code-Review 2026-11-17 — offene Befunde mit fertigen Patches

**Basis:** `deepseek_architecture_overhaul` @ `ee0f3bb` (Pi-Stand). Erstellt in einer Cloud-Session ohne Pi und ohne MCP; der Session-Lock liess sich dort nicht aufheben, deshalb liegen die Fixes hier als Patches statt als Code-Commits.

Jeder Patch ist geprueft: `git apply` auf eine Kopie des Pi-Stands, `py_compile`, ruff (F821/F823/E9) und — wo moeglich — ein isolierter Vorher/Nachher-Test.

**Auf dem Pi-Stand schon gefixt und deshalb nicht enthalten:** lokales `import threading` in `_execute_panel_cmd` (`42fe147`), TTS-Timeouts (`23bc50f`), `ptz_goto` (`29bf84c`), fehlende `core/vision`-Module (bereinigt).

Sicherheits- und Datenschutz-Befunde stehen bewusst **nicht** hier, solange das Repo oeffentlich ist (Session-Chat 2026-11-17).

## Anwenden (auf dem Pi, normales Startprotokoll)

```bash
cd ~/moloch
# 1. Patches extrahieren -> /tmp/moloch_patch_*.diff
awk '/^<!-- PATCH /{name=$3} /^```diff$/{if(name){f="/tmp/moloch_patch_" name ".diff"; next}} /^```$/{f=""; name=""} f{print > f}' docs/CODE_REVIEW_2026-11-17.md
ls /tmp/moloch_patch_*.diff

# 2. Pro Patch: pruefen, anwenden, Syntax, Restart, Audit, einzeln committen
git apply --check /tmp/moloch_patch_01_service_save_settings_kein_rumpf.diff
git apply /tmp/moloch_patch_01_service_save_settings_kein_rumpf.diff
python3 -c "import core.moloch_service"
find ~/moloch/core -name __pycache__ -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch
python3 ~/moloch/moloch_audit.py --auto
git add core/moloch_service.py && git commit -m "fix(service): _save_settings ueberschreibt kaputte settings.json nicht mehr"
```

Reihenfolge: 04 setzt 03 voraus. Alle anderen sind unabhaengig.

Die awk-Extraktion ist byte-genau. Patches 02, 05 und 06 enthalten CRLF-Zeilenenden, weil die Zieldateien CRLF haben. Wer aus der GitHub-Weboberflaeche kopiert, verliert die CRs: dann `git apply --ignore-whitespace` (funktioniert, ergibt aber gemischte Zeilenenden).

## Uebersicht

| # | Datei | Ampel | Agent | Danach |
|---|---|---|---|---|
| 01 | `core/moloch_service.py` | ROT | service | Service-Restart |
| 02 | `core/camera_manager.py` | – | hardware | Service-Restart |
| 03 | `core/gui/moloch_unified_panel.py` | GELB | gui | Panel neu starten |
| 04 | `core/gui/moloch_unified_panel.py` | GELB | gui | Panel neu starten |
| 05 | `core/gui/popups/popup_whisper.py, popup_settings.py` | GELB | gui | Panel neu starten |
| 06 | `moloch.service` | – | service | Unit nach /etc kopieren + **Reboot** (CLAUDE.md 0c), danach `rm -f /tmp/moloch_env` |

## Patch 01 — service save settings kein rumpf

**Was:** `_save_settings`: Ist `settings.json` nicht lesbar, wird abgebrochen statt mit `{"version": 1}` weiterzuschreiben. Vorher gingen dabei alle Keys anderer Prozesse verloren (`tentacle_llm`, `llm_profile`, `hand_detection_enabled` ...).

**Beleg:** Kaputte Datei mit `tentacle_llm` + `llm_profile`: vorher ueberschrieben (nur noch Service-Keys), nachher unveraendert. Normalfall: fremde Keys bleiben erhalten.

<!-- PATCH 01_service_save_settings_kein_rumpf -->
```diff
diff --git a/core/moloch_service.py b/core/moloch_service.py
index 4321e89..4f55b7c 100644
--- a/core/moloch_service.py
+++ b/core/moloch_service.py
@@ -3500,13 +3500,19 @@ class MolochService:
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

## Patch 02 — hardware ptz calibrate

**Was:** `ptz_calibrate` baute `CameraCloudBridgeSync()` ohne Pflichtargument `config` → `TypeError`. Nutzt jetzt den bestehenden CloudController wie `cloud_set_night_mode()`. (`ptz_goto` ist seit `29bf84c` gefixt.)

**Beleg:** Vorher `TypeError`, nachher Aufruf ueber `self._cloud.run(...)`.

<!-- PATCH 02_hardware_ptz_calibrate -->
```diff
--- a/core/camera_manager.py
+++ b/core/camera_manager.py
@@ -960,11 +960,16 @@
         logger.info(f"[PTZ] goto {position} ({pan}, {tilt})")
 
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

## Patch 03 — gui lambda closure

**Was:** 6× `except Exception as e: root.after(0, lambda: …{e})` → `lambda err=e:`. Python loescht `e` am Ende des except-Blocks; die Lambda lief spaeter in einen `NameError`, die Fehlermeldung kam nie an.

**Beleg:** ruff F821: 6 → 0.

<!-- PATCH 03_gui_lambda_closure -->
```diff
diff --git a/core/gui/moloch_unified_panel.py b/core/gui/moloch_unified_panel.py
index f373d47..13514fd 100755
--- a/core/gui/moloch_unified_panel.py
+++ b/core/gui/moloch_unified_panel.py
@@ -905,8 +905,8 @@ class MolochUnifiedPanel:
 
             except Exception as e:
                 logger.error(f"Service init failed: {e}\n{traceback.format_exc()}")
-                self.root.after(0, lambda: self.status_label.config(
-                    text=f"Service FEHLER: {e}", fg="#ff4444"))
+                self.root.after(0, lambda err=e: self.status_label.config(
+                    text=f"Service FEHLER: {err}", fg="#ff4444"))
 
         threading.Thread(target=do_init, daemon=True, name="ServiceInit").start()
 
@@ -2094,8 +2094,8 @@ class MolochUnifiedPanel:
                     proc.send_signal(signal.SIGINT)
                     proc.wait(timeout=3)
                 except Exception as e:
-                    self.root.after(0, lambda: self._append_chat(
-                        f"[Mic Test] Aufnahme fehlgeschlagen: {e}", "system"))
+                    self.root.after(0, lambda err=e: self._append_chat(
+                        f"[Mic Test] Aufnahme fehlgeschlagen: {err}", "system"))
                     return
 
                 if not os.path.exists(test_path) or os.path.getsize(test_path) < 1000:
@@ -2134,8 +2134,8 @@ class MolochUnifiedPanel:
                             self.root.after(0, lambda: self._append_chat(
                                 "[Mic Test] Nichts erkannt", "system"))
                     except Exception as e:
-                        self.root.after(0, lambda: self._append_chat(
-                            f"[Mic Test] Whisper Fehler: {e}", "system"))
+                        self.root.after(0, lambda err=e: self._append_chat(
+                            f"[Mic Test] Whisper Fehler: {err}", "system"))
 
                     # Whisper VDevice freigeben (sonst blockiert naechste Spracheingabe!)
                     try:
@@ -2258,8 +2258,8 @@ class MolochUnifiedPanel:
                     boxes, landmarks, _lb_px, _lb_py, _lb_rw, _lb_rh)
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] SCRFD Fehler: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] SCRFD Fehler: {err}", "system"))
                 return
 
             # Groesstes Gesicht finden
@@ -2303,8 +2303,8 @@ class MolochUnifiedPanel:
                 embedding = normalize_arcface(embedding)
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] ArcFace Fehler: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] ArcFace Fehler: {err}", "system"))
                 return
 
             # Save to face_embeddings.json
@@ -2344,8 +2344,8 @@ class MolochUnifiedPanel:
                 logger.info(f"[SNAPSHOT] Markus embedding saved to {db_path}")
 
             except Exception as e:
-                self.root.after(0, lambda: self._append_chat(
-                    f"[Snapshot] Speichern fehlgeschlagen: {e}", "system"))
+                self.root.after(0, lambda err=e: self._append_chat(
+                    f"[Snapshot] Speichern fehlgeschlagen: {err}", "system"))
 
         threading.Thread(target=do_snapshot, daemon=True).start()
 
```

## Patch 04 — gui deepseek history limit

**Was:** `conversation_history` auf 10 Wortwechsel + aktuelle Frage begrenzt (vorher unbegrenzt, komplett bei jeder DeepSeek-Anfrage). Bei Fehler wird die unbeantwortete Frage wieder entfernt.

**Beleg:** py_compile ok.

<!-- PATCH 04_gui_deepseek_history_limit -->
```diff
diff --git a/core/gui/moloch_unified_panel.py b/core/gui/moloch_unified_panel.py
index 13514fd..804cc97 100755
--- a/core/gui/moloch_unified_panel.py
+++ b/core/gui/moloch_unified_panel.py
@@ -2425,6 +2425,10 @@ class MolochUnifiedPanel:
             self.conversation_history.append({
                 "role": "user", "content": message_content
             })
+            # History begrenzen (10 Wortwechsel + aktuelle Frage) — sonst waechst
+            # jede Anfrage unbegrenzt (Kosten, Kontextlimit, 15s-Timeout)
+            if len(self.conversation_history) > 21:
+                del self.conversation_history[:-21]
 
             import requests
             api_msgs = [{"role": "system", "content": self.system_prompt}] + self.conversation_history
@@ -2463,6 +2467,9 @@ class MolochUnifiedPanel:
 
         except Exception as e:
             logger.error(f"DeepSeek error: {e}")
+            # Unbeantwortete Frage nicht in der History stehen lassen
+            if self.conversation_history and self.conversation_history[-1].get("role") == "user":
+                self.conversation_history.pop()
             return f"Fehler: {e}"
 
     def _append_chat(self, text, tag=None):
```

## Patch 05 — gui settings atomar

**Was:** Beide schrieben `settings.json` nicht atomar (NEVER #6). Jetzt tempfile + `os.replace`, gleiches Muster wie `popup_audio.py`.

**Beleg:** py_compile ok.

<!-- PATCH 05_gui_settings_atomar -->
```diff
diff --git a/core/gui/popups/popup_settings.py b/core/gui/popups/popup_settings.py
index 40cc8a7..915042e 100644
--- a/core/gui/popups/popup_settings.py
+++ b/core/gui/popups/popup_settings.py
@@ -20,6 +20,7 @@ import json
 import logging
 import os
 import shutil
+import tempfile
 import time
 import tkinter as tk
 from tkinter import messagebox
@@ -286,9 +287,19 @@ class SettingsPopup(tk.Toplevel):
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
diff --git a/core/gui/popups/popup_whisper.py b/core/gui/popups/popup_whisper.py
index 5762eeb..07eba64 100644
--- a/core/gui/popups/popup_whisper.py
+++ b/core/gui/popups/popup_whisper.py
@@ -17,6 +17,7 @@ Importiert NUR panel_styles und tkinter.
 import json
 import logging
 import os
+import tempfile
 import time
 import tkinter as tk
 from tkinter import scrolledtext
@@ -424,8 +425,19 @@ class WhisperPopup:
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
```

## Patch 06 — service unit kein env dump

**Was:** `ExecStartPre` schrieb Umgebungsvariablen (u. a. Zugangsdaten) bei jedem Start nach `/tmp/moloch_env`. Kein Code liest die Datei.

**Beleg:** Grep: einzige Referenz ist die Unit selbst.

<!-- PATCH 06_service_unit_kein_env_dump -->
```diff
diff --git a/moloch.service b/moloch.service
index b055214..5ccb1fd 100644
--- a/moloch.service
+++ b/moloch.service
@@ -8,7 +8,6 @@ Type=simple
 User=molochzuhause
 WorkingDirectory=/home/molochzuhause/moloch
 Environment=HOME=/home/molochzuhause
-ExecStartPre=/bin/bash -c 'source /home/molochzuhause/.profile && env | grep -E "^EWELINK_|^MOLOCH_" > /tmp/moloch_env'
 ExecStart=/bin/bash -c 'set -a; source /home/molochzuhause/.profile; set +a; exec /usr/bin/python3 -c "from core.moloch_service import MolochService; s = MolochService(); s.init(); s.start(blocking=True)"'
 Restart=always
 RestartSec=10
```

## Aufraeumen (auf dem Pi per `git rm`, jeweils ohne Referenz im Code)

| Datei | Grund |
|---|---|
| `simple_ptz_fix.sh` | Einmal-Migration: loescht Zeilen 579–582 im Panel per Zeilennummer und startet den Service neu. Erneut ausgefuehrt = kaputtes Panel. |
| `core/audio/audio_manager.py` | Nirgends importiert (Bluetooth-SmartMic-Altlast), enthaelt `shell=True` mit eingesetztem Text. |
| `core/gui/eye_control_panel.py` | Nirgends importiert (nur ein Kommentar im Unified Panel). |
| `scripts/face_capture.py` | Importiert `core.vision.xiao_vision`, das es nicht gibt — laeuft nicht. |

```bash
git rm simple_ptz_fix.sh core/audio/audio_manager.py core/gui/eye_control_panel.py scripts/face_capture.py
git commit -m "chore: tote/gefaehrliche Altlasten entfernt (Review 2026-11-17)"
```

## Offen — braucht eine Entscheidung

| Thema | Befund | Vorschlag |
|---|---|---|
| Mailbox-Push | `_mailbox_commit_push` (chat_server) macht add/commit/push **ohne Pull**. Sobald jemand anderes auf `deepseek_architecture_overhaul` pusht (z. B. Merge von PR #7), scheitert jeder weitere Mailbox-Push vom Pi, bis dort gepullt wird. | Vor dem Push `git pull --rebase` (betrifft nur die zwei Mailbox-Dateien). Bis dahin: nach jedem Merge auf den Pi-Branch einmal am Pi pullen. |
| Alter Enroll-Pfad | `moloch_unified_panel.py` (Snapshot-Enroll), `scripts/enroll_arcface.py`, `scripts/calibrate_tracking.py` importieren `letterbox_resize`/`_unletterbox_scrfd` aus `core.inference_engine` — gibt es dort nicht mehr (jetzt `core/perception/face_pipeline.py`). | Laut NEVER #3 laeuft Enrollment ueber `scripts/tappas_enroll.py` → Legacy-Pfade stilllegen statt reparieren. |
| Beat-Daten in `/api/state/current` | `chat_server.py` importiert das geloeschte `core.music.music_listener`; `get_last_beat()` existiert nicht; `chat_server` ist ein eigener Prozess. `music_bpm`/`music_beat_phase` sind immer 0. | `core/audio/music_listener.py` schreibt BPM + letzten Beat zusaetzlich nach `/dev/shm/moloch_music_beat.json`, `chat_server` liest das. Betrifft voice + bridge. |
| Doppelte Musik-Analyse | Service startet `MusicVisualizer` **und** `MusicListener`; beide FFT + Beat, beide schreiben `/dev/shm/moloch_music.bin`. | Einen abschalten (vermutlich Visualizer). |
| Pflicht-Audit | CLAUDE.md nennt `~/moloch/moloch_audit.py --auto`. Die Datei prueft 5 Dinge (Service, RAM, Temp, Qdrant, Log). Der grosse Audit (~90 Checks) ist `scripts/moloch_audit.py`. | CLAUDE.md und Hooks auf das gewollte Skript ausrichten. |
| `agents/` im Root | Doppelt zu `.claude/agents/`, wird aber von dort referenziert („Lies IMMER zuerst agents/AGENT_*.md“). Stand Maerz/April. | Inhalt in `.claude/agents/` zusammenfuehren, dann `agents/` entfernen. |
| Namenskollision | `core/tts.py` neben Paket `core/tts/`; `import core.tts` laedt immer das Paket, deshalb der importlib-Workaround in `personality_engine.py`. | Datei umbenennen (z. B. `core/tts_engine.py`). |
| Nicht erreichbare Module | ~80 Module in `core/` (~22.000 Zeilen) sind ueber Imports nicht erreichbar; einige laufen evtl. per Cron. | Crontab ins Repo (`docs/`), dann aufraeumen. |

## Pruef-Methode

Statische Analyse ueber alle Python-Dateien (ruff F/E9/PLE/B/S/ASYNC, mypy call-arg/name-defined, eigene AST-Checks: subprocess ohne timeout, nicht-atomare JSON-Writes, Imports nicht existierender Namen, Import-Graph ab den systemd-Einstiegspunkten), danach jeder Befund gegen den Pi-Stand nachgeprueft. Kein Laufzeittest (kein Pi).
