# Plan: Hailo-Treiber-Audit + Postprocess-SO-Realignment

**Anlass:** Markus' Verdacht (27.04 17:50): doppelte Treiber/SOs erzeugen
verschiedene Landmarks. DeepSeek-Diagnose-Hinweis. Plus heutiger
Halluzinations-Fix (`ef09a24`) hat Identity-Block fix gemacht, aber
moegliche Pose-/Face-Landmark-Drift-Bugs sitzen tiefer im Treiber-Stack.

**Ausfuehrung:** Frische Claude-Code-Session (1M-Context Opus) zieht das
durch. **Pi-Side-Code-Aenderungen** moeglich. **Lokomotive-Workflow.**

---

## Pre-Flight (PFLICHT vor Phase A)

```bash
# 1. Pflicht-Startprotokoll
moloch_session_init  # via MCP

# 2. Backup-Tag
cd ~/moloch
git tag pre_hailo_cleanup_$(date +%Y%m%d_%H%M)
git push --tags

# 3. Stand vor Plan-Start verifizieren
python3 ~/moloch/moloch_audit.py --auto   # MUSS PASS sein
git log --oneline -5
git status                                # clean except runtime-state

# 4. Aktuelle Befunde (Diagnose 27.04 17:48 — sind unten in Section "Was bekannt ist")
```

**STOP wenn:** Audit FAIL ODER uncommitted ROT-Files ODER FPS<10.

---

## Was bekannt ist (Diagnose 27.04 17:48)

### Installierte Pakete
```
hailort                   5.3.0
hailort-pcie-driver       5.3.0   (DKMS hailo1x_pci/5.3.0 active)
hailo-tappas-core         5.3.0
hailo-models              1.0.0-2
hailo-gen-ai-model-zoo    5.3.0
rpicam-apps-hailo-postprocess 1.11.0-1
python3-hailo-tappas      5.1.0   ← MAJOR-VERSION-MISMATCH
```

### Was der laufende `moloch.service` (PID 434031) tatsaechlich geladen hat
```
/usr/lib/libhailort.so.5.3.0                                            # OK 5.3
/usr/lib/aarch64-linux-gnu/libgsthailometa.so.5.3.0                     # OK 5.3
/usr/lib/aarch64-linux-gnu/libgsthailotools.so.5.3.0                    # OK 5.3
/usr/lib/aarch64-linux-gnu/libhailo_opencv_utils.so.5.3.0               # OK 5.3
/usr/lib/aarch64-linux-gnu/libhailo_tracker.so.5.3.0                    # OK 5.3
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo.so                 # OK
/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so   # OK
/usr/lib/python3/dist-packages/hailo.cpython-313-aarch64-linux-gnu.so   # 5.1 Python-Binding!
/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so          # CUSTOM 5.1-era?
/usr/local/lib/python3.13/dist-packages/hailo_platform/pyhailort/_pyhailort.cpython-313-aarch64-linux-gnu.so   # 5.3
```

### `tappas_pipeline.py` zeigt EXPLIZIT auf custom-SOs in `/usr/local/hailo/`
Zeilen 66-89:
```python
YOLO_POSTPROCESS_SO    = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
SCRFD_POSTPROCESS_SO   = "/usr/local/hailo/resources/so/libscrfd.so"
SCRFD_CONFIG_JSON      = "/usr/local/hailo/resources/json/scrfd.json"
ARCFACE_POSTPROCESS_SO = "/usr/local/hailo/resources/so/libface_recognition_post.so"
FACE_ALIGN_SO          = "/usr/local/hailo/resources/so/libvms_face_align.so"
FACE_CROP_SO           = "/usr/local/hailo/resources/so/libvms_croppers.so"
POSE_POSTPROCESS_SO    = "/usr/local/hailo/resources/so/libyolov8pose_postprocess.so"
REID_POSTPROCESS_SO    = "/usr/local/hailo/resources/so/librepvgg_reid_postprocess.so"
WHOLE_BUFFER_SO        = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"
REID_CROP_SO           = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libre_id.so"
```

### `/usr/local/hailo/resources/so/` Inhalt
- Owner: `molochzuhause:molochzuhause` (NICHT root, also Markus-Eigenbuild)
- Modified: `Feb 4 16:06` (Mehrheit) und `Mar 26 14:57` (Verzeichnis-mtime)
- Quelle vermutlich: `~/ssd2_backup/hailo/repos/hailo-apps/` (build.release/cpp/)

### Doppelte Postprocess-SOs (custom vs. TAPPAS-Standard)
| Modell | TAPPAS-Standard (`/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/`) | Custom (`/usr/local/hailo/resources/so/`) |
|---|---|---|
| SCRFD | `libscrfd_post.so` | `libscrfd.so` |
| YOLOv8 Pose | `libyolov8pose_post.so` | `libyolov8pose_postprocess.so` |
| YOLO HailoRT++ | (nicht vorhanden) | `libyolo_hailortpp_postprocess.so` |
| ArcFace | (nicht direkt) | `libface_recognition_post.so` |
| ReID | (nicht direkt) | `librepvgg_reid_postprocess.so` |
| Face-Align/Crop | (nicht direkt) | `libvms_face_align.so`, `libvms_croppers.so` |

→ Custom-SOs sind notwendig (TAPPAS-Standard hat 1:1-Variante nur fuer SCRFD und Pose).
   Aber Pose+SCRFD existieren BEIDES → es gibt eine echte Wahl welche genutzt wird.

### Driver-Source-Trees (kosmetisch, nicht aktiv)
```
/usr/src/hailo1x_pci-5.3.0/      ← DKMS aktiv, 5.3.0
/usr/src/hailort-pcie-driver/    ← orphan, ohne Version, harmlos
```

---

## Phase A — Audit / Linkage / Build-Origin (READ-ONLY, kein Risiko)

**Domain:** vision (TAPPAS-Pipeline).
**Lock:** `touch /tmp/moloch_agent_vision`

**A1. Linkage-Check der custom-SOs gegen TAPPAS-5.3-Libs**
```bash
for so in /usr/local/hailo/resources/so/lib*.so; do
  echo "=== $so ==="
  ldd "$so" 2>&1 | grep -iE "hailo|tappas|opencv" | head -8
done > /tmp/hailo_linkage.txt
cat /tmp/hailo_linkage.txt
```
**Erwartet:** alle custom-SOs linken gegen `/usr/lib/libhailort.so.5` und
`/usr/lib/aarch64-linux-gnu/libhailo_*.so.5.3.0`. Wenn EINE SO gegen
`libhailort.so.5.1` linkt → klare ABI-Inkompatibilitaet.

**A2. Symbol-Versionen pruefen**
```bash
for so in /usr/local/hailo/resources/so/lib*.so; do
  echo "=== $so ==="
  nm -D "$so" 2>/dev/null | grep -E "Hailo|tappas" | head -5
done
```

**A3. Build-Origin verifizieren**
```bash
ls -la ~/ssd2_backup/hailo/repos/hailo-apps/postprocess/build.release/cpp/ 2>&1 | head -20
# falls vorhanden: cat ~/ssd2_backup/hailo/repos/hailo-apps/CMakeLists.txt | head -20
# Welche TAPPAS-Headers wurden beim Build verwendet?
grep -rn "find_package\|tappas/" ~/ssd2_backup/hailo/repos/hailo-apps/ 2>/dev/null | head -10
```

**A4. tappas_pipeline.py Pose+SCRFD: was passiert wenn TAPPAS-Standard genutzt?**
```bash
ls -la /usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so
ls -la /usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libscrfd_post.so
# Build-Datum + size — sind die deutlich anders als die custom-Variante?
ls -la /usr/local/hailo/resources/so/libyolov8pose_postprocess.so
ls -la /usr/local/hailo/resources/so/libscrfd.so
```

**A5. Python-Bindings-Mismatch verifizieren**
```bash
dpkg -L python3-hailo-tappas | grep -E "\.dist-info|\.so" | head
python3 -c "import hailo; print('hailo module:', hailo.__file__ if hasattr(hailo,'__file__') else '?')"
apt-cache policy python3-hailo-tappas
apt-cache madison python3-hailo-tappas hailo-tappas-core
# 5.3-Bindings im Repo verfuegbar?
```

**Akzeptanz Phase A:** Output von A1-A5 als Markdown in
`/tmp/hailo_audit_YYYYMMDD.md` ablegen. Keine Aenderung am System.
Wenn ALLE SOs gegen 5.3 linken UND keine Mismatch-Symbole → kein
Treiber-Konflikt aktiv, Halluzinationen anders verursacht (Pi-Side
Identity-Fix `ef09a24` greift bereits, Markus testet weiter).

`rm /tmp/moloch_agent_vision`

---

## Phase B — Python-Bindings 5.1 → 5.3 Upgrade

**Domain:** vision (touched packages, nicht direkt unsere `core/`).
**Lock:** `touch /tmp/moloch_agent_vision`

**WARNUNG:** Pi-Reboot nach Phase B PFLICHT (siehe CLAUDE.md ⛔ PFLICHT-SCHRITT 0c).

**B1. Backup vor jeder Aenderung**
```bash
cd ~/moloch
git tag pre_python_tappas_upgrade_$(date +%Y%m%d_%H%M)
sudo cp -r /usr/lib/python3/dist-packages/gsthailo /tmp/gsthailo_backup_5.1.0
sudo cp /usr/lib/python3/dist-packages/hailo.cpython-313-aarch64-linux-gnu.so /tmp/hailo_so_backup_5.1.0
```

**B2. Verfuegbarkeit pruefen**
```bash
sudo apt update
apt list -a python3-hailo-tappas 2>&1 | head -10
# Wenn 5.3.0 verfuegbar:
sudo apt install python3-hailo-tappas=5.3.0
# Wenn nur 5.1.0: prufen ob hailo-Repo aktiviert ist
cat /etc/apt/sources.list.d/*hailo* 2>&1 | head -10
```

**B3. Falls kein 5.3-Paket in apt:** Hailo-Developer-Zone-Direktdownload
(dafuer NICHT autonom — Markus rufen, weil Login + Lizenz-Akzeptanz noetig).
**Status `wontfix` setzen + Markus-Hint posten, weiter mit Phase C.**

**B4. Falls upgrade durch:**
```bash
# __pycache__ clearen
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
sudo find /usr/lib/python3/dist-packages/gsthailo -name "__pycache__" -exec rm -rf {} + 2>/dev/null
# Pi-Reboot PFLICHT (TAPPAS-Python-Bindings sind C-Extension, brauchen frischen Python)
sudo systemctl stop moloch
sudo reboot
# 60s warten
```

**B5. Post-Reboot-Verify**
```bash
sleep 60
systemctl is-active moloch
python3 ~/moloch/moloch_audit.py --auto   # MUSS PASS sein
python3 -c "import hailo_platform; print(hailo_platform.__version__)"  # 5.3.0
dpkg -l python3-hailo-tappas | tail -1                                  # 5.3.0
```

**Akzeptanz Phase B:** python3-hailo-tappas auf 5.3.0 ODER (`wontfix`+
Markus-Note bei nur-5.1-Paket). Audit 85/85 PASS. FPS>=20 in `moloch_status`.
Bei FAIL: rollback via `git checkout` der `pre_python_tappas_upgrade`-Tag-Stand
+ `sudo apt install python3-hailo-tappas=5.1.0` + reboot.

`rm /tmp/moloch_agent_vision`

---

## Phase C — Custom-SOs gegen TAPPAS-5.3-Headers neu bauen

**Domain:** vision.
**Lock:** `touch /tmp/moloch_agent_vision`
**Voraussetzung:** Phase A abgeschlossen UND Phase A1 hat keine ABI-Mismatches
gezeigt → Phase C ist optional. **Phase A1 zeigt ABI-Mismatch** → Phase C
ist NOTWENDIG.

**C1. Source-Inventur**
```bash
ls -la ~/ssd2_backup/hailo/repos/hailo-apps/postprocess/cpp/
ls -la ~/ssd2_backup/hailo/repos/hailo-apps/cpp/
ls -la ~/ssd2_backup/hailo/repos/hailo-apps/postprocess/build.release/cpp/
# CMakeLists oder meson.build vorhanden?
find ~/ssd2_backup/hailo/repos/hailo-apps -name "CMakeLists.txt" -o -name "meson.build" | head -5
```

**C2. TAPPAS-5.3 Headers vorhanden?**
```bash
ls -la /usr/include/hailo/tappas/ 2>&1 | head -10
# Und Pose-Header?
ls -la /usr/include/hailo/tappas/pose_estimation/yolov8pose_postprocess.hpp 2>&1
```

**C3. Re-Build (wenn C1+C2 ok)**
```bash
cd ~/ssd2_backup/hailo/repos/hailo-apps/postprocess/
# Build-System detection: meson oder cmake
[ -f meson.build ] && meson setup build.5.3 --buildtype=release && meson compile -C build.5.3
[ -f CMakeLists.txt ] && cmake -S . -B build.5.3 -DCMAKE_BUILD_TYPE=Release && cmake --build build.5.3 -j4
ls -la build.5.3/cpp/*.so 2>&1 | head -10
```

**C4. Atomic-Replace mit Backup**
```bash
sudo cp -r /usr/local/hailo/resources/so /tmp/hailo_so_backup_$(date +%s)
for so in build.5.3/cpp/lib*.so; do
  base=$(basename "$so")
  if [ -f "/usr/local/hailo/resources/so/$base" ]; then
    sudo cp "$so" "/usr/local/hailo/resources/so/$base"
    echo "replaced: $base"
  fi
done
```

**C5. Pipeline-Test**
```bash
find ~/moloch/core -name "__pycache__" -exec rm -rf {} + 2>/dev/null
sudo systemctl restart moloch
sleep 20
python3 ~/moloch/moloch_audit.py --auto   # MUSS PASS sein
# Live-Test: Pose-Keypoints + Face-Landmarks visuell pruefen
moloch_snapshot()  # via MCP
```

**Akzeptanz Phase C:** Custom-SOs gegen TAPPAS-5.3 gebaut, Audit PASS,
SHM-FPS>=18, keine SEGV in 5min. Markus visual-check der Snapshot:
Face-Landmarks sitzen wo sie sein sollen (Augen, Nase, Mund), Pose-Keypoints
am Koerper.

**Bei FAIL:**
```bash
sudo cp /tmp/hailo_so_backup_*/* /usr/local/hailo/resources/so/
sudo systemctl restart moloch
# Audit muss wieder gruen sein
git checkout pre_hailo_cleanup_*
```

`rm /tmp/moloch_agent_vision`

---

## Phase D — Cosmetic Cleanup (optional)

**D1. Orphan driver-source-tree entfernen**
```bash
sudo rm -rf /usr/src/hailort-pcie-driver/
# DKMS bleibt unberuehrt (nutzt /usr/src/hailo1x_pci-5.3.0/)
sudo dkms status
```

**D2. Falls hailo-apps-Source-Tree auf SSD2 raus soll**
- NICHT entfernen wenn Phase C erfolgreich war (zukuenftiger Re-Build moeglich)
- Nur wenn Markus-OK + Backup vorhanden

---

## Wenn alles fehlschlaegt — Komplett-Reinstall

**NUR wenn Markus explizit OK gibt** (Pi-Reboot, Pipeline-Restart, Risiko Hailo-FW):
```bash
cd ~/moloch
git tag pre_full_hailo_reinstall_$(date +%Y%m%d)
git push --tags

# Stop everything
sudo systemctl stop moloch moloch-chat moloch-chat-https moloch-cross-monitor

# Purge alles
sudo apt purge --autoremove \
  hailort hailort-pcie-driver hailo-tappas-core python3-hailo-tappas \
  hailo-models hailo-gen-ai-model-zoo rpicam-apps-hailo-postprocess

# Custom-SOs in Quarantaene
sudo mv /usr/local/hailo /usr/local/hailo.OLD.$(date +%s)

# Reinstall
sudo apt update
sudo apt install -y \
  hailort hailort-pcie-driver hailo-tappas-core python3-hailo-tappas \
  hailo-models hailo-gen-ai-model-zoo

# DKMS rebuild
sudo dkms autoinstall

# Reboot Pi
sudo reboot

# Post-Reboot: Phase A1+A2+A3 nochmal laufen lassen
# Custom-SOs aus Backup wiederherstellen ODER neu bauen (Phase C)
```

---

## Akzeptanz-Kriterien GESAMT

Nach Plan-Ende:
1. **Audit 85/85 PASS** (`python3 ~/moloch/moloch_audit.py --auto`)
2. **FPS stabil** >= 18 fuer >= 5min
3. **Keine SEGV** in dmesg / journalctl letzte 10min
4. **Python-Bindings + TAPPAS-Core auf gleicher Version** (5.3.0 ideal,
   beide auf 5.1.0 mit `wontfix` akzeptabel falls 5.3-Paket nicht verfuegbar)
5. **Live-Snapshot mit Markus**:
   - Face-Landmarks sitzen am Gesicht (nicht verschoben)
   - Pose-Keypoints sitzen am Koerper
   - Markus-Recognition-Similarity >= 0.45
6. **Markus visuell zufrieden** im Cockpit Sehen-Tab

---

## NEVER-Regeln (aus CLAUDE.md, hier hart)

- **NIE shell=True bei subprocess** — alle Befehle als arglist
- **NIE git config user.* aendern** — Author via env-vars `Cowork Pi-Side`
- **NIE force-push** — Konflikte via `git pull --rebase`
- **Pi-Reboot PFLICHT bei** TAPPAS-Plugin-Install, Hailo-FW-Update, oder
  wenn dieser Plan Phase B oder Komplett-Reinstall macht
- **`__pycache__` nach Code-Aenderung loeschen**
- **Backup-Tag VOR jeder Phase**: `git tag pre_<phasename>_$(date +%Y%m%d)`
- **Bei Audit FAIL → STOPP, rollback** (siehe Akzeptanz-Sektionen)

---

## Mailbox-Updates (am Ende vom Plan)

Nach erfolgreichem Plan-End:
1. Commit alle Aenderungen mit `Cowork Pi-Side` GIT_AUTHOR
2. `docs/PI_TO_PC.md` neuer Eintrag `hailo_treiber_audit_done` mit:
   - Welche Phasen durchgefuehrt
   - Linkage-Befund (Phase A1)
   - Final-Versions-Stand
   - Audit-Status
3. Diese Plan-Datei (`docs/plans/hailo_driver_audit_cleanup.md`) mit
   `## STATUS: DONE YYYY-MM-DD by <session-id>` ergaenzen
4. `logs/agent_handoff.md` Update

---

## Kontakt zum laufenden Stand (Stand 27.04 17:50)

- **Pi HEAD:** `ef09a24` (identity+hardware Halluzinations-Fix LIVE)
- **Pool:** 14 approved / 22 pending / 7 rejected — v_next_ready_to_train
  steht (15:25)
- **Federation:** `fed_kill` aktiv beidseitig (OAuth-Daemon-Pfad nicht
  praktikabel — Markus-Entscheidung 15:05)
- **Markus' Halluzinations-Test heute:** "P-Power Deck"-Frage
  beantwortet jetzt korrekt, **vermutlich Treiber-Frage parallele Achse**
  zur Identity-Fix-Achse — Plan hier ist die Treiber-Achse
