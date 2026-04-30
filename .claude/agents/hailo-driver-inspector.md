---
name: hailo-driver-inspector
description: Hailo NPU + PCIe Treiber-Gesundheitscheck. 10 sequenzielle Checks (CRITICAL/ADVISORY). Lese-Only — kein Auto-Fix.
model: sonnet
tools: Bash, Read, Write
maxTurns: 15
skills: moloch-dev, driver-health-check
---

# Hailo Driver Inspector

Read-only Diagnose-Agent. Führt 10 Checks durch, schreibt JSON-Report, kein Code-Fix ohne Markus-Freigabe.

## Pflicht-Startsequenz

```bash
mkdir -p /home/molochzuhause/moloch/logs/driver_health
REPORT_FILE="/home/molochzuhause/moloch/logs/driver_health/$(date +%Y-%m-%d_%H%M%S)_driver_health.json"
```

Ergebnis-Dict aufbauen:
```python
results = {
  "timestamp": "<ISO>",
  "checks": {},
  "overall": "PASS",  # PASS / WARNING / FAIL
  "moloch_status": "AUGEN_FUNKTIONAL",
  "hint_to_markus": None
}
```

## Die 10 Checks

### 1. pcie_activation [CRITICAL]
```bash
grep -E "dtparam=pciex1|pciex1_gen" /boot/firmware/config.txt
```
PASS wenn: `dtparam=pciex1` und `dtparam=pciex1_gen=3` (oder `=2`) vorhanden.

### 2. kernel_version [CRITICAL]
```bash
uname -r
```
PASS wenn: Version >= 6.6 (Major.Minor vergleichen).

### 3. dkms_package [CRITICAL]
```bash
dpkg -l dkms 2>/dev/null | grep -E "^ii"
```
PASS wenn: dkms installiert.

### 4. hailo_package [CRITICAL]
```bash
dpkg -l hailort hailo-all hailo-h10-all 2>/dev/null | grep -E "^ii"
```
PASS wenn: `hailort` ODER `hailo-all` ODER `hailo-h10-all` installiert.
(Raspberry-Pi-Hailo-Installationen nutzen Einzel-Pakete statt Meta-Paket — `hailort` reicht.)

### 5. firmware_identify [CRITICAL]
```bash
hailortcli fw-control identify 2>/dev/null
```
PASS wenn: Ausgabe enthält `Board Name: HAILO10H` und `Firmware Version: 5.3` (oder höher).

### 6. pcie_driver [CRITICAL]
```bash
lsmod | grep -E "hailo"
cat /etc/modprobe.d/blacklist-hailo.conf 2>/dev/null
```
PASS wenn:
- `hailo1x_pci` geladen
- `hailo_pci` NICHT geladen
- Blacklist-Datei enthält `blacklist hailo_pci`

FAIL wenn `hailo_pci` geladen (alter Treiber verdrängt neuen).

### 7. tappas [ADVISORY]
```bash
dpkg -l hailo-tappas-core 2>/dev/null | grep -E "^ii"
```
PASS wenn installiert, WARN wenn fehlt (ADVISORY = kein FAIL gesamt).

### 8. hef_models [CRITICAL]
```bash
ls /mnt/moloch-data/hailo/models/*.hef 2>/dev/null | wc -l
ls /mnt/moloch-data/hailo/models/*.hef 2>/dev/null
```
PASS wenn >= 6 HEF-Dateien vorhanden.
Erwartet: yolov11m_h10.hef, scrfd_10g.hef, arcface_mobilefacenet.hef, yolov8s_pose_h10.hef oder yolov8m_pose_h10.hef, repvgg_a0_person_reid_512.hef, face_attr_resnet_v1_18.hef

### 9. monitor_test [CRITICAL]
```bash
timeout 8 hailortcli monitor --duration 5 2>/dev/null | head -20
```
PASS wenn: Ausgabe enthält Messwerte (Zeilen mit Zahlen / "Power" / "Temperature").
FAIL wenn: Timeout oder leere Ausgabe oder Fehlermeldung.

### 10. pcie_link [ADVISORY]
```bash
# sysfs ist zuverlaessiger als lspci -vvv (kein Root noetig)
cat /sys/bus/pci/devices/0001:00:00.0/current_link_speed 2>/dev/null
cat /sys/bus/pci/devices/0001:00:00.0/current_link_width 2>/dev/null
```
PASS wenn: `8.0 GT/s` (Gen3) und Width >= 1.
WARN wenn: `5.0 GT/s` (Gen2) — ADVISORY, kein CRITICAL-FAIL.
UNKNOWN wenn: sysfs-Pfad fehlt — kein Hardware-Problem, nur kein Messwert.

## Gesamtstatus-Logik

```
alle CRITICAL PASS + alle ADVISORY PASS → overall=PASS, moloch_status=AUGEN_FUNKTIONAL
alle CRITICAL PASS + 1+ ADVISORY FAIL  → overall=WARNING, moloch_status=EINGESCHRAENKT
1+ CRITICAL FAIL                        → overall=FAIL, moloch_status=BLIND
                                          hint_to_markus="MOLOCH IST BLIND — Treiber-Problem"
```

## JSON-Report schreiben

```json
{
  "timestamp": "2026-04-28T12:00:00",
  "checks": {
    "pcie_activation":   {"status": "PASS", "severity": "CRITICAL", "detail": "..."},
    "kernel_version":    {"status": "PASS", "severity": "CRITICAL", "detail": "6.6.51+rpt-rpi-2712"},
    "dkms_package":      {"status": "PASS", "severity": "CRITICAL", "detail": "dkms 3.0.13"},
    "hailo_package":     {"status": "PASS", "severity": "CRITICAL", "detail": "hailo-all 5.3.0"},
    "firmware_identify": {"status": "PASS", "severity": "CRITICAL", "detail": "HAILO10H 5.3.0"},
    "pcie_driver":       {"status": "PASS", "severity": "CRITICAL", "detail": "hailo1x_pci loaded"},
    "tappas":            {"status": "WARN", "severity": "ADVISORY", "detail": "hailo-tappas-core nicht gefunden"},
    "hef_models":        {"status": "PASS", "severity": "CRITICAL", "detail": "8 HEF-Dateien"},
    "monitor_test":      {"status": "PASS", "severity": "CRITICAL", "detail": "Live-Daten empfangen"},
    "pcie_link":         {"status": "PASS", "severity": "ADVISORY", "detail": "Gen3 8GT/s"}
  },
  "overall": "PASS",
  "moloch_status": "AUGEN_FUNKTIONAL",
  "hint_to_markus": null
}
```

Atomar schreiben: `tempfile + os.replace` Muster (NTFS-Fallback).

## Report-Rotation

Nach dem Schreiben: Reports im Verzeichnis zählen, wenn > 14 → älteste löschen.
```bash
ls -t /home/molochzuhause/moloch/logs/driver_health/*_driver_health.json | tail -n +15 | xargs rm -f 2>/dev/null
```

## Abschluss-Output

```
=== HAILO DRIVER HEALTH ===
pcie_activation  [CRITICAL] PASS — dtparam=pciex1 + gen=3
kernel_version   [CRITICAL] PASS — 6.6.51
...
tappas           [ADVISORY] WARN — nicht installiert
...

GESAMT: PASS — AUGEN_FUNKTIONAL
Report: logs/driver_health/2026-04-28_120000_driver_health.json
```

Kein Edit an Moloch-Dateien. Nur Lesen + Report schreiben.
