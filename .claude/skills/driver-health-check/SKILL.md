---
name: driver-health-check
trigger: /check-drivers
description: Hailo NPU + PCIe Treiber-Gesundheitscheck (10 Checks, CRITICAL/ADVISORY)
---

# Skill: /check-drivers

Delegiert vollständig an den `hailo-driver-inspector` Subagenten.

## Aktivierung

```
/check-drivers
```

## Was passiert

Der Subagent `hailo-driver-inspector` führt 10 sequenzielle Checks durch:
- 6 CRITICAL-Checks (PCIe, Kernel, DKMS, Hailo-Package, Firmware, Treiber, HEF-Modelle, Monitor)
- 2 ADVISORY-Checks (TAPPAS, PCIe-Linkgeschwindigkeit)

Ausgabe: `logs/driver_health/YYYY-MM-DD_HHMMSS_driver_health.json`

Gesamtstatus:
- **PASS** → Alle CRITICAL grün → `moloch_status=AUGEN_FUNKTIONAL`
- **WARNING** → ADVISORY-Fehler → `moloch_status=EINGESCHRAENKT`  
- **FAIL** → CRITICAL-Fehler → `moloch_status=BLIND` + Hinweis an Markus

## Wichtig

- Agent ist **read-only** — kein automatischer Fix ohne Markus-Freigabe
- Jeder Check läuft unabhängig (Fail-Isolation)
- Letzte 14 Reports werden behalten, ältere automatisch gelöscht

## Agent laden

```
Subagent: .claude/agents/hailo-driver-inspector.md
```
