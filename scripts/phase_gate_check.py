#!/usr/bin/env python3
"""
phase_gate_check — Taeglicher Readiness-Check fuer Phase 4e + Phase 6.

Zaehlt Journal-Tage (YYYY-MM-DD.jsonl in journal/ + journal/archive/),
aktualisiert config/phase_gates.json und gibt deutliche Meldungen aus,
wenn eine Phase scharf wird.

Aufruf taeglich via systemd-Timer moloch-phase-gate.timer.
"""
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

JOURNAL_DIR = "/mnt/moloch-data/memory/journal"
JOURNAL_ARCHIVE = "/mnt/moloch-data/memory/journal/archive"
PHASE_GATES_PATH = "/home/molochzuhause/moloch/config/phase_gates.json"

# YYYY-MM-DD.jsonl, optional mit "scored_" Prefix oder anderen Praefixen wird ignoriert.
_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

DEFAULT_GATES = {
    "phase4e": {
        "armed": False,
        "armed_at": None,
        "days_collected": 0,
        "required_days": 7,
        "description": "Weekly compactor scharf, sobald 7 Journal-Tage vorhanden sind",
    },
    "phase6": {
        "armed": False,
        "armed_at": None,
        "operation_days": 0,
        "required_days": 14,
        "description": "Phase 6 scharf, sobald 14 Journal-Tage vorhanden sind",
    },
    "last_check": None,
}


def count_journal_days() -> int:
    """Zaehlt einzigartige YYYY-MM-DD.jsonl Dateien in journal/ + journal/archive/."""
    days = set()
    for base in (JOURNAL_DIR, JOURNAL_ARCHIVE):
        if not os.path.isdir(base):
            continue
        try:
            for entry in os.listdir(base):
                m = _DAY_FILE_RE.match(entry)
                if m:
                    days.add(m.group(1))
        except OSError as e:
            print(f"[GATE] WARN: kann {base} nicht lesen: {e}", file=sys.stderr)
    return len(days)


def load_gates() -> dict:
    """Liest phase_gates.json, erstellt mit Defaults wenn nicht vorhanden."""
    path = Path(PHASE_GATES_PATH)
    if not path.exists():
        # Defaults zurueckgeben (wird beim ersten save_gates angelegt)
        return json.loads(json.dumps(DEFAULT_GATES))  # deep copy
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[GATE] WARN: phase_gates.json kaputt ({e}) — verwende Defaults", file=sys.stderr)
        return json.loads(json.dumps(DEFAULT_GATES))

    # Merge mit Defaults (vorwaerts-kompatibel)
    for k, v in DEFAULT_GATES.items():
        if k not in data:
            data[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if kk not in data[k]:
                    data[k][kk] = vv
    return data


def save_gates(gates: dict) -> None:
    """Atomic write nach phase_gates.json (NEVER 6)."""
    path = Path(PHASE_GATES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(gates, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
    except OSError:
        # NTFS-Fallback
        with open(str(path), "w") as f:
            json.dump(gates, f, indent=2, ensure_ascii=False)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def check_phase4e(gates: dict, days: int) -> None:
    """Phase 4e: armed wenn days >= required_days (Default 7)."""
    section = gates["phase4e"]
    was_armed = bool(section.get("armed", False))
    section["days_collected"] = days
    required = int(section.get("required_days", 7))

    if days >= required and not was_armed:
        section["armed"] = True
        section["armed_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"[GATE] *** PHASE 4E SCHARF *** ({days} Tage gesammelt, Schwelle {required})")
        print("[GATE] Hinweis: 'sudo systemctl enable --now moloch-weekly-compactor.timer' falls noch nicht aktiv")
    elif was_armed:
        print(f"[GATE] Phase 4e: armed (seit {section.get('armed_at', '?')}, {days} Tage)")
    else:
        print(f"[GATE] Phase 4e: {days}/{required} Tage")


def check_phase6(gates: dict, days: int) -> None:
    """Phase 6: armed wenn days >= required_days (Default 14)."""
    section = gates["phase6"]
    was_armed = bool(section.get("armed", False))
    section["operation_days"] = days
    required = int(section.get("required_days", 14))

    if days >= required and not was_armed:
        section["armed"] = True
        section["armed_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"[GATE] *** PHASE 6 SCHARF *** ({days} Tage gesammelt, Schwelle {required})")
    elif was_armed:
        print(f"[GATE] Phase 6: armed (seit {section.get('armed_at', '?')}, {days} Tage)")
    else:
        print(f"[GATE] Phase 6: {days}/{required} Tage")


def main() -> int:
    days = count_journal_days()
    gates = load_gates()
    check_phase4e(gates, days)
    check_phase6(gates, days)
    gates["last_check"] = datetime.now().isoformat(timespec="seconds")
    save_gates(gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
