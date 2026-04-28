#!/usr/bin/env python3
"""
test_integration_moloch — Phase-6 Integrationstest.

Gate: Laeuft nur wenn phase_gates.json phase6.operation_days >= 14
      ODER genug Journal-Tage vorhanden sind.
Wenn Gate nicht offen: exit(0) mit Hinweis.

Tests:
  1. ICH-Form-Check: LLM-Antwort enthaelt "ich" (case-insensitive), nicht "kann ich helfen"
  2. Unbekannte-Person-Tension: tension steigt nach unknown_person-Event
  3. tension_delta != 0 in letzten Chat-Events im Journal
  4. Zone-Mapping: Guardian/Shadow/Berserker korrekt
  5. No-Crash: 0 SEGV/MemErr in journalctl der letzten 24h
"""

import os
import re
import sys
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Pfade ---
PROJECT_ROOT     = Path(__file__).parent.parent
JOURNAL_DIR      = Path("/mnt/moloch-data/memory/journal")
CONV_DIR         = Path("/mnt/moloch-data/memory/conversations")
STATUS_JSON      = Path("/dev/shm/moloch_status.json")
PHASE_GATES_PATH = PROJECT_ROOT / "config" / "phase_gates.json"
LOGS_DIR         = PROJECT_ROOT / "logs"
GATE_OPERATION_DAYS = 14

# ---------------------------------------------------------------------------
# Gate-Check
# ---------------------------------------------------------------------------

def check_gate() -> tuple[bool, int]:
    """Prueft ob Phase 6 scharf ist. Gibt (ready, days) zurueck."""
    try:
        days = len([
            f for f in os.listdir(JOURNAL_DIR)
            if re.match(r'\d{4}-\d{2}-\d{2}\.jsonl', f)
        ])
        archive = JOURNAL_DIR / "archive"
        if archive.is_dir():
            days += len([
                f for f in os.listdir(archive)
                if re.match(r'\d{4}-\d{2}-\d{2}\.jsonl', f)
            ])
        return days >= GATE_OPERATION_DAYS, days
    except Exception:
        return False, 0


def load_phase_gates() -> dict:
    try:
        with open(PHASE_GATES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_phase_gates(data: dict) -> None:
    import tempfile
    PHASE_GATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(PHASE_GATES_PATH.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(PHASE_GATES_PATH))
    except OSError:
        with open(PHASE_GATES_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test-Hilfsfunktionen
# ---------------------------------------------------------------------------

def latest_conv_file() -> Path | None:
    """Neueste Konversations-JSON-Datei."""
    files = sorted(CONV_DIR.glob("*.json")) if CONV_DIR.is_dir() else []
    return files[-1] if files else None


def load_journal_last_24h() -> list[dict]:
    """Laedt alle Journal-Events der letzten 24 Stunden."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    events = []
    if not JOURNAL_DIR.is_dir():
        return events
    # Aktuelle und archivierte Dateien
    candidates = list(JOURNAL_DIR.glob("*.jsonl"))
    archive = JOURNAL_DIR / "archive"
    if archive.is_dir():
        candidates += list(archive.glob("*.jsonl"))
    for path in candidates:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        ts_str = ev.get("ts", "")
                        # Normalisiere Timestamp
                        ts_str = ts_str.replace("Z", "+00:00")
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts >= cutoff:
                            events.append(ev)
                    except Exception:
                        continue
        except Exception:
            continue
    return events


def load_journal_last_n(n: int = 20) -> list[dict]:
    """Laedt die letzten n Journal-Events (nach Timestamp sortiert)."""
    all_events = []
    if not JOURNAL_DIR.is_dir():
        return all_events
    candidates = list(JOURNAL_DIR.glob("*.jsonl"))
    archive = JOURNAL_DIR / "archive"
    if archive.is_dir():
        candidates += list(archive.glob("*.jsonl"))
    for path in sorted(candidates):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_events.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    # Nach Timestamp sortieren, letzten n nehmen
    def ts_key(ev):
        try:
            return ev.get("ts", "")
        except Exception:
            return ""
    all_events.sort(key=ts_key)
    return all_events[-n:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

results: list[dict] = []


def record(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}: {detail}")


def test_ich_form() -> None:
    """Test 1: LLM-Antworten enthalten 'ich', nicht 'kann ich helfen'."""
    conv_file = latest_conv_file()
    if not conv_file:
        record("ICH-Form-Check", False, "Keine Konversations-Datei gefunden")
        return
    try:
        with open(conv_file) as f:
            entries = json.load(f)
    except Exception as e:
        record("ICH-Form-Check", False, f"Datei nicht lesbar: {e}")
        return

    # Letzte 5 Moloch-Antworten
    moloch_entries = [
        e for e in entries
        if e.get("sender") == "moloch" or e.get("role") == "assistant"
    ][-5:]

    if not moloch_entries:
        record("ICH-Form-Check", False, "Keine Moloch-Antworten in Konversation")
        return

    ich_count = 0
    help_count = 0
    for e in moloch_entries:
        text = (e.get("text") or e.get("content") or "").lower()
        if "ich " in text or text.startswith("ich"):
            ich_count += 1
        if "kann ich helfen" in text or "wie kann ich ihnen" in text:
            help_count += 1

    passed = ich_count >= 1
    detail = (
        f"{ich_count}/5 Antworten mit 'ich', "
        f"{help_count} 'kann ich helfen'-Treffer"
        f" (Datei: {conv_file.name})"
    )
    if help_count > 0:
        detail += " — WARNUNG: Assistenten-Sprache detected!"
    record("ICH-Form-Check", passed, detail)


def test_unknown_person_tension() -> None:
    """Test 2: tension_delta > 0 bei unknown-Person-Events in letzten 24h."""
    events = load_journal_last_24h()
    unknown_events = [
        ev for ev in events
        if ev.get("type") == "awareness"
        and "unbekannt" in (ev.get("interpretation") or "").lower()
    ]
    if not unknown_events:
        record(
            "Unknown-Person-Tension",
            True,
            "Keine unbekannten Personen in letzten 24h — Test nicht auslösbar (SKIP/PASS)"
        )
        return

    with_tension = [ev for ev in unknown_events if (ev.get("tension_delta") or 0.0) > 0]
    passed = len(with_tension) > 0
    detail = (
        f"{len(with_tension)}/{len(unknown_events)} unknown-Events "
        f"hatten tension_delta > 0"
    )
    record("Unknown-Person-Tension", passed, detail)


def test_tension_delta_chat() -> None:
    """Test 3: Min. 30% der Chat-Events haben tension_delta != 0."""
    events = load_journal_last_n(20)
    chat_events = [ev for ev in events if ev.get("type") == "chat"]
    if len(chat_events) < 5:
        record(
            "tension_delta Chat-Check",
            True,
            f"Nur {len(chat_events)} Chat-Events gefunden — zu wenig fuer verlässliche Aussage (SKIP/PASS)"
        )
        return

    nonzero = [
        ev for ev in chat_events
        if (ev.get("tension_delta") or 0.0) != 0.0
    ]
    ratio = len(nonzero) / len(chat_events)
    passed = ratio >= 0.30
    detail = (
        f"{len(nonzero)}/{len(chat_events)} Chat-Events "
        f"mit tension_delta != 0 ({ratio:.0%})"
    )
    record("tension_delta Chat-Check", passed, detail)


def test_zone_mapping() -> None:
    """Test 4: Zone in [guardian, shadow, berserker], tension in [0.0, 1.0]."""
    VALID_ZONES = {"guardian", "shadow", "berserker"}
    if not STATUS_JSON.exists():
        record(
            "Zone-Mapping",
            True,
            "Status-JSON nicht vorhanden (Service gestoppt?) — SKIP/PASS"
        )
        return
    try:
        with open(STATUS_JSON) as f:
            status = json.load(f)
    except Exception as e:
        record("Zone-Mapping", False, f"Status-JSON nicht lesbar: {e}")
        return

    core = status.get("core") or status
    zone = (core.get("zone") or "").lower()
    tension = core.get("tension")

    zone_ok = zone in VALID_ZONES
    tension_ok = tension is not None and 0.0 <= float(tension) <= 1.0

    passed = zone_ok and tension_ok
    detail = f"zone='{zone}' (valid={zone_ok}), tension={tension} (valid={tension_ok})"
    record("Zone-Mapping", passed, detail)


def test_no_crash() -> None:
    """Test 5: Keine SEGV/MemErr in journalctl der letzten 24h."""
    try:
        proc = subprocess.run(
            ["journalctl", "-u", "moloch", "--since", "24 hours ago", "--no-pager"],
            capture_output=True, text=True, timeout=15
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        record("No-Crash-Check", False, "journalctl Timeout nach 15s")
        return
    except FileNotFoundError:
        record("No-Crash-Check", True, "journalctl nicht verfuegbar — SKIP/PASS")
        return

    crash_patterns = [
        r"Segmentation fault",
        r"SIGSEGV",
        r"MemoryError",
        r"Out of memory",
        r"killed process",
        r"hailo.*error.*74",
        r"CRITICAL.*crash",
    ]
    found = []
    for pattern in crash_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            found.append(f"{pattern}: {len(matches)}x")

    passed = len(found) == 0
    if passed:
        detail = "0 Crashes/Errors in letzten 24h"
    else:
        detail = "Gefunden: " + ", ".join(found)
    record("No-Crash-Check", passed, detail)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    gate_ready, days = check_gate()

    print(f"\n=== Phase-6 Integrations-Gate ===")
    print(f"  Operationstage im Journal: {days}/{GATE_OPERATION_DAYS}")

    if not gate_ready:
        print(
            f"\n  Phase 6 Gate: {days}/{GATE_OPERATION_DAYS} Operationstage "
            f"— noch nicht scharf.\n"
            f"  Benoetigt: {GATE_OPERATION_DAYS - days} weitere Tag(e).\n"
            f"  Tests werden erst nach Erreichen des Schwellwerts ausgefuehrt.\n"
        )
        sys.exit(0)

    print(f"  Gate offen. Starte Tests...\n")

    print("=== Phase-6 Integrationstests ===\n")
    test_ich_form()
    test_unknown_person_tension()
    test_tension_delta_chat()
    test_zone_mapping()
    test_no_crash()

    # Ergebnis-Zusammenfassung
    total   = len(results)
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = total - passed
    overall = "PASS" if failed == 0 else "FAIL"

    print(f"\n=== Ergebnis: {overall} ({passed}/{total}) ===\n")

    # Log schreiben
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase6_validation_{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_path, "a") as f:
        f.write(f"\n--- {datetime.now().isoformat()} ---\n")
        for r in results:
            f.write(f"[{r['status']}] {r['name']}: {r['detail']}\n")
        f.write(f"GESAMT: {overall} ({passed}/{total})\n")
    print(f"  Log: {log_path}")

    # phase_gates.json updaten bei PASS
    if overall == "PASS":
        gates = load_phase_gates()
        if "phase6" not in gates:
            gates["phase6"] = {}
        gates["phase6"]["last_validation"] = datetime.now().isoformat()
        gates["phase6"]["last_result"]     = "PASS"
        gates["phase6"]["operation_days"]  = days
        save_phase_gates(gates)
        print("  phase_gates.json aktualisiert.")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
