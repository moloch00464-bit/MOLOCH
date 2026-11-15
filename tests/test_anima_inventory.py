#!/usr/bin/env python3
"""
ANIMA Laufzeit-Inventar: Auswertefunktionen und Robustheit, ohne Hardware.

Run:
    python3 -m pytest tests/test_anima_inventory.py -v
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "anima_inventory", os.path.join(ROOT, "scripts", "anima_inventory.py"))
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)


def test_meminfo_4gb_pi():
    text = "MemTotal:        4147200 kB\nMemFree:  100 kB\nMemAvailable:    2073600 kB\n"
    assert inv.parse_meminfo(text) == {"total_gb": 3.96, "available_gb": 1.98}


def test_meminfo_unreadable():
    assert inv.parse_meminfo(None) == {} and inv.parse_meminfo("Unsinn") == {}


def test_throttled_flags():
    assert inv.parse_throttled("throttled=0x0") == {
        "unterspannung_jetzt": False, "gedrosselt_jetzt": False,
        "unterspannung_seit_start": False, "gedrosselt_seit_start": False}
    flags = inv.parse_throttled("throttled=0x50005")
    assert all(flags.values())
    assert inv.parse_throttled("throttled=0x10000")["unterspannung_seit_start"] is True
    assert inv.parse_throttled("throttled=0x10000")["unterspannung_jetzt"] is False


def test_throttled_unreadable_is_none_not_false():
    """Nicht messbar ist nicht dasselbe wie 'keine Unterspannung'."""
    assert inv.parse_throttled(None) is None
    assert inv.parse_throttled("kaputt") is None
    assert inv.parse_throttled("throttled=xyz") is None


def test_missing_command_gives_none():
    assert inv.run(["gibt-es-nicht-xyz"]) is None


def test_unknown_is_spelled_out():
    assert inv.val(None) == "unbekannt" and inv.val("") == "unbekannt" and inv.val(0) == "0"


def test_report_builds_without_any_hardware(monkeypatch):
    """Kein Pi, keine NPU, kein Netz: Bericht entsteht trotzdem, ohne erfundene Werte."""
    monkeypatch.setattr(inv, "http_json", lambda *a, **k: None)
    report = inv.build_report()
    for title, _ in inv.SECTIONS:
        assert f"## {title}" in report
    assert "unbekannt" in report
    assert "Abschnitt fehlgeschlagen" not in report


def test_broken_section_does_not_kill_report(monkeypatch):
    def boom():
        raise RuntimeError("kaputt")
    monkeypatch.setattr(inv, "SECTIONS", [("Test", boom)])
    assert "Abschnitt fehlgeschlagen" in inv.build_report()


def test_script_never_uses_shell_or_writes_outside_logs():
    src = open(os.path.join(ROOT, "scripts", "anima_inventory.py"), encoding="utf-8").read()
    assert "shell=True" not in src
    assert src.count("open(tmp, \"w\"") == 1      # einzige Schreibstelle: der Bericht
