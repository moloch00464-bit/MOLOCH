#!/usr/bin/env python3
"""
M.O.L.O.C.H. Self-Report — Was kann ich?
==========================================
Liest system_capabilities.json + evolution_log.json und gibt
einen Kurzbericht aus was Moloch aktuell kann und was sich
zuletzt geaendert hat.

Usage:
    python3 scripts/moloch_self_report.py
    python3 scripts/moloch_self_report.py --json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

MOLOCH_ROOT = Path.home() / "moloch"
CAPS_PATH = MOLOCH_ROOT / "config" / "system_capabilities.json"
EVOLOG_PATH = MOLOCH_ROOT / "state" / "moloch_evolution_log.json"


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def print_report(caps: dict, evolog: list):
    print()
    print("=" * 55)
    print("  M.O.L.O.C.H. SELF-REPORT")
    print(f"  Generiert: {caps.get('_meta', {}).get('generated', '?')}")
    print("=" * 55)

    # --- Hardware ---
    hw = caps.get("hardware", {})
    brain = hw.get("brain", {})
    npu = hw.get("npu", {})
    cam = hw.get("camera", {})
    print(f"\n  HARDWARE")
    print(f"    Brain:  {brain.get('device', '?')} ({brain.get('ram_gb', '?')}GB)")
    print(f"    NPU:    {npu.get('device', '?')} ({npu.get('tops', '?')} TOPS, {npu.get('ram_gb', '?')}GB)")
    npu_status = "ONLINE" if npu.get("online") else "offline"
    print(f"            Status: {npu_status}")
    print(f"    Kamera: {cam.get('device', '?')} ({cam.get('resolution', '?')} @{cam.get('fps', '?')}fps)")

    # --- Vision-Modelle ---
    models = caps.get("vision_models", [])
    active = [m for m in models if m.get("active")]
    reserve = [m for m in models if not m.get("active")]
    print(f"\n  VISION ({len(active)} aktiv, {len(reserve)} Reserve)")
    for m in active:
        print(f"    [AKTIV]   {m['name']:30s} {m.get('task', '?')}")
    for m in reserve:
        print(f"    [Reserve] {m['name']:30s} {m.get('task', '?')}")

    # --- Faehigkeiten ---
    abilities = caps.get("abilities", {})
    print(f"\n  FAEHIGKEITEN")
    for category, items in abilities.items():
        if not isinstance(items, dict):
            continue
        aktiv = [k for k, v in items.items() if v is True]
        inaktiv = [k for k, v in items.items() if v is False]
        marker = "+" if aktiv else "-"
        summary = ", ".join(aktiv) if aktiv else "keine"
        print(f"    {category:20s} [{marker}] {summary}")
        if inaktiv:
            print(f"    {'':20s}     (offen: {', '.join(inaktiv)})")

    # --- Pipeline ---
    pipe = caps.get("pipeline", {})
    print(f"\n  PIPELINE")
    print(f"    Typ:     {pipe.get('type', '?')}")
    print(f"    FPS:     {pipe.get('fps', '?')}")
    print(f"    Modelle: {', '.join(pipe.get('active_models', []))}")

    # --- Module ---
    modules = caps.get("modules", {})
    total = sum(len(v) for v in modules.values())
    print(f"\n  MODULE ({total} importierbar)")
    for pkg, mods in modules.items():
        print(f"    {pkg:15s} [{len(mods):2d}] {', '.join(mods[:5])}", end="")
        if len(mods) > 5:
            print(f" +{len(mods)-5} mehr", end="")
        print()

    # --- Gates ---
    gates = caps.get("gates", {})
    print(f"\n  GATES")
    for gate_id, info in gates.items():
        status = info.get("status", "?")
        name = info.get("name", "?")
        date = info.get("date", "")
        marker = {"PASS": "+", "AKTIV": ">", "GEPLANT": " "}.get(status, "?")
        date_str = f" ({date})" if date else ""
        print(f"    [{marker}] {gate_id:10s} {status:8s} {name}{date_str}")

    # --- Evolution Log ---
    print(f"\n  EVOLUTION LOG ({len(evolog)} Eintraege)")
    if not evolog:
        print("    (leer — noch keine Aenderungen aufgezeichnet)")
    else:
        # Letzte 5 Eintraege
        recent = evolog[-5:]
        for entry in reversed(recent):
            ts = entry.get("timestamp", "?")
            added = entry.get("added", [])
            removed = entry.get("removed", [])
            print(f"    {ts}:")
            for a in added[:3]:
                print(f"      + {a}")
            if len(added) > 3:
                print(f"      ... +{len(added)-3} weitere")
            for r in removed[:3]:
                print(f"      - {r}")
            if len(removed) > 3:
                print(f"      ... -{len(removed)-3} weitere")

    print()
    print("=" * 55)
    print()


def main():
    parser = argparse.ArgumentParser(description="M.O.L.O.C.H. Self-Report")
    parser.add_argument("--json", action="store_true", help="JSON Output")
    args = parser.parse_args()

    caps = load_json(CAPS_PATH)
    if caps is None:
        print(f"FEHLER: {CAPS_PATH} nicht gefunden oder unlesbar.")
        print("Tipp: capability_monitor.run() beim Systemstart ausfuehren.")
        sys.exit(1)

    evolog = load_json(EVOLOG_PATH)
    if evolog is None:
        evolog = []

    if args.json:
        report = {"capabilities": caps, "evolution_log": evolog}
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(caps, evolog)


if __name__ == "__main__":
    main()
