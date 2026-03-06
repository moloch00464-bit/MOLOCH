#!/usr/bin/env python3
"""
M.O.L.O.C.H. WIRING CHECKER — moloch wiring
=============================================
Vergleicht Architektur-Specs mit echtem Code.
Findet fehlende Pfeile, Module und Verdrahtungen.

Liest die JSON-Specs und prüft:
  1. Existieren die spezifizierten Dateien?
  2. Sind Event-Bus-Verbindungen verdrahtet (subscribe/publish)?
  3. Sind Chat-Integrationen vorhanden?
  4. Sind spezifizierte Funktionen/Klassen implementiert?
  5. Fehlen ganze Subsysteme?

Usage:
    python3 scripts/moloch_wiring_check.py           # Voller Check
    python3 scripts/moloch_wiring_check.py --json     # Maschinenlesbar
    python3 scripts/moloch_wiring_check.py --critical  # Nur Blocker

Erstellt: 06.03.2026
Quellen: Alle Architektur-Specs aus dem Projektordner
"""

import sys
import os
import json
import re
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple

MOLOCH_ROOT = Path.home() / "moloch"
sys.path.insert(0, str(MOLOCH_ROOT))


# ── Result Types ─────────────────────────────────────────────

@dataclass
class WiringGap:
    category: str       # FILE_MISSING, EVENT_UNWIRED, FUNCTION_MISSING, etc.
    severity: str       # CRITICAL, HIGH, MEDIUM, LOW
    spec_source: str    # Welche JSON-Spec hat das definiert
    expected: str       # Was laut Spec existieren sollte
    actual: str         # Was tatsächlich da ist
    fix_hint: str = ""  # Vorschlag was zu tun ist

    def to_dict(self):
        return asdict(self)


@dataclass
class WiringReport:
    timestamp: float = field(default_factory=time.time)
    gaps: List[dict] = field(default_factory=list)
    wired: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_gap(self, gap: WiringGap):
        self.gaps.append(gap.to_dict())

    def add_wired(self, category: str, description: str):
        self.wired.append({"category": category, "description": description})

    def finalize(self):
        critical = sum(1 for g in self.gaps if g["severity"] == "CRITICAL")
        high = sum(1 for g in self.gaps if g["severity"] == "HIGH")
        medium = sum(1 for g in self.gaps if g["severity"] == "MEDIUM")
        low = sum(1 for g in self.gaps if g["severity"] == "LOW")
        self.summary = {
            "total_gaps": len(self.gaps),
            "total_wired": len(self.wired),
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
            "verdict": "BROKEN" if critical > 0 else ("INCOMPLETE" if high > 0 else "OK"),
        }


report = WiringReport()
ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}


# ── Helper ───────────────────────────────────────────────────

def file_exists(rel_path: str) -> bool:
    return (MOLOCH_ROOT / rel_path).exists()


def file_contains(rel_path: str, pattern: str) -> bool:
    """Prüft ob eine Datei einen String/Pattern enthält."""
    fp = MOLOCH_ROOT / rel_path
    if not fp.exists():
        return False
    try:
        content = fp.read_text(errors="ignore")
        return pattern in content
    except Exception:
        return False


def file_contains_any(rel_path: str, patterns: List[str]) -> Tuple[bool, List[str]]:
    """Prüft welche Patterns in einer Datei vorkommen."""
    fp = MOLOCH_ROOT / rel_path
    if not fp.exists():
        return False, []
    try:
        content = fp.read_text(errors="ignore")
        found = [p for p in patterns if p in content]
        return bool(found), found
    except Exception:
        return False, []


def find_files_with_pattern(pattern: str, extensions=(".py",)) -> List[str]:
    """Sucht in allen Python-Dateien nach einem Pattern."""
    matches = []
    for ext in extensions:
        for fp in MOLOCH_ROOT.rglob(f"*{ext}"):
            if ".git" in str(fp) or "__pycache__" in str(fp):
                continue
            try:
                if pattern in fp.read_text(errors="ignore"):
                    matches.append(str(fp.relative_to(MOLOCH_ROOT)))
            except Exception:
                pass
    return matches


def find_event_subscribers(event_pattern: str) -> List[str]:
    """Findet alle Dateien die ein bestimmtes Event subscriben."""
    return find_files_with_pattern(f'subscribe("{event_pattern}') + \
           find_files_with_pattern(f"subscribe('{event_pattern}")


def find_event_publishers(event_pattern: str) -> List[str]:
    """Findet alle Dateien die ein bestimmtes Event publishen."""
    return find_files_with_pattern(f'publish("{event_pattern}') + \
           find_files_with_pattern(f"publish('{event_pattern}") + \
           find_files_with_pattern(f'emit("{event_pattern}') + \
           find_files_with_pattern(f"emit('{event_pattern}")


# ══════════════════════════════════════════════════════════════
# CHECKS — Jeder Check prüft eine Architektur-Spec
# ══════════════════════════════════════════════════════════════


def check_nervous_system():
    """
    Spec: moloch_nervous_system_spec.json
    Prüft ob Event Bus Quellen und Ziele verdrahtet sind.
    """
    print("\n  ── NERVENSYSTEM (Event Bus Verdrahtung) ──")

    # Spezifizierte Event-Quellen
    sources = {
        "vision_pipeline": ["perception.person", "perception.face", "perception.owner"],
        "hardware_events": ["hardware.ptz", "hardware.camera"],
        "system_state": ["system.mode", "system.health"],
        "capability_monitor": ["capability.added", "capability.changed"],
        "user_commands": ["user.command", "chat.message"],
    }

    # Spezifizierte Event-Ziele
    targets = {
        "action_bridge": "perception.",
        "memory_system": "perception.",
        "evolution_log": "capability.",
        "chat_interface": "perception.",
        "hardware_controller": "action.",
    }

    # Prüfe Quellen: wer publisht?
    for source_name, events in sources.items():
        for event in events:
            publishers = find_event_publishers(event)
            if publishers:
                report.add_wired("EVENT_SOURCE", f"{source_name} → {event} ({', '.join(publishers)})")
            else:
                severity = "HIGH" if event.startswith("perception.") else "MEDIUM"
                report.add_gap(WiringGap(
                    category="EVENT_UNWIRED",
                    severity=severity,
                    spec_source="moloch_nervous_system_spec.json",
                    expected=f"{source_name} publisht '{event}'",
                    actual="Kein Publisher gefunden",
                    fix_hint=f"Ein Modul muss bus.publish('{event}', ...) aufrufen",
                ))
                icon = ICONS[severity]
                print(f"  {icon} {event}: Kein Publisher — {source_name} sendet nicht")

    # Prüfe Ziele: wer subscribt?
    for target_name, event_prefix in targets.items():
        subscribers = find_event_subscribers(event_prefix)
        if subscribers:
            report.add_wired("EVENT_TARGET", f"{target_name} ← {event_prefix}* ({', '.join(subscribers)})")
        else:
            # Chat-Interface und Evolution Log sind die kritischen fehlenden
            severity = "CRITICAL" if target_name in ("chat_interface", "evolution_log") else "HIGH"
            report.add_gap(WiringGap(
                category="EVENT_UNWIRED",
                severity=severity,
                spec_source="moloch_nervous_system_spec.json",
                expected=f"{target_name} subscribt '{event_prefix}*' Events",
                actual="Kein Subscriber gefunden",
                fix_hint=f"{target_name} muss bus.subscribe('{event_prefix}*', handler) aufrufen",
            ))
            icon = ICONS[severity]
            print(f"  {icon} {target_name}: Subscribt NICHT auf {event_prefix}* Events")


def check_evolution_system():
    """
    Spec: moloch_evolution_log_design.json
    Prüft ob das Evolution Awareness System existiert.
    """
    print("\n  ── EVOLUTION AWARENESS (Selbstwahrnehmung) ──")

    required_files = {
        "system_capabilities.json": {
            "path": "config/system_capabilities.json",
            "alt_paths": ["state/system_capabilities.json"],
            "severity": "CRITICAL",
            "desc": "Capability Registry — Moloch weiß was er kann",
        },
        "capability_monitor.py": {
            "path": "core/capability_monitor.py",
            "alt_paths": ["core/evolution/capability_monitor.py", "scripts/capability_monitor.py"],
            "severity": "CRITICAL",
            "desc": "Capability Monitor — erkennt neue Fähigkeiten beim Start",
        },
        "moloch_evolution_log.json": {
            "path": "state/moloch_evolution_log.json",
            "alt_paths": ["logs/moloch_evolution_log.json", "config/moloch_evolution_log.json"],
            "severity": "HIGH",
            "desc": "Evolution Log — Lebensgeschichte des Systems",
        },
        "moloch_self_report.py": {
            "path": "scripts/moloch_self_report.py",
            "alt_paths": ["core/self_report.py"],
            "severity": "HIGH",
            "desc": "Self-Report — erzeugt aktuellen Fähigkeitsbericht",
        },
    }

    for name, spec in required_files.items():
        found = file_exists(spec["path"])
        if not found:
            for alt in spec.get("alt_paths", []):
                if file_exists(alt):
                    found = True
                    break

        if found:
            report.add_wired("EVOLUTION", f"{name} existiert")
            print(f"  ✅ {name}: vorhanden")
        else:
            report.add_gap(WiringGap(
                category="FILE_MISSING",
                severity=spec["severity"],
                spec_source="moloch_evolution_log_design.json",
                expected=f"{name} — {spec['desc']}",
                actual="Datei existiert nicht",
                fix_hint=f"Erstelle {spec['path']}",
            ))
            icon = ICONS[spec["severity"]]
            print(f"  {icon} {name}: FEHLT — {spec['desc']}")

    # Chat-Integration: Kann der Chat system_capabilities lesen?
    chat_files = find_files_with_pattern("system_capabilities")
    if chat_files:
        report.add_wired("EVOLUTION", f"Chat referenziert system_capabilities ({chat_files})")
    else:
        report.add_gap(WiringGap(
            category="INTEGRATION_MISSING",
            severity="CRITICAL",
            spec_source="moloch_evolution_log_design.json",
            expected="Chat liest system_capabilities.json für Selbstauskunft",
            actual="Kein Code referenziert system_capabilities",
            fix_hint="Chat-API-Call muss Capability-Daten als Kontext mitschicken",
        ))
        print(f"  🔴 Chat kennt system_capabilities NICHT — Moloch weiß nicht was er kann")


def check_chat_context_bridge():
    """
    Prüft ob der Chat-LLM Live-Systemdaten bekommt.
    DAS ist der Kern-Bug: Moloch redet blind.
    """
    print("\n  ── CHAT CONTEXT BRIDGE (Moloch's Bewusstsein) ──")

    # Finde Chat/API Module
    chat_modules = []
    for pattern in ["push_to_talk", "flask", "chat_handler", "chat_api", "llm_bridge"]:
        found = find_files_with_pattern(pattern)
        chat_modules.extend(found)
    chat_modules = list(set(chat_modules))

    if not chat_modules:
        report.add_gap(WiringGap(
            category="MODULE_MISSING",
            severity="HIGH",
            spec_source="System-Architektur",
            expected="Chat/LLM-Modul gefunden",
            actual="Kein Chat-Modul identifiziert",
        ))
        print(f"  🟠 Kein Chat-Modul gefunden")
        return

    print(f"  📋 Chat-Module gefunden: {', '.join(chat_modules)}")

    # Prüfe ob Live-Daten in den Chat-Kontext fließen
    live_data_checks = {
        "Kamera-Sicht (was sieht Moloch)": [
            "perception", "current_person", "face_id", "detection",
            "get_perception", "vision_state",
        ],
        "Bridge-State (was tut Moloch)": [
            "action_bridge", "get_action_bridge", "bridge_state",
            "fsm_state", "current_state",
        ],
        "Tension/Mood (was fühlt Moloch)": [
            "tension", "mood", "dominance", "get_tension",
            "mood_engine", "tension_integrator",
        ],
        "Spotify (was hört Moloch)": [
            "spotify", "current_track", "now_playing",
            "audio_features", "music_state",
        ],
        "Capabilities (was kann Moloch)": [
            "system_capabilities", "capability_registry",
            "evolution_log", "self_report",
        ],
        "Raumkontext (wo ist Moloch)": [
            "room_map", "current_zone", "zone",
            "context_evaluator", "activity",
        ],
    }

    for context_name, patterns in live_data_checks.items():
        found_in = []
        for module in chat_modules:
            has_any, matched = file_contains_any(module, patterns)
            if has_any:
                found_in.append(f"{module}({','.join(matched[:2])})")

        if found_in:
            report.add_wired("CHAT_CONTEXT", f"{context_name} → {', '.join(found_in)}")
            print(f"  ✅ {context_name}: verdrahtet")
        else:
            severity = "CRITICAL" if context_name in (
                "Kamera-Sicht (was sieht Moloch)",
                "Capabilities (was kann Moloch)",
            ) else "HIGH"
            report.add_gap(WiringGap(
                category="CONTEXT_MISSING",
                severity=severity,
                spec_source="Nervous System Spec + Evolution Log",
                expected=f"Chat-API bekommt {context_name} als Kontext",
                actual=f"Kein Chat-Modul referenziert relevante Daten",
                fix_hint=f"Vor dem API-Call: {context_name} aus dem System lesen und in System-Prompt injizieren",
            ))
            icon = ICONS[severity]
            print(f"  {icon} {context_name}: NICHT im Chat-Kontext")


def check_chat_action_return():
    """
    Prüft ob Chat-Antworten zurück ins System fließen.
    Wenn Moloch sagt 'ich drehe nach rechts' — passiert das?
    """
    print("\n  ── CHAT → ACTION (Molochs Wille → Handlung) ──")

    # Suche nach Action-Parsing in Chat-Modulen
    action_patterns = [
        "ptz_move", "camera.move", "pan", "tilt",
        "spotify.play", "music.change", "playlist",
        "led.set", "led.blink",
        "event_bus", "publish", "emit",
        "action_bridge",
    ]

    chat_files = find_files_with_pattern("push_to_talk") + \
                 find_files_with_pattern("chat_handler") + \
                 find_files_with_pattern("flask")
    chat_files = list(set(chat_files))

    action_wired = False
    for cf in chat_files:
        has_any, matched = file_contains_any(cf, action_patterns)
        if has_any:
            report.add_wired("CHAT_ACTION", f"{cf} hat Action-Anbindung: {matched}")
            action_wired = True

    if not action_wired:
        report.add_gap(WiringGap(
            category="ACTION_RETURN_MISSING",
            severity="CRITICAL",
            spec_source="Nervous System Spec",
            expected="Chat-Antworten werden nach Aktions-Intents geparst und auf Event Bus geworfen",
            actual="Kein Action-Return-Pfad im Chat gefunden",
            fix_hint="Chat-Response parsen: 'drehe Kamera' → bus.emit('action.ptz_move', ...), "
                     "'spiele Musik' → bus.emit('action.music_change', ...)",
        ))
        print(f"  🔴 Chat → Action: KEIN Rückkanal — Molochs Worte haben keine Wirkung")
    else:
        print(f"  ✅ Chat → Action: Teilweise verdrahtet")


def check_spotify_integration():
    """
    Spec: SPOTIFY_SOUL_INTEGRATION
    Prüft ob Spotify als emotionaler Kanal verdrahtet ist.
    """
    print("\n  ── SPOTIFY SOUL (Musik als Emotion) ──")

    spec_modules = {
        "core/music/spotify_bridge.py": "Spotify → Event Bus Bridge",
        "core/music/music_state_mapper.py": "Audio Features → Emotional States",
        "core/music/music_memory.py": "Track-Person-Mood Assoziationen",
    }

    for module_path, description in spec_modules.items():
        if file_exists(module_path):
            report.add_wired("SPOTIFY", f"{module_path} existiert")
            print(f"  ✅ {module_path}: vorhanden")
        else:
            # Suche nach alternativen Pfaden
            alt_name = Path(module_path).stem
            alts = find_files_with_pattern(alt_name)
            if alts:
                report.add_wired("SPOTIFY", f"{alt_name} gefunden in {alts}")
                print(f"  ✅ {alt_name}: gefunden in {', '.join(alts)}")
            else:
                report.add_gap(WiringGap(
                    category="FILE_MISSING",
                    severity="MEDIUM",
                    spec_source="SPOTIFY_SOUL_INTEGRATION",
                    expected=f"{module_path} — {description}",
                    actual="Nicht implementiert",
                    fix_hint=f"Erstelle {module_path}",
                ))
                print(f"  🟡 {module_path}: FEHLT — {description}")

    # Spotify Events auf dem Bus?
    spotify_events = ["music_track_started", "music_features_received", "music_mood_changed"]
    for event in spotify_events:
        pubs = find_event_publishers(event)
        subs = find_event_subscribers(event)
        if pubs or subs:
            report.add_wired("SPOTIFY_EVENT", f"{event}: pub={pubs}, sub={subs}")
        else:
            report.add_gap(WiringGap(
                category="EVENT_UNWIRED",
                severity="LOW",
                spec_source="SPOTIFY_SOUL_INTEGRATION",
                expected=f"Event '{event}' wird published und subscribed",
                actual="Weder Publisher noch Subscriber",
            ))


def check_gate_modules():
    """
    Prüft ob die Gate-spezifischen Module existieren.
    """
    print("\n  ── GATE MODULE (Implementierungsstatus) ──")

    gate_modules = {
        "Gate 1 — Action Bridge": {
            "core/action_bridge.py": "CRITICAL",
            "core/moloch_event_bus.py": "CRITICAL",
        },
        "Gate 2 — Identity/Memory": {
            "core/memory/episodic_memory.py": "HIGH",
            "core/memory/music_memory.py": "HIGH",
        },
        "Gate 3 — Awareness": {
            "core/awareness/room_map.py": "MEDIUM",
            "core/awareness/motion_analyzer.py": "MEDIUM",
            "core/awareness/activity_classifier.py": "MEDIUM",
            "core/awareness/context_evaluator.py": "MEDIUM",
        },
        "Gate 4 — Personality": {
            "core/personality/tension_integrator.py": "MEDIUM",
            "core/personality/mood_engine.py": "MEDIUM",
            "core/personality/behavior_rules.py": "MEDIUM",
        },
        "Gate 5 — Autonomy": {
            "core/autonomy/decision_engine.py": "MEDIUM",
            "core/autonomy/atmosphere_controller.py": "MEDIUM",
            "core/autonomy/homeostasis.py": "MEDIUM",
            "core/autonomy/night_cycle.py": "MEDIUM",
        },
    }

    for gate_name, modules in gate_modules.items():
        found = 0
        total = len(modules)
        for module_path, severity in modules.items():
            if file_exists(module_path):
                found += 1
            else:
                # Suche alternatives Layout
                alt_name = Path(module_path).name
                alts = list(MOLOCH_ROOT.rglob(alt_name))
                if alts:
                    found += 1
                else:
                    report.add_gap(WiringGap(
                        category="GATE_MODULE_MISSING",
                        severity=severity,
                        spec_source="Gate Roadmap",
                        expected=f"{gate_name}: {module_path}",
                        actual="Nicht gefunden",
                    ))

        status = "✅" if found == total else ("🟡" if found > 0 else "🔴")
        print(f"  {status} {gate_name}: {found}/{total} Module")


def check_cross_module_wiring():
    """
    Prüft kritische Modul-zu-Modul Verbindungen.
    """
    print("\n  ── CROSS-MODULE WIRING (Modul-Verbindungen) ──")

    connections = [
        {
            "name": "Perception → Event Bus",
            "source_pattern": "perception_engine",
            "target_pattern": "publish",
            "severity": "CRITICAL",
        },
        {
            "name": "Event Bus → Action Bridge",
            "source_pattern": "action_bridge",
            "target_pattern": "subscribe",
            "severity": "CRITICAL",
        },
        {
            "name": "Action Bridge → Tracker (PTZ)",
            "source_pattern": "action_bridge",
            "target_pattern": "ptz",
            "severity": "HIGH",
            "alt_patterns": ["tracker", "camera"],
        },
        {
            "name": "Tension → Mood Engine",
            "source_pattern": "tension_integrator",
            "target_pattern": "mood",
            "severity": "MEDIUM",
        },
        {
            "name": "Decision Engine → Event Bus",
            "source_pattern": "decision_engine",
            "target_pattern": "publish",
            "severity": "MEDIUM",
            "alt_patterns": ["emit"],
        },
        {
            "name": "Chat → System-Kontext",
            "source_pattern": "push_to_talk",
            "target_pattern": "system_capabilities",
            "severity": "CRITICAL",
            "alt_patterns": ["context_snapshot", "live_context", "get_status"],
        },
        {
            "name": "moloch_service → Bridge Start",
            "source_pattern": "moloch_service",
            "target_pattern": "action_bridge",
            "severity": "CRITICAL",
        },
        {
            "name": "moloch_service → Homeostasis Start",
            "source_pattern": "moloch_service",
            "target_pattern": "homeostasis",
            "severity": "HIGH",
        },
    ]

    for conn in connections:
        # Finde Source-Files
        source_files = find_files_with_pattern(conn["source_pattern"])
        if not source_files:
            report.add_gap(WiringGap(
                category="MODULE_MISSING",
                severity=conn["severity"],
                spec_source="System-Architektur",
                expected=f"{conn['name']}: Source-Modul '{conn['source_pattern']}'",
                actual="Source-Modul nicht gefunden",
            ))
            icon = ICONS[conn["severity"]]
            print(f"  {icon} {conn['name']}: Source fehlt")
            continue

        # Prüfe ob Source das Target referenziert
        wired = False
        for sf in source_files:
            patterns = [conn["target_pattern"]] + conn.get("alt_patterns", [])
            has_any, matched = file_contains_any(sf, patterns)
            if has_any:
                wired = True
                report.add_wired("CROSS_MODULE", f"{conn['name']}: {sf} → {matched}")
                break

        if wired:
            print(f"  ✅ {conn['name']}: verdrahtet")
        else:
            report.add_gap(WiringGap(
                category="WIRING_MISSING",
                severity=conn["severity"],
                spec_source="System-Architektur",
                expected=f"{conn['name']}",
                actual=f"Source '{conn['source_pattern']}' referenziert '{conn['target_pattern']}' nicht",
                fix_hint=f"Verbindung herstellen: {conn['source_pattern']} → {conn['target_pattern']}",
            ))
            icon = ICONS[conn["severity"]]
            print(f"  {icon} {conn['name']}: NICHT verdrahtet")


# ══════════════════════════════════════════════════════════════
# SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════

def print_summary():
    """Druckt die Zusammenfassung mit den wichtigsten Lücken."""
    report.finalize()
    s = report.summary

    print()
    print("═" * 60)
    print(f"  M.O.L.O.C.H. WIRING REPORT")
    print(f"═" * 60)
    print(f"  Verbunden:  {s['total_wired']}")
    print(f"  Lücken:     {s['total_gaps']}")
    print(f"    🔴 Critical: {s['critical']}")
    print(f"    🟠 High:     {s['high']}")
    print(f"    🟡 Medium:   {s['medium']}")
    print(f"    ⚪ Low:      {s['low']}")
    print(f"  Verdict:    {s['verdict']}")
    print(f"═" * 60)

    # Top-Lücken anzeigen
    criticals = [g for g in report.gaps if g["severity"] == "CRITICAL"]
    if criticals:
        print(f"\n  🔴 KRITISCHE LÜCKEN — Müssen gefixt werden:\n")
        for i, gap in enumerate(criticals, 1):
            print(f"  {i}. {gap['expected']}")
            print(f"     Status: {gap['actual']}")
            if gap.get("fix_hint"):
                print(f"     Fix: {gap['fix_hint']}")
            print()

    highs = [g for g in report.gaps if g["severity"] == "HIGH"]
    if highs:
        print(f"  🟠 HOHE PRIORITÄT — Sollten bald gefixt werden:\n")
        for i, gap in enumerate(highs, 1):
            print(f"  {i}. {gap['expected']}")
            if gap.get("fix_hint"):
                print(f"     Fix: {gap['fix_hint']}")
        print()


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

def run_full():
    print()
    print("═" * 60)
    print("  M.O.L.O.C.H. WIRING CHECK")
    print("  Spec vs. Reality — Fehlende Verbindungen finden")
    print("═" * 60)

    check_nervous_system()
    check_evolution_system()
    check_chat_context_bridge()
    check_chat_action_return()
    check_spotify_integration()
    check_gate_modules()
    check_cross_module_wiring()
    print_summary()


def run_critical_only():
    print("\n  M.O.L.O.C.H. WIRING CHECK — NUR CRITICAL\n")
    check_chat_context_bridge()
    check_chat_action_return()
    check_evolution_system()
    report.finalize()
    criticals = [g for g in report.gaps if g["severity"] == "CRITICAL"]
    print(f"\n  🔴 {len(criticals)} kritische Lücken gefunden\n")
    for i, gap in enumerate(criticals, 1):
        print(f"  {i}. {gap['expected']}")
        if gap.get("fix_hint"):
            print(f"     → {gap['fix_hint']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="M.O.L.O.C.H. Wiring Check — findet fehlende Verdrahtungen"
    )
    parser.add_argument("--json", action="store_true", help="JSON Output")
    parser.add_argument("--critical", action="store_true", help="Nur Critical Gaps")
    args = parser.parse_args()

    if args.critical:
        run_critical_only()
    else:
        run_full()

    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))

    sys.exit(0 if report.summary.get("verdict") != "BROKEN" else 1)


if __name__ == "__main__":
    main()
