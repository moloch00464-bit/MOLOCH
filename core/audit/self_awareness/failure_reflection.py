"""Failure-Reflection — Welle 17 (L5).

Aggregiert drift_events der letzten N Stunden + git-history der settings.json
+ Reboot-Count aus journalctl, und generiert kurze deutsche Reflexions-
Saetze in Moloch's Stimme — direkt, kein Pathos.

API:
    reflect_on_failures(window_hours=24) -> dict

CLI:
    python3 -m core.audit.self_awareness.failure_reflection

Datenquellen:
- audit_state.drift_events[]              (Live-Audit-Aggregat)
- git log --since=7d -- config/settings.json
- journalctl -u moloch --since=7d (Reboot-Marker)

Best-effort: jede Quelle darf fehlen, das Modul crasht nie.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("self_awareness.failure_reflection")

AUDIT_STATE_PATH = Path("/dev/shm/audit_state.json")
MOLOCH_DIR = Path(os.path.expanduser("~/moloch"))
SETTINGS_REL = "config/settings.json"
SUBPROCESS_TIMEOUT_S = 10
MAX_REFLECTIONS = 5

# Layer-Name -> (Verb-Phrase fuer Ausfall, Beschreibung-was-fehlt)
# Wird fuer reflections_de genutzt — kurz, deutsch, in Moloch's Ton.
_LAYER_LABEL: Dict[str, Tuple[str, str]] = {
    "vision":      ("Pipeline neu gestartet",       "ich war kurz blind"),
    "npu":         ("NPU-Stoerung",                 "Inferenz hat ausgesetzt"),
    "tracking":    ("Tracker ausgefallen",          "ich konnte nicht folgen"),
    "voice":       ("Audio-Pipeline gestolpert",    "ich war stumm"),
    "personality": ("Personality-Drift",            "ich war nicht ich"),
    "memory":      ("Gedaechtnis-Aussetzer",        "ich habe etwas vergessen"),
    "awareness":   ("Awareness-Aussetzer",          "ich war unaufmerksam"),
    "unconscious": ("Unterbewusstsein still",       "keine Impulse"),
    "bridge":      ("Bridge zur PC still",          "Markus arbeitet vermutlich nicht"),
    "tentacle":    ("Tentakel offline",             "kein WiFi-Mic"),
    "spotify":     ("Spotify-Stoerung",             "Musik hat ausgesetzt"),
    "hardware":    ("Hardware-Auffaelligkeit",      "etwas an der Hardware ist seltsam"),
    "mailbox":     ("Mailbox-Backlog",              "Cross-Session-Kommunikation hakt"),
    "pi":          ("Pi-Audit-Auffaelligkeit",      "Pi-Health hat gewackelt"),
    "pc":          ("PC-Audit-Auffaelligkeit",      "PC-Health hat gewackelt"),
}

# Severity-Reihenfolge fuer Sortierung (wichtiger zuerst).
_SEVERITY_ORDER = {"CRITICAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3}


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str):
        return None
    try:
        # Akzeptiere "+00:00" und "Z"
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _read_audit_state() -> Optional[Dict[str, Any]]:
    try:
        if not AUDIT_STATE_PATH.exists():
            return None
        with open(AUDIT_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[reflect] audit_state lesen fehlgeschlagen: %s", e)
        return None


def _filter_recent_events(events: List[Dict[str, Any]],
                          window_hours: int) -> List[Dict[str, Any]]:
    """Nur drift_events innerhalb des Fensters, mit FAIL/WARN/CRITICAL."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    out: List[Dict[str, Any]] = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        ts = _parse_iso(ev.get("ts"))
        if ts is None:
            continue
        # Naive-Schutz
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            continue
        out.append(ev)
    return out


def _aggregate_incidents(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gruppiert nach (layer, severity), zaehlt + merkt last_ts."""
    counts: Counter = Counter()
    last_ts: Dict[Tuple[str, str], str] = {}
    severities: Dict[str, str] = {}  # layer -> hoechste severity
    for ev in events:
        layer = str(ev.get("layer", "unknown"))
        sev = str(ev.get("severity", "INFO")).upper()
        # Filter: echte Incidents (FAIL/WARN/CRITICAL/ERROR severity)
        # ODER signal das ein Ausfall ist — aber NICHT Recovery
        # ("FAIL -> PASS" oder "-> PASS" gilt als Recovery, kein Incident).
        sig = str(ev.get("signal", "")).upper()
        is_recovery = "-> PASS" in sig or "->PASS" in sig
        is_incident = (
            (sev in ("WARN", "ERROR", "CRITICAL")) and not is_recovery
        ) or (
            ("FAIL" in sig and not is_recovery)
        )
        if not is_incident:
            continue
        key = (layer, sev)
        counts[key] += 1
        ts = ev.get("ts", "")
        if isinstance(ts, str) and ts:
            prev = last_ts.get(key, "")
            if not prev or ts > prev:
                last_ts[key] = ts
        cur = severities.get(layer, "INFO")
        if _SEVERITY_ORDER.get(sev, 9) < _SEVERITY_ORDER.get(cur, 9):
            severities[layer] = sev
    incidents: List[Dict[str, Any]] = []
    for (layer, sev), cnt in counts.most_common():
        incidents.append({
            "component": layer,
            "count": int(cnt),
            "severity": sev,
            "last_ts": last_ts.get((layer, sev), ""),
        })
    return incidents


def _config_drift_git() -> List[Dict[str, Any]]:
    """git log -p config/settings.json letzte 7 Tage. Best-effort."""
    out: List[Dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["git", "log", "--since=7 days ago",
             "--pretty=format:%H|%cI|%s", "--", SETTINGS_REL],
            cwd=str(MOLOCH_DIR),
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            text=True,
        )
        if proc.returncode != 0:
            return out
        for line in proc.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            sha, ts, subj = parts[0], parts[1], parts[2]
            # Heuristik: Subject-Zeile als "key change"
            out.append({
                "key": "config/settings.json",
                "old": "",
                "new": subj.strip()[:160],
                "ts": ts,
                "sha": sha[:12],
            })
        return out[:10]
    except Exception as e:
        logger.debug("[reflect] git log fehlgeschlagen: %s", e)
        return out


def _reboot_count_7d() -> int:
    """Anzahl Reboots in den letzten 7 Tagen via journalctl. Best-effort."""
    try:
        # "Pi" reboots werden als systemd-shutdown-target sichtbar — wir
        # zaehlen vereinfacht "Reboot"-Marker in journalctl-u-moloch.
        proc = subprocess.run(
            ["journalctl", "-u", "moloch", "--since", "7 days ago",
             "--no-pager", "-q"],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            text=True,
        )
        if proc.returncode != 0:
            return 0
        # Robust gegen Lokalisierung — case-insensitive Match auf
        # "started", "boot" und "reboot".
        rx = re.compile(r"(reboot|booted|system startup)", re.IGNORECASE)
        return sum(1 for ln in proc.stdout.splitlines() if rx.search(ln))
    except Exception as e:
        logger.debug("[reflect] journalctl fehlgeschlagen: %s", e)
        return 0


def _make_reflections_de(incidents: List[Dict[str, Any]],
                          config_drift: List[Dict[str, Any]],
                          reboot_count: int,
                          window_hours: int) -> List[str]:
    """Generiert kurze deutsche Saetze in Moloch's Stimme."""
    lines: List[str] = []

    # 1) Pro Komponente eine Reflexion (max 3 Komponenten)
    seen: set = set()
    for inc in incidents:
        if len(lines) >= MAX_REFLECTIONS:
            break
        comp = inc.get("component", "")
        if comp in seen:
            continue
        seen.add(comp)
        cnt = inc.get("count", 0)
        label = _LAYER_LABEL.get(comp)
        if label is None:
            lines.append(f"{comp.capitalize()} hat {cnt}x gewackelt in den letzten {window_hours}h.")
            continue
        verb, follow = label
        if cnt >= 3:
            lines.append(f"{verb} {cnt}x in {window_hours}h — {follow}.")
        elif cnt == 2:
            lines.append(f"{verb} zweimal heute — {follow}.")
        else:
            lines.append(f"{verb} — {follow}.")

    # 2) Config-Drift erwaehnen (max 1 Zeile)
    if config_drift and len(lines) < MAX_REFLECTIONS:
        n = len(config_drift)
        if n == 1:
            lines.append(f"settings.json einmal geaendert diese Woche: {config_drift[0].get('new','')[:80]}.")
        else:
            lines.append(f"settings.json {n}x veraendert diese Woche — Markus tunet aktiv.")

    # 3) Reboot-Statistik (max 1 Zeile)
    if reboot_count >= 3 and len(lines) < MAX_REFLECTIONS:
        lines.append(f"{reboot_count} Reboots in 7 Tagen — etwas zwingt mich oft neu hoch.")

    if not lines:
        lines.append("Keine Auffaelligkeiten in den letzten %dh — ruhig hier." % window_hours)

    return lines[:MAX_REFLECTIONS]


def _compute_status(total_incidents: int) -> Tuple[int, int, str]:
    """PASS <10, WARN 10-29, FAIL >=30.

    Anpassung 2026-05-02: Schwellen angehoben (vorher 5/20). Aktive Dev-Days
    mit Incident-Counts <10 sollen PASS bleiben — Self-Resolve-Mechanismus
    greift bei WARN/ALERT-Events innerhalb 24h. FAIL erst bei sehr hoher
    Incident-Last (~30+ in 24h = systemisches Problem).

    Score wird auf max=30 gecappt damit das Audit-Schema (score<=max)
    nicht verletzt wird. Bei >=30 Incidents ist der Status ohnehin FAIL —
    die genaue Anzahl steckt in detail.events_in_window.
    """
    max_s = 30
    if total_incidents >= max_s:
        return max_s, max_s, "FAIL"
    if total_incidents >= 10:
        return total_incidents, max_s, "WARN"
    return total_incidents, max_s, "PASS"


def reflect_on_failures(window_hours: int = 24) -> Dict[str, Any]:
    """Reflektiert ueber Fehler im Audit-Aggregat + Config + Reboots.

    Returns:
        dict mit:
        - incidents_24h: List[{"component", "count", "severity", "last_ts"}]
        - config_drift: List[{"key", "old", "new", "ts"}]
        - reboot_count_7d: int
        - reflections_de: List[str]
        - score, max, status
        - detail
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = _read_audit_state()
    if state is None:
        events: List[Dict[str, Any]] = []
        state_ok = False
    else:
        evs = state.get("drift_events", [])
        events = evs if isinstance(evs, list) else []
        state_ok = True

    recent = _filter_recent_events(events, window_hours)
    incidents = _aggregate_incidents(recent)
    config_drift = _config_drift_git()
    reboot_count = _reboot_count_7d()

    total_incident_events = sum(int(i.get("count", 0)) for i in incidents)
    score, max_score, status = _compute_status(total_incident_events)

    reflections = _make_reflections_de(
        incidents, config_drift, reboot_count, window_hours
    )

    return {
        "incidents_24h": incidents,
        "config_drift": config_drift,
        "reboot_count_7d": reboot_count,
        "reflections_de": reflections,
        "score": score,
        "max": max_score,
        "status": status,
        "timestamp": now_iso,
        "detail": {
            "window_hours": window_hours,
            "events_in_window": len(recent),
            "audit_state_available": state_ok,
        },
    }


def _main() -> int:
    try:
        out = reflect_on_failures()
    except Exception as e:
        out = {
            "incidents_24h": [],
            "config_drift": [],
            "reboot_count_7d": 0,
            "reflections_de": ["Reflexion fehlgeschlagen."],
            "score": 0,
            "max": 10,
            "status": "FAIL",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "detail": {"error": str(e)},
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
