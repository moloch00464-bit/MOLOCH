#!/usr/bin/env python3
"""
M.O.L.O.C.H. Watcher Agent — Stiller Beobachter (Wu-Wei)
==========================================================

Taeglicher Bericht ueber Systemzustand, Entscheidungen, Anomalien.
Greift NICHT ein — beobachtet nur.

Datenquellen:
  1. journalctl (moloch.service Logs)
  2. Event Bus JSONL (logs/events/events_YYYY-MM-DD.jsonl)
  3. Decision Engine (decision_made Events)
  4. Homeostasis (health_alert Events)
  5. Night Cycle (night_YYYY-MM-DD.json)
  6. System Metriken (CPU, RAM, Temp via /proc + vcgencmd)

Ausgabe:
  ~/moloch/logs/watcher/YYYY-MM-DD.md

Ausfuehrung:
  python3 ~/moloch/dev/agents/watcher_agent.py          # Heute
  python3 ~/moloch/dev/agents/watcher_agent.py 2026-03-05  # Bestimmter Tag

Cron (taeglich 06:15):
  15 6 * * * /usr/bin/python3 /home/molochzuhause/moloch/dev/agents/watcher_agent.py
"""

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, date
from pathlib import Path

# --- Pfade ---
MOLOCH_HOME = Path.home() / "moloch"
LOG_DIR = MOLOCH_HOME / "logs"
EVENTS_DIR = LOG_DIR / "events"
WATCHER_DIR = LOG_DIR / "watcher"
NIGHT_CYCLE_DIR = Path("/mnt/moloch-data/memory/night_cycle")


def get_target_date() -> date:
    """Zieldatum aus CLI-Argument oder gestern (Cron laeuft morgens)."""
    if len(sys.argv) > 1:
        return date.fromisoformat(sys.argv[1])
    # Standard: gestern (Cron laeuft um 06:15, bewertet den Vortag)
    from datetime import timedelta
    return date.today() - timedelta(days=1)


# ============================================================
# Datenquellen
# ============================================================

def read_journalctl(target: date) -> dict:
    """journalctl fuer moloch.service am Zieldatum auslesen."""
    since = target.isoformat()
    until = (target.replace(day=target.day)).__class__(
        target.year, target.month, target.day
    )
    from datetime import timedelta
    until_str = (target + timedelta(days=1)).isoformat()

    try:
        result = subprocess.run(
            ["journalctl", "-u", "moloch.service", "--no-pager",
             "--since", since, "--until", until_str, "-o", "short-iso"],
            capture_output=True, text=True, timeout=30
        )
        lines = result.stdout.strip().splitlines() if result.stdout else []
    except Exception:
        lines = []

    errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
    warnings = [l for l in lines if "WARNING" in l or "WARNUNG" in l]
    restarts = [l for l in lines if "Started" in l or "Stopped" in l or "SIGTERM" in l]

    return {
        "total_lines": len(lines),
        "errors": errors[:30],
        "warnings": warnings[:30],
        "restarts": restarts,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "restart_count": len(restarts),
    }


def read_event_log(target: date) -> dict:
    """Event Bus JSONL fuer den Tag auslesen und kategorisieren."""
    path = EVENTS_DIR / f"events_{target.isoformat()}.jsonl"
    events = []
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Nach Typ zaehlen
    type_counts = Counter(e.get("event_type", "unknown") for e in events)

    # Decisions extrahieren
    decisions = [e for e in events if e.get("event_type") == "decision_made"]

    # Health Alerts extrahieren
    health_alerts = [e for e in events if e.get("event_type") == "health_alert"]

    # Prioritaets-Verteilung
    prio_counts = Counter(e.get("priority", -1) for e in events)
    prio_labels = {0: "CRITICAL", 1: "perception", 2: "action",
                   3: "bridge", 4: "system", 5: "info", 8: "debug", 9: "logging"}
    prio_named = {prio_labels.get(k, f"prio_{k}"): v for k, v in sorted(prio_counts.items())}

    return {
        "total_events": len(events),
        "type_counts": dict(type_counts.most_common(25)),
        "priority_distribution": prio_named,
        "decisions": decisions,
        "health_alerts": health_alerts,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0,
    }


def read_night_cycle(target: date) -> dict:
    """Night Cycle Ergebnis fuer den Tag lesen."""
    path = NIGHT_CYCLE_DIR / f"night_{target.isoformat()}.json"
    if not path.exists():
        return {"ran": False}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {"ran": True, **data}
    except Exception:
        return {"ran": False, "error": "JSON parse Fehler"}


def read_system_metrics() -> dict:
    """Aktuelle System-Metriken (CPU-Temp, RAM, Disk, Uptime)."""
    metrics = {}

    # CPU-Temperatur
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=5
        )
        # Ausgabe: "temp=52.0'C"
        temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
        metrics["cpu_temp_c"] = float(temp_str)
    except Exception:
        metrics["cpu_temp_c"] = None

    # RAM
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        total = meminfo.get("MemTotal", 1)
        avail = meminfo.get("MemAvailable", 0)
        metrics["ram_total_mb"] = round(total / 1024)
        metrics["ram_used_mb"] = round((total - avail) / 1024)
        metrics["ram_percent"] = round((total - avail) / total * 100, 1)
    except Exception:
        pass

    # Disk
    try:
        st = os.statvfs("/home/molochzuhause/moloch")
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        metrics["disk_total_gb"] = round(total / 1024**3, 1)
        metrics["disk_used_gb"] = round((total - free) / 1024**3, 1)
        metrics["disk_percent"] = round((total - free) / total * 100, 1)
    except Exception:
        pass

    # Uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_s = float(f.read().split()[0])
        hours = int(uptime_s // 3600)
        mins = int((uptime_s % 3600) // 60)
        metrics["uptime"] = f"{hours}h {mins}m"
    except Exception:
        pass

    return metrics


# ============================================================
# Qi-Bewertung
# ============================================================

def assess_qi(journal: dict, events: dict, night: dict, metrics: dict) -> dict:
    """Qi-Fluss bewerten: 0-100 Punkte, Bewertung, Anomalien, Empfehlungen."""
    score = 100
    anomalies = []
    recommendations = []

    # --- Journal-Gesundheit ---
    if journal["error_count"] > 50:
        score -= 25
        anomalies.append(f"Hohe Fehlerrate: {journal['error_count']} Errors in journalctl")
    elif journal["error_count"] > 10:
        score -= 10
        anomalies.append(f"Erhoehte Fehlerrate: {journal['error_count']} Errors")

    if journal["restart_count"] > 3:
        score -= 15
        anomalies.append(f"Viele Neustarts: {journal['restart_count']}x Service restart")
    elif journal["restart_count"] > 1:
        score -= 5

    if journal["warning_count"] > 100:
        score -= 10
        anomalies.append(f"Warning-Flut: {journal['warning_count']} Warnings")

    # --- Event-Fluss ---
    if events["total_events"] == 0:
        score -= 30
        anomalies.append("KEIN Event-Log — Service lief nicht oder Bus defekt")
    elif events["total_events"] < 100:
        score -= 10
        anomalies.append(f"Wenig Events ({events['total_events']}) — reduzierte Aktivitaet")

    critical_count = events["priority_distribution"].get("CRITICAL", 0)
    if critical_count > 0:
        score -= min(20, critical_count * 5)
        anomalies.append(f"{critical_count} CRITICAL Events — sofort pruefen!")

    # --- Health Alerts ---
    alert_count = len(events["health_alerts"])
    if alert_count > 20:
        score -= 15
        anomalies.append(f"Homeostasis gestresst: {alert_count} Health-Alerts")
    elif alert_count > 5:
        score -= 5

    # RAM/Thermal Alerts einzeln zaehlen
    ram_alerts = [a for a in events["health_alerts"]
                  if "ram" in a.get("payload", {}).get("type", "")]
    temp_alerts = [a for a in events["health_alerts"]
                   if "temp" in a.get("payload", {}).get("type", "")]
    if len(ram_alerts) > 5:
        recommendations.append("RAM-Druck haeufig — GC-Intervall oder Pipeline-Buffer pruefen")
    if len(temp_alerts) > 5:
        recommendations.append("Thermische Alerts — Kuehlung oder Workload pruefen")

    # --- Entscheidungen ---
    decision_count = len(events["decisions"])
    if decision_count > 0:
        action_counts = Counter(
            d.get("payload", {}).get("action", "unknown") for d in events["decisions"]
        )
        silence_ratio = action_counts.get("silence", 0) / decision_count
        if silence_ratio > 0.9 and decision_count > 10:
            anomalies.append(f"Decision Engine passiv: {silence_ratio*100:.0f}% Silence")
            recommendations.append("Decision-Thresholds pruefen — System traut sich nichts")

    # --- Night Cycle ---
    if not night.get("ran"):
        score -= 5
        recommendations.append("Night Cycle lief nicht — Dreaming-Phase pruefen")

    # --- System Metriken (aktuell, nicht historisch) ---
    if metrics.get("ram_percent", 0) > 85:
        anomalies.append(f"RAM aktuell hoch: {metrics['ram_percent']}%")
    if metrics.get("cpu_temp_c") and metrics["cpu_temp_c"] > 75:
        anomalies.append(f"CPU-Temp aktuell hoch: {metrics['cpu_temp_c']}°C")
    if metrics.get("disk_percent", 0) > 85:
        recommendations.append(f"Disk {metrics['disk_percent']}% voll — Event-Logs rotieren?")

    # --- Gesamtbewertung ---
    score = max(0, min(100, score))
    if score >= 80:
        level = "Harmonisch"
        emoji = "🟢"
    elif score >= 60:
        level = "Leicht gestoert"
        emoji = "🟡"
    elif score >= 40:
        level = "Angespannt"
        emoji = "🟠"
    else:
        level = "Blockiert"
        emoji = "🔴"

    return {
        "score": score,
        "level": level,
        "emoji": emoji,
        "anomalies": anomalies,
        "recommendations": recommendations,
    }


# ============================================================
# Markdown-Report
# ============================================================

def generate_report(target: date, journal: dict, events: dict,
                    night: dict, metrics: dict, qi: dict) -> str:
    """Tagesbericht als Markdown generieren."""
    lines = []
    lines.append(f"# Watcher Report — {target.isoformat()}")
    lines.append(f"*Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Wu-Wei — beobachten, nicht eingreifen*")
    lines.append("")

    # --- Qi-Fluss ---
    lines.append(f"## {qi['emoji']} Qi-Fluss: {qi['score']}/100 — {qi['level']}")
    lines.append("")

    # --- System-Metriken ---
    lines.append("## System-Metriken (aktuell)")
    lines.append("")
    lines.append("| Metrik | Wert |")
    lines.append("|--------|------|")
    if metrics.get("cpu_temp_c") is not None:
        lines.append(f"| CPU-Temp | {metrics['cpu_temp_c']}°C |")
    if metrics.get("ram_used_mb"):
        lines.append(f"| RAM | {metrics['ram_used_mb']}/{metrics['ram_total_mb']} MB ({metrics['ram_percent']}%) |")
    if metrics.get("disk_used_gb"):
        lines.append(f"| Disk (SSD1) | {metrics['disk_used_gb']}/{metrics['disk_total_gb']} GB ({metrics['disk_percent']}%) |")
    if metrics.get("uptime"):
        lines.append(f"| Uptime | {metrics['uptime']} |")
    lines.append("")

    # --- Journal ---
    lines.append("## journalctl (moloch.service)")
    lines.append("")
    lines.append(f"- **Log-Zeilen:** {journal['total_lines']}")
    lines.append(f"- **Errors:** {journal['error_count']}")
    lines.append(f"- **Warnings:** {journal['warning_count']}")
    lines.append(f"- **Neustarts:** {journal['restart_count']}")
    if journal["restarts"]:
        lines.append("")
        lines.append("**Service-Events:**")
        for r in journal["restarts"][:10]:
            lines.append(f"- `{r[:120]}`")
    if journal["errors"][:5]:
        lines.append("")
        lines.append("**Top Errors (max 5):**")
        for e in journal["errors"][:5]:
            lines.append(f"- `{e[:150]}`")
    lines.append("")

    # --- Event Bus ---
    lines.append("## Event Bus")
    lines.append("")
    lines.append(f"- **Events gesamt:** {events['total_events']}")
    lines.append(f"- **Log-Groesse:** {events['file_size_mb']} MB")
    lines.append("")

    if events["priority_distribution"]:
        lines.append("**Prioritaets-Verteilung:**")
        lines.append("")
        lines.append("| Prioritaet | Anzahl |")
        lines.append("|------------|--------|")
        for prio, count in events["priority_distribution"].items():
            lines.append(f"| {prio} | {count} |")
        lines.append("")

    if events["type_counts"]:
        lines.append("**Top Event-Typen (max 15):**")
        lines.append("")
        lines.append("| Typ | Anzahl |")
        lines.append("|-----|--------|")
        for typ, count in list(events["type_counts"].items())[:15]:
            lines.append(f"| {typ} | {count} |")
        lines.append("")

    # --- Entscheidungen ---
    decisions = events["decisions"]
    lines.append(f"## Entscheidungen ({len(decisions)})")
    lines.append("")
    if decisions:
        action_counts = Counter(
            d.get("payload", {}).get("action", "?") for d in decisions
        )
        lines.append("**Aktions-Verteilung:**")
        lines.append("")
        lines.append("| Aktion | Anzahl |")
        lines.append("|--------|--------|")
        for action, count in action_counts.most_common():
            lines.append(f"| {action} | {count} |")
        lines.append("")

        # Letzte 5 nicht-silence Entscheidungen
        active_decisions = [d for d in decisions
                           if d.get("payload", {}).get("action") != "silence"][-5:]
        if active_decisions:
            lines.append("**Letzte aktive Entscheidungen:**")
            for d in active_decisions:
                p = d.get("payload", {})
                ts = d.get("timestamp", 0)
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                lines.append(f"- `{ts_str}` **{p.get('action', '?')}** "
                             f"(Score: {p.get('score', '?'):.2f}) — {p.get('reason', '?')}")
            lines.append("")
    else:
        lines.append("*Keine Entscheidungen getroffen.*")
        lines.append("")

    # --- Health Alerts ---
    alerts = events["health_alerts"]
    lines.append(f"## Homeostasis ({len(alerts)} Alerts)")
    lines.append("")
    if alerts:
        alert_types = Counter(
            a.get("payload", {}).get("type", "?") for a in alerts
        )
        lines.append("| Alert-Typ | Anzahl |")
        lines.append("|-----------|--------|")
        for typ, count in alert_types.most_common():
            lines.append(f"| {typ} | {count} |")
        lines.append("")
    else:
        lines.append("*Keine Health-Alerts — System stabil.*")
        lines.append("")

    # --- Night Cycle ---
    lines.append("## Night Cycle")
    lines.append("")
    if night.get("ran"):
        lines.append(f"- **Datum:** {night.get('date', '?')}")
        dur = night.get("duration_s", 0)
        lines.append(f"- **Dauer:** {dur:.1f}s")
        steps = night.get("steps", {})
        if "episodes" in steps:
            ep = steps["episodes"]
            lines.append(f"- **Episoden:** {ep.get('count', '?')} ({ep.get('note', '')})")
        if "music" in steps:
            mu = steps["music"]
            lines.append(f"- **Music Memory:** {mu.get('processed', 0)} verarbeitet, "
                         f"{mu.get('removed', 0)} entfernt, {mu.get('remaining', 0)} verbleibend")
        if "stats" in steps:
            st = steps["stats"]
            lines.append(f"- **Tages-Events:** {st.get('events', 0)}")
    else:
        lines.append("*Night Cycle lief nicht (oder Ergebnis nicht gefunden).*")
    lines.append("")

    # --- Anomalien ---
    lines.append("## Anomalien")
    lines.append("")
    if qi["anomalies"]:
        for a in qi["anomalies"]:
            lines.append(f"- {a}")
    else:
        lines.append("*Keine Anomalien erkannt.*")
    lines.append("")

    # --- Empfehlungen ---
    lines.append("## Empfehlungen")
    lines.append("")
    if qi["recommendations"]:
        for r in qi["recommendations"]:
            lines.append(f"- {r}")
    else:
        lines.append("*Keine Empfehlungen — Qi fliesst.*")
    lines.append("")

    lines.append("---")
    lines.append("*Watcher Agent — Wu-Wei: Beobachten ohne Eingriff.*")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    target = get_target_date()
    WATCHER_DIR.mkdir(parents=True, exist_ok=True)

    # Daten sammeln
    journal = read_journalctl(target)
    events = read_event_log(target)
    night = read_night_cycle(target)
    metrics = read_system_metrics()

    # Qi bewerten
    qi = assess_qi(journal, events, night, metrics)

    # Report generieren
    report = generate_report(target, journal, events, night, metrics, qi)

    # Schreiben
    out_path = WATCHER_DIR / f"{target.isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"Watcher Report: {out_path}")
    print(f"Qi-Fluss: {qi['emoji']} {qi['score']}/100 — {qi['level']}")
    if qi["anomalies"]:
        print(f"Anomalien: {len(qi['anomalies'])}")
    if qi["recommendations"]:
        print(f"Empfehlungen: {len(qi['recommendations'])}")


if __name__ == "__main__":
    main()
