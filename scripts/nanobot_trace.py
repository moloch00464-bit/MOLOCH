#!/usr/bin/env python3
"""
Nanobot Flow Tracer — Latenz durch die komplette MOLOCH Pipeline messen.

Stationen:
  1. SHM Frame         — wann wurde der letzte Frame nach /dev/shm/moloch_frame geschrieben
  2. TAPPAS Callback   — wann kam Detection zurueck (letztes perception.* Event im Log)
  3. Event Bus Publish — wann wurde perception.* auf den Bus geschickt (= Station 2, selber Aufruf)
  4. Event Bus Deliver — wann kam Event beim Subscriber an (approximiert via Bridge-Decision)
  5. Action Bridge     — wann hat die FSM auf das Event reagiert (letztes bridge_decision)

Datenquellen: /dev/shm/moloch_frame mtime, logs/events/*.jsonl, /dev/shm/moloch_status.json
Kein Eingriff in die laufende Pipeline — nur Lesen und Messen.

Ausgabe: logs/flow_trace.json + Terminal
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Pfade ---
BASE_DIR = Path(__file__).resolve().parent.parent
SHM_FRAME = Path("/dev/shm/moloch_frame")
SHM_STATUS = Path("/dev/shm/moloch_status.json")
EVENTS_DIR = BASE_DIR / "logs" / "events"
TRACE_OUT = BASE_DIR / "logs" / "flow_trace.json"


def _lade_events_letzte_n(sekunden: float = 60.0) -> list:
    """Letzte Eintraege aus dem Event-Log lesen (alle Dateien, letzten N Sekunden)."""
    jetzt = time.time()
    grenze = jetzt - sekunden
    events = []

    # Alle Dateien nach Datum sortiert, neueste zuerst
    dateien = sorted(EVENTS_DIR.glob("events_*.jsonl"), key=lambda f: f.name, reverse=True)

    for datei in dateien[:2]:  # Max 2 Tage zurueck
        try:
            with open(datei, "r") as f:
                for zeile in f:
                    zeile = zeile.strip()
                    if not zeile:
                        continue
                    try:
                        ev = json.loads(zeile)
                        ts = ev.get("timestamp", 0)
                        if ts >= grenze:
                            events.append(ev)
                    except Exception:
                        pass
        except Exception:
            pass

    return events


def _lade_status() -> dict:
    """moloch_status.json aus Shared Memory lesen."""
    try:
        return json.loads(SHM_STATUS.read_text())
    except Exception:
        return {}


# =========================================================================
# Station-Messungen
# =========================================================================

def station1_shm_frame() -> dict:
    """Station 1: SHM Frame — wann wurde der letzte Frame geschrieben."""
    s = {
        "station": 1,
        "name": "shm_frame",
        "beschreibung": "SHM Frame geschrieben (/dev/shm/moloch_frame)",
        "ts": None,
        "fehler": None,
    }
    try:
        stat = SHM_FRAME.stat()
        s["ts"] = stat.st_mtime
        s["extra"] = {"groesse_bytes": stat.st_size}
    except Exception as e:
        s["fehler"] = str(e)
    return s


def station2_tappas_callback(events: list) -> dict:
    """Station 2: TAPPAS Callback — wann feuerte der Detection-Callback zuletzt."""
    s = {
        "station": 2,
        "name": "tappas_callback",
        "beschreibung": "TAPPAS Detection Callback (letztes perception.* Event)",
        "ts": None,
        "fehler": None,
    }
    # Letztes perception.* Event von tappas_pipeline
    treffer = [
        ev for ev in events
        if ev.get("event_type", "").startswith("perception.")
        and ev.get("source") == "tappas_pipeline"
    ]
    if not treffer:
        # Fallback: beliebige Quelle
        treffer = [ev for ev in events if ev.get("event_type", "").startswith("perception.")]

    if not treffer:
        s["fehler"] = "Keine perception.* Events in den letzten 60s"
        return s

    letztes = max(treffer, key=lambda e: e.get("timestamp", 0))
    s["ts"] = letztes["timestamp"]
    s["extra"] = {
        "event_type": letztes.get("event_type"),
        "source": letztes.get("source"),
        "payload_keys": list(letztes.get("payload", {}).keys()),
    }
    return s


def station3_event_bus_publish(station2: dict) -> dict:
    """Station 3: Event Bus Publish — wann wurde perception.* auf den Bus geschickt.

    In der Praxis: EventBus.publish() wird direkt im TAPPAS-Callback aufgerufen.
    Der Timestamp ist identisch mit Station 2 (selber Python-Stack).
    Wird trotzdem separat ausgegeben um den Pfad explizit zu machen.
    """
    s = {
        "station": 3,
        "name": "event_bus_publish",
        "beschreibung": "Event Bus Publish (perception.* auf Bus geschickt)",
        "ts": station2.get("ts"),  # Selber Timestamp wie Station 2
        "fehler": None,
        "hinweis": "Timestamp = Station 2 (publish() im TAPPAS-Callback, kein Zeitversatz)",
    }
    if station2.get("fehler"):
        s["fehler"] = "Abhaengig von Station 2: " + station2["fehler"]
        s["ts"] = None
    return s


def station4_event_bus_deliver(events: list, station2: dict) -> dict:
    """Station 4: Event Bus Deliver — wann kam Event beim Subscriber an.

    Keine dedizierte 'delivered_at' Metrik im Event-Log vorhanden.
    Approximation: Erstes action.* Event das NACH dem letzten perception.* Event kam.
    Das action.* Event wird vom ActionBridge-Subscriber nach Empfang published.
    """
    s = {
        "station": 4,
        "name": "event_bus_deliver",
        "beschreibung": "Event Bus Deliver (bei ActionBridge Subscriber angekommen)",
        "ts": None,
        "fehler": None,
        "hinweis": "Approximiert: erstes action.* Event nach dem perception.* Event",
    }

    perception_ts = station2.get("ts")
    if not perception_ts:
        s["fehler"] = "Kein Referenz-Timestamp (Station 2 fehlgeschlagen)"
        return s

    # Erstes action.* Event NACH dem perception.* Event
    folge_events = [
        ev for ev in events
        if ev.get("event_type", "").startswith("action.")
        and ev.get("timestamp", 0) >= perception_ts
    ]

    if not folge_events:
        # Fallback: letztes action.* Event ueberhaupt (Bus-Aktivitaet vorhanden?)
        alle_action = [ev for ev in events if ev.get("event_type", "").startswith("action.")]
        if alle_action:
            letztes = max(alle_action, key=lambda e: e.get("timestamp", 0))
            s["ts"] = letztes["timestamp"]
            s["fehler"] = None
            s["hinweis"] = "Kein action.* nach perception.* gefunden — letztes action.* Event verwendet"
            s["extra"] = {"event_type": letztes.get("event_type"), "source": letztes.get("source")}
        else:
            s["fehler"] = "Kein action.* Event in den letzten 60s — ActionBridge hat nicht reagiert?"
        return s

    erstes = min(folge_events, key=lambda e: e.get("timestamp", 0))
    s["ts"] = erstes["timestamp"]
    s["extra"] = {
        "event_type": erstes.get("event_type"),
        "source": erstes.get("source"),
        "delta_ms_nach_perception": round((erstes["timestamp"] - perception_ts) * 1000, 2),
    }
    return s


def station5_action_bridge(status: dict) -> dict:
    """Station 5: Action Bridge FSM — wann hat die FSM auf das letzte Event reagiert."""
    s = {
        "station": 5,
        "name": "action_bridge",
        "beschreibung": "Action Bridge FSM Reaktion (letztes bridge_decision)",
        "ts": None,
        "fehler": None,
    }

    bridge = status.get("bridge", {})
    decisions = status.get("bridge_decisions", [])

    if not bridge:
        s["fehler"] = "bridge fehlt in moloch_status.json"
        return s

    if not decisions:
        # Kein Zustandswechsel — Bridge laeuft, aber stabil (kein neues Event)
        s["fehler"] = "Keine bridge_decisions — kein Zustandswechsel seit Service-Start"
        s["hinweis"] = f"Aktueller State: {bridge.get('state', 'unknown')} (seit {bridge.get('state_age_s', '?')}s)"
        return s

    letztes = max(decisions, key=lambda d: d.get("timestamp", 0))
    s["ts"] = letztes.get("timestamp")
    s["extra"] = {
        "state": bridge.get("state"),
        "state_age_s": bridge.get("state_age_s"),
        "thought": letztes.get("thought"),
        "action": letztes.get("action"),
        "uebergang": f"{letztes.get('old_state')} → {letztes.get('new_state')}",
    }
    return s


# =========================================================================
# Latenz-Berechnung
# =========================================================================

def berechne_latenz(stationen: list) -> dict:
    """Zeitabstaende zwischen den Stationen berechnen."""
    valide = [(s["station"], s) for s in stationen if s.get("ts") is not None]

    if len(valide) < 2:
        return {
            "total_ms": None,
            "segmente": [],
            "engpass": None,
            "unterbrochene_pfade": len(stationen) - len(valide),
        }

    segmente = []
    for i in range(1, len(valide)):
        _, s_vor = valide[i - 1]
        _, s_nach = valide[i]
        delta_ms = round((s_nach["ts"] - s_vor["ts"]) * 1000, 3)
        segmente.append({
            "von": s_vor["name"],
            "nach": s_nach["name"],
            "delta_ms": delta_ms,
        })

    total_ms = round((valide[-1][1]["ts"] - valide[0][1]["ts"]) * 1000, 3)

    # Engpass: groesstes positives Delta
    pos = [seg for seg in segmente if seg["delta_ms"] > 0]
    engpass = max(pos, key=lambda s: s["delta_ms"]) if pos else None

    return {
        "total_ms": total_ms,
        "segmente": segmente,
        "engpass": engpass,
        "unterbrochene_pfade": len(stationen) - len(valide),
    }


# =========================================================================
# Hauptfunktion
# =========================================================================

def trace_ausfuehren() -> dict:
    """Alle 5 Stationen messen und Ergebnis zusammenstellen."""
    trace_ts = time.time()
    trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Datenquellen laden
    events = _lade_events_letzte_n(sekunden=120.0)
    status = _lade_status()

    # Stationen messen
    s1 = station1_shm_frame()
    s2 = station2_tappas_callback(events)
    s3 = station3_event_bus_publish(s2)
    s4 = station4_event_bus_deliver(events, s2)
    s5 = station5_action_bridge(status)

    stationen = [s1, s2, s3, s4, s5]
    latenz = berechne_latenz(stationen)

    return {
        "trace_id": trace_id,
        "trace_timestamp": trace_ts,
        "trace_timestamp_human": datetime.fromtimestamp(trace_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "stationen": stationen,
        "latenz": latenz,
    }


# =========================================================================
# Terminal-Ausgabe
# =========================================================================

def ausgabe_terminal(ergebnis: dict):
    """Formatierte Ausgabe auf Terminal."""
    breite = 65
    print()
    print("=" * breite)
    print("  MOLOCH Nanobot Flow Tracer")
    print(f"  {ergebnis['trace_timestamp_human']}  |  ID: {ergebnis['trace_id']}")
    print("=" * breite)

    print("\n  STATIONEN:")
    for s in ergebnis["stationen"]:
        nr = s["station"]
        name = s.get("beschreibung", s["name"])
        ts = s.get("ts")
        fehler = s.get("fehler")
        hinweis = s.get("hinweis")

        if fehler:
            print(f"  [{nr}] {name}")
            print(f"       FEHLER: {fehler}")
            if hinweis:
                print(f"       Hinweis: {hinweis}")
        elif ts:
            ts_human = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{nr}] {name}")
            print(f"       {ts_human}", end="")
            extra = s.get("extra", {})
            if "event_type" in extra:
                print(f"  ({extra['event_type']})", end="")
            if "uebergang" in extra:
                print(f"  [{extra['uebergang']}]", end="")
            print()
            if hinweis:
                print(f"       Hinweis: {hinweis}")
        else:
            print(f"  [{nr}] {name} — kein Timestamp")

    latenz = ergebnis["latenz"]
    print("\n  LATENZEN:")
    if latenz["segmente"]:
        for seg in latenz["segmente"]:
            delta = seg["delta_ms"]
            ist_engpass = latenz["engpass"] and seg == latenz["engpass"]
            marker = "  << ENGPASS" if ist_engpass else ""
            farbe = "\033[91m" if ist_engpass else ("\033[93m" if delta > 50 else "")
            reset = "\033[0m" if farbe else ""
            print(f"    {seg['von']:28s} → {seg['nach']:22s}  {farbe}{delta:+8.3f} ms{reset}{marker}")

        print(f"\n  TOTAL PIPELINE-LATENZ:  {latenz['total_ms']:+.3f} ms")
    else:
        print("    Nicht genug valide Stationen fuer Messung.")

    if latenz["engpass"]:
        e = latenz["engpass"]
        print(f"\n  ENGPASS:  {e['von']} → {e['nach']}  ({e['delta_ms']:.3f} ms)")

    if latenz["unterbrochene_pfade"] > 0:
        n = latenz["unterbrochene_pfade"]
        print(f"\n  UNTERBROCHENE PFADE: {n} Station(en) ohne Timestamp")

    print()
    print(f"  Gespeichert: logs/flow_trace.json")
    print("=" * breite)
    print()


# =========================================================================
# Entry Point
# =========================================================================

def main() -> dict:
    """Trace ausfuehren, ausgeben, speichern."""
    ergebnis = trace_ausfuehren()

    ausgabe_terminal(ergebnis)

    # JSON speichern
    TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRACE_OUT.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False))

    return ergebnis


if __name__ == "__main__":
    main()
