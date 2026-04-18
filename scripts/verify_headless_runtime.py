#!/usr/bin/env python3
"""
verify_headless_runtime.py — Prüft ob der Backend-Service bei Display-Aus / GUI-Close
wirklich unabhängig weiterläuft.

Läuft N Sekunden (Default: 300), liest jede Sekunde /dev/shm/moloch_status.json
und schreibt eine CSV-Datei nach logs/verify_headless_<ISO-timestamp>.csv

Spalten: timestamp_iso, elapsed_s, fps, ram_percent, cpu_temp,
         tracking_state, person_present, face_id, status_age_ms
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Konstanten ---
STATUS_JSON = Path("/dev/shm/moloch_status.json")
LOGS_DIR    = Path(__file__).parent.parent / "logs"

# --- Globale Steuerung ---
_laufe     = True   # Signal-Handler setzt auf False
_zeilen    = []     # Gesammelte CSV-Zeilen (für Zusammenfassung und sauberes Schreiben)


def _signal_handler(sig, frame):
    """SIGINT abfangen — Zusammenfassung ausgeben, dann sauber beenden."""
    global _laufe
    _laufe = False


def status_lesen() -> dict | None:
    """
    Liest /dev/shm/moloch_status.json atomar.
    Gibt None zurück wenn Datei fehlt oder kein gültiges JSON.
    """
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def status_age_ms() -> int | None:
    """
    Gibt Alter der Status-Datei in Millisekunden zurück.
    Erkennt eingefrorenen Status-File.
    """
    try:
        mtime = STATUS_JSON.stat().st_mtime
        return int((time.time() - mtime) * 1000)
    except (FileNotFoundError, OSError):
        return None


def felder_extrahieren(status: dict | None, age_ms: int | None) -> dict:
    """
    Extrahiert die relevanten Felder aus dem Status-Dict.
    Fehlende oder fehlerhafte Felder werden als Leerstring zurückgegeben.

    Bekannte Status-JSON-Struktur (Stand 2026-04-18):
      - fps: dict mit Keys scrfd/arcface/yolov8m/total → wir nutzen 'total'
      - watchdog: dict mit cpu_temp, ram_percent
      - ptz: dict mit tracker_state
      - person_detected: bool
      - face_id: str
    """
    if status is None:
        return {
            "fps":            "",
            "ram_percent":    "",
            "cpu_temp":       "",
            "tracking_state": "",
            "person_present": "",
            "face_id":        "",
            "status_age_ms":  age_ms if age_ms is not None else "",
        }

    # FPS: entweder direkt float oder dict mit 'total'-Key
    fps_raw = status.get("fps") or status.get("pipeline_fps") or status.get("tappas_fps")
    if isinstance(fps_raw, dict):
        fps = fps_raw.get("total", "")
    elif isinstance(fps_raw, (int, float)):
        fps = float(fps_raw)
    else:
        fps = ""

    # RAM + CPU-Temp aus watchdog-Sub-Dict
    watchdog = status.get("watchdog", {}) or {}
    ram  = watchdog.get("ram_percent", status.get("ram_percent", ""))
    temp = watchdog.get("cpu_temp", status.get("cpu_temp", ""))

    # Tracking-State aus ptz-Sub-Dict
    ptz_dict = status.get("ptz", {}) or {}
    tracking = (
        ptz_dict.get("tracker_state")
        or status.get("tracking_state")
        or status.get("tracker_state")
        or ""
    )

    # Person vorhanden?
    person = status.get("person_detected", status.get("person_present", ""))

    # Face-ID des zuletzt erkannten Gesichts
    face_id = status.get("face_id", "")

    return {
        "fps":            fps,
        "ram_percent":    ram,
        "cpu_temp":       temp,
        "tracking_state": tracking,
        "person_present": person,
        "face_id":        face_id,
        "status_age_ms":  age_ms if age_ms is not None else "",
    }


def stdout_zeile(elapsed: int, felder: dict) -> None:
    """Gibt eine kurze Statuszeile auf stdout aus."""
    fps_str  = f"{felder['fps']:.1f}" if isinstance(felder['fps'], (int, float)) else str(felder['fps'] or "-")
    ram_str  = f"{felder['ram_percent']:.0f}%" if isinstance(felder['ram_percent'], (int, float)) else str(felder['ram_percent'] or "-")
    track    = str(felder['tracking_state'] or "?")
    person   = int(bool(felder['person_present'])) if felder['person_present'] != "" else 0
    age      = f"{felder['status_age_ms']}ms" if felder['status_age_ms'] != "" else "?ms"

    print(f"[{elapsed:4d}s] fps={fps_str} ram={ram_str} track={track} person={person} age={age}")


def zusammenfassung(zeilen: list[dict], csv_pfad: Path) -> None:
    """Gibt am Ende eine Zusammenfassung auf stdout aus."""
    if not zeilen:
        print("\n[Zusammenfassung] Keine Daten gesammelt.")
        return

    fps_werte = [
        float(z["fps"]) for z in zeilen
        if z["fps"] not in ("", None)
    ]
    age_werte = [
        int(z["status_age_ms"]) for z in zeilen
        if z["status_age_ms"] not in ("", None)
    ]

    print("\n" + "=" * 60)
    print("HEADLESS-RUNTIME ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Gesammelte Zeilen   : {len(zeilen)}")

    if fps_werte:
        stalls = sum(1 for v in fps_werte if v < 10)
        print(f"FPS  min/avg/max    : {min(fps_werte):.1f} / {sum(fps_werte)/len(fps_werte):.1f} / {max(fps_werte):.1f}")
        print(f"FPS-Stalls (<10)    : {stalls}")
    else:
        print("FPS                 : keine Daten")

    if age_werte:
        print(f"Status-Age max      : {max(age_werte)} ms")
        frozen = sum(1 for v in age_werte if v > 5000)
        if frozen:
            print(f"  WARNUNG: {frozen}x Status-File > 5s alt (Service haengt?)")
    else:
        print("Status-Age          : keine Daten")

    print(f"CSV gespeichert     : {csv_pfad}")
    print("=" * 60)


def main() -> None:
    global _laufe

    # --- CLI-Argumente ---
    parser = argparse.ArgumentParser(
        description="Prüft ob der MOLOCH Backend-Service headless weiterläuft."
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=300,
        help="Laufzeit in Sekunden (Default: 300)"
    )
    args = parser.parse_args()

    # --- Signal-Handler für sauberes SIGINT ---
    signal.signal(signal.SIGINT, _signal_handler)

    # --- CSV-Datei vorbereiten ---
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    iso_ts   = datetime.now().strftime("%Y%m%dT%H%M%S")
    csv_pfad = LOGS_DIR / f"verify_headless_{iso_ts}.csv"

    csv_felder = [
        "timestamp_iso",
        "elapsed_s",
        "fps",
        "ram_percent",
        "cpu_temp",
        "tracking_state",
        "person_present",
        "face_id",
        "status_age_ms",
    ]

    print(f"verify_headless_runtime.py — Laufzeit: {args.duration}s")
    print(f"CSV: {csv_pfad}")
    print("-" * 60)

    start_time = time.time()
    elapsed    = 0

    # --- Haupt-Loop ---
    while _laufe and elapsed < args.duration:
        schleife_start = time.time()

        # Status lesen
        now_iso = datetime.now(timezone.utc).isoformat()
        age     = status_age_ms()
        status  = status_lesen()
        felder  = felder_extrahieren(status, age)

        # CSV-Zeile zusammenbauen
        zeile = {
            "timestamp_iso":  now_iso,
            "elapsed_s":      elapsed,
            **felder,
        }
        _zeilen.append(zeile)

        # Stdout-Ausgabe
        stdout_zeile(elapsed, felder)

        # Genau 1 Sekunde warten (Drift-tolerant)
        vergangen = time.time() - schleife_start
        restzeit  = max(0.0, 1.0 - vergangen)
        time.sleep(restzeit)

        elapsed = int(time.time() - start_time)

    # --- CSV schreiben ---
    try:
        with open(csv_pfad, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_felder)
            writer.writeheader()
            writer.writerows(_zeilen)
    except OSError as e:
        print(f"[FEHLER] CSV konnte nicht geschrieben werden: {e}", file=sys.stderr)

    # --- Zusammenfassung ---
    zusammenfassung(_zeilen, csv_pfad)


if __name__ == "__main__":
    main()
