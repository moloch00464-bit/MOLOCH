#!/usr/bin/env python3
"""
Tracking-Diagnose: 30 Sekunden alle 500ms loggen.

Liest NUR aus /dev/shm/moloch_status.json und /tmp/moloch_face_state.json.
KEIN NPU-Zugriff, KEINE Systemänderungen.

Output: scripts/tracking_log.csv

HINWEIS: detection.center_x/center_y ist NICHT im Status-JSON enthalten.
Diese Daten leben nur intern im autonomous_tracker. Stattdessen werden
tracker_state, smoothed_person, pan/tilt und face_id geloggt.
"""

import json
import csv
import time
import os
import sys

STATUS_PATH = "/dev/shm/moloch_status.json"
FACE_STATE_PATH = "/tmp/moloch_face_state.json"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking_log.csv")

DURATION_SEC = 30
INTERVAL_SEC = 0.5

CSV_FIELDS = [
    "timestamp",
    "elapsed_s",
    "current_pan",
    "current_tilt",
    "tracker_state",
    "ptz_stage",
    "ptz_velocity",
    "ptz_restless_score",
    "smoothed_person",
    "smoothed_face_id",
    "smoothed_distance",
    "approaching",
    "leaving",
    "presence_duration",
    "absence_duration",
    "npu_stage",
    "active_models",
    "face_name",
    "face_similarity",
    "person_count",
    "tracking_moves",
    "search_moves",
]


def read_json(path):
    """JSON lesen, bei Fehler leeres Dict."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def sample():
    """Ein Sample aus Status + Face-State zusammenbauen."""
    status = read_json(STATUS_PATH)
    face = read_json(FACE_STATE_PATH)

    ptz = status.get("ptz", {})
    core = status.get("core", {})
    trends = core.get("trends", {})

    return {
        "current_pan": ptz.get("current_pan", ""),
        "current_tilt": ptz.get("current_tilt", ""),
        "tracker_state": ptz.get("tracker_state", ""),
        "ptz_stage": ptz.get("ptz_stage", ""),
        "ptz_velocity": ptz.get("ptz_velocity", ""),
        "ptz_restless_score": ptz.get("ptz_restless_score", ""),
        "smoothed_person": trends.get("smoothed_person", ""),
        "smoothed_face_id": trends.get("smoothed_face_id", ""),
        "smoothed_distance": trends.get("smoothed_distance", ""),
        "approaching": trends.get("approaching", ""),
        "leaving": trends.get("leaving", ""),
        "presence_duration": trends.get("presence_duration", ""),
        "absence_duration": trends.get("absence_duration", ""),
        "npu_stage": status.get("npu_stage", ""),
        "active_models": ",".join(status.get("active_models", [])),
        "face_name": face.get("name", ""),
        "face_similarity": face.get("similarity", ""),
        "person_count": face.get("person_count", ""),
        "tracking_moves": ptz.get("tracking_moves", ""),
        "search_moves": ptz.get("search_moves", ""),
    }


def analyze(rows):
    """Zusammenfassung drucken: Pan-Korrelation, State-Verteilung."""
    if not rows:
        print("Keine Daten gesammelt.")
        return

    print(f"\n{'='*60}")
    print(f"  TRACKING DIAGNOSE — {len(rows)} Samples in {DURATION_SEC}s")
    print(f"{'='*60}")

    # --- Pan/Tilt Range ---
    pans = [r["current_pan"] for r in rows if r["current_pan"] != ""]
    tilts = [r["current_tilt"] for r in rows if r["current_tilt"] != ""]

    if pans:
        print(f"\n  Pan:  {min(pans):+.1f}° .. {max(pans):+.1f}°  (Delta: {max(pans)-min(pans):.1f}°)")
    if tilts:
        print(f"  Tilt: {min(tilts):+.1f}° .. {max(tilts):+.1f}°  (Delta: {max(tilts)-min(tilts):.1f}°)")

    # --- Tracker State Verteilung ---
    states = {}
    for r in rows:
        s = r.get("tracker_state", "?")
        states[s] = states.get(s, 0) + 1
    print(f"\n  Tracker States:")
    for state, count in sorted(states.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        print(f"    {state:20s}  {count:3d}x  ({pct:.0f}%)")

    # --- Person-Erkennung ---
    person_count = sum(1 for r in rows if r.get("smoothed_person") is True)
    no_person = len(rows) - person_count
    print(f"\n  Person erkannt:  {person_count}x ({person_count/len(rows)*100:.0f}%)")
    print(f"  Keine Person:    {no_person}x ({no_person/len(rows)*100:.0f}%)")

    # --- Face IDs ---
    face_ids = {}
    for r in rows:
        fid = r.get("smoothed_face_id") or r.get("face_name", "")
        if fid:
            face_ids[fid] = face_ids.get(fid, 0) + 1
    if face_ids:
        print(f"\n  Erkannte Gesichter:")
        for fid, count in sorted(face_ids.items(), key=lambda x: -x[1]):
            print(f"    {fid:20s}  {count:3d}x")

    # --- Pan-Bewegungsanalyse ---
    if len(pans) >= 4:
        print(f"\n  Pan-Bewegungsanalyse:")
        # Pan-Richtung ueber Zeit
        pan_deltas = []
        for i in range(1, len(pans)):
            pan_deltas.append(pans[i] - pans[i-1])

        moves_left = sum(1 for d in pan_deltas if d < -0.5)  # negative = links
        moves_right = sum(1 for d in pan_deltas if d > 0.5)  # positive = rechts
        stationary = len(pan_deltas) - moves_left - moves_right

        print(f"    Bewegungen links:  {moves_left}x")
        print(f"    Bewegungen rechts: {moves_right}x")
        print(f"    Stationaer:        {stationary}x")

        avg_delta = sum(pan_deltas) / len(pan_deltas)
        print(f"    Mittlerer Pan-Drift: {avg_delta:+.2f}°/Sample")

        # Korrelation: Person + Pan
        print(f"\n  Pan-Verhalten bei Person:")
        pan_with_person = []
        pan_without_person = []
        for i, r in enumerate(rows):
            if r.get("current_pan") == "":
                continue
            if r.get("smoothed_person") is True:
                pan_with_person.append(r["current_pan"])
            else:
                pan_without_person.append(r["current_pan"])

        if pan_with_person:
            print(f"    Bei Person:  Pan {min(pan_with_person):+.1f}° .. {max(pan_with_person):+.1f}°")
        else:
            print(f"    Bei Person:  (keine Samples)")
        if pan_without_person:
            print(f"    Ohne Person: Pan {min(pan_without_person):+.1f}° .. {max(pan_without_person):+.1f}°")
        else:
            print(f"    Ohne Person: (keine Samples)")

    # --- Velocity ---
    velocities = [r["ptz_velocity"] for r in rows if r.get("ptz_velocity") not in ("", None)]
    if velocities:
        print(f"\n  PTZ Velocity:  {min(velocities):.1f} .. {max(velocities):.1f}  (Avg: {sum(velocities)/len(velocities):.1f})")

    # --- FAZIT ---
    print(f"\n{'='*60}")
    if len(pans) >= 4:
        # Trend: Geht Pan in eine Richtung?
        first_half = pans[:len(pans)//2]
        second_half = pans[len(pans)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        drift = avg_second - avg_first

        if abs(drift) < 1.0:
            print(f"  FAZIT: Pan war stabil (Drift {drift:+.1f}°)")
        elif drift > 0:
            print(f"  FAZIT: Pan driftet nach RECHTS ({drift:+.1f}°)")
        else:
            print(f"  FAZIT: Pan driftet nach LINKS ({drift:+.1f}°)")

        if person_count > 0 and len(pans) >= 4:
            # War Pan-Bewegung korreliert mit Person?
            print(f"  Person in {person_count/len(rows)*100:.0f}% der Samples erkannt")
    else:
        print(f"  FAZIT: Zu wenig Pan-Daten fuer Analyse")

    print(f"{'='*60}")
    print(f"\n  CSV: {OUTPUT_PATH}")


def main():
    # Pruefe ob Status-Datei existiert
    if not os.path.exists(STATUS_PATH):
        print(f"FEHLER: {STATUS_PATH} nicht gefunden!")
        print("Laeuft der moloch Service?")
        sys.exit(1)

    print(f"Tracking-Diagnose startet...")
    print(f"  Dauer:    {DURATION_SEC}s")
    print(f"  Interval: {INTERVAL_SEC*1000:.0f}ms")
    print(f"  Output:   {OUTPUT_PATH}")
    print(f"  Quelle:   {STATUS_PATH}")
    print()

    rows = []
    start = time.time()

    with open(OUTPUT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        sample_nr = 0
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= DURATION_SEC:
                break

            data = sample()
            data["timestamp"] = f"{now:.3f}"
            data["elapsed_s"] = f"{elapsed:.1f}"

            writer.writerow(data)
            rows.append(data)
            sample_nr += 1

            # Fortschritt
            bar_len = 40
            progress = elapsed / DURATION_SEC
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            state = data.get("tracker_state", "?")
            pan = data.get("current_pan", "?")
            person = "👤" if data.get("smoothed_person") is True else "  "
            sys.stdout.write(f"\r  [{bar}] {elapsed:.0f}s/{DURATION_SEC}s  {state:12s}  Pan:{pan}° {person}")
            sys.stdout.flush()

            # Naechstes Sample-Timing
            next_time = start + (sample_nr * INTERVAL_SEC)
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    print()  # Newline nach Fortschrittsbalken
    print(f"\n{len(rows)} Samples geschrieben nach {OUTPUT_PATH}")

    # Fuer Analyse: numerische Werte konvertieren
    for r in rows:
        for key in ("current_pan", "current_tilt", "ptz_velocity", "ptz_restless_score",
                     "presence_duration", "absence_duration", "face_similarity"):
            try:
                r[key] = float(r[key])
            except (ValueError, TypeError):
                pass
        # smoothed_person String -> Bool
        if r.get("smoothed_person") == "True":
            r["smoothed_person"] = True
        elif r.get("smoothed_person") == "False":
            r["smoothed_person"] = False

    analyze(rows)


if __name__ == "__main__":
    main()
