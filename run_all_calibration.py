#!/usr/bin/env python3
"""Alle Kalibrierungs-Phasen autonom durchlaufen.

Tempo 1 = 1 Bild/Sekunde (langsam, besser fuer NPU-Verarbeitung).
Phasen: emotions -> gender -> age -> emotions_hd -> gestures

Laeuft als IPC-Client - sendet Kommandos an den Service
und pollt den Fortschritt.
"""
import json
import os
import time
import sys

CAL_EVENT = "/dev/shm/moloch_cal_event.json"
# Tempo pro Phase: CPU-Phasen schnell, NPU-Phasen langsam
SPEED_CPU = 5   # Emotions, Gender, Age (nur CPU)
SPEED_NPU = 2   # Emotions HD, Gesten (NPU braucht Zeit)

PHASES = [
    ("emotions",    "Emotionen (FER2013, 35.887 Bilder)",    SPEED_CPU),
    ("gender",      "Gender (FairFace, 400 Bilder)",         SPEED_CPU),
    ("age",         "Alter (FairFace, 800 Bilder)",          SPEED_CPU),
    ("emotions_hd", "Emotionen HD (FairFace, 1.200 Bilder)", SPEED_NPU),
    ("gestures",    "Gesten (HaGRID, 100 Bilder)",           SPEED_NPU),
]

def send_cmd(cmd):
    seq = int(time.monotonic_ns())
    tmp = f'/tmp/moloch_cmd_{seq}.tmp'
    dst = f'/tmp/moloch_cmd_{seq}.json'
    with open(tmp, 'w') as f:
        json.dump(cmd, f)
    os.rename(tmp, dst)

def wait_for_finish(phase_name, timeout=7200):
    """Warte bis Phase fertig (finished/done/error) oder Timeout."""
    start = time.time()
    last_progress = ""
    last_ts = 0

    while time.time() - start < timeout:
        try:
            if os.path.exists(CAL_EVENT):
                with open(CAL_EVENT) as f:
                    ev = json.load(f)
                ts = ev.get("ts", 0)
                if ts != last_ts:
                    last_ts = ts
                    event = ev.get("event", "")
                    data = ev.get("data", {})

                    if event == "calibration_result":
                        prog = data.get("progress", (0, 0))
                        detected = data.get("detected", "?")
                        correct = data.get("correct", False)
                        mark = "OK" if correct else "X"
                        pct = prog[0] / prog[1] * 100 if prog[1] > 0 else 0
                        msg = f"  [{mark}] {prog[0]}/{prog[1]} ({pct:.0f}%) -> {detected}"
                        if msg != last_progress:
                            print(msg, flush=True)
                            last_progress = msg

                    elif event == "calibration_status":
                        status = data.get("status", "")
                        if status in ("finished", "done"):
                            rate = data.get("rate", 0)
                            total = data.get("total", 0)
                            correct = data.get("correct", 0)
                            dur = data.get("duration", 0)
                            print(f"\n  FERTIG: {correct}/{total} ({rate:.1%}) in {dur:.0f}s")
                            return True
                        elif status == "error":
                            print(f"\n  FEHLER: {data.get('message', '?')}")
                            return False
                        elif status == "stopped":
                            print("\n  GESTOPPT")
                            return False
        except Exception:
            pass

        time.sleep(0.5)

    print(f"\n  TIMEOUT nach {timeout}s!")
    return False


def main():
    print("=" * 60)
    print("M.O.L.O.C.H. BILDERBUCH - Autonome Kalibrierung")
    print(f"Tempo: CPU={SPEED_CPU}/s, NPU={SPEED_NPU}/s")
    print("=" * 60)

    # Altes Event loeschen
    try:
        if os.path.exists(CAL_EVENT):
            os.remove(CAL_EVENT)
    except Exception:
        pass

    results = {}

    for phase_id, phase_desc, speed in PHASES:
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_desc} (Tempo {speed}/s)")
        print(f"{'='*60}")

        # Phase starten
        send_cmd({
            "action": "calibration_start",
            "phase": phase_id,
            "speed": speed,
        })

        time.sleep(1)  # Kurz warten

        # Auf Ergebnis warten
        ok = wait_for_finish(phase_id)
        results[phase_id] = ok

        if not ok:
            print(f"  Phase {phase_id} fehlgeschlagen - weiter zur naechsten")

        # Kurze Pause zwischen Phasen
        time.sleep(3)

    # Zusammenfassung
    print(f"\n{'='*60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*60}")
    for phase_id, phase_desc, _ in PHASES:
        status = "OK" if results.get(phase_id) else "FEHLER"
        print(f"  {phase_desc}: {status}")

    # Ergebnisdatei anzeigen
    results_path = os.path.expanduser("~/moloch/data/calibration_results.json")
    if os.path.exists(results_path):
        print(f"\nDetaillierte Ergebnisse: {results_path}")
        with open(results_path) as f:
            data = json.load(f)
        for phase, r in data.get("phases", {}).items():
            rate = r.get("rate", 0)
            total = r.get("total", 0)
            correct = r.get("correct", 0)
            dur = r.get("duration_seconds", 0)
            print(f"  {phase}: {correct}/{total} ({rate:.1%}) - {dur:.0f}s")

    print(f"\n{'='*60}")
    print("KALIBRIERUNG ABGESCHLOSSEN")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
