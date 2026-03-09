#!/usr/bin/env python3
"""
Test 3: FFT Live — MusicListener 5 Sekunden, dann ausgeben.
Laeuft auf dem Pi, empfaengt UDP Port 12346 (48kHz Stereo vom ESP32).

Voraussetzung: ESP32 im 48kHz-Modus (oder manuell testen via test_respeaker_udp.py)
"""
import sys
import time
sys.path.insert(0, "/home/molochzuhause/moloch")

from core.audio.music_listener import get_music_listener
from core.moloch_event_bus import get_event_bus

events_received = []

def on_bands(event):
    payload = event.get("payload", {}) if isinstance(event, dict) else {}
    events_received.append({
        "time": round(time.monotonic(), 2),
        "bass": round(payload.get("bass", 0), 3),
        "mid":  round(payload.get("mid", 0), 3),
        "high": round(payload.get("high", 0), 3),
        "energy": round(payload.get("overall_energy", 0), 3),
    })

def on_beat(event):
    payload = event.get("payload", {}) if isinstance(event, dict) else {}
    print(f"  [BEAT] strength={payload.get('strength', 0):.3f} bpm~{payload.get('bpm_estimate', 0):.0f}")

bus = get_event_bus()
bus.subscribe("music.frequency_bands", on_bands)
bus.subscribe("music.beat", on_beat)

# MusicListener starten und manuell auf 48kHz-Modus setzen
ml = get_music_listener()
ml.start()

# Manuell aktivieren (simuliert mic.mode_changed Event)
import threading
import time as _time
def _activate():
    _time.sleep(0.1)
    ml._active = True  # Direkt aktivieren fuer Test
threading.Thread(target=_activate, daemon=True).start()

print("MusicListener gestartet — 5 Sekunden empfangen...")
print("(ESP32 muss auf 48kHz-Modus sein — Port 12346)")
time.sleep(5)

ml.stop()

print(f"\n=== Ergebnis: {len(events_received)} Band-Events in 5s ===")
if events_received:
    # Erste und letzte 3 ausgeben
    for e in events_received[:3]:
        print(f"  t={e['time']} bass={e['bass']} mid={e['mid']} high={e['high']} E={e['energy']}")
    if len(events_received) > 6:
        print("  ...")
    for e in events_received[-3:]:
        print(f"  t={e['time']} bass={e['bass']} mid={e['mid']} high={e['high']} E={e['energy']}")
    avg_energy = sum(e['energy'] for e in events_received) / len(events_received)
    print(f"\nDurchschnitt Energy: {avg_energy:.3f}")
    print("PASS" if avg_energy > 0.001 else "INFO: Kein Audio-Signal empfangen")
else:
    print("INFO: Keine Events — ESP32 im 48kHz-Modus? UDP Port 12346 offen?")
