#!/usr/bin/env python3
"""
Test 4+5: Eye Visualizer — Atmen und simulierte Beat-Events.

Modus:
  --mode=breathe          Ruhiges Atmen (keine Musik)
  --mode=simulate_music   Simulierte music.beat + music.frequency_bands Events

Ausgabe: Render-State alle 33ms fuer 5 Sekunden.
"""
import sys
import time
import argparse
sys.path.insert(0, "/home/molochzuhause/moloch")

from core.ui.eye_visualizer import get_eye_visualizer
from core.moloch_event_bus import get_event_bus

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["breathe", "simulate_music"], default="breathe")
args = parser.parse_args()

ev = get_eye_visualizer()
ev.start()
ev.set_guardian_state("GUARDIAN")

bus = get_event_bus()

if args.mode == "simulate_music":
    print("Modus: simulate_music — simulierte Beat + Frequenz Events")
    import threading
    import math

    def simulate():
        t = 0.0
        while t < 5.0:
            # Frequenz-Baender (simuliert Musik mit 120 BPM Sinus)
            phase = t * 2 * math.pi * 2.0  # 2 Hz Grundfrequenz
            bass = max(0, math.sin(phase) * 0.8)
            mid  = max(0, math.sin(phase * 1.3 + 0.5) * 0.5)
            high = max(0, math.sin(phase * 2.1 + 1.0) * 0.3)
            energy = (bass + mid + high) / 3

            bus.publish(
                event_type="music.frequency_bands",
                source="test_simulator",
                payload={"bass": round(bass, 3), "mid": round(mid, 3),
                         "high": round(high, 3), "overall_energy": round(energy, 3)},
            )

            # Beat alle 500ms
            if int(t * 2) != int((t - 0.05) * 2):
                bus.publish(
                    event_type="music.beat",
                    source="test_simulator",
                    payload={"strength": bass, "bpm_estimate": 120.0},
                )
                print(f"  [SIM-BEAT] t={t:.2f}s bass={bass:.2f}")

            time.sleep(0.05)
            t += 0.05

    sim_thread = threading.Thread(target=simulate, daemon=True)
    sim_thread.start()
else:
    print("Modus: breathe — ruhiges Atmen ohne Musik")

# Render-Loop 30 FPS fuer 5 Sekunden
print("\n--- Render State (alle 500ms) ---")
start = time.monotonic()
last_print = 0.0
while time.monotonic() - start < 5.0:
    ev.tick()
    now = time.monotonic() - start
    if now - last_print >= 0.5:
        state = ev.get_render_state()
        print(f"  t={now:.1f}s iris={state['iris_radius']:.1f}px "
              f"pupil={state['pupil_radius']:.1f}px "
              f"bright={state['brightness']:.3f} "
              f"glow={state['glow_alpha']} "
              f"jitter={state['ray_jitter']:.2f} "
              f"beat={state['beat']}")
        last_print = now
    time.sleep(0.033)  # 30 FPS

ev.stop()
print("\nPASS: EyeVisualizer laeuft")
