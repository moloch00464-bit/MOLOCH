#!/usr/bin/env python3
"""
Whisper Mikrofon-Test fuer ReSpeaker Lite.

Nimmt ueber ReSpeaker auf, transkribiert mit Whisper (NPU oder CPU).
Fuer schnelle Tests ohne das komplette Push-to-Talk GUI.

Usage:
    python3 scripts/whisper_mic_test.py              # 5s Aufnahme
    python3 scripts/whisper_mic_test.py --duration 10 # 10s Aufnahme
    python3 scripts/whisper_mic_test.py --loop        # Endlosschleife
"""

import os
import sys
import time
import signal
import subprocess
import argparse

sys.path.insert(0, os.path.expanduser("~/moloch"))

RESPEAKER_NODE = "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo"
RATE = 16000
TEMP_WAV = "/tmp/moloch_mic_test.wav"


def record(duration):
    """Nimm Audio auf via pw-record."""
    print(f"Aufnahme ({duration}s)... Sprich jetzt!")
    cmd = [
        "pw-record",
        "--target", RESPEAKER_NODE,
        "--channels", "1",
        "--rate", str(RATE),
        TEMP_WAV
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(duration)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=3)
    time.sleep(0.2)

    size = os.path.getsize(TEMP_WAV) if os.path.exists(TEMP_WAV) else 0
    print(f"Aufnahme fertig: {size} Bytes ({size/RATE/2:.1f}s Audio)")
    return size > 1000


def transcribe():
    """Transkribiere mit Whisper."""
    print("Transkribiere...")
    t0 = time.perf_counter()

    from core.speech.hailo_whisper import get_whisper
    whisper = get_whisper()
    text = whisper.transcribe(TEMP_WAV, language="de")

    dt = time.perf_counter() - t0
    print(f"Backend: {whisper.backend}")
    print(f"Zeit: {dt:.1f}s")

    if text:
        print(f"Text: {text}")
    else:
        print("Kein Text erkannt (Stille?)")
    return text


def main():
    parser = argparse.ArgumentParser(description="Whisper Mikrofon-Test (ReSpeaker Lite)")
    parser.add_argument("--duration", type=int, default=5, help="Aufnahmedauer in Sekunden")
    parser.add_argument("--loop", action="store_true", help="Endlosschleife")
    parser.add_argument("--record-only", action="store_true", help="Nur aufnehmen, nicht transkribieren")
    args = parser.parse_args()

    # Check ob ReSpeaker erreichbar
    check = subprocess.run(
        ["pw-record", "--target", RESPEAKER_NODE, "--channels", "1", "--rate", str(RATE), "/dev/null"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=2
    )
    # pw-record startet immer, also einfach pruefen ob Datei kommt

    print("=" * 50)
    print(" M.O.L.O.C.H. Whisper Mikrofon-Test")
    print(f" Device: ReSpeaker Lite ({RATE} Hz)")
    print("=" * 50)

    while True:
        print()
        if record(args.duration):
            if not args.record_only:
                transcribe()
        else:
            print("FEHLER: Aufnahme fehlgeschlagen!")

        if not args.loop:
            break

        input("\n[Enter] fuer naechste Aufnahme, Ctrl+C zum Beenden...")


if __name__ == "__main__":
    main()
