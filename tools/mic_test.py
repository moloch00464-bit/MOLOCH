#!/usr/bin/env python3
"""
M.O.L.O.C.H. Mikrofon-Test
==========================
Testet PS3 Eye Mikrofon mit Gain-Boost und NPU-Whisper.

Usage:
    python3 tools/mic_test.py [--gain GAIN] [--duration SEC]

    --gain: Software gain multiplier (default: 10)
    --duration: Recording duration in seconds (default: 5)
"""

import subprocess
import sys
import os
import tempfile
import argparse
import struct
import math

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PS3 Eye device
PS3_EYE_DEVICE = "plughw:3,0"
SAMPLE_RATE = 16000


def record_audio(duration: int, output_path: str) -> bool:
    """Record from PS3 Eye."""
    print(f"\n🎤 Aufnahme startet - {duration} Sekunden SPRECHEN!")
    print("=" * 40)

    try:
        result = subprocess.run([
            "arecord",
            "-D", PS3_EYE_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", "1",
            "-d", str(duration),
            output_path
        ], capture_output=True, text=True, timeout=duration + 5)

        if result.returncode == 0:
            print("✅ Aufnahme fertig")
            return True
        else:
            print(f"❌ Fehler: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False


def analyze_audio(wav_path: str) -> dict:
    """Analyze audio levels."""
    try:
        result = subprocess.run(
            ["sox", wav_path, "-n", "stat"],
            capture_output=True, text=True
        )
        stats = result.stderr

        info = {}
        for line in stats.split('\n'):
            if 'RMS' in line and 'amplitude' in line:
                try:
                    info['rms'] = float(line.split()[-1])
                except:
                    pass
            elif 'Maximum amplitude' in line:
                try:
                    info['peak'] = float(line.split()[-1])
                except:
                    pass
            elif 'Length' in line:
                try:
                    info['duration'] = float(line.split()[-1])
                except:
                    pass

        return info
    except Exception as e:
        print(f"Analyse-Fehler: {e}")
        return {}


def apply_gain(input_path: str, output_path: str, gain: float) -> bool:
    """Apply software gain and normalization."""
    try:
        # Apply gain and normalize to -3dB
        subprocess.run([
            "sox", input_path, output_path,
            "gain", str(gain),  # Apply gain in dB
            "norm", "-3"        # Normalize to -3dB peak
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Gain-Fehler: {e}")
        return False


def transcribe(wav_path: str) -> str:
    """Transcribe with NPU Whisper."""
    try:
        from core.speech.hailo_whisper import get_whisper
        whisper = get_whisper()
        print(f"🧠 Backend: {whisper.backend}")
        result = whisper.transcribe(wav_path, language='de')
        return result or ""
    except Exception as e:
        print(f"❌ Whisper-Fehler: {e}")
        return ""


def rms_to_db(rms: float) -> float:
    """Convert RMS to dB."""
    if rms <= 0:
        return -100.0
    return 20 * math.log10(rms)


def main():
    parser = argparse.ArgumentParser(description="M.O.L.O.C.H. Mikrofon-Test")
    parser.add_argument("--gain", type=float, default=20, help="Software gain in dB (default: 20)")
    parser.add_argument("--duration", type=int, default=5, help="Aufnahme-Dauer in Sekunden")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("   M.O.L.O.C.H. MIKROFON-TEST (PS3 Eye)")
    print("=" * 50)
    print(f"   Gain: +{args.gain}dB | Dauer: {args.duration}s")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw.wav")
        processed_path = os.path.join(tmpdir, "processed.wav")

        # 1. Record
        if not record_audio(args.duration, raw_path):
            return 1

        # 2. Analyze raw audio
        print("\n📊 Audio-Analyse (RAW):")
        raw_stats = analyze_audio(raw_path)
        if raw_stats:
            rms_db = rms_to_db(raw_stats.get('rms', 0))
            peak_db = rms_to_db(raw_stats.get('peak', 0))
            print(f"   RMS:  {rms_db:+.1f} dB")
            print(f"   Peak: {peak_db:+.1f} dB")

            if rms_db < -50:
                print("   ⚠️  Sehr leise - näher ans Mikrofon!")
            elif rms_db < -30:
                print("   📢 OK - Software-Boost wird angewendet")
            else:
                print("   ✅ Guter Pegel!")

        # 3. Apply gain
        print(f"\n🔊 Wende +{args.gain}dB Gain an...")
        if not apply_gain(raw_path, processed_path, args.gain):
            return 1

        # 4. Analyze processed audio
        print("\n📊 Audio-Analyse (PROCESSED):")
        proc_stats = analyze_audio(processed_path)
        if proc_stats:
            rms_db = rms_to_db(proc_stats.get('rms', 0))
            peak_db = rms_to_db(proc_stats.get('peak', 0))
            print(f"   RMS:  {rms_db:+.1f} dB")
            print(f"   Peak: {peak_db:+.1f} dB")

        # 5. Transcribe
        print("\n🎯 NPU-Whisper Transkription:")
        print("-" * 40)
        result = transcribe(processed_path)
        if result:
            print(f'   "{result}"')
            print("-" * 40)
            print("   ✅ Erkennung erfolgreich!")
        else:
            print("   ❌ Keine Sprache erkannt")
            print("   Tipps: Lauter sprechen, näher ans Mikrofon")

        print("\n" + "=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
