#!/usr/bin/env python3
"""
M.O.L.O.C.H. TTS Test Script
Tests different voices with various German phrases.
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path.home() / "moloch" / "core"))

import tts
import time


def main():
    print("=" * 70)
    print("M.O.L.O.C.H. TTS TEST")
    print("=" * 70)

    # Initialize TTS engine
    engine = tts.get_tts_engine()

    print(f"\nAvailable voices: {len(engine.list_voices())}")
    for i, voice in enumerate(engine.list_voices(), 1):
        print(f"  {i}. {voice}")

    print(f"\nCurrent voice: {engine.current_voice}")

    # Test phrases
    test_phrases = [
        "M.O.L.O.C.H. ist online.",
        "System läuft stabil.",
        "Temperatur bei 50 Grad.",
        "Guten Morgen, Markus.",
        "Die dunkle Seite grüßt.",
        "Alle Systeme bereit.",
        "Ich bin bereit, dir zu dienen.",
        "Hailo Beschleuniger erkannt.",
        "NVMe Speicher verfügbar.",
        "Kamera System aktiv."
    ]

    # Test current voice with all phrases
    print("\n" + "=" * 70)
    print(f"Testing voice: {engine.current_voice}")
    print("=" * 70)

    for i, phrase in enumerate(test_phrases, 1):
        print(f"\n[{i}/{len(test_phrases)}] Speaking: {phrase}")

        success = tts.speak(phrase)

        if success:
            print("  ✓ Success")
        else:
            print("  ✗ Failed")

        # Brief pause between phrases
        time.sleep(0.5)

    # Test multiple voices with one phrase
    print("\n" + "=" * 70)
    print("Testing all voices with: 'M.O.L.O.C.H. ist online.'")
    print("=" * 70)

    test_phrase = "M.O.L.O.C.H. ist online."

    for voice in engine.list_voices():
        print(f"\n[Voice: {voice}]")

        success = tts.speak(test_phrase, voice=voice)

        if success:
            print("  ✓ Success")
        else:
            print("  ✗ Failed")

        time.sleep(1)

    # Voice comparison - let M.O.L.O.C.H. speak about itself with different voices
    print("\n" + "=" * 70)
    print("M.O.L.O.C.H. introduces itself with different voices")
    print("=" * 70)

    introduction_phrases = {
        "de_DE-thorsten-high": "Ich bin M.O.L.O.C.H., dein lokaler Assistent.",
        "de_DE-thorsten-medium": "Maschinelle Organisation für Logische Operationen und Computergestützte Hilfe.",
        "de_DE-thorsten-low": "Ich lerne und wachse mit jedem Tag.",
        "de_DE-eva_k-x_low": "Ich bin hier, um dir zu helfen.",
        "de_DE-karlsson-low": "Alle meine Systeme sind online.",
        "de_DE-kerstin-low": "Ich kann verschiedene Stimmen nutzen.",
        "de_DE-pavoque-low": "Welche Stimme gefällt dir am besten?",
        "de_DE-ramona-low": "M.O.L.O.C.H. steht bereit."
    }

    for voice, phrase in introduction_phrases.items():
        if voice in engine.available_voices:
            print(f"\n[{voice}]: {phrase}")
            tts.speak(phrase, voice=voice)
            time.sleep(1.5)

    print("\n" + "=" * 70)
    print("TTS TEST COMPLETE")
    print("=" * 70)
    print("\nCheck ~/moloch/logs/tts.log for detailed logs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
