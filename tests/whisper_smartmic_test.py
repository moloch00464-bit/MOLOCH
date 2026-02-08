#!/usr/bin/env python3
"""
Whisper SmartMic Test
=====================

Testet Spracherkennung mit SmartMic Bluetooth Mikrofon.

Features:
- SmartMic Bluetooth Verbindung
- Echtzeit Audio-Aufnahme
- Whisper Speech-to-Text
- Optional: M.O.L.O.C.H. antwortet via TTS

Usage:
  python whisper_smartmic_test.py --test          # Schnelltest
  python whisper_smartmic_test.py --live          # Live-Erkennung
  python whisper_smartmic_test.py --respond       # Mit TTS Antwort

Author: M.O.L.O.C.H. System
"""

import subprocess
import sys
import time
import tempfile
import wave
import io
import argparse
import logging
from pathlib import Path

# Check dependencies
def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    # Check for whisper
    whisper_available = False
    whisper_type = None

    try:
        import whisper
        whisper_available = True
        whisper_type = "openai-whisper"
    except ImportError:
        try:
            import faster_whisper
            whisper_available = True
            whisper_type = "faster-whisper"
        except ImportError:
            missing.append("whisper (openai-whisper or faster-whisper)")

    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall with:")
        print("  pip install numpy openai-whisper")
        print("  # or for faster inference:")
        print("  pip install numpy faster-whisper")
        return False, None

    return True, whisper_type


class WhisperSmartMicTest:
    """Test Whisper STT with SmartMic."""

    # SmartMic config
    SMARTMIC_MAC = "54:B7:E5:AA:3B:8E"
    SMARTMIC_NAME = "SmartMic"

    def __init__(self, whisper_type: str = "openai-whisper", model_size: str = "base"):
        self.whisper_type = whisper_type
        self.model_size = model_size
        self.model = None
        self.logger = logging.getLogger("WhisperTest")

    def load_model(self):
        """Load Whisper model."""
        print(f"Loading Whisper model ({self.whisper_type}, {self.model_size})...")

        if self.whisper_type == "openai-whisper":
            import whisper
            self.model = whisper.load_model(self.model_size)
        else:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, compute_type="int8")

        print("Model loaded!")

    def connect_smartmic(self) -> bool:
        """Connect to SmartMic."""
        print(f"\nConnecting to {self.SMARTMIC_NAME}...")

        # Check if already connected
        result = subprocess.run(
            ["bluetoothctl", "info", self.SMARTMIC_MAC],
            capture_output=True, text=True
        )

        if "Connected: yes" in result.stdout:
            print("Already connected!")
            return True

        # Connect
        result = subprocess.run(
            ["bluetoothctl", "connect", self.SMARTMIC_MAC],
            capture_output=True, text=True, timeout=10
        )

        if "successful" in result.stdout.lower():
            print("Connected!")
            time.sleep(2)  # Wait for PipeWire
            return True

        print(f"Connection failed: {result.stdout}")
        return False

    def set_headset_profile(self) -> bool:
        """Set SmartMic to headset profile for microphone access."""
        print("Setting headset profile (mSBC)...")

        # Get device ID
        result = subprocess.run(["wpctl", "status"], capture_output=True, text=True)
        device_id = None

        for line in result.stdout.split("\n"):
            if self.SMARTMIC_NAME in line and "bluez5" in line:
                parts = line.strip().split(".")
                if len(parts) >= 1:
                    num = parts[0].strip().replace("*", "").strip()
                    if num.isdigit():
                        device_id = int(num)
                        break

        if device_id:
            subprocess.run(["wpctl", "set-profile", str(device_id), "headset-head-unit"])
            time.sleep(1)
            print(f"Profile set (device {device_id})")
            return True

        print("Device not found in PipeWire")
        return False

    def record_audio(self, duration: float = 5.0) -> bytes:
        """Record audio from SmartMic."""
        print(f"\nRecording for {duration} seconds...")
        print(">>> SPEAK NOW <<<")

        source_name = f"bluez_input.{self.SMARTMIC_MAC.replace(':', '_')}"

        # Record to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            cmd = [
                "pw-record",
                "--target", source_name,
                "--format", "s16",
                "--rate", "16000",
                "--channels", "1",
                temp_path
            ]

            # Start recording
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(duration)
            process.terminate()
            process.wait()

            print("Recording complete!")

            # Read audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()

            Path(temp_path).unlink()
            return audio_data

        except Exception as e:
            print(f"Recording error: {e}")
            return b""

    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio with Whisper."""
        if not audio_data:
            return ""

        print("\nTranscribing...")

        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            if self.whisper_type == "openai-whisper":
                result = self.model.transcribe(
                    temp_path,
                    language="de",
                    fp16=False
                )
                text = result["text"].strip()
            else:
                segments, info = self.model.transcribe(
                    temp_path,
                    language="de",
                    beam_size=5
                )
                text = " ".join([s.text for s in segments]).strip()

            Path(temp_path).unlink()
            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def speak_response(self, text: str):
        """Speak response via SmartMic speaker (TTS)."""
        print(f"\nM.O.L.O.C.H. says: {text}")

        sink_name = f"bluez_output.{self.SMARTMIC_MAC.replace(':', '_')}.1"

        # Try espeak-ng
        cmd = f'espeak-ng -v de --stdout "{text}" | pw-play --target {sink_name} -'

        try:
            subprocess.run(cmd, shell=True, timeout=30)
        except Exception as e:
            print(f"TTS error: {e}")

    def run_quick_test(self):
        """Run a quick test."""
        print("=" * 60)
        print("WHISPER SMARTMIC QUICK TEST")
        print("=" * 60)

        if not self.connect_smartmic():
            return False

        self.set_headset_profile()
        self.load_model()

        # Record and transcribe
        audio = self.record_audio(5.0)
        text = self.transcribe(audio)

        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        if text:
            print(f"Erkannt: \"{text}\"")
        else:
            print("Keine Sprache erkannt")

        return True

    def run_live_mode(self, respond: bool = False):
        """Run continuous live recognition."""
        print("=" * 60)
        print("WHISPER SMARTMIC LIVE MODE")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        print()

        if not self.connect_smartmic():
            return

        self.set_headset_profile()
        self.load_model()

        try:
            while True:
                print("\n" + "-" * 40)
                audio = self.record_audio(4.0)
                text = self.transcribe(audio)

                if text:
                    print(f"\n>>> Erkannt: \"{text}\"")

                    if respond:
                        # Simple response
                        if "hallo" in text.lower():
                            self.speak_response("Hallo! Ich bin M.O.L.O.C.H.")
                        elif "wie geht" in text.lower():
                            self.speak_response("Mir geht es gut, danke der Nachfrage!")
                        elif "zeit" in text.lower() or "uhrzeit" in text.lower():
                            import datetime
                            now = datetime.datetime.now().strftime("%H:%M")
                            self.speak_response(f"Es ist {now} Uhr.")
                        else:
                            self.speak_response(f"Du hast gesagt: {text}")
                else:
                    print("(keine Sprache erkannt)")

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Whisper SmartMic Test")
    parser.add_argument("--test", action="store_true", help="Quick test (5s recording)")
    parser.add_argument("--live", action="store_true", help="Live recognition mode")
    parser.add_argument("--respond", action="store_true", help="Live mode with TTS responses")
    parser.add_argument("--model", type=str, default="base",
                        help="Whisper model (tiny/base/small/medium/large)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Check dependencies
    ok, whisper_type = check_dependencies()
    if not ok:
        sys.exit(1)

    print(f"Using: {whisper_type}")

    tester = WhisperSmartMicTest(whisper_type=whisper_type, model_size=args.model)

    if args.test:
        tester.run_quick_test()
    elif args.live or args.respond:
        tester.run_live_mode(respond=args.respond)
    else:
        # Default: quick test
        tester.run_quick_test()


if __name__ == "__main__":
    main()
