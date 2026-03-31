"""
M.O.L.O.C.H. TTS Module
Text-to-Speech using Piper for local, offline voice synthesis.
M.O.L.O.C.H. can choose its own voice.
"""

import os
import re
import socket
import struct
import subprocess
import threading
import wave
import logging
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
import json

# Setup logging
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "tts.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Voice models directory
MODELS_DIR = Path.home() / "moloch" / "models" / "voices"
PIPER_BIN = Path.home() / ".local" / "bin" / "piper"

# TTS settings for clear speech
PITCH_SHIFT = 0      # 0=normal, 300=kobold (higher can cause distortion)
LENGTH_SCALE = 1.0   # 1.0 = normal speed, keine kuenstlichen Pausen
TMP_DIR = Path("/tmp")

# Audio-Sink Routing
# Leer = PipeWire Default (HDMI)
SPEAKER_SINK = ""  # Leer = HDMI (PipeWire Default)

# ReSpeaker Wireless Speaker — sendet TTS-Audio parallel via UDP an ESP32
# ESP32 spielt es via I2S TX → XMOS → 3,5mm Klinke ab
RESPEAKER_IP   = "10.42.0.2"
RESPEAKER_PORT = 12347        # UDP_SPEAKER_PORT in config.h
RESPEAKER_ENABLED = True      # False = nur HDMI


class VoiceModel:
    """Represents a Piper voice model."""

    def __init__(self, name: str, path: Path, config: Dict):
        self.name = name
        self.path = path
        self.config = config
        self.quality = config.get("quality", "unknown")
        self.language = config.get("language", {}).get("code", "unknown")
        self.speaker = config.get("speaker", "unknown")

    def __repr__(self):
        return f"VoiceModel({self.name}, quality={self.quality})"


class TTSEngine:
    """Text-to-Speech Engine using Piper."""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.piper_bin = PIPER_BIN
        self.available_voices: Dict[str, VoiceModel] = {}
        self.current_voice: Optional[str] = None
        self.available = False

        # Verify Piper is installed
        if not self.piper_bin.exists():
            logger.warning(f"Piper not found at {self.piper_bin} - TTS disabled")
            return

        # Load available voices
        self._load_voices()

        # Select default voice (Thorsten high quality)
        if "de_DE-thorsten-high" in self.available_voices:
            self.current_voice = "de_DE-thorsten-high"
        elif self.available_voices:
            self.current_voice = list(self.available_voices.keys())[0]
        else:
            logger.warning("No voice models found - TTS disabled")
            return

        self.available = True
        logger.info(f"TTS Engine initialized with {len(self.available_voices)} voices")
        logger.info(f"Current voice: {self.current_voice}")

    def _load_voices(self):
        """Load all available voice models from the models directory."""
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return

        # Find all .onnx files
        onnx_files = list(self.models_dir.glob("*.onnx"))

        for onnx_file in onnx_files:
            # Load corresponding JSON config
            json_file = onnx_file.with_suffix(".onnx.json")

            if not json_file.exists():
                logger.warning(f"Config file not found for {onnx_file.name}")
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Extract voice name from filename (without .onnx)
                voice_name = onnx_file.stem

                voice_model = VoiceModel(voice_name, onnx_file, config)
                self.available_voices[voice_name] = voice_model

                logger.info(f"Loaded voice: {voice_name} (quality: {voice_model.quality})")

            except Exception as e:
                logger.error(f"Failed to load voice {onnx_file.name}: {e}")

    def list_voices(self) -> List[str]:
        """Return list of available voice names."""
        return list(self.available_voices.keys())

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into speakable sentence chunks."""
        parts = re.split(r'(?<=[.!?;:])\s+', text.strip())
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if result and len(result[-1]) < 30:
                result[-1] += " " + part
            else:
                result.append(part)
        return result if result else [text]

    def _synthesize_raw(self, text: str, voice_model) -> bytes:
        """Run Piper and return raw PCM audio bytes."""
        cmd = [
            str(self.piper_bin),
            "--model", str(voice_model.path),
            "--length-scale", str(LENGTH_SCALE),
            "--output-raw"
        ]
        result = subprocess.run(
            cmd, input=text.encode('utf-8'),
            capture_output=True, check=True)
        return result.stdout

    def _build_pw_cmd(self, sample_rate: int) -> list:
        """Baut pw-cat Kommando mit optionalem Sink-Target."""
        cmd = ["pw-cat", "-p", "--raw", "--rate", str(sample_rate),
               "--channels", "1", "--format", "s16"]
        if SPEAKER_SINK:
            cmd += ["--target", SPEAKER_SINK]
        cmd.append("-")
        return cmd

    def _send_to_respeaker(self, raw_audio_22k_mono: bytes):
        """Sendet TTS-Audio parallel via UDP an ESP32 (ReSpeaker 3,5mm Klinke).

        Konvertiert 22050Hz Mono S16_LE → 48000Hz Stereo S16_LE via sox,
        dann UDP-Chunks a 1920 Bytes (960 Stereo-Samples) mit 4B Seq-Header.
        Laeuft in eigenem Thread damit HDMI-Ausgabe nicht blockiert wird.
        """
        if not RESPEAKER_ENABLED:
            return

        def _send():
            try:
                # Resample 22050Hz Mono → 48000Hz Stereo via sox
                result = subprocess.run(
                    ["sox", "-t", "raw", "-r", "22050", "-e", "signed",
                     "-b", "16", "-c", "1", "-",
                     "-t", "raw", "-r", "48000", "-c", "2", "-"],
                    input=raw_audio_22k_mono,
                    capture_output=True, timeout=30
                )
                stereo_48k = result.stdout
                if not stereo_48k:
                    return

                # UDP senden: [4B seq][1920B Audio]
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                chunk_size = 1920
                seq = 0
                for off in range(0, len(stereo_48k), chunk_size):
                    chunk = stereo_48k[off:off + chunk_size]
                    packet = struct.pack("<I", seq) + chunk
                    sock.sendto(packet, (RESPEAKER_IP, RESPEAKER_PORT))
                    seq += 1
                    # Timing: 1920 Bytes = 960 Stereo-Samples @ 48kHz = 20ms
                    # Kein Sleep noetig — ESP32 Ring-Buffer puffert
                sock.close()
                logger.info(f"[TTS] ReSpeaker: {seq} UDP-Chunks gesendet")
            except Exception as e:
                logger.warning(f"[TTS] ReSpeaker UDP fehlgeschlagen: {e}")

        threading.Thread(target=_send, daemon=True, name="TTS-ReSpeaker").start()

    def _play_raw(self, raw_audio: bytes, sample_rate: int):
        """Play raw PCM via HDMI (pw-cat) + parallel an ReSpeaker senden."""
        self._send_to_respeaker(raw_audio)
        subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True)

    def set_speed(self, speed: float):
        """TTS Geschwindigkeit aendern. 0.8=schnell, 1.0=normal, 1.2=langsam."""
        global LENGTH_SCALE
        speed = max(0.6, min(1.5, speed))
        LENGTH_SCALE = speed
        logger.info(f"[TTS] Speed: {speed}")

    def set_voice(self, voice_name: str) -> bool:
        """
        Set the current voice.

        Args:
            voice_name: Name of the voice model to use

        Returns:
            True if successful, False otherwise
        """
        if voice_name not in self.available_voices:
            logger.error(f"Voice '{voice_name}' not found. Available: {self.list_voices()}")
            return False

        self.current_voice = voice_name
        logger.info(f"Voice changed to: {voice_name}")
        return True

    def speak(self, text: str, voice: Optional[str] = None, output_file: Optional[Path] = None, **kwargs) -> bool:
        """
        Convert text to speech and play it via direct pipe streaming.

        Uses Unix pipe: piper --output-raw | aplay
        Piper outputs PCM incrementally, aplay plays as data arrives.
        No temp files, no gaps, truly continuous audio.

        Args:
            text: Text to speak
            voice: Optional voice to use (overrides current_voice)
            output_file: Optional path to save audio file (if None, plays directly)

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            logger.debug("TTS not available - skipping speak()")
            return False

        voice_to_use = voice if voice else self.current_voice

        if voice_to_use not in self.available_voices:
            logger.error(f"Voice '{voice_to_use}' not available")
            return False

        voice_model = self.available_voices[voice_to_use]
        sample_rate = voice_model.config.get("audio", {}).get("sample_rate", 22050)

        logger.info(f"Speaking with voice '{voice_to_use}': {text[:50]}...")

        try:
            # File output: full synthesis (no streaming)
            if output_file:
                raw_audio = self._synthesize_raw(text, voice_model)
                if not raw_audio:
                    logger.error("Piper generated no audio data")
                    return False
                self._save_wav(raw_audio, output_file, voice_model.config)
                logger.info(f"Audio saved to {output_file}")
                return True

            # Pitch shift needs temp file path (fallback)
            if PITCH_SHIFT != 0:
                raw_audio = self._synthesize_raw(text, voice_model)
                if raw_audio:
                    self._play_audio(raw_audio, voice_model.config)
                return bool(raw_audio)

            # Direct pipe streaming: piper | pw-cat (PipeWire, raw PCM)
            # Piper outputs raw PCM incrementally, pw-cat spielt via PipeWire
            piper_cmd = [
                str(self.piper_bin),
                "--model", str(voice_model.path),
                "--length-scale", str(LENGTH_SCALE),
                "--output-raw"
            ]

            logger.info(f"[TTS] Piper → Buffer → HDMI+ReSpeaker")

            # Piper komplett in RAM puffern (kein underrun, ReSpeaker parallel)
            piper_proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            raw_audio, _ = piper_proc.communicate(input=text.encode('utf-8'))

            if not raw_audio:
                return False

            # Parallel: ReSpeaker (non-blocking Thread)
            self._send_to_respeaker(raw_audio)

            # HDMI via pw-cat (blockierend bis fertig)
            subprocess.run(self._build_pw_cmd(sample_rate),
                           input=raw_audio, stderr=subprocess.DEVNULL)

            logger.info("[TTS] Ausgabe fertig")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Piper failed: {e.stderr.decode('utf-8', errors='ignore')}")
            return False
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return False

    def _save_wav(self, raw_audio: bytes, output_file: Path, config: Dict):
        """Save raw audio as WAV file."""
        sample_rate = config.get("audio", {}).get("sample_rate", 22050)

        with wave.open(str(output_file), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)

    def _play_audio(self, raw_audio: bytes, config: Dict):
        """
        Play audio using buffered approach for smooth Bluetooth playback.

        Pipeline: raw audio → WAV → pitch shift (sox) → play (mpv)
        """
        sample_rate = config.get("audio", {}).get("sample_rate", 22050)

        # Temp files for buffered playback
        wav_file = TMP_DIR / "moloch_tts.wav"
        pitched_file = TMP_DIR / "moloch_tts_pitched.wav"

        try:
            # Step 1: Save raw audio as WAV (buffered, not streaming)
            with wave.open(str(wav_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_audio)

            # Step 2: Apply pitch shift with sox for younger kobold voice
            if PITCH_SHIFT != 0:
                sox_cmd = [
                    "sox",
                    str(wav_file),
                    str(pitched_file),
                    "pitch", str(PITCH_SHIFT)
                ]
                subprocess.run(sox_cmd, check=True, capture_output=True)
                play_file = pitched_file
            else:
                play_file = wav_file

            # Step 3: Play with mpv (best for Bluetooth)
            mpv_cmd = [
                "mpv",
                "--no-video",
                "--no-terminal",
                "--really-quiet",
                str(play_file)
            ]
            subprocess.run(mpv_cmd, check=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"Audio playback failed: {e}")
            # Fallback zu pw-cat wenn mpv/sox fehlschlaegt
            try:
                subprocess.run(self._build_pw_cmd(sample_rate), input=raw_audio, check=True)
            except Exception:
                raise
        finally:
            # Cleanup temp files
            try:
                wav_file.unlink(missing_ok=True)
                pitched_file.unlink(missing_ok=True)
            except Exception:
                pass


# Global TTS engine instance
_tts_engine: Optional[TTSEngine] = None


def get_tts_engine() -> Optional[TTSEngine]:
    """Get or create the global TTS engine instance."""
    global _tts_engine
    if _tts_engine is None:
        try:
            _tts_engine = TTSEngine()
        except Exception as e:
            logger.error(f"Failed to create TTS engine: {e}")
            _tts_engine = TTSEngine.__new__(TTSEngine)
            _tts_engine.available = False
            _tts_engine.available_voices = {}
            _tts_engine.current_voice = None
    return _tts_engine


def speak(text: str, voice: Optional[str] = None) -> bool:
    """
    Convenience function to speak text using the global TTS engine.

    Args:
        text: Text to speak
        voice: Optional voice name to use

    Returns:
        True if successful, False otherwise
    """
    engine = get_tts_engine()
    return engine.speak(text, voice=voice)


def list_voices() -> List[str]:
    """List all available voices."""
    engine = get_tts_engine()
    return engine.list_voices()


def set_voice(voice_name: str) -> bool:
    """Set the current voice."""
    engine = get_tts_engine()
    return engine.set_voice(voice_name)


if __name__ == "__main__":
    # Quick test
    print("M.O.L.O.C.H. TTS Module")
    print("=" * 50)

    engine = get_tts_engine()

    print(f"\nAvailable voices ({len(engine.list_voices())}):")
    for voice in engine.list_voices():
        print(f"  - {voice}")

    print(f"\nCurrent voice: {engine.current_voice}")
    print("\nTesting speech...")

    speak("M.O.L.O.C.H. ist online.")
