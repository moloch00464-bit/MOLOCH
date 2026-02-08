#!/usr/bin/env python3
"""
AudioManager - SmartMic Bluetooth Audio Controller
===================================================

Vollständige Audio-Verwaltung für M.O.L.O.C.H. mit SmartMic Bluetooth Mikrofon.

Features:
- Mikrofon-Input für Spracherkennung (Speech-to-Text)
- Lautsprecher-Output für Sprachausgabe (Text-to-Speech)
- Profil-Wechsel zwischen A2DP (HiFi) und Headset (Bidirektional)
- Automatische Geräteerkennung und Verbindung
- Lautstärkeregelung

Profile:
- A2DP: Hohe Audioqualität für Wiedergabe, kein echtes Mikrofon
- Headset (mSBC): Bidirektional - Mikrofon UND Lautsprecher, 16kHz
- Headset (CVSD): Bidirektional - niedrigere Qualität, 8kHz

Author: M.O.L.O.C.H. System
"""

import subprocess
import logging
import time
import threading
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path


class AudioProfile(Enum):
    """Bluetooth Audio Profile."""
    OFF = "off"
    A2DP = "a2dp-sink"                    # HiFi Output only
    A2DP_XQ = "a2dp-sink-sbc_xq"          # HiFi Output (better SBC)
    HEADSET_MSBC = "headset-head-unit"     # Bidirectional 16kHz
    HEADSET_CVSD = "headset-head-unit-cvsd"  # Bidirectional 8kHz


class AudioMode(Enum):
    """Audio operation mode."""
    IDLE = auto()
    LISTENING = auto()      # Mikrofon aktiv
    SPEAKING = auto()       # Lautsprecher aktiv
    DUPLEX = auto()         # Beides aktiv (Headset mode)


@dataclass
class AudioDevice:
    """Audio device information."""
    name: str
    mac_address: str
    connected: bool = False
    battery: int = 0
    profile: AudioProfile = AudioProfile.OFF
    pipewire_id: int = 0
    source_id: int = 0      # Microphone
    sink_id: int = 0        # Speaker


@dataclass
class AudioConfig:
    """Audio configuration."""
    # SmartMic settings
    smartmic_mac: str = "54:B7:E5:AA:3B:8E"
    smartmic_name: str = "SmartMic"

    # Default profile
    default_profile: AudioProfile = AudioProfile.HEADSET_MSBC

    # Volume settings (0.0 - 1.0)
    mic_volume: float = 1.0
    speaker_volume: float = 0.8

    # Auto-connect
    auto_connect: bool = True
    auto_switch_profile: bool = True

    # Speech settings
    enable_tts_output: bool = True      # M.O.L.O.C.H. kann sprechen
    enable_stt_input: bool = True       # Spracherkennung aktiv

    # Fallback
    use_camera_mic_fallback: bool = False  # RTSP Kamera als Fallback


class AudioManager:
    """
    Complete audio management for M.O.L.O.C.H.

    Handles SmartMic Bluetooth microphone/speaker with all profiles.
    """

    def __init__(self, config: Optional[AudioConfig] = None, log_level: int = logging.INFO):
        """Initialize AudioManager."""
        self.config = config or AudioConfig()

        # Logging
        self.logger = logging.getLogger("AudioManager")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)

        # State
        self.mode = AudioMode.IDLE
        self.smartmic: Optional[AudioDevice] = None
        self._connected = False

        # Callbacks
        self.on_audio_received: Optional[Callable[[bytes], None]] = None
        self.on_connection_change: Optional[Callable[[bool], None]] = None

        # Threading
        self._lock = threading.Lock()
        self._recording = False
        self._record_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to SmartMic and initialize audio."""
        self.logger.info(f"Connecting to {self.config.smartmic_name}...")

        # Check if already paired
        result = self._run_cmd(["bluetoothctl", "devices"])
        if self.config.smartmic_mac not in result:
            self.logger.error(f"SmartMic not paired: {self.config.smartmic_mac}")
            return False

        # Connect Bluetooth
        result = self._run_cmd(["bluetoothctl", "connect", self.config.smartmic_mac])
        if "successful" not in result.lower() and "already connected" not in result.lower():
            self.logger.error(f"Connection failed: {result}")
            return False

        time.sleep(2)  # Wait for PipeWire to register device

        # Get device info
        self._update_device_info()

        if self.smartmic and self.smartmic.connected:
            self._connected = True
            self.logger.info(f"SmartMic connected (Battery: {self.smartmic.battery}%)")

            # Set default profile
            if self.config.auto_switch_profile:
                self.set_profile(self.config.default_profile)

            if self.on_connection_change:
                self.on_connection_change(True)

            return True

        return False

    def disconnect(self):
        """Disconnect SmartMic."""
        self.stop_recording()
        self._run_cmd(["bluetoothctl", "disconnect", self.config.smartmic_mac])
        self._connected = False
        self.logger.info("SmartMic disconnected")

        if self.on_connection_change:
            self.on_connection_change(False)

    def _update_device_info(self):
        """Update SmartMic device information from system."""
        # Get Bluetooth info
        result = self._run_cmd(["bluetoothctl", "info", self.config.smartmic_mac])

        connected = "Connected: yes" in result
        battery = 0
        if "Battery Percentage:" in result:
            try:
                battery = int(result.split("Battery Percentage:")[1].split("(")[1].split(")")[0])
            except:
                pass

        # Get PipeWire device ID
        pw_result = self._run_cmd(["wpctl", "status"])
        device_id = 0
        source_id = 0
        sink_id = 0

        for line in pw_result.split("\n"):
            if self.config.smartmic_name in line:
                try:
                    # Extract ID from line like "79. SmartMic [bluez5]"
                    parts = line.strip().split(".")
                    if len(parts) >= 1:
                        num = parts[0].strip().replace("*", "").strip()
                        if num.isdigit():
                            if "bluez5" in line:
                                device_id = int(num)
                            elif "Audio/Source" in line or "bluez_input" in line:
                                source_id = int(num)
                            elif "Audio/Sink" in line or "bluez_output" in line:
                                sink_id = int(num)
                except:
                    pass

        # Get current profile
        profile = AudioProfile.OFF
        if device_id > 0:
            inspect = self._run_cmd(["wpctl", "inspect", str(device_id)])
            if "a2dp-sink-sbc_xq" in inspect:
                profile = AudioProfile.A2DP_XQ
            elif "a2dp-sink" in inspect:
                profile = AudioProfile.A2DP
            elif "headset-head-unit-cvsd" in inspect:
                profile = AudioProfile.HEADSET_CVSD
            elif "headset-head-unit" in inspect:
                profile = AudioProfile.HEADSET_MSBC

        self.smartmic = AudioDevice(
            name=self.config.smartmic_name,
            mac_address=self.config.smartmic_mac,
            connected=connected,
            battery=battery,
            profile=profile,
            pipewire_id=device_id,
            source_id=source_id,
            sink_id=sink_id
        )

    @property
    def is_connected(self) -> bool:
        """Check if SmartMic is connected."""
        return self._connected and self.smartmic is not None and self.smartmic.connected

    # -------------------------------------------------------------------------
    # Profile Management
    # -------------------------------------------------------------------------

    def set_profile(self, profile: AudioProfile) -> bool:
        """
        Set SmartMic audio profile.

        Profiles:
        - A2DP: High quality output only (no mic)
        - HEADSET_MSBC: Bidirectional 16kHz (mic + speaker)
        - HEADSET_CVSD: Bidirectional 8kHz (lower quality)
        """
        if not self.smartmic or self.smartmic.pipewire_id == 0:
            self._update_device_info()
            if not self.smartmic or self.smartmic.pipewire_id == 0:
                self.logger.error("SmartMic device not found in PipeWire")
                return False

        self.logger.info(f"Switching to profile: {profile.value}")

        result = self._run_cmd([
            "wpctl", "set-profile",
            str(self.smartmic.pipewire_id),
            profile.value
        ])

        time.sleep(1)
        self._update_device_info()

        if self.smartmic.profile == profile:
            self.logger.info(f"Profile set to {profile.value}")
            return True
        else:
            self.logger.warning(f"Profile change may have failed")
            return False

    def get_profile(self) -> AudioProfile:
        """Get current profile."""
        self._update_device_info()
        return self.smartmic.profile if self.smartmic else AudioProfile.OFF

    def enable_microphone(self) -> bool:
        """Enable microphone by switching to headset profile."""
        return self.set_profile(AudioProfile.HEADSET_MSBC)

    def enable_hifi_output(self) -> bool:
        """Enable high-fidelity output (A2DP, no mic)."""
        return self.set_profile(AudioProfile.A2DP_XQ)

    def enable_bidirectional(self) -> bool:
        """Enable bidirectional audio (mic + speaker)."""
        return self.set_profile(AudioProfile.HEADSET_MSBC)

    # -------------------------------------------------------------------------
    # Volume Control
    # -------------------------------------------------------------------------

    def set_mic_volume(self, volume: float) -> bool:
        """Set microphone volume (0.0 - 1.0)."""
        volume = max(0.0, min(1.0, volume))
        self._update_device_info()

        if self.smartmic and self.smartmic.source_id > 0:
            self._run_cmd(["wpctl", "set-volume", str(self.smartmic.source_id), f"{volume:.2f}"])
            self.config.mic_volume = volume
            self.logger.debug(f"Mic volume set to {volume*100:.0f}%")
            return True
        return False

    def set_speaker_volume(self, volume: float) -> bool:
        """Set speaker volume (0.0 - 1.0)."""
        volume = max(0.0, min(1.0, volume))
        self._update_device_info()

        if self.smartmic and self.smartmic.sink_id > 0:
            self._run_cmd(["wpctl", "set-volume", str(self.smartmic.sink_id), f"{volume:.2f}"])
            self.config.speaker_volume = volume
            self.logger.debug(f"Speaker volume set to {volume*100:.0f}%")
            return True
        return False

    def mute_mic(self, mute: bool = True) -> bool:
        """Mute/unmute microphone."""
        if self.smartmic and self.smartmic.source_id > 0:
            action = "1" if mute else "0"
            self._run_cmd(["wpctl", "set-mute", str(self.smartmic.source_id), action])
            return True
        return False

    def mute_speaker(self, mute: bool = True) -> bool:
        """Mute/unmute speaker."""
        if self.smartmic and self.smartmic.sink_id > 0:
            action = "1" if mute else "0"
            self._run_cmd(["wpctl", "set-mute", str(self.smartmic.sink_id), action])
            return True
        return False

    # -------------------------------------------------------------------------
    # Audio Recording (for Speech-to-Text)
    # -------------------------------------------------------------------------

    def start_recording(self, callback: Optional[Callable[[bytes], None]] = None) -> bool:
        """
        Start recording from SmartMic microphone.

        Audio is captured and passed to callback for processing (e.g., STT).
        Requires headset profile for real microphone access.
        """
        if not self.is_connected:
            self.logger.error("SmartMic not connected")
            return False

        # Ensure headset profile for mic access
        if self.smartmic.profile not in [AudioProfile.HEADSET_MSBC, AudioProfile.HEADSET_CVSD]:
            self.logger.info("Switching to headset profile for mic access")
            self.set_profile(AudioProfile.HEADSET_MSBC)
            time.sleep(1)

        self._update_device_info()

        if callback:
            self.on_audio_received = callback

        self._recording = True
        self.mode = AudioMode.LISTENING

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

        self.logger.info("Recording started")
        return True

    def stop_recording(self):
        """Stop recording."""
        self._recording = False
        if self._record_thread:
            self._record_thread.join(timeout=2)
            self._record_thread = None
        self.mode = AudioMode.IDLE
        self.logger.info("Recording stopped")

    def _record_loop(self):
        """Recording thread loop."""
        # Use pw-record for PipeWire audio capture
        try:
            self._update_device_info()

            # Find the SmartMic source node name
            source_name = f"bluez_input.{self.config.smartmic_mac.replace(':', '_')}"

            cmd = [
                "pw-record",
                "--target", source_name,
                "--format", "s16",
                "--rate", "16000",
                "--channels", "1",
                "-"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            chunk_size = 1600  # 100ms at 16kHz mono 16-bit

            while self._recording and process.poll() is None:
                data = process.stdout.read(chunk_size)
                if data and self.on_audio_received:
                    self.on_audio_received(data)

            process.terminate()

        except Exception as e:
            self.logger.error(f"Recording error: {e}")
            self._recording = False

    # -------------------------------------------------------------------------
    # Audio Playback (for Text-to-Speech)
    # -------------------------------------------------------------------------

    def play_audio(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """
        Play audio through SmartMic speaker.

        For M.O.L.O.C.H. to speak to the user.
        Requires headset profile or A2DP for speaker access.
        """
        if not self.is_connected:
            self.logger.error("SmartMic not connected")
            return False

        self._update_device_info()

        # Ensure we have speaker access
        if self.smartmic.profile == AudioProfile.OFF:
            self.set_profile(AudioProfile.HEADSET_MSBC)
            time.sleep(1)

        self.mode = AudioMode.SPEAKING

        try:
            sink_name = f"bluez_output.{self.config.smartmic_mac.replace(':', '_')}.1"

            cmd = [
                "pw-play",
                "--target", sink_name,
                "--format", "s16",
                "--rate", str(sample_rate),
                "--channels", "1",
                "-"
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            process.communicate(input=audio_data, timeout=30)

            self.mode = AudioMode.IDLE
            return True

        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            self.mode = AudioMode.IDLE
            return False

    def play_file(self, file_path: str) -> bool:
        """Play audio file through SmartMic speaker."""
        if not Path(file_path).exists():
            self.logger.error(f"File not found: {file_path}")
            return False

        self._update_device_info()

        if self.smartmic.profile == AudioProfile.OFF:
            self.set_profile(AudioProfile.HEADSET_MSBC)
            time.sleep(1)

        self.mode = AudioMode.SPEAKING

        try:
            sink_name = f"bluez_output.{self.config.smartmic_mac.replace(':', '_')}.1"

            # Use ffplay or pw-play
            cmd = ["pw-play", "--target", sink_name, file_path]
            subprocess.run(cmd, capture_output=True, timeout=60)

            self.mode = AudioMode.IDLE
            return True

        except Exception as e:
            self.logger.error(f"File playback error: {e}")
            self.mode = AudioMode.IDLE
            return False

    def speak(self, text: str, lang: str = "de") -> bool:
        """
        Text-to-Speech: M.O.L.O.C.H. spricht zum Benutzer.

        Uses espeak-ng or piper for TTS, output to SmartMic speaker.
        """
        if not self.config.enable_tts_output:
            self.logger.warning("TTS output disabled in config")
            return False

        if not self.is_connected:
            self.logger.error("SmartMic not connected")
            return False

        self._update_device_info()

        # Ensure speaker profile
        if self.smartmic.sink_id == 0:
            self.set_profile(AudioProfile.HEADSET_MSBC)
            time.sleep(1)
            self._update_device_info()

        self.mode = AudioMode.SPEAKING
        self.logger.info(f"Speaking: {text[:50]}...")

        try:
            sink_name = f"bluez_output.{self.config.smartmic_mac.replace(':', '_')}.1"

            # Try piper first (better quality), fall back to espeak
            tts_cmd = None

            # Check for piper
            if self._cmd_exists("piper"):
                tts_cmd = f'echo "{text}" | piper --model de_DE-thorsten-high --output-raw | pw-play --target {sink_name} --format s16 --rate 22050 --channels 1 -'
            # Fall back to espeak-ng
            elif self._cmd_exists("espeak-ng"):
                tts_cmd = f'espeak-ng -v {lang} --stdout "{text}" | pw-play --target {sink_name} -'
            # Fall back to espeak
            elif self._cmd_exists("espeak"):
                tts_cmd = f'espeak -v {lang} --stdout "{text}" | pw-play --target {sink_name} -'
            else:
                self.logger.error("No TTS engine found (piper, espeak-ng, espeak)")
                self.mode = AudioMode.IDLE
                return False

            subprocess.run(tts_cmd, shell=True, timeout=60)

            self.mode = AudioMode.IDLE
            return True

        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            self.mode = AudioMode.IDLE
            return False

    # -------------------------------------------------------------------------
    # Status and Info
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get complete audio status."""
        self._update_device_info()

        return {
            "connected": self.is_connected,
            "mode": self.mode.name,
            "device": {
                "name": self.smartmic.name if self.smartmic else None,
                "mac": self.smartmic.mac_address if self.smartmic else None,
                "battery": self.smartmic.battery if self.smartmic else 0,
                "profile": self.smartmic.profile.value if self.smartmic else "off",
                "pipewire_id": self.smartmic.pipewire_id if self.smartmic else 0,
                "source_id": self.smartmic.source_id if self.smartmic else 0,
                "sink_id": self.smartmic.sink_id if self.smartmic else 0,
            },
            "config": {
                "mic_volume": self.config.mic_volume,
                "speaker_volume": self.config.speaker_volume,
                "tts_enabled": self.config.enable_tts_output,
                "stt_enabled": self.config.enable_stt_input,
            },
            "capabilities": {
                "microphone": self.smartmic.profile in [AudioProfile.HEADSET_MSBC, AudioProfile.HEADSET_CVSD] if self.smartmic else False,
                "speaker": self.smartmic.profile != AudioProfile.OFF if self.smartmic else False,
                "bidirectional": self.smartmic.profile in [AudioProfile.HEADSET_MSBC, AudioProfile.HEADSET_CVSD] if self.smartmic else False,
            }
        }

    def get_available_profiles(self) -> List[Dict[str, Any]]:
        """Get list of available profiles with descriptions."""
        return [
            {
                "id": AudioProfile.A2DP.value,
                "name": "A2DP HiFi",
                "description": "Hohe Audioqualität für Wiedergabe",
                "microphone": False,
                "speaker": True,
                "quality": "Hoch (SBC)"
            },
            {
                "id": AudioProfile.A2DP_XQ.value,
                "name": "A2DP HiFi+",
                "description": "Beste Audioqualität (SBC-XQ)",
                "microphone": False,
                "speaker": True,
                "quality": "Sehr hoch (SBC-XQ)"
            },
            {
                "id": AudioProfile.HEADSET_MSBC.value,
                "name": "Headset (mSBC)",
                "description": "Bidirektional - Mikrofon & Lautsprecher",
                "microphone": True,
                "speaker": True,
                "quality": "Gut (16kHz mSBC)"
            },
            {
                "id": AudioProfile.HEADSET_CVSD.value,
                "name": "Headset (CVSD)",
                "description": "Bidirektional - Telefonie-Qualität",
                "microphone": True,
                "speaker": True,
                "quality": "Niedrig (8kHz CVSD)"
            },
        ]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _run_cmd(self, cmd: List[str]) -> str:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout + result.stderr
        except Exception as e:
            self.logger.error(f"Command failed: {cmd} - {e}")
            return ""

    def _cmd_exists(self, cmd: str) -> bool:
        """Check if command exists."""
        result = subprocess.run(["which", cmd], capture_output=True)
        return result.returncode == 0


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartMic Audio Manager Test")
    parser.add_argument("--connect", action="store_true", help="Connect to SmartMic")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--profile", type=str, help="Set profile (a2dp/headset)")
    parser.add_argument("--speak", type=str, help="Speak text via TTS")
    parser.add_argument("--record", type=int, help="Record for N seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    manager = AudioManager(log_level=logging.DEBUG)

    if args.connect or args.status or args.profile or args.speak or args.record:
        if manager.connect():
            print("\nSmartMic connected!")
        else:
            print("\nConnection failed!")
            exit(1)

    if args.status:
        status = manager.get_status()
        print("\n=== AUDIO STATUS ===")
        print(f"Connected: {status['connected']}")
        print(f"Mode: {status['mode']}")
        print(f"Device: {status['device']['name']}")
        print(f"Battery: {status['device']['battery']}%")
        print(f"Profile: {status['device']['profile']}")
        print(f"Microphone: {'Yes' if status['capabilities']['microphone'] else 'No'}")
        print(f"Speaker: {'Yes' if status['capabilities']['speaker'] else 'No'}")

    if args.profile:
        if args.profile.lower() == "a2dp":
            manager.set_profile(AudioProfile.A2DP_XQ)
        elif args.profile.lower() == "headset":
            manager.set_profile(AudioProfile.HEADSET_MSBC)
        print(f"Profile set to: {manager.get_profile().value}")

    if args.speak:
        print(f"Speaking: {args.speak}")
        manager.speak(args.speak)

    if args.record:
        print(f"Recording for {args.record} seconds...")

        audio_data = []
        def on_audio(data):
            audio_data.append(data)

        manager.start_recording(on_audio)
        time.sleep(args.record)
        manager.stop_recording()

        print(f"Recorded {len(audio_data)} chunks ({len(b''.join(audio_data))} bytes)")

    print("\nDone!")
