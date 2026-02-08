#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Whisper Speech-to-Text
===================================

NPU-accelerated Whisper using Hailo-10H.
Falls back to CPU if NPU unavailable.

Usage:
    whisper = get_whisper()
    text = whisper.transcribe("/path/to/audio.wav", language="de")
"""

import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Import Hailo resource manager
try:
    from core.hardware.hailo_manager import get_hailo_manager, HailoConsumer
    HAILO_MANAGER_AVAILABLE = True
except ImportError:
    HAILO_MANAGER_AVAILABLE = False
    logger.warning("HailoManager not available - running without resource management")

# Singleton instance
_whisper_instance = None


class MolochWhisper:
    """
    Hailo NPU-accelerated Whisper transcription.

    Primary: Hailo-10H NPU with Whisper-Base
    Fallback: CPU with faster-whisper small model
    """

    def __init__(self, lazy_npu: bool = True):
        """
        Initialize Whisper.

        Args:
            lazy_npu: If True, don't claim NPU until first transcribe() call.
                     This allows GstHailoDetector to use NPU for detection.
        """
        self.backend = "none"
        self._npu_processor = None
        self._vdevice = None
        self._cpu_model = None
        self._npu_initialized = False
        self._lazy_npu = lazy_npu

        if lazy_npu:
            # Lazy loading: don't claim NPU yet, let detection use it first
            logger.info("Whisper: NPU will be loaded lazily on first transcribe")
            self.backend = "lazy-npu"
        else:
            # Immediate loading (old behavior)
            if not self._init_npu():
                logger.warning("NPU init failed, falling back to CPU")
                self._init_cpu()

    def _init_npu(self) -> bool:
        """Initialize Hailo NPU-based Whisper."""
        try:
            from hailo_platform import VDevice
            from hailo_platform.genai import Speech2Text

            # Import hailo-apps utilities
            import sys
            sys.path.insert(0, str(Path.home() / "hailo-apps"))
            from hailo_apps.python.core.common.core import resolve_hef_path
            from hailo_apps.python.core.common.defines import HAILO10H_ARCH, WHISPER_CHAT_APP, SHARED_VDEVICE_GROUP_ID

            logger.info("Initializing Hailo NPU Whisper...")

            # Create SHARED VDevice (allows multiple users of NPU)
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self._vdevice = VDevice(params)

            # Resolve Whisper HEF path
            hef_path = resolve_hef_path(
                hef_path=None,
                app_name=WHISPER_CHAT_APP,
                arch=HAILO10H_ARCH
            )

            if hef_path is None:
                logger.error("Whisper HEF not found. Run: hailo-download-resources --group whisper_chat")
                return False

            logger.info(f"Loading Whisper model: {hef_path}")

            # Create Speech2Text processor
            self._npu_processor = Speech2Text(self._vdevice, str(hef_path))

            self.backend = "npu-whisper-base"
            logger.info("Hailo NPU Whisper initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"Hailo imports not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize NPU Whisper: {e}")
            return False

    def _init_cpu(self) -> bool:
        """Initialize CPU-based faster_whisper as fallback."""
        try:
            from faster_whisper import WhisperModel

            # Use "medium" model for better German recognition
            # (larger than "small" but much better accuracy for German)
            logger.info("Loading CPU Whisper-medium as fallback (better German)...")

            self._cpu_model = WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8"
            )

            self.backend = "cpu-medium"
            logger.info("CPU Whisper-medium loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"faster_whisper not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize CPU Whisper: {e}")
            return False

    def _load_wav_as_numpy(self, audio_path: str) -> Optional[np.ndarray]:
        """Load WAV file and convert to float32 numpy array."""
        try:
            with wave.open(audio_path, 'rb') as wf:
                # Get audio parameters
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()

                # Read raw audio data
                raw_data = wf.readframes(n_frames)

            # Convert to numpy array based on sample width
            if sample_width == 2:  # 16-bit (s16)
                audio = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio = np.frombuffer(raw_data, dtype=np.int32)
            else:
                logger.error(f"Unsupported sample width: {sample_width}")
                return None

            # Convert to mono if stereo
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Convert to float32 normalized [-1.0, 1.0]
            audio = audio.astype(np.float32)
            if sample_width == 2:
                audio /= 32768.0
            elif sample_width == 4:
                audio /= 2147483648.0

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple resampling (for production, use librosa or scipy)
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sample_rate))

            logger.debug(f"Loaded audio: {len(audio)} samples, {len(audio)/16000:.2f}s")
            return audio

        except Exception as e:
            logger.error(f"Failed to load WAV file: {e}")
            return None

    def transcribe(self, audio_path: str, language: str = "de",
                   timeout_ms: int = 15000, npu_already_acquired: bool = False) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to WAV audio file
            language: Language code (de, en, etc.)
            timeout_ms: Timeout in milliseconds
            npu_already_acquired: If True, skip NPU acquire/release (caller handles it)

        Returns:
            Transcribed text string
        """
        # === HAILO MANAGER: Acquire NPU for voice ===
        hailo_acquired = False
        if npu_already_acquired:
            # Caller (push_to_talk) already acquired NPU before recording
            logger.info("[Whisper] NPU already acquired by caller - skipping acquire")
            hailo_acquired = False  # Don't release in finally!
        elif HAILO_MANAGER_AVAILABLE:
            manager = get_hailo_manager()
            if manager.acquire_for_voice(timeout=10.0):
                hailo_acquired = True
                logger.info("[Whisper] Hailo NPU acquired via manager")
            else:
                logger.warning("[Whisper] Failed to acquire Hailo - using CPU fallback")
                if not self._cpu_model:
                    self._init_cpu()
                if self._cpu_model:
                    return self._transcribe_cpu(audio_path, language)
                logger.error("No CPU fallback available!")
                return ""

        try:
            # Lazy-load NPU once and keep it alive (DO NOT release between calls!)
            if self._lazy_npu and not self._npu_initialized:
                logger.info("Whisper: Lazy-loading NPU for first transcription...")
                if self._init_npu():
                    self._npu_initialized = True
                else:
                    logger.warning("NPU not available, falling back to CPU")
                    self._init_cpu()
                    self._npu_initialized = True

            # Try NPU first
            if self._npu_processor:
                return self._transcribe_npu(audio_path, language, timeout_ms)

            # Fallback to CPU
            if self._cpu_model:
                return self._transcribe_cpu(audio_path, language)

            logger.error("No Whisper backend available!")
            return ""

        finally:
            # === HAILO MANAGER: Release NPU (will auto-restart vision) ===
            # Only release if WE acquired it (not if caller did)
            if hailo_acquired and HAILO_MANAGER_AVAILABLE:
                try:
                    manager = get_hailo_manager()
                    manager.release_voice(restart_vision=True)
                    logger.info("[Whisper] Hailo NPU released via manager")
                except Exception as e:
                    logger.error(f"[Whisper] Error releasing via manager: {e}")

    def _transcribe_npu(self, audio_path: str, language: str, timeout_ms: int) -> str:
        """Transcribe using Hailo NPU."""
        try:
            from hailo_platform.genai import Speech2TextTask

            # Load audio as numpy array
            audio_data = self._load_wav_as_numpy(audio_path)
            if audio_data is None:
                return ""

            logger.info(f"NPU transcribing {len(audio_data)/16000:.1f}s audio...")

            # Run inference on NPU
            segments = self._npu_processor.generate_all_segments(
                audio_data=audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language=language,
                timeout_ms=timeout_ms
            )

            if not segments:
                logger.warning("No speech detected in audio")
                return ""

            # Combine all segments and clean Whisper tokens
            text = "".join([seg.text for seg in segments]).strip()
            # Remove Whisper special tokens like <|de|>, <|en|>, <|transcribe|>, etc.
            import re
            text = re.sub(r'<\|[^>]+\|>', '', text).strip()

            logger.info(f"NPU transcribed: {text[:50]}..." if len(text) > 50 else f"NPU transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"NPU transcription error: {e}")
            # Try CPU fallback
            if self._cpu_model:
                logger.info("Falling back to CPU...")
                return self._transcribe_cpu(audio_path, language)
            return ""

    def _transcribe_cpu(self, audio_path: str, language: str) -> str:
        """Transcribe using CPU (fallback)."""
        try:
            # German context prompt helps with accuracy
            initial_prompt = "Dies ist eine Sprachaufnahme auf Deutsch." if language == "de" else None

            segments, info = self._cpu_model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                ),
                initial_prompt=initial_prompt,
                condition_on_previous_text=False  # Prevent hallucinations
            )

            text_parts = []
            for seg in segments:
                text_parts.append(seg.text)

            text = " ".join(text_parts).strip()

            if text:
                logger.info(f"CPU transcribed ({info.language}): {text[:50]}...")
            else:
                logger.warning("No speech detected in audio")

            return text

        except Exception as e:
            logger.error(f"CPU transcription error: {e}")
            return ""

    def release(self):
        """Release NPU resources so other processes can use the NPU.

        Only called by HailoManager when vision needs the NPU,
        NOT after every transcription!
        """
        try:
            if self._npu_processor:
                self._npu_processor = None
            if self._vdevice:
                self._vdevice = None

            # Force garbage collection to release the Hailo resources
            import gc
            gc.collect()

            # Reset so next transcribe() will re-init
            self.backend = "lazy-npu"
            self._npu_initialized = False

            logger.info("[Whisper] NPU resources released (VDevice freed)")
        except Exception as e:
            logger.error(f"[Whisper] Error during release: {e}")

    @property
    def is_available(self) -> bool:
        """Check if any backend is available."""
        # Lazy-load mode: assume available until proven otherwise
        if self._lazy_npu and not self._npu_initialized:
            return True
        return self._npu_processor is not None or self._cpu_model is not None

    def __str__(self) -> str:
        return f"MolochWhisper(backend={self.backend}, available={self.is_available})"


# Keep old class name for compatibility
HailoWhisper = MolochWhisper


def get_whisper() -> MolochWhisper:
    """
    Get or create singleton Whisper instance.

    Returns:
        MolochWhisper instance
    """
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = MolochWhisper()
    return _whisper_instance
