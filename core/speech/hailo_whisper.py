#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Whisper Speech-to-Text
===================================

NPU-accelerated Whisper auf Hailo-10H (8GB RAM).
Shared VDevice mit Vision-Pipeline — alle Modelle permanent geladen.

Hailo-10H 8GB: SCRFD(6MB) + ArcFace(3MB) + YOLOv8m(21MB) + Pose(14MB) + Whisper(137MB) = ~180MB
Das passt locker. Kein Pausieren, kein VDevice-Wechsel, kein Laden/Entladen.

Usage:
    whisper = get_whisper()
    whisper.set_vdevice(service_vdevice)  # Shared VDevice vom Service
    text = whisper.transcribe("/path/to/audio.wav", language="de")
"""

import logging
import re
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Singleton instance
_whisper_instance = None


class MolochWhisper:
    """
    Hailo NPU Whisper — permanent geladen auf shared VDevice.

    Nutzt das GLEICHE VDevice wie die Vision-Pipeline.
    Kein eigenes VDevice, kein Pausieren, kein HailoManager-Acquire.
    8GB NPU-RAM reichen fuer alle Modelle gleichzeitig.
    """

    def __init__(self):
        self.backend = "none"
        self._npu_processor = None
        self._vdevice = None        # Shared VDevice vom Service
        self._shared_vdevice = None  # Referenz auf Service-VDevice (nicht freigeben!)
        self._npu_initialized = False

        logger.info("Whisper: Wartet auf shared VDevice vom Service")
        self.backend = "waiting-for-vdevice"

    def set_vdevice(self, vdevice):
        """Shared VDevice vom Service uebernehmen und Whisper sofort laden.

        Hailo-10H hat 8GB — alle Modelle (Vision + Whisper) passen gleichzeitig.
        Kein Pausieren noetig, kein VDevice-Wechsel.
        """
        self._shared_vdevice = vdevice
        logger.info("[Whisper] Shared VDevice vom Service uebernommen")

        if self._init_npu():
            logger.info("[Whisper] Permanent auf NPU geladen (shared VDevice)")
        else:
            logger.error("[Whisper] NPU init mit shared VDevice fehlgeschlagen!")

    def _init_npu(self) -> bool:
        """Whisper Speech2Text auf NPU laden.

        Nutzt shared VDevice vom Service — kein eigenes VDevice erstellen!
        Bleibt permanent geladen.
        """
        try:
            from hailo_platform.genai import Speech2Text

            import sys
            sys.path.insert(0, str(Path.home() / "hailo-apps"))
            from hailo_apps.python.core.common.core import resolve_hef_path
            from hailo_apps.python.core.common.defines import HAILO10H_ARCH, WHISPER_CHAT_APP

            logger.info("Initializing Hailo NPU Whisper...")

            # Shared VDevice vom Service nutzen
            if self._shared_vdevice:
                self._vdevice = self._shared_vdevice
                logger.info("[Whisper] Nutze shared VDevice vom Service")
            else:
                # Standalone-Fallback: eigenes VDevice (nur fuer Tests)
                from hailo_platform import VDevice
                from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID
                logger.warning("[Whisper] Kein shared VDevice — erstelle eigenes (Standalone-Modus)")
                params = VDevice.create_params()
                params.group_id = SHARED_VDEVICE_GROUP_ID
                self._vdevice = VDevice(params)

            # Whisper HEF finden
            hef_path = resolve_hef_path(
                hef_path=None,
                app_name=WHISPER_CHAT_APP,
                arch=HAILO10H_ARCH
            )

            if hef_path is None:
                logger.error("Whisper HEF not found. Run: hailo-download-resources --group whisper_chat")
                return False

            logger.info(f"Loading Whisper model: {hef_path}")

            # Speech2Text auf shared VDevice laden
            self._npu_processor = Speech2Text(self._vdevice, str(hef_path))

            self.backend = "npu-whisper-base"
            self._npu_initialized = True
            logger.info("Hailo NPU Whisper permanent geladen")
            return True

        except ImportError as e:
            logger.warning(f"Hailo imports not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize NPU Whisper: {e}")
            return False

    def _load_wav_as_numpy(self, audio_path: str) -> Optional[np.ndarray]:
        """Load WAV file and convert to float32 numpy array."""
        try:
            with wave.open(audio_path, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

            if sample_width == 2:
                audio = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 4:
                audio = np.frombuffer(raw_data, dtype=np.int32)
            else:
                logger.error(f"Unsupported sample width: {sample_width}")
                return None

            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            audio = audio.astype(np.float32)
            if sample_width == 2:
                audio /= 32768.0
            elif sample_width == 4:
                audio /= 2147483648.0

            if sample_rate != 16000:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sample_rate))

            logger.debug(f"Loaded audio: {len(audio)} samples, {len(audio)/16000:.2f}s")
            return audio

        except Exception as e:
            logger.error(f"Failed to load WAV file: {e}")
            return None

    def transcribe(self, audio_path: str, language: str = "de",
                   timeout_ms: int = 0, **kwargs) -> str:
        """
        Audio transkribieren. NPU ist permanent geladen — kein Pause/Resume.

        Shared VDevice: Vision + Whisper laufen parallel auf der gleichen NPU.
        Kein HailoManager-Acquire noetig.

        Args:
            audio_path: Pfad zur WAV-Datei
            language: Sprache (de, en, etc.)
            timeout_ms: Timeout in Millisekunden (0 = auto: 4s pro Sekunde Audio, min 30s)

        Returns:
            Transkribierter Text
        """
        # Lazy-init falls set_vdevice noch nicht aufgerufen wurde
        if not self._npu_initialized:
            logger.info("Whisper: Lazy-loading NPU fuer erste Transkription...")
            if not self._init_npu():
                logger.error("NPU init fehlgeschlagen — kein Whisper verfuegbar")
                return ""

        if self._npu_processor:
            return self._transcribe_npu(audio_path, language, timeout_ms)

        logger.error("Kein Whisper-Backend verfuegbar (NPU nicht initialisiert)")
        return ""

    def _transcribe_npu(self, audio_path: str, language: str, timeout_ms: int) -> str:
        """Transcribe using Hailo NPU."""
        try:
            from hailo_platform.genai import Speech2TextTask

            audio_data = self._load_wav_as_numpy(audio_path)
            if audio_data is None:
                return ""

            audio_duration_s = len(audio_data) / 16000.0
            # Dynamischer Timeout: 4s Verarbeitung pro Sekunde Audio, mindestens 30s
            if timeout_ms <= 0:
                timeout_ms = max(30000, int(audio_duration_s * 4000))

            logger.info(f"NPU transcribing {audio_duration_s:.1f}s audio (timeout={timeout_ms}ms)...")

            segments = self._npu_processor.generate_all_segments(
                audio_data=audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language=language,
                timeout_ms=timeout_ms
            )

            if not segments:
                logger.warning("No speech detected in audio")
                return ""

            text = "".join([seg.text for seg in segments]).strip()
            text = re.sub(r'<\|[^>]+\|>', '', text).strip()

            logger.info(f"NPU transcribed: {text[:50]}..." if len(text) > 50 else f"NPU transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"NPU transcription error: {e}")
            return ""

    def release(self):
        """NPU Whisper-Ressourcen freigeben.
        ACHTUNG: Shared VDevice wird NICHT freigegeben (gehoert dem Service)."""
        try:
            if self._npu_processor:
                self._npu_processor = None
            # Shared VDevice NICHT freigeben!
            if self._vdevice and self._vdevice is not self._shared_vdevice:
                self._vdevice = None
            self._vdevice = None

            import gc
            gc.collect()

            self.backend = "released"
            self._npu_initialized = False

            logger.info("[Whisper] NPU-Ressourcen freigegeben (VDevice bleibt beim Service)")
        except Exception as e:
            logger.error(f"[Whisper] Error during release: {e}")

    @property
    def is_available(self) -> bool:
        """Check if NPU backend is available."""
        return self._npu_initialized and self._npu_processor is not None

    def __str__(self) -> str:
        return f"MolochWhisper(backend={self.backend}, available={self.is_available})"


# Kompatibilitaet
HailoWhisper = MolochWhisper


def get_whisper() -> MolochWhisper:
    """Get or create singleton Whisper instance."""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = MolochWhisper()
    return _whisper_instance
