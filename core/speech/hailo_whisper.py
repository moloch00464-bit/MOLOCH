#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Whisper Speech-to-Text
===================================

NPU-accelerated Whisper auf Hailo-10H (8GB RAM).
On-Demand: Wird NUR bei Push-to-Talk geladen und danach entladen.
Vision-Pipeline behaelt volle NPU-Bandbreite wenn Whisper nicht aktiv.

Ablauf:
  1. System startet → Whisper NICHT geladen → Vision hat volle NPU
  2. User drueckt Push-to-Talk → Whisper wird auf NPU geladen (1-2s)
  3. Spracherkennung laeuft
  4. Ergebnis da → Whisper wird von NPU entladen
  5. Vision hat wieder volle NPU

Usage:
    whisper = get_whisper()
    whisper.set_vdevice(service_vdevice)  # VDevice speichern (kein Laden!)
    text = whisper.transcribe("/path/to/audio.wav", language="de")
    # Whisper wird automatisch entladen nach Transkription
"""

import gc
import logging
import re
import time
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Singleton instance
_whisper_instance = None


class MolochWhisper:
    """
    Hailo NPU Whisper — On-Demand Laden/Entladen.

    Wird NUR bei Push-to-Talk auf die NPU geladen und danach entladen.
    Nutzt shared VDevice vom Service. Vision hat volle NPU wenn Whisper nicht aktiv.
    """

    def __init__(self):
        self.backend = "on-demand"
        self._npu_processor = None
        self._vdevice = None        # Aktives VDevice (nur waehrend Transkription)
        self._shared_vdevice = None  # Referenz auf Service-VDevice (nicht freigeben!)
        self._npu_initialized = False
        self._load_lock = threading.Lock()  # Schutz gegen parallele Load/Unload

        logger.info("Whisper: On-Demand Modus (wird bei PTT geladen)")

    def set_vdevice(self, vdevice):
        """Shared VDevice vom Service speichern. Whisper wird NICHT geladen.

        Das VDevice wird nur gespeichert und bei Bedarf (Push-to-Talk)
        fuer das Laden von Whisper verwendet.
        """
        self._shared_vdevice = vdevice
        logger.info("[Whisper] VDevice gespeichert — On-Demand (kein permanentes Laden)")

    def _load_npu(self) -> bool:
        """Whisper Speech2Text auf NPU laden (On-Demand).

        Nutzt shared VDevice vom Service. Wird nach Transkription entladen.
        """
        if self._npu_initialized and self._npu_processor:
            return True

        try:
            from hailo_platform.genai import Speech2Text

            import sys
            sys.path.insert(0, str(Path.home() / "hailo-apps"))
            from hailo_apps.python.core.common.core import resolve_hef_path
            from hailo_apps.python.core.common.defines import HAILO10H_ARCH, WHISPER_CHAT_APP

            # Shared VDevice vom Service nutzen
            if self._shared_vdevice:
                self._vdevice = self._shared_vdevice
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

            logger.info(f"[Whisper] On-Demand: Lade {hef_path}...")

            # Speech2Text auf shared VDevice laden
            self._npu_processor = Speech2Text(self._vdevice, str(hef_path))

            self.backend = "npu-whisper-base"
            self._npu_initialized = True
            return True

        except ImportError as e:
            logger.warning(f"Hailo imports not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load NPU Whisper: {e}")
            return False

    def _unload_npu(self):
        """Whisper von NPU entladen — gibt Ressourcen fuer Vision frei."""
        try:
            if self._npu_processor:
                self._npu_processor = None
            # VDevice-Referenz freigeben (shared VDevice bleibt beim Service)
            self._vdevice = None
            gc.collect()
            self._npu_initialized = False
            self.backend = "on-demand"
            logger.info("[Whisper] NPU entladen — Vision hat volle Bandbreite")
        except Exception as e:
            logger.error(f"[Whisper] Entladen fehlgeschlagen: {e}")
            self._npu_initialized = False
            self.backend = "on-demand"

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

            # Audio-Preprocessing: DC-Offset entfernen + Normalisierung auf -3dBFS
            dc_offset = np.mean(audio)
            if abs(dc_offset) > 0.001:
                audio = audio - dc_offset
                logger.debug(f"DC-Offset entfernt: {dc_offset:.4f}")

            peak = np.max(np.abs(audio))
            if peak > 0.001:
                # -3dBFS = 10^(-3/20) ≈ 0.7079
                target = 0.7079
                gain = target / peak
                if gain > 10.0:
                    gain = 10.0  # Max 20dB Verstaerkung
                audio = audio * gain
                logger.debug(f"Normalisiert: Peak {peak:.4f} → {peak * gain:.4f} "
                             f"(Gain {20 * np.log10(gain):.1f}dB)")

            logger.debug(f"Loaded audio: {len(audio)} samples, {len(audio)/16000:.2f}s")
            return audio

        except Exception as e:
            logger.error(f"Failed to load WAV file: {e}")
            return None

    def transcribe(self, audio_path: str, language: str = "de",
                   timeout_ms: int = 0, **kwargs) -> str:
        """
        On-Demand Transkription: Hailo NPU Base als Primary.

        Strategie:
        1. Hailo NPU Base (schnell, CPU-sparend, <1s Latenz)
        2. Fallback: faster-whisper Small auf CPU (~5s fuer 10s Audio)

        Args:
            audio_path: Pfad zur WAV-Datei
            language: Sprache (de, en, etc.) — IMMER explizit angeben!
            timeout_ms: Timeout in ms (0 = automatisch)

        Returns:
            Transkribierter Text
        """
        # Primary: NPU Whisper Base (schnell, kein CPU-Load)
        with self._load_lock:
            t_load = time.perf_counter()
            if self._load_npu():
                dt_load = (time.perf_counter() - t_load) * 1000
                logger.info(f"[Whisper] NPU geladen in {dt_load:.0f}ms")
                try:
                    if self._npu_processor:
                        text = self._transcribe_npu(audio_path, language, timeout_ms)
                        if text:
                            return text
                        logger.warning("[Whisper] NPU: kein Ergebnis — Fallback auf CPU")
                finally:
                    self._unload_npu()
            else:
                logger.warning("[Whisper] NPU init fehlgeschlagen — Fallback auf CPU")

        # Fallback: faster-whisper Small auf CPU
        logger.warning("[Whisper] Fallback: faster-whisper Small auf CPU")
        return self._transcribe_faster_whisper(audio_path, language)

    def _transcribe_faster_whisper(self, audio_path: str, language: str) -> str:
        """Transkription mit faster-whisper Small auf CPU (Fallback wenn NPU fehlschlaegt).

        Modell wird geladen, transkribiert, und sofort wieder entladen (RAM sparen).
        ~626MB RAM waehrend Transkription, danach wieder frei.
        """
        try:
            from faster_whisper import WhisperModel

            t0 = time.perf_counter()
            # Small statt Medium: halb so viel RAM, doppelt so schnell
            model = WhisperModel("small", device="cpu", compute_type="int8")
            dt_load = (time.perf_counter() - t0) * 1000
            self.backend = "faster-whisper-small"

            t1 = time.perf_counter()
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=True,
                initial_prompt="Ich spreche Deutsch mit fraenkischem Dialekt.",
                vad_filter=True,
                vad_parameters=dict(
                    # 1200ms: nicht bei Atempausen oder kurzen Pausen abschneiden
                    min_silence_duration_ms=1200,
                    # 400ms Padding: Wortanfaenge/-enden nicht abschneiden
                    speech_pad_ms=400,
                ),
            )
            text = "".join(seg.text for seg in segments).strip()
            dt_trans = (time.perf_counter() - t1) * 1000

            # Modell sofort entladen (RAM freigeben)
            del model
            gc.collect()
            self.backend = "on-demand"

            logger.info(f"[Whisper] faster-whisper Small: "
                        f"Laden={dt_load:.0f}ms, Transkription={dt_trans:.0f}ms, "
                        f"Sprache={info.language} ({info.language_probability:.0%})")

            if text:
                logger.info(f"[Whisper] faster-whisper transcribed: {text}")
            return text

        except ImportError:
            logger.warning("[Whisper] faster-whisper nicht installiert")
            return ""
        except Exception as e:
            logger.error(f"[Whisper] faster-whisper Fehler: {e}")
            try:
                del model
            except NameError:
                pass
            gc.collect()
            self.backend = "on-demand"
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

            # Hailo API aufrufen — optionale Parameter per try/except absichern
            try:
                # Vollstaendiger Aufruf: language=de, vad_filter=False (kein internes Cutting)
                segments = self._npu_processor.generate_all_segments(
                    audio_data=audio_data,
                    task=Speech2TextTask.TRANSCRIBE,
                    language=language,
                    timeout_ms=timeout_ms,
                    initial_prompt="Ich spreche Deutsch. Fränkischer Dialekt möglich.",
                    vad_filter=False,  # VAD=0: kein internes Segmentschneiden
                )
            except TypeError:
                # Aeltere Hailo API-Version: ohne optionale Parameter
                logger.debug("Hailo API: optionale Parameter nicht unterstuetzt — Basis-Aufruf")
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

            # Bekannte Whisper-Halluzinationen filtern (Stille/Hintergrundgeraeusch)
            halluzinationen = [
                "Untertitel", "Danke", "Tschüss", "Tschüss!", "Auf Wiedersehen",
                "Bitte abonnieren", "Abonnieren", "Vielen Dank", "Vielen Dank!",
                "Thank you", "Thanks", "Bye", "Goodbye", "Subscribe",
                "[Musik]", "[musik]", "[MUSIK]", "[Applaus]", "[applaus]",
                "[Gelächter]", "(Musik)", "(Stille)",
            ]
            text_lower = text.lower().strip()
            for h in halluzinationen:
                if text_lower == h.lower().strip():
                    logger.info(f"Halluzination gefiltert: '{text}'")
                    return ""

            logger.info(f"NPU transcribed: {text[:50]}..." if len(text) > 50 else f"NPU transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"NPU transcription error: {e}")
            return ""

    def release(self):
        """NPU Whisper-Ressourcen freigeben.
        ACHTUNG: Shared VDevice wird NICHT freigegeben (gehoert dem Service)."""
        with self._load_lock:
            self._unload_npu()
            logger.info("[Whisper] Release abgeschlossen")

    @property
    def is_available(self) -> bool:
        """Check ob Whisper grundsaetzlich verfuegbar ist (VDevice vorhanden)."""
        return self._shared_vdevice is not None

    def __str__(self) -> str:
        loaded = "geladen" if self._npu_initialized else "on-demand"
        return f"MolochWhisper(backend={self.backend}, {loaded})"


# Kompatibilitaet
HailoWhisper = MolochWhisper


def get_whisper() -> MolochWhisper:
    """Get or create singleton Whisper instance."""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = MolochWhisper()
    return _whisper_instance
