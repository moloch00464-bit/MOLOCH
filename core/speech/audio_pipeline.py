#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Audio Pipeline
===========================

Layer 1 (Raw Facts) - Audio preprocessing and quality assessment.

Features:
- Audio diagnostics (RMS, SNR, clipping detection)
- Noise gate with adaptive threshold
- RMS normalization for consistent Whisper input
- Multi-stage VAD (energy + webrtcvad)
- Segment validation and quality scoring

Hardware Assumptions:
- Pipewire audio backend
- 16kHz sample rate (Whisper standard)
- 16-bit signed PCM
"""

import logging
import wave
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Try to import webrtcvad for advanced VAD
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logger.warning("webrtcvad not available - using energy-based VAD only")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AudioConfig:
    """Audio pipeline configuration."""

    # === AUDIO FORMAT ===
    sample_rate: int = 16000        # Whisper expects 16kHz
    sample_width: int = 2           # 16-bit = 2 bytes
    channels: int = 1               # Mono

    # === NOISE GATE ===
    noise_gate_threshold_db: float = -60.0  # Sehr empfindlich für leise Sprache
    noise_gate_attack_ms: float = 5.0       # Schnellere Reaktion
    noise_gate_release_ms: float = 150.0    # Längeres Halten nach Sprache
    noise_gate_hold_ms: float = 100.0       # Längeres Halten

    # === NORMALIZATION ===
    target_rms_db: float = -18.0    # Etwas lauter für Whisper
    max_gain_db: float = 65.0       # Mehr Verstärkung erlaubt
    peak_limit_db: float = -1.0     # Hard limiter threshold

    # === VAD ===
    vad_aggressiveness: int = 0     # webrtcvad: 0 = am wenigsten aggressiv, mehr Sprache erkannt
    min_speech_duration_ms: int = 150       # Kürzere Segmente erlaubt (deutsche Wörter)
    max_speech_duration_ms: int = 30000     # Maximum segment length
    speech_pad_ms: int = 250                # Mehr Padding vor/nach Sprache
    silence_threshold_ms: int = 600         # Längere Pause bevor Segment endet

    # === QUALITY THRESHOLDS ===
    min_rms_db: float = -70.0  # Sehr empfindlich - leisere Sprache akzeptieren
    max_rms_db: float = -3.0        # Above this = likely clipping
    min_snr_db: float = 10.0        # Minimum signal-to-noise ratio
    max_clipping_ratio: float = 0.01  # Max 1% clipped samples


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioDiagnostics:
    """Audio quality diagnostics."""
    duration_ms: int = 0
    rms_db: float = -100.0
    peak_db: float = -100.0
    noise_floor_db: float = -100.0
    snr_db: float = 0.0
    clipping_ratio: float = 0.0
    speech_ratio: float = 0.0           # Ratio of speech to total
    quality_score: float = 0.0          # 0-1 overall quality
    issues: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Check if audio quality is acceptable for ASR."""
        return self.quality_score >= 0.5 and len(self.issues) == 0


@dataclass
class AudioSegment:
    """Validated audio segment ready for ASR."""
    audio_data: np.ndarray
    sample_rate: int
    duration_ms: int
    diagnostics: AudioDiagnostics
    confidence: float               # VAD confidence


@dataclass
class ProcessingResult:
    """Result of audio processing pipeline."""
    success: bool
    segments: List[AudioSegment]
    diagnostics: AudioDiagnostics
    output_path: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= 0:
        return -100.0
    return 20.0 * np.log10(linear)


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS level of audio."""
    if len(audio) == 0:
        return 0.0
    return np.sqrt(np.mean(audio.astype(np.float64) ** 2))


def calculate_rms_db(audio: np.ndarray, max_amplitude: float = 32768.0) -> float:
    """Calculate RMS level in dB (relative to max amplitude)."""
    rms = calculate_rms(audio)
    if rms == 0:
        return -100.0
    return linear_to_db(rms / max_amplitude)


# =============================================================================
# AUDIO DIAGNOSTICS
# =============================================================================

class AudioAnalyzer:
    """Analyze audio quality and characteristics."""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> AudioDiagnostics:
        """
        Analyze audio and return diagnostics.

        Args:
            audio: Audio samples as int16 numpy array
            sample_rate: Sample rate in Hz

        Returns:
            AudioDiagnostics with quality metrics
        """
        diag = AudioDiagnostics()

        if len(audio) == 0:
            diag.issues.append("empty_audio")
            return diag

        # Duration
        diag.duration_ms = int(len(audio) / sample_rate * 1000)

        # Convert to float for analysis
        audio_float = audio.astype(np.float32)
        max_amp = 32768.0

        # RMS level
        diag.rms_db = calculate_rms_db(audio_float, max_amp)

        # Peak level
        peak = np.max(np.abs(audio_float))
        diag.peak_db = linear_to_db(peak / max_amp)

        # Clipping detection (samples at or near max)
        clipping_threshold = 32000  # Near max int16
        clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
        diag.clipping_ratio = clipped_samples / len(audio)

        # Noise floor estimation (lowest 10% RMS)
        frame_size = int(sample_rate * 0.02)  # 20ms frames
        if len(audio) >= frame_size:
            n_frames = len(audio) // frame_size
            frame_rms = []
            for i in range(n_frames):
                frame = audio_float[i * frame_size:(i + 1) * frame_size]
                frame_rms.append(calculate_rms(frame))

            if frame_rms:
                sorted_rms = sorted(frame_rms)
                noise_frames = sorted_rms[:max(1, len(sorted_rms) // 10)]
                noise_rms = np.mean(noise_frames)
                diag.noise_floor_db = linear_to_db(noise_rms / max_amp)

                # SNR estimation
                signal_rms = np.mean(sorted_rms[-len(sorted_rms) // 2:])
                if noise_rms > 0:
                    diag.snr_db = linear_to_db(signal_rms / noise_rms)

        # Speech ratio (frames above noise floor + margin)
        if HAS_WEBRTCVAD and diag.duration_ms >= 30:
            diag.speech_ratio = self._estimate_speech_ratio(audio, sample_rate)
        else:
            # Fallback: use energy-based estimation
            speech_threshold = db_to_linear(diag.noise_floor_db + 10) * max_amp
            speech_samples = np.sum(np.abs(audio_float) > speech_threshold)
            diag.speech_ratio = speech_samples / len(audio)

        # Quality scoring
        diag.quality_score = self._calculate_quality_score(diag)

        # Issue detection
        self._detect_issues(diag)

        return diag

    def _estimate_speech_ratio(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate speech ratio using webrtcvad."""
        try:
            vad = webrtcvad.Vad(self.config.vad_aggressiveness)
            frame_duration_ms = 30  # webrtcvad supports 10, 20, 30ms
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            n_frames = len(audio) // frame_size

            speech_frames = 0
            for i in range(n_frames):
                frame = audio[i * frame_size:(i + 1) * frame_size]
                frame_bytes = frame.tobytes()
                if vad.is_speech(frame_bytes, sample_rate):
                    speech_frames += 1

            return speech_frames / max(1, n_frames)
        except Exception as e:
            logger.debug(f"VAD speech ratio estimation failed: {e}")
            return 0.5

    def _calculate_quality_score(self, diag: AudioDiagnostics) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0

        # Penalize low RMS
        if diag.rms_db < self.config.min_rms_db:
            score *= 0.3
        elif diag.rms_db < self.config.min_rms_db + 10:
            score *= 0.7

        # Penalize clipping
        if diag.clipping_ratio > self.config.max_clipping_ratio:
            score *= 0.5
        elif diag.clipping_ratio > self.config.max_clipping_ratio / 2:
            score *= 0.8

        # Penalize low SNR
        if diag.snr_db < self.config.min_snr_db:
            score *= 0.5
        elif diag.snr_db < self.config.min_snr_db + 5:
            score *= 0.8

        # Penalize very short duration
        if diag.duration_ms < self.config.min_speech_duration_ms:
            score *= 0.5

        # Penalize low speech ratio
        if diag.speech_ratio < 0.1:
            score *= 0.5
        elif diag.speech_ratio < 0.3:
            score *= 0.8

        return max(0.0, min(1.0, score))

    def _detect_issues(self, diag: AudioDiagnostics):
        """Detect and log audio quality issues."""
        issues = []

        if diag.rms_db < self.config.min_rms_db:
            issues.append("too_quiet")
        if diag.rms_db > self.config.max_rms_db:
            issues.append("too_loud")
        if diag.clipping_ratio > self.config.max_clipping_ratio:
            issues.append("clipping")
        if diag.snr_db < self.config.min_snr_db:
            issues.append("low_snr")
        if diag.speech_ratio < 0.1:
            issues.append("no_speech")
        if diag.duration_ms < self.config.min_speech_duration_ms:
            issues.append("too_short")

        diag.issues = issues


# =============================================================================
# AUDIO PREPROCESSING
# =============================================================================

class AudioPreprocessor:
    """Preprocess audio for optimal ASR performance."""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.analyzer = AudioAnalyzer(config)

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, AudioDiagnostics]:
        """
        Process audio through the full pipeline.

        Pipeline:
        1. Analyze input
        2. Apply noise gate
        3. Normalize RMS
        4. Apply peak limiter
        5. Analyze output

        Args:
            audio: Input audio as int16 numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (processed_audio, diagnostics)
        """
        # Initial analysis
        input_diag = self.analyzer.analyze(audio, sample_rate)
        logger.debug(f"Input: {input_diag.rms_db:.1f}dB RMS, {input_diag.snr_db:.1f}dB SNR")

        # Convert to float for processing
        audio_float = audio.astype(np.float32) / 32768.0

        # Apply noise gate
        audio_float = self._apply_noise_gate(audio_float, sample_rate, input_diag.noise_floor_db)

        # Normalize RMS
        audio_float = self._normalize_rms(audio_float)

        # Apply peak limiter
        audio_float = self._apply_limiter(audio_float)

        # Convert back to int16
        audio_out = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)

        # Final analysis
        output_diag = self.analyzer.analyze(audio_out, sample_rate)
        logger.debug(f"Output: {output_diag.rms_db:.1f}dB RMS, {output_diag.snr_db:.1f}dB SNR")

        return audio_out, output_diag

    def _apply_noise_gate(self, audio: np.ndarray, sample_rate: int, noise_floor_db: float) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        # Threshold slightly above noise floor
        threshold_db = max(noise_floor_db + 6, self.config.noise_gate_threshold_db)
        threshold_linear = db_to_linear(threshold_db)

        # Calculate envelope
        frame_ms = 10
        frame_size = int(sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_size

        envelope = np.zeros(len(audio))
        for i in range(n_frames):
            start = i * frame_size
            end = start + frame_size
            frame_rms = calculate_rms(audio[start:end])
            envelope[start:end] = frame_rms

        # Smooth envelope
        attack_samples = int(sample_rate * self.config.noise_gate_attack_ms / 1000)
        release_samples = int(sample_rate * self.config.noise_gate_release_ms / 1000)

        # Simple gate: open when above threshold
        gate = np.where(envelope > threshold_linear, 1.0, 0.0)

        # Smooth transitions
        from scipy.ndimage import uniform_filter1d
        gate = uniform_filter1d(gate.astype(float), size=max(attack_samples, 1))

        return audio * gate

    def _normalize_rms(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target RMS level."""
        current_rms = calculate_rms(audio)
        if current_rms == 0:
            return audio

        target_rms = db_to_linear(self.config.target_rms_db)
        gain = target_rms / current_rms

        # Limit maximum gain
        max_gain = db_to_linear(self.config.max_gain_db)
        gain = min(gain, max_gain)

        return audio * gain

    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Apply peak limiter to prevent clipping."""
        threshold = db_to_linear(self.config.peak_limit_db)
        return np.clip(audio, -threshold, threshold)


# =============================================================================
# VAD (VOICE ACTIVITY DETECTION)
# =============================================================================

class VoiceActivityDetector:
    """Multi-stage voice activity detection."""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._vad = None
        if HAS_WEBRTCVAD:
            self._vad = webrtcvad.Vad(config.vad_aggressiveness if config else 2)

    def detect_speech_segments(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[int, int]]:
        """
        Detect speech segments in audio.

        Args:
            audio: Audio as int16 numpy array
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_sample, end_sample) tuples
        """
        if not HAS_WEBRTCVAD:
            return self._detect_energy_based(audio, sample_rate)

        return self._detect_webrtcvad(audio, sample_rate)

    def _detect_energy_based(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        """Fallback energy-based VAD."""
        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_size

        # Calculate frame energies
        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energies.append(calculate_rms(frame.astype(np.float32)))

        if not energies:
            return []

        # Adaptive threshold
        sorted_energies = sorted(energies)
        noise_floor = np.mean(sorted_energies[:max(1, len(sorted_energies) // 10)])
        threshold = noise_floor * 3  # 10dB above noise floor

        # Find speech regions
        speech_frames = [e > threshold for e in energies]

        # Convert to sample ranges with padding
        segments = []
        in_speech = False
        start_frame = 0
        pad_frames = self.config.speech_pad_ms // frame_ms

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = max(0, i - pad_frames)
                in_speech = True
            elif not is_speech and in_speech:
                end_frame = min(n_frames, i + pad_frames)
                segments.append((start_frame * frame_size, end_frame * frame_size))
                in_speech = False

        # Handle segment still open at end
        if in_speech:
            segments.append((start_frame * frame_size, len(audio)))

        # Merge close segments
        return self._merge_segments(segments, sample_rate)

    def _detect_webrtcvad(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        """WebRTC VAD-based detection."""
        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_size

        speech_frames = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            try:
                is_speech = self._vad.is_speech(frame.tobytes(), sample_rate)
                speech_frames.append(is_speech)
            except Exception:
                speech_frames.append(False)

        # Convert to sample ranges
        segments = []
        in_speech = False
        start_frame = 0
        silence_frames = 0
        silence_threshold = self.config.silence_threshold_ms // frame_ms
        pad_frames = self.config.speech_pad_ms // frame_ms

        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                if not in_speech:
                    start_frame = max(0, i - pad_frames)
                    in_speech = True
                silence_frames = 0
            else:
                if in_speech:
                    silence_frames += 1
                    if silence_frames >= silence_threshold:
                        end_frame = min(n_frames, i + pad_frames)
                        segments.append((start_frame * frame_size, end_frame * frame_size))
                        in_speech = False

        if in_speech:
            segments.append((start_frame * frame_size, len(audio)))

        return self._merge_segments(segments, sample_rate)

    def _merge_segments(self, segments: List[Tuple[int, int]], sample_rate: int) -> List[Tuple[int, int]]:
        """Merge segments that are close together."""
        if not segments:
            return []

        merge_gap = int(sample_rate * self.config.silence_threshold_ms / 1000)
        min_duration = int(sample_rate * self.config.min_speech_duration_ms / 1000)

        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        # Filter by minimum duration
        return [(s, e) for s, e in merged if e - s >= min_duration]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AudioPipeline:
    """
    Complete audio processing pipeline for M.O.L.O.C.H.

    Combines diagnostics, preprocessing, and VAD for optimal ASR input.
    """

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.analyzer = AudioAnalyzer(self.config)
        self.preprocessor = AudioPreprocessor(self.config)
        self.vad = VoiceActivityDetector(self.config)

    def process_file(self, input_path: str, output_path: str = None) -> ProcessingResult:
        """
        Process audio file through the pipeline.

        Args:
            input_path: Path to input WAV file
            output_path: Optional output path (auto-generated if None)

        Returns:
            ProcessingResult with processed audio and diagnostics
        """
        try:
            # Load audio
            audio, sample_rate = self._load_wav(input_path)
            if audio is None:
                return ProcessingResult(success=False, segments=[], diagnostics=AudioDiagnostics(),
                                       error="Failed to load audio file")

            # Process
            return self.process_audio(audio, sample_rate, output_path)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return ProcessingResult(success=False, segments=[], diagnostics=AudioDiagnostics(),
                                   error=str(e))

    def process_audio(self, audio: np.ndarray, sample_rate: int = 16000,
                     output_path: str = None) -> ProcessingResult:
        """
        Process audio array through the pipeline.

        Args:
            audio: Input audio as int16 numpy array
            sample_rate: Sample rate in Hz
            output_path: Optional output path

        Returns:
            ProcessingResult with processed audio and diagnostics
        """
        # Initial diagnostics
        input_diag = self.analyzer.analyze(audio, sample_rate)
        logger.info(f"Input: {input_diag.duration_ms}ms, {input_diag.rms_db:.1f}dB, "
                   f"SNR:{input_diag.snr_db:.1f}dB, Speech:{input_diag.speech_ratio:.0%}")

        # Check if audio is worth processing
        if input_diag.duration_ms < self.config.min_speech_duration_ms:
            return ProcessingResult(
                success=False, segments=[], diagnostics=input_diag,
                error=f"Audio too short: {input_diag.duration_ms}ms"
            )

        # Preprocess
        processed_audio, processed_diag = self.preprocessor.process(audio, sample_rate)

        # Detect speech segments
        speech_segments = self.vad.detect_speech_segments(processed_audio, sample_rate)

        if not speech_segments:
            return ProcessingResult(
                success=False, segments=[], diagnostics=processed_diag,
                error="No speech detected"
            )

        # Create audio segments
        segments = []
        for start, end in speech_segments:
            segment_audio = processed_audio[start:end]
            segment_diag = self.analyzer.analyze(segment_audio, sample_rate)
            segments.append(AudioSegment(
                audio_data=segment_audio,
                sample_rate=sample_rate,
                duration_ms=int((end - start) / sample_rate * 1000),
                diagnostics=segment_diag,
                confidence=segment_diag.quality_score
            ))

        # Save processed audio if path provided
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav', prefix='moloch_processed_')

        self._save_wav(output_path, processed_audio, sample_rate)

        logger.info(f"Output: {len(segments)} segments, {processed_diag.rms_db:.1f}dB, "
                   f"Quality:{processed_diag.quality_score:.0%}")

        return ProcessingResult(
            success=True,
            segments=segments,
            diagnostics=processed_diag,
            output_path=output_path
        )

    def _load_wav(self, path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load WAV file as numpy array."""
        try:
            with wave.open(path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

                # Handle different sample widths
                sample_width = wf.getsampwidth()
                if sample_width == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 1:
                    audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.int16) - 128) * 256
                else:
                    logger.error(f"Unsupported sample width: {sample_width}")
                    return None, 0

                # Convert to mono if needed
                if wf.getnchannels() == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    from scipy.signal import resample
                    n_samples = int(len(audio) * 16000 / sample_rate)
                    audio = resample(audio, n_samples).astype(np.int16)
                    sample_rate = 16000

                return audio, sample_rate

        except Exception as e:
            logger.error(f"Failed to load WAV: {e}")
            return None, 0

    def _save_wav(self, path: str, audio: np.ndarray, sample_rate: int):
        """Save numpy array as WAV file."""
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())
        except Exception as e:
            logger.error(f"Failed to save WAV: {e}")


# =============================================================================
# HARDWARE VALIDATION
# =============================================================================

class HardwareValidator:
    """Validate audio hardware setup."""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.analyzer = AudioAnalyzer(config)

    def run_silence_test(self, duration_ms: int = 1000) -> AudioDiagnostics:
        """
        Test silence/ambient noise level.

        Records silence and measures noise floor.
        """
        logger.info("Running silence test...")
        audio = self._record(duration_ms)
        if audio is None:
            return AudioDiagnostics(issues=["recording_failed"])

        diag = self.analyzer.analyze(audio, self.config.sample_rate)
        logger.info(f"Silence test: {diag.noise_floor_db:.1f}dB noise floor")
        return diag

    def run_impulse_test(self) -> Tuple[bool, str]:
        """
        Test microphone response with impulse (clap).

        Returns (passed, message)
        """
        logger.info("Impulse test: Clap your hands...")
        audio = self._record(2000)
        if audio is None:
            return False, "Recording failed"

        # Look for impulse
        peak = np.max(np.abs(audio))
        peak_db = linear_to_db(peak / 32768.0)

        if peak_db < -20:
            return False, f"No impulse detected (peak: {peak_db:.1f}dB)"
        elif peak_db > -3:
            return False, f"Impulse clipped (peak: {peak_db:.1f}dB)"
        else:
            return True, f"Impulse detected at {peak_db:.1f}dB"

    def _record(self, duration_ms: int) -> Optional[np.ndarray]:
        """Record audio using Pipewire."""
        import subprocess
        import tempfile

        path = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run([
                'pw-record',
                '--rate', str(self.config.sample_rate),
                '--channels', '1',
                '--format', 's16',
                path
            ], timeout=duration_ms / 1000 + 1, capture_output=True)

            with wave.open(path, 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            return audio

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None
        finally:
            Path(path).unlink(missing_ok=True)


# =============================================================================
# SINGLETON & CLI
# =============================================================================

_pipeline_instance: Optional[AudioPipeline] = None


def get_pipeline(config: AudioConfig = None) -> AudioPipeline:
    """Get or create audio pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AudioPipeline(config)
    return _pipeline_instance


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("M.O.L.O.C.H. Audio Pipeline")
    print("=" * 50)
    print(f"webrtcvad available: {HAS_WEBRTCVAD}")

    if len(sys.argv) > 1:
        # Process file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"\nProcessing: {input_file}")
        pipeline = get_pipeline()
        result = pipeline.process_file(input_file, output_file)

        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Segments: {len(result.segments)}")
        print(f"  Quality: {result.diagnostics.quality_score:.0%}")
        print(f"  RMS: {result.diagnostics.rms_db:.1f}dB")
        print(f"  SNR: {result.diagnostics.snr_db:.1f}dB")
        print(f"  Issues: {result.diagnostics.issues or 'None'}")
        if result.output_path:
            print(f"  Output: {result.output_path}")
        if result.error:
            print(f"  Error: {result.error}")
    else:
        print("\nUsage: python audio_pipeline.py <input.wav> [output.wav]")
        print("\nRunning hardware validation...")

        validator = HardwareValidator()
        silence = validator.run_silence_test()
        print(f"\nSilence test:")
        print(f"  Noise floor: {silence.noise_floor_db:.1f}dB")
        print(f"  Issues: {silence.issues or 'None'}")
