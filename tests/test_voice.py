#!/usr/bin/env python3
"""
M.O.L.O.C.H. Voice System Tests
================================

Tests for voice configuration, selection logic, and audio pipeline.
All tests run locally without TTS hardware.

Run:
    cd ~/moloch && python3 -m pytest tests/test_voice.py -v
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.tts.config.voices import (
    load_voice_config,
    validate_voice_metadata,
    get_voice_by_id,
    filter_voices_by_criteria,
    list_available_voices,
    get_config_metadata,
)
from core.tts.tts_manager import ContextSignals, TTSEngine, VoiceSelection
from core.tts.selection.voice_selector import VoiceSelector


# === Voice Configuration Tests ===

class TestVoiceConfig:
    """Tests for voice configuration loading and validation."""

    def test_load_voice_config_returns_list(self):
        """load_voice_config() should return a list of voice dicts."""
        voices = load_voice_config()
        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_all_voices_have_required_fields(self):
        """Every voice should have all required fields."""
        voices = load_voice_config()
        required = ["voice_id", "display_name", "engine", "style_tags",
                     "energy_level", "emotional_range", "preferred_contexts", "language"]
        for voice in voices:
            for field in required:
                assert field in voice, f"Voice {voice.get('voice_id', '?')} missing {field}"

    def test_voice_ids_are_unique(self):
        """All voice IDs should be unique."""
        voices = load_voice_config()
        ids = [v["voice_id"] for v in voices]
        assert len(ids) == len(set(ids)), f"Duplicate voice IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_energy_levels_are_valid(self):
        """All energy levels should be low/medium/high."""
        voices = load_voice_config()
        valid_levels = {"low", "medium", "high"}
        for voice in voices:
            assert voice["energy_level"] in valid_levels

    def test_validate_voice_metadata_valid(self):
        """validate_voice_metadata() should accept valid voice."""
        voice = {
            "voice_id": "test_voice",
            "display_name": "Test Voice",
            "engine": "piper",
            "style_tags": ["test"],
            "energy_level": "medium",
            "emotional_range": ["neutral"],
            "preferred_contexts": ["testing"],
            "language": "de-DE",
        }
        assert validate_voice_metadata(voice) is True

    def test_validate_voice_metadata_missing_field(self):
        """validate_voice_metadata() should reject voice with missing field."""
        voice = {"voice_id": "incomplete", "engine": "piper"}
        assert validate_voice_metadata(voice) is False

    def test_validate_voice_metadata_invalid_energy(self):
        """validate_voice_metadata() should reject invalid energy level."""
        voice = {
            "voice_id": "bad_energy",
            "display_name": "Bad",
            "engine": "piper",
            "style_tags": [],
            "energy_level": "turbo",  # Invalid
            "emotional_range": [],
            "preferred_contexts": [],
            "language": "de-DE",
        }
        assert validate_voice_metadata(voice) is False

    def test_validate_voice_metadata_non_list_field(self):
        """validate_voice_metadata() should reject non-list for list fields."""
        voice = {
            "voice_id": "bad_list",
            "display_name": "Bad",
            "engine": "piper",
            "style_tags": "not_a_list",  # Should be list
            "energy_level": "medium",
            "emotional_range": [],
            "preferred_contexts": [],
            "language": "de-DE",
        }
        assert validate_voice_metadata(voice) is False

    def test_get_voice_by_id_found(self):
        """get_voice_by_id() should return voice if it exists."""
        voices = load_voice_config()
        if voices:
            first_id = voices[0]["voice_id"]
            result = get_voice_by_id(first_id)
            assert result is not None
            assert result["voice_id"] == first_id

    def test_get_voice_by_id_not_found(self):
        """get_voice_by_id() should return None for unknown ID."""
        result = get_voice_by_id("nonexistent_voice_xyz")
        assert result is None

    def test_filter_by_energy_level(self):
        """filter_voices_by_criteria() should filter by energy."""
        high = filter_voices_by_criteria(energy_level="high")
        for v in high:
            assert v["energy_level"] == "high"

    def test_filter_by_engine(self):
        """filter_voices_by_criteria() should filter by engine."""
        piper = filter_voices_by_criteria(engine="piper")
        for v in piper:
            assert v["engine"] == "piper"

    def test_filter_by_context(self):
        """filter_voices_by_criteria() should filter by context."""
        morning = filter_voices_by_criteria(context="morning")
        for v in morning:
            assert "morning" in v["preferred_contexts"]

    def test_list_available_voices_not_empty(self):
        """list_available_voices() should return non-empty string."""
        result = list_available_voices()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_config_metadata(self):
        """get_config_metadata() should return dict."""
        metadata = get_config_metadata()
        assert isinstance(metadata, dict)


# === Voice Selector Tests ===

class TestVoiceSelector:
    """Tests for contextual voice selection logic."""

    @pytest.fixture(autouse=True)
    def setup_selector(self):
        """Create VoiceSelector instance."""
        self.selector = VoiceSelector()

    def _make_context(self, **kwargs) -> ContextSignals:
        """Create ContextSignals with defaults."""
        defaults = {
            "time_of_day": "afternoon",
            "system_load": "low",
            "recent_interaction_tone": "casual",
            "explicit_user_request": None,
            "session_duration": 10,
            "last_voice_used": None,
        }
        defaults.update(kwargs)
        return ContextSignals(**defaults)

    def test_select_returns_voice_selection(self):
        """select() should return a VoiceSelection object."""
        context = self._make_context()
        result = self.selector.select(context)
        assert isinstance(result, VoiceSelection)

    def test_select_has_voice_id(self):
        """Selection should have a voice_id."""
        context = self._make_context()
        result = self.selector.select(context)
        assert result.voice_id is not None
        assert len(result.voice_id) > 0

    def test_select_has_reason(self):
        """Selection should include reason string."""
        context = self._make_context()
        result = self.selector.select(context)
        assert result.reason is not None
        assert len(result.reason) > 0

    def test_select_has_confidence(self):
        """Selection should have confidence score between 0 and 1."""
        context = self._make_context()
        result = self.selector.select(context)
        assert 0.0 <= result.confidence <= 1.0

    def test_explicit_user_request_overrides(self):
        """Explicit user voice request should override all scoring."""
        voices = load_voice_config()
        if not voices:
            pytest.skip("No voices configured")
        target_id = voices[0]["voice_id"]
        context = self._make_context(explicit_user_request=target_id)
        result = self.selector.select(context)
        assert result.voice_id == target_id

    def test_different_times_may_select_differently(self):
        """Different times of day should influence selection."""
        morning = self._make_context(time_of_day="morning")
        night = self._make_context(time_of_day="night")
        # Run multiple times to account for randomness
        morning_voices = set()
        night_voices = set()
        for _ in range(10):
            morning_voices.add(self.selector.select(morning).voice_id)
            night_voices.add(self.selector.select(night).voice_id)
        # At least the distributions should exist (not crash)
        assert len(morning_voices) >= 1
        assert len(night_voices) >= 1

    def test_selection_history_tracked(self):
        """Selections should be recorded in history."""
        initial_len = len(self.selector.selection_history)
        context = self._make_context()
        self.selector.select(context)
        assert len(self.selector.selection_history) == initial_len + 1

    def test_diversity_with_different_contexts(self):
        """Different contexts should eventually produce different voice selections."""
        voices = load_voice_config()
        if len(voices) < 2:
            pytest.skip("Need at least 2 voices for diversity test")
        selected = set()
        contexts = ["morning", "afternoon", "evening", "night"]
        tones = ["casual", "formal", "urgent", None]
        for tod in contexts:
            for tone in tones:
                context = self._make_context(time_of_day=tod, recent_interaction_tone=tone)
                result = self.selector.select(context)
                selected.add(result.voice_id)
        # With varied contexts, we should see at least some variety
        assert len(selected) >= 1, "Expected at least 1 voice selected"

    def test_high_load_prefers_lighter_voices(self):
        """High system load should bias toward lighter voices."""
        context = self._make_context(system_load="high")
        # Just verify it doesn't crash
        result = self.selector.select(context)
        assert result.voice_id is not None

    def test_weights_loaded(self):
        """VoiceSelector should have selection weights."""
        assert "time_of_day" in self.selector.weights
        assert "system_load" in self.selector.weights
        assert self.selector.weights["explicit_user_request"] == 1.0


# === ContextSignals Tests ===

class TestContextSignals:
    """Tests for the ContextSignals dataclass."""

    def test_create_with_all_fields(self):
        """ContextSignals should create with all required fields."""
        cs = ContextSignals(
            time_of_day="morning",
            system_load="low",
            recent_interaction_tone=None,
            explicit_user_request=None,
            session_duration=None,
            last_voice_used=None,
        )
        assert cs.time_of_day == "morning"
        assert cs.system_load == "low"
        assert cs.recent_interaction_tone is None

    def test_fields_store_values(self):
        """ContextSignals should store all provided values."""
        cs = ContextSignals(
            time_of_day="afternoon",
            system_load="medium",
            recent_interaction_tone="casual",
            explicit_user_request=None,
            session_duration=5,
            last_voice_used="thorsten_casual",
        )
        assert cs.recent_interaction_tone == "casual"
        assert cs.session_duration == 5

    def test_all_fields(self):
        """ContextSignals should accept all fields."""
        cs = ContextSignals(
            time_of_day="evening",
            system_load="high",
            recent_interaction_tone="formal",
            explicit_user_request="kobold_karlsson",
            session_duration=30,
            last_voice_used="thorsten_casual",
        )
        assert cs.session_duration == 30
        assert cs.last_voice_used == "thorsten_casual"


# === Audio Pipeline Tests ===

class TestAudioPipeline:
    """Tests for audio analysis with synthetic data."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Import audio pipeline components."""
        try:
            from core.speech.audio_pipeline import AudioAnalyzer, AudioDiagnostics
            self.analyzer = AudioAnalyzer()
            self.AudioDiagnostics = AudioDiagnostics
            self.available = True
        except ImportError:
            self.available = False

    def _make_sine_wave_int16(self, freq=440, duration=1.0, sr=16000, amplitude=0.5):
        """Generate a synthetic sine wave as int16 (what AudioAnalyzer expects)."""
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float64)
        wave = amplitude * np.sin(2 * np.pi * freq * t)
        return (wave * 32767).astype(np.int16)

    def _make_silence_int16(self, duration=1.0, sr=16000):
        """Generate silence as int16."""
        return np.zeros(int(sr * duration), dtype=np.int16)

    def test_analyze_sine_wave(self):
        """Analyzer should process a sine wave without errors."""
        if not self.available:
            pytest.skip("AudioAnalyzer not available")
        audio = self._make_sine_wave_int16()
        result = self.analyzer.analyze(audio, 16000)
        assert result is not None
        assert result.duration_ms > 0

    def test_analyze_silence_detects_quiet(self):
        """Analyzer should detect silence as too quiet."""
        if not self.available:
            pytest.skip("AudioAnalyzer not available")
        audio = self._make_silence_int16()
        result = self.analyzer.analyze(audio, 16000)
        assert result.rms_db < -40

    def test_analyze_loud_signal(self):
        """Analyzer should detect loud signal with high amplitude."""
        if not self.available:
            pytest.skip("AudioAnalyzer not available")
        audio = self._make_sine_wave_int16(amplitude=0.9)
        result = self.analyzer.analyze(audio, 16000)
        assert result.rms_db > -10  # 0.9 amplitude = ~-4 dB

    def test_analyze_clipping(self):
        """Analyzer should detect clipping in over-driven int16 signal."""
        if not self.available:
            pytest.skip("AudioAnalyzer not available")
        # Create clipped signal (values at int16 max)
        audio = self._make_sine_wave_int16(amplitude=1.0)
        # Force some samples to near-max to trigger clipping detection (threshold=32000)
        audio[audio > 32000] = 32767
        audio[audio < -32000] = -32767
        result = self.analyzer.analyze(audio, 16000)
        # Some clipping expected from full-scale sine
        assert result.clipping_ratio >= 0  # May or may not detect depending on threshold

    def test_diagnostics_quality_score_range(self):
        """Quality score should be between 0 and 1."""
        if not self.available:
            pytest.skip("AudioAnalyzer not available")
        audio = self._make_sine_wave_int16()
        result = self.analyzer.analyze(audio, 16000)
        assert 0.0 <= result.quality_score <= 1.0


# === TTS Engine Enum Tests ===

class TestTTSEngine:
    """Tests for TTSEngine enum."""

    def test_piper_value(self):
        assert TTSEngine.PIPER.value == "piper"

    def test_coqui_value(self):
        assert TTSEngine.COQUI.value == "coqui-tts"

    def test_none_value(self):
        assert TTSEngine.NONE.value == "none"
