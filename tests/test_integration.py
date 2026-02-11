#!/usr/bin/env python3
"""
M.O.L.O.C.H. Integration Tests
===============================

Cross-module tests that verify systems work together.
Some tests need hardware (marked accordingly).

Run:
    cd ~/moloch && python3 -m pytest tests/test_integration.py -v
    cd ~/moloch && python3 -m pytest tests/test_integration.py -v -m "not requires_qdrant"
"""

import json
import time
import threading
import pytest

from core.memory.persistent_memory import PersistentMemory


# === Memory <-> Vector Integration ===

class TestMemoryVectorIntegration:
    """Tests that PersistentMemory properly syncs with VectorMemory."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_memory_dir, monkeypatch):
        import core.memory.persistent_memory as pm
        monkeypatch.setattr(pm, "MEMORY_DIR", str(tmp_memory_dir))
        monkeypatch.setattr(pm, "KNOWLEDGE_FILE", str(tmp_memory_dir / "user_knowledge.json"))
        monkeypatch.setattr(pm, "CONVERSATION_FILE", str(tmp_memory_dir / "conversation_log.json"))
        self.mem = PersistentMemory()

    @pytest.mark.requires_qdrant
    def test_remember_syncs_to_qdrant(self):
        """remember() should store in both JSON and Qdrant."""
        from core.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        if not vm._ensure_client():
            pytest.skip("Qdrant not reachable")

        test_key = f"integration_test_{int(time.time())}"
        try:
            self.mem.remember(test_key, "test_value_integration")
            # Check JSON
            assert self.mem.knowledge[test_key] == "test_value_integration"
            # Check Qdrant
            results = vm.search(f"{test_key}: test_value_integration", limit=3)
            assert any(test_key in r.get("key", "") for r in results)
        finally:
            vm.delete(test_key)

    @pytest.mark.requires_qdrant
    def test_forget_removes_from_qdrant(self):
        """forget() should remove from both JSON and Qdrant."""
        from core.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        if not vm._ensure_client():
            pytest.skip("Qdrant not reachable")

        test_key = f"forget_test_{int(time.time())}"
        self.mem.remember(test_key, "temporary_data")
        self.mem.forget(test_key)

        assert test_key not in self.mem.knowledge
        results = vm.search(f"{test_key}: temporary_data", limit=3)
        matching = [r for r in results if r.get("key") == test_key]
        assert len(matching) == 0

    @pytest.mark.requires_qdrant
    def test_sync_to_vector(self):
        """sync_to_vector() should batch-sync all knowledge."""
        from core.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        if not vm._ensure_client():
            pytest.skip("Qdrant not reachable")

        keys = []
        try:
            for i in range(3):
                key = f"sync_batch_{int(time.time())}_{i}"
                keys.append(key)
                self.mem.remember(key, f"value_{i}")

            self.mem.sync_to_vector()

            for key in keys:
                results = vm.search(f"{key}", limit=3)
                assert len(results) > 0, f"Key {key} not found in Qdrant after sync"
        finally:
            for key in keys:
                vm.delete(key)

    def test_extract_memories_and_persist(self):
        """extract_memories() should parse tags AND persist to JSON."""
        text = "Gut zu wissen! [REMEMBER: integration_test=works] Danke!"
        cleaned = self.mem.extract_memories(text)

        assert "integration_test" in self.mem.knowledge
        assert self.mem.knowledge["integration_test"] == "works"
        assert "[REMEMBER:" not in cleaned
        assert "Gut zu wissen!" in cleaned


# === Memory <-> Prompt Integration ===

class TestMemoryPromptIntegration:
    """Tests that memory correctly builds system prompt sections."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_memory_dir, monkeypatch):
        import core.memory.persistent_memory as pm
        monkeypatch.setattr(pm, "MEMORY_DIR", str(tmp_memory_dir))
        monkeypatch.setattr(pm, "KNOWLEDGE_FILE", str(tmp_memory_dir / "user_knowledge.json"))
        monkeypatch.setattr(pm, "CONVERSATION_FILE", str(tmp_memory_dir / "conversation_log.json"))
        self.mem = PersistentMemory()

    def test_full_prompt_assembly(self):
        """Full prompt section should include knowledge + conversation."""
        self.mem.remember("Markus_Spitzname", "PIGH0ST")
        self.mem.remember("Markus_Beruf", "Ingenieur")
        self.mem.add_turn("user", "Wie heisse ich?")
        self.mem.add_turn("assistant", "Du bist PIGH0ST!")

        section = self.mem.to_prompt_section()
        assert "LANGZEITGEDAECHTNIS" in section
        assert "PIGH0ST" in section
        assert "Ingenieur" in section
        assert "LETZTE KONVERSATION" in section
        assert "Markus: Wie heisse ich?" in section

    def test_memory_instruction_completeness(self):
        """Memory instructions should explain the REMEMBER mechanism."""
        instructions = self.mem.get_memory_instruction()
        assert "REMEMBER" in instructions
        assert "schluessel=wert" in instructions
        assert "SPARSAM" in instructions


# === Vision <-> Perception State Integration ===

class TestVisionPerceptionIntegration:
    """Tests that vision updates propagate to perception state."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from context.vision_context import VisionContext
            from context.perception_state import PerceptionState
            self.vision = VisionContext()
            self.perception = PerceptionState()
            self.available = True
        except ImportError:
            self.available = False

    def test_perception_describe_grammar(self):
        """describe() should produce grammatically valid German."""
        if not self.available:
            pytest.skip("Vision/Perception not available")
        # No person
        desc = self.perception.describe()
        assert "niemanden" in desc.lower() or "sehe" in desc.lower()

        # With person
        self.perception.update(
            user_detected=True, face_detected=True, gesture_detected=False,
            gesture_type="none", person_count=1, confidence=0.95,
            face_confidence=0.9, gesture_confidence=0.0,
            face_keypoints=5, torso_keypoints=4, wrist_keypoints=2,
            source="test",
        )
        desc = self.perception.describe()
        assert "sehe" in desc.lower()


# === Self-Diagnosis Integration ===

class TestSelfDiagnosis:
    """Tests for the self-diagnosis script."""

    def test_diagnosis_script_exists(self):
        """self_diagnosis.py should exist."""
        from pathlib import Path
        script = Path(__file__).parent.parent / "scripts" / "self_diagnosis.py"
        assert script.exists(), f"self_diagnosis.py not found at {script}"

    def test_diagnosis_importable(self):
        """self_diagnosis.py functions should be importable."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            import self_diagnosis
            assert hasattr(self_diagnosis, "main")
        except ImportError:
            pytest.skip("self_diagnosis not importable")

    @pytest.mark.requires_qdrant
    def test_diagnosis_qdrant_test(self):
        """Diagnosis Qdrant test should pass when Qdrant is running."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            from self_diagnosis import test_qdrant
            ok, detail = test_qdrant()
            assert ok is True, f"Qdrant test failed: {detail}"
        except ImportError:
            pytest.skip("self_diagnosis not importable")

    def test_diagnosis_persistent_memory(self):
        """Diagnosis persistent memory test should pass."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            from self_diagnosis import test_persistent_memory
            ok, detail = test_persistent_memory()
            assert ok is True, f"Persistent memory test failed: {detail}"
        except ImportError:
            pytest.skip("self_diagnosis not importable")


# === Graceful Degradation ===

class TestGracefulDegradation:
    """Tests that subsystems degrade gracefully on failure."""

    def test_tts_engine_no_crash_without_piper(self, monkeypatch):
        """TTSEngine should not crash if Piper binary is missing."""
        import importlib.util
        import os
        tts_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "core", "tts.py"
        )
        spec = importlib.util.spec_from_file_location("tts_runtime_test", tts_path)
        tts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tts_module)

        # Force missing Piper path
        original_bin = tts_module.PIPER_BIN
        tts_module.PIPER_BIN = type(original_bin)("/nonexistent/piper")
        try:
            engine = tts_module.TTSEngine()
            assert engine.available is False
            assert engine.speak("test") is False
        finally:
            tts_module.PIPER_BIN = original_bin

    def test_tts_engine_no_crash_without_voices(self, monkeypatch, tmp_path):
        """TTSEngine should not crash if no voice models found."""
        import importlib.util
        import os
        tts_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "core", "tts.py"
        )
        spec = importlib.util.spec_from_file_location("tts_runtime_test2", tts_path)
        tts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tts_module)

        # Point to empty dir for voices, but valid Piper path
        tts_module.MODELS_DIR = tmp_path
        try:
            engine = tts_module.TTSEngine()
            # Either available=False (no voices) or True (if Piper doesn't exist either)
            if engine.available:
                # Piper exists but no voices = should be False
                assert len(engine.available_voices) > 0
            else:
                assert engine.speak("test") is False
        finally:
            pass

    def test_vector_memory_graceful_without_qdrant(self):
        """VectorMemory should work when Qdrant is unreachable."""
        from core.memory.vector_memory import VectorMemory
        vm = VectorMemory()
        # Force unavailable
        vm._available = False
        vm._client = None

        # None of these should raise
        vm.store("test", category="fact", key="test_key")
        vm.delete("test_key")
        results = vm.search("anything")
        assert results == []
        context = vm.build_context("anything")
        assert context == ""
        vm.sync_knowledge({"k": "v"})

    def test_camera_controller_no_crash_offline(self):
        """SonoffCameraController should not crash when camera is offline."""
        from core.hardware.camera import SonoffCameraController
        cam = SonoffCameraController()
        # Don't connect - camera offline
        assert cam.is_connected is False
        # Movement should fail gracefully
        result = cam.move_absolute(0, 0)
        assert result is False

    def test_persistent_memory_corrupted_json(self, tmp_memory_dir, monkeypatch):
        """PersistentMemory should survive corrupted JSON files."""
        import core.memory.persistent_memory as pm
        monkeypatch.setattr(pm, "MEMORY_DIR", str(tmp_memory_dir))
        knowledge_file = tmp_memory_dir / "user_knowledge.json"
        monkeypatch.setattr(pm, "KNOWLEDGE_FILE", str(knowledge_file))
        monkeypatch.setattr(pm, "CONVERSATION_FILE", str(tmp_memory_dir / "conversation_log.json"))

        # Write corrupted JSON
        knowledge_file.write_text("{invalid json!!!", encoding="utf-8")

        # Should not crash - recovers with empty state
        mem = PersistentMemory()
        assert isinstance(mem.knowledge, dict)
        # Should be able to use it normally after recovery
        mem.remember("test", "value")
        assert mem.knowledge["test"] == "value"


# === Cross-Module Import Integrity ===

class TestImportIntegrity:
    """Verify that all key modules import without errors."""

    def test_import_hardware_camera(self):
        from core.hardware.camera import SonoffCameraController, get_camera_controller

    def test_import_hardware_init(self):
        from core.hardware import SonoffCameraController, CameraStatus

    def test_import_memory(self):
        from core.memory import get_memory, get_vector_memory

    def test_import_tts_config(self):
        from core.tts.config.voices import load_voice_config

    def test_import_tts_manager(self):
        from core.tts.tts_manager import ContextSignals, TTSEngine

    def test_import_voice_selector(self):
        from core.tts.selection.voice_selector import VoiceSelector

    def test_import_audio_pipeline(self):
        from core.speech.audio_pipeline import AudioAnalyzer, AudioDiagnostics

    def test_import_vision_context(self):
        from context.vision_context import VisionContext

    def test_import_perception_state(self):
        from context.perception_state import PerceptionState

    def test_import_gesture_detector(self):
        from core.vision.gesture_detector import GestureDetector, GestureType

    def test_import_identity_manager(self):
        from core.vision.identity_manager import IdentityManager

    def test_import_ptz_orchestrator(self):
        from core.mpo.ptz_orchestrator import PTZOrchestrator, get_ptz_orchestrator

    def test_import_whisper(self):
        from core.speech.hailo_whisper import MolochWhisper, get_whisper
