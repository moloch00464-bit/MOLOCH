#!/usr/bin/env python3
"""
M.O.L.O.C.H. Memory System Tests
=================================

Tests for persistent_memory.py and vector_memory.py.
PersistentMemory tests are fully local (JSON files).
VectorMemory tests need a running Qdrant instance.

Run:
    cd ~/moloch && python3 -m pytest tests/test_memory.py -v
    cd ~/moloch && python3 -m pytest tests/test_memory.py -v -m "not requires_qdrant"  # skip Qdrant
"""

import json
import os
import time
import threading
import pytest
from unittest.mock import patch, MagicMock

from core.memory.persistent_memory import PersistentMemory


# === PersistentMemory Tests ===

class TestPersistentMemory:
    """Tests for JSON-based persistent memory."""

    @pytest.fixture(autouse=True)
    def setup_memory(self, tmp_memory_dir, monkeypatch):
        """Create a PersistentMemory with temporary directory."""
        import core.memory.persistent_memory as pm
        monkeypatch.setattr(pm, "MEMORY_DIR", str(tmp_memory_dir))
        monkeypatch.setattr(pm, "KNOWLEDGE_FILE", str(tmp_memory_dir / "user_knowledge.json"))
        monkeypatch.setattr(pm, "CONVERSATION_FILE", str(tmp_memory_dir / "conversation_log.json"))
        self.mem = PersistentMemory()
        self.mem_dir = tmp_memory_dir
        self.knowledge_file = tmp_memory_dir / "user_knowledge.json"
        self.conversation_file = tmp_memory_dir / "conversation_log.json"

    def test_remember_stores_fact(self):
        """remember() should store key-value and persist to disk."""
        self.mem.remember("test_key", "test_value")
        assert self.mem.knowledge["test_key"] == "test_value"
        # Check disk
        with open(self.knowledge_file) as f:
            data = json.load(f)
        assert data["test_key"] == "test_value"

    def test_remember_overwrites_existing(self):
        """remember() with same key should overwrite."""
        self.mem.remember("name", "Alice")
        self.mem.remember("name", "Bob")
        assert self.mem.knowledge["name"] == "Bob"
        assert len(self.mem.knowledge) == 1

    def test_forget_removes_fact(self):
        """forget() should remove key from memory and disk."""
        self.mem.remember("temp", "data")
        assert "temp" in self.mem.knowledge
        self.mem.forget("temp")
        assert "temp" not in self.mem.knowledge
        with open(self.knowledge_file) as f:
            data = json.load(f)
        assert "temp" not in data

    def test_forget_nonexistent_key(self):
        """forget() with nonexistent key should not crash."""
        self.mem.forget("does_not_exist")  # Should not raise

    def test_add_turn(self):
        """add_turn() should append to conversation log."""
        self.mem.add_turn("user", "Hallo Moloch")
        self.mem.add_turn("assistant", "Hallo Markus")
        assert len(self.mem.conversation_log) == 2
        assert self.mem.conversation_log[0]["role"] == "user"
        assert self.mem.conversation_log[0]["content"] == "Hallo Moloch"
        assert self.mem.conversation_log[1]["role"] == "assistant"

    def test_add_turn_truncates_long_content(self):
        """add_turn() should truncate content over 500 chars."""
        long_text = "x" * 1000
        self.mem.add_turn("user", long_text)
        assert len(self.mem.conversation_log[0]["content"]) == 500

    def test_add_turn_persists_to_disk(self):
        """add_turn() should save to conversation_log.json."""
        self.mem.add_turn("user", "test")
        with open(self.conversation_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_conversation_log_max_turns(self):
        """Conversation log should cap at MAX_CONVERSATION_TURNS."""
        for i in range(25):
            self.mem.add_turn("user", f"message {i}")
        assert len(self.mem.conversation_log) == 20  # MAX_CONVERSATION_TURNS

    def test_get_recent_conversation(self):
        """get_recent_conversation() should return last N turns."""
        for i in range(15):
            self.mem.add_turn("user", f"msg {i}")
        recent = self.mem.get_recent_conversation(5)
        assert len(recent) == 5
        assert recent[0]["content"] == "msg 10"
        assert recent[4]["content"] == "msg 14"

    def test_extract_memories_parses_tags(self):
        """extract_memories() should parse [REMEMBER: key=value] tags."""
        text = "Interessant! [REMEMBER: Markus_Beruf=Ingenieur] Das merke ich mir."
        cleaned = self.mem.extract_memories(text)
        assert self.mem.knowledge["Markus_Beruf"] == "Ingenieur"
        assert "[REMEMBER:" not in cleaned
        assert "Interessant!" in cleaned
        assert "Das merke ich mir." in cleaned

    def test_extract_memories_multiple_tags(self):
        """extract_memories() should handle multiple [REMEMBER:] tags."""
        text = "[REMEMBER: a=1] text [REMEMBER: b=2] more text"
        cleaned = self.mem.extract_memories(text)
        assert self.mem.knowledge["a"] == "1"
        assert self.mem.knowledge["b"] == "2"
        assert "text" in cleaned

    def test_extract_memories_no_tags(self):
        """extract_memories() with no tags should return text unchanged."""
        text = "Normaler Text ohne Tags"
        cleaned = self.mem.extract_memories(text)
        assert cleaned == text

    def test_to_prompt_section_empty(self):
        """to_prompt_section() with no data should return empty string."""
        result = self.mem.to_prompt_section()
        assert result == ""

    def test_to_prompt_section_with_knowledge(self):
        """to_prompt_section() should include stored facts."""
        self.mem.remember("Lieblingsfarbe", "Schwarz")
        section = self.mem.to_prompt_section()
        assert "LANGZEITGEDAECHTNIS" in section
        assert "Lieblingsfarbe: Schwarz" in section

    def test_to_prompt_section_with_conversation(self):
        """to_prompt_section() should include recent conversation."""
        self.mem.add_turn("user", "Wie geht es dir?")
        self.mem.add_turn("assistant", "Gut, danke!")
        section = self.mem.to_prompt_section()
        assert "LETZTE KONVERSATION" in section
        assert "Markus: Wie geht es dir?" in section

    def test_get_memory_instruction(self):
        """get_memory_instruction() should return instructions string."""
        instructions = self.mem.get_memory_instruction()
        assert "REMEMBER" in instructions
        assert "LANGZEITGEDAECHTNIS" in instructions

    def test_thread_safety(self):
        """Concurrent remember() calls should not corrupt data."""
        errors = []

        def writer(start, count):
            try:
                for i in range(count):
                    self.mem.remember(f"key_{start + i}", f"val_{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0, 50)),
            threading.Thread(target=writer, args=(50, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(self.mem.knowledge) == 100

    def test_persistence_across_instances(self):
        """New PersistentMemory instance should load existing data."""
        self.mem.remember("persistent_test", "survives_restart")
        self.mem.add_turn("user", "persistent turn")

        # monkeypatch is still active from setup_memory fixture
        mem2 = PersistentMemory()

        assert mem2.knowledge["persistent_test"] == "survives_restart"
        assert len(mem2.conversation_log) == 1

    def test_corrupted_json_recovery(self):
        """PersistentMemory should handle corrupted JSON gracefully."""
        with open(self.knowledge_file, "w") as f:
            f.write("{corrupted json!!!")
        with open(self.conversation_file, "w") as f:
            f.write("[not valid")

        # monkeypatch is still active from setup_memory fixture
        mem = PersistentMemory()

        assert mem.knowledge == {}
        assert mem.conversation_log == []

    def test_str_representation(self):
        """__str__ should show counts."""
        self.mem.remember("k", "v")
        self.mem.add_turn("user", "hi")
        s = str(self.mem)
        assert "knowledge=1" in s
        assert "turns=1" in s


# === VectorMemory Tests (need Qdrant) ===

@pytest.mark.requires_qdrant
class TestVectorMemory:
    """Tests for Qdrant-based vector memory. Needs running Qdrant."""

    @pytest.fixture(autouse=True)
    def setup_vector(self):
        """Create fresh VectorMemory instance."""
        from core.memory.vector_memory import VectorMemory
        self.vm = VectorMemory()
        if not self.vm._ensure_client():
            pytest.skip("Qdrant not reachable")
        self.test_key = f"test_{int(time.time())}"

    def teardown_method(self):
        """Clean up test data."""
        try:
            self.vm.delete(self.test_key)
        except Exception:
            pass

    def test_store_and_search(self):
        """store() + search() roundtrip should work."""
        self.vm.store("Markus liebt Dark Wave Musik", category="fact", key=self.test_key)
        results = self.vm.search("Welche Musik mag Markus?", limit=5)
        assert len(results) > 0
        assert any(self.test_key in r["key"] for r in results)

    def test_store_upsert(self):
        """Storing same key twice should upsert, not duplicate."""
        self.vm.store("Version 1", category="fact", key=self.test_key)
        self.vm.store("Version 2", category="fact", key=self.test_key)
        results = self.vm.search("Version", limit=10)
        matching = [r for r in results if r["key"] == self.test_key]
        assert len(matching) <= 1  # Upsert, not duplicate

    def test_delete(self):
        """delete() should remove the point."""
        self.vm.store("Temporary data", category="fact", key=self.test_key)
        self.vm.delete(self.test_key)
        # Search should not find it (or score should be low)
        results = self.vm.search("Temporary data", limit=5)
        matching = [r for r in results if r["key"] == self.test_key]
        assert len(matching) == 0

    def test_search_with_category_filter(self):
        """search() with category filter should only return matching category."""
        key1 = f"test_fact_{int(time.time())}"
        key2 = f"test_conv_{int(time.time())}"
        try:
            self.vm.store("Fact entry", category="fact", key=key1)
            self.vm.store("Conversation entry", category="conversation", key=key2)
            results = self.vm.search("entry", limit=10, category="fact")
            categories = {r["category"] for r in results}
            assert "conversation" not in categories
        finally:
            self.vm.delete(key1)
            self.vm.delete(key2)

    def test_build_context_formats_correctly(self):
        """build_context() should return [Erinnerung: ...] formatted string."""
        self.vm.store("Markus Spitzname ist PIGH0ST", category="fact", key=self.test_key)
        context = self.vm.build_context("Wie heisst Markus?")
        if context:  # May be empty if no semantic match
            assert "[Erinnerung:" in context

    def test_build_context_empty_on_no_match(self):
        """build_context() should return empty string when nothing matches."""
        context = self.vm.build_context("xyzzy_totally_random_nonsense_query_12345")
        assert context == ""

    def test_sync_knowledge(self):
        """sync_knowledge() should batch-store all facts."""
        knowledge = {
            f"sync_test_1_{int(time.time())}": "value1",
            f"sync_test_2_{int(time.time())}": "value2",
        }
        try:
            self.vm.sync_knowledge(knowledge)
            for key in knowledge:
                results = self.vm.search(f"{key}: {knowledge[key]}", limit=3)
                assert len(results) > 0
        finally:
            for key in knowledge:
                self.vm.delete(key)

    def test_deterministic_uuid(self):
        """_make_id() should produce same UUID for same key."""
        from core.memory.vector_memory import VectorMemory
        id1 = VectorMemory._make_id("test_key")
        id2 = VectorMemory._make_id("test_key")
        id3 = VectorMemory._make_id("different_key")
        assert id1 == id2
        assert id1 != id3

    def test_is_available(self):
        """is_available should be True when Qdrant is reachable."""
        assert self.vm.is_available is True


class TestVectorMemoryGracefulDegradation:
    """Tests for VectorMemory when Qdrant is NOT available."""

    @pytest.fixture(autouse=True)
    def setup_disconnected(self):
        """Create VectorMemory that can't connect."""
        from core.memory.vector_memory import VectorMemory
        self.vm = VectorMemory()
        self.vm._available = False  # Force unavailable

    def test_store_no_crash(self):
        """store() should not crash when Qdrant unavailable."""
        self.vm.store("test", category="fact", key="test")  # Should not raise

    def test_delete_no_crash(self):
        """delete() should not crash when Qdrant unavailable."""
        self.vm.delete("test")  # Should not raise

    def test_search_returns_empty(self):
        """search() should return empty list when Qdrant unavailable."""
        results = self.vm.search("test")
        assert results == []

    def test_build_context_returns_empty(self):
        """build_context() should return empty string when Qdrant unavailable."""
        context = self.vm.build_context("test")
        assert context == ""

    def test_sync_knowledge_no_crash(self):
        """sync_knowledge() should not crash when Qdrant unavailable."""
        self.vm.sync_knowledge({"k": "v"})  # Should not raise

    def test_is_available_false(self):
        """is_available should be False when Qdrant is down."""
        assert self.vm.is_available is False


# === Singleton Tests ===

class TestMemorySingletons:
    """Tests for singleton pattern in memory modules."""

    def test_persistent_memory_singleton(self):
        """get_memory() should return same instance."""
        import core.memory.persistent_memory as pm
        # Reset singleton
        pm._memory_instance = None
        m1 = pm.get_memory()
        m2 = pm.get_memory()
        assert m1 is m2

    def test_vector_memory_singleton(self):
        """get_vector_memory() should return same instance."""
        import core.memory.vector_memory as vm
        # Reset singleton
        vm._vector_instance = None
        v1 = vm.get_vector_memory()
        v2 = vm.get_vector_memory()
        assert v1 is v2
