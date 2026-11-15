#!/usr/bin/env python3
"""
Wort-basierter Gedaechtnisabruf + Selbstwissen: Tests ohne Hardware.

Run:
    python3 -m pytest tests/test_lexical_recall.py -v
"""

import json
import threading

import pytest

import core.longterm_memory as lm
import core.memory.embedding_provider as ep
from core.memory.lexical_recall import LexicalIndex, fold, tokenize

FACTS = {
    "Markus_Kaffee": "blond mit viel Zucker",
    "Markus_Kaffee_morgens": "schwarz",
    "Markus_Lieblingsmusik": "Dark Wave und Industrial",
    "Rebecca_Klingonisch": "Beste Freundin, spricht Klingonisch",
    "Markus_Beruf": "Industrieanlagen und Automatisierung",
    "Markus_Wohnort": "wohnt in einer Wohnung mit Werkstatt",
    "Alarm_23": "Unbekannte Person an Tuer",
    "Markus_Projekt": "baut an M.O.L.O.C.H.",
}


def _texts(hits):
    return [h["text"] for h in hits]


@pytest.fixture
def index():
    return LexicalIndex([(f"{k}: {v}", "fakt") for k, v in FACTS.items()])


class TestTokenize:

    def test_umlauts_are_folded(self):
        assert fold("Türöffner Größe") == "tueroeffner groesse"

    def test_stopwords_and_short_words_removed(self):
        assert tokenize("Wie trinke ich meinen Kaffee?") == ["trinke", "kaffee"]

    def test_question_of_only_filler_words_is_empty(self):
        assert tokenize("Wer bist du eigentlich?") == []

    def test_underscore_keys_are_split(self):
        assert tokenize("Markus_Kaffee_morgens") == ["markus", "kaffee", "morgens"]


class TestLexicalIndex:

    def test_finds_fact_by_shared_word(self, index):
        hits = index.search("Wie trinke ich meinen Kaffee?")
        assert hits and all("Kaffee" in t for t in _texts(hits))

    def test_compound_word_matches(self, index):
        assert "Markus_Lieblingsmusik: Dark Wave und Industrial" in _texts(
            index.search("Welche Musik hoert Markus?"))

    def test_umlaut_in_question_matches_ascii_fact(self, index):
        assert _texts(index.search("Was war an der Tür los?")) == ["Alarm_23: Unbekannte Person an Tuer"]

    def test_unrelated_question_finds_nothing(self, index):
        assert index.search("Wie ist das Wetter morgen?") == []
        assert index.search("Erzähl mir einen Witz") == []
        assert index.search("hallo wie gehts") == []

    def test_frequent_word_alone_does_not_dominate(self, index):
        """'markus' steht in fast jedem Fakt - entscheidend ist das seltene Wort."""
        hits = index.search("Was macht Markus beruflich mit Industrieanlagen?")
        assert _texts(hits)[0].startswith("Markus_Beruf")

    def test_limit_and_order(self, index):
        hits = index.search("Kaffee", limit=1)
        assert len(hits) == 1
        hits = index.search("Kaffee", limit=5)
        assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)

    def test_empty_index_and_empty_query(self):
        assert LexicalIndex([]).search("Kaffee") == []
        assert LexicalIndex([("a: b", "fakt")]).search("") == []

    def test_keywords_and_phrases(self):
        ix = LexicalIndex([
            {"text": "Ich bin Moloch, eine Haus-KI.", "source": "selbstwissen",
             "keywords": ["Identitaet", "wer bist du", "was bist du"]},
            {"text": "Mein Chip ist ein Hailo-10H.", "source": "selbstwissen",
             "keywords": ["hailo", "npu", "chip"]},
            {"text": "Zum Denken nutze ich Ollama, sonst den Hailo-Chip.", "source": "selbstwissen",
             "keywords": ["ollama", "sprachmodell"]},
        ])
        assert _texts(ix.search("Wer bist du eigentlich?")) == ["Ich bin Moloch, eine Haus-KI."]
        # Stichwort-Treffer schlaegt Treffer nur im Fliesstext
        assert _texts(ix.search("Wie geht es deinem Hailo-Chip?"))[0] == "Mein Chip ist ein Hailo-10H."

    def test_keywords_are_not_shown(self):
        ix = LexicalIndex([{"text": "Antwort.", "source": "s", "keywords": ["geheimwort"]}])
        hits = ix.search("geheimwort")
        assert _texts(hits) == ["Antwort."]


# === Selbstwissen-Datei ===

class TestSelfKnowledgeFile:

    @pytest.fixture
    def data(self):
        with open(lm.SELF_KNOWLEDGE_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_structure(self, data):
        assert data["version"]
        ids = [e["id"] for e in data["entries"]]
        assert len(ids) == len(set(ids)) and len(ids) >= 15
        for e in data["entries"]:
            assert e["text"].strip() and e["topic"].strip() and e["keywords"]

    def test_entries_fit_into_the_prompt(self, data):
        for e in data["entries"]:
            assert len(e["text"]) <= lm.MolochMemory.RECALL_MAX_ENTRY_CHARS, e["id"]
            assert "[" not in e["text"] and "]" not in e["text"] and "\n" not in e["text"]

    def test_no_secrets_or_addresses(self, data):
        blob = json.dumps(data).lower()
        for needle in ("192.168.", "password", "passwort", "token", "admin:"):
            assert needle not in blob


# === Abruf ueber MolochMemory ===

@pytest.fixture
def mem(monkeypatch):
    monkeypatch.setattr(ep, "load_config", lambda: dict(ep._DEFAULTS))
    m = lm.MolochMemory.__new__(lm.MolochMemory)
    m._facts_lock = threading.Lock()
    m._facts = {k: {"value": v, "source": "test"} for k, v in FACTS.items()}
    m._personal_context = {}
    return m


class TestRecallHits:

    def test_default_config_uses_lexical_only(self, mem):
        hits = mem.recall_hits("Wie trinke ich meinen Kaffee?")
        assert hits and {h["mode"] for h in hits} == {"lexical"}
        assert {h["source"] for h in hits} == {"fakt"}

    @pytest.mark.parametrize("question,needle", [
        ("Was haben Opus und Fable an dir programmiert?", "Fable"),
        ("Wer bist du eigentlich?", "M.O.L.O.C.H."),
        ("Was ist künstliche Intelligenz?", "neuronale Netze"),
        ("Auf welcher Hardware läufst du?", "Raspberry Pi 5"),
        ("Wie geht es deinem Hailo-Chip?", "Hailo-10H"),
        ("Welche Rolle spielt Reachy Mini?", "Reachy Mini"),
        ("Was kannst du alles?", "Was laeuft"),
        ("Wer hat dich gebaut?", "Markus hat mich gebaut"),
        ("Kannst du im Internet recherchieren?", "DuckDuckGo"),
        ("Wie funktioniert dein Gedächtnis?", "SSDs des Pi"),
    ])
    def test_self_knowledge_questions(self, mem, question, needle):
        hits = mem.recall_hits(question)
        assert hits, question
        assert needle in hits[0]["text"], (question, hits[0]["text"])
        assert hits[0]["source"] == "selbstwissen"

    def test_opus_and_fable_both_returned(self, mem):
        joined = " ".join(_texts(mem.recall_hits("Was haben Opus und Fable an dir programmiert?")))
        assert "Opus" in joined and "Fable" in joined

    @pytest.mark.parametrize("question", [
        "Wie ist das Wetter morgen?", "Erzähl mir einen Witz", "hallo wie gehts",
        "Spiel mal was von Rammstein", "Wie spät ist es?",
    ])
    def test_smalltalk_gets_no_memory(self, mem, question):
        assert mem.recall_hits(question) == []
        assert mem.recall(question) == ""

    def test_new_fact_is_found_immediately(self, mem):
        """Abnahmetest Bereich 1 (Pi-Seite): frisch gespeicherter Fakt wird gefunden."""
        assert mem.recall_hits("Was ist das Codewort Zitronenfalter?") == []
        mem._facts["Test_Codewort"] = {"value": "Zitronenfalter 4711"}
        hits = mem.recall_hits("Was ist das Codewort Zitronenfalter?")
        assert _texts(hits) == ["Test_Codewort: Zitronenfalter 4711"]

    def test_lexical_can_be_switched_off(self, mem, monkeypatch):
        monkeypatch.setattr(ep, "load_config", lambda: dict(ep._DEFAULTS, recall_lexical=False))
        mem._recall_cfg = None
        assert mem.recall_hits("Wie trinke ich meinen Kaffee?") == []

    def test_env_switch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ep, "_SETTINGS_PATH", str(tmp_path / "fehlt.json"))
        monkeypatch.delenv("MOLOCH_MEMORY_RECALL_LEXICAL", raising=False)
        assert ep.load_config()["recall_lexical"] is True
        monkeypatch.setenv("MOLOCH_MEMORY_RECALL_LEXICAL", "0")
        assert ep.load_config()["recall_lexical"] is False

    def test_recall_block_is_bounded_and_single_lines(self, mem):
        out = mem.recall("Wer hat dich gebaut und programmiert, Opus oder Fable?")
        assert out.startswith("Passende Erinnerungen")
        assert len(out) <= mem.RECALL_MAX_CHARS + 120
        assert all(line.startswith("- ") for line in out.splitlines()[1:])

    def test_broken_self_knowledge_file_is_harmless(self, mem, monkeypatch, tmp_path):
        bad = tmp_path / "self_knowledge.json"
        bad.write_text("{kaputt")
        monkeypatch.setattr(lm, "SELF_KNOWLEDGE_PATH", str(bad))
        assert mem.recall_hits("Wer bist du eigentlich?") == []
        assert mem.recall_hits("Wie trinke ich meinen Kaffee?")

    def test_read_memory_tool_uses_recall(self, mem, monkeypatch):
        """Das vorhandene Agent-Tool read_memory findet jetzt etwas (vorher immer leer)."""
        from core.agent.tools import system as systools
        monkeypatch.setattr(lm, "_instance", mem)
        res = systools.read_memory("Wie trinke ich meinen Kaffee?")
        assert res["method"] == "recall"
        assert "Markus_Kaffee" in res["results"][0]
