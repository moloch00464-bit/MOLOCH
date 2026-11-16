#!/usr/bin/env python3
"""
Recherche-Schleuse (Issue #8): Quellenpruefung V4 und lokaler Spotify-Abgleich V2.

Run:
    python3 -m pytest tests/test_web_sources.py -v
"""

import os

import pytest

from core.bridge import web_sources as w

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = ["https://www.wave-gotik-treffen.de/bands", "https://monkeypress.de/wgt-2026"]


class TestCheckSources:

    def test_fetched_url_is_kept(self):
        text, rep = w.check_sources("Steht auf https://wave-gotik-treffen.de/bands/.", ALLOWED)
        assert "https://wave-gotik-treffen.de/bands/" in text
        assert rep["unbelegt"] == [] and rep["ohne_quelle"] is False

    def test_subpage_counts_as_same_source(self):
        _, rep = w.check_sources("https://monkeypress.de/wgt-2026/tag-1", ALLOWED)
        assert rep["unbelegt"] == []

    def test_invented_url_is_replaced(self):
        text, rep = w.check_sources("Quelle: https://wgt.de/2026/lineup, glaub mir.", ALLOWED)
        assert w.UNBELEGT in text and "wgt.de" not in text
        assert rep["unbelegt"] == ["https://wgt.de/2026/lineup"]
        assert text.endswith(", glaub mir.")

    def test_same_host_other_path_is_not_enough(self):
        _, rep = w.check_sources("https://monkeypress.de/ganz-anderer-artikel", ALLOWED)
        assert rep["unbelegt"] == ["https://monkeypress.de/ganz-anderer-artikel"]

    def test_no_url_means_no_source(self):
        text, rep = w.check_sources("Ungefähr 200 Bands.", ALLOWED)
        assert text == "Ungefähr 200 Bands." and rep["ohne_quelle"] is True

    def test_lookalike_host_is_rejected(self):
        _, rep = w.check_sources("https://wave-gotik-treffen.de.evil.example/bands", ALLOWED)
        assert len(rep["unbelegt"]) == 1

    def test_urls_in_strips_punctuation(self):
        assert w.urls_in("a (https://x.de/y), b https://z.de.") == ["https://x.de/y", "https://z.de"]

    def test_markdown_link(self):
        text, rep = w.check_sources("[Lineup](https://erfunden.example/x)", ALLOWED)
        assert rep["unbelegt"] == ["https://erfunden.example/x"] and w.UNBELEGT in text


SNIPPET = ("=== MARKUS' SPOTIFY-ARTISTS (Filter: P) — ALLE 3 jemals gehörten, sortiert nach Plays ===\n"
           "  120p — Project Pitchfork (Stamm-Hörer)\n"
           "  7p — Peter Heppner (paar mal gehört)\n"
           "  2p — Pi (mal gehört)\n")


class TestLocalMatches:

    def test_parse_snippet(self):
        assert w.spotify_artists_from_snippet(SNIPPET) == [
            ("Project Pitchfork", 120), ("Peter Heppner", 7), ("Pi", 2)]

    def test_whole_word_match_sorted_by_plays(self):
        web = "WEB-RESULTS:\n- Lineup | x | Peter Heppner, Project Pitchfork, Pink Turns Blue"
        hits = w.local_artist_matches(web, w.spotify_artists_from_snippet(SNIPPET))
        assert hits == [("Project Pitchfork", 120), ("Peter Heppner", 7)]

    def test_short_names_are_ignored(self):
        """'Pi' wuerde in jedem Text mit 'Pink' oder 'Pi' zufaellig treffen."""
        hits = w.local_artist_matches("Pi spielt nicht, Pink schon", [("Pi", 2)])
        assert hits == []

    def test_format(self):
        assert "keine Überschneidung" in w.format_local_matches([])
        text = w.format_local_matches([("A", 3), ("B", 1)])
        assert "lokal" in text and "A (3×)" in text


class TestChatServerWiring:
    """Ohne FastAPI pruefbar: die Stellen im Web-Pfad sind verdrahtet."""

    @pytest.fixture
    def src(self):
        with open(os.path.join(ROOT, "core", "bridge", "chat_server.py"), encoding="utf-8") as f:
            return f.read()

    def test_v4_applied_to_web_answer(self, src):
        assert "web_sources.check_sources(" in src
        assert "web_fetched_urls.append(url)" in src

    def test_v2_switch_keeps_default_and_guards_cloud_prompt(self, src):
        assert 'get("web_private_context", True)' in src
        assert "if spotify_ctx and private_ok:" in src
