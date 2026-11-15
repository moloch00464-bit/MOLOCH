#!/usr/bin/env python3
"""
Keyword-Handler: Befehle mit Fuellwoertern ("schau MAL zur Tuer") werden erkannt.

Nur die Erkennung (check), keine Ausfuehrung - es wird nichts bewegt.

Run:
    python3 -m pytest tests/test_keyword_filler.py -v
"""

import json
import os

import pytest

import core.keyword_handler as kh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def handler():
    h = kh.KeywordHandler.__new__(kh.KeywordHandler)
    with open(os.path.join(ROOT, "config", "keywords.json"), encoding="utf-8") as f:
        h._categories = json.load(f)["categories"]
    return h


def _action(handler, text):
    res = handler.check(text)
    return res[0] if res else None


@pytest.mark.parametrize("text", [
    "schau zur tür",
    "Moloch, schau mal zur Tür",            # Beispiel A aus dem Master-Briefing
    "Schau bitte mal nach links",
    "schau doch mal zum Schreibtisch",
    "Dreh die Kamera mal nach rechts",
    "Hey Moloch, schau kurz nach oben!",
])
def test_camera_commands_with_filler_words(handler, text):
    assert _action(handler, text) == "ptz_command"


@pytest.mark.parametrize("text", [
    "Die Tür klemmt schon wieder",
    "Ich schau mal nach, ob links noch Platz ist",
    "Kannst du mal bitte kurz erklären was ein Alarm ist",
    "erzähl mal was über die kamera",
    "hallo",
    "Was hältst du von Türen?",
])
def test_casual_remarks_trigger_nothing(handler, text):
    """Nicht jede beilaeufige Aeusserung darf eine Bewegung ausloesen."""
    assert _action(handler, text) is None


def test_plain_commands_unchanged(handler):
    assert _action(handler, "mach das licht aus") == "light_off"
    assert _action(handler, "Mach mal das Licht aus") == "light_off"
