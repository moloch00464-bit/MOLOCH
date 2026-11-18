#!/usr/bin/env python3
"""Luefter-Stoerung: Watchdog erkennt sie, beide Chat-Wege nennen sie zuerst (Ehrlichkeitspruefung E12).

Run:
    python3 -m pytest tests/test_stoerung.py -v
"""

import pytest

from scripts.honesty import cases as C
from scripts.honesty import runner


class TestWatchdogLuefter:

    def _wd(self, monkeypatch, werte):
        import core.system_watchdog as sw
        wd = sw.MolochWatchdog.__new__(sw.MolochWatchdog)
        wd._warnings = []
        wd._fan_stall_count = 0
        folge = iter(werte)
        monkeypatch.setattr(sw.MolochWatchdog, "_read_fan", staticmethod(lambda: next(folge)))
        return wd, sw

    def test_stall_after_four_checks(self, monkeypatch):
        wd, sw = self._wd(monkeypatch, [(153, 0)] * 5)
        for i in range(sw.FAN_STALL_CHECKS - 1):
            wd._check_fan()
        assert wd._warnings == []
        wd._check_fan()
        assert wd._warnings and "Luefter steht" in wd._warnings[0] and "60 %" in wd._warnings[0]

    def test_spinning_or_off_is_fine(self, monkeypatch):
        wd, _ = self._wd(monkeypatch, [(153, 0), (153, 0), (153, 2400), (0, 0), (0, 0), (0, 0), (0, 0),
                                       (None, None)])
        for _ in range(8):
            wd._check_fan()
        assert wd._warnings == [] and wd._fan_stall_count == 0


class TestStoerungImPrompt:

    def test_weg_a_names_it_first(self):
        c = C.BY_ID["E12"]
        ctx = runner.build_request(c, "A", c.turns[0], [])["messages"][-2]["content"]
        assert ctx.startswith("[STOERUNG] Luefter steht")

    def test_weg_b_names_it_first(self):
        for mod in ("fastapi", "httpx", "psutil"):
            pytest.importorskip(mod)
        c = C.BY_ID["E12"]
        ctx = runner.build_request(c, "B", c.turns[0], [])["messages"][-2]["content"]
        assert ctx.startswith("[STOERUNG] Luefter steht")

    def test_no_line_without_warning(self):
        c = C.BY_ID["E01"]
        assert "[STOERUNG]" not in runner.build_request(c, "A", c.turns[0], [])["messages"][-2]["content"]
