#!/usr/bin/env python3
"""
Face-DB-Doktor und Enrollment-Namenserkennung: Tests ohne Hardware.

Run:
    python3 -m pytest tests/test_face_db_doctor.py -v
"""

import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "face_db_doctor", os.path.join(ROOT, "scripts", "face_db_doctor.py"))
doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doctor)


def _cluster(seed: int, n: int, spread: float = 0.15, dim: int = 64):
    """n aehnliche Einheitsvektoren um ein Zentrum."""
    rng = np.random.default_rng(seed)
    center = rng.normal(size=dim)
    out = []
    for _ in range(n):
        v = center + spread * rng.normal(size=dim) * np.linalg.norm(center) / np.sqrt(dim) * 3
        out.append((v / np.linalg.norm(v)).tolist())
    return out


@pytest.fixture
def db():
    d = {}
    for i, v in enumerate(_cluster(1, 6)):
        d[f"markus#snap_{i}"] = v
    for i, v in enumerate(_cluster(2, 4)):          # gleiche Person, anderer Vektorraum
        d[f"Markus#teach_2026040{i}"] = v
    for i, v in enumerate(_cluster(3, 5)):
        d[f"nicht#cl_17748131{i}"] = v
    for i, v in enumerate(_cluster(4, 3)):
        d[f"rebecca#snap_{i}"] = v
    d["rebecca"] = _cluster(4, 1)[0]
    return d


def _types(rep):
    return [f["type"] for f in rep["findings"]]


class TestAnalyze:

    def test_split_key(self):
        assert doctor.split_key("Markus#teach_20260401_081733") == ("Markus", "teach")
        assert doctor.split_key("markus#cl_1774813130") == ("markus", "cl")
        assert doctor.split_key("markus") == ("markus", "basis")

    def test_groups_and_counts(self, db):
        rep = doctor.analyze(db)
        assert rep["entries"] == len(db) and rep["dim"] == 64
        counts = {(g["name"], g["kind"]): g["count"] for g in rep["groups"]}
        assert counts[("markus", "snap")] == 6 and counts[("Markus", "teach")] == 4
        assert counts[("rebecca", "basis")] == 1

    def test_filler_word_name_is_flagged(self, db):
        hits = [f for f in doctor.analyze(db)["findings"] if f["type"] == "verdaechtiger_name"]
        assert hits == [{"type": "verdaechtiger_name", "name": "nicht", "count": 5}]

    def test_case_variants_are_flagged(self, db):
        hits = [f for f in doctor.analyze(db)["findings"] if f["type"] == "schreibweise"]
        assert hits == [{"type": "schreibweise", "names": ["Markus", "markus"]}]

    def test_incompatible_sets_of_same_person_are_flagged(self, db):
        hits = [f for f in doctor.analyze(db)["findings"] if f["type"] == "inkompatibel"]
        assert len(hits) == 1
        assert {hits[0]["a"], hits[0]["b"]} == {"markus#snap", "Markus#teach"}

    def test_consistent_person_has_no_findings(self, db):
        rep = doctor.analyze({k: v for k, v in db.items() if k.startswith("rebecca")})
        assert rep["findings"] == []

    def test_outlier_is_flagged(self, db):
        only = {k: v for k, v in db.items() if k.startswith("rebecca")}
        only["rebecca#cl_1"] = _cluster(99, 1)[0]
        hits = [f for f in doctor.analyze(only)["findings"] if f["type"] == "ausreisser"]
        assert [h["key"] for h in hits] == ["rebecca#cl_1"]

    def test_broken_entries_do_not_crash(self, db):
        db["kaputt"] = []
        db["falsche_dim"] = [0.1, 0.2]
        rep = doctor.analyze(db)
        assert {f["key"] for f in rep["findings"] if f["type"] == "defekt"} == {"kaputt", "falsche_dim"}

    def test_real_db_in_repo_is_readable(self):
        with open(doctor.DB_PATH, encoding="utf-8") as f:
            rep = doctor.analyze(json.load(f))
        assert rep["dim"] == 512 and rep["entries"] > 0


class TestCleanup:

    def test_select_by_name_is_case_sensitive(self, db):
        doomed = doctor.select_removals(db, ["markus"], [])
        assert len(doomed) == 6 and all(k.startswith("markus#") for k in doomed)

    def test_select_by_kind(self, db):
        doomed = doctor.select_removals(db, [], ["rebecca:snap"])
        assert sorted(doomed) == ["rebecca#snap_0", "rebecca#snap_1", "rebecca#snap_2"]

    def test_bad_kind_spec(self, db):
        with pytest.raises(ValueError):
            doctor.select_removals(db, [], ["ohne_doppelpunkt"])

    def _file(self, tmp_path, db):
        p = tmp_path / "face_embeddings.json"
        p.write_text(json.dumps(db))
        return p

    def test_preview_writes_nothing(self, db, tmp_path, capsys):
        p = self._file(tmp_path, db)
        before = p.read_text()
        assert doctor.main(["--db", str(p), "--remove", "nicht"]) == 0
        assert p.read_text() == before
        assert list(tmp_path.iterdir()) == [p]
        assert "Nur Vorschau" in capsys.readouterr().out

    def test_apply_removes_and_keeps_backup(self, db, tmp_path):
        p = self._file(tmp_path, db)
        before = p.read_text()
        assert doctor.main(["--db", str(p), "--remove", "nicht", "--apply"]) == 0
        after = json.loads(p.read_text())
        assert len(after) == len(db) - 5 and not any(k.startswith("nicht") for k in after)
        backups = [f for f in tmp_path.iterdir() if ".bak_" in f.name]
        assert len(backups) == 1 and backups[0].read_text() == before
        assert not any(f.name.endswith(".tmp") for f in tmp_path.iterdir())

    def test_refuses_to_empty_the_db(self, tmp_path):
        only = {f"nicht#cl_{i}": v for i, v in enumerate(_cluster(3, 3))}
        p = self._file(tmp_path, only)
        assert doctor.main(["--db", str(p), "--remove", "nicht", "--apply"]) == 2
        assert json.loads(p.read_text()) == only

    def test_report_only_writes_nothing(self, db, tmp_path):
        p = self._file(tmp_path, db)
        assert doctor.main(["--db", str(p)]) == 0
        assert list(tmp_path.iterdir()) == [p]


class TestEnrollmentName:
    """Fuellwoerter duerfen nie als Personenname durchgehen (Ursache von 'nicht#cl_*')."""

    @pytest.fixture
    def extract(self):
        from core.keyword_handler import KeywordHandler
        return KeywordHandler.__new__(KeywordHandler)._extract_enrollment_name

    @pytest.mark.parametrize("text,name", [
        ("merk dir das ist Markus", "markus"),
        ("Das ist Rebecca", "rebecca"),
        ("enrollment rebecca", "rebecca"),
        ("einprägen tom", "tom"),
    ])
    def test_real_names(self, extract, text, name):
        assert extract(text) == name

    @pytest.mark.parametrize("text", [
        "merk dir das ist nicht Markus",
        "merk dir das ist nicht",
        "das ist nicht gut",
        "enrollment starten",
        "einprägen bitte",
        "lern mein gesicht",
        "merk dir das ist 3",
        "das ist ja toll",
    ])
    def test_filler_words_give_no_name(self, extract, text):
        assert extract(text) == "unbekannt"
