#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Character Journal
================================

Single Source of Truth fuer charakter-formende Events.
Append-only JSONL, tagesweise rotiert, auf SSD2.

Phase 2 von Gate 1.5 (Character Evolution Loop).
Phase 4 (kommt spaeter): Distiller liest dieses Journal nachts und
destilliert daraus Mood-Drift + Persoenlichkeits-Updates.

Storage:
  /mnt/moloch-data/memory/journal/YYYY-MM-DD.jsonl   - tagesweise Eintraege
  /mnt/moloch-data/memory/journal/_state.json        - persistenter event_id Counter

Singleton: get_journal()

API:
  journal.write_event(type, interpretation, tension_delta=0.0, context="",
                      tags=None, relevance=None, importance=None, citation=None)
  journal.read_recent(n=50) -> List[Dict]

Schema pro Eintrag:
  ts, event_id, type, interpretation, tension_delta, context,
  recency, relevance, importance, citation, tags
"""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CharacterJournal")

JOURNAL_DIR = "/mnt/moloch-data/memory/journal"
STATE_PATH = os.path.join(JOURNAL_DIR, "_state.json")

ALLOWED_TYPES = frozenset({
    "camera", "audio", "tension", "mode_switch",
    "spotify", "chat", "protective",
})

MAX_CONTEXT_LEN = 200
MAX_INTERPRETATION_LEN = 300


def _utc_iso_ms() -> str:
    """Aktueller UTC-Zeitstempel im ISO-Format mit Millisekunden + Z-Suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_write_json(path: str, data: Any) -> None:
    """JSON atomar schreiben (tempfile + os.replace, NTFS-Fallback).

    Pattern uebernommen aus core/longterm_memory.py:_safe_write_json.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError:
            with open(tmp_path, "r", encoding="utf-8") as f_src:
                content = f_src.read()
            with open(path, "w", encoding="utf-8") as f_dst:
                f_dst.write(content)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[JOURNAL] _state.json Schreiben fehlgeschlagen: {e}")
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


class CharacterJournal:
    """Append-only Character Journal — Single Source of Truth fuer Charakter-Events.

    Thread-safe. Pro write_event ein einzelner JSONL-Append + fsync auf SSD2.
    event_id Counter ueberlebt Reboot via _state.json.
    """

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        self._last_id = self._load_last_id()
        # Dedup: hash -> timestamp (innerhalb 5min-Fenster werden Duplikate verworfen)
        self._dedup_cache: Dict[str, float] = {}
        self._dedup_window_s: float = 300.0
        logger.info(f"[JOURNAL] Initialisiert: dir={JOURNAL_DIR}, last_id={self._last_id}")

    def _load_last_id(self) -> int:
        """Letzten event_id Counter aus _state.json laden (0 wenn neu)."""
        if not os.path.exists(STATE_PATH):
            return 0
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            return int(state.get("last_id", 0))
        except Exception as e:
            logger.warning(f"[JOURNAL] _state.json kaputt, starte bei 0: {e}")
            return 0

    def _save_last_id(self, last_id: int) -> None:
        """event_id Counter atomar persistieren."""
        _safe_write_json(STATE_PATH, {"last_id": last_id, "updated": _utc_iso_ms()})

    def _today_path(self) -> str:
        """Pfad zum heutigen Journal-File."""
        return os.path.join(JOURNAL_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")

    def write_event(
        self,
        type: str,
        interpretation: str,
        tension_delta: float = 0.0,
        context: str = "",
        tags: Optional[List[str]] = None,
        relevance: Optional[float] = None,
        importance: Optional[float] = None,
        citation: Optional[str] = None,
        referenced_event_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Schreibt einen Charakter-Event in das Journal.

        Args:
            type: Eine der ALLOWED_TYPES (camera, audio, tension, mode_switch,
                  spotify, chat, protective).
            interpretation: Abstrakte Bedeutung (KEIN raw data). Pflicht, nicht leer.
            tension_delta: Aenderung der Tension durch diesen Event. Default 0.0.
            context: Kurzer Kontext-String (auto-truncated auf MAX_CONTEXT_LEN).
            tags: Liste von Tag-Strings. Default leer.
            relevance, importance, citation: Distiller-Felder (Phase 4).
                Caller darf optional setzen, sonst null.
            referenced_event_ids: Optional, Liste von event_id-Strings auf die
                dieser Event verweist (z.B. Reflection-Events die andere Events
                zusammenfassen). Wird nur ins Entry-Dict geschrieben wenn nicht leer.

        Returns:
            event_id (z.B. "evt_00000042") bei Erfolg, sonst None.
        """
        # Validation
        if type not in ALLOWED_TYPES:
            logger.warning(f"[JOURNAL] Unbekannter type='{type}' — Eintrag verworfen")
            return None
        if not interpretation or not interpretation.strip():
            logger.warning(f"[JOURNAL] Leere interpretation fuer type={type} — Eintrag verworfen")
            return None

        # Truncation
        interpretation = interpretation.strip()[:MAX_INTERPRETATION_LEN]
        context = (context or "").strip()[:MAX_CONTEXT_LEN]
        tags = list(tags) if tags else []

        with self._lock:
            # Dedup-Check: identische Events in 5min-Fenster verwerfen.
            # Hash-Formel: MD5(type|interpretation|minute)[:8] -> Events gleicher Minute kollidieren.
            _dedup_hash = hashlib.md5(
                f"{type}|{interpretation}|{datetime.now().minute}".encode("utf-8")
            ).hexdigest()[:8]
            now_ts = time.time()
            # GC: alte Eintraege rausschmeissen
            self._dedup_cache = {
                k: v for k, v in self._dedup_cache.items()
                if now_ts - v < self._dedup_window_s
            }
            if _dedup_hash in self._dedup_cache:
                age = now_ts - self._dedup_cache[_dedup_hash]
                logger.debug(
                    f"[JOURNAL] Dedup: {type}/{interpretation[:30]} "
                    f"bereits vor {age:.0f}s geschrieben"
                )
                return None
            self._dedup_cache[_dedup_hash] = now_ts

            new_id = self._last_id + 1
            event_id = f"evt_{new_id:08d}"

            entry = {
                "ts": _utc_iso_ms(),
                "event_id": event_id,
                "type": type,
                "interpretation": interpretation,
                "tension_delta": float(tension_delta),
                "context": context,
                "recency": 1.0,
                "relevance": relevance,
                "importance": importance,
                "citation": citation,
                "tags": tags,
            }
            # Reflection-Linking: nur eintragen wenn Caller eine Liste mitgibt.
            if referenced_event_ids:
                entry["referenced_event_ids"] = list(referenced_event_ids)

            path = self._today_path()
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                logger.error(f"[JOURNAL] Append fehlgeschlagen ({path}): {e}")
                return None

            # Counter erst nach erfolgreichem Append persistieren
            self._last_id = new_id
            self._save_last_id(new_id)
            return event_id

    def read_recent(self, n: int = 50) -> List[Dict]:
        """Letzte N Eintraege ueber bis zu 3 Tagesfiles zurueckgeben.

        Reihenfolge: aelteste zuerst, neueste zuletzt.
        """
        entries: List[Dict] = []
        for days_back in range(3):
            date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            path = os.path.join(JOURNAL_DIR, f"{date}.jsonl")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    day_entries = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            day_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    entries = day_entries + entries  # aelterer Tag voran
            except Exception as e:
                logger.error(f"[JOURNAL] Lesen fehlgeschlagen ({path}): {e}")

        return entries[-n:]


# =============================================================================
# Singleton
# =============================================================================

_instance: Optional[CharacterJournal] = None
_instance_lock = threading.Lock()


def get_journal() -> CharacterJournal:
    """Globale CharacterJournal-Instanz (Singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CharacterJournal()
    return _instance


# =============================================================================
# Self-Test — `python3 -m core.memory.character_journal`
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    j = get_journal()
    start_id = j._last_id

    samples = [
        ("camera", "Markus betritt Bild", 0.0, "sim=0.58", ["entry"]),
        ("audio", "Sprache erkannt (12 Zeichen)", 0.0, "dur=1.4s", []),
        ("tension", "Beleidigung erkannt", 0.31, "whisper:'du depp'", ["rudeness"]),
        ("mode_switch", "Zone guardian->shadow", 0.0, "trigger=tension>0.5", ["shadow"]),
        ("spotify", "Spielt: VNV Nation - Beloved", 0.0, "album=Futureperfect", ["guardian"]),
    ]

    written: List[str] = []
    for t, interp, td, ctx, tags in samples:
        eid = j.write_event(t, interp, tension_delta=td, context=ctx, tags=tags)
        assert eid is not None, f"write_event failed for type={t}"
        written.append(eid)
        print(f"  wrote {eid}: {t} -> {interp}")

    # Sequenz-Check
    expected = [f"evt_{start_id + i + 1:08d}" for i in range(len(samples))]
    assert written == expected, f"event_id-Sequenz falsch:\n  got {written}\n  exp {expected}"

    # Schema-Check
    recent = j.read_recent(len(samples))
    assert len(recent) >= len(samples), f"read_recent: nur {len(recent)} von {len(samples)}"
    last = recent[-1]
    expected_keys = {
        "ts", "event_id", "type", "interpretation", "tension_delta",
        "context", "recency", "relevance", "importance", "citation", "tags",
    }
    assert set(last.keys()) == expected_keys, f"Schema-Mismatch: {set(last.keys())} != {expected_keys}"
    assert last["recency"] == 1.0
    assert last["relevance"] is None
    assert last["importance"] is None
    assert last["citation"] is None

    # Validation-Check
    assert j.write_event("invalid_type", "x") is None
    assert j.write_event("camera", "") is None

    # Distiller-Felder optional
    eid_dist = j.write_event(
        "spotify", "Top-Track erkannt", context="Lieblings",
        tags=["favorite"], importance=0.85, citation="track_index.json",
    )
    assert eid_dist is not None
    rec = j.read_recent(1)[-1]
    assert rec["importance"] == 0.85
    assert rec["citation"] == "track_index.json"

    print(f"\nSelf-Test PASS — geschrieben: {len(written) + 1} Eintraege ({start_id + 1}..{j._last_id})")
    print(f"File: {j._today_path()}")
    print(f"State: {STATE_PATH}")
