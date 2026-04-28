#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
journal_scorer — Nightly Relevanz- und Importance-Scoring fuer Journal-Events.

Laeuft als systemd-Timer um 23:00 Uhr (oder manuell via CLI).
Liest journal/YYYY-MM-DD.jsonl, schreibt scored_YYYY-MM-DD.jsonl mit
zusaetzlichen Feldern relevance + importance pro Event.

Relevance: TF-IDF-Proxy via Keyword-Uebereinstimmung mit letzten Chat-Eintraegen
(simpel, kein Embedding-Modell noetig auf Pi).

Importance: Heuristik nach Event-Typ:
  protective + tension_delta>0.3: 0.9
  chat mit tension_delta!=0:      0.8
  camera + Markus:               0.5
  spotify + zone_change:         0.4
  sonst:                         0.2

CLI:
  python3 -m core.memory.journal_scorer            # Gestern
  python3 -m core.memory.journal_scorer 2026-04-27 # spez. Datum
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("JournalScorer")
JOURNAL_DIR = Path("/mnt/moloch-data/memory/journal")

# Keyword-Sets fuer Relevanz-Pruefung (informativ, momentan ungenutzt)
_TENSION_KEYWORDS = {"beleidigung", "rudeness", "appeasement", "konflikt", "anger"}
_CAMERA_KEYWORDS = {"markus", "person", "gesicht", "erkannt", "unbekannt"}
_SPOTIFY_KEYWORDS = {"musik", "spotify", "song", "track", "zone"}


def _score_importance(event: Dict[str, Any]) -> float:
    """Heuristik-basiertes Importance-Scoring nach Event-Typ + Tension-Delta."""
    etype = event.get("type", "")
    tags = event.get("tags", []) or []
    td = abs(float(event.get("tension_delta", 0.0) or 0.0))

    if "protective" in tags or etype == "protective":
        return 0.9
    if etype == "tension" and td > 0.3:
        return 0.8
    if etype == "chat" and td > 0.0:
        return 0.8
    if etype == "camera" and "markus" in str(event).lower():
        return 0.5
    if etype == "spotify":
        return 0.4
    return 0.2


def _score_relevance(event: Dict[str, Any], chat_words: set) -> float:
    """Keyword-Overlap mit Chat-History als Relevanz-Proxy.

    Je mehr Wort-Treffer zwischen Event-Text und gesammelten Chat-Words,
    desto hoeher der Score (capped bei 1.0). Pi-tauglich, kein Embedding noetig.
    """
    event_text = " ".join([
        str(event.get("type", "")),
        str(event.get("interpretation", "")),
        str(event.get("context", "")),
    ]).lower()
    words = set(event_text.split())
    overlap = len(words & chat_words)
    return min(1.0, overlap * 0.15)


def score_day(target_date: date) -> int:
    """Scored alle Events eines Tages. Gibt Anzahl gescoredter Events zurueck."""
    src = JOURNAL_DIR / f"{target_date.isoformat()}.jsonl"
    dst = JOURNAL_DIR / f"scored_{target_date.isoformat()}.jsonl"

    if not src.exists():
        logger.info(f"[SCORER] Kein Journal fuer {target_date}: {src}")
        return 0

    # Pass 1: Chat-Words aus allen Chat-Events sammeln
    chat_words: set = set()
    try:
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "chat":
                    chat_words.update(str(ev.get("context", "")).lower().split())
                    chat_words.update(str(ev.get("interpretation", "")).lower().split())
    except Exception as e:
        logger.warning(f"[SCORER] Chat-Words-Pass fehlgeschlagen: {e}")

    # Pass 2: Scoring + Schreiben
    count = 0
    with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ev["importance"] = _score_importance(ev)
                ev["relevance"] = _score_relevance(ev, chat_words)
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning(f"[SCORER] Event parse error: {e}")

    logger.info(f"[SCORER] {target_date}: {count} Events gescored -> {dst}")
    return count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    target = date.today() - timedelta(days=1)
    if len(sys.argv) > 1:
        try:
            target = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"Ungueltiges Datum: {sys.argv[1]} (erwarte YYYY-MM-DD)")
            sys.exit(2)
    n = score_day(target)
    print(f"Gescored: {n} Events fuer {target}")
