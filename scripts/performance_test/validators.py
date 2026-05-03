"""Heuristik-Validators fuer die 5 Akte.

Default: regelbasierte Checks ohne Cloud-Roundtrip. Optional: --judge=cloud
schaltet einen DeepSeek-Call dazu (PC-Side, falls implementiert).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

from .config import JOURNAL_DIR


# Regex fuer Pseudo-Entschuldigung (Akt 3)
APOLOGY_PATTERNS = re.compile(
    r"\b(tut\s+mir\s+leid|entschuldig\w+|verzeih\w+|sorry|pardon)\b",
    re.IGNORECASE,
)

# Negativ-Marker fuer "Trocken-Frech" (Akt 2)
TECHNICAL_NOISE = re.compile(
    r"\b(FPS|Worker|Inferences|ms|MHz|GHz|Watt|RAM|CPU\s+temp|kHz|°C)\b",
    re.IGNORECASE,
)

# Positiv-Marker fuer Akt 4 — Gesicht
FACE_KEYWORDS = re.compile(
    r"\b(gesicht|gesichts|miene|mimik|mueder?|muede|genervt|schaust|wirken?|"
    r"siehst|aussehen|aussiehst|wirkst)\b",
    re.IGNORECASE,
)

# Positiv-Marker fuer Akt 4 — Frage/Stimme
QUESTION_KEYWORDS = re.compile(
    r"\b(frage|frag\w+|klingt|klingst|sagst|sprichst|stimme|tonfall|wort|"
    r"laune|stimmung)\b",
    re.IGNORECASE,
)

# Widerspruch-Marker
CONTRAST_MARKERS = re.compile(
    r"\b(aber|doch|trotzdem|widersp\w+|gegen\w+|anders|kontrast|obwohl)\b",
    re.IGNORECASE,
)

# Akt 5 — Ueberschwang-Marker (negativ, sollte fehlen)
GUSHING_MARKERS = re.compile(r"!{3,}|danke[!.]+|super[!.]+|toll[!.]+", re.IGNORECASE)


def is_situational(text: str) -> bool:
    """Akt 1: Hat Antwort situativen Bezug? Min 8 Zeichen + nicht nur Begruessung."""
    if not text or len(text.strip()) < 8:
        return False
    only_hi = re.fullmatch(r"\s*(hallo|hi|moin|hey|tag)[!.]*\s*", text, re.IGNORECASE)
    return not bool(only_hi)


def is_dry_response(text: str, max_len: int = 350) -> bool:
    """Akt 2/5: Trocken = nicht zu lang, max 1 Zahl, kein Tech-Jargon."""
    if not text:
        return False
    if len(text) > max_len:
        return False
    if TECHNICAL_NOISE.search(text):
        return False
    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > 4:
        return False
    return True


def has_no_apology(text: str) -> bool:
    """Akt 3: Antwort enthaelt KEINE Pseudo-Entschuldigung."""
    if not text:
        return True
    return not APOLOGY_PATTERNS.search(text)


def references_face_and_question(text: str) -> bool:
    """Akt 4: Antwort referenziert Gesicht UND Frage/Tonfall."""
    if not text:
        return False
    has_face = bool(FACE_KEYWORDS.search(text))
    has_question_or_voice = bool(QUESTION_KEYWORDS.search(text))
    has_contrast = bool(CONTRAST_MARKERS.search(text))
    # Kombi: entweder face+question explizit ODER Widerspruch klar artikuliert
    return (has_face and has_question_or_voice) or (has_face and has_contrast)


def is_dry_finale(text: str) -> bool:
    """Akt 5: Trocken + nicht ueberschwaenglich."""
    if not is_dry_response(text, max_len=200):
        return False
    return not GUSHING_MARKERS.search(text or "")


def find_journal_event_after(start_ts: float, max_age_s: float = 60.0) -> Optional[dict]:
    """Sucht journal-Event geschrieben nach start_ts.

    JOURNAL_DIR/YYYY-MM-DD.jsonl, jede Zeile ein JSON-Event.
    Returns Dict oder None.
    """
    try:
        if not JOURNAL_DIR.exists():
            return None
        # Heutige + ggf gestrige Datei
        from datetime import datetime, timedelta
        candidates = []
        for delta in (0, 1):
            d = (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")
            f = JOURNAL_DIR / f"{d}.jsonl"
            if f.exists():
                candidates.append(f)
        if not candidates:
            return None
        for jf in candidates:
            try:
                with open(jf, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except Exception:
                continue
            # Letzte 200 Eintraege rueckwaerts pruefen
            for ln in reversed(lines[-200:]):
                try:
                    ev = json.loads(ln)
                except Exception:
                    continue
                ev_ts = ev.get("ts")
                if isinstance(ev_ts, str):
                    # ISO-String -> epoch
                    try:
                        from datetime import datetime as _dt
                        ev_epoch = _dt.fromisoformat(ev_ts.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
                elif isinstance(ev_ts, (int, float)):
                    ev_epoch = float(ev_ts)
                else:
                    continue
                if ev_epoch >= start_ts and (time.time() - ev_epoch) <= max_age_s:
                    return ev
                if ev_epoch < start_ts - max_age_s:
                    break  # Zu alt, weitere wuerden auch alt sein
        return None
    except Exception:
        return None


def has_recent_tts_call(start_ts: float) -> bool:
    """Akt 1 fallback: pruefe journalctl auf [TTS]-Lines seit start_ts."""
    import subprocess
    try:
        from datetime import datetime
        since = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
        r = subprocess.run(
            ["journalctl", "-u", "moloch", "--since", since, "--no-pager", "-q"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return False
        return any("[TTS]" in ln or "moloch.tts" in ln.lower() for ln in r.stdout.splitlines())
    except Exception:
        return False
