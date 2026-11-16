#!/usr/bin/env python3
"""
Quellenpruefung und lokaler Abgleich fuer den Web-Pfad (Recherche-Schleuse, Issue #8).

V4 - Quellen zaehlen nur, wenn sie aus der Suchantwort stammen:
    check_sources() vergleicht jede URL in der Modellantwort mit den URLs, die
    tatsaechlich abgerufen wurden. Fremde URLs werden ersetzt und gemeldet.
    Damit kann ein Sprachmodell keine Quelle erfinden, die nach Beleg aussieht.

V2 - private Daten bleiben im Haus:
    local_artist_matches() gleicht Suchergebnisse lokal mit Markus' Spotify-
    Kuenstlern ab, statt die Hoerdaten in den Cloud-Prompt zu legen.

Reine Funktionen ohne Netz und ohne Zustand.
"""

import re
from typing import Iterable, List, Set, Tuple
from urllib.parse import urlsplit

UNBELEGT = "[Quelle nicht belegt]"
_URL_RE = re.compile(r"https?://[^\s<>\"'`\])}|]+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}>'\"»“”"


def _norm(url: str) -> str:
    """Vergleichsform: ohne Schema, ohne www., ohne Fragment, ohne Schlussstrich."""
    url = url.strip().rstrip(_TRAILING)
    try:
        p = urlsplit(url)
    except ValueError:
        return url.lower()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (p.path or "").rstrip("/")
    query = f"?{p.query}" if p.query else ""
    return f"{host}{path}{query}"


def urls_in(text: str) -> List[str]:
    """Alle URLs in einem Text, ohne angehaengte Satzzeichen."""
    return [u.rstrip(_TRAILING) for u in _URL_RE.findall(text or "")]


def check_sources(answer: str, allowed: Iterable[str]) -> Tuple[str, dict]:
    """URLs in der Antwort gegen die abgerufenen Quellen pruefen.

    Eine URL gilt als belegt, wenn sie (normalisiert) einer abgerufenen URL
    entspricht oder eine Unterseite davon ist. Nicht belegte URLs werden durch
    UNBELEGT ersetzt.

    Returns (bereinigte_antwort, {"belegt": [...], "unbelegt": [...], "ohne_quelle": bool})
    """
    allowed_n: Set[str] = {_norm(u) for u in allowed if u}
    belegt: List[str] = []
    unbelegt: List[str] = []

    def ok(n: str) -> bool:
        return any(n == a or n.startswith(a + "/") or n.startswith(a + "?") for a in allowed_n)

    def repl(m: "re.Match") -> str:
        raw = m.group(0)
        url = raw.rstrip(_TRAILING)
        tail = raw[len(url):]
        if ok(_norm(url)):
            belegt.append(url)
            return raw
        unbelegt.append(url)
        return UNBELEGT + tail

    cleaned = _URL_RE.sub(repl, answer or "")
    return cleaned, {"belegt": belegt, "unbelegt": unbelegt, "ohne_quelle": not belegt}


_ARTIST_LINE = re.compile(r"^\s*(\d+)p\s+[—-]\s+(.+?)(?:\s+\((?:mal gehört|paar mal gehört|kennst du|Stamm-Hörer)\))?\s*$")


def spotify_artists_from_snippet(snippet: str) -> List[Tuple[str, int]]:
    """Kuenstler und Wiedergaben aus _build_spotify_artists_snippet() zurueckgewinnen."""
    out = []
    for line in (snippet or "").splitlines():
        m = _ARTIST_LINE.match(line)
        if m:
            out.append((m.group(2).strip(), int(m.group(1))))
    return out


def local_artist_matches(web_text: str, artists: Iterable[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Kuenstler, die in den Suchergebnissen als ganzes Wort vorkommen (lokal, ohne Modell)."""
    text = (web_text or "").casefold()
    hits = []
    for name, plays in artists:
        n = name.casefold().strip()
        if len(n) < 3:
            continue  # zu kurze Namen treffen zufaellig
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text):
            hits.append((name, plays))
    hits.sort(key=lambda x: -x[1])
    return hits


def format_local_matches(matches: List[Tuple[str, int]], max_n: int = 15) -> str:
    """Kurzer Anhang an die Antwort - entsteht lokal, nicht im Sprachmodell."""
    if not matches:
        return "\n\n(Abgleich mit deinen Spotify-Daten, lokal: keine Überschneidung gefunden.)"
    parts = [f"{name} ({plays}×)" for name, plays in matches[:max_n]]
    more = f" und {len(matches) - max_n} weitere" if len(matches) > max_n else ""
    return "\n\n(Abgleich mit deinen Spotify-Daten, lokal: " + ", ".join(parts) + more + ".)"
