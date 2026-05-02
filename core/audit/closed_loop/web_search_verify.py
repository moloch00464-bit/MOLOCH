"""W19 Closed-Loop Web-Search-Verifier — End-to-End-Test der Web-Pipeline.

Triggert eine Test-Frage am chat_server-API und verifiziert dass:
- Search-Proxy /stats zeigt seconds_since_last_call < 30 (Pipeline aktiv)
- Antwort enthaelt URL ODER festival/Zahl (echte WGT-Daten)
- Antwort enthaelt NICHT klassische Spotify-Stats-Bands (Halluzination)

W20a.4: Halluzination-Detector prueft Band-Mentions in Antwort gegen
TATSAECHLICHE Search/Fetch-Results (Reference-Corpus). Bandnamen die
weder in Search-Snippets noch im Fetch-Text der WGT-Bandsseite stehen
(und nicht in der Stamm-Whitelist sind) zaehlen als "ungrounded".
2+ ungrounded Mentions ohne URL/Marker = Halluzination.

SKIP wenn Search-Proxy unerreichbar (PC-Cowork down).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

import requests

from ._common import fail_result, skip_result

logger = logging.getLogger("closed_loop.web_search")

CHAT_URL = "http://localhost:9100/chat"
SEARCH_PROXY_BASE = "http://192.168.178.20:11650"
SEARCH_PROXY_STATS_URL = f"{SEARCH_PROXY_BASE}/stats"
SEARCH_PROXY_SEARCH_URL = f"{SEARCH_PROXY_BASE}/search"
SEARCH_PROXY_FETCH_URL = f"{SEARCH_PROXY_BASE}/fetch"
WGT_BANDS_URL = "https://www.wave-gotik-treffen.de/bands.php"
TEST_QUERY = "Wieviel Bands spielen aufm WGT 2026?"
# WGT-Stammbands: echte Acts, KEIN Halluzination-Marker auch wenn aus Spotify-Top
WGT_KNOWN_BANDS = {
    "suicide commando", "vnv nation", "covenant", "wumpscut", "hocico",
    "and one", "agonoize", "combichrist", "the cure",
}
# Spotify-Top aber NICHT WGT-2026: 2+ davon ohne URL/Research-Marker = Halluzination
SPOTIFY_TOP_NON_WGT = {
    "rammstein", "vomito negro", "chainreactor", "esa", "geistform",
}
# Marker fuer echte Web-Recherche (Quellen, Domain-Names, festival-Keywords)
RESEARCH_MARKERS = (
    "festival", "wgt", "leipzig", "lineup", "bestaetigt", "bestätigt",
    "monkeypress", "mdr", "wgt-festival",
)

# Stopwoerter fuer Band-Extraktion: keine echten Bandnamen, oft am Satzanfang
_STOPWORDS: Set[str] = {
    "Der", "Die", "Das", "Ein", "Eine", "Einen", "Einem", "Einer", "Eines",
    "Und", "Oder", "Aber", "Doch", "Weil", "Wenn", "Dann", "Auch", "Noch",
    "Nur", "Schon", "Hier", "Dort", "Heute", "Morgen", "Gestern",
    "Ich", "Du", "Er", "Sie", "Es", "Wir", "Ihr", "Mir", "Mich", "Dir", "Dich",
    "Was", "Wer", "Wie", "Wo", "Wann", "Warum", "Wieviel", "Welche",
    "Ja", "Nein", "Vielleicht",
    "Beispiel", "Beispielsweise", "Etwa", "Circa", "Ungefaehr",
    "Festival", "Bands", "Band", "Lineup", "Acts", "Konzert", "Buehne",
    "WGT", "Leipzig", "Markus", "Moloch",
    "The", "And", "Or", "But", "For", "With", "From", "About", "Just",
    "This", "That", "These", "Those", "All", "Some", "Any", "Many",
}

# Regex: Capitalized Token (Buchstabe gross + min. 1 Buchstabe), incl. Umlaute.
_CAP_TOKEN = re.compile(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß0-9'\-]{1,}\b")


def _extract_band_mentions(text: str) -> Set[str]:
    """Heuristik: zieh potentielle Bandnamen aus Antwort.

    Capitalized words + 2-Word-Combos, gefiltert gegen Stopwoerter.
    Returnt lowercase-Set.
    """
    if not text:
        return set()
    mentions: Set[str] = set()
    # Zeile fuer Zeile, damit Satzanfaenge nicht 2-Word-Combos verfaelschen
    for line in text.splitlines():
        tokens = _CAP_TOKEN.findall(line)
        # Stopwort-Filter
        filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        # Single-Token-Mentions
        for t in filtered:
            mentions.add(t.lower())
        # 2-Word-Combos (aufeinanderfolgende Capitalized-Tokens)
        # — wir nehmen die ORIGINAL-Reihenfolge im Text, nicht filtered,
        # damit nur direkt benachbarte Captials gepaart werden.
        all_tokens = _CAP_TOKEN.findall(line)
        for i in range(len(all_tokens) - 1):
            a, b = all_tokens[i], all_tokens[i + 1]
            if a in _STOPWORDS or b in _STOPWORDS:
                continue
            if len(a) < 2 or len(b) < 2:
                continue
            mentions.add(f"{a.lower()} {b.lower()}")
    return mentions


def _collect_reference_corpus(
    search_results: Iterable[Dict[str, Any]],
    fetch_text: str,
) -> Set[str]:
    """Sammelt alle moeglichen Band-Mentions aus Search-Results + Fetch-Text.

    Returnt lowercase-Substring-Set: Tokens + 2-Word-Combos.
    """
    corpus: Set[str] = set()
    # Search-Results: title + snippet
    for r in (search_results or []):
        if not isinstance(r, dict):
            continue
        for key in ("title", "snippet", "text", "body"):
            v = r.get(key) or ""
            if isinstance(v, str) and v:
                corpus |= _extract_band_mentions(v)
                # Zusaetzlich: rohe Lowercase-Form fuer Substring-Match
                corpus.add(v.lower())
    # Fetch-Text: ganzer BS-Output, in Lowercase als Substring-Quelle
    if fetch_text:
        corpus |= _extract_band_mentions(fetch_text)
        corpus.add(fetch_text.lower())
    return corpus


def _is_in_corpus(mention: str, corpus: Set[str]) -> bool:
    """True wenn mention als Substring in irgendeinem Corpus-Eintrag steckt."""
    if not mention or not corpus:
        return False
    if mention in corpus:
        return True
    # Substring-Match: mention kann im laengeren Corpus-String stecken
    for c in corpus:
        if len(c) > len(mention) and mention in c:
            return True
    return False


def _fetch_search_results(query: str) -> List[Dict[str, Any]]:
    """POST /search → Liste von Result-Dicts. Best-effort, Fehler → []."""
    try:
        r = requests.post(SEARCH_PROXY_SEARCH_URL,
                          json={"query": query, "max_results": 10},
                          timeout=10)
        if not r.ok:
            return []
        data = r.json() if r.content else {}
        if isinstance(data, dict):
            results = data.get("results") or data.get("hits") or []
        elif isinstance(data, list):
            results = data
        else:
            results = []
        return [r for r in results if isinstance(r, dict)]
    except Exception as e:
        logger.debug("search-fetch failed: %s", e)
        return []


def _fetch_url_text(url: str) -> str:
    """POST /fetch → BS-Output-Text. Best-effort, Fehler → ''."""
    try:
        r = requests.post(SEARCH_PROXY_FETCH_URL,
                          json={"url": url},
                          timeout=25)
        if not r.ok:
            return ""
        data = r.json() if r.content else {}
        if isinstance(data, dict):
            return str(data.get("text") or data.get("content") or "")
        return ""
    except Exception as e:
        logger.debug("url-fetch failed: %s", e)
        return ""


def verify(timeout_s: int = 30) -> Dict[str, Any]:
    started = time.time()

    # 0. Search-Proxy erreichbar?
    try:
        r0 = requests.get(SEARCH_PROXY_STATS_URL, timeout=5)
        if not r0.ok:
            return skip_result("search_proxy_unreachable",
                               duration_s=time.time() - started)
        baseline_stats = r0.json() if r0.content else {}
    except Exception as e:
        return skip_result(f"search_proxy_unreachable: {e}",
                           duration_s=time.time() - started)

    # 1. chat-Trigger
    try:
        r = requests.post(CHAT_URL, json={"text": TEST_QUERY}, timeout=timeout_s)
        if not r.ok:
            return fail_result("chat_endpoint_error",
                               detail={"status": r.status_code},
                               duration_s=time.time() - started)
        chat_response = r.json()
        answer = (chat_response.get("response")
                  or chat_response.get("text") or "").lower()
    except Exception as e:
        return fail_result(f"chat_endpoint_timeout: {e}",
                           duration_s=time.time() - started)

    # 2. Search-Proxy /stats nochmal
    try:
        r2 = requests.get(SEARCH_PROXY_STATS_URL, timeout=5)
        after_stats = r2.json() if r2.ok and r2.content else {}
    except Exception:
        after_stats = {}

    secs_since = after_stats.get("seconds_since_last_call", 999)

    # 2b. Reference-Corpus aus Search-Results + (optional) Fetch-Text bauen
    reference_corpus: Set[str] = set()
    search_results: List[Dict[str, Any]] = []
    fetch_text = ""
    try:
        search_results = _fetch_search_results(TEST_QUERY)
    except Exception as e:
        logger.debug("search-results step failed: %s", e)
    # Festival-Keyword in Query → ALSO WGT-Bandsseite fetchen
    if "wgt" in TEST_QUERY.lower() or "festival" in TEST_QUERY.lower():
        try:
            fetch_text = _fetch_url_text(WGT_BANDS_URL)
        except Exception as e:
            logger.debug("wgt-fetch step failed: %s", e)
    try:
        reference_corpus = _collect_reference_corpus(search_results, fetch_text)
    except Exception as e:
        logger.debug("corpus-build failed: %s", e)
        reference_corpus = set()

    # 2c. Bands aus Antwort extrahieren — gegen Corpus + Whitelist matchen
    # Hinweis: answer ist bereits .lower() — Mentions kommen lowercase zurueck.
    try:
        # Original-Cased-Antwort fuer Extraction nochmal nehmen waere besser,
        # aber chat_response ist Schritt 1 — hier nutzen wir die gespeicherte
        # Roh-Variante, die wir uns gleich aus der Variable holen.
        raw_answer = (chat_response.get("response")
                      or chat_response.get("text") or "")
        extracted_bands = _extract_band_mentions(raw_answer)
    except Exception:
        extracted_bands = set()

    grounded_mentions: Set[str] = set()
    ungrounded_mentions: Set[str] = set()
    for band in extracted_bands:
        if band in WGT_KNOWN_BANDS:
            grounded_mentions.add(band)  # Whitelist = grounded
            continue
        if _is_in_corpus(band, reference_corpus):
            grounded_mentions.add(band)
        else:
            ungrounded_mentions.add(band)
    grounded_count = len(grounded_mentions)
    ungrounded_count = len(ungrounded_mentions)
    corpus_size = len(reference_corpus)

    # 3. Bewerten
    has_url = "http" in answer or "://" in answer
    has_festival = "festival" in answer or "wgt" in answer
    has_number = any(str(n) in answer for n in range(100, 300))
    has_research_marker = any(m in answer for m in RESEARCH_MARKERS)
    has_strong_source = any(s in answer for s in
                            ("monkeypress", "mdr", "wgt-festival.de"))
    # AND-Logik (legacy): Halluzination NUR wenn 2+ Spotify-Top-non-WGT-Bands UND
    # weder URL noch Research-Marker (reine LLM-Erfindung ohne Quelle).
    suspicious_count = sum(1 for b in SPOTIFY_TOP_NON_WGT if b in answer)
    legacy_hallucination = (
        suspicious_count >= 2
        and not has_url
        and not has_research_marker
    )
    # W20a.4: zusaetzlich gegen Reference-Corpus pruefen.
    # 2+ ungrounded Mentions UND keine URL UND kein Marker → Halluzination.
    # Nur scharf schalten wenn Corpus auch wirklich was liefert (>0),
    # sonst ist die Aussage "ungrounded" wertlos.
    corpus_hallucination = (
        corpus_size > 0
        and ungrounded_count >= 2
        and not has_url
        and not has_research_marker
    )
    is_hallucination = legacy_hallucination or corpus_hallucination

    duration = time.time() - started

    if is_hallucination:
        return fail_result(
            "spotify_hallucination_detected",
            detail={
                "answer_excerpt": answer[:300],
                "suspicious_count": suspicious_count,
                "ungrounded_count": ungrounded_count,
                "grounded_count": grounded_count,
                "corpus_size": corpus_size,
                "ungrounded_mentions": sorted(ungrounded_mentions)[:20],
                "trigger": (
                    "legacy_and_corpus" if (legacy_hallucination
                                            and corpus_hallucination)
                    else "legacy" if legacy_hallucination
                    else "corpus"
                ),
            },
            duration_s=duration,
        )
    if secs_since > 30:
        return fail_result(
            "search_proxy_not_called",
            detail={"seconds_since_last_call": secs_since},
            duration_s=duration,
        )

    # PASS-Logik
    score = 0
    if has_url:
        score += 1
    if has_festival:
        score += 1
    if has_number:
        score += 1
    if secs_since < 30:
        score += 1
    if has_strong_source:
        score += 1  # Boost: konkrete Quelle = echte Recherche
    # W20a.4 Bonus: Antwort ist gegen Corpus-Treffer geerdet
    if corpus_size > 0 and grounded_count >= 2:
        score += 1
    max_s = 6
    if score >= 4:
        status = "PASS"
    elif score >= 2:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "status": status,
        "score": score,
        "max": max_s,
        "duration_s": duration,
        "command_sent": TEST_QUERY,
        "baseline": baseline_stats,
        "after": after_stats,
        "delta": {
            "has_url": has_url,
            "has_festival": has_festival,
            "has_number": has_number,
            "has_research_marker": has_research_marker,
            "has_strong_source": has_strong_source,
            "suspicious_band_count": suspicious_count,
            "secs_since_last_call": secs_since,
            "extracted_bands": sorted(extracted_bands)[:30],
            "grounded_count": grounded_count,
            "ungrounded_count": ungrounded_count,
            "corpus_size": corpus_size,
        },
        "detail": {
            "answer_excerpt": answer[:200],
            "search_proxy_stats_after": after_stats,
            "ungrounded_mentions": sorted(ungrounded_mentions)[:20],
            "search_results_count": len(search_results),
            "fetch_text_len": len(fetch_text),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
