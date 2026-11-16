#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wort-basierter Gedaechtnisabruf (ohne Embeddings, ohne Netzwerk).

Findet zu einer Frage die passenden Eintraege ueber gemeinsame Woerter.
Laeuft komplett auf dem Pi, braucht weder Qdrant noch den PC, und ist
damit die Grundversorgung: funktioniert auch bei ausgeschaltetem PC und
fuer das kleine NPU-Modell. Die Vektorsuche (vector_memory.py) ist die
optionale Ergaenzung fuer Fragen ohne gemeinsame Woerter.

Verfahren:
- Kleinschreibung, Umlaute gefaltet (ae/oe/ue/ss), Fuellwoerter entfernt - auch
  Woerter UEBER das Gedaechtnis ("gespeichert", "gemerkt", "heisst"): sie stehen in
  der Frage, aber nie im Fakt, und druecken sonst passende Treffer unter die Schwelle.
- Ein Frage-Wort trifft ein Eintrags-Wort, wenn eines das andere enthaelt
  ("musik" trifft "lieblingsmusik") - deutsche Komposita und Endungen.
- Seltene Woerter zaehlen mehr als haeufige (IDF): "markus" steht in fast
  jedem Fakt und entscheidet deshalb wenig.
- Eintraege koennen Stichwoerter haben; Mehrwort-Stichwoerter ("wer bist du")
  treffen als ganze Wendung.
- Score = Gewicht der getroffenen Frage-Woerter / Gewicht aller Frage-Woerter.
"""

import math
import re
from typing import Dict, Iterable, List

MIN_TOKEN = 3          # kuerzere Woerter ignorieren
MIN_PARTIAL = 4        # Teilwort-Treffer erst ab dieser Laenge
MIN_PHRASE = 6         # Mehrwort-Stichwort erst ab dieser Laenge als Wendung werten
DEFAULT_MIN_SCORE = 0.35
PARTIAL_FACTOR = 0.6   # Teilwort-Treffer zaehlen weniger als gleiche Woerter
BODY_FACTOR = 0.7      # Treffer im Fliesstext zaehlen weniger als auf Stichwoertern
RELATIVE_CUT = 0.65    # nur Treffer nahe am besten zurueckgeben

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset("""
aber alle allem allen aller alles als also auch auf aus bei bin bis bist bitte das dass dein deine
deinem deinen deiner dem den denn der des dich die dies diese diesem diesen dieser dieses dir doch
dort du durch ein eine einem einen einer eines einmal er es etwas euch euer fuer gegen gern gerne
gibt hab habe haben hast hat hatte hier ihm ihn ihr ihre ihrem ihren ihrer im in ins ist ja jede
jedem jeden jeder jedes jetzt kann kannst kein keine keinem keinen keiner man mal mein meine meinem
meinen meiner mich mir mit moloch nach nein nicht nichts noch nun nur ob oder ohne sag sage schon
sehr sein seine seinem seinen seiner sich sie sind so soll sollen sollst ueber um und uns unser unter
viel vom von vor wann war waren warum was weil welche welchem welchen welcher welches wem wen wenn
wer werde werden wie wieder wieso will wir wird wirst wo wohl wurde zu zum zur
kennst weisst weiss erzaehl erzaehle erklaer erklaere eigentlich genau denn mal bitte
mag magst moegen moechte moechtest gefaellt geht gehts machst macht hallo danke okay alles
funktioniert funktionieren laeuft laeufst spielt spiel
ich hast hatten wurden waere wuerde koennte muss musst darf mehr immer heute gerade
heisst heisse heissen hiess hiessen nochmal vorhin damals gesagt sagtest sag
gespeichert speichern speicher gemerkt merken merk erinnerst erinnere erinnern
""".split())


def fold(text: str) -> str:
    """Kleinschreibung + Umlaute falten."""
    return str(text).lower().translate(_UMLAUTS)


def tokenize(text: str) -> List[str]:
    """Inhaltswoerter eines Textes (ohne Fuellwoerter, ohne Dubletten, Reihenfolge erhalten)."""
    seen, out = set(), []
    for tok in _WORD.findall(fold(text)):
        if len(tok) < MIN_TOKEN or tok in STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _match_strength(q: str, entry_tokens: Iterable[str]) -> float:
    """1.0 bei gleichem Wort, PARTIAL_FACTOR bei Teilwort ("musik" in "lieblingsmusik"), sonst 0."""
    best = 0.0
    for t in entry_tokens:
        if q == t:
            return 1.0
        if len(q) >= MIN_PARTIAL and len(t) >= MIN_PARTIAL and (q in t or t in q):
            best = PARTIAL_FACTOR
    return best


class LexicalIndex:
    """Kleiner Suchindex ueber kurze Texte. Fuer einige hundert Eintraege gedacht.

    Ein Eintrag ist (text, quelle) oder ein dict {"text", "source", "keywords": [...]}.
    Mit Stichwoertern gilt: Treffer auf ein Stichwort zaehlen voll, Treffer nur im
    Fliesstext zaehlen BODY_FACTOR. Mehrwort-Stichwoerter ("wer bist du") treffen
    als ganze Wendung in der Frage - so finden auch Fragen aus lauter Fuellwoertern.
    """

    def __init__(self, entries: Iterable):
        self._entries: List[Dict] = []
        for e in entries:
            if not isinstance(e, dict):
                e = {"text": e[0], "source": e[1]}
            text = " ".join(str(e.get("text", "")).split())
            keywords = [fold(k).strip() for k in e.get("keywords", []) if str(k).strip()]
            key_tokens = tokenize(" ".join(keywords))
            body_tokens = tokenize(text)
            if not text or not (key_tokens or body_tokens):
                continue
            self._entries.append({
                "text": text, "source": e.get("source", ""),
                "key": key_tokens, "body": body_tokens,
                "phrases": [k for k in keywords if " " in k and len(k) >= MIN_PHRASE],
            })

    def __len__(self) -> int:
        return len(self._entries)

    def _strength(self, q: str, entry: Dict) -> float:
        key = _match_strength(q, entry["key"])
        body = _match_strength(q, entry["body"])
        if entry["key"]:
            body *= BODY_FACTOR
        return max(key, body)

    def search(self, query: str, limit: int = 3,
               min_score: float = DEFAULT_MIN_SCORE) -> List[Dict]:
        """Beste Eintraege zur Frage. Returns [{text, source, score}], absteigend.

        Nur Eintraege nahe am besten Treffer (RELATIVE_CUT) kommen zurueck -
        lieber eine passende Erinnerung als drei halb passende.
        """
        if not self._entries:
            return []
        folded = " ".join(_WORD.findall(fold(query)))
        scores: Dict[int, float] = {}

        # Ganze Wendungen ("wer bist du") - unabhaengig von Fuellwoertern
        for i, e in enumerate(self._entries):
            if any(f" {p} " in f" {folded} " for p in e["phrases"]):
                scores[i] = 1.0

        q_tokens = tokenize(query)
        if q_tokens:
            n = len(self._entries)
            total = 0.0
            acc: Dict[int, float] = {}
            for q in q_tokens:
                strengths = [(i, self._strength(q, e)) for i, e in enumerate(self._entries)]
                hits = [(i, st) for i, st in strengths if st > 0]
                # Seltene Woerter zaehlen mehr (IDF), aber gedaempft auf 1..2: ein
                # Frage-Wort ganz ohne Treffer ("trinke") darf einen klaren
                # Treffer ("kaffee") nicht erdruecken.
                w = 1.0 + math.log(1.0 + n / (1.0 + len(hits))) / math.log(1.0 + n)
                total += w
                for i, st in hits:
                    acc[i] = acc.get(i, 0.0) + w * st
            for i, sc in acc.items():
                scores[i] = max(scores.get(i, 0.0), sc / total)

        if not scores:
            return []
        ranked = sorted(((sc, -i) for i, sc in scores.items()), reverse=True)
        best = ranked[0][0]
        out = []
        for score, neg_i in ranked:
            if score < min_score or score < best * RELATIVE_CUT or len(out) >= limit:
                break
            e = self._entries[-neg_i]
            out.append({"text": e["text"], "source": e["source"], "score": round(score, 3)})
        return out
