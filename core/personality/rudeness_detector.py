#!/usr/bin/env python3
"""
M.O.L.O.C.H. Rudeness/Appeasement Detector — Phase 2 Task 2a+2b
================================================================

Klassifiziert Text als rude/appeasement und gibt einen Score 0.0-1.0 zurueck.

V0:
- Keyword-Fastpath (Liste aus tension_integrator.py uebernommen)
- Match → Score 0.7 (skaliert nach Hit-Anzahl)
- Kein Match → Score 0.0
- TF-IDF Pickle-Fallback ist Stub (Platzhalter fuer spaeteres Training)

V1 (geplant):
- TF-IDF Modell aus character_journal Trainingsdaten
- Sentiment-Score per kleinem Modell

Crash-safe: Jede Exception → Score 0.0.
"""

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger("MolochRudenessDetector")

# Keywords aus tension_integrator.py uebernommen — Single Source bleibt dort,
# hier eigene Kopie als Fastpath fuer Detector-Standalone-Nutzung.
_RUDENESS_KEYWORDS: List[str] = [
    "blöd", "dumm", "scheiß", "idiot", "nutzlos", "kaputt", "schrott", "müll",
    "bescheuert", "depp", "doof", "schwachsinn", "mist", "dreck", "arschloch",
    "wichser", "hurensohn", "vollidiot", "trottel", "spacken",
    "stupid", "useless", "trash", "garbage", "broken", "crap", "fuck", "shit",
    "asshole", "idiot", "moron", "dumbass",
]

_APPEASEMENT_KEYWORDS: List[str] = [
    "sorry", "entschuldigung", "tut mir leid", "bitte", "danke", "schön",
    "toll", "super", "gut gemacht", "prima", "klasse", "wunderbar",
    "ich mag dich", "du bist gut", "respekt", "brav", "okay okay",
    "alles gut", "peace", "calm down", "beruhig dich",
]

# Modell-Pfad fuer spaetere TF-IDF-Erweiterung
_TFIDF_MODEL_PATH = os.path.expanduser("~/moloch/config/rudeness_tfidf.pkl")


class _BaseDetector:
    """Gemeinsamer Code fuer Rudeness und Appeasement."""

    _keywords: List[str] = []
    _name: str = "base"

    def __init__(self):
        self._lock = threading.Lock()
        self._tfidf_model = None  # Lazy-loaded falls vorhanden
        self._tfidf_tried = False

    def detect(self, text: str) -> float:
        """Gibt Score 0.0-1.0 zurueck. Crash-safe."""
        try:
            if not text or not isinstance(text, str):
                return 0.0
            text_lower = text.lower().strip()
            if len(text_lower) < 2:
                return 0.0

            # Keyword-Fastpath
            hits = sum(1 for kw in self._keywords if kw in text_lower)
            if hits >= 1:
                # Basis 0.7 bei 1 Hit, +0.1 pro weiterem Hit, capped bei 1.0
                score = min(1.0, 0.7 + (hits - 1) * 0.1)
                logger.debug(f"[{self._name}] Keyword-Hit: {hits} → score={score:.2f}")
                return score

            # TF-IDF Fallback (Stub)
            tfidf_score = self._try_tfidf_model(text_lower)
            if tfidf_score is not None:
                return tfidf_score

            return 0.0
        except Exception as e:
            logger.warning(f"[{self._name}] detect() Exception: {e}")
            return 0.0

    def _try_tfidf_model(self, text: str) -> Optional[float]:
        """Pickle-Fallback-Stub. Gibt None zurueck wenn kein Modell geladen.

        Spaeter: TF-IDF + LogReg auf character_journal Daten trainieren,
        Modell nach _TFIDF_MODEL_PATH speichern, hier laden.
        """
        if self._tfidf_tried and self._tfidf_model is None:
            return None
        with self._lock:
            if self._tfidf_tried:
                return None
            self._tfidf_tried = True
            if not os.path.exists(_TFIDF_MODEL_PATH):
                return None
            try:
                import pickle
                with open(_TFIDF_MODEL_PATH, "rb") as f:
                    self._tfidf_model = pickle.load(f)
                logger.info(f"[{self._name}] TF-IDF Modell geladen aus {_TFIDF_MODEL_PATH}")
            except Exception as e:
                logger.debug(f"[{self._name}] TF-IDF Lade-Fehler: {e}")
                self._tfidf_model = None
        return None  # V0: kein echtes Inferenz-Path


class RudenessDetector(_BaseDetector):
    """Erkennt verbale Aggression / Beleidigung."""
    _keywords = _RUDENESS_KEYWORDS
    _name = "rudeness"


class AppeasementDetector(_BaseDetector):
    """Erkennt Besaenftigung / nette Worte."""
    _keywords = _APPEASEMENT_KEYWORDS
    _name = "appeasement"
