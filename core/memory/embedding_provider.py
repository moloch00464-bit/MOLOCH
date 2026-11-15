#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Embedding-Provider (Text -> Vektor)
================================================

Der Pi hat zu wenig RAM fuer sentence-transformers (PyTorch, 500-800 MB).
Deshalb rechnet der PC die Embeddings: Ollama laeuft dort ohnehin
(LLM-Tentakel) und liefert ueber /api/embed Vektoren. Das Modell
`all-minilm` hat 384 Dimensionen, passend zur Collection moloch_memory.

Standard: ABGESCHALTET. Ohne Konfiguration verhaelt sich VectorMemory wie
bisher (keine Embeddings, JSON bleibt Ground Truth).

Konfiguration - settings.json, Schluessel `embedding_bridge`:
    {"enabled": true, "host": "...", "port": 11434, "model": "all-minilm",
     "path": "/api/embed", "connect_timeout_sec": 2, "timeout_sec": 8,
     "backoff_sec": 60, "max_chars": 2000}
Fehlen host/port, gelten die Werte aus `tentacle_llm`.

Umgebungsvariablen (haben Vorrang):
    MOLOCH_EMBED_ENABLED=1|0
    MOLOCH_EMBED_URL=http://rechner:11434     (Basis-URL ohne Pfad)
    MOLOCH_EMBED_MODEL=all-minilm
    MOLOCH_EMBED_TOKEN=...                    (optional, Authorization: Bearer)
    MOLOCH_MEMORY_RECALL=1|0                  (Fakten-Recall im LLM-Kontext)
"""

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

_SETTINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "settings.json",
)

_DEFAULTS = {
    "enabled": False,
    "host": "",
    "port": 11434,
    "model": "all-minilm",
    "path": "/api/embed",
    "connect_timeout_sec": 2,
    "timeout_sec": 8,
    "backoff_sec": 60,
    "max_chars": 2000,
    # Recall: passende Fakten in den LLM-Kontext (core/longterm_memory.py recall()).
    # Eigener Schalter, damit Speichern und Abrufen getrennt aktivierbar sind.
    "recall_enabled": False,
    "recall_limit": 3,
    "recall_min_score": 0.45,
}


class EmbeddingProvider(Protocol):
    """Schnittstelle: Text rein, Vektor (oder None bei Fehler) raus."""

    model: str

    def embed(self, text: str) -> Optional[List[float]]:
        ...


class _Vec(list):
    """Liste mit tolist() - damit Aufrufer, die ein numpy-Array erwarten
    (core/moloch_sprache.py nutzt VectorMemory._embedder.encode().tolist()),
    unveraendert weiterlaufen."""

    def tolist(self) -> List[float]:
        return list(self)


def load_config() -> Dict:
    """Konfiguration aus settings.json + Umgebungsvariablen."""
    cfg = dict(_DEFAULTS)
    try:
        with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
        cfg.update(settings.get("embedding_bridge", {}) or {})
        tentacle = settings.get("tentacle_llm", {}) or {}
        if not cfg.get("host"):
            cfg["host"] = tentacle.get("host", "")
            cfg["port"] = tentacle.get("port", cfg["port"])
    except Exception as e:
        logger.debug(f"[Embedding] settings.json nicht lesbar: {e}")

    env_enabled = os.environ.get("MOLOCH_EMBED_ENABLED")
    if env_enabled is not None:
        cfg["enabled"] = env_enabled.strip().lower() in ("1", "true", "yes", "on")
    env_recall = os.environ.get("MOLOCH_MEMORY_RECALL")
    if env_recall is not None:
        cfg["recall_enabled"] = env_recall.strip().lower() in ("1", "true", "yes", "on")
    if os.environ.get("MOLOCH_EMBED_MODEL"):
        cfg["model"] = os.environ["MOLOCH_EMBED_MODEL"]
    base = os.environ.get("MOLOCH_EMBED_URL", "").rstrip("/")
    if not base and cfg.get("host"):
        base = f"http://{cfg['host']}:{cfg['port']}"
    cfg["base_url"] = base
    return cfg


class RemoteEmbeddingProvider:
    """Holt Embeddings per HTTP vom PC (Ollama /api/embed).

    - Kurze Timeouts + Circuit-Breaker: ein ausgeschalteter PC kostet einen
      einzigen Timeout, danach `backoff_sec` lang gar nichts.
    - Kein automatischer Wiederholungsversuch im selben Aufruf.
    - Vektor wird auf die erwartete Dimension geprueft und L2-normalisiert.
    """

    def __init__(self, cfg: Dict, expected_dim: int):
        self.model = str(cfg["model"])
        self._url = cfg["base_url"] + str(cfg["path"])
        self._timeout = (float(cfg["connect_timeout_sec"]), float(cfg["timeout_sec"]))
        self._backoff = float(cfg["backoff_sec"])
        self._max_chars = int(cfg["max_chars"])
        self._dim = int(expected_dim)
        self._down_until = 0.0
        self._headers = {}
        token = os.environ.get("MOLOCH_EMBED_TOKEN")
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    @property
    def is_down(self) -> bool:
        return time.monotonic() < self._down_until

    def _mark_down(self, reason: str) -> None:
        self._down_until = time.monotonic() + self._backoff
        logger.warning(f"[Embedding] {reason} - Pause {int(self._backoff)}s")

    def embed(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None
        if self.is_down:
            return None
        import requests
        try:
            r = requests.post(
                self._url,
                json={"model": self.model, "input": text[: self._max_chars]},
                headers=self._headers,
                timeout=self._timeout,
            )
            if r.status_code != 200:
                self._mark_down(f"HTTP {r.status_code}")
                return None
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            self._mark_down(f"nicht erreichbar: {str(e)[:120]}")
            return None

        # Ollama /api/embed: {"embeddings": [[...]]}, aelteres /api/embeddings: {"embedding": [...]}
        vec = None
        if isinstance(data, dict):
            embs = data.get("embeddings")
            if isinstance(embs, list) and embs and isinstance(embs[0], list):
                vec = embs[0]
            elif isinstance(data.get("embedding"), list):
                vec = data["embedding"]
        if not vec:
            self._mark_down("Antwort ohne Embedding")
            return None
        if len(vec) != self._dim:
            # Falsches Modell konfiguriert - nie in die Collection schreiben.
            self._mark_down(f"Dimension {len(vec)} statt {self._dim} (Modell '{self.model}')")
            return None
        try:
            vec = [float(x) for x in vec]
        except (TypeError, ValueError):
            self._mark_down("Embedding enthaelt Nicht-Zahlen")
            return None
        norm = math.sqrt(sum(x * x for x in vec))
        if norm <= 0.0 or not math.isfinite(norm):
            return None
        return [x / norm for x in vec]

    def encode(self, text: str, normalize_embeddings: bool = True) -> _Vec:
        """Kompatibel zu SentenceTransformer.encode() fuer bestehende Aufrufer.
        Wirft bei Fehler, weil die Aufrufer Ausnahmen bereits abfangen."""
        vec = self.embed(text)
        if vec is None:
            raise RuntimeError("Embedding nicht verfuegbar")
        return _Vec(vec)


def create_provider(expected_dim: int) -> Optional[RemoteEmbeddingProvider]:
    """Provider laut Konfiguration bauen. None wenn abgeschaltet oder ohne Ziel."""
    cfg = load_config()
    if not cfg.get("enabled"):
        return None
    if not cfg.get("base_url"):
        logger.warning("[Embedding] aktiviert, aber kein Host/URL konfiguriert")
        return None
    logger.info(f"[Embedding] Remote-Provider: {cfg['base_url']}{cfg['path']} Modell {cfg['model']}")
    return RemoteEmbeddingProvider(cfg, expected_dim)
