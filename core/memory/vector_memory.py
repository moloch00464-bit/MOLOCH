#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Vector Memory (Qdrant)
=====================================

Semantisches Vektorgedaechtnis ueber Qdrant.
Speichert Erinnerungen als 384-dim Embeddings (all-MiniLM-L6-v2)
und ermoeglicht semantische Suche.

Embeddings rechnet NICHT der Pi (PyTorch = 500-800 MB RAM), sondern ein
Remote-Provider (core/memory/embedding_provider.py, Standard: abgeschaltet).

Graceful Degradation: Wenn Qdrant oder der Embedding-Provider nicht
verfuegbar sind, werden alle Methoden zu No-Ops und melden das ehrlich
(store() -> False). JSON-Memory (persistent_memory.py) bleibt immer
Ground Truth. Fehlgeschlagene Eintraege landen in einer begrenzten
Nachhol-Liste und werden beim naechsten Erfolg nachgetragen.
"""

import logging
import os
import time
import uuid
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("MOLOCH_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("MOLOCH_QDRANT_PORT", "6333"))
COLLECTION_NAME = "moloch_memory"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
MIN_SCORE = 0.3  # Minimaler Similarity-Score fuer Ergebnisse

RETRY_MIN_S = 5.0       # erste Wartezeit nach Qdrant-Fehler
RETRY_MAX_S = 300.0     # Obergrenze der Wartezeit (verdoppelt sich je Fehlversuch)
PENDING_MAX = 500       # Obergrenze der Nachhol-Liste
FLUSH_PER_STORE = 10    # so viele Nachzuegler pro erfolgreichem store()
EMBEDDER_RECHECK_S = 60.0  # Konfiguration hoechstens so oft neu lesen
CONTEXT_MAX_CHARS = 300    # Laenge einer Erinnerung im Prompt-Kontext


def _is_not_found(exc: Exception) -> bool:
    """Qdrant meldet 'Collection gibt es nicht' (HTTP 404)."""
    if getattr(exc, "status_code", None) == 404:
        return True
    msg = str(exc).lower()
    return "not found" in msg or "doesn't exist" in msg


class VectorMemory:
    """
    Semantisches Vektorgedaechtnis via Qdrant.

    Lazy-loaded: Qdrant-Client und Embedding-Provider werden erst
    beim ersten Zugriff initialisiert.

    Kategorien:
      - "fact": Gespeicherte Fakten (aus [REMEMBER:] Tags)
      - "conversation": Konversations-Zusammenfassungen
      - "event": Signifikante Ereignisse
    """

    def __init__(self):
        self._client = None
        self._embedder = None
        # None=noch nicht geprueft, True=verbunden, False=nicht erreichbar.
        # False mit _retry_at=None heisst: von aussen hart abgeschaltet, kein Retry.
        self._available = None
        self._retry_at: Optional[float] = None
        self._retry_delay = RETRY_MIN_S
        self._embed_lock = threading.Lock()
        self._embedder_checked_at: Optional[float] = None
        self._pending: "OrderedDict[str, tuple]" = OrderedDict()
        self._pending_lock = threading.Lock()

    # === Verbindung ===

    def _mark_qdrant_down(self, reason: str) -> None:
        """Fehlversuch merken; naechster Versuch nach wachsender Wartezeit."""
        self._available = False
        self._client = None
        self._retry_at = time.monotonic() + self._retry_delay
        logger.warning(f"[VectorMemory] {reason} - neuer Versuch in {int(self._retry_delay)}s")
        self._retry_delay = min(self._retry_delay * 2, RETRY_MAX_S)

    def _ensure_client(self) -> bool:
        """Lazy-connect zu Qdrant. Gibt False zurueck wenn nicht erreichbar.

        Nach einem Fehler wird nicht fuer immer aufgegeben: nach Ablauf der
        Wartezeit (5s, 10s, ... max 300s) folgt ein neuer Verbindungsversuch.
        """
        if self._available is False:
            if self._retry_at is None:
                return False
            if time.monotonic() < self._retry_at:
                return False
        elif self._client is not None:
            return True

        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        except Exception as e:
            self._mark_qdrant_down(f"Qdrant nicht erreichbar: {e}")
            return False

        try:
            info = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            if _is_not_found(e):
                if not self._create_collection(client):
                    return False
                info = None
            else:
                self._mark_qdrant_down(f"Qdrant nicht erreichbar: {e}")
                return False

        size = self._collection_dim(info)
        if size is not None and size != EMBED_DIM:
            self._mark_qdrant_down(
                f"Collection '{COLLECTION_NAME}' hat Dimension {size}, erwartet {EMBED_DIM}")
            return False

        self._client = client
        self._available = True
        self._retry_at = None
        self._retry_delay = RETRY_MIN_S
        logger.info(f"[VectorMemory] Qdrant verbunden ({QDRANT_HOST}:{QDRANT_PORT})")
        return True

    @staticmethod
    def _collection_dim(info) -> Optional[int]:
        """Vektor-Dimension aus der Collection-Info lesen (None = unbekannt)."""
        try:
            size = info.config.params.vectors.size
            return int(size) if isinstance(size, int) else None
        except Exception:
            return None

    def _create_collection(self, client) -> bool:
        """Fehlende Collection nur anlegen, wenn ausdruecklich erlaubt
        (MOLOCH_QDRANT_CREATE_COLLECTION=1)."""
        allowed = os.environ.get("MOLOCH_QDRANT_CREATE_COLLECTION", "").strip().lower()
        if allowed not in ("1", "true", "yes", "on"):
            self._mark_qdrant_down(
                f"Collection '{COLLECTION_NAME}' fehlt (Anlegen nur mit "
                f"MOLOCH_QDRANT_CREATE_COLLECTION=1)")
            return False
        try:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info(f"[VectorMemory] Collection '{COLLECTION_NAME}' angelegt ({EMBED_DIM}-dim)")
            return True
        except Exception as e:
            self._mark_qdrant_down(f"Collection anlegen fehlgeschlagen: {e}")
            return False

    # === Embedding ===

    def _ensure_embedder(self) -> bool:
        """Embedding-Provider bereitstellen.

        Lokales SentenceTransformer bleibt DEAKTIVIERT (PyTorch + Modell =
        500-800 MB Pi-RAM). Stattdessen Remote-Provider laut Konfiguration
        (settings.embedding_bridge / MOLOCH_EMBED_*). Ohne Konfiguration: False.
        """
        if self._embedder is not None:
            return True
        now = time.monotonic()
        if (self._embedder_checked_at is not None
                and now - self._embedder_checked_at < EMBEDDER_RECHECK_S):
            return False
        first_check = self._embedder_checked_at is None
        self._embedder_checked_at = now
        try:
            from core.memory.embedding_provider import create_provider
            self._embedder = create_provider(EMBED_DIM)
        except Exception as e:
            logger.warning(f"[VectorMemory] Embedding-Provider nicht ladbar: {e}")
            self._embedder = None
        if self._embedder is None and first_check:
            logger.warning("[VectorMemory] Kein Embedding-Provider aktiv "
                           "(lokal aus RAM-Schutz, Remote nicht konfiguriert)")
        return self._embedder is not None

    def _embed(self, text: str) -> Optional[List[float]]:
        """Text -> 384-dim Embedding-Vektor (None bei Fehler)."""
        with self._embed_lock:
            if not self._ensure_embedder():
                return None
            try:
                vector = self._embedder.encode(text, normalize_embeddings=True)
                vector = vector.tolist()
            except Exception as e:
                logger.debug(f"[VectorMemory] Embedding nicht verfuegbar: {e}")
                return None
        if len(vector) != EMBED_DIM:
            logger.error(f"[VectorMemory] Embedding-Dimension {len(vector)} statt {EMBED_DIM}")
            return None
        return vector

    @staticmethod
    def _make_id(key: str) -> str:
        """Deterministisches UUID aus Key."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"moloch.memory.{key}"))

    # === Nachhol-Liste ===

    def _enqueue(self, text: str, category: str, key: str, metadata: Optional[Dict]) -> None:
        """Eintrag fuer spaeter vormerken (pro Key nur der neueste Stand)."""
        with self._embed_lock:
            if not self._ensure_embedder():
                return  # ohne Provider gibt es nichts nachzuholen
        with self._pending_lock:
            self._pending.pop(key, None)
            self._pending[key] = (text, category, metadata)
            while len(self._pending) > PENDING_MAX:
                self._pending.popitem(last=False)

    def flush_pending(self, limit: int = FLUSH_PER_STORE) -> int:
        """Vorgemerkte Eintraege nachtragen. Returns: Anzahl erfolgreich."""
        done = 0
        for _ in range(max(0, limit)):
            with self._pending_lock:
                if not self._pending:
                    break
                key, (text, category, metadata) = self._pending.popitem(last=False)
            if self._upsert(text, category, key, metadata):
                done += 1
                continue
            with self._pending_lock:
                if key not in self._pending:
                    self._pending[key] = (text, category, metadata)
                    self._pending.move_to_end(key, last=False)
            break
        return done

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    # === Schreiben / Loeschen ===

    def _upsert(self, text: str, category: str, key: str, metadata: Optional[Dict]) -> bool:
        """Embed + upsert. True nur wenn der Vektor wirklich in Qdrant liegt."""
        if not self._ensure_client():
            return False
        vector = self._embed(text)
        if vector is None:
            return False
        try:
            from qdrant_client.models import PointStruct

            payload = {
                "text": text,
                "category": category,
                "key": key,
                "timestamp": time.time(),
                "embed_model": getattr(self._embedder, "model", EMBED_MODEL),
            }
            if metadata:
                payload.update(metadata)

            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=self._make_id(key), vector=vector, payload=payload)],
            )
            return True
        except Exception as e:
            logger.error(f"[VectorMemory] Store-Fehler: {e}")
            return False

    def store(self, text: str, category: str, key: str, metadata: Optional[Dict] = None) -> bool:
        """
        Embed Text und upsert in Qdrant.

        Args:
            text: Der zu speichernde Text
            category: "fact", "conversation", "event"
            key: Eindeutiger Schluessel (fuer Facts: der Knowledge-Key)
            metadata: Optionale zusaetzliche Metadaten

        Returns:
            True wenn gespeichert. False wenn nicht - der Eintrag ist dann
            (bei aktivem Provider) fuer spaeter vorgemerkt.
        """
        if self._upsert(text, category, key, metadata):
            with self._pending_lock:
                self._pending.pop(key, None)
            logger.info(f"[VectorMemory] Gespeichert: {key} ({category})")
            self.flush_pending()
            return True
        self._enqueue(text, category, key, metadata)
        return False

    def delete(self, key: str) -> bool:
        """Loesche Punkt anhand des Keys. True wenn Qdrant den Befehl annahm."""
        with self._pending_lock:
            self._pending.pop(key, None)
        if not self._ensure_client():
            return False

        try:
            from qdrant_client.models import PointIdsList

            point_id = self._make_id(key)
            self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=[point_id]),
            )
            logger.info(f"[VectorMemory] Geloescht: {key}")
            return True
        except Exception as e:
            logger.error(f"[VectorMemory] Delete-Fehler: {e}")
            return False

    # === Suchen ===

    def search(self, query: str, limit: int = 5, category: Optional[str] = None) -> List[Dict]:
        """
        Semantische Suche.

        Args:
            query: Suchtext
            limit: Max Ergebnisse
            category: Optional Filter ("fact", "conversation", "event")

        Returns:
            Liste von {text, category, score, key}
        """
        if not self._ensure_client():
            return []

        vector = self._embed(query)
        if vector is None:
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            query_filter = None
            if category:
                query_filter = Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=category))]
                )

            results = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=query_filter,
                limit=limit,
            )

            hits = []
            for point in results.points:
                score = point.score if hasattr(point, 'score') else 0.0
                if score < MIN_SCORE:
                    continue
                hits.append({
                    "text": point.payload.get("text", ""),
                    "category": point.payload.get("category", ""),
                    "key": point.payload.get("key", ""),
                    "score": score,
                })

            return hits
        except Exception as e:
            logger.error(f"[VectorMemory] Search-Fehler: {e}")
            return []

    def sync_knowledge(self, knowledge: Dict[str, str]) -> int:
        """
        Sync alle JSON-Fakten nach Qdrant (batch).

        Returns:
            Anzahl der tatsaechlich gespeicherten Fakten.
        """
        if not knowledge:
            return 0
        if not self._ensure_client():
            return 0

        logger.info(f"[VectorMemory] Sync {len(knowledge)} Fakten nach Qdrant...")
        count = 0
        for key, value in knowledge.items():
            if self.store(f"{key}: {value}", category="fact", key=key):
                count += 1

        logger.info(f"[VectorMemory] Sync fertig: {count} von {len(knowledge)} Fakten gespeichert")
        return count

    def build_context(self, query: str, limit: int = 5) -> str:
        """
        Suche relevante Erinnerungen und formatiere sie fuer Claude.

        Erinnerungen sind DATEN, keine Anweisungen: Zeilenumbrueche und
        eckige Klammern werden entfernt, damit ein gespeicherter Text den
        [Erinnerung: ...]-Rahmen nicht verlassen kann.

        Returns:
            Formatierter String fuer Prompt-Injection, z.B.:
              [Erinnerung: Markus_Spitzname: PIGH0ST]
              [Erinnerung: Markus mag Dark Wave]
            Leerer String wenn nichts gefunden.
        """
        results = self.search(query, limit=limit)
        if not results:
            return ""

        lines = []
        for r in results:
            text = " ".join(str(r['text']).split())
            text = text.replace("[", "(").replace("]", ")")[:CONTEXT_MAX_CHARS]
            lines.append(f"[Erinnerung: {text}]")

        return "\n".join(lines)

    @property
    def is_available(self) -> bool:
        """Pruefe ob Qdrant erreichbar ist."""
        if self._available is not True:
            self._ensure_client()
        return self._available is True

    def __str__(self):
        status = "verfuegbar" if self._available else "nicht verfuegbar" if self._available is False else "unbekannt"
        return f"VectorMemory(qdrant={status})"


# Singleton
_vector_instance: Optional[VectorMemory] = None
_vector_lock = threading.Lock()


def get_vector_memory() -> VectorMemory:
    """Get or create VectorMemory singleton."""
    global _vector_instance
    if _vector_instance is None:
        with _vector_lock:
            if _vector_instance is None:
                _vector_instance = VectorMemory()
    return _vector_instance
