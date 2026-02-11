#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Vector Memory (Qdrant)
=====================================

Semantisches Vektorgedaechtnis ueber Qdrant.
Speichert Erinnerungen als 384-dim Embeddings (all-MiniLM-L6-v2)
und ermoeglicht semantische Suche.

Graceful Degradation: Wenn Qdrant oder sentence-transformers
nicht verfuegbar sind, werden alle Methoden zu No-Ops.
JSON-Memory (persistent_memory.py) bleibt immer Ground Truth.
"""

import logging
import time
import uuid
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "moloch_memory"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
MIN_SCORE = 0.3  # Minimaler Similarity-Score fuer Ergebnisse


class VectorMemory:
    """
    Semantisches Vektorgedaechtnis via Qdrant.

    Lazy-loaded: Qdrant-Client und Embedding-Modell werden erst
    beim ersten Zugriff initialisiert.

    Kategorien:
      - "fact": Gespeicherte Fakten (aus [REMEMBER:] Tags)
      - "conversation": Konversations-Zusammenfassungen
      - "event": Signifikante Ereignisse
    """

    def __init__(self):
        self._client = None
        self._embedder = None
        self._available = None  # None=noch nicht geprueft
        self._embed_lock = threading.Lock()

    def _ensure_client(self) -> bool:
        """Lazy-connect zu Qdrant. Gibt False zurueck wenn nicht erreichbar."""
        if self._available is False:
            return False
        if self._client is not None:
            return True

        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            # Verbindungstest
            self._client.get_collection(COLLECTION_NAME)
            self._available = True
            logger.info(f"[VectorMemory] Qdrant verbunden ({QDRANT_HOST}:{QDRANT_PORT})")
            return True
        except Exception as e:
            self._available = False
            self._client = None
            logger.warning(f"[VectorMemory] Qdrant nicht erreichbar: {e}")
            return False

    def _ensure_embedder(self) -> bool:
        """Lazy-load sentence-transformers Modell (~2s auf Pi5)."""
        if self._embedder is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[VectorMemory] Lade Embedding-Modell {EMBED_MODEL}...")
            self._embedder = SentenceTransformer(EMBED_MODEL)
            logger.info(f"[VectorMemory] Embedding-Modell geladen")
            return True
        except Exception as e:
            logger.warning(f"[VectorMemory] sentence-transformers nicht verfuegbar: {e}")
            return False

    def _embed(self, text: str) -> Optional[List[float]]:
        """Text -> 384-dim Embedding-Vektor."""
        with self._embed_lock:
            if not self._ensure_embedder():
                return None
            try:
                vector = self._embedder.encode(text, normalize_embeddings=True)
                return vector.tolist()
            except Exception as e:
                logger.error(f"[VectorMemory] Embedding-Fehler: {e}")
                return None

    @staticmethod
    def _make_id(key: str) -> str:
        """Deterministisches UUID aus Key."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"moloch.memory.{key}"))

    def store(self, text: str, category: str, key: str, metadata: Optional[Dict] = None):
        """
        Embed Text und upsert in Qdrant.

        Args:
            text: Der zu speichernde Text
            category: "fact", "conversation", "event"
            key: Eindeutiger Schluessel (fuer Facts: der Knowledge-Key)
            metadata: Optionale zusaetzliche Metadaten
        """
        if not self._ensure_client():
            return

        vector = self._embed(text)
        if vector is None:
            return

        try:
            from qdrant_client.models import PointStruct

            point_id = self._make_id(key)
            payload = {
                "text": text,
                "category": category,
                "key": key,
                "timestamp": time.time(),
            }
            if metadata:
                payload.update(metadata)

            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            logger.info(f"[VectorMemory] Gespeichert: {key} ({category})")
        except Exception as e:
            logger.error(f"[VectorMemory] Store-Fehler: {e}")

    def delete(self, key: str):
        """Loesche Punkt anhand des Keys."""
        if not self._ensure_client():
            return

        try:
            from qdrant_client.models import PointIdsList

            point_id = self._make_id(key)
            self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=[point_id]),
            )
            logger.info(f"[VectorMemory] Geloescht: {key}")
        except Exception as e:
            logger.error(f"[VectorMemory] Delete-Fehler: {e}")

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

    def sync_knowledge(self, knowledge: Dict[str, str]):
        """
        Sync alle JSON-Fakten nach Qdrant (batch).

        Wird einmal beim Start aufgerufen um sicherzustellen
        dass Qdrant alle Fakten hat.
        """
        if not knowledge:
            return
        if not self._ensure_client():
            return

        logger.info(f"[VectorMemory] Sync {len(knowledge)} Fakten nach Qdrant...")
        count = 0
        for key, value in knowledge.items():
            text = f"{key}: {value}"
            self.store(text, category="fact", key=key)
            count += 1

        logger.info(f"[VectorMemory] Sync fertig: {count} Fakten")

    def build_context(self, query: str, limit: int = 5) -> str:
        """
        Suche relevante Erinnerungen und formatiere sie fuer Claude.

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
            lines.append(f"[Erinnerung: {r['text']}]")

        return "\n".join(lines)

    @property
    def is_available(self) -> bool:
        """Pruefe ob Qdrant erreichbar ist."""
        if self._available is None:
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
