#!/usr/bin/env python3
"""
M.O.L.O.C.H. Episodisches Gedaechtnis — Qdrant Vektor-DB
==========================================================

Speichert Wahrnehmungs-Episoden als 512-dim ArcFace-Embeddings in Qdrant.
Embedded Modus (kein Server noetig) — Daten auf SSD2.

Funktionen:
  - store_episode(): Ereignis mit Embedding + Metadaten speichern
  - recall(): Aehnlichste Episoden per Cosine Similarity abrufen
  - get_stats(): Collection-Statistiken

Datenbank: /mnt/moloch-data/qdrant/ (persistent auf SSD2)

Singleton: get_episodic_memory()

Gate 2: Identity (ReID + Qdrant VITALE)
"""

import logging
import threading
import time
import uuid
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger("MolochEpisodicMemory")

QDRANT_PATH = "/home/molochzuhause/moloch/data/qdrant"  # ext4 SSD1 (NTFS hatte SQLite-Journal-Bug)
COLLECTION_NAME = "episodes"
EMBEDDING_DIM = 512


class EpisodicMemory:
    """
    Episodisches Gedaechtnis via Qdrant (embedded, kein Server).

    Speichert Wahrnehmungs-Episoden als Vektoren mit Metadaten.
    Abruf via Cosine Similarity — "Wer war das letzte Mal hier?"
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._client = None
        self._ready = False
        self._init_qdrant()

    def _init_qdrant(self):
        """Qdrant Client im embedded Modus starten + Collection anlegen.

        DEAKTIVIERT: Qdrant embedded (in-process) allokiert via Rust/RocksDB/MMAP
        hunderte MB bis GB im Pi-RAM. Pi5 hat nur 4 GB → System wird unbenutzbar.
        TODO: Auf Qdrant-Client (Docker, port 6333) umstellen wenn Docker-Qdrant
        repariert ist (aktuell Crash-Loop wegen NTFS Stale file handle).
        """
        logger.warning("[EPISODIC] Qdrant embedded DEAKTIVIERT (RAM-Schutz: Pi5 hat nur 4 GB)")
        return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                VectorParams, Distance, PointStruct,
            )
            self._client = QdrantClient(path=QDRANT_PATH)

            # Collection erstellen falls nicht vorhanden
            collections = [c.name for c in self._client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"[EPISODIC] Collection '{COLLECTION_NAME}' erstellt ({EMBEDDING_DIM}-dim, Cosine)")
            else:
                info = self._client.get_collection(COLLECTION_NAME)
                logger.info(f"[EPISODIC] Collection '{COLLECTION_NAME}' geladen: {info.points_count} Punkte")

            self._ready = True
        except Exception as e:
            logger.error(f"[EPISODIC] Qdrant Init fehlgeschlagen: {e}")
            self._client = None
            self._ready = False

    # =====================================================================
    # Speichern
    # =====================================================================

    def store_episode(self, person: str, event_type: str,
                      embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Episode in Qdrant speichern.

        Args:
            person: Identitaets-Name (z.B. "markus", "unknown")
            event_type: Ereignis-Typ (z.B. "reid_match", "face_detected", "entered")
            embedding: 512-dim ArcFace-Embedding
            metadata: Optionale Zusatzinfos (distance, confidence, zone, ...)
        """
        if not self._ready or self._client is None:
            return

        from qdrant_client.models import PointStruct

        # Embedding normalisieren
        emb = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        payload = {
            "person": person.lower(),
            "event_type": event_type,
            "timestamp": time.time(),
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            payload.update(metadata)

        point_id = str(uuid.uuid4())

        with self._lock:
            try:
                self._client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=emb.tolist(),
                            payload=payload,
                        )
                    ],
                )
            except Exception as e:
                logger.error(f"[EPISODIC] store_episode fehlgeschlagen: {e}")

    # =====================================================================
    # Abrufen
    # =====================================================================

    def recall(self, embedding: np.ndarray, limit: int = 5,
               person: Optional[str] = None) -> List[Dict[str, Any]]:
        """Aehnlichste Episoden abrufen (Cosine Similarity).

        Args:
            embedding: 512-dim Query-Embedding
            limit: Max Anzahl Ergebnisse
            person: Optional — nur Episoden dieser Person

        Returns:
            Liste von Dicts mit score, person, event_type, timestamp, ...
        """
        if not self._ready or self._client is None:
            return []

        # Embedding normalisieren
        emb = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        query_filter = None
        if person:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="person", match=MatchValue(value=person.lower()))]
            )

        with self._lock:
            try:
                results = self._client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=emb.tolist(),
                    query_filter=query_filter,
                    limit=limit,
                )
                episodes = []
                for hit in results.points:
                    entry = dict(hit.payload) if hit.payload else {}
                    entry["score"] = hit.score
                    entry["id"] = str(hit.id)
                    episodes.append(entry)
                return episodes
            except Exception as e:
                logger.error(f"[EPISODIC] recall fehlgeschlagen: {e}")
                return []

    # =====================================================================
    # Status
    # =====================================================================

    def get_stats(self) -> dict:
        """Collection-Statistiken fuer IPC/Panel."""
        if not self._ready or self._client is None:
            return {"ready": False, "points": 0}
        try:
            info = self._client.get_collection(COLLECTION_NAME)
            return {
                "ready": True,
                "points": info.points_count,
                "path": QDRANT_PATH,
                "collection": COLLECTION_NAME,
            }
        except Exception as e:
            return {"ready": False, "error": str(e)}

    def close(self):
        """Qdrant Client sauber schliessen."""
        if self._client:
            try:
                self._client.close()
                logger.info("[EPISODIC] Qdrant Client geschlossen")
            except Exception:
                pass
            self._client = None
            self._ready = False


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[EpisodicMemory] = None
_instance_lock = threading.Lock()


def get_episodic_memory() -> EpisodicMemory:
    """Singleton-Zugriff auf Episodisches Gedaechtnis."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = EpisodicMemory()
    return _instance
