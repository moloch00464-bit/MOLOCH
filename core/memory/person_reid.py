#!/usr/bin/env python3
"""
M.O.L.O.C.H. Person Re-Identification — Embedding-basierte Identitaet
=======================================================================

Nutzt ArcFace-Embeddings aus der TAPPAS-Pipeline (512-dim).
Kein eigenes HEF noetig — Hailo reid_multisource macht es genauso.

Funktionen:
  - Embedding-Datenbank verwalten (add/remove/persist)
  - Matching via Cosine Similarity
  - Tracking-IDs mit Identitaeten verknuepfen

Datenbank: /mnt/moloch-data/memory/reid_db.json (persistent auf SSD2)

Singleton: get_reid()

Gate 2: Identity (ReID + Qdrant VITALE)
"""

import json
import logging
import threading
import os
import time
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import numpy as np

logger = logging.getLogger("MolochReID")

# Persistent DB auf SSD2
REID_DB_PATH = "/mnt/moloch-data/memory/reid_db.json"
REID_EMBEDDING_DIM = 512
DEFAULT_THRESHOLD = 0.60


class PersonReID:
    """
    Person Re-Identification via ArcFace-Embeddings.

    Embeddings kommen aus der TAPPAS-Pipeline (GStreamer ArcFace).
    Kein eigenes NPU-Inference — nur DB-Verwaltung + Matching.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._db: Dict[str, np.ndarray] = {}  # {name: avg_embedding}
        self._db_counts: Dict[str, int] = {}  # {name: n_embeddings} fuer avg
        self._threshold = DEFAULT_THRESHOLD
        self._load_db()

    # =====================================================================
    # DB Persistence (SSD2)
    # =====================================================================

    def _load_db(self):
        """Embedding-DB von Disk laden."""
        if not os.path.exists(REID_DB_PATH):
            logger.info("[REID] Keine DB gefunden, starte leer")
            return
        try:
            with open(REID_DB_PATH, "r") as f:
                data = json.load(f)
            for name, entry in data.items():
                emb = np.array(entry["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self._db[name] = emb
                self._db_counts[name] = entry.get("count", 1)
            logger.info(f"[REID] DB geladen: {len(self._db)} Identitaeten")
        except Exception as e:
            logger.error(f"[REID] DB laden fehlgeschlagen: {e}")

    def _save_db(self):
        """Embedding-DB atomar auf Disk speichern (NEVER 6: tmp + replace)."""
        try:
            os.makedirs(os.path.dirname(REID_DB_PATH), exist_ok=True)
            data = {}
            for name, emb in self._db.items():
                data[name] = {
                    "embedding": emb.tolist(),
                    "count": self._db_counts.get(name, 1),
                }
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(REID_DB_PATH), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, REID_DB_PATH)
            logger.info(f"[REID] DB gespeichert: {len(data)} Identitaeten")
        except Exception as e:
            logger.error(f"[REID] DB speichern fehlgeschlagen: {e}")

    # =====================================================================
    # DB Verwaltung
    # =====================================================================

    def add_embedding(self, name: str, embedding: np.ndarray, save: bool = True):
        """Embedding zur DB hinzufuegen (Running Average).

        Args:
            name: Identitaets-Name (lowercase)
            embedding: 512-dim ArcFace-Embedding (wird L2-normalisiert)
            save: Sofort auf Disk speichern
        """
        name = name.lower()
        emb = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        with self._lock:
            if name in self._db:
                # Running Average: (old * n + new) / (n + 1)
                n = self._db_counts.get(name, 1)
                self._db[name] = (self._db[name] * n + emb) / (n + 1)
                # Re-normalisieren
                norm = np.linalg.norm(self._db[name])
                if norm > 0:
                    self._db[name] = self._db[name] / norm
                self._db_counts[name] = n + 1
            else:
                self._db[name] = emb
                self._db_counts[name] = 1

        if save:
            self._save_db()
        logger.info(f"[REID] Embedding fuer '{name}' hinzugefuegt "
                    f"(n={self._db_counts.get(name, 1)})")

    def remove_identity(self, name: str):
        """Identitaet aus DB entfernen."""
        name = name.lower()
        with self._lock:
            self._db.pop(name, None)
            self._db_counts.pop(name, None)
        self._save_db()
        logger.info(f"[REID] '{name}' entfernt")

    def get_identities(self) -> List[str]:
        """Alle bekannten Identitaeten."""
        with self._lock:
            return list(self._db.keys())

    # =====================================================================
    # Matching
    # =====================================================================

    def match(self, embedding: np.ndarray,
              threshold: float = None) -> Tuple[Optional[str], float]:
        """Embedding gegen DB matchen (Cosine Similarity).

        Args:
            embedding: 512-dim ArcFace-Embedding
            threshold: Minimum Similarity (default: self._threshold)

        Returns:
            (name, similarity) bei Match, oder (None, best_similarity)
        """
        if embedding is None:
            return None, 0.0

        thresh = threshold if threshold is not None else self._threshold
        emb = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_name = None
        best_sim = 0.0

        with self._lock:
            for name, db_emb in self._db.items():
                sim = float(np.dot(emb, db_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= thresh:
            return best_name, best_sim
        return None, best_sim

    def set_threshold(self, threshold: float):
        """Matching-Threshold setzen."""
        self._threshold = max(0.0, min(1.0, threshold))
        logger.info(f"[REID] Threshold: {self._threshold:.2f}")

    def get_status(self) -> dict:
        """Status-Dict fuer IPC/Panel."""
        with self._lock:
            return {
                "identities": len(self._db),
                "names": list(self._db.keys()),
                "threshold": self._threshold,
                "db_path": REID_DB_PATH,
            }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[PersonReID] = None
_instance_lock = threading.Lock()


def get_reid() -> PersonReID:
    """Singleton-Zugriff auf Person ReID."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PersonReID()
    return _instance
