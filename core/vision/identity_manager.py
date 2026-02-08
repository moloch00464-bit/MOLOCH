#!/usr/bin/env python3
"""
M.O.L.O.C.H. Identity Manager
=============================

Verwaltet Gesichts-Embeddings für Identity Verification.
- Enrollment: Speichert normalisierte ArcFace-Embeddings
- Matching: Cosine Similarity gegen gespeicherte Identitäten
- Single-User Verification (kein ANN, kein Multi-Match)

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import json
import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent / "config" / "identity_registry.json"


class IdentityManager:
    """
    Verwaltet Gesichts-Identitäten für Verification.

    Verwendet Cosine Similarity auf CPU - kein Hailo erforderlich.
    Thread-safe für read operations, nicht für write.
    """

    def __init__(self, registry_path: str = None):
        """
        Args:
            registry_path: Pfad zur identity_registry.json
        """
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self.identities: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

        logger.info(f"[IDENTITY] Manager initialized, {len(self.identities)} identities loaded")

    def _load_registry(self):
        """Lade Identitäten aus JSON."""
        if not self.registry_path.exists():
            logger.warning(f"[IDENTITY] Registry not found: {self.registry_path}")
            self.identities = {}
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            self.identities = data.get("identities", {})
            logger.info(f"[IDENTITY] Loaded {len(self.identities)} identities from registry")

        except Exception as e:
            logger.error(f"[IDENTITY] Failed to load registry: {e}")
            self.identities = {}

    def _save_registry(self):
        """Speichere Identitäten in JSON."""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.registry_path, "w") as f:
                json.dump({
                    "version": "1.0",
                    "identities": self.identities
                }, f, indent=2)

            logger.info(f"[IDENTITY] Registry saved ({len(self.identities)} identities)")

        except Exception as e:
            logger.error(f"[IDENTITY] Failed to save registry: {e}")

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """L2-Normalisierung für Cosine Similarity."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine Similarity zwischen zwei normalisierten Vektoren."""
        return float(np.dot(vec1, vec2))

    def enroll(self, name: str, embedding: np.ndarray, threshold: float = 0.65) -> bool:
        """
        Registriere neue Identität.

        Args:
            name: Name der Person
            embedding: 512-dim ArcFace Embedding
            threshold: Min. Similarity für Match (default 0.65)

        Returns:
            True wenn erfolgreich
        """
        try:
            # Normalize embedding
            embedding = self.normalize(embedding)

            # Validate embedding size
            if len(embedding) != 512:
                logger.error(f"[IDENTITY] Invalid embedding size: {len(embedding)} (expected 512)")
                return False

            self.identities[name] = {
                "embedding": embedding.tolist(),
                "threshold": threshold,
                "enabled": True
            }

            self._save_registry()
            logger.info(f"[IDENTITY] Enrolled: {name} (threshold={threshold})")
            return True

        except Exception as e:
            logger.error(f"[IDENTITY] Enrollment failed for {name}: {e}")
            return False

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Matche Embedding gegen gespeicherte Identitäten.

        Args:
            embedding: 512-dim ArcFace Embedding

        Returns:
            Tuple (name, score) oder (None, 0.0) wenn kein Match
        """
        if len(self.identities) == 0:
            return None, 0.0

        embedding = self.normalize(embedding)

        best_match: Optional[str] = None
        best_score: float = 0.0

        for name, data in self.identities.items():
            if not data.get("enabled", True):
                continue

            ref_vec = np.array(data["embedding"])
            score = self.cosine_similarity(embedding, ref_vec)

            threshold = data.get("threshold", 0.65)

            if score > threshold and score > best_score:
                best_match = name
                best_score = score

        if best_match:
            logger.debug(f"[IDENTITY] Match: {best_match} (score={best_score:.3f})")

        return best_match, best_score

    def get_identity(self, name: str) -> Optional[Dict[str, Any]]:
        """Hole Identität nach Name."""
        return self.identities.get(name)

    def remove_identity(self, name: str) -> bool:
        """Entferne Identität."""
        if name in self.identities:
            del self.identities[name]
            self._save_registry()
            logger.info(f"[IDENTITY] Removed: {name}")
            return True
        return False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Aktiviere/Deaktiviere Identität."""
        if name in self.identities:
            self.identities[name]["enabled"] = enabled
            self._save_registry()
            logger.info(f"[IDENTITY] {name} enabled={enabled}")
            return True
        return False

    def list_identities(self) -> list:
        """Liste alle Identitäten."""
        return [
            {
                "name": name,
                "threshold": data.get("threshold", 0.65),
                "enabled": data.get("enabled", True),
                "has_embedding": len(data.get("embedding", [])) == 512
            }
            for name, data in self.identities.items()
        ]

    def get_status(self) -> Dict[str, Any]:
        """Status für Debugging."""
        return {
            "registry_path": str(self.registry_path),
            "identity_count": len(self.identities),
            "enabled_count": sum(1 for d in self.identities.values() if d.get("enabled", True)),
            "identities": list(self.identities.keys())
        }


# Singleton instance
_identity_manager: Optional[IdentityManager] = None


def get_identity_manager() -> IdentityManager:
    """Hole IdentityManager Singleton."""
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = IdentityManager()
    return _identity_manager


# Convenience functions
def match_identity(embedding: np.ndarray) -> Tuple[Optional[str], float]:
    """Matche Embedding gegen bekannte Identitäten."""
    return get_identity_manager().match(embedding)


def enroll_identity(name: str, embedding: np.ndarray, threshold: float = 0.65) -> bool:
    """Registriere neue Identität."""
    return get_identity_manager().enroll(name, embedding, threshold)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    manager = IdentityManager()
    print(f"Status: {manager.get_status()}")

    # Test with dummy embedding
    dummy_embedding = np.random.randn(512).astype(np.float32)

    print("\n--- Enroll test identity ---")
    manager.enroll("Test Person", dummy_embedding, threshold=0.65)

    print("\n--- Match test ---")
    # Should match with same embedding
    name, score = manager.match(dummy_embedding)
    print(f"Match: {name} (score={score:.3f})")

    # Should not match with different embedding
    other_embedding = np.random.randn(512).astype(np.float32)
    name, score = manager.match(other_embedding)
    print(f"Match: {name} (score={score:.3f})")

    print("\n--- List identities ---")
    for identity in manager.list_identities():
        print(f"  {identity}")

    print("\n--- Remove test identity ---")
    manager.remove_identity("Test Person")
    print(f"Status: {manager.get_status()}")
