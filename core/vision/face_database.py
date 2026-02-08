#!/usr/bin/env python3
"""
M.O.L.O.C.H. Face Database
===========================

LanceDB-based face embedding storage.
Wraps hailo-apps DatabaseHandler with M.O.L.O.C.H.-specific interface.

Storage:
- ~/moloch/data/faces/database/  - LanceDB vector database
- ~/moloch/data/faces/samples/   - Face sample images
- ~/moloch/data/faces/train/     - Training images by person
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path.home() / "moloch" / "data" / "faces" / "database"
DEFAULT_SAMPLES_PATH = Path.home() / "moloch" / "data" / "faces" / "samples"


@dataclass
class KnownPerson:
    """Known person record."""
    person_id: str
    name: str
    num_samples: int
    last_seen: float
    confidence_threshold: float


@dataclass
class SearchResult:
    """Face search result."""
    person_id: str
    name: str
    confidence: float  # 0.0 - 1.0
    is_known: bool
    distance: float  # Cosine distance


class FaceDatabase:
    """
    Face embedding database using LanceDB.

    Wraps hailo-apps DatabaseHandler with simplified interface.
    Stores 512-dimensional embeddings from arcface_mobilefacenet.
    """

    def __init__(self,
                 db_path: str = None,
                 samples_path: str = None,
                 threshold: float = 0.5):
        """
        Initialize face database.

        Args:
            db_path: Directory for LanceDB database
            samples_path: Directory for face sample images
            threshold: Default confidence threshold (0.0-1.0)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.samples_path = Path(samples_path) if samples_path else DEFAULT_SAMPLES_PATH
        self.threshold = threshold

        # Ensure directories exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.samples_path.mkdir(parents=True, exist_ok=True)

        # Handler (lazy loaded)
        self._handler = None

        logger.info(f"FaceDatabase initialized (db={self.db_path})")

    def _ensure_handler(self):
        """Ensure database handler is initialized."""
        if self._handler is not None:
            return

        try:
            # Import hailo-apps DatabaseHandler
            from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record

            self._handler = DatabaseHandler(
                db_name='persons.db',
                table_name='persons',
                schema=Record,
                threshold=self.threshold,
                database_dir=str(self.db_path),
                samples_dir=str(self.samples_path)
            )
            logger.info("DatabaseHandler initialized")

        except ImportError as e:
            logger.error(f"Failed to import hailo-apps: {e}")
            logger.error("Install hailo-apps: pip install hailo-apps")
            raise RuntimeError("hailo-apps not available") from e

    def add_person(self,
                   name: str,
                   embedding: np.ndarray,
                   sample_path: str = None) -> str:
        """
        Add new person to database.

        Args:
            name: Person's name/label
            embedding: 512-dim face embedding vector
            sample_path: Optional path to sample image

        Returns:
            person_id (UUID string)
        """
        self._ensure_handler()

        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Create sample path if not provided
        if sample_path is None:
            sample_path = str(self.samples_path / f"{name}_{int(time.time())}.jpg")

        timestamp = int(time.time())

        record = self._handler.create_record(
            embedding=embedding,
            sample=sample_path,
            timestamp=timestamp,
            label=name
        )

        person_id = record['global_id']
        logger.info(f"Added person: {name} (id={person_id})")

        return person_id

    def add_sample(self,
                   person_id: str,
                   embedding: np.ndarray,
                   sample_path: str = None) -> bool:
        """
        Add additional sample to existing person.

        Args:
            person_id: Existing person's ID
            embedding: New 512-dim embedding
            sample_path: Optional path to sample image

        Returns:
            True if successful
        """
        self._ensure_handler()

        try:
            record = self._handler.get_record_by_id(person_id)
            if record is None:
                logger.warning(f"Person not found: {person_id}")
                return False

            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)

            if sample_path is None:
                name = record.get('label', 'unknown')
                sample_path = str(self.samples_path / f"{name}_{int(time.time())}.jpg")

            timestamp = int(time.time())

            self._handler.insert_new_sample(
                record=record,
                embedding=embedding,
                sample=sample_path,
                timestamp=timestamp
            )

            logger.info(f"Added sample to person: {person_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add sample: {e}")
            return False

    def search(self, embedding: np.ndarray) -> SearchResult:
        """
        Search for matching person by embedding.

        Args:
            embedding: 512-dim face embedding to search for

        Returns:
            SearchResult with match info
        """
        self._ensure_handler()

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        result = self._handler.search_record(
            embedding=embedding,
            top_k=1,
            metric_type="cosine"
        )

        # Calculate confidence from distance
        # LanceDB returns cosine distance (0 = identical, 2 = opposite)
        distance = result.get('_distance', 1.0)
        confidence = 1.0 - distance  # Convert to similarity

        is_known = result.get('label', 'Unknown') != 'Unknown'

        return SearchResult(
            person_id=result.get('global_id', ''),
            name=result.get('label', 'Unknown'),
            confidence=max(0.0, min(1.0, confidence)),
            is_known=is_known,
            distance=distance
        )

    def list_persons(self) -> List[KnownPerson]:
        """
        List all known persons in database.

        Returns:
            List of KnownPerson records (excludes 'Unknown')
        """
        self._ensure_handler()

        records = self._handler.get_all_records(only_unknowns=False)

        persons = []
        for record in records:
            label = record.get('label', 'Unknown')
            if label == 'Unknown':
                continue

            samples = record.get('samples_json', [])
            num_samples = len(samples) if isinstance(samples, list) else 0

            persons.append(KnownPerson(
                person_id=record.get('global_id', ''),
                name=label,
                num_samples=num_samples,
                last_seen=record.get('last_sample_recieved_time', 0),
                confidence_threshold=record.get('classificaiton_confidence_threshold', self.threshold)
            ))

        return persons

    def get_person(self, person_id: str) -> Optional[KnownPerson]:
        """Get person by ID."""
        self._ensure_handler()

        try:
            record = self._handler.get_record_by_id(person_id)
            if record is None:
                return None

            samples = record.get('samples_json', [])
            num_samples = len(samples) if isinstance(samples, list) else 0

            return KnownPerson(
                person_id=record.get('global_id', ''),
                name=record.get('label', 'Unknown'),
                num_samples=num_samples,
                last_seen=record.get('last_sample_recieved_time', 0),
                confidence_threshold=record.get('classificaiton_confidence_threshold', self.threshold)
            )
        except Exception as e:
            logger.error(f"Failed to get person: {e}")
            return None

    def get_person_by_name(self, name: str) -> Optional[KnownPerson]:
        """Get person by name/label."""
        self._ensure_handler()

        try:
            record = self._handler.get_record_by_label(name)
            if record is None:
                return None

            samples = record.get('samples_json', [])
            num_samples = len(samples) if isinstance(samples, list) else 0

            return KnownPerson(
                person_id=record.get('global_id', ''),
                name=record.get('label', 'Unknown'),
                num_samples=num_samples,
                last_seen=record.get('last_sample_recieved_time', 0),
                confidence_threshold=record.get('classificaiton_confidence_threshold', self.threshold)
            )
        except Exception as e:
            logger.error(f"Failed to get person by name: {e}")
            return None

    def update_name(self, person_id: str, new_name: str) -> bool:
        """
        Update person's name/label.

        Args:
            person_id: Person's ID
            new_name: New name to set

        Returns:
            True if successful
        """
        self._ensure_handler()

        try:
            self._handler.update_record_label(person_id, new_name)
            logger.info(f"Updated name for {person_id}: {new_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update name: {e}")
            return False

    def delete_person(self, person_id: str) -> bool:
        """
        Delete person from database.

        Args:
            person_id: Person's ID

        Returns:
            True if successful
        """
        self._ensure_handler()

        try:
            self._handler.delete_record(person_id)
            logger.info(f"Deleted person: {person_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete person: {e}")
            return False

    def clear_unknown(self) -> int:
        """
        Remove all 'Unknown' persons from database.

        Returns:
            Number of records removed
        """
        self._ensure_handler()

        try:
            before = len(self._handler.get_all_records(only_unknowns=True))
            self._handler.clear_unknown_labels()
            after = len(self._handler.get_all_records(only_unknowns=True))
            removed = before - after
            logger.info(f"Cleared {removed} unknown records")
            return removed
        except Exception as e:
            logger.error(f"Failed to clear unknown: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        self._ensure_handler()

        try:
            all_records = self._handler.get_all_records()
            known = [r for r in all_records if r.get('label', 'Unknown') != 'Unknown']
            unknown = [r for r in all_records if r.get('label', 'Unknown') == 'Unknown']

            total_samples = sum(
                len(r.get('samples_json', []))
                for r in all_records
                if isinstance(r.get('samples_json'), list)
            )

            return {
                'total_persons': len(all_records),
                'known_persons': len(known),
                'unknown_persons': len(unknown),
                'total_samples': total_samples,
                'db_path': str(self.db_path),
                'samples_path': str(self.samples_path)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}


# Singleton instance
_database: Optional[FaceDatabase] = None


def get_face_database() -> FaceDatabase:
    """Get or create FaceDatabase instance."""
    global _database
    if _database is None:
        _database = FaceDatabase()
    return _database
