#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hailo Analyzer
============================

On-demand face recognition using Hailo-10H NPU.

State Machine:
- STANDBY: NPU idle, no inference
- LOADING: Loading models
- ACTIVE: Running inference

Uses hailo-apps face recognition when available.
Falls back to simple mode (detection only) when not.
"""

import os
import sys
import time
import base64
import logging
import threading
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
HAILO_DEVICE = "/dev/hailo0"
DB_PATH = Path.home() / "moloch" / "data" / "faces"


class HailoState(Enum):
    """Hailo analyzer states."""
    STANDBY = "standby"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class RecognitionResult:
    """Face recognition result."""
    person_id: str
    person_name: str
    confidence: float  # 0.0 - 1.0
    is_known: bool
    face_count: int
    bbox: Optional[Dict] = None
    embedding: Optional[List[float]] = None


class HailoAnalyzer:
    """
    Hailo-10H face recognition analyzer.

    On-demand activation for energy efficiency.
    Uses hailo-apps face recognition pipeline.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize Hailo analyzer.

        Args:
            db_path: Path to face database directory
        """
        self.state = HailoState.STANDBY
        self._state_lock = threading.Lock()

        # Database path
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Hailo device
        self._device_path = HAILO_DEVICE

        # Face database
        self._face_db = None

        # GStreamer pipeline (for future full integration)
        self._pipeline = None

        # Callbacks
        self._on_state_change: Optional[Callable] = None

        # Active time tracking
        self._active_since = 0

        # Check availability
        self._hailo_available = os.path.exists(self._device_path)
        self._hailo_apps_available = self._check_hailo_apps()

        logger.info(f"HailoAnalyzer initialized (hailo={self._hailo_available}, hailo_apps={self._hailo_apps_available})")

    def _check_hailo_apps(self) -> bool:
        """Check if hailo-apps is available."""
        try:
            from hailo_apps.python.core.common.db_handler import DatabaseHandler
            return True
        except ImportError:
            return False

    def _set_state(self, new_state: HailoState):
        """Set analyzer state with callback."""
        with self._state_lock:
            old_state = self.state
            self.state = new_state
            logger.info(f"HailoAnalyzer state: {old_state.value} -> {new_state.value}")

            if self._on_state_change:
                try:
                    self._on_state_change(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    def wake(self) -> bool:
        """
        Wake Hailo from standby.

        Loads models and initializes inference pipeline.

        Returns:
            True if successful
        """
        if self.state == HailoState.ACTIVE:
            logger.debug("Already active")
            return True

        if not self._hailo_available:
            logger.warning("Hailo device not available")
            self._set_state(HailoState.ERROR)
            return False

        self._set_state(HailoState.LOADING)

        try:
            # Initialize face database
            from .face_database import get_face_database
            self._face_db = get_face_database()

            # For now, we work in simplified mode
            # Full GStreamer pipeline integration would go here
            self._active_since = time.time()
            self._set_state(HailoState.ACTIVE)
            logger.info("Hailo analyzer active")
            return True

        except Exception as e:
            logger.error(f"Failed to wake Hailo: {e}")
            self._set_state(HailoState.ERROR)
            return False

    def standby(self) -> float:
        """
        Put Hailo in standby mode.

        Releases resources to save power.

        Returns:
            Seconds that Hailo was active
        """
        if self.state == HailoState.STANDBY:
            return 0

        active_time = time.time() - self._active_since if self._active_since else 0

        # Release resources
        if self._pipeline:
            try:
                self._pipeline = None
            except Exception as e:
                logger.error(f"Error releasing pipeline: {e}")

        self._set_state(HailoState.STANDBY)
        logger.info(f"Hailo standby (was active {active_time:.1f}s)")

        return active_time

    def recognize(self, image_data: bytes, detections: List[Dict] = None) -> Optional[RecognitionResult]:
        """
        Recognize face from image data.

        Args:
            image_data: JPEG image bytes (base64 decoded)
            detections: Face detections (optional, for bbox)

        Returns:
            RecognitionResult or None if no face found
        """
        if self.state != HailoState.ACTIVE:
            if not self.wake():
                return None

        try:
            # Get face count from detections
            face_count = len(detections) if detections else 0
            bbox = detections[0] if detections else None

            # TODO: Full Hailo inference would go here
            # For now, return unknown (face detected but not identified)
            # This will be enhanced when we integrate the full pipeline

            if face_count > 0:
                # In simplified mode: we know a face is present
                # but we can't identify WHO without the full pipeline
                return RecognitionResult(
                    person_id="unknown",
                    person_name="Unknown",
                    confidence=0.0,
                    is_known=False,
                    face_count=face_count,
                    bbox=bbox
                )

            return None

        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None

    def recognize_with_embedding(self, embedding: np.ndarray) -> Optional[RecognitionResult]:
        """
        Recognize face from pre-computed embedding.

        Args:
            embedding: 512-dim face embedding vector

        Returns:
            RecognitionResult or None
        """
        if not self._face_db:
            from .face_database import get_face_database
            self._face_db = get_face_database()

        try:
            result = self._face_db.search(embedding)

            return RecognitionResult(
                person_id=result.person_id,
                person_name=result.name,
                confidence=result.confidence,
                is_known=result.is_known,
                face_count=1,
                embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            )

        except Exception as e:
            logger.error(f"Embedding search error: {e}")
            return None

    def train_face(self, person_name: str, image_paths: List[str]) -> bool:
        """
        Train a new face into the database.

        Note: This requires the full hailo-apps pipeline.
        For now, this is a placeholder that will be implemented
        when we integrate the full GStreamer pipeline.

        Args:
            person_name: Label for the person
            image_paths: List of image file paths

        Returns:
            True if training successful
        """
        if not self._hailo_apps_available:
            logger.warning("hailo-apps not available for training")
            return False

        # TODO: Integrate with hailo-apps training mode
        # This would run the face_recognition pipeline in --mode train

        train_dir = self.db_path / "train" / person_name
        train_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to training directory
        import shutil
        for path in image_paths:
            if os.path.exists(path):
                dest = train_dir / os.path.basename(path)
                shutil.copy2(path, dest)
                logger.info(f"Copied training image: {dest}")

        logger.info(f"Training images prepared for {person_name}")
        logger.info("Run hailo-apps face_recognition --mode train to complete training")

        return True

    def run_training(self) -> bool:
        """
        Run face recognition training using hailo-apps.

        Processes all images in ~/moloch/data/faces/train/

        Returns:
            True if training successful
        """
        if not self._hailo_apps_available:
            logger.error("hailo-apps not available")
            return False

        train_dir = self.db_path / "train"
        if not train_dir.exists() or not any(train_dir.iterdir()):
            logger.warning(f"No training data in {train_dir}")
            return False

        try:
            # Run hailo-apps face_recognition in train mode
            cmd = [
                sys.executable, "-m",
                "hailo_apps.python.pipeline_apps.face_recognition.face_recognition",
                "--mode", "train"
            ]

            logger.info(f"Running training: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("Training completed successfully")
                return True
            else:
                logger.error(f"Training failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return False
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def delete_face(self, person_name: str) -> bool:
        """
        Remove a person from the database.

        Args:
            person_name: Name of person to delete

        Returns:
            True if successful
        """
        if not self._face_db:
            from .face_database import get_face_database
            self._face_db = get_face_database()

        try:
            person = self._face_db.get_person_by_name(person_name)
            if person:
                return self._face_db.delete_person(person.person_id)
            else:
                logger.warning(f"Person not found: {person_name}")
                return False

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def list_known_persons(self) -> List[Dict[str, Any]]:
        """
        List all known persons in database.

        Returns:
            List of person records
        """
        if not self._face_db:
            from .face_database import get_face_database
            self._face_db = get_face_database()

        try:
            persons = self._face_db.list_persons()
            return [
                {
                    "person_id": p.person_id,
                    "name": p.name,
                    "num_samples": p.num_samples,
                    "last_seen": p.last_seen
                }
                for p in persons
            ]

        except Exception as e:
            logger.error(f"List error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        if not self._face_db:
            try:
                from .face_database import get_face_database
                self._face_db = get_face_database()
            except Exception:
                pass

        status = {
            "state": self.state.value,
            "hailo_available": self._hailo_available,
            "hailo_apps_available": self._hailo_apps_available,
            "device": self._device_path,
            "db_path": str(self.db_path)
        }

        if self._face_db:
            try:
                stats = self._face_db.get_stats()
                status["known_persons"] = stats.get("known_persons", 0)
                status["total_samples"] = stats.get("total_samples", 0)
            except Exception:
                pass

        if self.state == HailoState.ACTIVE:
            status["active_seconds"] = time.time() - self._active_since

        return status

    @property
    def is_available(self) -> bool:
        """Check if Hailo device is available."""
        return self._hailo_available

    @property
    def is_active(self) -> bool:
        """Check if analyzer is active."""
        return self.state == HailoState.ACTIVE

    def set_state_callback(self, callback: Callable):
        """Set callback for state changes."""
        self._on_state_change = callback


# Singleton instance
_analyzer: Optional[HailoAnalyzer] = None
_analyzer_lock = threading.Lock()


def get_hailo_analyzer() -> HailoAnalyzer:
    """Get or create HailoAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = HailoAnalyzer()
    return _analyzer
