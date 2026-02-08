#!/usr/bin/env python3
"""
M.O.L.O.C.H. Identity Enrollment CLI
====================================

Registriert neue Gesichter in der Identity Registry.
Verwendet bestehende Hailo Pipeline - kein paralleler Device Context.

Usage:
    python scripts/enroll_identity.py --name "First Moloch" --image face.jpg
    python scripts/enroll_identity.py --name "First Moloch" --camera  # Live capture
    python scripts/enroll_identity.py --list
    python scripts/enroll_identity.py --remove "Test Person"

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import argparse
import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.vision.identity_manager import IdentityManager, get_identity_manager

logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = PROJECT_ROOT / "config"
IDENTITY_REGISTRY = CONFIG_DIR / "identity_registry.json"
MODEL_REGISTRY = CONFIG_DIR / "model_registry.json"


def load_image(image_path: str) -> np.ndarray:
    """Lade Bild als numpy array."""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    except ImportError:
        raise ImportError("OpenCV (cv2) required for image loading")


def detect_face(image: np.ndarray) -> tuple:
    """
    Erkenne Gesicht im Bild.

    Returns:
        Tuple (face_crop, bbox) oder (None, None)
    """
    try:
        import cv2

        # Simple face detection with Haar Cascade (CPU fallback)
        # In production: Use SCRFD via Hailo
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            logger.warning("No face detected in image")
            return None, None

        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using largest")

        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        face_crop = image[y1:y2, x1:x2]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        logger.info(f"Face detected: {bbox}")
        return face_crop, bbox

    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return None, None


def compute_embedding_cpu(face_crop: np.ndarray) -> np.ndarray:
    """
    Berechne Face Embedding auf CPU (Fallback).

    HINWEIS: Dies ist ein Platzhalter!
    In Production sollte das ArcFace Modell über Hailo laufen.
    """
    try:
        import cv2

        # Resize to ArcFace input size
        face_resized = cv2.resize(face_crop, (112, 112))

        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0

        # PLACEHOLDER: Generate deterministic embedding from image
        # This should be replaced with actual ArcFace inference!
        logger.warning("Using CPU placeholder embedding - NOT for production!")

        # Create embedding from image statistics (NOT a real face embedding!)
        # This is just for testing the pipeline
        flat = face_normalized.flatten()
        embedding = np.zeros(512, dtype=np.float32)

        # Fill with image-derived features
        for i in range(512):
            idx = (i * len(flat)) // 512
            embedding[i] = flat[idx]

        # Add some noise for uniqueness
        embedding += np.random.randn(512).astype(np.float32) * 0.01

        return embedding

    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return None


def compute_embedding_hailo(face_crop: np.ndarray) -> np.ndarray:
    """
    Berechne Face Embedding über Hailo NPU.

    Verwendet bestehende Pipeline - KEIN neuer Device Context!
    """
    try:
        # Import Hailo embedding model
        # TODO: Implement when face_embedding pipeline is ready
        logger.warning("Hailo embedding not yet implemented, using CPU fallback")
        return compute_embedding_cpu(face_crop)

    except Exception as e:
        logger.error(f"Hailo embedding failed: {e}")
        return None


def enroll_from_image(name: str, image_path: str, threshold: float = 0.65) -> bool:
    """Enrollment von Bild-Datei."""
    print(f"Enrolling '{name}' from image: {image_path}")

    # Load image
    image = load_image(image_path)
    print(f"  Image loaded: {image.shape}")

    # Detect face
    face_crop, bbox = detect_face(image)
    if face_crop is None:
        print("  ERROR: No face detected!")
        return False
    print(f"  Face detected: {bbox}")

    # Compute embedding
    embedding = compute_embedding_cpu(face_crop)  # TODO: Use Hailo
    if embedding is None:
        print("  ERROR: Embedding computation failed!")
        return False
    print(f"  Embedding computed: {embedding.shape}")

    # Enroll
    manager = get_identity_manager()
    success = manager.enroll(name, embedding, threshold)

    if success:
        print(f"  SUCCESS: '{name}' enrolled with threshold {threshold}")
    else:
        print(f"  ERROR: Enrollment failed!")

    return success


def enroll_from_camera(name: str, threshold: float = 0.65) -> bool:
    """Enrollment von Live-Kamera (TODO)."""
    print(f"Live camera enrollment not yet implemented")
    print("Use --image instead")
    return False


def list_identities():
    """Liste alle Identitäten."""
    manager = get_identity_manager()
    identities = manager.list_identities()

    print(f"\nRegistered Identities ({len(identities)}):")
    print("-" * 50)

    if not identities:
        print("  (none)")
    else:
        for identity in identities:
            status = "enabled" if identity["enabled"] else "disabled"
            embedding_status = "OK" if identity["has_embedding"] else "MISSING"
            print(f"  {identity['name']}")
            print(f"    Threshold: {identity['threshold']}")
            print(f"    Status: {status}")
            print(f"    Embedding: {embedding_status}")


def remove_identity(name: str) -> bool:
    """Entferne Identität."""
    manager = get_identity_manager()
    success = manager.remove_identity(name)

    if success:
        print(f"Removed: '{name}'")
    else:
        print(f"Not found: '{name}'")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="M.O.L.O.C.H. Identity Enrollment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Enroll from image:
    python enroll_identity.py --name "First Moloch" --image face.jpg

  List all identities:
    python enroll_identity.py --list

  Remove identity:
    python enroll_identity.py --remove "Test Person"
        """
    )

    parser.add_argument("--name", "-n", help="Name for enrollment")
    parser.add_argument("--image", "-i", help="Image file for enrollment")
    parser.add_argument("--camera", "-c", action="store_true", help="Use live camera")
    parser.add_argument("--threshold", "-t", type=float, default=0.65,
                        help="Match threshold (default: 0.65)")
    parser.add_argument("--list", "-l", action="store_true", help="List all identities")
    parser.add_argument("--remove", "-r", help="Remove identity by name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Execute command
    if args.list:
        list_identities()
        return 0

    if args.remove:
        success = remove_identity(args.remove)
        return 0 if success else 1

    if args.name:
        if args.image:
            success = enroll_from_image(args.name, args.image, args.threshold)
            return 0 if success else 1
        elif args.camera:
            success = enroll_from_camera(args.name, args.threshold)
            return 0 if success else 1
        else:
            print("ERROR: --name requires --image or --camera")
            return 1

    # No command specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
