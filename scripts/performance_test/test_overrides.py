"""Mock-Schreiber fuer Akt 4 face_attr-Override.

Hook in core/autonomy/local_llm_bridge.py liest /dev/shm/moloch_test_face_attr_override.json
nur wenn Datei existiert + valid_until_ts in Zukunft. Cleanup garantiert via
context-manager.
"""
from __future__ import annotations

import json
import os
import time
import tempfile
from contextlib import contextmanager
from typing import Iterator

from .config import TEST_OVERRIDE_FACE, OVERRIDE_VALID_DURATION_S


def _atomic_write_override(face_attr: str, valid_seconds: int) -> None:
    """Schreibt Override-File atomic via tempfile + os.replace (NEVER 6)."""
    payload = {
        "face_attr": face_attr,
        "valid_until_ts": time.time() + valid_seconds,
        "created_by": "performance_test",
    }
    dir_path = os.path.dirname(str(TEST_OVERRIDE_FACE))
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(TEST_OVERRIDE_FACE))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _remove_override() -> None:
    try:
        TEST_OVERRIDE_FACE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


@contextmanager
def face_attr_override(
    face_attr: str,
    valid_seconds: int = OVERRIDE_VALID_DURATION_S,
) -> Iterator[None]:
    """Setzt face_attr-Override fuer Dauer des with-Blocks.

    Bei Exit (auch bei Exception): Override-File geloescht. valid_until_ts
    schuetzt zusaetzlich gegen vergessene Cleanup (Auto-Expire).
    """
    _atomic_write_override(face_attr, valid_seconds)
    try:
        yield
    finally:
        _remove_override()
