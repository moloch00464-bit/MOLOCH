#!/usr/bin/env python3
"""
IPCRouter - Panel IPC via /dev/shm und /tmp.

Extrahiert aus moloch_service.py (Phase 4, Regel 10).

Verantwortlichkeiten:
  - Frame-Daten nach /dev/shm/moloch_frame schreiben (fuer Panel Preview)
  - Status-JSON nach /dev/shm/moloch_status.json schreiben
  - Face-State nach /tmp/moloch_face_state.json schreiben
  - Panel-Commands aus /tmp/moloch_cmd_*.json lesen
  - IPC-Dateien beim Shutdown aufraeumen
"""

import os
import json
import struct
import time
import threading
import logging

import numpy as np

logger = logging.getLogger("IPCRouter")

FACE_STATE_PATH = "/tmp/moloch_face_state.json"

# Preview-Groesse fuer Panel IPC (1080p waere 6MB/Frame)
PREVIEW_W = 640
PREVIEW_H = 360


class IPCRouter:
    """Panel IPC: Frame/Status-Write + Command-Polling."""

    # Preview-Groesse (Klassenvariablen fuer externen Zugriff)
    PREVIEW_W = PREVIEW_W
    PREVIEW_H = PREVIEW_H

    def __init__(self):
        self._shm_seq = 0
        self._status_write_lock = threading.Lock()

    # =================================================================
    # Frame Write
    # =================================================================

    def write_frame(self, frame: np.ndarray):
        """Frame + Header nach /dev/shm/moloch_frame schreiben.

        Args:
            frame: BGR-Frame (sollte PREVIEW_W x PREVIEW_H sein)
        """
        try:
            self._shm_seq = (self._shm_seq + 1) & 0xFFFFFFFF
            h, w = frame.shape[:2]
            c = frame.shape[2] if len(frame.shape) > 2 else 1
            header = struct.pack('<IIII', h, w, c, self._shm_seq)
            with open('/dev/shm/moloch_frame.tmp', 'wb') as f:
                f.write(header)
                f.write(frame.tobytes())
            os.rename('/dev/shm/moloch_frame.tmp', '/dev/shm/moloch_frame')
        except Exception:
            pass

    # =================================================================
    # Status Write
    # =================================================================

    def write_status(self, status: dict):
        """Status-JSON nach /dev/shm/moloch_status.json schreiben.

        Args:
            status: Fertig aufgebautes Status-Dict (wird vom Service zusammengebaut)
        """
        try:
            with self._status_write_lock:
                with open('/dev/shm/moloch_status.tmp', 'w') as f:
                    json.dump(status, f)
                os.rename('/dev/shm/moloch_status.tmp', '/dev/shm/moloch_status.json')
        except Exception:
            pass

    # =================================================================
    # Face State Write
    # =================================================================

    def write_face_state(self, name, similarity, person_count,
                         emotion=None, gender=None, age_range=None,
                         head_pose=None, detected_objects=None):
        """Face-Recognition-State fuer IPC mit push_to_talk schreiben.

        Atomic write (tmp + rename) verhindert halb-geschriebene JSON.
        """
        try:
            state = {
                "name": name,
                "similarity": round(similarity, 3),
                "person_count": person_count,
                "emotion": emotion,
                "gender": gender,
                "age_range": age_range,
                "head_pose": {"pitch": head_pose[0], "yaw": head_pose[1],
                              "roll": head_pose[2]} if head_pose else None,
                "detected_objects": detected_objects or [],
                "timestamp": time.time(),
                "source": "moloch_service"
            }
            tmp = str(FACE_STATE_PATH) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.rename(tmp, str(FACE_STATE_PATH))
        except Exception:
            pass

    # =================================================================
    # Command Polling
    # =================================================================

    def poll_commands(self) -> list:
        """Panel-Commands aus /tmp/moloch_cmd_*.json lesen.

        Returns:
            Liste von parsed Command-Dicts (chronologisch sortiert)
        """
        import glob as _glob
        commands = []
        try:
            # Alle cmd-Files lesen (sortiert = chronologisch)
            cmd_files = sorted(_glob.glob('/tmp/moloch_cmd_*.json'))
            # Legacy single-file auch noch unterstuetzen
            legacy = '/tmp/moloch_cmd.json'
            if os.path.exists(legacy):
                cmd_files.insert(0, legacy)
            for cf in cmd_files:
                try:
                    with open(cf) as f:
                        cmd = json.load(f)
                    os.unlink(cf)
                    commands.append(cmd)
                except Exception as e:
                    logger.debug(f"Panel cmd poll ({cf}): {e}")
                    try:
                        os.unlink(cf)
                    except FileNotFoundError:
                        pass
        except Exception:
            pass
        return commands

    # =================================================================
    # Cleanup
    # =================================================================

    def cleanup(self):
        """IPC-Dateien beim Shutdown loeschen."""
        for path in ['/dev/shm/moloch_frame', '/dev/shm/moloch_frame.tmp',
                     '/dev/shm/moloch_status.json', '/dev/shm/moloch_status.tmp',
                     FACE_STATE_PATH]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
