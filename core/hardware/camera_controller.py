#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Camera Controller
==============================

Verwaltet die Sonoff PTZ Kamera fuer M.O.L.O.C.H.
Bietet API fuer Bilderfassung, PTZ-Steuerung und Autonomie.

Kamera:
- Sonoff CAM-PT2: RTSP Stream + ONVIF PTZ (192.168.178.25)
"""

import os
import cv2
import time
import json
import logging
import threading
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

# PTZ Controller
from .sonoff_ptz_controller import SonoffPTZController, AutonomyMode

logger = logging.getLogger(__name__)



# Load camera config
import json
from pathlib import Path

def load_camera_config():
    config_path = Path(__file__).parent.parent.parent / 'config' / 'sonoff_camera.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None

CAMERA_CONFIG = load_camera_config()

class CameraID(Enum):
    """Kamera-Identifikatoren"""
    SONOFF = "sonoff"
    XIAO = "xiao"  # Legacy, no longer supported


@dataclass
class CameraFrame:
    """Ein Kamera-Frame mit Metadaten"""
    camera_id: CameraID
    image: np.ndarray  # BGR OpenCV Format
    timestamp: float
    width: int
    height: int
    detections: list = None  # Optional: Erkennungen im Bild


@dataclass
class CameraStatus:
    """Status einer Kamera"""
    camera_id: CameraID
    connected: bool
    resolution: Tuple[int, int]
    fps: float
    error: Optional[str] = None


class CameraController:
    """
    Camera Controller fuer M.O.L.O.C.H.

    Verwaltet Sonoff PTZ Kamera und bietet:
    - Frame-Erfassung via RTSP
    - PTZ-Steuerung
    - Snapshot-Speicherung
    """

    # Sonoff Konfiguration
    SONOFF_IP = os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP")
    SONOFF_RTSP = CAMERA_CONFIG["camera"]["connection"]["url"] if CAMERA_CONFIG else os.environ.get("MOLOCH_RTSP_URL", "rtsp://USER:PASS@CAMERA_IP:554/av_stream/ch0")
    SONOFF_ONVIF_USER = os.environ.get("MOLOCH_CAMERA_USER", "CHANGE_ME")
    SONOFF_ONVIF_PASS = os.environ.get("MOLOCH_CAMERA_PASS", "CHANGE_ME")

    # Snapshot-Verzeichnis
    SNAPSHOT_DIR = Path("/home/molochzuhause/moloch/snapshots")

    def __init__(self):
        self.sonoff_cap: Optional[cv2.VideoCapture] = None
        self.ptz: Optional[SonoffPTZController] = None

        # Status
        self.sonoff_connected = False
        self._stop_event = threading.Event()

        # Frame Caches
        self._sonoff_frame: Optional[CameraFrame] = None
        self._frame_lock = threading.Lock()

        # Background Capture Threads
        self._sonoff_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_person_detected: Optional[Callable] = None

        # Ensure snapshot dir exists
        self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("CameraController initialisiert")

    def connect_all(self) -> Dict[str, bool]:
        """Verbinde Kamera. Gibt Status zurueck."""
        results = {}

        # Sonoff via RTSP
        results["sonoff"] = self._connect_sonoff()

        # PTZ Controller
        if results["sonoff"]:
            self._connect_ptz()

        return results

    def _connect_sonoff(self) -> bool:
        """Verbinde Sonoff RTSP Stream - optimiert fuer stabile 20+ FPS"""
        try:
            import os
            # UDP + Low-Delay fuer minimale Latenz (5-9ms)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

            self.sonoff_cap = cv2.VideoCapture(self.SONOFF_RTSP, cv2.CAP_FFMPEG)

            if self.sonoff_cap.isOpened():
                # Buffer auf 1 fuer minimale Latenz
                self.sonoff_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frame = self.sonoff_cap.read()
                if ret and frame is not None:
                    self.sonoff_connected = True
                    h, w = frame.shape[:2]
                    logger.info(f"Sonoff verbunden: {w}x{h} (20+ FPS optimiert)")
                    return True

            logger.warning("Sonoff RTSP konnte nicht geoeffnet werden")
            self.sonoff_connected = False
            return False

        except Exception as e:
            logger.error(f"Sonoff Verbindungsfehler: {e}")
            self.sonoff_connected = False
            return False

    def _connect_ptz(self) -> bool:
        """Verbinde Sonoff PTZ via ONVIF"""
        try:
            self.ptz = SonoffPTZController(
                ip=self.SONOFF_IP,
                user=self.SONOFF_ONVIF_USER,
                password=self.SONOFF_ONVIF_PASS
            )
            if self.ptz.connect():
                logger.info("PTZ Controller verbunden")
                return True
            return False
        except Exception as e:
            logger.warning(f"PTZ nicht verfuegbar: {e}")
            return False

    def start_capture(self):
        """Starte Background-Thread fuer kontinuierliche Frame-Erfassung"""
        self._stop_event.clear()

        if self.sonoff_connected:
            self._sonoff_thread = threading.Thread(
                target=self._sonoff_capture_loop,
                daemon=True
            )
            self._sonoff_thread.start()

        logger.info("Capture-Thread gestartet")

    def stop_capture(self):
        """Stoppe Capture-Thread"""
        self._stop_event.set()

        if self._sonoff_thread:
            self._sonoff_thread.join(timeout=2)

        logger.info("Capture-Thread gestoppt")

    def _sonoff_capture_loop(self):
        """Background Loop fuer Sonoff Frame-Erfassung - LOW LATENCY"""
        frame_count = 0
        last_fps_time = time.time()

        while not self._stop_event.is_set():
            try:
                if self.sonoff_cap and self.sonoff_cap.isOpened():
                    # Leere Buffer: grab() verwirft alte Frames schnell
                    # Dann retrieve() fuer den neuesten Frame
                    grabbed = self.sonoff_cap.grab()
                    if grabbed:
                        ret, frame = self.sonoff_cap.retrieve()
                        if ret and frame is not None:
                            frame_count += 1
                            now = time.time()

                            # FPS logging alle 5 Sekunden
                            if now - last_fps_time >= 5.0:
                                fps = frame_count / (now - last_fps_time)
                                logger.debug(f"Sonoff capture: {fps:.1f} FPS")
                                frame_count = 0
                                last_fps_time = now

                            with self._frame_lock:
                                self._sonoff_frame = CameraFrame(
                                    camera_id=CameraID.SONOFF,
                                    image=frame,
                                    timestamp=now,
                                    width=frame.shape[1],
                                    height=frame.shape[0]
                                )
                        # Minimaler Sleep um CPU zu schonen
                        time.sleep(0.001)
                    else:
                        self._connect_sonoff()
                        time.sleep(1)
                else:
                    time.sleep(0.1)  # Kurz warten wenn Cap nicht offen
            except Exception as e:
                logger.error(f"Sonoff capture error: {e}")
                time.sleep(1)

    def get_frame(self, camera: CameraID) -> Optional[CameraFrame]:
        """Hole aktuellen Frame einer Kamera"""
        with self._frame_lock:
            if camera == CameraID.SONOFF:
                return self._sonoff_frame
        return None

    def take_snapshot(self, camera: CameraID = None) -> Optional[str]:
        """
        Speichere Snapshot.

        Args:
            camera: Welche Kamera (default: Sonoff)

        Returns:
            Pfad zum Snapshot oder None
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        frame = self.get_frame(CameraID.SONOFF)
        if frame and frame.image is not None:
            filename = f"sonoff_{timestamp}.jpg"
            filepath = self.SNAPSHOT_DIR / filename
            cv2.imwrite(str(filepath), frame.image)
            logger.info(f"Snapshot gespeichert: {filepath}")
            return str(filepath)

        return None

    # === Status ===

    def get_status(self) -> Dict[str, CameraStatus]:
        """Hole Status der Kamera"""
        status = {}

        if self.sonoff_connected and self._sonoff_frame:
            status["sonoff"] = CameraStatus(
                camera_id=CameraID.SONOFF,
                connected=True,
                resolution=(self._sonoff_frame.width, self._sonoff_frame.height),
                fps=15.0
            )
        else:
            status["sonoff"] = CameraStatus(
                camera_id=CameraID.SONOFF,
                connected=False,
                resolution=(0, 0),
                fps=0,
                error="Nicht verbunden"
            )

        return status

    def disconnect(self):
        """Trenne Verbindung"""
        self.stop_capture()

        if self.sonoff_cap:
            self.sonoff_cap.release()
            self.sonoff_cap = None

        self.sonoff_connected = False

        logger.info("Kamera getrennt")


# Singleton
_camera_controller: Optional[CameraController] = None


def get_camera_controller() -> CameraController:
    """Hole oder erstelle CameraController Singleton"""
    global _camera_controller
    if _camera_controller is None:
        _camera_controller = CameraController()
    return _camera_controller
