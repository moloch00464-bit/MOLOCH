#!/usr/bin/env python3
"""
CloudController - Persistente async eWeLink Cloud-Verbindung.

Extrahiert aus moloch_service.py (Regel 10, Phase 1.1).
Kapselt Event Loop, Bridge-Instanz und Cloud-API-Aufrufe.
"""

import os
import time
import asyncio
import threading
import logging

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

logger = logging.getLogger("CloudController")


class CloudController:
    """Persistent async eWeLink cloud controller."""

    def __init__(self):
        self.bridge = None
        self.loop = None
        self._thread = None
        self.connected = False

    def start(self):
        """Event Loop starten und mit eWeLink Cloud verbinden."""
        config = CloudConfig(
            enabled=True,
            api_base_url="https://eu-apia.coolkit.cc",
            app_id=os.environ.get("EWELINK_APP_ID_1", ""),
            app_secret=os.environ.get("EWELINK_APP_SECRET_1", ""),
            device_id="1002817609",
            username=os.environ.get("EWELINK_USERNAME", ""),
            password=os.environ.get("EWELINK_PASSWORD", ""),
        )
        self.bridge = CameraCloudBridge(config)

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.2)
        future = asyncio.run_coroutine_threadsafe(self.bridge.connect(), self.loop)
        try:
            self.connected = future.result(timeout=10)
        except Exception as e:
            logger.error(f"Cloud connect failed: {e}")
            self.connected = False

    def run(self, coro):
        """Async Coroutine im Cloud Event Loop ausfuehren."""
        if not self.loop:
            return False
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Cloud call failed: {e}")
            return False

    def set_smart_tracking(self, on: bool) -> bool:
        """Smart Tracking setzen. Returns True bei Erfolg."""
        if not self.connected:
            return False
        try:
            self.run(self.bridge.set_smart_tracking(on))
            return True
        except Exception as e:
            logger.error(f"Smart Tracking setzen fehlgeschlagen: {e}")
            return False
