#!/usr/bin/env python3
"""
M.O.L.O.C.H. Tentacle Bridge — Gate 9
=======================================
Verbindet MOLOCH mit externen Smart-Home Geraeten.

Bridges:
1. WLED: WS2812 LED-Strips ueber HTTP (Raum-Stimmungsbeleuchtung)
2. MQTT: Sensor-Nodes, Tuer-Kontakte, Bewegungsmelder
3. Home Assistant: Licht, Heizung, Steckdosen via REST API

Jede Bridge ist optional — fehlendes Geraet = kein Crash.
Publiziert Events auf dem Event Bus bei Sensor-Aenderungen.

Singleton: get_tentacle_bridge()
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger("TentacleBridge")


class WLEDBridge:
    """WLED Controller fuer RGB LED-Strips.

    WLED API: http://<ip>/json/state
    Setzt Farbe, Helligkeit, Effekt basierend auf Atmosphere.
    """

    # Atmosphere → WLED Mapping (Preset-IDs oder RGB)
    ATMOSPHERE_PRESETS = {
        "guardian": {"r": 0, "g": 100, "b": 255, "bri": 80},    # Blau, ruhig
        "shadow": {"r": 255, "g": 30, "b": 0, "bri": 120},      # Rot, bedrohlich
        "berserker": {"r": 255, "g": 0, "b": 0, "bri": 255},    # Rot, voll
        "intimate": {"r": 255, "g": 150, "b": 50, "bri": 40},   # Warm, gedimmt
        "focused": {"r": 200, "g": 200, "b": 255, "bri": 100},  # Kalt-weiss
        "night": {"r": 20, "g": 10, "b": 40, "bri": 10},        # Fast aus
    }

    def __init__(self, host: Optional[str] = None):
        self._host = host  # z.B. "192.168.178.50"
        self._available = False
        self._last_atmosphere: Optional[str] = None
        if host:
            self._check_availability()

    def _check_availability(self):
        try:
            import requests
            resp = requests.get(f"http://{self._host}/json/state", timeout=2)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"[WLED] Erreichbar: {self._host}")
        except Exception:
            self._available = False

    def set_atmosphere(self, atmosphere: str):
        """WLED Farbe basierend auf Atmosphere setzen."""
        if not self._available or not self._host:
            return
        if atmosphere == self._last_atmosphere:
            return
        preset = self.ATMOSPHERE_PRESETS.get(atmosphere)
        if not preset:
            return
        try:
            import requests
            payload = {"on": True, "bri": preset["bri"],
                       "seg": [{"col": [[preset["r"], preset["g"], preset["b"]]]}]}
            requests.post(f"http://{self._host}/json/state",
                          json=payload, timeout=2)
            self._last_atmosphere = atmosphere
            logger.debug(f"[WLED] Atmosphere: {atmosphere}")
        except Exception as e:
            logger.debug(f"[WLED] Fehler: {e}")

    @property
    def available(self) -> bool:
        return self._available


class MQTTBridge:
    """MQTT Bridge fuer Sensor-Nodes.

    Empfaengt: Temperatur, Tuer-Kontakte, Bewegungsmelder
    Publiziert auf Event Bus als sensor.* Events.
    """

    def __init__(self, broker: Optional[str] = None, port: int = 1883):
        self._broker = broker
        self._port = port
        self._available = False
        self._client = None
        self._last_values: Dict[str, Any] = {}
        if broker:
            self._connect()

    def _connect(self):
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id="moloch")
            self._client.on_message = self._on_message
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.subscribe("moloch/sensors/#")
            self._client.loop_start()
            self._available = True
            logger.info(f"[MQTT] Verbunden: {self._broker}:{self._port}")
        except Exception as e:
            self._available = False
            logger.debug(f"[MQTT] Nicht verfuegbar: {e}")

    def _on_message(self, client, userdata, msg):
        """MQTT Nachricht → Event Bus."""
        try:
            topic = msg.topic
            payload = msg.payload.decode("utf-8")
            self._last_values[topic] = payload
            # Event publizieren
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="sensor.mqtt_update",
                source="mqtt_bridge",
                priority=7,
                payload={"topic": topic, "value": payload},
            )
        except Exception as e:
            logger.debug(f"[MQTT] Message-Fehler: {e}")

    @property
    def available(self) -> bool:
        return self._available


class HomeAssistantBridge:
    """Home Assistant REST API Bridge.

    Steuert: Licht, Heizung, Steckdosen, Szenen.
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self._url = url  # z.B. "http://192.168.178.10:8123"
        self._token = token
        self._available = False
        if url and token:
            self._check_availability()

    def _check_availability(self):
        try:
            import requests
            resp = requests.get(f"{self._url}/api/",
                                headers={"Authorization": f"Bearer {self._token}"},
                                timeout=3)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"[HA] Erreichbar: {self._url}")
        except Exception:
            self._available = False

    def call_service(self, domain: str, service: str,
                     entity_id: str, data: Dict = None) -> bool:
        """HA Service aufrufen (z.B. light.turn_on)."""
        if not self._available:
            return False
        try:
            import requests
            payload = {"entity_id": entity_id}
            if data:
                payload.update(data)
            resp = requests.post(
                f"{self._url}/api/services/{domain}/{service}",
                headers={"Authorization": f"Bearer {self._token}",
                         "Content-Type": "application/json"},
                json=payload, timeout=5)
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.debug(f"[HA] Service-Fehler: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._available


class TentacleBridge:
    """Vereint alle externen Bridges. Konfiguration aus Environment."""

    def __init__(self):
        import os
        self.wled = WLEDBridge(host=os.environ.get("MOLOCH_WLED_HOST"))
        self.mqtt = MQTTBridge(broker=os.environ.get("MOLOCH_MQTT_BROKER"))
        self.ha = HomeAssistantBridge(
            url=os.environ.get("MOLOCH_HA_URL"),
            token=os.environ.get("MOLOCH_HA_TOKEN"),
        )
        active = sum([self.wled.available, self.mqtt.available, self.ha.available])
        logger.info(f"[TENTACLE] {active}/3 Bridges aktiv "
                    f"(WLED={self.wled.available} MQTT={self.mqtt.available} HA={self.ha.available})")

    def set_atmosphere(self, atmosphere: str):
        """Atmosphere an alle verfuegbaren Bridges weiterleiten."""
        self.wled.set_atmosphere(atmosphere)

    def get_status(self) -> Dict:
        return {
            "wled": self.wled.available,
            "mqtt": self.mqtt.available,
            "ha": self.ha.available,
        }


# Singleton
_instance: Optional[TentacleBridge] = None

def get_tentacle_bridge() -> TentacleBridge:
    global _instance
    if _instance is None:
        _instance = TentacleBridge()
    return _instance
