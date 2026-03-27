#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.O.L.O.C.H. Power Monitor — PiPower5 Tentakel
=================================================

Moloch's Bewusstsein fuer seine eigene Stromversorgung.
Liest PiPower5 Board-Daten via I2C (SPC) und meldet:
- Netzteil/Akkubetrieb Wechsel
- Akkustand (Prozent, Spannung)
- Stromverbrauch
- Kritische Zustaende (Low Battery, Power Loss)

Integration:
- CoreIntegrator: environmental_stress bei Akkubetrieb
- EventBus: power_state_changed Events
- Status-JSON: Fuer Panel-Anzeige

Polling: Alle 10s (I2C ist langsam, SSD-schonend)
"""

import logging
import threading
import time

logger = logging.getLogger("PowerMonitor")

# PiPower5 SPC Library (I2C)
try:
    import sys
    sys.path.insert(0, "/opt/pipower5/venv/lib/python3.13/site-packages")
    from spc.spc import SPC
    SPC_AVAILABLE = True
except ImportError:
    SPC_AVAILABLE = False
    logger.warning("[POWER] SPC Library nicht verfuegbar — PiPower5 deaktiviert")


class PowerMonitor:
    """PiPower5 als Moloch-Tentakel: Strom-Bewusstsein.

    Liest alle 10s den Zustand und meldet Aenderungen.
    Kein InfluxDB, kein Disk-Write — nur RAM + Events.
    """

    POLL_INTERVAL = 10.0  # Sekunden zwischen Readings
    LOW_BATTERY_PCT = 20  # Warnung
    CRITICAL_BATTERY_PCT = 10  # Kritisch — Shutdown vorbereiten

    def __init__(self):
        self._spc = None
        self._available = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Letzter bekannter Zustand
        self._state = {
            "battery_pct": -1,
            "battery_voltage": 0,
            "battery_current": 0,
            "input_voltage": 0,
            "input_current": 0,
            "output_voltage": 0,
            "output_current": 0,
            "power_source": "unknown",  # "netzteil" oder "akku"
            "is_charging": False,
            "is_plugged_in": False,
            "power_watts": 0.0,
            "last_update": 0.0,
        }

        # Callbacks
        self._on_power_change = None  # Netzteil <-> Akku Wechsel
        self._on_low_battery = None   # Akkustand niedrig
        self._core_integrator = None  # CoreIntegrator fuer Tension

        # Init SPC
        if SPC_AVAILABLE:
            try:
                self._spc = SPC()
                # Verbindungstest
                data = self._spc.read_all()
                if data and "battery_percentage" in data:
                    self._available = True
                    self._update_state(data)
                    logger.info(f"[POWER] PiPower5 verbunden: "
                               f"Akku={self._state['battery_pct']}% "
                               f"Quelle={self._state['power_source']} "
                               f"Verbrauch={self._state['power_watts']:.1f}W")
            except Exception as e:
                logger.warning(f"[POWER] PiPower5 Init fehlgeschlagen: {e}")

    def set_core_integrator(self, ci):
        """CoreIntegrator anbinden fuer Tension-Feeds."""
        self._core_integrator = ci

    def set_on_power_change(self, callback):
        """Callback bei Netzteil/Akku Wechsel: callback(power_source: str)"""
        self._on_power_change = callback

    def set_on_low_battery(self, callback):
        """Callback bei niedrigem Akku: callback(pct: int)"""
        self._on_low_battery = callback

    def start(self):
        """Polling-Thread starten."""
        if not self._available:
            logger.info("[POWER] PiPower5 nicht verfuegbar — Monitor deaktiviert")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="PowerMonitor")
        self._thread.start()
        logger.info("[POWER] Monitor gestartet (10s Intervall)")

    def stop(self):
        """Polling stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_status(self) -> dict:
        """Aktueller Power-Status (fuer Panel/IPC)."""
        with self._lock:
            return self._state.copy()

    def _poll_loop(self):
        """Polling-Thread: alle 10s Board-Daten lesen."""
        while self._running:
            try:
                data = self._spc.read_all()
                if data:
                    old_source = self._state["power_source"]
                    old_pct = self._state["battery_pct"]

                    self._update_state(data)

                    # Event: Stromquelle gewechselt
                    new_source = self._state["power_source"]
                    if old_source != "unknown" and old_source != new_source:
                        logger.warning(f"[POWER] Stromquelle: {old_source} → {new_source}")
                        if self._on_power_change:
                            try:
                                self._on_power_change(new_source)
                            except Exception:
                                pass
                        # CoreIntegrator: Akkubetrieb = Stress
                        self._feed_tension()

                    # Event: Low Battery
                    new_pct = self._state["battery_pct"]
                    if new_pct <= self.LOW_BATTERY_PCT and old_pct > self.LOW_BATTERY_PCT:
                        logger.warning(f"[POWER] Akku niedrig: {new_pct}%!")
                        if self._on_low_battery:
                            try:
                                self._on_low_battery(new_pct)
                            except Exception:
                                pass

                    # CoreIntegrator regelmaessig fuettern
                    self._feed_tension()

            except Exception as e:
                logger.error(f"[POWER] Lesefehler: {e}")

            time.sleep(self.POLL_INTERVAL)

    def _update_state(self, data: dict):
        """SPC-Rohdaten in lesbaren State umwandeln."""
        with self._lock:
            self._state["battery_pct"] = data.get("battery_percentage", -1)
            self._state["battery_voltage"] = data.get("battery_voltage", 0)
            self._state["battery_current"] = data.get("battery_current", 0)
            self._state["input_voltage"] = data.get("input_voltage", 0)
            self._state["input_current"] = data.get("input_current", 0)
            self._state["output_voltage"] = data.get("output_voltage", 0)
            self._state["output_current"] = data.get("output_current", 0)
            self._state["is_charging"] = data.get("is_charging", False)
            self._state["is_plugged_in"] = data.get("is_input_plugged_in", False)
            self._state["last_update"] = time.time()

            # Stromquelle bestimmen
            if data.get("is_input_plugged_in", False):
                self._state["power_source"] = "netzteil"
            else:
                self._state["power_source"] = "akku"

            # Leistung berechnen (Output V * A, in mV/mA → Watt)
            v_mv = data.get("output_voltage", 0)
            i_ma = data.get("output_current", 0)
            self._state["power_watts"] = round(v_mv * i_ma / 1_000_000, 2)

    def _feed_tension(self):
        """CoreIntegrator mit Power-Zustand fuettern."""
        if not self._core_integrator:
            return
        try:
            source = self._state["power_source"]
            pct = self._state["battery_pct"]

            if source == "akku":
                # Akkubetrieb = Stress (proportional zum Ladezustand)
                # 100% Akku = leichter Stress (0.2), 10% = hoher Stress (0.8)
                stress = 0.2 + (1.0 - pct / 100.0) * 0.6
                self._core_integrator.update_input("power", "environmental_stress",
                                                    min(0.8, stress))
            else:
                # Netzteil = kein Stress
                self._core_integrator.update_input("power", "environmental_stress", 0.0)
        except Exception:
            pass


# Singleton
_instance = None

def get_power_monitor() -> PowerMonitor:
    """Singleton PowerMonitor."""
    global _instance
    if _instance is None:
        _instance = PowerMonitor()
    return _instance
