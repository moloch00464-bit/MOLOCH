#!/usr/bin/env python3
"""
LEDController - Status-LED Signaling via eWeLink Cloud.

Extrahiert aus moloch_service.py (Phase 4, Regel 10).
Steuert blaue Status-LED + weisse Flutlicht-LED der Sonoff CAM-PT2.

Verantwortlichkeiten:
  - LED on/off/blink (Status-LED)
  - Markus-Standlicht (LED an wenn Markus erkannt, Hysterese gegen Flackern)
  - Flash (weisse LED kurz an fuer Teachen Snapshot)
  - CoreIntegrator-Integration (Berserker-Zone -> blinken statt Standlicht)
"""

import time
import threading
import logging

logger = logging.getLogger("LEDController")


class LEDController:
    """Status-LED Steuerung mit Hysterese und CoreIntegrator-Integration."""

    def __init__(self, cloud=None, core_integrator=None):
        """
        Args:
            cloud: CloudController Instanz fuer API-Calls
            core_integrator: CoreIntegrator fuer Zone/Attention (optional)
        """
        self._cloud = cloud
        self._core_integrator = core_integrator

        # LED Erkennungs-Indikator State
        self._indicator_state = False
        self._markus_on = False
        self._markus_last_seen = 0

        # Hysterese (Frame-Counter gegen Flackern)
        self._markus_on_streak = 0
        self._markus_off_streak = 0
        self._ON_THRESHOLD = 3
        self._OFF_THRESHOLD = 30

        # Generelle Detection Hysterese
        self._detect_on_streak = 0
        self._detect_off_streak = 0
        self._DETECT_ON_THRESH = 3
        self._DETECT_OFF_THRESH = 15

        # API Rate-Limit: mindestens 2s zwischen LED-Calls
        self._last_api_call = 0
        self._API_MIN_INTERVAL = 2.0

        # Gate0 Phase 6: Personality-Modus (EINE Wahrheit mit Iris)
        self._personality_mode = "guardian"

    def set_cloud(self, cloud):
        """Cloud-Referenz setzen (fuer spaete Initialisierung)."""
        self._cloud = cloud

    def set_core_integrator(self, ci):
        """CoreIntegrator-Referenz setzen."""
        self._core_integrator = ci

    @property
    def markus_on(self) -> bool:
        """Ob Markus-Standlicht gerade aktiv ist."""
        return self._markus_on

    @property
    def personality_mode(self) -> str:
        """Aktueller Personality-Modus (fuer Panel-Indikator)."""
        return self._personality_mode

    # =================================================================
    # Basis-LED-Operationen
    # =================================================================

    def on(self):
        """Status-LED AN (blau, sichtbar)."""
        if not self._cloud or not self._cloud.connected:
            return
        try:
            self._cloud.run(self._cloud.bridge.set_status_led(True))
        except Exception:
            pass

    def off(self):
        """Status-LED AUS."""
        if not self._cloud or not self._cloud.connected:
            return
        try:
            self._cloud.run(self._cloud.bridge.set_status_led(False))
        except Exception:
            pass

    def blink(self, count=6, interval=0.3):
        """Status-LED blinken, danach AN lassen (MOLOCH hat noch Kontrolle).

        Mini-Bug Fix 2026-04-28: Markus-Standlicht hat Prioritaet. Wenn
        _markus_on=True → kein Blink-Override, sonst flackert LED wild
        durch konkurrierende Atmosphere/Behavior-Trigger und Markus sieht
        sie als "leuchtet nicht".
        """
        if self._markus_on:
            return  # Markus erkannt -> Standlicht haelt, kein Blink
        def do_blink():
            for _ in range(count):
                self.off()
                time.sleep(interval)
                self.on()
                time.sleep(interval)
        threading.Thread(target=do_blink, daemon=True).start()

    # =================================================================
    # Indikator-Logik (State-Change-Detection + Rate-Limit)
    # =================================================================

    def indicator_set(self, on: bool):
        """LED Indikator: Setzt LED nur bei State-Aenderung (vermeidet API-Spam).

        Markus-Standlicht hat Prioritaet: wenn markus_on, wird LED nicht ausgeschaltet.
        CoreIntegrator: led_feedback_frequency steuert ob LED sofort oder verzoegert reagiert.
        API Rate-Limit: mindestens _API_MIN_INTERVAL zwischen Calls.
        """
        if not on and self._markus_on:
            return
        if on == self._indicator_state:
            return

        # API Rate-Limit
        now = time.time()
        if now - self._last_api_call < self._API_MIN_INTERVAL:
            return

        # CoreIntegrator: Bei niedriger Attention LED-Feedback verzoegern
        if on and self._core_integrator:
            try:
                led_freq = self._core_integrator.get_effects().get("led_feedback_frequency", 0.5)
                if led_freq < 0.3:
                    return
            except Exception:
                pass

        self._indicator_state = on
        self._last_api_call = now
        if on:
            self.on()
        else:
            self.off()

    def indicator_markus_seen(self):
        """Markus erkannt (nach Hysterese-Pruefung): LED an (Standlicht).

        CoreIntegrator: Berserker-Zone -> LED blinkt statt Standlicht.
        """
        self._markus_last_seen = time.time()
        self._markus_off_streak = 0

        # Berserker-Zone: LED blinken statt Standlicht
        # ArbitrationEngine hat Vorrang, Fallback auf CoreIntegrator
        if self._core_integrator:
            try:
                from core.arbitration import get_arbitration
                zone = get_arbitration().get_zone()
            except Exception:
                try:
                    zone = self._core_integrator.get_personality_zone()
                except Exception:
                    zone = "guardian"
            if zone == "berserker":
                now = time.time()
                if now - self._last_api_call >= self._API_MIN_INTERVAL:
                    self._last_api_call = now
                    self.blink(count=3, interval=0.15)
                return

        if not self._markus_on:
            now = time.time()
            if now - self._last_api_call < self._API_MIN_INTERVAL:
                self._markus_on = True
                self._indicator_state = True
                return

            self._markus_on = True
            self._indicator_state = True
            self._last_api_call = now
            self.on()
            logger.info("[LED] Markus Hysterese: On-Streak erreicht -> Standlicht AN")

    # =================================================================
    # Hysterese-Update (pro Frame in der Inference Loop aufgerufen)
    # =================================================================

    def update_hysteresis(self, markus_recognized: bool,
                          face_detected: bool, persons_detected: bool,
                          moloch_has_control: bool,
                          personality_mode: str = "guardian"):
        """LED-Hysterese aktualisieren. Wird pro Inference-Frame aufgerufen.

        Gate0 Phase 6: LED zeigt Wahrheit.
        LED blau NUR wenn Guardian + Markus erkannt.
        Shadow/Berserker = LED AUS (physisch blau nicht moeglich -> aus).

        Args:
            markus_recognized: Markus in diesem Frame erkannt
            face_detected: Irgendein Gesicht erkannt
            persons_detected: Person (YOLO) erkannt
            moloch_has_control: MOLOCH hat Kamera-Kontrolle
            personality_mode: Aktueller Modus (guardian/shadow/berserker)
        """
        self._personality_mode = personality_mode

        # Shadow/Berserker: LED IMMER aus — keine blaue LED bei Bedrohung
        if personality_mode in ("shadow", "berserker"):
            if self._markus_on or self._indicator_state:
                self._markus_on = False
                self._markus_on_streak = 0
                self._markus_off_streak = 0
                self.indicator_set(False)
                logger.info(f"[LED] Modus {personality_mode} -> LED AUS (Wahrheit)")
            return

        # Guardian: LED blau NUR wenn Markus erkannt (Hysterese)
        if markus_recognized:
            self._markus_on_streak += 1
            self._markus_off_streak = 0
            if self._markus_on_streak >= self._ON_THRESHOLD:
                self.indicator_markus_seen()
        else:
            self._markus_on_streak = 0
            if self._markus_on:
                self._markus_off_streak += 1
                if self._markus_off_streak >= self._OFF_THRESHOLD:
                    self._markus_off_streak = 0
                    self._markus_on = False
                    self.indicator_set(False)
                    logger.info("[LED] Markus Hysterese: Off-Streak erreicht -> LED AUS")

    # =================================================================
    # Weisse LED (Flutlicht/Flash)
    # =================================================================

    def flash_white(self):
        """Kurzer Blitz der weissen LED (200ms) - laeuft in Daemon-Thread."""
        if not self._cloud or not self._cloud.connected:
            return
        try:
            self._cloud.run(self._cloud.bridge.set_night('night'))
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"[TEACHEN] Flash-LED AN Fehler: {e}")
        finally:
            try:
                self._cloud.run(self._cloud.bridge.set_night('day'))
            except Exception as e2:
                logger.error(f"[TEACHEN] Flash-LED AUS Fehler (LED koennte haengen!): {e2}")
        logger.info("[TEACHEN] Flash-LED Blitz")
