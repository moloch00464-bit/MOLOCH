#!/usr/bin/env python3
"""
Sonoff CAM-PT2 PTZ Controller fuer M.O.L.O.C.H.
Autonome Kamerasteuerung mit Sicherheitslimits.

Philosophie: Die Kamera ist kein Ueberwachungsgeraet,
sondern ein Aufmerksamkeits-Device das "lebendig" wirken soll.
"""

import json
import time
import logging
from datetime import datetime, time as dt_time
from enum import Enum
from typing import Optional, Tuple
from pathlib import Path
from collections import deque

# ONVIF wird spaeter importiert wenn verfuegbar
try:
    from onvif import ONVIFCamera
    ONVIF_AVAILABLE = True
except ImportError:
    ONVIF_AVAILABLE = False
    logging.warning("onvif-zeep nicht installiert: pip install onvif-zeep")

# WSDL Pfad fuer ONVIF
WSDL_PATH = "/home/molochzuhause/.local/lib/python3.13/site-packages/wsdl/"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PTZ] %(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)


class AutonomyMode(Enum):
    """Autonomie-Modi fuer Kamerasteuerung"""
    IDLE = "IDLE"
    FOLLOW_MARKUS = "FOLLOW_MARKUS"
    SEARCH_MODE = "SEARCH_MODE"
    IDLE_TRACKING = "IDLE_TRACKING"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class SonoffPTZController:
    """
    PTZ Controller fuer Sonoff CAM-PT2 via ONVIF.
    Implementiert autonome Tracking-Modi mit Sicherheitslimits.
    """

    # Kamera-Credentials (aus moloch_context.json)
    DEFAULT_IP = "192.168.178.25"
    DEFAULT_PORT = 80
    DEFAULT_USER = "Moloch_4.5"
    DEFAULT_PASS = "Auge666"

    # Sicherheitslimits
    MAX_MOVEMENTS_PER_MINUTE = 20
    MIN_MOVEMENT_INTERVAL = 2.0  # Sekunden zwischen Bewegungen
    NIGHT_START = dt_time(23, 0)
    NIGHT_END = dt_time(6, 0)
    MANUAL_OVERRIDE_COOLDOWN = 300  # 5 Minuten

    def __init__(self, ip: str = None, port: int = None,
                 user: str = None, password: str = None):
        self.ip = ip or self.DEFAULT_IP
        self.port = port or self.DEFAULT_PORT
        self.user = user or self.DEFAULT_USER
        self.password = password or self.DEFAULT_PASS

        # State
        self.camera: Optional[ONVIFCamera] = None
        self.ptz_service = None
        self.media_service = None
        self.profile_token = None

        self.current_mode = AutonomyMode.IDLE
        self.last_movement_time = 0.0
        self.movement_history = deque(maxlen=60)  # Letzte 60 Bewegungen
        self.manual_override_until = 0.0

        # Position tracking (relative, 0-1 range)
        self.current_pan = 0.5   # Mitte
        self.current_tilt = 0.5  # Mitte

        self.connected = False

    def connect(self) -> bool:
        """Verbinde mit Kamera via ONVIF"""
        if not ONVIF_AVAILABLE:
            log.error("ONVIF nicht verfuegbar - pip install onvif-zeep")
            return False

        try:
            log.info(f"Verbinde mit Sonoff CAM @ {self.ip}:{self.port}")
            self.camera = ONVIFCamera(
                self.ip, self.port, self.user, self.password, WSDL_PATH
            )

            # Services holen
            self.media_service = self.camera.create_media_service()
            self.ptz_service = self.camera.create_ptz_service()

            # Erstes Profil fuer PTZ nutzen
            profiles = self.media_service.GetProfiles()
            if profiles:
                self.profile_token = profiles[0].token
                log.info(f"ONVIF Profil: {self.profile_token}")

            self.connected = True
            log.info("Sonoff CAM verbunden!")
            return True

        except Exception as e:
            log.error(f"ONVIF Verbindung fehlgeschlagen: {e}")
            self.connected = False
            return False

    def _check_safety_limits(self) -> Tuple[bool, str]:
        """Pruefe ob Bewegung erlaubt ist"""
        now = time.time()
        current_time = datetime.now().time()

        # Nachtsperre
        if self._is_night_time(current_time):
            return False, "Nachtsperre aktiv (23:00-06:00)"

        # Manual Override aktiv?
        if now < self.manual_override_until:
            remaining = int(self.manual_override_until - now)
            return False, f"Manual Override Cooldown ({remaining}s)"

        # Bewegungslimit pro Minute
        recent_movements = [t for t in self.movement_history
                          if now - t < 60]
        if len(recent_movements) >= self.MAX_MOVEMENTS_PER_MINUTE:
            return False, f"Max {self.MAX_MOVEMENTS_PER_MINUTE} Bewegungen/min erreicht"

        # Mindestabstand zwischen Bewegungen
        if now - self.last_movement_time < self.MIN_MOVEMENT_INTERVAL:
            return False, "Mindestabstand 2s nicht erreicht"

        return True, "OK"

    def _is_night_time(self, t: dt_time) -> bool:
        """Pruefe ob Nachtzeit (23:00-06:00)"""
        return t >= self.NIGHT_START or t < self.NIGHT_END

    def _record_movement(self):
        """Bewegung in History aufzeichnen"""
        now = time.time()
        self.movement_history.append(now)
        self.last_movement_time = now

    def move(self, pan: float = 0.0, tilt: float = 0.0,
             speed: float = 0.5, force: bool = False) -> bool:
        """
        Bewege Kamera.

        Args:
            pan: -1.0 (links) bis 1.0 (rechts), 0 = keine Bewegung
            tilt: -1.0 (runter) bis 1.0 (hoch), 0 = keine Bewegung
            speed: 0.0-1.0 Geschwindigkeit
            force: Ignoriere Sicherheitslimits (nur fuer Manual Override!)

        Returns:
            True wenn Bewegung ausgefuehrt
        """
        if not self.connected:
            log.warning("Nicht verbunden - keine Bewegung moeglich")
            return False

        # Sicherheitscheck (ausser bei force)
        if not force:
            allowed, reason = self._check_safety_limits()
            if not allowed:
                log.debug(f"Bewegung blockiert: {reason}")
                return False

        try:
            # ONVIF ContinuousMove
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile_token
            request.Velocity = {
                'PanTilt': {'x': pan * speed, 'y': tilt * speed},
                'Zoom': {'x': 0}
            }

            self.ptz_service.ContinuousMove(request)

            # Kurz bewegen, dann stoppen
            time.sleep(0.3)
            self.stop()

            # Position tracken (geschaetzt)
            self.current_pan = max(0, min(1, self.current_pan + pan * 0.1))
            self.current_tilt = max(0, min(1, self.current_tilt + tilt * 0.1))

            self._record_movement()
            log.info(f"PTZ Move: pan={pan:.2f}, tilt={tilt:.2f}")
            return True

        except Exception as e:
            log.error(f"PTZ Bewegung fehlgeschlagen: {e}")
            return False

    def stop(self):
        """Stoppe alle Bewegungen"""
        if not self.connected:
            return

        try:
            self.ptz_service.Stop({'ProfileToken': self.profile_token})
        except Exception as e:
            log.error(f"PTZ Stop fehlgeschlagen: {e}")

    def center(self, force: bool = False) -> bool:
        """Fahre zur Mittelposition"""
        log.info("PTZ: Zentriere Kamera")
        # TODO: GotoHomePosition oder AbsoluteMove zur Mitte
        self.current_pan = 0.5
        self.current_tilt = 0.5
        return True

    def pan_left(self, speed: float = 0.5) -> bool:
        """Schwenke nach links"""
        return self.move(pan=-1.0, tilt=0, speed=speed)

    def pan_right(self, speed: float = 0.5) -> bool:
        """Schwenke nach rechts"""
        return self.move(pan=1.0, tilt=0, speed=speed)

    def tilt_up(self, speed: float = 0.5) -> bool:
        """Neige nach oben"""
        return self.move(pan=0, tilt=1.0, speed=speed)

    def tilt_down(self, speed: float = 0.5) -> bool:
        """Neige nach unten"""
        return self.move(pan=0, tilt=-1.0, speed=speed)

    # --- Autonomie-Modi ---

    def set_mode(self, mode: AutonomyMode):
        """Setze Autonomie-Modus"""
        old_mode = self.current_mode
        self.current_mode = mode
        log.info(f"Modus: {old_mode.value} -> {mode.value}")

    def trigger_manual_override(self):
        """Aktiviere Manual Override (stoppt Autonomie fuer 5 min)"""
        self.stop()
        self.manual_override_until = time.time() + self.MANUAL_OVERRIDE_COOLDOWN
        self.current_mode = AutonomyMode.MANUAL_OVERRIDE
        log.info(f"MANUAL OVERRIDE aktiviert fuer {self.MANUAL_OVERRIDE_COOLDOWN}s")

    def follow_face(self, face_x: float, face_y: float,
                    frame_width: int, frame_height: int) -> bool:
        """
        Folge einem Gesicht im Bild.

        Args:
            face_x, face_y: Gesichtsmitte in Pixeln
            frame_width, frame_height: Bildgroesse

        Returns:
            True wenn Bewegung ausgefuehrt
        """
        if self.current_mode == AutonomyMode.MANUAL_OVERRIDE:
            return False

        # Normalisiere Position (0-1)
        norm_x = face_x / frame_width
        norm_y = face_y / frame_height

        # Berechne Abweichung von Bildmitte
        offset_x = norm_x - 0.5  # -0.5 bis 0.5
        offset_y = 0.5 - norm_y  # Invertiert (oben = positiv)

        # Deadzone - kleine Abweichungen ignorieren
        DEADZONE = 0.15

        pan = 0.0
        tilt = 0.0

        if abs(offset_x) > DEADZONE:
            pan = offset_x * 2  # Skaliere auf -1 bis 1

        if abs(offset_y) > DEADZONE:
            tilt = offset_y * 2

        if pan == 0 and tilt == 0:
            return False  # Gesicht ist zentriert genug

        # Langsame, organische Bewegung
        return self.move(pan=pan, tilt=tilt, speed=0.3)

    def search_scan(self) -> bool:
        """
        Langsamer Scan von links nach rechts (SEARCH_MODE).
        Rufe wiederholt auf fuer kontinuierlichen Scan.
        """
        if self.current_mode != AutonomyMode.SEARCH_MODE:
            return False

        # Einfacher Links-Rechts Scan
        if self.current_pan < 0.3:
            return self.pan_right(speed=0.2)
        elif self.current_pan > 0.7:
            return self.pan_left(speed=0.2)
        else:
            # In der Mitte - zufaellige Richtung
            return self.pan_right(speed=0.2)



    # ===== M.O.L.O.C.H. Event-basiertes Tracking =====
    
    def move_pan_tilt(self, x_velocity: float, y_velocity: float, duration: float = 0.1) -> bool:
        """Bewegt Kamera mit Geschwindigkeit fuer bestimmte Dauer, dann Stop."""
        import time
        x_velocity = max(-1.0, min(1.0, x_velocity))
        y_velocity = max(-1.0, min(1.0, y_velocity))
        success = self.move(pan=x_velocity, tilt=y_velocity)
        if success:
            time.sleep(duration)
            self.stop()
        return success
    
    def track_target(self, target_x: int, frame_width: int = 1920, deadzone: int = 100) -> dict:
        """Berechnet PTZ-Korrektur (nur Berechnung, keine Bewegung!)."""
        frame_center = frame_width // 2
        offset = target_x - frame_center
        if abs(offset) < deadzone:
            return {'action': 'none', 'offset': offset, 'velocity': 0.0, 'should_move': False}
        velocity = min(abs(offset) / frame_center, 1.0) * 0.6
        action = 'right' if offset > 0 else 'left'
        return {'action': action, 'offset': offset, 'velocity': velocity, 'should_move': True}
    
    def execute_tracking(self, target_x: int, frame_width: int = 1920, deadzone: int = 100, duration: float = 0.1) -> dict:
        """track_target + move_pan_tilt kombiniert (Event-basiert)."""
        result = self.track_target(target_x, frame_width, deadzone)
        result['executed'] = False
        if result['should_move']:
            x_vel = result['velocity'] if result['action'] == 'right' else -result['velocity']
            if self.move_pan_tilt(x_vel, 0, duration):
                result['executed'] = True
                logger.info(f"Tracking: {result['action']} vel={result['velocity']:.2f} offset={result['offset']}px")
        return result


if __name__ == "__main__":
    import time
    print("=== Sonoff PTZ Controller Test ===")
    controller = SonoffPTZController()
    if controller.connect():
        print("Verbunden! Teste track_target...")
        r = controller.track_target(1500, 1920, 100)
        print(f"track_target(1500): {r}")
        print("Test OK")
    else:
        print("Verbindung fehlgeschlagen!")
