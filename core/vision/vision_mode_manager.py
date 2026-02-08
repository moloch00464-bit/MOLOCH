#!/usr/bin/env python3
"""
M.O.L.O.C.H. Vision Mode Manager
================================

Verwaltet die verschiedenen Vision-Modi:
- VISION_TRACKING: Nur Personenerkennung
- VISION_IDENTITY: + Gesichtserkennung
- VISION_FULL: + Landmarks + Hände

Author: M.O.L.O.C.H. System
Date: 2026-02-05
"""

import json
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Config path
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "vision_modes.json"


class VisionMode(Enum):
    """Verfügbare Vision-Modi."""
    VISION_TRACKING = "VISION_TRACKING"
    VISION_IDENTITY = "VISION_IDENTITY"
    VISION_FULL = "VISION_FULL"


@dataclass
class ModeConfig:
    """Konfiguration eines Vision-Modus."""
    name: str
    description: str
    models: List[str]
    identity_enabled: bool
    face_enabled: bool
    hands_enabled: bool


class VisionModeManager:
    """
    Verwaltet Vision-Modi und deren Umschaltung.

    Singleton - nur eine Instanz erlaubt.
    """

    _instance: Optional["VisionModeManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._config: Dict[str, Any] = {}
        self._current_mode: VisionMode = VisionMode.VISION_TRACKING
        self._mode_configs: Dict[str, ModeConfig] = {}
        self._callbacks: List[Callable[[VisionMode], None]] = []
        self._state_lock = threading.RLock()

        # Lade Konfiguration
        self._load_config()

        logger.info(f"[VISION_MODE] Manager initialisiert, aktiver Modus: {self._current_mode.value}")

    def _load_config(self):
        """Lade Konfiguration aus JSON."""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r') as f:
                    self._config = json.load(f)

                # Parse modes
                for name, cfg in self._config.get("modes", {}).items():
                    self._mode_configs[name] = ModeConfig(
                        name=name,
                        description=cfg.get("description", ""),
                        models=cfg.get("models", []),
                        identity_enabled=cfg.get("identity_enabled", False),
                        face_enabled=cfg.get("face_enabled", False),
                        hands_enabled=cfg.get("hands_enabled", False)
                    )

                # Setze aktiven Modus
                active = self._config.get("active_mode", "VISION_TRACKING")
                try:
                    self._current_mode = VisionMode(active)
                except ValueError:
                    logger.warning(f"[VISION_MODE] Unbekannter Modus '{active}', verwende VISION_TRACKING")
                    self._current_mode = VisionMode.VISION_TRACKING

                logger.info(f"[VISION_MODE] Konfiguration geladen: {len(self._mode_configs)} Modi")
            else:
                logger.warning(f"[VISION_MODE] Keine Konfiguration gefunden: {CONFIG_PATH}")
                self._create_default_config()

        except Exception as e:
            logger.error(f"[VISION_MODE] Fehler beim Laden der Konfiguration: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Erstelle Standard-Konfiguration."""
        self._mode_configs = {
            "VISION_TRACKING": ModeConfig(
                name="VISION_TRACKING",
                description="Person detection only",
                models=["person_detector"],
                identity_enabled=False,
                face_enabled=False,
                hands_enabled=False
            ),
            "VISION_IDENTITY": ModeConfig(
                name="VISION_IDENTITY",
                description="Person + face verification",
                models=["person_detector", "face_detector", "face_embedding"],
                identity_enabled=True,
                face_enabled=True,
                hands_enabled=False
            ),
            "VISION_FULL": ModeConfig(
                name="VISION_FULL",
                description="Person + face + landmarks + hands",
                models=["person_detector", "face_detector", "face_embedding",
                       "face_landmarks", "hand_detector", "hand_landmarks"],
                identity_enabled=True,
                face_enabled=True,
                hands_enabled=True
            )
        }

    def _save_config(self):
        """Speichere aktuelle Konfiguration."""
        try:
            self._config["active_mode"] = self._current_mode.value

            with open(CONFIG_PATH, 'w') as f:
                json.dump(self._config, f, indent=2)

            logger.info(f"[VISION_MODE] Konfiguration gespeichert")
        except Exception as e:
            logger.error(f"[VISION_MODE] Fehler beim Speichern: {e}")

    @property
    def current_mode(self) -> VisionMode:
        """Aktueller Vision-Modus."""
        with self._state_lock:
            return self._current_mode

    @property
    def current_config(self) -> Optional[ModeConfig]:
        """Konfiguration des aktuellen Modus."""
        with self._state_lock:
            return self._mode_configs.get(self._current_mode.value)

    @property
    def identity_enabled(self) -> bool:
        """Ist Identitätserkennung aktiviert?"""
        cfg = self.current_config
        return cfg.identity_enabled if cfg else False

    @property
    def face_enabled(self) -> bool:
        """Ist Gesichtserkennung aktiviert?"""
        cfg = self.current_config
        return cfg.face_enabled if cfg else False

    @property
    def hands_enabled(self) -> bool:
        """Ist Handerkennung aktiviert?"""
        cfg = self.current_config
        return cfg.hands_enabled if cfg else False

    def set_mode(self, mode: VisionMode) -> bool:
        """
        Setze neuen Vision-Modus.

        Args:
            mode: Neuer Modus

        Returns:
            True wenn erfolgreich gewechselt
        """
        with self._state_lock:
            if mode == self._current_mode:
                logger.debug(f"[VISION_MODE] Bereits im Modus {mode.value}")
                return True

            old_mode = self._current_mode
            self._current_mode = mode

            # Speichere Änderung
            self._save_config()

            logger.info(f"[VISION_MODE] Modus gewechselt: {old_mode.value} -> {mode.value}")

            # Benachrichtige Callbacks
            for callback in self._callbacks:
                try:
                    callback(mode)
                except Exception as e:
                    logger.error(f"[VISION_MODE] Callback-Fehler: {e}")

            return True

    def set_mode_by_name(self, mode_name: str) -> bool:
        """Setze Modus nach Name."""
        try:
            mode = VisionMode(mode_name)
            return self.set_mode(mode)
        except ValueError:
            logger.error(f"[VISION_MODE] Unbekannter Modus: {mode_name}")
            return False

    def register_callback(self, callback: Callable[[VisionMode], None]):
        """Registriere Callback für Moduswechsel."""
        with self._state_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[VisionMode], None]):
        """Entferne Callback."""
        with self._state_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_available_modes(self) -> List[str]:
        """Liste verfügbarer Modi."""
        return [m.value for m in VisionMode]

    def get_mode_info(self, mode_name: str = None) -> Dict[str, Any]:
        """
        Info über einen Modus.

        Args:
            mode_name: Modusname oder None für aktuellen
        """
        if mode_name is None:
            mode_name = self._current_mode.value

        cfg = self._mode_configs.get(mode_name)
        if not cfg:
            return {}

        return {
            "name": cfg.name,
            "description": cfg.description,
            "models": cfg.models,
            "identity_enabled": cfg.identity_enabled,
            "face_enabled": cfg.face_enabled,
            "hands_enabled": cfg.hands_enabled,
            "is_active": mode_name == self._current_mode.value
        }

    def get_status(self) -> Dict[str, Any]:
        """Status-Info für Debugging."""
        with self._state_lock:
            return {
                "current_mode": self._current_mode.value,
                "identity_enabled": self.identity_enabled,
                "face_enabled": self.face_enabled,
                "hands_enabled": self.hands_enabled,
                "available_modes": self.get_available_modes(),
                "callbacks_registered": len(self._callbacks)
            }


# Singleton-Getter
_manager_instance: Optional[VisionModeManager] = None


def get_vision_mode_manager() -> VisionModeManager:
    """Hole VisionModeManager Singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = VisionModeManager()
    return _manager_instance


# Convenience-Funktionen
def get_current_vision_mode() -> str:
    """Aktueller Modus als String."""
    return get_vision_mode_manager().current_mode.value


def set_vision_mode(mode_name: str) -> bool:
    """Setze Vision-Modus."""
    return get_vision_mode_manager().set_mode_by_name(mode_name)


def is_identity_enabled() -> bool:
    """Ist Identitätserkennung aktiviert?"""
    return get_vision_mode_manager().identity_enabled


def is_face_enabled() -> bool:
    """Ist Gesichtserkennung aktiviert?"""
    return get_vision_mode_manager().face_enabled


def is_hands_enabled() -> bool:
    """Ist Handerkennung aktiviert?"""
    return get_vision_mode_manager().hands_enabled


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    manager = get_vision_mode_manager()
    print(f"Status: {manager.get_status()}")

    print("\n--- Verfügbare Modi ---")
    for mode in manager.get_available_modes():
        info = manager.get_mode_info(mode)
        active = " (AKTIV)" if info.get("is_active") else ""
        print(f"  {mode}{active}: {info.get('description')}")

    print("\n--- Wechsel zu VISION_IDENTITY ---")
    manager.set_mode(VisionMode.VISION_IDENTITY)
    print(f"Neuer Status: {manager.get_status()}")

    print("\n--- Zurück zu VISION_TRACKING ---")
    manager.set_mode(VisionMode.VISION_TRACKING)
    print(f"Finaler Status: {manager.get_status()}")
