"""Bewegungs-Backend: Schnittstelle + Mock.

Ein Backend fuer den echten Reachy Mini fehlt absichtlich. Es wird diese
vier Methoden implementieren, sobald SDK-Version und API verifiziert sind.
"""

from typing import List, Protocol, Tuple

from .pose import BodyPose


class MotionBackend(Protocol):
    """Was der Controller von einem Koerper braucht - mehr nicht."""

    def connect(self) -> bool:
        """Verbindung aufbauen. False wenn der Koerper nicht erreichbar ist."""
        ...

    def is_connected(self) -> bool:
        ...

    def goto(self, pose: BodyPose) -> None:
        """Zielpose anfahren (nicht blockierend). Darf bei Fehler werfen."""
        ...

    def stop(self) -> None:
        """Sofort anhalten und Motoren entlasten (Not-Aus). Darf werfen."""
        ...


class MockMotionBackend:
    """Zeichnet alle Befehle auf. Fuer Tests und als Trockenlauf ohne Roboter."""

    def __init__(self, connected: bool = True):
        self.reachable = connected      # ob connect() gelingt
        self._connected = False
        self.fail_goto = False          # goto() wirft (Verbindungsabbruch)
        self.fail_stop = False
        self.log: List[Tuple[str, object]] = []
        self.connect_calls = 0

    def connect(self) -> bool:
        self.connect_calls += 1
        self._connected = bool(self.reachable)
        return self._connected

    def is_connected(self) -> bool:
        return self._connected

    def goto(self, pose: BodyPose) -> None:
        if self.fail_goto:
            self._connected = False
            raise ConnectionError("Mock: Verbindung verloren")
        self.log.append(("goto", pose))

    def stop(self) -> None:
        if self.fail_stop:
            raise ConnectionError("Mock: stop fehlgeschlagen")
        self.log.append(("stop", None))

    # Hilfen fuer Tests
    @property
    def poses(self) -> List[BodyPose]:
        return [p for kind, p in self.log if kind == "goto"]

    @property
    def stops(self) -> int:
        return sum(1 for kind, _ in self.log if kind == "stop")
