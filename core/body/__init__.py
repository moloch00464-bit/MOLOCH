"""
M.O.L.O.C.H. Koerper (Reachy Mini) - ENTWURF, nicht mit dem Service verbunden.

Konzept: docs/REACHY_MINI_KOERPER_KONZEPT.md

    expression.py  Gefuehl -> Zielpose (reine Funktion, kein I/O)
    controller.py  Prioritaeten, Not-Aus, Grenzen, Rate-Limit, Timeout
    backend.py     MotionBackend-Schnittstelle + MockMotionBackend

Es gibt bewusst noch KEIN Backend fuer den echten Roboter: SDK-Version,
Methodennamen und Grenzwerte sind nicht verifiziert (Phase 0 im Konzept).
"""

from .pose import BodyPose, clamp_pose, load_mappings
from .backend import MotionBackend, MockMotionBackend
from .expression import BodyState, compute_pose
from .controller import BodyController, Priority

__all__ = [
    "BodyPose", "clamp_pose", "load_mappings",
    "MotionBackend", "MockMotionBackend",
    "BodyState", "compute_pose",
    "BodyController", "Priority",
]
