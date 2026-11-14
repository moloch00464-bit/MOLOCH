# M.O.L.O.C.H. Vision Module
"""
Vision-Hilfsmodule (Legacy-Paket)
=================================

Die Live-Pipeline liegt in core/perception/ (TAPPAS + Stage-2-Worker).
Hier bleiben nur noch:
- HailoAnalyzer: On-demand Hailo-Gesichtserkennung
- FaceDatabase: Face-Embedding-Speicher
- Detektor-Module (gesture_detector, hand_gesture_detector, face_attr_npu, ...)

unified_pipeline und hybrid_vision wurden am 2026-03-30 geloescht (Commit
642f454). Ihre Importe standen hier noch und liessen JEDEN Import aus
core.vision scheitern - deshalb entfernt.
"""

# Hailo Analyzers
from .hailo_analyzer import HailoAnalyzer, HailoState, get_hailo_analyzer, RecognitionResult
from .face_database import FaceDatabase, KnownPerson, SearchResult, get_face_database

# Legacy compatibility stubs
def get_vision():
    """Legacy stub - returns None."""
    return None

def get_xiao_trigger():
    """Legacy stub - returns None."""
    return None

class XiaoVision:
    """Legacy stub class."""
    connected = False
    connection_mode = "none"
    def get_inference(self, with_image=False):
        return {}

class XiaoTrigger:
    """Legacy stub class."""
    pass

__all__ = [
    # Legacy stubs
    "XiaoVision",
    "get_vision",
    "XiaoTrigger",
    "get_xiao_trigger",
    # Hailo
    "HailoAnalyzer",
    "HailoState",
    "get_hailo_analyzer",
    "RecognitionResult",
    # Database
    "FaceDatabase",
    "KnownPerson",
    "SearchResult",
    "get_face_database",
]
