# M.O.L.O.C.H. Vision Module
"""
Vision Pipeline
===============

Components:
- UnifiedVisionPipeline: Single source of truth
- GstHailoDetector: GStreamer + Hailo-10H pose detection
- GstHailoPoseDetector: Pose detection with keypoint validation
- HailoAnalyzer: On-demand Hailo face recognition
- HybridVision: Orchestrator
- FaceDatabase: LanceDB face embedding storage

RECOMMENDED Usage:
    from core.vision import get_unified_pipeline

    pipeline = get_unified_pipeline()
    pipeline.start()

    # Get frame with overlay
    frame = pipeline.get_frame_with_overlay()

Legacy Usage:
    from core.vision import get_hybrid_vision
    hv = get_hybrid_vision()
"""

# Unified Pipeline (RECOMMENDED)
from .unified_pipeline import (
    UnifiedVisionPipeline,
    PipelineConfig,
    PipelineState,
    Detection,
    FrameResult,
    get_unified_pipeline
)

# Hailo Analyzers
from .hailo_analyzer import HailoAnalyzer, HailoState, get_hailo_analyzer, RecognitionResult
from .hybrid_vision import HybridVision, HybridVisionState, PersonEvent, get_hybrid_vision
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
    # Unified Pipeline (RECOMMENDED)
    "UnifiedVisionPipeline",
    "PipelineConfig",
    "PipelineState",
    "Detection",
    "FrameResult",
    "get_unified_pipeline",
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
    # Orchestrator
    "HybridVision",
    "HybridVisionState",
    "PersonEvent",
    "get_hybrid_vision",
    # Database
    "FaceDatabase",
    "KnownPerson",
    "SearchResult",
    "get_face_database",
]
