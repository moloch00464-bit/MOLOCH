"""
M.O.L.O.C.H. Text-to-Speech System
==================================

PREPARATION ONLY - NO RUNTIME ACTIVATION

Multi-voice TTS system with contextual voice selection.
Voices are treated as resources, not fixed personalities.

Usage:
    from core.tts import create_tts_manager, ContextSignals

    # Create manager (preparation mode)
    tts = create_tts_manager(enabled=False)

    # Test voice selection logic
    context = ContextSignals(
        time_of_day="morning",
        system_load="low",
        recent_interaction_tone="casual",
        explicit_user_request=None
    )

    selection = tts.select_voice(context)
    print(f"Voice: {selection.voice_id}")
    print(f"Reason: {selection.reason}")
"""

from .tts_manager import (
    TTSManagerInterface,
    PreparedTTSManager,
    create_tts_manager,
    ContextSignals,
    VoiceSelection,
    TTSEngine
)

__all__ = [
    "TTSManagerInterface",
    "PreparedTTSManager",
    "create_tts_manager",
    "ContextSignals",
    "VoiceSelection",
    "TTSEngine"
]

__version__ = "1.0.0"
__status__ = "PREPARATION"
