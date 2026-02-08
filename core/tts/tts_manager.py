"""
M.O.L.O.C.H. Text-to-Speech Manager
====================================

PREPARATION ONLY - NO RUNTIME ACTIVATION

This module defines the interface for M.O.L.O.C.H.'s multi-voice TTS system.
Voices are treated as contextual resources, not fixed personalities.

Philosophy:
-----------
- Voices are selected dynamically per utterance based on context
- No hardcoded personality-to-voice mappings
- Local-first: prefer Piper, fallback to Coqui-TTS
- Human-in-the-loop: all activation requires explicit user permission
- Logging only: voice selection decisions are logged, not audio

Future Activation:
------------------
This interface will be activated explicitly by Markus in a later phase.
Until then, this serves as architectural preparation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import logging


logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """Supported TTS engines (local-first priority)."""
    PIPER = "piper"
    COQUI = "coqui-tts"
    NONE = "none"  # Fallback when TTS is disabled


@dataclass
class ContextSignals:
    """
    Input signals that inform voice selection.

    These signals are gathered from the system state and recent interactions
    to help select an appropriate voice for the current utterance.
    """
    time_of_day: str  # "morning", "afternoon", "evening", "night"
    system_load: str  # "low", "medium", "high"
    recent_interaction_tone: Optional[str]  # "formal", "casual", "urgent", etc.
    explicit_user_request: Optional[str]  # User can request specific voice
    session_duration: Optional[int]  # Minutes in current session
    last_voice_used: Optional[str]  # For diversity/continuity balance


@dataclass
class VoiceSelection:
    """
    Result of voice selection process.

    Contains the chosen voice ID and the reasoning behind the selection
    for logging and transparency.
    """
    voice_id: str
    reason: str
    context_snapshot: Dict[str, Any]
    timestamp: str
    confidence: float  # 0.0 to 1.0


class TTSManagerInterface(ABC):
    """
    Abstract interface for M.O.L.O.C.H.'s TTS system.

    This interface is PREPARATION ONLY. Implementation and activation
    will occur in a future phase with explicit user permission.
    """

    @abstractmethod
    def initialize(self, engine: TTSEngine = TTSEngine.PIPER) -> bool:
        """
        Initialize the TTS engine (future use only).

        Args:
            engine: Preferred TTS engine to use

        Returns:
            bool: Success status

        Note:
            This method is a placeholder. Actual initialization will
            require explicit user permission and system checks.
        """
        pass

    @abstractmethod
    def select_voice(self, context: ContextSignals) -> VoiceSelection:
        """
        Select appropriate voice based on context signals.

        Args:
            context: Current context signals for decision making

        Returns:
            VoiceSelection: The selected voice and reasoning

        Note:
            This is a decision-making function only. No audio is generated.
            Selection is logged for transparency.
        """
        pass

    @abstractmethod
    def list_available_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voices with their metadata.

        Returns:
            List of voice metadata dictionaries

        Note:
            This queries voice configuration, does not download anything.
        """
        pass

    @abstractmethod
    def get_voice_metadata(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific voice.

        Args:
            voice_id: Unique identifier for the voice

        Returns:
            Voice metadata dict or None if not found
        """
        pass

    @abstractmethod
    def synthesize_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        context: Optional[ContextSignals] = None
    ) -> Optional[bytes]:
        """
        Synthesize text to speech (FUTURE USE ONLY).

        Args:
            text: Text to synthesize
            voice_id: Explicit voice ID (overrides context-based selection)
            context: Context for automatic voice selection

        Returns:
            Audio bytes or None if disabled

        CRITICAL:
            This method will NOT execute until explicitly activated.
            Currently returns None and logs the request only.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of TTS system."""
        pass


class PreparedTTSManager(TTSManagerInterface):
    """
    Prepared (non-executing) implementation of TTS Manager.

    This implementation provides logging and interface validation
    without actual TTS execution. It serves as:
    1. Architecture preparation
    2. Integration point for future activation
    3. Decision logging and transparency
    """

    def __init__(self):
        self.enabled = False  # Explicitly disabled until activation
        self.engine = TTSEngine.NONE
        logger.info("M.O.L.O.C.H. TTS Manager initialized (PREPARATION MODE)")

    def initialize(self, engine: TTSEngine = TTSEngine.PIPER) -> bool:
        """Log initialization request without execution."""
        logger.info(
            f"TTS initialization requested with engine: {engine.value}. "
            "Execution deferred until explicit activation by user."
        )
        return False  # Not actually initialized

    def select_voice(self, context: ContextSignals) -> VoiceSelection:
        """
        Perform voice selection logic and log the decision.

        This is the core decision-making function that will be used
        even before full TTS activation. It demonstrates how M.O.L.O.C.H.
        makes contextual voice choices.
        """
        from .selection.voice_selector import VoiceSelector

        selector = VoiceSelector()
        selection = selector.select(context)

        # Log the selection for transparency
        logger.info(
            f"Voice selected: {selection.voice_id} | "
            f"Reason: {selection.reason} | "
            f"Context: {selection.context_snapshot}"
        )

        return selection

    def list_available_voices(self) -> List[Dict[str, Any]]:
        """List voices from configuration."""
        from .config.voices import load_voice_config
        voices = load_voice_config()
        logger.debug(f"Listed {len(voices)} available voices")
        return voices

    def get_voice_metadata(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific voice metadata."""
        voices = self.list_available_voices()
        for voice in voices:
            if voice.get("voice_id") == voice_id:
                return voice
        return None

    def synthesize_text(
        self,
        text: str,
        voice_id: Optional[str] = None,
        context: Optional[ContextSignals] = None
    ) -> Optional[bytes]:
        """
        Log synthesis request without execution.

        CRITICAL: No audio is generated in preparation mode.
        """
        if voice_id:
            logger.info(
                f"TTS synthesis requested (explicit voice: {voice_id}). "
                f"Text length: {len(text)} chars. "
                "Execution deferred until activation."
            )
        elif context:
            selection = self.select_voice(context)
            logger.info(
                f"TTS synthesis requested (context-based: {selection.voice_id}). "
                f"Text length: {len(text)} chars. "
                "Execution deferred until activation."
            )
        else:
            logger.warning(
                "TTS synthesis requested without voice or context. "
                "Execution deferred until activation."
            )

        return None  # No audio generated

    def shutdown(self) -> None:
        """Log shutdown request."""
        logger.info("TTS Manager shutdown requested (preparation mode)")


# Factory function for future use
def create_tts_manager(enabled: bool = False) -> TTSManagerInterface:
    """
    Factory function to create appropriate TTS manager.

    Args:
        enabled: If True, creates active manager (requires permission check)
                 If False, creates prepared manager (logs only)

    Returns:
        TTSManagerInterface implementation

    Note:
        Currently always returns PreparedTTSManager regardless of enabled flag.
        Active manager will be implemented in future phase.
    """
    if enabled:
        logger.warning(
            "Active TTS manager requested but not yet implemented. "
            "Returning prepared manager. Activation requires explicit user permission."
        )

    return PreparedTTSManager()
