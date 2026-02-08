"""
M.O.L.O.C.H. Voice Selection Logic
===================================

PREPARATION ONLY - DECISION LOGIC PLACEHOLDER

This module contains the logic for selecting appropriate voices based on
contextual signals. The philosophy is:

1. Voices are RESOURCES, not personalities
2. Selection is DYNAMIC per utterance
3. Context drives the decision (time, load, tone, user preference)
4. No hardcoded mappings
5. Transparent reasoning (all decisions are logged)

Future Enhancement:
-------------------
This placeholder logic will be refined with:
- Machine learning for pattern recognition
- User feedback integration
- A/B testing of voice choices
- Adaptive selection based on effectiveness
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import random
import logging


logger = logging.getLogger(__name__)


class VoiceSelector:
    """
    Contextual voice selection engine for M.O.L.O.C.H.

    This class implements the decision logic for choosing appropriate
    voices based on current context signals.
    """

    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.weights = self._load_selection_weights()

    def _load_selection_weights(self) -> Dict[str, float]:
        """
        Load or initialize selection weights.

        These weights determine how much each context signal influences
        voice selection. They can be tuned over time.
        """
        # Default weights (can be overridden by config)
        return {
            "time_of_day": 0.3,
            "system_load": 0.2,
            "recent_interaction_tone": 0.25,
            "explicit_user_request": 1.0,  # Always override
            "diversity_bonus": 0.15,  # Encourage voice variety
            "continuity_bonus": 0.1,  # But maintain some consistency
        }

    def select(self, context: 'ContextSignals') -> 'VoiceSelection':
        """
        Select the most appropriate voice for given context.

        Args:
            context: Current context signals

        Returns:
            VoiceSelection with chosen voice and reasoning

        Note:
            This is DECISION LOGIC ONLY. No audio is generated.
        """
        from ..tts_manager import VoiceSelection

        # 1. Handle explicit user request (highest priority)
        if context.explicit_user_request:
            return self._handle_explicit_request(context)

        # 2. Load available voices
        voices = self._load_available_voices()

        if not voices:
            logger.warning("No voices configured. Using fallback.")
            return self._create_fallback_selection(context)

        # 3. Score each voice based on context
        scores = self._score_voices(voices, context)

        # 4. Select highest scoring voice
        best_voice_id = max(scores.items(), key=lambda x: x[1])[0]
        best_voice = next(v for v in voices if v["voice_id"] == best_voice_id)

        # 5. Build reasoning explanation
        reason = self._build_reasoning(best_voice, context, scores[best_voice_id])

        # 6. Create selection result
        selection = VoiceSelection(
            voice_id=best_voice_id,
            reason=reason,
            context_snapshot=self._snapshot_context(context),
            timestamp=datetime.now().isoformat(),
            confidence=scores[best_voice_id]
        )

        # 7. Log selection for analysis
        self._log_selection(selection)

        return selection

    def _handle_explicit_request(self, context: 'ContextSignals') -> 'VoiceSelection':
        """Handle user's explicit voice request."""
        from ..tts_manager import VoiceSelection

        voice_id = context.explicit_user_request
        logger.info(f"Explicit voice request: {voice_id}")

        return VoiceSelection(
            voice_id=voice_id,
            reason=f"Explicit user request for voice '{voice_id}'",
            context_snapshot=self._snapshot_context(context),
            timestamp=datetime.now().isoformat(),
            confidence=1.0
        )

    def _load_available_voices(self) -> List[Dict[str, Any]]:
        """Load available voices from configuration."""
        try:
            from ..config.voices import load_voice_config
            return load_voice_config()
        except Exception as e:
            logger.error(f"Failed to load voice config: {e}")
            return []

    def _score_voices(
        self,
        voices: List[Dict[str, Any]],
        context: 'ContextSignals'
    ) -> Dict[str, float]:
        """
        Score each voice based on context fit.

        This is the core selection algorithm. It evaluates how well
        each voice matches the current context.

        Scoring factors:
        1. Time of day alignment
        2. System load (prefer lighter voices under high load)
        3. Interaction tone matching
        4. Diversity (avoid repeating same voice too much)
        5. Continuity (maintain some consistency within session)
        """
        scores: Dict[str, float] = {}

        for voice in voices:
            voice_id = voice["voice_id"]
            score = 0.0

            # Factor 1: Time of day
            score += self._score_time_alignment(voice, context) * self.weights["time_of_day"]

            # Factor 2: System load
            score += self._score_load_suitability(voice, context) * self.weights["system_load"]

            # Factor 3: Tone matching
            score += self._score_tone_match(voice, context) * self.weights["recent_interaction_tone"]

            # Factor 4: Diversity bonus (avoid recent voices)
            score += self._score_diversity(voice_id, context) * self.weights["diversity_bonus"]

            # Factor 5: Continuity bonus (slight preference for last voice)
            score += self._score_continuity(voice_id, context) * self.weights["continuity_bonus"]

            # Normalize to 0-1 range
            scores[voice_id] = min(1.0, max(0.0, score))

        return scores

    def _score_time_alignment(
        self,
        voice: Dict[str, Any],
        context: 'ContextSignals'
    ) -> float:
        """
        Score how well voice matches time of day.

        Morning: Energetic, clear voices
        Afternoon: Neutral, professional voices
        Evening: Warm, relaxed voices
        Night: Calm, lower energy voices
        """
        preferred_contexts = voice.get("preferred_contexts", [])
        time_of_day = context.time_of_day

        if time_of_day in preferred_contexts:
            return 1.0

        # Partial matches based on energy level
        energy = voice.get("energy_level", "medium")

        if time_of_day == "morning" and energy == "high":
            return 0.8
        elif time_of_day == "night" and energy == "low":
            return 0.8
        elif time_of_day in ["afternoon", "evening"] and energy == "medium":
            return 0.7

        return 0.5  # Neutral if no clear match

    def _score_load_suitability(
        self,
        voice: Dict[str, Any],
        context: 'ContextSignals'
    ) -> float:
        """
        Score based on system load.

        Under high load, prefer computationally lighter voices.
        """
        system_load = context.system_load
        # Placeholder: would check actual voice file size / complexity
        # For now, use energy level as proxy (lower energy = lighter)

        energy = voice.get("energy_level", "medium")

        if system_load == "high":
            return 1.0 if energy == "low" else 0.5
        elif system_load == "medium":
            return 0.8 if energy == "medium" else 0.6
        else:  # low load
            return 1.0  # Any voice is fine

    def _score_tone_match(
        self,
        voice: Dict[str, Any],
        context: 'ContextSignals'
    ) -> float:
        """
        Score based on recent interaction tone.

        Match voice emotional range to conversation tone.
        """
        if not context.recent_interaction_tone:
            return 0.5  # Neutral

        tone = context.recent_interaction_tone
        emotional_range = voice.get("emotional_range", [])

        # Direct match
        if tone in emotional_range:
            return 1.0

        # Semantic similarity (simplified)
        tone_groups = {
            "formal": ["professional", "serious", "authoritative"],
            "casual": ["friendly", "relaxed", "conversational"],
            "urgent": ["intense", "energetic", "alert"],
            "empathetic": ["warm", "caring", "gentle"]
        }

        for group_key, group_values in tone_groups.items():
            if tone in group_values and any(er in group_values for er in emotional_range):
                return 0.7

        return 0.4  # Weak or no match

    def _score_diversity(self, voice_id: str, context: 'ContextSignals') -> float:
        """
        Bonus for voices not recently used.

        Encourages variety while maintaining contextual appropriateness.
        """
        if not context.last_voice_used:
            return 0.5  # Neutral

        if voice_id == context.last_voice_used:
            return 0.0  # No diversity bonus for same voice
        else:
            return 1.0  # Full bonus for different voice

    def _score_continuity(self, voice_id: str, context: 'ContextSignals') -> float:
        """
        Slight bonus for continuing with same voice.

        Balance diversity with consistency within session.
        """
        if not context.last_voice_used:
            return 0.5  # Neutral

        if voice_id == context.last_voice_used:
            return 1.0  # Continuity bonus
        else:
            return 0.0  # No continuity bonus

    def _build_reasoning(
        self,
        voice: Dict[str, Any],
        context: 'ContextSignals',
        score: float
    ) -> str:
        """
        Build human-readable explanation for voice selection.

        Transparency is critical for trust and debugging.
        """
        reasons = []

        # Primary factors
        if context.explicit_user_request:
            return f"User explicitly requested voice '{voice['voice_id']}'"

        if context.time_of_day in voice.get("preferred_contexts", []):
            reasons.append(f"optimal for {context.time_of_day}")

        if context.recent_interaction_tone:
            if context.recent_interaction_tone in voice.get("emotional_range", []):
                reasons.append(f"matches {context.recent_interaction_tone} tone")

        if context.system_load == "high" and voice.get("energy_level") == "low":
            reasons.append("lightweight for high system load")

        # Diversity
        if context.last_voice_used and voice["voice_id"] != context.last_voice_used:
            reasons.append("provides variety")
        elif context.last_voice_used and voice["voice_id"] == context.last_voice_used:
            reasons.append("maintains session continuity")

        if reasons:
            reason_text = ", ".join(reasons)
            return f"Selected '{voice['voice_id']}': {reason_text} (score: {score:.2f})"
        else:
            return f"Selected '{voice['voice_id']}' as best contextual match (score: {score:.2f})"

    def _snapshot_context(self, context: 'ContextSignals') -> Dict[str, Any]:
        """Create a serializable snapshot of context for logging."""
        return {
            "time_of_day": context.time_of_day,
            "system_load": context.system_load,
            "recent_interaction_tone": context.recent_interaction_tone,
            "session_duration": context.session_duration,
            "last_voice_used": context.last_voice_used,
        }

    def _create_fallback_selection(self, context: 'ContextSignals') -> 'VoiceSelection':
        """Create fallback selection when no voices configured."""
        from ..tts_manager import VoiceSelection

        return VoiceSelection(
            voice_id="default",
            reason="No voices configured, using fallback",
            context_snapshot=self._snapshot_context(context),
            timestamp=datetime.now().isoformat(),
            confidence=0.0
        )

    def _log_selection(self, selection: 'VoiceSelection') -> None:
        """
        Log voice selection for analysis and transparency.

        LOGGING POLICY:
        - Log voice_id, reason, and context snapshot
        - Do NOT log audio data
        - Do NOT log user speech content
        - Only selection metadata
        """
        self.selection_history.append({
            "voice_id": selection.voice_id,
            "reason": selection.reason,
            "context": selection.context_snapshot,
            "timestamp": selection.timestamp,
            "confidence": selection.confidence
        })

        logger.info(
            f"[VOICE_SELECTION] {selection.voice_id} | "
            f"Confidence: {selection.confidence:.2f} | "
            f"Reason: {selection.reason}"
        )

    def get_selection_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve selection history for analysis.

        Returns:
            List of past selections with metadata
        """
        return self.selection_history.copy()

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze selection patterns (future use).

        This method will provide insights into:
        - Most frequently selected voices
        - Context-voice correlations
        - Selection confidence trends
        - Potential improvements
        """
        if not self.selection_history:
            return {"message": "No selection history available"}

        voice_counts = {}
        total_confidence = 0.0

        for selection in self.selection_history:
            voice_id = selection["voice_id"]
            voice_counts[voice_id] = voice_counts.get(voice_id, 0) + 1
            total_confidence += selection["confidence"]

        avg_confidence = total_confidence / len(self.selection_history)

        return {
            "total_selections": len(self.selection_history),
            "unique_voices_used": len(voice_counts),
            "voice_distribution": voice_counts,
            "average_confidence": round(avg_confidence, 3),
            "most_common_voice": max(voice_counts.items(), key=lambda x: x[1])[0] if voice_counts else None
        }
