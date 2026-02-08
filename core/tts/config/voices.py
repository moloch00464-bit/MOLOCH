"""
M.O.L.O.C.H. Voice Configuration Loader
========================================

This module provides utilities for loading and managing voice configurations
from the voices.json file.

Philosophy:
-----------
- Voices are external resources, not hardcoded
- Configuration is declarative and versioned
- Easy to add/remove voices without code changes
- Metadata-driven selection logic
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """
    Get the path to voices.json configuration file.

    Returns:
        Path to voices.json
    """
    current_dir = Path(__file__).parent
    return current_dir / "voices.json"


def load_voice_config() -> List[Dict[str, Any]]:
    """
    Load voice configuration from voices.json.

    Returns:
        List of voice metadata dictionaries

    Raises:
        FileNotFoundError: If voices.json doesn't exist
        json.JSONDecodeError: If voices.json is malformed
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.error(f"Voice configuration not found at {config_path}")
        return []

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        voices = config_data.get("voices", [])
        logger.info(f"Loaded {len(voices)} voice configurations")

        # Validate voices
        validated_voices = []
        for voice in voices:
            if validate_voice_metadata(voice):
                validated_voices.append(voice)
            else:
                logger.warning(f"Skipping invalid voice: {voice.get('voice_id', 'unknown')}")

        return validated_voices

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse voices.json: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading voice config: {e}")
        return []


def validate_voice_metadata(voice: Dict[str, Any]) -> bool:
    """
    Validate that voice metadata has all required fields.

    Args:
        voice: Voice metadata dictionary

    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = [
        "voice_id",
        "display_name",
        "engine",
        "style_tags",
        "energy_level",
        "emotional_range",
        "preferred_contexts",
        "language"
    ]

    for field in required_fields:
        if field not in voice:
            logger.warning(f"Voice missing required field: {field}")
            return False

    # Validate energy_level
    if voice["energy_level"] not in ["low", "medium", "high"]:
        logger.warning(f"Invalid energy_level: {voice['energy_level']}")
        return False

    # Validate that lists are actually lists
    for list_field in ["style_tags", "emotional_range", "preferred_contexts"]:
        if not isinstance(voice[list_field], list):
            logger.warning(f"Field {list_field} must be a list")
            return False

    return True


def get_voice_by_id(voice_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific voice by its ID.

    Args:
        voice_id: Unique voice identifier

    Returns:
        Voice metadata dict or None if not found
    """
    voices = load_voice_config()
    for voice in voices:
        if voice["voice_id"] == voice_id:
            return voice
    return None


def filter_voices_by_criteria(
    engine: Optional[str] = None,
    energy_level: Optional[str] = None,
    language: Optional[str] = None,
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filter voices by specific criteria.

    Args:
        engine: Filter by TTS engine (e.g., "piper")
        energy_level: Filter by energy level ("low", "medium", "high")
        language: Filter by language code (e.g., "en-US")
        context: Filter by preferred context (e.g., "morning", "technical")

    Returns:
        List of matching voice metadata dictionaries
    """
    voices = load_voice_config()
    filtered = voices

    if engine:
        filtered = [v for v in filtered if v.get("engine") == engine]

    if energy_level:
        filtered = [v for v in filtered if v.get("energy_level") == energy_level]

    if language:
        filtered = [v for v in filtered if v.get("language") == language]

    if context:
        filtered = [
            v for v in filtered
            if context in v.get("preferred_contexts", [])
        ]

    logger.debug(
        f"Filtered voices: {len(filtered)} matches "
        f"(engine={engine}, energy={energy_level}, lang={language}, context={context})"
    )

    return filtered


def list_available_voices(detail: bool = False) -> str:
    """
    Get a human-readable list of available voices.

    Args:
        detail: If True, include full metadata. If False, brief summary.

    Returns:
        Formatted string listing voices
    """
    voices = load_voice_config()

    if not voices:
        return "No voices configured."

    lines = [f"Available voices ({len(voices)} total):", ""]

    for voice in voices:
        voice_id = voice["voice_id"]
        display_name = voice["display_name"]
        energy = voice["energy_level"]
        contexts = ", ".join(voice["preferred_contexts"][:3])

        if detail:
            lines.append(f"  {voice_id} ({display_name})")
            lines.append(f"    Energy: {energy}")
            lines.append(f"    Contexts: {contexts}")
            lines.append(f"    Description: {voice.get('description', 'N/A')}")
            lines.append("")
        else:
            lines.append(f"  {voice_id} - {display_name} ({energy} energy, {contexts})")

    return "\n".join(lines)


def get_config_metadata() -> Dict[str, Any]:
    """
    Get metadata about the voice configuration itself.

    Returns:
        Configuration metadata (version, last_updated, notes, etc.)
    """
    config_path = get_config_path()

    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data.get("metadata", {})
    except Exception as e:
        logger.error(f"Failed to load config metadata: {e}")
        return {}


def add_custom_voice(voice_metadata: Dict[str, Any]) -> bool:
    """
    Add a custom voice to the configuration (future use).

    Args:
        voice_metadata: Complete voice metadata dictionary

    Returns:
        bool: Success status

    Note:
        This is a placeholder for future functionality.
        Requires write permissions and validation.
    """
    logger.warning(
        "add_custom_voice() is not yet implemented. "
        "Manual editing of voices.json required."
    )
    return False


# Convenience function for quick testing
def demo_voice_listing():
    """
    Demo function to show voice configuration loading.

    This can be called to verify the configuration is working.
    """
    print("=" * 60)
    print("M.O.L.O.C.H. Voice Configuration Demo")
    print("=" * 60)
    print()

    # Load all voices
    print(list_available_voices(detail=True))
    print()

    # Show metadata
    metadata = get_config_metadata()
    if metadata:
        print("Configuration Metadata:")
        print(f"  Version: {metadata.get('version', 'N/A')}")
        print(f"  Last Updated: {metadata.get('last_updated', 'N/A')}")
        print()
        if "notes" in metadata:
            print("Notes:")
            for note in metadata["notes"]:
                print(f"  - {note}")
        print()

    # Show some filters
    print("High-energy voices:")
    high_energy = filter_voices_by_criteria(energy_level="high")
    for voice in high_energy:
        print(f"  - {voice['voice_id']}")
    print()

    print("Morning-optimized voices:")
    morning = filter_voices_by_criteria(context="morning")
    for voice in morning:
        print(f"  - {voice['voice_id']}")
    print()

    print("=" * 60)


if __name__ == "__main__":
    # Allow direct execution for testing
    demo_voice_listing()
