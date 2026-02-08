"""
M.O.L.O.C.H. 3.0 Brain Module
=============================

This module provides the structural framework for future intelligence.

IMPORTANT: This module is DORMANT by design.
- No automatic execution on import
- No background threads
- No autonomous behavior
- Brain activation requires explicit user command

The brain exists as a potential, not an active process.

To interact with the brain:
    from brain.legacy_brain import create_brain
    brain = create_brain()
    brain.mount()  # DORMANT -> LOADED
    brain.activate("AWAKEN_MOLOCH")  # LOADED -> ACTIVE

Or via command line:
    python3 -m brain.awake --mount   # Mount brain
    python3 -m brain.awake --awake   # Interactive awakening
"""

__version__ = "3.0.0"
__status__ = "dormant"

# Exports (no automatic instantiation)
from .interfaces import BrainState, IBrain, IBrainLoader, IBrainContext
from .loader import DormantBrainLoader
from .legacy_brain import LegacyBrain, create_brain, get_brain

__all__ = [
    "BrainState",
    "IBrain",
    "IBrainLoader",
    "IBrainContext",
    "DormantBrainLoader",
    "LegacyBrain",
    "create_brain",
    "get_brain"
]
