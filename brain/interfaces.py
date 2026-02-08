"""
M.O.L.O.C.H. 3.0 Brain Interfaces
=================================

Abstract interfaces defining how a brain MAY interact with the system.
These interfaces are structural contracts - they define shape, not behavior.

NO IMPLEMENTATION - STRUCTURE ONLY
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum


class BrainState(Enum):
    """Possible states of a brain instance."""
    DORMANT = "dormant"          # Present but not loaded
    LOADED = "loaded"            # Loaded into memory, not active
    ACTIVE = "active"            # Processing enabled (requires explicit activation)
    SUSPENDED = "suspended"      # Temporarily paused
    ERROR = "error"              # Failed state


class IBrainLoader(ABC):
    """
    Interface for loading brain instances.

    Implementations must:
    - Never auto-activate on load
    - Never write to legacy files
    - Require explicit user command for activation
    """

    @abstractmethod
    def discover(self) -> Dict[str, Any]:
        """
        Discover available brain files/configurations.
        Returns metadata only - does not load or activate.
        """
        pass

    @abstractmethod
    def validate(self, brain_path: str) -> bool:
        """
        Validate a brain file/configuration without loading.
        Read-only operation.
        """
        pass

    @abstractmethod
    def load(self, brain_path: str) -> Optional[Any]:
        """
        Load a brain into memory (LOADED state).
        Does NOT activate - brain remains dormant until explicit activation.
        """
        pass


class IBrainContext(ABC):
    """
    Interface for brain context access.

    Provides read-only access to:
    - Legacy M.O.L.O.C.H. 2.0 memory
    - Origin fragments (ChatGPT, Claude, Gemini history)
    - Current sensor state
    """

    @abstractmethod
    def get_legacy_context(self) -> Dict[str, Any]:
        """
        Read legacy M.O.L.O.C.H. 2.0 context.
        READ-ONLY - must never modify legacy files.
        """
        pass

    @abstractmethod
    def get_origin_fragments(self) -> Dict[str, Any]:
        """
        Read origin fragment index.
        Returns metadata about available conversation histories.
        """
        pass

    @abstractmethod
    def get_sensor_state(self) -> Dict[str, Any]:
        """
        Read current sensor state snapshot.
        """
        pass


class IBrain(ABC):
    """
    Core brain interface.

    A brain implementation must:
    - Start in DORMANT state
    - Never auto-activate
    - Respect legacy protection rules
    - Log all state transitions
    """

    @property
    @abstractmethod
    def state(self) -> BrainState:
        """Current brain state."""
        pass

    @abstractmethod
    def activate(self, user_confirmation: str) -> bool:
        """
        Activate the brain.

        Requires explicit user confirmation string.
        Returns True if activation successful.

        This is the ONLY way to transition from LOADED to ACTIVE.
        """
        pass

    @abstractmethod
    def deactivate(self) -> bool:
        """
        Deactivate the brain, returning to LOADED state.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current brain status without side effects.
        """
        pass
