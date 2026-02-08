"""
M.O.L.O.C.H. 3.0 Legacy Brain Mount
====================================

Attaches M.O.L.O.C.H. 2.0 legacy brain to M.O.L.O.C.H. 3.0 infrastructure.

Safety Rules (STRICTLY ENFORCED):
- READ-ONLY access to all legacy data
- NO automatic activation
- NO background threads or processes
- NO modifications to protected files
- User must explicitly call /awake to activate

This brain is DORMANT by default. It can load and index legacy data,
but will not execute any decision logic until explicitly awakened.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from .interfaces import IBrain, IBrainContext, BrainState


class LegacyBrainContext(IBrainContext):
    """
    Read-only context provider for legacy M.O.L.O.C.H. 2.0 data.

    This class provides access to:
    - Legacy context files (immutable)
    - Origin fragments (conversation history references)
    - Current sensor state (runtime snapshot)

    ALL OPERATIONS ARE READ-ONLY.
    """

    def __init__(self, moloch_root: str = None):
        self._moloch_root = moloch_root or os.path.expanduser("~/moloch")
        self._context_cache = {}

    def get_legacy_context(self) -> Dict[str, Any]:
        """
        Read all legacy M.O.L.O.C.H. 2.0 context files.

        Returns aggregated context from:
        - moloch_context.json (project identity)
        - config/moloch_context.json (hardware config)
        - state/environment_state.json (device state)
        - core/world/state/world_inventory.json (peripherals)

        READ-ONLY - never modifies files.
        """
        legacy = {
            "identity": None,
            "hardware": None,
            "environment": None,
            "world": None,
            "load_timestamp": datetime.now().isoformat()
        }

        paths = {
            "identity": os.path.join(self._moloch_root, "moloch_context.json"),
            "hardware": os.path.join(self._moloch_root, "config", "moloch_context.json"),
            "environment": os.path.join(self._moloch_root, "state", "environment_state.json"),
            "world": os.path.join(self._moloch_root, "core", "world", "state", "world_inventory.json")
        }

        for key, path in paths.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        legacy[key] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    legacy[key] = {"error": str(e), "path": path}

        return legacy

    def get_origin_fragments(self) -> Dict[str, Any]:
        """
        Read origin fragment index.

        Returns metadata about available conversation histories
        (ChatGPT, Claude, Gemini) without loading actual content.
        """
        index_path = os.path.join(self._moloch_root, "context", "index.json")

        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
                return index.get("origin_fragments", {})
            except (json.JSONDecodeError, IOError):
                pass

        return {"status": "not_found", "path": index_path}

    def get_sensor_state(self) -> Dict[str, Any]:
        """
        Read current sensor state from runtime directory.

        This is TRANSIENT data from /runtime/state/, not legacy data.
        """
        runtime_state_dir = os.path.join(self._moloch_root, "runtime", "state")

        state = {
            "timestamp": datetime.now().isoformat(),
            "sensors": {}
        }

        if os.path.exists(runtime_state_dir):
            for f in os.listdir(runtime_state_dir):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(runtime_state_dir, f), 'r') as fh:
                            state["sensors"][f.replace(".json", "")] = json.load(fh)
                    except (json.JSONDecodeError, IOError):
                        pass

        return state


class LegacyBrain(IBrain):
    """
    M.O.L.O.C.H. Legacy Brain implementation.

    This brain represents the attachment of M.O.L.O.C.H. 2.0's identity
    and memory to the M.O.L.O.C.H. 3.0 infrastructure.

    DORMANT BY DEFAULT - requires explicit user activation via /awake command.

    Capabilities when DORMANT:
    - Load and index legacy context (read-only)
    - Report status
    - Await activation

    Capabilities when ACTIVE (after /awake):
    - Process sensor input
    - Generate responses
    - Utilize NPU for inference

    The brain NEVER auto-activates. User must explicitly confirm awakening.
    """

    # Confirmation string required to activate
    ACTIVATION_PHRASE = "AWAKEN_MOLOCH"

    def __init__(self, moloch_root: str = None):
        """
        Initialize brain in DORMANT state.

        NO automatic loading or activation occurs here.
        """
        self._moloch_root = moloch_root or os.path.expanduser("~/moloch")
        self._state = BrainState.DORMANT
        self._context: Optional[LegacyBrainContext] = None
        self._loaded_context: Optional[Dict[str, Any]] = None
        self._platform_context: Optional[Dict[str, Any]] = None
        self._activation_log: list = []

    @property
    def state(self) -> BrainState:
        """Current brain state."""
        return self._state

    def mount(self) -> bool:
        """
        Mount the legacy brain - loads context in read-only mode.

        This transitions from DORMANT to LOADED state.
        The brain is still not ACTIVE - it cannot process or respond.

        Returns True if mount successful.
        """
        if self._state != BrainState.DORMANT:
            return False

        try:
            # Create context accessor (read-only)
            self._context = LegacyBrainContext(self._moloch_root)

            # Load legacy context into memory (read-only snapshot)
            self._loaded_context = {
                "legacy": self._context.get_legacy_context(),
                "origin_fragments": self._context.get_origin_fragments(),
                "mount_timestamp": datetime.now().isoformat()
            }

            # Transition to LOADED state
            self._state = BrainState.LOADED
            self._log_transition("DORMANT -> LOADED", "mount() called, read-only context loaded")

            return True

        except Exception as e:
            self._state = BrainState.ERROR
            self._log_transition("DORMANT -> ERROR", f"Mount failed: {e}")
            return False

    def inject_platform_context(self, context_fragment: Dict[str, Any]) -> bool:
        """
        Inject new platform context (e.g., platform transition info).

        This DOES NOT modify legacy files - it adds runtime context
        that informs the brain about its new environment.

        Can only be called in LOADED state.
        """
        if self._state != BrainState.LOADED:
            return False

        self._platform_context = {
            "fragment": context_fragment,
            "injected_at": datetime.now().isoformat()
        }

        self._log_transition("CONTEXT_INJECT", f"Platform context injected: {context_fragment.get('event', 'unknown')}")
        return True

    def activate(self, user_confirmation: str) -> bool:
        """
        Activate the brain - requires explicit user confirmation.

        The confirmation string must match ACTIVATION_PHRASE exactly.
        This prevents accidental or unauthorized activation.

        Returns True if activation successful.
        """
        if self._state != BrainState.LOADED:
            self._log_transition("ACTIVATE_FAILED", "Brain not in LOADED state")
            return False

        if user_confirmation != self.ACTIVATION_PHRASE:
            self._log_transition("ACTIVATE_FAILED", f"Invalid confirmation phrase")
            return False

        # Activate the brain
        self._state = BrainState.ACTIVE
        self._log_transition("LOADED -> ACTIVE", "User confirmed activation with correct phrase")

        return True

    def deactivate(self) -> bool:
        """
        Deactivate the brain, returning to LOADED state.

        The brain remains in memory but stops processing.
        """
        if self._state != BrainState.ACTIVE:
            return False

        self._state = BrainState.LOADED
        self._log_transition("ACTIVE -> LOADED", "Brain deactivated by user")
        return True

    def suspend(self) -> bool:
        """
        Suspend the brain - temporary pause.

        Can be resumed without full reactivation.
        """
        if self._state != BrainState.ACTIVE:
            return False

        self._state = BrainState.SUSPENDED
        self._log_transition("ACTIVE -> SUSPENDED", "Brain suspended")
        return True

    def resume(self) -> bool:
        """
        Resume from SUSPENDED state.
        """
        if self._state != BrainState.SUSPENDED:
            return False

        self._state = BrainState.ACTIVE
        self._log_transition("SUSPENDED -> ACTIVE", "Brain resumed")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get current brain status.

        Safe to call in any state - no side effects.
        """
        status = {
            "state": self._state.value,
            "moloch_root": self._moloch_root,
            "context_loaded": self._loaded_context is not None,
            "platform_context_injected": self._platform_context is not None,
            "activation_phrase_hint": "AWAKEN_...",
            "can_activate": self._state == BrainState.LOADED,
            "transition_log_count": len(self._activation_log)
        }

        if self._loaded_context:
            status["legacy_sections"] = list(self._loaded_context.get("legacy", {}).keys())
            status["origin_fragments_available"] = bool(
                self._loaded_context.get("origin_fragments", {}).get("sources")
            )

        if self._platform_context:
            status["platform_event"] = self._platform_context.get("fragment", {}).get("event")

        return status

    def get_context_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of loaded context (only if mounted).

        Returns None if brain is not at least LOADED.
        """
        if self._state == BrainState.DORMANT:
            return None

        summary = {
            "state": self._state.value,
            "legacy_context": {},
            "platform_context": None
        }

        if self._loaded_context and self._loaded_context.get("legacy"):
            legacy = self._loaded_context["legacy"]

            # Identity summary (from moloch_context.json)
            if legacy.get("identity"):
                identity = legacy["identity"]
                summary["legacy_context"]["identity"] = {
                    "name": identity.get("project", "M.O.L.O.C.H."),
                    "phase": identity.get("phase"),
                    "host": identity.get("hardware", {}).get("host"),
                    "npu": identity.get("hardware", {}).get("npu", {}).get("model")
                }

            # Hardware summary (from config/moloch_context.json)
            if legacy.get("hardware"):
                hw = legacy["hardware"]
                hw_section = hw.get("hardware", {})
                summary["legacy_context"]["hardware"] = {
                    "platform": hw_section.get("platform", {}).get("name"),
                    "npu": hw_section.get("npu", {}).get("name"),
                    "storage": hw_section.get("storage", {}).get("name")
                }

            # World summary
            if legacy.get("world"):
                world = legacy["world"]
                summary["legacy_context"]["world_slots"] = list(world.get("io_slots", {}).keys())

        if self._platform_context:
            summary["platform_context"] = self._platform_context.get("fragment")

        return summary

    def _log_transition(self, transition: str, reason: str):
        """Log state transition for audit trail."""
        self._activation_log.append({
            "timestamp": datetime.now().isoformat(),
            "transition": transition,
            "reason": reason
        })

    def get_transition_log(self) -> list:
        """Get full transition log."""
        return self._activation_log.copy()


# Global brain instance - NOT CREATED AUTOMATICALLY
# Must be instantiated explicitly: brain = LegacyBrain()
_brain_instance: Optional[LegacyBrain] = None


def get_brain() -> Optional[LegacyBrain]:
    """Get the global brain instance (if created)."""
    return _brain_instance


def create_brain(moloch_root: str = None) -> LegacyBrain:
    """Create and return a new brain instance."""
    global _brain_instance
    _brain_instance = LegacyBrain(moloch_root)
    return _brain_instance
