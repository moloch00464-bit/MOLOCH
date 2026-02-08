"""
M.O.L.O.C.H. 3.0 Brain Loader (Dormant)
=======================================

A brain loader that can detect and validate brains, but NEVER activates them
automatically.

Safety Rules:
- Loader requires explicit user command to activate
- Loader may only read, never write
- No execution on import
- No background threads
- No autonomous behavior

This file defines the loading mechanism. The actual brain remains dormant
until the user explicitly decides to awaken it.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from .interfaces import IBrainLoader, BrainState


class DormantBrainLoader(IBrainLoader):
    """
    Brain loader implementation that maintains dormancy.

    This loader can:
    - Discover available brain configurations
    - Validate brain files
    - Load brain metadata into memory

    This loader CANNOT:
    - Auto-activate brains
    - Modify any files
    - Start background processes
    - Execute brain logic
    """

    def __init__(self, moloch_root: str = None):
        """
        Initialize the loader.

        NOTE: Initialization does NOT load or activate anything.
        """
        self._moloch_root = moloch_root or os.path.expanduser("~/moloch")
        self._safeguards_path = os.path.join(self._moloch_root, "safeguards")
        self._legacy_manifest_path = os.path.join(self._safeguards_path, "legacy_manifest.json")
        self._discovered_brains: Dict[str, Any] = {}
        self._state = BrainState.DORMANT

    @property
    def state(self) -> BrainState:
        """Current loader state - always starts DORMANT."""
        return self._state

    def _load_legacy_manifest(self) -> Dict[str, Any]:
        """Load the legacy protection manifest (read-only)."""
        if os.path.exists(self._legacy_manifest_path):
            with open(self._legacy_manifest_path, 'r') as f:
                return json.load(f)
        return {}

    def _is_protected(self, path: str) -> bool:
        """Check if a path is protected by legacy manifest."""
        manifest = self._load_legacy_manifest()
        protected = manifest.get("protected_files", {})

        for category in protected.values():
            if isinstance(category, list):
                for item in category:
                    if isinstance(item, dict) and item.get("path") == path:
                        return True
                    elif isinstance(item, dict) and path.startswith(item.get("path", "")):
                        return True
        return False

    def discover(self) -> Dict[str, Any]:
        """
        Discover available brain configurations.

        Scans for:
        - Brain definition files in moloch/brain/
        - Context configurations in moloch/context/
        - Legacy adapters

        Returns metadata dictionary - does NOT load or activate.
        """
        discovered = {
            "timestamp": None,  # Will be set by caller if needed
            "brain_configs": [],
            "context_sources": [],
            "legacy_adapters": [],
            "state": "discovery_complete"
        }

        brain_dir = os.path.join(self._moloch_root, "brain")
        context_dir = os.path.join(self._moloch_root, "context")

        # Discover brain config files
        if os.path.exists(brain_dir):
            for f in os.listdir(brain_dir):
                if f.endswith(".json"):
                    discovered["brain_configs"].append({
                        "name": f,
                        "path": os.path.join(brain_dir, f),
                        "status": "discovered"
                    })

        # Discover context sources
        if os.path.exists(context_dir):
            index_path = os.path.join(context_dir, "index.json")
            if os.path.exists(index_path):
                discovered["context_sources"].append({
                    "name": "context_index",
                    "path": index_path,
                    "status": "discovered"
                })

            # Check origin fragments
            fragments_dir = os.path.join(context_dir, "origin_fragments")
            if os.path.exists(fragments_dir):
                for f in os.listdir(fragments_dir):
                    if f.endswith(".json"):
                        discovered["context_sources"].append({
                            "name": f.replace(".json", ""),
                            "path": os.path.join(fragments_dir, f),
                            "type": "origin_fragment",
                            "status": "discovered"
                        })

        self._discovered_brains = discovered
        return discovered

    def validate(self, brain_path: str) -> bool:
        """
        Validate a brain configuration file.

        Checks:
        - File exists and is readable
        - JSON is valid
        - Does not reference protected legacy files for writing
        - Contains required structure

        READ-ONLY operation - never modifies files.
        """
        if not os.path.exists(brain_path):
            return False

        try:
            with open(brain_path, 'r') as f:
                config = json.load(f)

            # Validate basic structure
            required_fields = ["meta", "type"]
            for field in required_fields:
                if field not in config:
                    return False

            # Check for unsafe operations (writing to protected files)
            if "write_targets" in config:
                for target in config["write_targets"]:
                    if self._is_protected(target):
                        return False  # Would violate legacy protection

            return True

        except (json.JSONDecodeError, IOError):
            return False

    def load(self, brain_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a brain configuration into memory.

        This loads the brain into LOADED state - NOT ACTIVE.
        The brain exists in memory but does not process or respond.

        Activation requires explicit user command via IBrain.activate().
        """
        if not self.validate(brain_path):
            return None

        try:
            with open(brain_path, 'r') as f:
                config = json.load(f)

            # Return loaded config with metadata
            loaded = {
                "config": config,
                "path": brain_path,
                "state": BrainState.LOADED.value,
                "activation_required": True,
                "activation_command": "User must explicitly call brain.activate(confirmation_string)"
            }

            self._state = BrainState.LOADED
            return loaded

        except (json.JSONDecodeError, IOError):
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current loader status."""
        return {
            "loader_state": self._state.value,
            "discovered_count": len(self._discovered_brains.get("brain_configs", [])),
            "moloch_root": self._moloch_root,
            "safeguards_active": os.path.exists(self._legacy_manifest_path),
            "auto_activation": False,  # Always False - by design
            "background_threads": False  # Always False - by design
        }


# Module-level instance is NOT created automatically
# User must explicitly instantiate: loader = DormantBrainLoader()
