"""
M.O.L.O.C.H. 3.0 Legacy Guardian
================================

Validation functions for legacy protection.
Called by other modules before file operations.

This is NOT a daemon or background service.
It provides on-demand validation only.
"""

import json
import os
from datetime import datetime
from typing import Optional, Tuple


class LegacyGuardian:
    """
    Guardian that validates operations against legacy protection rules.

    Usage:
        guardian = LegacyGuardian()
        allowed, reason = guardian.can_write("/path/to/file")
        if not allowed:
            print(f"Blocked: {reason}")
    """

    def __init__(self, moloch_root: str = None):
        self._moloch_root = moloch_root or os.path.expanduser("~/moloch")
        self._manifest_path = os.path.join(
            self._moloch_root, "safeguards", "legacy_manifest.json"
        )
        self._violation_log_path = os.path.join(
            self._moloch_root, "safeguards", "violation_log.json"
        )
        self._manifest_cache = None

    def _load_manifest(self) -> dict:
        """Load legacy manifest (cached)."""
        if self._manifest_cache is None:
            if os.path.exists(self._manifest_path):
                with open(self._manifest_path, 'r') as f:
                    self._manifest_cache = json.load(f)
            else:
                self._manifest_cache = {"protected_files": {}}
        return self._manifest_cache

    def _get_protected_paths(self) -> list:
        """Extract all protected paths from manifest."""
        manifest = self._load_manifest()
        protected = []

        for category in manifest.get("protected_files", {}).values():
            if isinstance(category, list):
                for item in category:
                    if isinstance(item, dict):
                        path = item.get("path")
                        if path:
                            protected.append({
                                "path": path,
                                "protection": item.get("protection", "immutable")
                            })
        return protected

    def can_read(self, path: str) -> Tuple[bool, str]:
        """
        Check if reading a path is allowed.
        Reading is generally allowed for all files.
        """
        return True, "Read operations are allowed"

    def can_write(self, path: str) -> Tuple[bool, str]:
        """
        Check if writing to a path is allowed.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        abs_path = os.path.abspath(path)

        for protected in self._get_protected_paths():
            protected_path = protected["path"]
            protection_level = protected["protection"]

            # Check exact match
            if abs_path == protected_path:
                return False, f"Path is protected ({protection_level}): {protected_path}"

            # Check if path is inside protected directory
            if protected_path.endswith("/") and abs_path.startswith(protected_path):
                if protection_level == "immutable":
                    return False, f"Path is inside protected directory: {protected_path}"
                elif protection_level == "append-only":
                    # Append-only allows new files but not modification
                    if os.path.exists(abs_path):
                        return False, f"Cannot modify existing file in append-only directory: {protected_path}"

        return True, "Write allowed"

    def can_delete(self, path: str) -> Tuple[bool, str]:
        """
        Check if deleting a path is allowed.
        Deletion is blocked for all protected paths.
        """
        abs_path = os.path.abspath(path)

        for protected in self._get_protected_paths():
            protected_path = protected["path"]

            if abs_path == protected_path or abs_path.startswith(protected_path):
                return False, f"Cannot delete protected path: {protected_path}"

        return True, "Delete allowed"

    def log_violation(self, operation: str, path: str, reason: str) -> None:
        """Log a protection violation."""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "path": path,
            "reason": reason,
            "blocked": True
        }

        # Load existing log or create new
        violations = []
        if os.path.exists(self._violation_log_path):
            try:
                with open(self._violation_log_path, 'r') as f:
                    data = json.load(f)
                    violations = data.get("violations", [])
            except (json.JSONDecodeError, IOError):
                pass

        violations.append(violation)

        # Write updated log
        log_data = {
            "meta": {
                "purpose": "Legacy protection violation log",
                "created": datetime.now().isoformat()
            },
            "violations": violations
        }

        with open(self._violation_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_status(self) -> dict:
        """Get guardian status."""
        manifest = self._load_manifest()
        protected_count = len(self._get_protected_paths())

        violation_count = 0
        if os.path.exists(self._violation_log_path):
            try:
                with open(self._violation_log_path, 'r') as f:
                    data = json.load(f)
                    violation_count = len(data.get("violations", []))
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "guardian_active": True,
            "manifest_loaded": self._manifest_cache is not None,
            "protected_paths": protected_count,
            "violations_logged": violation_count,
            "background_monitoring": False  # By design - passive only
        }


# Module-level guardian is NOT created automatically
# Other modules must explicitly instantiate: guardian = LegacyGuardian()
