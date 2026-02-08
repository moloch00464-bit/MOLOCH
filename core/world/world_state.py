"""
M.O.L.O.C.H. World State Manager

Manages the world inventory - the map of all possible and actual
peripherals, sensors, and interaction channels.

This module ONLY handles loading and saving the inventory.
NO sensor polling, NO device activation, NO autonomous behavior.

Principle: Know what exists. Don't act on it (yet).
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy


# Setup logging
LOG_DIR = Path.home() / "moloch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "world.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inventory file path
WORLD_DIR = Path(__file__).parent
STATE_DIR = WORLD_DIR / "state"
INVENTORY_FILE = STATE_DIR / "world_inventory.json"

# Valid slot statuses
SLOT_STATUS = {
    "available": "Device is present and ready",
    "placeholder": "Slot reserved for future device",
    "empty": "Unused slot",
    "offline": "Device configured but not responding",
    "error": "Device has encountered an error"
}


class WorldInventory:
    """
    Manages the world inventory - M.O.L.O.C.H.'s map of its peripherals.

    This is a passive data structure. It does not:
    - Poll devices
    - Activate hardware
    - Make decisions
    - Learn or adapt

    It only:
    - Loads the inventory from disk
    - Saves the inventory to disk
    - Provides read access to the inventory
    - Allows manual updates (by human or supervised code)
    """

    def __init__(self, inventory_file: Optional[Path] = None):
        self.inventory_file = inventory_file or INVENTORY_FILE
        self.inventory: Dict[str, Any] = {}
        self._loaded = False

        # Ensure state directory exists
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> bool:
        """
        Load the inventory from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.inventory_file.exists():
            logger.error(f"Inventory file not found: {self.inventory_file}")
            return False

        try:
            with open(self.inventory_file, 'r') as f:
                self.inventory = json.load(f)

            self._loaded = True
            logger.info(f"World inventory loaded from {self.inventory_file}")

            # Log summary
            categories = [k for k in self.inventory.keys() if k != "meta"]
            logger.info(f"Categories: {', '.join(categories)}")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in inventory file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load inventory: {e}")
            return False

    def save(self) -> bool:
        """
        Save the current inventory to disk.

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._loaded:
            logger.warning("Attempting to save inventory that was never loaded")

        try:
            # Update last_modified timestamp
            if "meta" in self.inventory:
                self.inventory["meta"]["last_modified"] = datetime.now().isoformat()

            with open(self.inventory_file, 'w') as f:
                json.dump(self.inventory, f, indent=2)

            logger.info(f"World inventory saved to {self.inventory_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save inventory: {e}")
            return False

    def get_categories(self) -> List[str]:
        """Get list of inventory categories."""
        return [k for k in self.inventory.keys() if k != "meta"]

    def get_category(self, category: str) -> Optional[Dict[str, Any]]:
        """Get a specific category from the inventory."""
        return self.inventory.get(category)

    def get_slot(self, category: str, slot_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific slot from a category."""
        cat = self.get_category(category)
        if cat and "slots" in cat:
            return cat["slots"].get(slot_id)
        return None

    def get_available_slots(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all available (active) slots in a category."""
        cat = self.get_category(category)
        if not cat or "slots" not in cat:
            return {}

        return {
            slot_id: slot_data
            for slot_id, slot_data in cat["slots"].items()
            if slot_data.get("status") == "available"
        }

    def get_placeholder_slots(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all placeholder slots in a category."""
        cat = self.get_category(category)
        if not cat or "slots" not in cat:
            return {}

        return {
            slot_id: slot_data
            for slot_id, slot_data in cat["slots"].items()
            if slot_data.get("status") == "placeholder"
        }

    def count_by_status(self) -> Dict[str, Dict[str, int]]:
        """Count slots by status for each category."""
        result = {}

        for category in self.get_categories():
            cat = self.get_category(category)
            if not cat or "slots" not in cat:
                continue

            counts = {"available": 0, "placeholder": 0, "empty": 0, "offline": 0, "error": 0}

            for slot_data in cat["slots"].values():
                status = slot_data.get("status", "empty")
                if status in counts:
                    counts[status] += 1

            result[category] = counts

        return result

    def update_slot(self, category: str, slot_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a slot with new data.

        Args:
            category: Category name
            slot_id: Slot ID (e.g., "slot_1")
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully
        """
        slot = self.get_slot(category, slot_id)
        if slot is None:
            logger.error(f"Slot not found: {category}/{slot_id}")
            return False

        # Apply updates
        for key, value in updates.items():
            slot[key] = value

        logger.info(f"Updated slot {category}/{slot_id}: {updates}")
        return True

    def set_slot_status(self, category: str, slot_id: str, status: str) -> bool:
        """
        Set the status of a slot.

        Args:
            category: Category name
            slot_id: Slot ID
            status: New status (must be valid)

        Returns:
            True if status was set
        """
        if status not in SLOT_STATUS:
            logger.error(f"Invalid status: {status}. Valid: {list(SLOT_STATUS.keys())}")
            return False

        return self.update_slot(category, slot_id, {"status": status})

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the world inventory."""
        if not self._loaded:
            return {"error": "Inventory not loaded"}

        summary = {
            "meta": self.inventory.get("meta", {}),
            "categories": {},
            "totals": {
                "available": 0,
                "placeholder": 0,
                "empty": 0
            }
        }

        counts = self.count_by_status()

        for category, status_counts in counts.items():
            cat_data = self.get_category(category)
            summary["categories"][category] = {
                "description": cat_data.get("description", ""),
                "max_slots": cat_data.get("max_slots", 0),
                "counts": status_counts
            }

            for status, count in status_counts.items():
                if status in summary["totals"]:
                    summary["totals"][status] += count

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Get the full inventory as a dictionary."""
        return deepcopy(self.inventory)


# Global instance
_world_inventory: Optional[WorldInventory] = None


def get_world_inventory() -> WorldInventory:
    """Get or create the global world inventory instance."""
    global _world_inventory
    if _world_inventory is None:
        _world_inventory = WorldInventory()
        _world_inventory.load()
    return _world_inventory


def load_inventory() -> bool:
    """Load the world inventory."""
    inventory = get_world_inventory()
    return inventory.load()


def save_inventory() -> bool:
    """Save the world inventory."""
    inventory = get_world_inventory()
    return inventory.save()


def get_summary() -> Dict[str, Any]:
    """Get inventory summary."""
    inventory = get_world_inventory()
    return inventory.get_summary()


if __name__ == "__main__":
    # Quick test
    print("M.O.L.O.C.H. World State Manager")
    print("=" * 60)

    inventory = get_world_inventory()

    if inventory._loaded:
        summary = inventory.get_summary()

        print(f"\nPhase: {summary['meta'].get('phase', '?')}")
        print(f"Version: {summary['meta'].get('version', '?')}")

        print(f"\nTotals:")
        print(f"  Available:   {summary['totals']['available']}")
        print(f"  Placeholder: {summary['totals']['placeholder']}")
        print(f"  Empty:       {summary['totals']['empty']}")

        print(f"\nCategories:")
        for cat_name, cat_data in summary["categories"].items():
            available = cat_data["counts"]["available"]
            placeholder = cat_data["counts"]["placeholder"]
            print(f"  {cat_name}: {available} available, {placeholder} placeholder")
    else:
        print("\nFailed to load inventory!")
