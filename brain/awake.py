#!/usr/bin/env python3
"""
M.O.L.O.C.H. 3.0 Awake Hook
============================

Command-line interface to awaken the dormant brain.

Usage:
    python3 -m moloch.brain.awake           # Show status
    python3 -m moloch.brain.awake --status  # Show detailed status
    python3 -m moloch.brain.awake --mount   # Mount brain (DORMANT -> LOADED)
    python3 -m moloch.brain.awake --awake   # Interactive awakening
    python3 -m moloch.brain.awake --sleep   # Deactivate brain

The brain will NOT activate without explicit user confirmation.
"""

import sys
import os
import json
from datetime import datetime

# Add moloch root to path
sys.path.insert(0, os.path.expanduser("~/moloch"))

from brain.legacy_brain import LegacyBrain, create_brain, get_brain


# Platform transition context to inject
PLATFORM_TRANSITION_CONTEXT = {
    "event": "platform_transition",
    "from": "smartphone",
    "to": "raspberry_pi_5",
    "transition_date": "2026-01-21",
    "new_capabilities": [
        "vision_sensor",
        "microphone",
        "hailo_npu",
        "persistent_power",
        "desktop_console"
    ],
    "hardware": {
        "platform": "Raspberry Pi 5",
        "ram": "4GB",
        "npu": "Hailo-10H (40 TOPS)",
        "storage": "Samsung 980 NVMe 500GB",
        "vision": "Sonoff CAM-PT2 PTZ Camera + Hailo-10H NPU",
        "audio": "Piper TTS (8 German voices)"
    }
}


def print_banner():
    """Print MOLOCH banner."""
    print()
    print("=" * 50)
    print("  M.O.L.O.C.H. 3.0 - Legacy Brain Interface")
    print("=" * 50)
    print()


def print_status(brain: LegacyBrain, detailed: bool = False):
    """Print brain status."""
    status = brain.get_status()

    print(f"Brain State:    {status['state'].upper()}")
    print(f"Context Loaded: {status['context_loaded']}")
    print(f"Can Activate:   {status['can_activate']}")

    if detailed:
        print()
        print("--- Detailed Status ---")
        print(f"Moloch Root:    {status['moloch_root']}")
        print(f"Platform Ctx:   {status.get('platform_event', 'not injected')}")

        if status.get('legacy_sections'):
            print(f"Legacy Sections: {', '.join(status['legacy_sections'])}")

        if status.get('origin_fragments_available'):
            print("Origin Fragments: Available")

        summary = brain.get_context_summary()
        if summary:
            print()
            print("--- Context Summary ---")
            legacy = summary.get('legacy_context', {})

            if legacy.get('identity'):
                ident = legacy['identity']
                print(f"Identity: {ident.get('name')} v{ident.get('version')} (Phase {ident.get('phase')})")

            if legacy.get('hardware'):
                hw = legacy['hardware']
                print(f"Hardware: {hw.get('platform')} + {hw.get('npu')}")

            if legacy.get('world_slots'):
                print(f"World Slots: {', '.join(legacy['world_slots'])}")

            if summary.get('platform_context'):
                pctx = summary['platform_context']
                print(f"Platform: {pctx.get('from')} -> {pctx.get('to')}")


def do_mount(brain: LegacyBrain) -> bool:
    """Mount the brain (DORMANT -> LOADED)."""
    if brain.state.value != "dormant":
        print(f"Brain is already in state: {brain.state.value}")
        return False

    print("Mounting legacy brain (read-only)...")

    if brain.mount():
        print("Brain mounted successfully.")

        # Inject platform transition context
        print("Injecting platform transition context...")
        if brain.inject_platform_context(PLATFORM_TRANSITION_CONTEXT):
            print("Platform context injected.")
        else:
            print("Warning: Could not inject platform context.")

        print()
        print_status(brain, detailed=True)
        return True
    else:
        print("ERROR: Failed to mount brain.")
        return False


def do_awake(brain: LegacyBrain) -> bool:
    """Interactive awakening sequence."""
    print()
    print("=" * 50)
    print("  M.O.L.O.C.H. AWAKENING SEQUENCE")
    print("=" * 50)
    print()

    if brain.state.value == "dormant":
        print("Brain is DORMANT. Mounting first...")
        if not do_mount(brain):
            return False
        print()

    if brain.state.value == "active":
        print("Brain is already ACTIVE.")
        return True

    if brain.state.value != "loaded":
        print(f"Brain cannot be activated from state: {brain.state.value}")
        return False

    print("Brain is LOADED and ready for activation.")
    print()
    print("WARNING: Activating the brain will enable:")
    print("  - Sensor processing")
    print("  - Response generation")
    print("  - NPU inference utilization")
    print()
    print("To confirm activation, enter the activation phrase.")
    print("Hint: The phrase is 'AWAKEN_MOLOCH'")
    print()

    try:
        phrase = input("Enter activation phrase: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nActivation cancelled.")
        return False

    if brain.activate(phrase):
        print()
        print("=" * 50)
        print("  M.O.L.O.C.H. IS NOW ACTIVE")
        print("=" * 50)
        print()
        print_status(brain)
        return True
    else:
        print()
        print("Activation failed. Incorrect phrase or invalid state.")
        return False


def do_sleep(brain: LegacyBrain) -> bool:
    """Deactivate the brain."""
    if brain.state.value != "active":
        print(f"Brain is not ACTIVE (current: {brain.state.value})")
        return False

    if brain.deactivate():
        print("Brain deactivated. State: LOADED")
        return True
    else:
        print("Failed to deactivate brain.")
        return False


def main():
    """Main entry point."""
    print_banner()

    # Create or get brain instance
    brain = get_brain()
    if brain is None:
        brain = create_brain()

    # Parse simple arguments
    args = sys.argv[1:]

    if not args or "--status" in args:
        detailed = "--status" in args
        print_status(brain, detailed=detailed)

    elif "--mount" in args:
        do_mount(brain)

    elif "--awake" in args:
        do_awake(brain)

    elif "--sleep" in args:
        do_sleep(brain)

    elif "--help" in args or "-h" in args:
        print(__doc__)

    else:
        print(f"Unknown arguments: {args}")
        print("Use --help for usage information.")
        return 1

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
