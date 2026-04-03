#!/usr/bin/env python3
"""
M.O.L.O.C.H. Environment Watcher Test Script

Tests the environment detection and change monitoring.
"""

import sys
import time
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path.home() / "moloch" / "core"))

import environment_watcher as env_watcher


def print_separator(char="=", length=70):
    print(char * length)


def print_section(title):
    print_separator()
    print(f"  {title}")
    print_separator()


def display_current_state(watcher):
    """Display the current environment state."""
    state = watcher.get_current_state()

    if not state:
        print("No state available yet.")
        return

    print(f"\nTimestamp: {state.get('timestamp', 'unknown')}")

    sections = [
        ("dev_devices", "/dev Devices"),
        ("video_devices", "Video Devices"),
        ("audio_devices", "Audio Devices"),
        ("models", "Models"),
        ("hardware_files", "Hardware Files"),
        ("usb_devices", "USB Devices")
    ]

    for key, label in sections:
        items = state.get(key, [])
        print(f"\n{label} ({len(items)}):")
        if items:
            for item in sorted(items)[:10]:  # Show first 10
                print(f"  • {item}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
        else:
            print("  (none)")


def test_basic_check():
    """Test basic environment check."""
    print_section("TEST 1: Basic Environment Check")

    watcher = env_watcher.get_watcher()

    print("\nPerforming first check (establishing baseline)...")
    changes = watcher.check()

    if changes:
        print(f"\nDetected {len(changes)} change(s):")
        for change in changes:
            print(f"  [{change.change_type.upper()}] {change.category}: {change.details}")
    else:
        print("\n✓ Baseline established (no previous state to compare)")

    display_current_state(watcher)


def test_change_detection():
    """Test change detection with simulated changes."""
    print_section("TEST 2: Change Detection")

    watcher = env_watcher.get_watcher()

    print("\nWaiting 3 seconds...")
    print("(If you plug/unplug a USB device now, it will be detected)")
    time.sleep(3)

    print("\nPerforming second check...")
    changes = watcher.check()

    if changes:
        print(f"\n✓ Detected {len(changes)} change(s):")
        for change in changes:
            print(f"  [{change.change_type.upper()}] {change.category}: {change.details}")
    else:
        print("\n✓ No changes detected (environment stable)")


def test_state_persistence():
    """Test that state persists across runs."""
    print_section("TEST 3: State Persistence")

    state_file = Path.home() / "moloch" / "state" / "environment_state.json"

    if state_file.exists():
        print(f"\n✓ State file exists: {state_file}")
        print(f"  Size: {state_file.stat().st_size} bytes")

        # Read and display snippet
        import json
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
            print(f"  Tracked categories: {len([k for k, v in data.items() if k != 'timestamp' and v])}")
        except Exception as e:
            print(f"  ✗ Error reading state: {e}")
    else:
        print(f"\n✗ State file not found: {state_file}")


def test_force_baseline():
    """Test forcing a new baseline."""
    print_section("TEST 4: Force New Baseline")

    watcher = env_watcher.get_watcher()

    print("\nForcing new baseline (will ignore previous state)...")
    changes = watcher.force_baseline()

    print("✓ New baseline established")
    display_current_state(watcher)


def test_monitoring_loop():
    """Test continuous monitoring."""
    print_section("TEST 5: Continuous Monitoring (10 seconds)")

    watcher = env_watcher.get_watcher()

    print("\nMonitoring for 10 seconds (Ctrl+C to stop early)...")
    print("Try plugging/unplugging devices to see detection in action!\n")

    try:
        for i in range(10):
            print(f"[{i+1}/10] Checking...", end=" ")

            changes = watcher.check()

            if changes:
                print(f"CHANGES DETECTED ({len(changes)}):")
                for change in changes:
                    print(f"    • [{change.change_type.upper()}] {change.category}: {change.details}")
            else:
                print("No changes")

            time.sleep(1)

        print("\n✓ Monitoring test complete")

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring interrupted by user")


def show_log_file():
    """Show recent log entries."""
    print_section("Environment Watcher Logs")

    log_file = Path.home() / "moloch" / "logs" / "environment.log"

    if not log_file.exists():
        print(f"\nLog file not found: {log_file}")
        return

    print(f"\nLog file: {log_file}")
    print(f"Size: {log_file.stat().st_size} bytes\n")

    # Show last 20 lines
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        recent_lines = lines[-20:] if len(lines) > 20 else lines

        print("Recent entries:")
        print_separator("-")
        for line in recent_lines:
            print(line.rstrip())
        print_separator("-")

    except Exception as e:
        print(f"Error reading log: {e}")


def main():
    print_separator("=")
    print("  M.O.L.O.C.H. ENVIRONMENT WATCHER TEST SUITE")
    print_separator("=")

    tests = [
        ("Basic Check", test_basic_check),
        ("Change Detection", test_change_detection),
        ("State Persistence", test_state_persistence),
        ("Force Baseline", test_force_baseline),
        ("Continuous Monitoring", test_monitoring_loop),
        ("View Logs", show_log_file)
    ]

    # Run all tests
    for i, (name, test_func) in enumerate(tests, 1):
        try:
            test_func()
            print()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n✗ Test suite interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    print_separator("=")
    print("  TEST SUITE COMPLETE")
    print_separator("=")
    print(f"\nState file: {Path.home() / 'moloch' / 'state' / 'environment_state.json'}")
    print(f"Log file: {Path.home() / 'moloch' / 'logs' / 'environment.log'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
