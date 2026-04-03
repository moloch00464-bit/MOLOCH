#!/usr/bin/env python3
"""
Cloud Bridge Integration Test
==============================

Tests the integration of CameraCloudBridge with UnifiedCameraController.

Tests:
- Cloud bridge initialization
- Feature availability detection
- Sleep mode control (via cloud)
- LED control (via cloud)
- Night mode control (via cloud)
- Mic gain control (via cloud)
- Graceful fallback when cloud unavailable
- Status query with cloud information

Usage:
    python3 test_cloud_bridge.py
    python3 test_cloud_bridge.py --enable-cloud  # Test with cloud enabled
"""

import sys
import time
import logging
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera import (
    SonoffCameraController as UnifiedCameraController,
    NightMode,
    LEDLevel
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print formatted section."""
    print(f"\n{title}")
    print("-" * 80)


def print_result(test_name: str, expected: bool, actual: bool, details: str = ""):
    """Print test result."""
    passed = (expected == actual)
    status = "✓ PASS" if passed else "✗ FAIL"
    exp_str = "Success" if expected else "Failure"
    act_str = "Success" if actual else "Failure"
    print(f"  {status:8s} | {test_name:40s} | Expected: {exp_str:7s}, Got: {act_str:7s} | {details}")
    return passed


def test_cloud_bridge_initialization(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test cloud bridge initialization."""
    print_section("TEST 1: Cloud Bridge Initialization")

    # Get status
    status = controller.get_status()

    # Check cloud fields
    tests = [
        ("Cloud enabled flag", cloud_enabled, status.cloud_enabled),
        ("Cloud connected", cloud_enabled, status.cloud_connected),
        ("Cloud status set", True, len(status.cloud_status) > 0),
    ]

    all_passed = True
    for test_name, expected, actual in tests:
        passed = print_result(test_name, expected, actual)
        all_passed = all_passed and passed

    print(f"\n  Cloud Status:")
    print(f"    Enabled:    {status.cloud_enabled}")
    print(f"    Connected:  {status.cloud_connected}")
    print(f"    Status:     {status.cloud_status}")

    return all_passed


def test_feature_availability(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test feature availability flags."""
    print_section("TEST 2: Feature Availability")

    status = controller.get_status()

    # Features should be available if cloud is enabled AND connected
    expected_available = cloud_enabled and status.cloud_connected

    tests = [
        ("Night mode available", expected_available, status.night_mode_available),
        ("LED control available", expected_available, status.led_control_available),
        ("Sleep mode available", expected_available, status.sleep_mode_available),
        ("Mic gain available", expected_available, status.mic_gain_available),
        ("PTZ available (ONVIF)", True, status.ptz_available),
        ("Audio available (ONVIF)", True, status.audio_available),
    ]

    all_passed = True
    for test_name, expected, actual in tests:
        passed = print_result(test_name, expected, actual)
        all_passed = all_passed and passed

    return all_passed


def test_sleep_mode_control(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test sleep/privacy mode control."""
    print_section("TEST 3: Sleep/Privacy Mode Control")

    expected_success = cloud_enabled

    # Test enable
    print("\n  Enabling sleep mode...")
    result_on = controller.set_sleep_mode(True)
    passed_on = print_result("Enable sleep mode", expected_success, result_on)

    time.sleep(0.5)

    # Test disable
    print("\n  Disabling sleep mode...")
    result_off = controller.set_sleep_mode(False)
    passed_off = print_result("Disable sleep mode", expected_success, result_off)

    return passed_on and passed_off


def test_led_control(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test LED brightness control."""
    print_section("TEST 4: LED Brightness Control")

    expected_success = cloud_enabled

    levels = [
        (LEDLevel.OFF, "OFF"),
        (LEDLevel.LOW, "LOW"),
        (LEDLevel.MEDIUM, "MEDIUM"),
        (LEDLevel.HIGH, "HIGH"),
    ]

    all_passed = True
    for level, name in levels:
        print(f"\n  Setting LED to {name}...")
        result = controller.set_led_level(level)
        passed = print_result(f"Set LED to {name}", expected_success, result)
        all_passed = all_passed and passed
        time.sleep(0.3)

    return all_passed


def test_night_mode_control(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test night/IR mode control."""
    print_section("TEST 5: Night/IR Mode Control")

    expected_success = cloud_enabled

    modes = [
        (NightMode.AUTO, "AUTO"),
        (NightMode.DAY, "DAY"),
        (NightMode.NIGHT, "NIGHT"),
    ]

    all_passed = True
    for mode, name in modes:
        print(f"\n  Setting night mode to {name}...")
        result = controller.set_night_mode(mode)
        passed = print_result(f"Set night mode to {name}", expected_success, result)
        all_passed = all_passed and passed
        time.sleep(0.3)

    return all_passed


def test_mic_gain_control(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test microphone gain control."""
    print_section("TEST 6: Microphone Gain Control")

    expected_success = cloud_enabled

    gain_levels = [
        (0.0, "0%"),
        (0.5, "50%"),
        (1.0, "100%"),
    ]

    all_passed = True
    for gain, name in gain_levels:
        print(f"\n  Setting mic gain to {name}...")
        result = controller.set_mic_gain(gain)
        passed = print_result(f"Set mic gain to {name}", expected_success, result)
        all_passed = all_passed and passed
        time.sleep(0.3)

    return all_passed


def test_graceful_fallback(controller: UnifiedCameraController, cloud_enabled: bool):
    """Test graceful fallback when cloud unavailable."""
    print_section("TEST 7: Graceful Fallback")

    print("\n  Testing that calls don't crash when cloud unavailable...")

    try:
        # All these should return False but not raise exceptions
        controller.set_night_mode(NightMode.AUTO)
        controller.set_led_level(LEDLevel.MEDIUM)
        controller.set_sleep_mode(True)
        controller.set_mic_gain(0.7)

        # Get status should still work
        status = controller.get_status()

        print_result("Graceful fallback", True, True, "No exceptions raised")
        return True

    except Exception as e:
        print_result("Graceful fallback", True, False, f"Exception: {e}")
        return False


def test_status_query(controller: UnifiedCameraController):
    """Test complete status query."""
    print_section("TEST 8: Status Query with Cloud Info")

    try:
        status = controller.get_status()
        status_dict = status.to_dict()

        print("\n  Full Status:")
        print(json.dumps(status_dict, indent=4))

        # Verify cloud section exists
        has_cloud_section = 'cloud' in status_dict
        print_result("Status has cloud section", True, has_cloud_section)

        return has_cloud_section

    except Exception as e:
        print_result("Status query", True, False, f"Exception: {e}")
        return False


def run_all_tests(cloud_enabled: bool):
    """Run all tests."""
    print_header("CLOUD BRIDGE INTEGRATION TEST")
    print(f"Cloud Enabled: {cloud_enabled}")

    # Modify config temporarily if needed
    config_path = Path(__file__).parent.parent / "config" / "camera_cloud.json"
    original_config = None

    if config_path.exists():
        with open(config_path, 'r') as f:
            original_config = json.load(f)

        # Update config
        modified_config = original_config.copy()
        modified_config['cloud_enabled'] = cloud_enabled

        with open(config_path, 'w') as f:
            json.dump(modified_config, f, indent=2)

        print(f"Config updated: cloud_enabled = {cloud_enabled}")
    else:
        print(f"Config not found: {config_path}")

    # Create controller
    controller = UnifiedCameraController()

    # Track results
    results = {}

    try:
        # Connect to camera
        print_section("Connecting to Camera")
        if not controller.connect():
            print("✗ Camera connection failed - some tests may not work")
        else:
            print("✓ Camera connected")

        # Run tests
        results['initialization'] = test_cloud_bridge_initialization(controller, cloud_enabled)
        results['feature_availability'] = test_feature_availability(controller, cloud_enabled)
        results['sleep_mode'] = test_sleep_mode_control(controller, cloud_enabled)
        results['led_control'] = test_led_control(controller, cloud_enabled)
        results['night_mode'] = test_night_mode_control(controller, cloud_enabled)
        results['mic_gain'] = test_mic_gain_control(controller, cloud_enabled)
        results['graceful_fallback'] = test_graceful_fallback(controller, cloud_enabled)
        results['status_query'] = test_status_query(controller)

    finally:
        # Cleanup
        print_section("Cleanup")
        controller.disconnect()
        print("✓ Disconnected")

        # Restore original config
        if original_config and config_path.exists():
            with open(config_path, 'w') as f:
                json.dump(original_config, f, indent=2)
            print("✓ Config restored")

    # Summary
    print_header("TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    print()

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} | {test_name}")

    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return passed_tests == total_tests


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Cloud Bridge Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--enable-cloud', action='store_true',
                       help='Test with cloud bridge enabled')

    args = parser.parse_args()

    # Run tests
    success = run_all_tests(cloud_enabled=args.enable_cloud)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
