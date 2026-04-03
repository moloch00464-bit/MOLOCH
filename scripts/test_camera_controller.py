import os
#!/usr/bin/env python3
"""
Unified Camera Controller Test Script
======================================

Comprehensive test script for UnifiedCameraController.

Tests:
- PTZ movement (pan, tilt, absolute, relative)
- Night mode toggle (shows not supported)
- LED level changes (shows not supported)
- Sleep/Privacy mode (shows not supported)
- Mic gain control (shows not supported)
- Status query
- Audio stream URL

All tests are local - no cloud connectivity.

Usage:
    python3 test_camera_controller.py              # Run all tests
    python3 test_camera_controller.py --quick      # Quick test (no delays)
    python3 test_camera_controller.py --ptz-only   # PTZ tests only
    python3 test_camera_controller.py --status     # Status query only
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera import (
    SonoffCameraController as UnifiedCameraController,
    NightMode,
    LEDLevel,
    ControlMode
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


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status:8s} | {test_name:40s} | {details}")


def test_connection(controller: UnifiedCameraController) -> bool:
    """Test camera connection."""
    print_section("TEST 1: Connection")

    try:
        success = controller.connect()
        if success:
            print_result("Connect to camera", True, f"Connected to {controller.camera_ip}")
            return True
        else:
            print_result("Connect to camera", False, "Connection failed")
            return False
    except Exception as e:
        print_result("Connect to camera", False, f"Exception: {e}")
        return False


def test_status_query(controller: UnifiedCameraController) -> bool:
    """Test status query."""
    print_section("TEST 2: Status Query")

    try:
        status = controller.get_status()

        # Check all fields
        tests = [
            ("Status.connected", status.connected, f"Connected: {status.connected}"),
            ("Status.model", status.model != "Unknown", f"Model: {status.model}"),
            ("Status.firmware", status.firmware != "Unknown", f"Firmware: {status.firmware}"),
            ("Status.position", status.position is not None,
             f"Pan={status.position.pan:.1f}°, Tilt={status.position.tilt:.1f}°" if status.position else "No position"),
            ("Status.pan_range", len(status.pan_range) == 2, f"Pan: {status.pan_range}"),
            ("Status.tilt_range", len(status.tilt_range) == 2, f"Tilt: {status.tilt_range}"),
            ("Status.rtsp_url", len(status.rtsp_url) > 0, f"RTSP URL available"),
        ]

        all_passed = True
        for test_name, passed, details in tests:
            print_result(test_name, passed, details)
            all_passed = all_passed and passed

        # Print feature support
        print("\n  Feature Support:")
        print(f"    PTZ Control:    {'✓ Supported' if status.ptz_available else '✗ Not Available'}")
        print(f"    Audio Stream:   {'✓ Supported' if status.audio_available else '✗ Not Available'}")
        print(f"    Night Mode:     {'✓ Supported' if status.night_mode_available else '✗ Not Available'}")
        print(f"    LED Control:    {'✓ Supported' if status.led_control_available else '✗ Not Available'}")
        print(f"    Sleep Mode:     {'✓ Supported' if status.sleep_mode_available else '✗ Not Available'}")
        print(f"    Mic Gain:       {'✓ Supported' if status.mic_gain_available else '✗ Not Available'}")

        # Print full status dict
        print("\n  Full Status Dictionary:")
        status_dict = status.to_dict()
        import json
        print(json.dumps(status_dict, indent=4))

        return all_passed

    except Exception as e:
        print_result("Status query", False, f"Exception: {e}")
        return False


def test_ptz_movement(controller: UnifiedCameraController, quick: bool = False) -> bool:
    """Test PTZ movement."""
    print_section("TEST 3: PTZ Movement")

    delay = 0.5 if quick else 2.0
    all_passed = True

    tests = [
        ("Move to home (0, 0)", lambda: controller.goto_home()),
        ("Move right (-30°)", lambda: controller.move_relative(-30.0, 0.0)),
        ("Move left (+60°)", lambda: controller.move_relative(60.0, 0.0)),
        ("Move up (+20°)", lambda: controller.move_relative(0.0, 20.0)),
        ("Move down (-40°)", lambda: controller.move_relative(0.0, -40.0)),
        ("Return to center", lambda: controller.goto_home()),
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n  {test_name}...")
            result = test_func()
            time.sleep(delay)

            # Get position
            pos = controller.get_position()
            details = f"Pan={pos.pan:+.1f}°, Tilt={pos.tilt:+.1f}°"

            print_result(test_name, result, details)
            all_passed = all_passed and result

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            all_passed = False

    return all_passed


def test_ptz_absolute(controller: UnifiedCameraController, quick: bool = False) -> bool:
    """Test absolute PTZ positioning."""
    print_section("TEST 4: Absolute Positioning")

    delay = 0.5 if quick else 2.0
    all_passed = True

    # Test positions
    positions = [
        ("Center (0, 0)", 0.0, 0.0),
        ("Far Right (-160°)", -160.0, 0.0),
        ("Far Left (+160°)", 160.0, 0.0),
        ("Look Up (0, +70°)", 0.0, 70.0),
        ("Look Down (0, -70°)", 0.0, -70.0),
        ("Back to Center", 0.0, 0.0),
    ]

    for test_name, pan, tilt in positions:
        try:
            print(f"\n  {test_name}...")
            result = controller.move_absolute(pan, tilt, speed=0.5)
            time.sleep(delay)

            # Verify position
            pos = controller.get_position()
            pan_error = abs(pos.pan - pan)
            tilt_error = abs(pos.tilt - tilt)

            # Allow 2 degree tolerance
            position_ok = pan_error < 2.0 and tilt_error < 2.0
            details = f"Target=({pan:.0f}°, {tilt:.0f}°), Actual=({pos.pan:+.1f}°, {pos.tilt:+.1f}°), Error=({pan_error:.1f}°, {tilt_error:.1f}°)"

            print_result(test_name, result and position_ok, details)
            all_passed = all_passed and result and position_ok

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            all_passed = False

    return all_passed


def test_night_mode(controller: UnifiedCameraController) -> bool:
    """Test night mode control (expected to fail - not supported)."""
    print_section("TEST 5: Night Mode Control")

    print("  NOTE: Night mode control is NOT supported on Sonoff GK-200MP2-B")
    print("        (No ONVIF Imaging Service)")
    print()

    tests = [
        ("Set night mode to AUTO", NightMode.AUTO),
        ("Set night mode to DAY", NightMode.DAY),
        ("Set night mode to NIGHT", NightMode.NIGHT),
        ("Query night mode", None),
    ]

    expected_results = []

    for test_name, mode in tests:
        try:
            if mode is None:
                # Query
                result = controller.get_night_mode()
                is_expected = (result is None)
                details = f"Returned None (expected)" if is_expected else f"Unexpected: {result}"
            else:
                # Set
                result = controller.set_night_mode(mode)
                is_expected = (result is False)
                details = f"Returned False (expected)" if is_expected else f"Unexpected: {result}"

            print_result(test_name, is_expected, details)
            expected_results.append(is_expected)

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            expected_results.append(False)

    # All should return expected (False/None)
    return all(expected_results)


def test_led_control(controller: UnifiedCameraController) -> bool:
    """Test LED control (expected to fail - not supported)."""
    print_section("TEST 6: LED Control")

    print("  NOTE: LED control is NOT supported on Sonoff GK-200MP2-B")
    print("        (No ONVIF Device I/O Service)")
    print()

    tests = [
        ("Set LED to OFF", LEDLevel.OFF),
        ("Set LED to LOW", LEDLevel.LOW),
        ("Set LED to MEDIUM", LEDLevel.MEDIUM),
        ("Set LED to HIGH", LEDLevel.HIGH),
        ("Query LED level", None),
    ]

    expected_results = []

    for test_name, level in tests:
        try:
            if level is None:
                # Query
                result = controller.get_led_level()
                is_expected = (result is None)
                details = f"Returned None (expected)" if is_expected else f"Unexpected: {result}"
            else:
                # Set
                result = controller.set_led_level(level)
                is_expected = (result is False)
                details = f"Returned False (expected)" if is_expected else f"Unexpected: {result}"

            print_result(test_name, is_expected, details)
            expected_results.append(is_expected)

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            expected_results.append(False)

    return all(expected_results)


def test_sleep_mode(controller: UnifiedCameraController) -> bool:
    """Test sleep/privacy mode (expected to fail - not supported)."""
    print_section("TEST 7: Sleep/Privacy Mode")

    print("  NOTE: Sleep/Privacy mode is NOT supported on Sonoff GK-200MP2-B")
    print("        (No ONVIF Privacy API)")
    print()

    tests = [
        ("Enable sleep mode", True),
        ("Disable sleep mode", False),
        ("Query sleep mode", None),
    ]

    expected_results = []

    for test_name, enabled in tests:
        try:
            if enabled is None:
                # Query
                result = controller.get_sleep_mode()
                is_expected = (result is None)
                details = f"Returned None (expected)" if is_expected else f"Unexpected: {result}"
            else:
                # Set
                result = controller.set_sleep_mode(enabled)
                is_expected = (result is False)
                details = f"Returned False (expected)" if is_expected else f"Unexpected: {result}"

            print_result(test_name, is_expected, details)
            expected_results.append(is_expected)

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            expected_results.append(False)

    return all(expected_results)


def test_mic_gain(controller: UnifiedCameraController) -> bool:
    """Test microphone gain control (expected to fail - not supported)."""
    print_section("TEST 8: Microphone Gain Control")

    print("  NOTE: Mic gain control is NOT supported on Sonoff GK-200MP2-B")
    print("        (No ONVIF Audio Input Configuration)")
    print()

    tests = [
        ("Set mic gain to 0%", 0.0),
        ("Set mic gain to 50%", 0.5),
        ("Set mic gain to 100%", 1.0),
        ("Query mic gain", None),
    ]

    expected_results = []

    for test_name, gain in tests:
        try:
            if gain is None:
                # Query
                result = controller.get_mic_gain()
                is_expected = (result is None)
                details = f"Returned None (expected)" if is_expected else f"Unexpected: {result}"
            else:
                # Set
                result = controller.set_mic_gain(gain)
                is_expected = (result is False)
                details = f"Returned False (expected)" if is_expected else f"Unexpected: {result}"

            print_result(test_name, is_expected, details)
            expected_results.append(is_expected)

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            expected_results.append(False)

    return all(expected_results)


def test_audio_stream(controller: UnifiedCameraController) -> bool:
    """Test audio stream URL retrieval."""
    print_section("TEST 9: Audio Stream")

    try:
        rtsp_url = controller.get_rtsp_url("main")
        audio_url = controller.get_audio_stream_url()

        # Verify URLs
        has_rtsp = "rtsp://" in rtsp_url
        has_credentials = controller.username in rtsp_url and controller.password in rtsp_url
        has_ip = controller.camera_ip in rtsp_url

        print_result("Get RTSP URL (main)", has_rtsp and has_credentials and has_ip, rtsp_url)
        print_result("Get audio stream URL", len(audio_url) > 0, audio_url)

        print(f"\n  Stream Details:")
        print(f"    Main Stream:  {controller.get_rtsp_url('main')}")
        print(f"    Sub Stream:   {controller.get_rtsp_url('sub')}")
        print(f"    Audio Codec:  G.711 A-law")
        print(f"    Sample Rate:  8000 Hz")
        print(f"    Channels:     Mono (1)")

        return has_rtsp and has_credentials and has_ip

    except Exception as e:
        print_result("Audio stream test", False, f"Exception: {e}")
        return False


def test_control_modes(controller: UnifiedCameraController) -> bool:
    """Test control mode switching."""
    print_section("TEST 10: Control Modes")

    modes = [
        ("Set AUTONOMOUS mode", ControlMode.AUTONOMOUS),
        ("Set MANUAL_OVERRIDE mode", ControlMode.MANUAL_OVERRIDE),
        ("Set SAFE_MODE", ControlMode.SAFE_MODE),
    ]

    all_passed = True

    for test_name, mode in modes:
        try:
            controller.set_mode(mode)
            is_correct = (controller.mode == mode)
            details = f"Current mode: {controller.mode.name}"

            print_result(test_name, is_correct, details)
            all_passed = all_passed and is_correct

        except Exception as e:
            print_result(test_name, False, f"Exception: {e}")
            all_passed = False

    # Reset to autonomous
    controller.set_mode(ControlMode.AUTONOMOUS)

    return all_passed


def run_all_tests(args):
    """Run all tests."""
    print_header("UNIFIED CAMERA CONTROLLER - TEST SUITE")
    print(f"Camera IP: {os.environ.get("MOLOCH_CAMERA_HOST", "CAMERA_IP")}")
    print(f"Quick Mode: {args.quick}")

    # Create controller
    controller = UnifiedCameraController()

    # Track results
    results = {}

    try:
        # Test 1: Connection
        if not test_connection(controller):
            print("\n✗ Connection failed - aborting tests")
            return False

        # Test 2: Status Query
        if not args.ptz_only:
            results['status'] = test_status_query(controller)

        # Test 3-4: PTZ Movement
        if not args.status:
            results['ptz_relative'] = test_ptz_movement(controller, args.quick)
            results['ptz_absolute'] = test_ptz_absolute(controller, args.quick)

        # Test 5-8: Unsupported features
        if not args.ptz_only and not args.status:
            results['night_mode'] = test_night_mode(controller)
            results['led_control'] = test_led_control(controller)
            results['sleep_mode'] = test_sleep_mode(controller)
            results['mic_gain'] = test_mic_gain(controller)

        # Test 9: Audio
        if not args.ptz_only:
            results['audio'] = test_audio_stream(controller)

        # Test 10: Control modes
        if not args.ptz_only:
            results['control_modes'] = test_control_modes(controller)

    finally:
        # Cleanup
        print_section("Cleanup")
        print("  Returning camera to home position...")
        controller.goto_home()
        time.sleep(1)
        print("  Disconnecting...")
        controller.disconnect()
        print("  ✓ Cleanup complete")

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
        description="Test Unified Camera Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--quick', action='store_true', help='Quick test mode (shorter delays)')
    parser.add_argument('--ptz-only', action='store_true', help='Test PTZ movement only')
    parser.add_argument('--status', action='store_true', help='Test status query only')

    args = parser.parse_args()

    # Run tests
    success = run_all_tests(args)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
