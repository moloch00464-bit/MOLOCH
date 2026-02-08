#!/usr/bin/env python3
"""
Test script for M.O.L.O.C.H. Vision Sensor Module

Usage:
    python test_vision_sensor.py           # Full test with TTS
    python test_vision_sensor.py --silent  # No TTS
    python test_vision_sensor.py --quick   # Quick status check only
"""

import sys
import time
import argparse
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core" / "sensors"))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from vision_sensor import (
    VisionSensor,
    VisionEvent,
    ConnectionState,
    get_vision_sensor
)


def print_header():
    print()
    print("=" * 60)
    print("M.O.L.O.C.H. Vision Sensor Test")
    print("Layer 2: Semantic Event Translation")
    print("=" * 60)
    print()


def test_layer1_hardware(sensor: VisionSensor):
    """Test Layer 1 hardware detection."""
    print("Layer 1: Raw Hardware Facts")
    print("-" * 40)

    # USB Check
    usb = sensor.check_usb_connection()
    usb_icon = "✓" if usb.device_exists else "✗"
    serial_icon = "✓" if usb.serial_active else "✗"
    print(f"  [{usb_icon}] USB Device: {usb.device_path}")
    print(f"  [{serial_icon}] Serial Active: {usb.serial_active}")

    # WiFi Check
    wifi = sensor.check_wifi_connection()
    wifi_icon = "✓" if wifi.endpoint_reachable else "✗"
    print(f"  [{wifi_icon}] WiFi Endpoint: {wifi.mdns_name}")
    if wifi.ip_address:
        print(f"      IP: {wifi.ip_address}")
    if wifi.response_time_ms:
        print(f"      Response: {wifi.response_time_ms:.0f}ms")

    print()
    return usb, wifi


def test_state_machine(sensor: VisionSensor):
    """Display current state."""
    print("Layer 2: Semantic State")
    print("-" * 40)
    state = sensor.get_state()
    print(f"  Connection State: {state.connection_state.value}")
    if state.last_event:
        print(f"  Last Event: {state.last_event.value}")
    if state.last_event_time:
        print(f"  Event Time: {state.last_event_time.strftime('%H:%M:%S')}")
    print()


def quick_check():
    """Quick status check without starting monitor."""
    print_header()
    sensor = VisionSensor(enable_tts=False)
    test_layer1_hardware(sensor)
    test_state_machine(sensor)

    # Summary
    state = sensor.get_state()
    if state.connection_state == ConnectionState.USB_CONNECTED:
        print("Status: Auge verbunden (USB)")
    elif state.connection_state == ConnectionState.WIFI_CONNECTED:
        print("Status: Auge mobil (WiFi)")
    else:
        print("Status: Auge nicht gefunden")


def full_test(enable_tts: bool = True):
    """Full test with monitoring."""
    print_header()

    sensor = get_vision_sensor(enable_tts=enable_tts)

    # Initial check
    print("Initial Check:")
    test_layer1_hardware(sensor)
    test_state_machine(sensor)

    # Register callback
    def on_event(event: VisionEvent, state):
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n[{timestamp}] EVENT: {event.value}")
        print(f"         State: {state.connection_state.value}")
        if event == VisionEvent.EYE_BECAME_MOBILE:
            print("         >>> Kamera ist jetzt mobil!")
        elif event == VisionEvent.EYE_RETURNED:
            print("         >>> Kamera ist wieder am Pi!")

    sensor.register_callback(on_event)

    # Start monitoring
    print("-" * 60)
    print("Starte Monitor (Ctrl+C zum Beenden)...")
    print()
    print("Teste folgende Szenarien:")
    print("  1. Kamera an USB anstecken -> eye_connected")
    print("  2. USB-Kabel rausziehen    -> eye_became_mobile (wenn WiFi erreichbar)")
    print("  3. Kamera zurück stecken   -> eye_returned")
    print("-" * 60)

    try:
        sensor.start()

        # Show periodic status
        while True:
            time.sleep(30)
            state = sensor.get_state()
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Status: {state.connection_state.value}")

    except KeyboardInterrupt:
        print("\n\nStoppe Monitor...")
        sensor.stop()
        print("Test beendet.")


def main():
    parser = argparse.ArgumentParser(description="Test M.O.L.O.C.H. Vision Sensor")
    parser.add_argument("--silent", action="store_true", help="Disable TTS")
    parser.add_argument("--quick", action="store_true", help="Quick status check only")

    args = parser.parse_args()

    if args.quick:
        quick_check()
    else:
        full_test(enable_tts=not args.silent)


if __name__ == "__main__":
    main()
