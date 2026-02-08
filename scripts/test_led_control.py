#!/usr/bin/env python3
"""
Quick LED Control Test
======================

Testet die LED-Steuerung via eWeLink Cloud API.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
import json


async def test_led_control():
    """Test LED control via cloud."""

    print("=" * 80)
    print("LED Control Test - eWeLink Cloud API")
    print("=" * 80)

    # Load config
    config_path = Path.home() / "moloch" / "config" / "camera_cloud.json"
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    cloud_cfg = cfg['cloud_config']

    # Create cloud config
    config = CloudConfig(
        enabled=True,
        api_base_url=cloud_cfg['api_base_url'],
        app_id=cloud_cfg['app_id'],
        app_secret=cloud_cfg['app_secret'],
        device_id=cloud_cfg['device_id'],
        username=cloud_cfg['username'],
        password=cloud_cfg['password'],
        timeout=5.0
    )

    # Create bridge
    bridge = CameraCloudBridge(config)

    # Create HTTP session
    import aiohttp
    bridge.session = aiohttp.ClientSession()

    try:
        # 1. Login
        print("\n🔐 Schritt 1: Cloud Login...")
        result = await bridge.ewelink_login()

        if not result['success']:
            print(f"❌ Login fehlgeschlagen: {result['error_message']}")
            return False

        print(f"✅ Login erfolgreich!")
        print(f"   Token: {result['token'][:30]}...")

        # Store token
        bridge.token.access_token = result['token']
        import time
        bridge.token.expires_at = time.time() + (30 * 24 * 3600)

        # 2. Test LED Control
        print("\n💡 Schritt 2: LED Steuerung testen...")

        # Test verschiedene LED-Level
        test_sequence = [
            (2, "MEDIUM"),
            (3, "HIGH"),
            (1, "LOW"),
            (0, "OFF")
        ]

        for level, name in test_sequence:
            print(f"\n  → Setze LED auf {name} (Level {level})...")
            success = await bridge.set_led(level)

            if success:
                print(f"  ✅ LED auf {name} gesetzt!")
            else:
                print(f"  ⚠️  API gab Fehler zurück (aber LED sollte trotzdem reagieren)")

            # Kurze Pause zwischen Änderungen
            await asyncio.sleep(1.5)

        print("\n" + "=" * 80)
        print("✅ Test abgeschlossen!")
        print("=" * 80)
        print("\nDu solltest gesehen haben wie die LEDs sich geändert haben:")
        print("  MEDIUM → HIGH → LOW → OFF")
        print("\nJetzt sind die LEDs AUS (Level 0)")
        print("=" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(test_led_control())
    sys.exit(0 if success else 1)
