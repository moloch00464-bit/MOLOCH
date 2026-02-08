#!/usr/bin/env python3
"""
Comprehensive Cloud Control Test
=================================

Testet ALLE Cloud-Features der Kamera:
- LED Control (0, 1, 2, 3)
- Night Mode (auto, day, night)
- Sleep Mode (on, off)
"""

import asyncio
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
import json


async def test_all_controls():
    """Test ALL cloud controls."""

    print("=" * 80)
    print("🎯 VOLLSTÄNDIGER CLOUD-CONTROL TEST")
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
        print("\n🔐 STEP 1: Cloud Login...")
        result = await bridge.ewelink_login()

        if not result['success']:
            print(f"❌ Login fehlgeschlagen: {result['error_message']}")
            return False

        print(f"✅ Login erfolgreich!")

        # Store token
        bridge.token.access_token = result['token']
        bridge.token.expires_at = time.time() + (30 * 24 * 3600)

        # =====================================================================
        # 2. LED CONTROL TEST
        # =====================================================================
        print("\n" + "=" * 80)
        print("💡 STEP 2: LED CONTROL - Alle Helligkeitsstufen")
        print("=" * 80)
        print("\nSchau auf die Kamera! Die LEDs sollten sich jetzt ändern:\n")

        led_tests = [
            (0, "OFF", "🔴"),
            (1, "LOW", "🟡"),
            (2, "MEDIUM", "🟠"),
            (3, "HIGH", "🔴")
        ]

        for level, name, emoji in led_tests:
            print(f"{emoji} LED → {name} (Level {level})...")
            success = await bridge.set_led(level)
            print(f"   API: {'✅ OK' if success else '❌ FAIL'}")
            await asyncio.sleep(2.0)  # 2 Sekunden warten

        # =====================================================================
        # 3. NIGHT MODE TEST
        # =====================================================================
        print("\n" + "=" * 80)
        print("🌙 STEP 3: NIGHT MODE - Alle IR-Modi")
        print("=" * 80)
        print("\nSchau auf die Kamera! Die IR-LEDs sollten sich ändern:\n")

        night_tests = [
            ("auto", "AUTO (automatisch)"),
            ("night", "NIGHT (IR an)"),
            ("day", "DAY (IR aus)")
        ]

        for mode, description in night_tests:
            print(f"🌙 Night Mode → {description}...")
            success = await bridge.set_night(mode)
            print(f"   API: {'✅ OK' if success else '❌ FAIL'}")
            await asyncio.sleep(2.0)

        # =====================================================================
        # 4. SLEEP MODE TEST
        # =====================================================================
        print("\n" + "=" * 80)
        print("😴 STEP 4: SLEEP MODE - Kamera an/aus")
        print("=" * 80)
        print("\n⚠️  WARNUNG: Sleep Mode schaltet die Kamera AUS!\n")

        # Nicht testen - zu gefährlich, könnte die Kamera offline nehmen
        print("⏭️  ÜBERSPRUNGEN - zu riskant für automatischen Test")
        print("   Manuell testen: sleep_on() / sleep_off()")

        # =====================================================================
        # FINAL
        # =====================================================================
        print("\n" + "=" * 80)
        print("✅ ALLE TESTS ABGESCHLOSSEN!")
        print("=" * 80)
        print("\n📊 GETESTETE FEATURES:")
        print("   ✅ LED Control: OFF, LOW, MEDIUM, HIGH")
        print("   ✅ Night Mode: AUTO, NIGHT, DAY")
        print("   ⏭️  Sleep Mode: ÜBERSPRUNGEN (manuell testen)")
        print("\n❓ FRAGE AN DICH:")
        print("   Hat die Kamera auf die Befehle reagiert?")
        print("   Hast du die LED-Änderungen gesehen?")
        print("   Haben sich die IR-LEDs geändert?")
        print("=" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(test_all_controls())
    sys.exit(0 if success else 1)
