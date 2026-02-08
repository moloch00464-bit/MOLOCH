#!/usr/bin/env python3
"""
Test State Change Detection
============================

Prüft ob sich die Parameter in der API ändern wenn wir Befehle senden.
"""

import asyncio
import sys
from pathlib import Path
import time
import json as json_lib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
import json


async def get_params(bridge, config):
    """Get current device parameters."""
    endpoint = f"{config.api_base_url}/v2/device/thing"
    params_url = {"id": config.device_id}
    headers = {
        'Authorization': f'Bearer {bridge.token.access_token}',
        'X-CK-Appid': config.app_id
    }

    async with bridge.session.get(endpoint, params=params_url, headers=headers) as response:
        resp_body = await response.json()
        if resp_body.get('error') == 0 and 'data' in resp_body:
            thing_list = resp_body['data'].get('thingList', [])
            if thing_list:
                return thing_list[0]['itemData']['params']
    return {}


async def test_state_changes():
    """Test if state changes in API after sending commands."""

    print("=" * 80)
    print("🔍 STATE CHANGE DETECTION TEST")
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
        print("\n🔐 Login...")
        result = await bridge.ewelink_login()

        if not result['success']:
            print(f"❌ Login fehlgeschlagen")
            return False

        print(f"✅ Login erfolgreich!")

        # Store token
        bridge.token.access_token = result['token']
        bridge.token.expires_at = time.time() + (30 * 24 * 3600)

        # =====================================================================
        # TEST 1: LED Control
        # =====================================================================
        print("\n" + "=" * 80)
        print("💡 TEST 1: LED State Change")
        print("=" * 80)

        # Get initial state
        print("\n📊 VORHER: Hole aktuellen Zustand...")
        params_before = await get_params(bridge, config)
        led_before = params_before.get('lightStrength', '???')
        print(f"   lightStrength = {led_before}")

        # Send command
        print(f"\n🔴 BEFEHL: Setze LED auf 3 (HIGH)...")
        success = await bridge.set_led(3)
        print(f"   API Response: {'✅ OK' if success else '❌ FAIL'}")

        # Wait a bit
        print("   ⏳ Warte 2 Sekunden...")
        await asyncio.sleep(2)

        # Get state after
        print("\n📊 NACHHER: Hole neuen Zustand...")
        params_after = await get_params(bridge, config)
        led_after = params_after.get('lightStrength', '???')
        print(f"   lightStrength = {led_after}")

        # Compare
        print("\n" + "─" * 80)
        if led_before != led_after:
            print(f"✅ ÄNDERUNG ERKANNT! {led_before} → {led_after}")
        else:
            print(f"❌ KEINE ÄNDERUNG! Bleibt bei {led_before}")
        print("─" * 80)

        # =====================================================================
        # TEST 2: Night Mode
        # =====================================================================
        print("\n" + "=" * 80)
        print("🌙 TEST 2: Night Mode State Change")
        print("=" * 80)

        # Get initial state
        print("\n📊 VORHER: Hole aktuellen Zustand...")
        params_before = await get_params(bridge, config)
        night_before = params_before.get('nightVision', '???')
        print(f"   nightVision = {night_before}")

        # Send command - toggle to different value
        target_mode = 'night' if night_before != 2 else 'auto'
        target_value = 2 if target_mode == 'night' else 1

        print(f"\n🌙 BEFEHL: Setze Night Mode auf '{target_mode}' (Wert {target_value})...")
        success = await bridge.set_night(target_mode)
        print(f"   API Response: {'✅ OK' if success else '❌ FAIL'}")

        # Wait a bit
        print("   ⏳ Warte 2 Sekunden...")
        await asyncio.sleep(2)

        # Get state after
        print("\n📊 NACHHER: Hole neuen Zustand...")
        params_after = await get_params(bridge, config)
        night_after = params_after.get('nightVision', '???')
        print(f"   nightVision = {night_after}")

        # Compare
        print("\n" + "─" * 80)
        if night_before != night_after:
            print(f"✅ ÄNDERUNG ERKANNT! {night_before} → {night_after}")
        else:
            print(f"❌ KEINE ÄNDERUNG! Bleibt bei {night_before}")
        print("─" * 80)

        # =====================================================================
        # FINAL RESULT
        # =====================================================================
        print("\n" + "=" * 80)
        print("🎯 FAZIT")
        print("=" * 80)
        print("\nWenn die API-Werte sich NICHT ändern:")
        print("   → Die Befehle kommen NICHT bei der Kamera an")
        print("   → Wir brauchen eine andere Methode (WebSocket? Anderer Endpoint?)")
        print("\nWenn die API-Werte sich ÄNDERN aber du physisch nichts siehst:")
        print("   → Die Befehle kommen an, aber die Parameter sind falsch")
        print("   → Oder die LEDs sind kaputt")
        print("=" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(test_state_changes())
    sys.exit(0 if success else 1)
