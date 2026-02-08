#!/usr/bin/env python3
"""
Real-Time Camera Parameter Monitor
===================================

Überwacht die Kamera-Parameter in Echtzeit und zeigt Änderungen an.
"""

import asyncio
import sys
from pathlib import Path
import time
import json as json_lib
from datetime import datetime

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

    try:
        async with bridge.session.get(endpoint, params=params_url, headers=headers, timeout=3) as response:
            resp_body = await response.json()
            if resp_body.get('error') == 0 and 'data' in resp_body:
                thing_list = resp_body['data'].get('thingList', [])
                if thing_list:
                    return thing_list[0]['itemData']['params']
    except:
        pass
    return {}


async def monitor_camera():
    """Monitor camera parameters for changes."""

    print("=" * 80)
    print("👁️  KAMERA PARAMETER MONITOR - 30 SEKUNDEN")
    print("=" * 80)
    print("\n⚡ JETZT IN DER APP ALLES DURCHSCHALTEN!")
    print("   - LEDs an/aus/hell/dunkel")
    print("   - Night Mode")
    print("   - Alles was du findest!\n")
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
        # Login
        result = await bridge.ewelink_login()
        if not result['success']:
            print("❌ Login fehlgeschlagen")
            return False

        bridge.token.access_token = result['token']
        bridge.token.expires_at = time.time() + (30 * 24 * 3600)

        # Parameters we care about
        watch_params = [
            'lightStrength',
            'nightVision',
            'power',
            'microphoneVolume',
            'speakerVolume',
            'screenFlip',
            'smartTraceEnable',
            'moveDetection'
        ]

        # Get initial state
        prev_params = await get_params(bridge, config)
        prev_values = {k: prev_params.get(k, '?') for k in watch_params}

        print(f"\n⏱️  START: {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 80)
        print("📊 INITIAL STATE:")
        for key, value in prev_values.items():
            print(f"   {key:20s} = {value}")
        print("─" * 80)
        print("\n🔍 ÜBERWACHE ÄNDERUNGEN...\n")

        # Monitor for 30 seconds
        start_time = time.time()
        check_count = 0
        changes_detected = 0

        while time.time() - start_time < 30:
            check_count += 1

            # Get current state
            curr_params = await get_params(bridge, config)
            curr_values = {k: curr_params.get(k, '?') for k in watch_params}

            # Check for changes
            changed = False
            for key in watch_params:
                if prev_values.get(key) != curr_values.get(key):
                    changed = True
                    changes_detected += 1
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    old_val = prev_values.get(key)
                    new_val = curr_values.get(key)
                    print(f"🔥 [{timestamp}] {key:20s} : {old_val} → {new_val}")

            prev_values = curr_values

            # Wait before next check
            await asyncio.sleep(2)

        # Final report
        print("\n" + "=" * 80)
        print(f"✅ MONITORING BEENDET nach {check_count} Checks")
        print(f"📊 ÄNDERUNGEN ERKANNT: {changes_detected}")
        print("=" * 80)

        if changes_detected == 0:
            print("\n⚠️  KEINE ÄNDERUNGEN ERKANNT!")
            print("   Mögliche Gründe:")
            print("   1. Die App ändert die Cloud-Werte nicht sofort")
            print("   2. Die Parameter haben andere Namen")
            print("   3. Die Kamera synchronisiert nicht mit der Cloud")
        else:
            print("\n✅ ÄNDERUNGEN GEFUNDEN!")
            print("   Jetzt wissen wir welche Parameter funktionieren!")

        print("\n📊 FINAL STATE:")
        print("─" * 80)
        for key, value in prev_values.items():
            print(f"   {key:20s} = {value}")
        print("─" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(monitor_camera())
    sys.exit(0 if success else 1)
