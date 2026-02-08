#!/usr/bin/env python3
"""
Get Current Device State
=========================

Holt die aktuellen Parameter der Kamera um zu sehen welche Werte möglich sind.
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


async def get_device_state():
    """Get current device state and parameters."""

    print("=" * 80)
    print("🔍 DEVICE STATE ABFRAGE")
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
            print(f"❌ Login fehlgeschlagen: {result['error_message']}")
            return False

        print(f"✅ Login erfolgreich!")

        # Store token
        bridge.token.access_token = result['token']
        bridge.token.expires_at = time.time() + (30 * 24 * 3600)

        # 2. Get device info
        print("\n📡 Hole Device Info...")

        endpoint = f"{config.api_base_url}/v2/device/thing"
        params = {
            "id": config.device_id
        }
        headers = {
            'Authorization': f'Bearer {bridge.token.access_token}',
            'X-CK-Appid': config.app_id
        }

        async with bridge.session.get(endpoint, params=params, headers=headers) as response:
            status_code = response.status
            resp_body = await response.json()

            print(f"\n📊 HTTP Status: {status_code}")
            print(f"📊 Response:")
            print("=" * 80)
            print(json_lib.dumps(resp_body, indent=2, ensure_ascii=False))
            print("=" * 80)

            if resp_body.get('error') == 0 and 'data' in resp_body:
                data = resp_body['data']

                print("\n🎯 WICHTIGE INFOS:")
                print("=" * 80)

                # itemData enthält die aktuellen Parameter
                if 'itemData' in data:
                    item_data = data['itemData']

                    print(f"📷 Device ID: {item_data.get('id')}")
                    print(f"📷 Name: {item_data.get('name')}")
                    print(f"📷 Model: {item_data.get('extra', {}).get('model')}")
                    print(f"📷 Online: {item_data.get('online')}")

                    print(f"\n💡 AKTUELLE PARAMETER:")
                    print("-" * 80)

                    params = item_data.get('params', {})
                    for key, value in params.items():
                        print(f"   {key}: {value}")

                    print("-" * 80)

                print("=" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(get_device_state())
    sys.exit(0 if success else 1)
