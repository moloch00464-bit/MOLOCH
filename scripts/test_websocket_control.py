#!/usr/bin/env python3
"""
eWeLink WebSocket Camera Control Test
======================================

Steuert die Kamera via WebSocket (wie die App es macht).
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import hmac
import hashlib
import base64

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig
import websockets


async def test_websocket_control():
    """Test camera control via WebSocket."""

    print("=" * 80)
    print("🌐 eWeLink WebSocket Control Test")
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

    # Create bridge for login
    bridge = CameraCloudBridge(config)

    import aiohttp
    bridge.session = aiohttp.ClientSession()

    try:
        # 1. Login to get access token
        print("\n🔐 Step 1: Login to get access token...")
        result = await bridge.ewelink_login()

        if not result['success']:
            print(f"❌ Login failed: {result['error_message']}")
            return False

        access_token = result['token']
        user_apikey = result['response_body']['data']['user']['apikey']

        print(f"✅ Login successful!")
        print(f"   Token: {access_token[:30]}...")
        print(f"   API Key: {user_apikey}")

        # 2. Connect to WebSocket
        print("\n🌐 Step 2: Connecting to WebSocket...")

        # WebSocket URL for EU region
        ws_url = "wss://eu-pconnect6.coolkit.cc:8080/api/ws"
        print(f"   URL: {ws_url}")

        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connected!")

            # 3. Send handshake/login message
            print("\n🤝 Step 3: Sending handshake...")

            # Generate nonce (timestamp in milliseconds)
            nonce = str(int(time.time() * 1000))

            handshake = {
                "action": "userOnline",
                "userAgent": "app",
                "version": 8,
                "nonce": nonce,
                "apkVesrion": "1.8",
                "os": "ios",
                "at": access_token,
                "apikey": user_apikey,
                "ts": int(time.time()),
                "model": "iPhone10,6",
                "romVersion": "11.1.2",
                "sequence": str(int(time.time() * 1000))
            }

            await websocket.send(json.dumps(handshake))
            print(f"   Sent: {handshake['action']}")

            # 4. Wait for handshake response
            print("\n⏳ Step 4: Waiting for handshake response...")

            response = await websocket.recv()
            resp_data = json.loads(response)
            print(f"   Response: {json.dumps(resp_data, indent=2)}")

            if resp_data.get('error') == 0:
                print("✅ Handshake successful!")
            else:
                print(f"❌ Handshake failed: {resp_data}")
                return False

            # 5. Send device control command
            print("\n🎮 Step 5: Sending LED control command...")
            print("   Setting lightStrength to 3 (HIGH)...")

            sequence = str(int(time.time() * 1000))

            control_msg = {
                "action": "update",
                "apikey": user_apikey,
                "deviceid": config.device_id,
                "params": {
                    "lightStrength": 3
                },
                "userAgent": "app",
                "sequence": sequence,
                "ts": int(time.time())
            }

            await websocket.send(json.dumps(control_msg))
            print(f"   Sent: {json.dumps(control_msg, indent=2)}")

            # 6. Wait for response
            print("\n⏳ Step 6: Waiting for response...")

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                resp_data = json.loads(response)
                print(f"   Response: {json.dumps(resp_data, indent=2)}")

                if resp_data.get('error') == 0:
                    print("✅ LED control command successful!")
                    print("\n🔥 SCHAU AUF DIE KAMERA - SIND DIE LEDS JETZT AN?")
                else:
                    print(f"⚠️  Response: {resp_data}")

            except asyncio.TimeoutError:
                print("⏱️  No immediate response (might still work)")

            # 7. Test another command
            print("\n🎮 Step 7: Setting lightStrength to 0 (OFF)...")
            await asyncio.sleep(2)

            sequence = str(int(time.time() * 1000))
            control_msg["params"] = {"lightStrength": 0}
            control_msg["sequence"] = sequence
            control_msg["ts"] = int(time.time())

            await websocket.send(json.dumps(control_msg))
            print(f"   Sent: lightStrength=0")

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                resp_data = json.loads(response)
                print(f"   Response: {json.dumps(resp_data, indent=2)}")

                if resp_data.get('error') == 0:
                    print("✅ LED OFF command successful!")
                    print("\n🔥 SIND DIE LEDS JETZT AUS?")

            except asyncio.TimeoutError:
                print("⏱️  No immediate response")

            # Keep connection alive to listen for updates
            print("\n👂 Step 8: Listening for 10 seconds...")
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    resp_data = json.loads(response)
                    print(f"   📩 Received: {resp_data}")
            except asyncio.TimeoutError:
                print("⏱️  Timeout - no more messages")

        print("\n" + "=" * 80)
        print("✅ WebSocket test completed!")
        print("=" * 80)

        return True

    finally:
        await bridge.session.close()


if __name__ == "__main__":
    success = asyncio.run(test_websocket_control())
    sys.exit(0 if success else 1)
