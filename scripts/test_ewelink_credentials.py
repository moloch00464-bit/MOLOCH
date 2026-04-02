import os
#!/usr/bin/env python3
"""
eWeLink Credentials Tester
===========================

Testet automatisch bekannte eWeLink API Credentials.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig


# Bekannte öffentliche eWeLink Credentials aus verschiedenen Quellen
KNOWN_CREDENTIALS = [
    # Community credentials #1
    {
        "app_id": os.environ.get("EWELINK_APP_ID_1", "CHANGE_ME"),
        "app_secret": os.environ.get("EWELINK_APP_SECRET_1", "CHANGE_ME"),
        "source": "Community #1 (am häufigsten)"
    },
    # Community credentials #2
    {
        "app_id": os.environ.get("EWELINK_APP_ID_2", "CHANGE_ME"),
        "app_secret": os.environ.get("EWELINK_APP_SECRET_2", "CHANGE_ME"),
        "source": "Community #2"
    },
    # Community credentials #3
    {
        "app_id": os.environ.get("EWELINK_APP_ID_3", "CHANGE_ME"),
        "app_secret": os.environ.get("EWELINK_APP_SECRET_3", "CHANGE_ME"),
        "source": "Community #3"
    },
    # eWeLink official app (alt)
    {
        "app_id": os.environ.get("EWELINK_APP_ID_4", "CHANGE_ME"),
        "app_secret": os.environ.get("EWELINK_APP_SECRET_4", "CHANGE_ME"),
        "source": "Official App (alt)"
    },
]


async def test_credentials(app_id: str, app_secret: str, source: str) -> bool:
    """Test a single credential pair."""

    print(f"\n{'='*80}")
    print(f"🔍 Teste: {source}")
    print(f"{'='*80}")
    print(f"AppID:     {app_id}")
    print(f"AppSecret: {app_secret}")
    print(f"{'='*80}")

    config = CloudConfig(
        enabled=True,
        api_base_url="https://eu-apia.coolkit.cc",  # CORRECTED: with "a"!
        app_id=app_id,
        app_secret=app_secret,
        username=os.environ.get("EWELINK_USERNAME", "CHANGE_ME"),
        password=os.environ.get("EWELINK_PASSWORD", "CHANGE_ME"),
        timeout=5.0
    )

    bridge = CameraCloudBridge(config)

    # Create HTTP session
    import aiohttp
    bridge.session = aiohttp.ClientSession()

    try:
        result = await bridge.ewelink_login()

        if result['success']:
            print(f"\n🎉 ✅ SUCCESS! Funktionierende Credentials gefunden!")
            print(f"{'='*80}")
            print(f"Token: {result['token'][:40]}...")
            return True
        else:
            print(f"\n❌ Fehlgeschlagen: {result['error_message']}")
            print(f"   Status: {result['status_code']}")
            return False

    finally:
        await bridge.session.close()


async def main():
    """Test all known credentials."""

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              eWeLink Credentials Auto-Tester                                 ║
║                                                                              ║
║  Testet {len(KNOWN_CREDENTIALS)} bekannte eWeLink API Credential-Paare                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    for i, creds in enumerate(KNOWN_CREDENTIALS, 1):
        print(f"\n[{i}/{len(KNOWN_CREDENTIALS)}] Teste Credentials...")

        success = await test_credentials(
            creds['app_id'],
            creds['app_secret'],
            creds['source']
        )

        if success:
            # Speichere funktionierende Credentials
            print(f"\n💾 Speichere funktionierende Credentials...")

            import json
            config_path = Path.home() / "moloch" / "config" / "camera_cloud.json"

            with open(config_path, 'r') as f:
                config = json.load(f)

            config['cloud_config']['app_id'] = creds['app_id']
            config['cloud_config']['app_secret'] = creds['app_secret']

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✅ Credentials gespeichert in: {config_path}")
            print(f"\n{'='*80}")
            print(f"🎉 FERTIG! Teste jetzt:")
            print(f"   cd ~/moloch")
            print(f"   python -m core.gui.camera_control_panel")
            print(f"{'='*80}")
            return True

        # Kurze Pause zwischen Tests
        await asyncio.sleep(1)

    print(f"\n{'='*80}")
    print(f"❌ Keine funktionierenden Credentials gefunden!")
    print(f"{'='*80}")
    print(f"\nNächste Schritte:")
    print(f"1. Traffic Sniffing mit mitmproxy (siehe Anleitung)")
    print(f"2. APK dekompilieren")
    print(f"3. Offiziellen Developer Account beantragen")
    print(f"{'='*80}")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
