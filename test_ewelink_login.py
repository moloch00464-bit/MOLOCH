#!/usr/bin/env python3
"""
Test Script für eWeLink API v2 Login
=====================================

Testet die neue ewelink_login() Methode mit vollständigem Response-Logging.

Usage:
    python test_ewelink_login.py

Author: M.O.L.O.C.H. System
Date: 2026-02-07
"""

import asyncio
import logging
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_ewelink_login():
    """Test eWeLink API v2 Login."""

    # Konfiguration (Werte müssen noch eingetragen werden)
    config = CloudConfig(
        enabled=True,
        api_base_url="https://eu-apia.coolkit.cc",  # EU Region API
        app_id="YOUR_APP_ID_HERE",  # eWeLink App ID
        app_secret="YOUR_APP_SECRET_HERE",  # eWeLink App Secret
        username="YOUR_EMAIL_HERE",  # eWeLink Account Email
        password="YOUR_PASSWORD_HERE",  # eWeLink Account Password
        timeout=10.0
    )

    print("=" * 80)
    print("eWeLink API v2 Login Test")
    print("=" * 80)
    print()

    # Bridge erstellen und Session initialisieren
    bridge = CameraCloudBridge(config)

    try:
        # HTTP Session initialisieren
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        bridge.session = aiohttp.ClientSession(timeout=timeout)

        # Login durchführen
        print("Calling ewelink_login()...\n")
        result = await bridge.ewelink_login()

        # Ergebnis ausgeben
        print("\n" + "=" * 80)
        print("Login Result")
        print("=" * 80)
        print(f"Success: {result['success']}")
        print(f"Status Code: {result['status_code']}")
        print(f"Error Message: {result['error_message']}")
        print(f"Token: {result['token'][:50]}..." if result['token'] else "Token: (none)")
        print(f"Request Timestamp: {result['request_timestamp']}")
        print()

        if result['success']:
            print("✓ Login erfolgreich!")
        else:
            print("✗ Login fehlgeschlagen!")
            print(f"  Fehler: {result['error_message']}")

        print("=" * 80)

    finally:
        # Session schließen
        if bridge.session:
            await bridge.session.close()


if __name__ == "__main__":
    print()
    print("HINWEIS: Bitte Config-Werte in test_ewelink_login.py eintragen:")
    print("  - app_id")
    print("  - app_secret")
    print("  - username (Email)")
    print("  - password")
    print()

    asyncio.run(test_ewelink_login())
