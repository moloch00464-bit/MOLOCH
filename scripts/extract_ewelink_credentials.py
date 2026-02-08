#!/usr/bin/env python3
"""
eWeLink Credentials Extractor - mitmproxy addon
================================================

Automatisch AppID und AppSecret aus eWeLink App Traffic extrahieren.

Usage:
    mitmdump -s extract_ewelink_credentials.py

oder interaktiv:
    mitmproxy -s extract_ewelink_credentials.py
"""

import json
import re
from pathlib import Path
from mitmproxy import http

class EWeLinkCredentialsExtractor:
    """Extrahiert eWeLink Credentials aus HTTP Traffic."""

    def __init__(self):
        self.found_credentials = {
            'app_id': None,
            'app_secret': None,
            'nonce': None,
            'timestamp': None,
            'signature': None
        }
        self.saved = False

    def request(self, flow: http.HTTPFlow) -> None:
        """Analysiert jeden Request."""

        # Nur eWeLink API Requests
        if 'coolkit.cc' not in flow.request.pretty_host:
            return

        print(f"\n{'='*80}")
        print(f"🔍 eWeLink Request gefunden!")
        print(f"{'='*80}")
        print(f"URL: {flow.request.pretty_url}")
        print(f"Method: {flow.request.method}")

        # Headers analysieren
        headers = dict(flow.request.headers)

        # AppID extrahieren
        if 'X-CK-Appid' in headers or 'x-ck-appid' in headers:
            app_id = headers.get('X-CK-Appid') or headers.get('x-ck-appid')
            if app_id and not self.found_credentials['app_id']:
                self.found_credentials['app_id'] = app_id
                print(f"\n✅ AppID gefunden: {app_id}")

        # Nonce extrahieren
        if 'X-CK-Nonce' in headers or 'x-ck-nonce' in headers:
            nonce = headers.get('X-CK-Nonce') or headers.get('x-ck-nonce')
            if nonce:
                self.found_credentials['nonce'] = nonce
                print(f"✅ Nonce: {nonce}")

        # Timestamp extrahieren
        if 'X-CK-Timestamp' in headers or 'x-ck-timestamp' in headers:
            timestamp = headers.get('X-CK-Timestamp') or headers.get('x-ck-timestamp')
            if timestamp:
                self.found_credentials['timestamp'] = timestamp
                print(f"✅ Timestamp: {timestamp}")

        # Signature extrahieren
        if 'Authorization' in headers:
            auth = headers.get('Authorization')
            if auth and auth.startswith('Sign '):
                signature = auth.replace('Sign ', '')
                self.found_credentials['signature'] = signature
                print(f"✅ Signature: {signature[:40]}...")

        # Alle Headers ausgeben
        print(f"\n📋 Request Headers:")
        for key, value in headers.items():
            if key.lower().startswith('x-ck') or key.lower() == 'authorization':
                print(f"  {key}: {value}")

        # Request Body analysieren
        if flow.request.content:
            try:
                body = flow.request.content.decode('utf-8')
                print(f"\n📦 Request Body:")
                print(f"  {body[:200]}")

                # Versuche JSON zu parsen
                try:
                    body_json = json.loads(body)
                    print(f"\n📝 JSON Body:")
                    print(json.dumps(body_json, indent=2))
                except:
                    pass

            except Exception as e:
                pass

        # AppSecret aus Query-Parametern versuchen
        if 'appsecret' in flow.request.pretty_url.lower():
            match = re.search(r'appsecret=([^&]+)', flow.request.pretty_url, re.IGNORECASE)
            if match:
                self.found_credentials['app_secret'] = match.group(1)
                print(f"\n✅ AppSecret aus URL: {match.group(1)}")

        # Wenn wir AppID haben, aber noch kein AppSecret, versuche HMAC reverse
        if self.found_credentials['app_id'] and not self.found_credentials['app_secret']:
            print(f"\n⚠️  AppSecret nicht gefunden - muss aus APK extrahiert werden")
            print(f"    Bekannte Secrets für diese AppID:")
            self._suggest_app_secret()

        # Speichern wenn wir genug haben
        if self.found_credentials['app_id']:
            self._save_credentials()

    def response(self, flow: http.HTTPFlow) -> None:
        """Analysiert jede Response."""

        # Nur eWeLink API Responses
        if 'coolkit.cc' not in flow.request.pretty_host:
            return

        print(f"\n📥 Response Status: {flow.response.status_code}")

        # Response Body analysieren
        if flow.response.content:
            try:
                body = flow.response.content.decode('utf-8')
                try:
                    body_json = json.loads(body)
                    print(f"📝 Response JSON:")
                    print(json.dumps(body_json, indent=2)[:500])
                except:
                    print(f"📝 Response Text:")
                    print(body[:200])
            except:
                pass

    def _suggest_app_secret(self):
        """Schlägt bekannte AppSecrets vor basierend auf AppID."""

        known_pairs = {
            "oeVkj2lYFGnJu5XUtWisfW4utiN4u9Mq": "6Nz4n0LrnJGWLazlWZLd0JmkKYFZ6pNz",
            "YzfeftUVcZ6twZw1OoVKPRFYTrGEg01Q": "4G91qSoboqYO4Y0XJ0LPPKIsq8reHdfa",
            "R8Oq3y0eSZSYdKccHlrQzT1ACCOUT9Gv": "1ve5Qk9GXfUhKAn1svnKwpAlxXkMarru",
        }

        app_id = self.found_credentials['app_id']
        if app_id in known_pairs:
            suggested_secret = known_pairs[app_id]
            print(f"    Möglich: {suggested_secret}")
            self.found_credentials['app_secret'] = suggested_secret
            print(f"\n✅ AppSecret (vorgeschlagen): {suggested_secret}")
        else:
            print(f"    Keine bekannten Secrets für diese AppID")
            print(f"    Du musst die APK dekompilieren oder weitere Requests analysieren")

    def _save_credentials(self):
        """Speichert gefundene Credentials in camera_cloud.json."""

        if self.saved:
            return

        if not self.found_credentials['app_id']:
            return

        # Versuche AppSecret zu erraten falls nicht gefunden
        if not self.found_credentials['app_secret']:
            self._suggest_app_secret()

        config_path = Path.home() / "moloch" / "config" / "camera_cloud.json"

        if not config_path.exists():
            print(f"\n⚠️  Config nicht gefunden: {config_path}")
            return

        try:
            # Config laden
            with open(config_path, 'r') as f:
                config = json.load(f)

            # AppID updaten
            if self.found_credentials['app_id']:
                config['cloud_config']['app_id'] = self.found_credentials['app_id']

            # AppSecret updaten (falls gefunden)
            if self.found_credentials['app_secret']:
                config['cloud_config']['app_secret'] = self.found_credentials['app_secret']

            # Speichern
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"\n{'='*80}")
            print(f"✅ Credentials gespeichert in: {config_path}")
            print(f"{'='*80}")
            print(f"AppID:     {self.found_credentials['app_id']}")
            print(f"AppSecret: {self.found_credentials['app_secret'] or 'NICHT GEFUNDEN'}")
            print(f"{'='*80}")

            if self.found_credentials['app_secret']:
                print(f"\n🎉 Fertig! Teste jetzt mit:")
                print(f"    cd ~/moloch")
                print(f"    python -m core.gui.camera_control_panel")
            else:
                print(f"\n⚠️  AppSecret fehlt noch!")
                print(f"    Dekompiliere die APK oder analysiere mehr Traffic")

            self.saved = True

        except Exception as e:
            print(f"\n❌ Fehler beim Speichern: {e}")


# mitmproxy addon
addons = [EWeLinkCredentialsExtractor()]


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   eWeLink Credentials Extractor                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dieses Script MUSS mit mitmproxy gestartet werden:

    mitmproxy -s extract_ewelink_credentials.py

oder im Hintergrund:

    mitmdump -s extract_ewelink_credentials.py

Dann:
1. Handy-Proxy konfigurieren (Pi-IP:8080)
2. CA-Zertifikat installieren (http://mitm.it)
3. eWeLink App öffnen und einloggen
4. Script extrahiert automatisch AppID & AppSecret!
""")
