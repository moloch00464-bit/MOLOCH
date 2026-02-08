# eWeLink API v2 Login Implementation

## Übersicht

Implementierung des eWeLink API v2 Login-Verfahrens in `camera_cloud_bridge.py` mit vollständigem Request/Response-Logging.

## Neue Methode: `ewelink_login()`

### Signatur

```python
async def ewelink_login(self) -> Dict[str, Any]
```

### Authentifizierung

Die Methode implementiert das eWeLink API v2 Authentifizierungsschema:

1. **HMAC-SHA256 Signatur**:
   ```
   message = appid + timestamp
   signature = HMAC-SHA256(message, appsecret)
   sign = base64_encode(signature)
   ```

2. **Timestamp**: In Millisekunden (z.B. 1738933200000)

3. **Nonce**: 8-Zeichen zufälliger String (z.B. "a3f5c2e1")

### HTTP Request

**Endpoint**: `{api_base_url}/v2/user/login`

**Method**: POST

**Headers**:
- `X-CK-Appid`: {appid}
- `X-CK-Nonce`: {8-char random nonce}
- `Authorization`: Sign {sign}
- `Content-Type`: application/json

**Body**:
```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

### Return Value

Die Methode gibt ein Dictionary mit folgenden Feldern zurück:

```python
{
    'success': bool,              # True wenn Login erfolgreich
    'status_code': int,           # HTTP Status Code (z.B. 200, 401, 500)
    'error_message': str,         # Fehlerbeschreibung (leer wenn erfolgreich)
    'token': str,                 # Access Token (leer wenn fehlgeschlagen)
    'response_headers': dict,     # Alle Response Headers
    'response_body': dict,        # Vollständiger Response Body
    'request_timestamp': str,     # Timestamp des Requests (ISO Format)
    'request_details': dict       # Details des Requests (für Debugging)
}
```

### Logging

Die Methode loggt **vollständig**:

#### Request Logging:
- Timestamp
- Endpoint URL
- Timestamp in Millisekunden
- Nonce
- AppID
- Signatur-Message
- Signatur (base64)
- Email (Username)
- Alle Headers

#### Response Logging:
- HTTP Status Code
- Alle Response Headers
- Response Body (vollständig als JSON)

### Verwendung

#### Beispiel 1: Basic Usage

```python
import asyncio
from core.hardware.camera_cloud_bridge import CameraCloudBridge, CloudConfig

async def login_test():
    config = CloudConfig(
        enabled=True,
        api_base_url="https://eu-apia.coolkit.cc",
        app_id="YOUR_APP_ID",
        app_secret="YOUR_APP_SECRET",
        username="your@email.com",
        password="yourpassword"
    )

    bridge = CameraCloudBridge(config)

    # Session muss initialisiert sein
    import aiohttp
    bridge.session = aiohttp.ClientSession()

    try:
        result = await bridge.ewelink_login()

        if result['success']:
            print(f"✓ Login erfolgreich!")
            print(f"Token: {result['token']}")
        else:
            print(f"✗ Login fehlgeschlagen: {result['error_message']}")
            print(f"Status Code: {result['status_code']}")

    finally:
        await bridge.session.close()

asyncio.run(login_test())
```

#### Beispiel 2: Mit Test-Script

```bash
cd /home/molochzuhause/moloch
python test_ewelink_login.py
```

(Vorher Config-Werte im Script anpassen!)

### API Regionen

eWeLink hat verschiedene API-Endpunkte je nach Region:

- **EU**: `https://eu-apia.coolkit.cc`
- **US**: `https://us-apia.coolkit.cc`
- **Asia**: `https://as-apia.coolkit.cc`
- **China**: `https://cn-apia.coolkit.cc`

### Fehlerbehandlung

Die Methode fängt alle Exceptions ab und gibt ein strukturiertes Ergebnis zurück:

- **Timeout**: `status_code: 0`, `error_message: "Request timeout"`
- **Exception**: `status_code: 0`, `error_message: "Exception: {details}"`
- **HTTP Error**: `status_code: {code}`, `error_message: {msg from response}`

### Token-Extraktion

Die Methode versucht den Token aus verschiedenen möglichen Feldern zu extrahieren:

```python
token = (
    response_body.get('at', '') or
    response_body.get('access_token', '') or
    response_body.get('token', '') or
    response_body.get('data', {}).get('at', '') or
    response_body.get('data', {}).get('access_token', '')
)
```

### Debugging

Bei Problemen:

1. **Logging Level auf DEBUG setzen**:
   ```python
   bridge = CameraCloudBridge(config, log_level=logging.DEBUG)
   ```

2. **Request Details auslesen**:
   ```python
   result = await bridge.ewelink_login()
   print(result['request_details'])
   ```

3. **Response Body überprüfen**:
   ```python
   print(json.dumps(result['response_body'], indent=2))
   ```

## Hinzugefügte Imports

In `camera_cloud_bridge.py` wurden folgende Imports hinzugefügt:

```python
import hmac
import hashlib
import base64
import secrets
```

## Files Modifiziert

1. **`/home/molochzuhause/moloch/core/hardware/camera_cloud_bridge.py`**
   - Neue Imports hinzugefügt
   - Neue Methode `ewelink_login()` implementiert (Zeile 326-547)

2. **`/home/molochzuhause/moloch/test_ewelink_login.py`** (neu)
   - Test-Script für eWeLink Login

3. **`/home/molochzuhause/moloch/docs/ewelink_api_v2_login.md`** (neu)
   - Diese Dokumentation

## Nächste Schritte

1. Config-Werte (app_id, app_secret, username, password) eintragen
2. Test-Script ausführen
3. Response-Logging analysieren
4. Bei Erfolg: Token in CloudConfig integrieren
5. Weitere API-Calls implementieren (Gerätesteuerung)

## Referenzen

- eWeLink API v2 Dokumentation
- HMAC-SHA256: RFC 2104
- Base64 Encoding: RFC 4648
