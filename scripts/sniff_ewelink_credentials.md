# eWeLink Credentials via Traffic Sniffing extrahieren

## Ziel
Echte `app_id` und `app_secret` aus der eWeLink Mobile App extrahieren.

## Vorbereitung

### 1. mitmproxy installieren (auf dem Pi)
```bash
sudo apt update
sudo apt install mitmproxy -y
```

### 2. Proxy starten
```bash
# Starte mitmproxy auf Port 8080
mitmproxy -p 8080 --set block_global=false
```

## Handy konfigurieren

### 3. Proxy auf dem Handy einstellen

**Android:**
- Einstellungen → WLAN → Dein WLAN lange drücken → Erweitert
- Proxy: Manuell
- Hostname: `<Pi-IP-Adresse>` (z.B. 192.168.178.XX)
- Port: `8080`
- Speichern

**iOS:**
- Einstellungen → WLAN → (i) neben deinem WLAN
- HTTP-Proxy: Manuell
- Server: `<Pi-IP-Adresse>`
- Port: `8080`

### 4. CA-Zertifikat installieren

**Im Handy-Browser öffnen:**
```
http://mitm.it
```

- "Get mitmproxy-ca-cert.pem" herunterladen
- Zertifikat installieren
  - Android: Als "CA-Zertifikat" installieren
  - iOS: Profil installieren + in Einstellungen aktivieren

## Credentials extrahieren

### 5. eWeLink App öffnen und einloggen

- eWeLink App öffnen (vorher ausloggen!)
- Mit deinem Account einloggen
- Login-Request wird von mitmproxy abgefangen

### 6. Traffic in mitmproxy analysieren

In der mitmproxy UI (Terminal):
- Mit Pfeiltasten navigieren
- Suche nach Request zu: `eu-api.coolkit.cc`
- Enter drücken zum Öffnen
- Tab drücken für Details

**Zu suchende Header:**
```
X-CK-Appid: <DEINE_APP_ID>
Authorization: Sign <signature>
```

**Oder im Request Body / URL Parameter:**
- `appid=...`
- `appsecret=...` (manchmal sichtbar)

### 7. Credentials notieren

Kopiere:
- **AppID** (aus X-CK-Appid Header)
- **AppSecret** (schwieriger - kann aus vorherigen Requests kommen)

**AppSecret finden:**
- Scrolle durch mehrere Requests
- Manchmal in App-Initialisierung oder Update-Requests
- Oder analysiere die APK (komplexer)

## Alternative: APK dekompilieren

Falls AppSecret nicht im Traffic sichtbar:

```bash
# APK von Handy extrahieren oder von APKMirror herunterladen
# Mit apktool dekompilieren
apktool d ewelink.apk

# In den Dateien nach appid/appsecret suchen
grep -r "appid" ewelink/
grep -r "appsecret" ewelink/
```

## Credentials in Config eintragen

```bash
nano ~/moloch/config/camera_cloud.json
```

Ersetze:
```json
"app_id": "<DEINE_APP_ID>",
"app_secret": "<DEIN_APP_SECRET>",
```

## Test

```bash
cd ~/moloch
python -m core.gui.camera_control_panel
```

Login-Button klicken → sollte HTTP 200 + Token bekommen!

## Hinweise

- **AppID** ist relativ einfach zu finden (in jedem Request)
- **AppSecret** ist schwieriger (nicht immer im Traffic sichtbar)
- Wenn nur AppID gefunden: APK dekompilieren notwendig
- Credentials sind app-version-spezifisch (können sich bei Updates ändern)

## Cleanup

Nach dem Sniffing:
- Proxy auf Handy deaktivieren
- CA-Zertifikat vom Handy löschen (Sicherheit!)
- mitmproxy beenden (Ctrl+C)
