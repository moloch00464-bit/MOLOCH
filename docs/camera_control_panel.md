# Camera Control Panel

**Created:** 2026-02-07
**Type:** Manual Camera Control GUI

---

## 📋 Überblick

Das **Camera Control Panel** ist eine GUI-Anwendung für vollständige manuelle Kamera-Steuerung.

**Hauptfunktionen:**
- ✅ Exklusive Kamera-Kontrolle (pausiert autonome Systeme)
- ✅ Manuelle PTZ-Steuerung mit Pfeiltasten
- ✅ Sleep/Privacy-Modus Steuerung
- ✅ LED-Helligkeits-Steuerung
- ✅ Night/IR-Modus Steuerung
- ✅ Mikrofon-Verstärkungs-Steuerung
- ✅ Status-Monitor in Echtzeit
- ✅ Watchdog für Crash-Recovery

---

## 🚀 Start

### Via Desktop-Icon
1. **Doppelklick** auf **"Camera Control Panel"** Icon
2. GUI öffnet sich in Terminal-Fenster
3. Klick auf **"Connect Camera"**
4. Exclusive Lock wird automatisch erworben

### Via Kommandozeile
```bash
cd ~/moloch
python3 -m core.gui.camera_control_panel
```

---

## 🎛️ GUI-Elemente

### Connection Section
- **Camera IP:** IP-Adresse der Kamera (Standard: 192.168.178.25)
- **Connect Camera:** Verbindet zur Kamera und erwirbt Exclusive Lock
- **Disconnect:** Trennt Verbindung und gibt Lock frei

### Cloud Login Section
- **Username:** Sonoff Cloud Benutzername
- **Password:** Sonoff Cloud Passwort (maskiert)
- **Login:** Testet Cloud-Anmeldung
- **Save to Config:** Speichert Credentials in config/camera_cloud.json

### Status Section
- **ONVIF:** Verbindungsstatus zur Kamera (via ONVIF)
- **Cloud:** Verbindungsstatus zur Cloud-API
- **Exclusive Lock:** Status des exklusiven Locks
- **Position:** Aktuelle PTZ-Position (Pan/Tilt in Grad)

### PTZ Control Section
```
        ↑
    ←   ⌂   →
        ↓
     [STOP]
```
- **Pfeiltasten:** Bewegen in 10°-Schritten
- **⌂ (Home):** Zurück zur Mittelposition (0°, 0°)
- **STOP:** Stoppt alle Bewegungen

### Camera Features Section
- **Sleep/Privacy Mode:** ON/OFF Buttons
- **LED Brightness:** Dropdown (OFF/LOW/MEDIUM/HIGH)
- **Night/IR Mode:** Dropdown (AUTO/DAY/NIGHT)
- **Mic Gain:** Slider (0-100%)

### Log Section
- Zeigt alle Aktionen und Statusmeldungen
- Farbcodiert: Grün=Erfolg, Orange=Warnung, Rot=Fehler

---

## 🔒 Exclusive Lock System

### Was passiert beim Connect?

1. **Exclusive Lock erwerben:**
   ```python
   controller.acquire_exclusive("camera_control_panel")
   ```

2. **Control Mode auf MANUAL setzen:**
   ```python
   controller.set_mode(ControlMode.MANUAL_OVERRIDE)
   ```

3. **Autonome Systeme pausieren:**
   - SystemAutonomy: User Override aktiviert
   - Vision Pipeline gestoppt
   - AutonomousTracker gestoppt
   - Alle autonomen Loops prüfen Exclusive Lock

### Was passiert beim Disconnect?

1. **Lock freigeben:**
   ```python
   controller.release_exclusive("camera_control_panel")
   ```

2. **Control Mode auf AUTONOMOUS setzen:**
   ```python
   controller.set_mode(ControlMode.AUTONOMOUS)
   ```

3. **Autonome Systeme fortsetzen:**
   - SystemAutonomy: User Override deaktiviert
   - Vision Pipeline gestartet
   - AutonomousTracker gestartet
   - Autonome Kontrolle wiederhergestellt

---

## 🐕 Watchdog System

**Zweck:** Automatische Recovery wenn GUI crasht

**Funktionsweise:**
```python
WATCHDOG_TIMEOUT = 10.0  # Sekunden

# Watchdog überwacht Heartbeat
if time_since_last_heartbeat > 10.0:
    # Automatisch:
    controller.release_exclusive()
    controller.set_mode(ControlMode.AUTONOMOUS)
```

**Status-Updates als Heartbeat:**
- Jede Sekunde wird Status aktualisiert
- Heartbeat-Zeitstempel wird gesetzt
- Watchdog prüft Zeitstempel
- Bei Stillstand > 10s → Automatic Release

---

## ⚡ Threadsafe Operations

### GUI Thread
- Alle UI-Updates
- Button-Clicks
- Status-Label-Updates

### Status Update Thread
- Läuft alle 1 Sekunde
- Fragt Kamera-Status ab
- Aktualisiert GUI via `root.after()`
- Setzt Heartbeat

### Feature Control Threads
- Sleep/LED/Night/Mic-Operationen
- Laufen in separaten Threads
- Blockieren GUI nicht
- 3-Sekunden Timeout pro Operation

---

## 🎨 UI-Features

### Farb-Codierung

**Status-Anzeigen:**
- 🟢 **Grün:** Verbunden / Aktiv / Erfolg
- 🔴 **Rot:** Getrennt / Fehler
- 🟠 **Orange:** Warnung / Inaktiv

**Log-Meldungen:**
- ✓ **Grün:** Erfolgreiche Aktion
- ⚠ **Orange:** Warnung (z.B. Feature nicht verfügbar)
- ✗ **Rot:** Fehler

### Echtzeit-Updates

**Position:**
```
Position: Pan=+45.2°, Tilt=-10.5°
```
Aktualisiert sich jede Sekunde automatisch.

---

## 🔧 Integration mit UnifiedCameraController

### Exclusive Lock Mechanismus

```python
# In UnifiedCameraController:
def acquire_exclusive(self, owner: str) -> bool:
    if self._exclusive_owner and self._exclusive_owner != owner:
        return False  # Bereits belegt

    self._exclusive_owner = owner
    self.set_mode(ControlMode.MANUAL_OVERRIDE)
    return True

def release_exclusive(self, owner: str):
    if self._exclusive_owner == owner:
        self._exclusive_owner = None

def has_exclusive_control(self, owner: str) -> bool:
    return self._exclusive_owner == owner

def is_exclusive_locked(self) -> bool:
    return self._exclusive_owner is not None
```

### Blockierung bei aktivem Lock

Wenn Lock aktiv ist:
- Andere Komponenten können PTZ **nicht** bewegen
- Vision kann **nicht** tracken
- Autonomous Tracker ist **inaktiv**

---

## 📊 Status-Monitoring

### Was wird überwacht?

| Status | Quelle | Update-Rate |
|--------|--------|-------------|
| ONVIF Connected | `status.connected` | 1 Hz |
| Cloud Connected | `status.cloud_connected` | 1 Hz |
| Exclusive Lock | `self.has_exclusive_lock` | 1 Hz |
| Position | `status.position.pan/tilt` | 1 Hz |

### Status-Query

```python
status = controller.get_status()

# ONVIF
status.connected  # bool

# Cloud
status.cloud_connected  # bool
status.cloud_status  # "connected", "error", "disabled"

# Features
status.night_mode_available  # bool (True wenn Cloud connected)
status.led_control_available  # bool
status.sleep_mode_available  # bool
status.mic_gain_available  # bool

# Position
status.position.pan  # float (Grad)
status.position.tilt  # float (Grad)
status.position.moving  # bool
```

---

## 🛠️ Troubleshooting

### GUI startet nicht

**Problem:** Desktop-Icon funktioniert nicht

**Lösung:**
```bash
# Manuell starten
cd ~/moloch
python3 -m core.gui.camera_control_panel

# Prüfen ob tkinter installiert ist
python3 -c "import tkinter; print('OK')"
```

### Kein Exclusive Lock

**Problem:** "Failed to acquire exclusive lock"

**Ursache:** Kamera wird bereits gesteuert

**Lösung:**
```bash
# Prüfen wer Lock hat
# In Python:
print(controller.exclusive_owner)

# Anderen Owner beenden oder:
controller.release_exclusive("other_owner")
```

### Features nicht verfügbar

**Problem:** "✗ Sleep mode control failed (not available)"

**Ursache:** Cloud nicht verbunden

**Lösung:**
```bash
# Cloud Config prüfen
cat ~/moloch/config/camera_cloud.json

# Cloud enabled setzen
# Credentials eintragen
# App neu starten
```

### Watchdog triggert

**Problem:** "Watchdog triggered! No heartbeat for 10.5s"

**Ursache:** GUI hat sich aufgehängt

**Ergebnis:** ✓ Lock automatisch freigegeben, autonome Systeme wieder aktiv

**Lösung:** GUI neu starten

---

## 🔐 Sicherheit

### Automatische Recovery

1. **GUI crasht:** Watchdog gibt Lock frei (10s)
2. **Verbindung verloren:** Disconnect-Handler gibt Lock frei
3. **Fenster geschlossen:** `on_closing()` gibt Lock frei

### Keine Race Conditions

- Alle Lock-Operationen sind thread-safe (`threading.Lock`)
- Status-Updates laufen in separatem Thread
- GUI-Updates via `root.after()` im Main-Thread

---

## 📁 Dateien

| Datei | Pfad | Zweck |
|-------|------|-------|
| **GUI App** | `core/gui/camera_control_panel.py` | Hauptanwendung |
| **Launcher** | `run_camera_control_panel.sh` | Start-Script |
| **Desktop Icon** | `~/Desktop/CameraControlPanel.desktop` | Desktop-Shortcut |
| **Dokumentation** | `docs/camera_control_panel.md` | Diese Datei |

---

## 🎯 Use Cases

### Use Case 1: Manuelle Kamera-Tests

```
1. Control Panel öffnen
2. Connect Camera
3. PTZ testen (Pfeiltasten)
4. Cloud-Features testen (Sleep/LED/Night)
5. Disconnect
→ Autonome Systeme laufen wieder
```

### Use Case 2: Kamera-Setup

```
1. Control Panel öffnen
2. Night Mode auf DAY setzen
3. LED auf LOW setzen
4. Position justieren
5. Disconnect
→ Einstellungen bleiben
```

### Use Case 3: Cloud Login Setup

```
1. Control Panel öffnen
2. Username/Password eingeben
3. "Login" klicken → Test
4. "Save to Config" klicken → Persistent
5. Disconnect
→ Credentials gespeichert in config/camera_cloud.json
```

### Use Case 4: Debugging

```
1. Control Panel öffnen
2. Log beobachten
3. Features einzeln testen
4. Fehlermeldungen analysieren
5. Disconnect
```

---

## 🚀 Erweiterte Funktionen

### Geplante Features

1. **Password Change Button** (TODO)
   - Ändert Cloud-Passwort
   - Speichert in Config

2. **Preset Positions** (TODO)
   - Speichern von Positionen
   - Abrufen per Klick

3. **Recording Control** (TODO)
   - Start/Stop Recording
   - Snapshot-Funktion

4. **Multi-Camera Support** (TODO)
   - Mehrere Kameras
   - Dropdown zur Auswahl

---

## 💡 Best Practices

### Beim Verwenden

1. **Immer Disconnect beim Beenden:**
   - Lock wird freigegeben
   - Autonome Systeme laufen wieder

2. **Watchdog beachten:**
   - Status-Updates müssen laufen
   - Bei Freeze gibt Watchdog Lock frei

3. **Cloud-Features:**
   - Funktionieren nur wenn Cloud connected
   - Orange-Meldungen sind normal wenn Cloud aus

### Beim Entwickeln

1. **Keine Blocking-Calls im GUI-Thread:**
   ```python
   # Falsch:
   def _set_sleep(self, enabled):
       self.controller.set_sleep_mode(enabled)  # Blockiert GUI!

   # Richtig:
   def _set_sleep(self, enabled):
       def task():
           self.controller.set_sleep_mode(enabled)
       threading.Thread(target=task, daemon=True).start()
   ```

2. **GUI-Updates nur im Main-Thread:**
   ```python
   # Richtig:
   self.root.after(0, self._update_status_labels, status)
   ```

3. **Exclusive Lock prüfen:**
   ```python
   if not self.has_exclusive_lock:
       self._log("⚠ No exclusive lock", "orange")
       return
   ```

---

## 📊 Statistiken

**Entwicklungszeit:** ~2 Stunden
**Code-Zeilen:** ~600 Zeilen Python + 100 Zeilen Docs
**Dependencies:** tkinter (built-in), threading (built-in)
**Getestet auf:** Raspberry Pi 5 mit Raspberry Pi OS

---

## 📝 Changelog

### Version 1.1 (2026-02-07)
- ✅ Cloud Login UI (Username/Password mit Maskierung)
- ✅ Save to Config Button (speichert Credentials)
- ✅ SystemAutonomy Integration (set_user_override)
- ✅ Verbesserte System-Pause (Vision + Tracker)
- ✅ Exclusive Lock Prüfung in autonomen Loops

### Version 1.0 (2026-02-07)
- ✅ Initiale Implementation
- ✅ Exclusive Lock System
- ✅ PTZ Control
- ✅ Cloud Feature Control (Sleep/LED/Night/Mic)
- ✅ Status Monitoring
- ✅ Watchdog System
- ✅ Desktop Integration

---

## 🎓 Technische Details

### Threading-Modell

```
┌─────────────────┐
│   GUI Thread    │  (Main Thread)
│  - UI Updates   │
│  - Button Clicks│
└────────┬────────┘
         │
         ├─────────────────────┐
         │                     │
┌────────▼────────┐   ┌───────▼────────┐
│ Status Thread   │   │ Feature Threads│
│ - 1 Hz Updates  │   │ - Async Calls  │
│ - Heartbeat     │   │ - 3s Timeout   │
└────────┬────────┘   └───────┬────────┘
         │                     │
         └──────────┬──────────┘
                    │
         ┌──────────▼───────────┐
         │ UnifiedCameraCtrl    │
         │ - Thread-safe Locks  │
         └──────────────────────┘
```

### Exclusive Lock State Machine

```
IDLE (No Owner)
      │
      ├─ acquire_exclusive("panel")
      ↓
LOCKED (Owner: panel)
      │
      ├─ release_exclusive("panel")
      ↓
IDLE (No Owner)
```

**Watchdog:**
```
LOCKED → 10s no heartbeat → FORCE_RELEASE → IDLE
```

---

## ✅ Zusammenfassung

Das Camera Control Panel ist eine vollständige manuelle Steuerungs-GUI mit:
- ✅ Exklusiver Kamera-Kontrolle
- ✅ Automatischer System-Pause
- ✅ Crash-Recovery via Watchdog
- ✅ Threadsafe Operations
- ✅ Desktop-Integration

**Einsatzbereit und getestet!** 🎉
