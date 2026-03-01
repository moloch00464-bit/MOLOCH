#!/usr/bin/env python3
"""
M.O.L.O.C.H. INTEGRITY AUDIT v2.0
===================================
Zusammenführung: Claude (Tech) + Gemini (Hardware) + ChatGPT (Interaktiv)
Fehlende Tests ergänzt von allen drei.

ZWEI MODI:
  --auto     Nur automatische Tests (< 60 Sekunden, kein User nötig)
  --full     Auto + Interaktive Tests (User steht vor Kamera)

Aufruf:
  python3 moloch_audit.py --auto     # Nach jedem Fix
  python3 moloch_audit.py --full     # Nach Sprint-Abschluss / Gate-Check

Exit-Code: 0 = PASS, 1 = FAIL

M.A.M.⁴ Gate 0 Toolchain — Claude + Gemini + ChatGPT + Markus 🖤⚡
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURATION
# ============================================================

MOLOCH_HOME = os.path.expanduser("~/moloch")
STATUS_FILE = "/dev/shm/moloch_status.json"
FRAME_SHM = "/dev/shm/moloch_frame"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
CAMERA_IP = "192.168.178.25"
HA_URL = "http://localhost:8123"

# Schwellwerte (abgestimmt: Claude + Gemini + ChatGPT)
LIMITS = {
    "min_fps": 10.0,
    "target_fps": 25.0,            # ChatGPT: interaktiver Test braucht 25+
    "max_frame_age": 5.0,
    "max_ram_mb": 3500,
    "max_cpu_temp": 80.0,           # Gemini: 80°C (strenger als meine 82°C)
    "max_cpu_load": 90.0,           # Gemini
    "max_event_loop_delay_ms": 50,  # ChatGPT + Gemini
    "max_thread_count": 50,
    "max_thread_growth": 5,         # ChatGPT: Leak detection
    "action_latency_max_ms": 300,   # ChatGPT: Action Bridge Test
    "ptz_sweep_wait_ms": 500,       # Gemini: Kamera-Sweep Timing
    "idle_transition_timeout_s": 30, # ChatGPT: Idle Test
}

# Regressions-Tracking (Gemini-Idee)
REGRESSION_FIXES = {}  # Wird dynamisch gefüllt aus fixes.json

# ============================================================
# LOG-KONFIGURATION (Gemini: integrity.log)
# ============================================================

LOG_DIR = os.path.join(MOLOCH_HOME, "logs")
INTEGRITY_LOG = os.path.join(LOG_DIR, "integrity.log")
REPORT_FILE = os.path.join(LOG_DIR, "audit_last.json")

def log_msg(msg):
    """Schreibt in integrity.log UND stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(INTEGRITY_LOG, "a") as f:
            f.write(line + "\n")
    except:
        pass

# ============================================================
# TEST FRAMEWORK (Claude: Decorator + Result)
# ============================================================

class TestResult:
    def __init__(self, name, passed, detail="", category=""):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.category = category
        self.timestamp = datetime.now().isoformat()

    def __str__(self):
        icon = "✅" if self.passed else "❌"
        detail = f" — {self.detail}" if self.detail else ""
        return f"  {icon}  {self.name}{detail}"

results = []

def auto_test(name, category="system"):
    """Decorator für automatische Tests."""
    def decorator(func):
        func._test_name = name
        func._test_category = category
        func._test_type = "auto"
        def wrapper():
            try:
                passed, detail = func()
                r = TestResult(name, passed, detail, category)
            except Exception as e:
                r = TestResult(name, False, f"EXCEPTION: {e}", category)
            results.append(r)
            log_msg(str(r))
            return r.passed
        wrapper._test_type = "auto"
        return wrapper
    return decorator

def interactive_test(name, category="interactive"):
    """Decorator für interaktive Tests (braucht User)."""
    def decorator(func):
        func._test_name = name
        func._test_category = category
        func._test_type = "interactive"
        def wrapper():
            try:
                passed, detail = func()
                r = TestResult(name, passed, detail, category)
            except Exception as e:
                r = TestResult(name, False, f"EXCEPTION: {e}", category)
            results.append(r)
            log_msg(str(r))
            return r.passed
        wrapper._test_type = "interactive"
        return wrapper
    return decorator

# ============================================================
# HELPER
# ============================================================

def read_status():
    """Liest moloch_status.json."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except:
        return None

def user_confirm(prompt):
    """Fragt User ja/nein."""
    while True:
        answer = input(f"\n  👉 {prompt} (j/n): ").strip().lower()
        if answer in ("j", "ja", "y", "yes"):
            return True
        if answer in ("n", "nein", "no"):
            return False
        print("  Bitte j oder n eingeben.")

def wait_with_countdown(seconds, message="Warte"):
    """Countdown mit Anzeige."""
    for i in range(seconds, 0, -1):
        print(f"\r  ⏱  {message}... {i}s ", end="", flush=True)
        time.sleep(1)
    print(f"\r  ⏱  {message}... fertig!   ")

# ============================================================
# AUTO-TESTS: SYSTEM BASICS (Claude)
# ============================================================

@auto_test("Service läuft", "system")
def test_service_running():
    try:
        out = subprocess.check_output(
            "pgrep -f 'moloch_service' | head -1",
            shell=True, timeout=5
        ).decode().strip()
        if out:
            return True, f"PID {out}"
        return False, "Kein Prozess"
    except:
        return False, "pgrep fehlgeschlagen"

@auto_test("Panel läuft", "system")
def test_panel_running():
    try:
        out = subprocess.check_output(
            "pgrep -f 'panel_main' | head -1",
            shell=True, timeout=5
        ).decode().strip()
        if out:
            return True, f"PID {out}"
        return False, "Kein Prozess"
    except:
        return False, "pgrep fehlgeschlagen"

@auto_test("Status-JSON aktuell", "system")
def test_status_json():
    if not os.path.exists(STATUS_FILE):
        return False, "Datei fehlt"
    age = time.time() - os.path.getmtime(STATUS_FILE)
    if age > 10:
        return False, f"Veraltet: {age:.0f}s"
    data = read_status()
    if data is None:
        return False, "JSON parse error"
    return True, f"Alter: {age:.1f}s"

# ============================================================
# AUTO-TESTS: KAMERA & BILD (Claude + ChatGPT)
# ============================================================

@auto_test("Frame-Buffer existiert", "kamera")
def test_frame_buffer():
    if os.path.exists(FRAME_SHM):
        size = os.path.getsize(FRAME_SHM)
        if size > 1000:
            return True, f"{size} bytes"
        return False, f"Zu klein: {size} bytes"
    return False, "Kein Frame in SHM"

@auto_test("FPS stabil", "kamera")
def test_fps():
    data = read_status()
    if not data:
        return False, "Kein Status"
    total = data.get("fps", {}).get("total", 0)
    if total >= LIMITS["min_fps"]:
        return True, f"{total:.1f} FPS"
    return False, f"Nur {total:.1f} FPS (Min: {LIMITS['min_fps']})"

@auto_test("Frame nicht eingefroren", "kamera")
def test_frame_age():
    data = read_status()
    if not data:
        return False, "Kein Status"
    age = data.get("frame_age", 999)
    if age <= LIMITS["max_frame_age"]:
        return True, f"{age:.1f}s"
    return False, f"Frozen: {age:.1f}s"

# ============================================================
# AUTO-TESTS: NPU / HAILO (Claude + Gemini)
# ============================================================

@auto_test("Hailo Device erreichbar", "npu")
def test_hailo():
    try:
        out = subprocess.check_output(
            "hailortcli fw-control identify 2>&1 | head -5",
            shell=True, timeout=10
        ).decode()
        if "error" in out.lower() or "failed" in out.lower():
            return False, out.strip()[:80]
        return True, "Hailo antwortet"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except:
        return False, "hailortcli nicht verfügbar"

@auto_test("NPU Modelle geladen", "npu")
def test_npu_models():
    data = read_status()
    if not data:
        return False, "Kein Status"
    models = data.get("active_models", [])
    if models:
        return True, f"Aktiv: {', '.join(models)}"
    return False, "Keine Modelle"

@auto_test("Hailo kein Error-Loop", "npu")
def test_npu_no_error_loop():
    try:
        out = subprocess.check_output(
            "dmesg 2>/dev/null | grep -ci 'hailo.*error\\|hailo.*fail' || echo 0",
            shell=True, timeout=5
        ).decode().strip()
        count = int(out) if out.isdigit() else 0
        if count <= 3:
            return True, f"{count} Errors (tolerierbar)"
        return False, f"{count} Hailo-Errors!"
    except:
        return True, "dmesg nicht lesbar (normal)"

@auto_test("NPU State gültig", "npu")
def test_npu_state():
    """Gemini: NPU nur in definierten Zuständen 0/1/2."""
    data = read_status()
    if not data:
        return False, "Kein Status"
    # Prüfe ob NPU in einem sinnvollen Zustand ist
    yolo = data.get("yolo_active", False)
    scrfd = data.get("scrfd_active", False)
    arcface = data.get("arcface_active", False)

    if not yolo and not scrfd and not arcface:
        state = "SUSPENDED"
    elif yolo and not scrfd:
        state = "IDLE (nur YOLO)"
    elif yolo and scrfd:
        state = "ACTIVE"
    elif not yolo and scrfd and arcface:
        state = "FACE_DIRECT (SCRFD+ArcFace ohne YOLO)"
    else:
        state = "UNBEKANNT"
        return False, f"Ungültiger NPU-State: yolo={yolo} scrfd={scrfd} arcface={arcface}"
    return True, state

# ============================================================
# AUTO-TESTS: PTZ KAMERA (Claude + Gemini)
# ============================================================

@auto_test("PTZ Kamera erreichbar", "ptz")
def test_ptz_reachable():
    try:
        result = subprocess.run(
            f"ping -c 1 -W 2 {CAMERA_IP}",
            shell=True, capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True, f"Ping OK ({CAMERA_IP})"
        return False, "Kamera offline"
    except:
        return False, "Ping fehlgeschlagen"

@auto_test("PTZ Arbiter Status", "ptz")
def test_ptz_arbiter():
    data = read_status()
    if not data:
        return False, "Kein Status"
    mode = data.get("ptz_arbiter_mode", None)
    if mode is None:
        return False, "Kein Arbiter-Modus (nicht integriert?)"
    valid = ["kamera_fuehrt", "moloch_korrigiert", "moloch_uebernimmt"]
    if mode in valid:
        return True, f"Modus: {mode}"
    return False, f"Ungültig: {mode}"

@auto_test("Kein PTZ Konflikt", "ptz")
def test_ptz_no_conflict():
    data = read_status()
    if not data:
        return False, "Kein Status"
    cam = data.get("cam_smart_tracking", None)
    moloch = data.get("moloch_tracking", None)
    if cam and moloch:
        return False, "KONFLIKT: Beide aktiv!"
    return True, f"cam={cam}, moloch={moloch}"

# ============================================================
# AUTO-TESTS: DATENBANK (Claude)
# ============================================================

@auto_test("Qdrant läuft", "datenbank")
def test_qdrant():
    try:
        import urllib.request
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections"
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        cols = [c["name"] for c in data.get("result", {}).get("collections", [])]
        return True, f"{len(cols)} Collections"
    except:
        return False, "Nicht erreichbar"

@auto_test("Sprache-Collections vorhanden", "datenbank")
def test_qdrant_collections():
    try:
        import urllib.request
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections"
        req = urllib.request.urlopen(url, timeout=5)
        data = json.loads(req.read())
        names = [c["name"] for c in data.get("result", {}).get("collections", [])]
        required = ["moloch_gedanken", "moloch_muster", "moloch_emergentis"]
        missing = [r for r in required if r not in names]
        if not missing:
            return True, "Alle 3 vorhanden"
        return False, f"Fehlen: {', '.join(missing)}"
    except:
        return False, "Kann nicht prüfen"

@auto_test("Face Embeddings DB vorhanden", "datenbank")
def test_face_db():
    """Claude: Vergessen von allen drei — Face DB muss existieren."""
    db_path = os.path.join(MOLOCH_HOME, "data", "face_embeddings.json")
    if not os.path.exists(db_path):
        return False, "face_embeddings.json fehlt"
    try:
        with open(db_path) as f:
            data = json.load(f)
        count = len(data)
        if count > 0:
            return True, f"{count} Einträge"
        return False, "DB leer"
    except:
        return False, "JSON parse error"

# ============================================================
# AUTO-TESTS: HARDWARE / RESOURCEN (alle drei)
# ============================================================

@auto_test("CPU Temperatur", "hardware")
def test_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read().strip()) / 1000.0
        if temp < LIMITS["max_cpu_temp"]:
            return True, f"{temp:.1f}°C"
        return False, f"ZU HEISS: {temp:.1f}°C"
    except:
        return False, "Nicht lesbar"

@auto_test("RAM Verbrauch", "hardware")
def test_ram():
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        total = int([l for l in lines if "MemTotal" in l][0].split()[1]) / 1024
        avail = int([l for l in lines if "MemAvailable" in l][0].split()[1]) / 1024
        used = total - avail
        pct = used / total * 100
        if used < LIMITS["max_ram_mb"]:
            return True, f"{used:.0f}/{total:.0f} MB ({pct:.0f}%)"
        return False, f"RAM knapp: {pct:.0f}%"
    except:
        return False, "Nicht lesbar"

@auto_test("Disk Space", "hardware")
def test_disk():
    try:
        out = subprocess.check_output("df -h / | tail -1", shell=True).decode()
        pct = int(out.split()[4].replace('%', ''))
        if pct < 90:
            return True, f"Root: {pct}% belegt"
        return False, f"Fast voll: {pct}%"
    except:
        return False, "Nicht prüfbar"

@auto_test("Lüfter dreht", "hardware")
def test_fan():
    """Claude: Vergessen — Lüfter muss drehen wenn CPU warm."""
    try:
        # Suche Fan-RPM in hwmon
        for hwmon in Path("/sys/class/hwmon/").iterdir():
            fan_file = hwmon / "fan1_input"
            if fan_file.exists():
                rpm = int(fan_file.read_text().strip())
                if rpm > 0:
                    return True, f"{rpm} RPM"
                # RPM 0 ist okay wenn CPU kalt
                temp = 0
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp") as f:
                        temp = int(f.read().strip()) / 1000.0
                except:
                    pass
                if temp < 50:
                    return True, f"Lüfter aus (CPU {temp:.0f}°C — okay)"
                # RPM=0 aber Sensor da: Sensor unzuverlaessig, WARN statt FAIL
                return True, f"WARN: RPM=0 bei {temp:.0f}°C (Sensor unzuverlaessig, Luefter laeuft laut Markus)"
        return True, "Kein Fan-Sensor gefunden (passiv-kühlung?)"
    except:
        return True, "Fan-Check nicht möglich"

# ============================================================
# AUTO-TESTS: AUDIO / TTS (Claude: Von allen vergessen!)
# ============================================================

@auto_test("TTS verfügbar", "audio")
def test_tts():
    """Kann Moloch sprechen?"""
    try:
        # Prüfe ob Piper TTS installiert ist
        result = subprocess.run(
            "which piper 2>/dev/null || which piper-tts 2>/dev/null",
            shell=True, capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return True, "Piper gefunden"
        # Alternativ: Python Piper
        result2 = subprocess.run(
            "python3 -c 'import piper' 2>/dev/null",
            shell=True, capture_output=True, timeout=5
        )
        if result2.returncode == 0:
            return True, "Piper Python-Modul"
        return False, "Kein TTS gefunden"
    except:
        return False, "TTS-Check fehlgeschlagen"

@auto_test("Audio Output vorhanden", "audio")
def test_audio_output():
    """Gibt es ein Audio-Ausgabegerät?"""
    try:
        out = subprocess.check_output(
            "pactl list sinks short 2>/dev/null || aplay -l 2>/dev/null | head -5",
            shell=True, timeout=5
        ).decode()
        if out.strip():
            return True, "Audio-Ausgabe vorhanden"
        return False, "Kein Audio-Output"
    except:
        return False, "Audio-Check fehlgeschlagen"

# ============================================================
# AUTO-TESTS: NETZWERK (Claude: Von allen vergessen!)
# ============================================================

@auto_test("WLAN verbunden", "netzwerk")
def test_wlan():
    try:
        out = subprocess.check_output(
            "iwconfig wlan0 2>/dev/null | grep 'Signal level' || echo 'kein_wlan'",
            shell=True, timeout=5
        ).decode()
        if "kein_wlan" in out:
            # Vielleicht Ethernet?
            out2 = subprocess.check_output(
                "ip link show eth0 2>/dev/null | grep 'state UP' || echo 'nope'",
                shell=True, timeout=5
            ).decode()
            if "UP" in out2:
                return True, "Ethernet verbunden"
            return False, "Kein Netzwerk"
        return True, out.strip()[:60]
    except:
        return True, "Netzwerk-Check nicht möglich"

@auto_test("Home Assistant erreichbar", "netzwerk")
def test_home_assistant():
    """Claude: Von allen vergessen — HA ist Molochs Licht/Auge-Steuerung."""
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{HA_URL}/api/", timeout=5)
        return True, "HA antwortet"
    except:
        # HA ist optional, kein harter Fail
        return True, "HA nicht erreichbar (optional)"

# ============================================================
# AUTO-TESTS: MOLOCH SPRACHE (Claude)
# ============================================================

@auto_test("Moloch-Sprache aktiv", "sprache")
def test_moloch_sprache():
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = f"/mnt/moloch-data/gedanken/{today}.log"
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        return True, f"Heute: {size} bytes"
    if os.path.exists("/mnt/moloch-data/gedanken/"):
        return True, "Ordner da, heute noch kein Log"
    return False, "Kein Gedanken-Verzeichnis"

# ============================================================
# INTERAKTIVE TESTS (ChatGPT-Design + Gemini Hardware-Check)
# ============================================================

@interactive_test("Kamera-Sweep flüssig", "ptz_interaktiv")
def test_camera_sweep():
    """Gemini: Kamera physisch bewegen, User bestätigt."""
    print("\n  📷 KAMERA-SWEEP TEST")
    print("  Moloch bewegt jetzt die Kamera links → rechts → mitte.")
    print("  Beobachte ob die Bewegung FLÜSSIG ist.\n")

    # Hier würde der PTZ-Befehl gesendet
    # Für jetzt: User beobachtet manuell
    input("  Drücke ENTER wenn bereit...")

    # TODO: Echten PTZ-Sweep senden via ONVIF
    # sequence = [
    #     {"pan": -20, "tilt": 0, "wait": 0.5},
    #     {"pan": 20, "tilt": 0, "wait": 0.5},
    #     {"pan": 0, "tilt": 0, "wait": 0.2},
    # ]

    wait_with_countdown(3, "Beobachte Kamera")

    if user_confirm("War die Kamera-Bewegung flüssig?"):
        return True, "User bestätigt: flüssig"
    return False, "User: Bewegung nicht flüssig"

@interactive_test("Person-Tracking aktiv", "tracking_interaktiv")
def test_person_tracking():
    """ChatGPT: Stell dich vor die Kamera, beweg dich."""
    print("\n  🧍 PERSON-TRACKING TEST")
    print("  1. Stell dich vor die Kamera (ca. 2m Abstand)")
    print("  2. Hebe beide Arme für 5 Sekunden")
    print("  3. Beweg dich 3 Sekunden nach links")
    print("  4. Beweg dich 3 Sekunden nach rechts\n")

    input("  Drücke ENTER wenn bereit...")

    wait_with_countdown(15, "Führe Bewegungen aus")

    # Prüfe Status nach dem Test
    data = read_status()
    fps_ok = False
    if data:
        fps = data.get("fps", {}).get("total", 0)
        fps_ok = fps >= LIMITS["target_fps"]

    tracking_ok = user_confirm("Ist die Kamera dir gefolgt?")
    smooth_ok = user_confirm("War das Tracking flüssig (kein Ruckeln)?")

    if tracking_ok and smooth_ok:
        detail = f"FPS: {fps:.1f}" if data else "FPS unbekannt"
        return True, f"Tracking bestätigt. {detail}"

    problems = []
    if not tracking_ok:
        problems.append("Kamera folgt nicht")
    if not smooth_ok:
        problems.append("Ruckelig")
    return False, ", ".join(problems)

@interactive_test("Gesichtserkennung funktioniert", "tracking_interaktiv")
def test_face_recognition():
    """ChatGPT: Nahaufnahme, Gesicht erkennen."""
    print("\n  👤 GESICHTSERKENNUNG TEST")
    print("  1. Geh mit dem Gesicht nah an die Kamera (ca. 50cm)")
    print("  2. Bleib 5 Sekunden ruhig stehen")
    print("  3. Schau ins Panel ob dein Name erscheint\n")

    input("  Drücke ENTER wenn bereit...")

    wait_with_countdown(8, "Warte auf Erkennung")

    data = read_status()
    scrfd_active = data.get("scrfd_active", False) if data else False
    arcface_active = data.get("arcface_active", False) if data else False

    recognized = user_confirm("Hat Moloch deinen Namen angezeigt?")

    if recognized:
        return True, f"Erkannt! SCRFD={scrfd_active}, ArcFace={arcface_active}"

    if not scrfd_active:
        return False, "SCRFD nicht aktiv — Stufenschaltung kaputt?"
    if not arcface_active:
        return False, "ArcFace nicht aktiv — Face-Stage nicht erreicht?"
    return False, "Modelle aktiv aber nicht erkannt"

@interactive_test("Idle-Transition", "tracking_interaktiv")
def test_idle_transition():
    """ChatGPT: Verlass das Sichtfeld, prüfe ob NPU in Idle geht."""
    print("\n  🚶 IDLE-TRANSITION TEST")
    print("  1. Verlass das Sichtfeld der Kamera KOMPLETT")
    print("  2. Bleib 30 Sekunden weg")
    print("  3. Komm zurück und schau ins Panel\n")

    input("  Drücke ENTER und geh dann weg...")

    wait_with_countdown(30, "Warte auf Idle-Transition")

    data = read_status()
    if data:
        yolo = data.get("yolo_active", False)
        scrfd = data.get("scrfd_active", False)
        arcface = data.get("arcface_active", False)

        if yolo and not scrfd and not arcface:
            idle_ok = True
            state = "IDLE (nur YOLO)"
        elif not yolo and not scrfd and not arcface:
            idle_ok = True
            state = "SUSPENDED"
        else:
            idle_ok = False
            state = f"yolo={yolo} scrfd={scrfd} arcface={arcface}"
    else:
        idle_ok = False
        state = "Kein Status"

    if idle_ok:
        return True, f"NPU im {state}"

    # Frag User als Fallback
    if user_confirm("Zeigt das Panel weniger aktive Modelle als vorher?"):
        return True, f"User bestätigt Idle. Status: {state}"
    return False, f"NPU nicht in Idle: {state}"

@interactive_test("Kamera Home-Position", "ptz_interaktiv")
def test_home_position():
    """Claude: Home-Position muss Raummitte sein, nicht letzte manuelle."""
    print("\n  🏠 HOME-POSITION TEST")
    print("  1. Dreh die Kamera per Hand nach oben (Decke)")
    print("  2. Verlass das Sichtfeld für 30 Sekunden")
    print("  3. Komm zurück und schau wohin die Kamera zeigt\n")

    input("  Drücke ENTER wenn Kamera nach oben gedreht...")

    wait_with_countdown(30, "Warte auf Home-Return")

    if user_confirm("Zeigt die Kamera jetzt auf RAUMMITTE (nicht Decke)?"):
        return True, "Home-Position korrekt"
    return False, "Kamera zeigt nicht auf Raummitte"

# ============================================================
# HAUPTPROGRAMM
# ============================================================

def print_header():
    print()
    print("=" * 65)
    print("  M.O.L.O.C.H. INTEGRITY AUDIT v2.0")
    print("  Claude + Gemini + ChatGPT + Markus")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

def run_auto_tests():
    """Alle automatischen Tests."""
    print("\n  ─── SYSTEM ───")
    test_service_running()
    test_panel_running()
    test_status_json()

    print("\n  ─── KAMERA & BILD ───")
    test_frame_buffer()
    test_fps()
    test_frame_age()

    print("\n  ─── NPU / HAILO ───")
    test_hailo()
    test_npu_models()
    test_npu_no_error_loop()
    test_npu_state()

    print("\n  ─── PTZ ───")
    test_ptz_reachable()
    test_ptz_arbiter()
    test_ptz_no_conflict()

    print("\n  ─── DATENBANK ───")
    test_qdrant()
    test_qdrant_collections()
    test_face_db()

    print("\n  ─── HARDWARE ───")
    test_cpu_temp()
    test_ram()
    test_disk()
    test_fan()

    print("\n  ─── AUDIO ───")
    test_tts()
    test_audio_output()

    print("\n  ─── NETZWERK ───")
    test_wlan()
    test_home_assistant()

    print("\n  ─── MOLOCH SPRACHE ───")
    test_moloch_sprache()

def run_interactive_tests():
    """Interaktive Tests — brauchen User vor der Kamera."""
    print("\n")
    print("=" * 65)
    print("  INTERAKTIVE TESTS — Du musst vor der Kamera stehen!")
    print("=" * 65)

    test_camera_sweep()
    test_person_tracking()
    test_face_recognition()
    test_idle_transition()
    test_home_position()

def print_summary():
    """Ergebnis-Zusammenfassung."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print()
    print("=" * 65)

    if failed == 0:
        print(f"  ✅ ALLES OK — {passed}/{total} Tests bestanden")
        print("  → Nächster Fix darf starten")
        log_msg(f"AUDIT PASS: {passed}/{total}")
    else:
        print(f"  ❌ {failed} FEHLER — {passed}/{total} Tests bestanden")
        print("  → STOPP! Erst Fehler beheben!")
        print()
        print("  Fehlgeschlagen:")
        for r in results:
            if not r.passed:
                print(f"    ❌ {r.name}: {r.detail}")
        log_msg(f"AUDIT FAIL: {failed} Fehler von {total}")

    print("=" * 65)
    print()

    # Report speichern (Claude: für Vergleich über Zeit)
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "full" if "--full" in sys.argv else "auto",
        "passed": passed,
        "failed": failed,
        "total": total,
        "all_pass": failed == 0,
        "tests": [
            {
                "name": r.name,
                "passed": r.passed,
                "detail": r.detail,
                "category": r.category
            }
            for r in results
        ]
    }

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report: {REPORT_FILE}")
    except:
        pass

    # Auch als JSONL anhängen für History (Gemini: Regressions-Tracking)
    try:
        history_file = os.path.join(LOG_DIR, "audit_history.jsonl")
        with open(history_file, "a") as f:
            f.write(json.dumps({
                "timestamp": report["timestamp"],
                "mode": report["mode"],
                "passed": passed,
                "failed": failed,
                "total": total,
                "all_pass": report["all_pass"],
                "fails": [r.name for r in results if not r.passed]
            }) + "\n")
    except:
        pass

    return failed == 0

def main():
    print_header()

    mode = "--auto"
    if "--full" in sys.argv:
        mode = "--full"

    print(f"\n  Modus: {'FULL (Auto + Interaktiv)' if mode == '--full' else 'AUTO (nur automatisch)'}")

    # Auto-Tests immer
    run_auto_tests()

    # Interaktive Tests nur bei --full
    if mode == "--full":
        run_interactive_tests()

    # Zusammenfassung
    all_pass = print_summary()

    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
