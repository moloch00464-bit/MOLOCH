#!/usr/bin/env python3
"""
M.O.L.O.C.H. System Watchdog — Nervensystem
============================================

Überwacht alle Moloch-Subsysteme. Schmerz-Signale fließen über
CoreIntegrator in Persönlichkeit und Verhalten — wenn etwas schiefläuft,
spürt Moloch es (Tension-Anstieg, Zone-Shift, TTS).

Checks (alle 3s):
  1. Frame-Freeze        (SHM-Timestamp > FRAME_FREEZE_TIMEOUT)
  2. TAPPAS-Pipeline     (_running Flag)
  3. ONVIF/PTZ           (Fehler-Counter via report_onvif_error/success())
  4. CPU-Temperatur      (> CPU_TEMP_WARN / CPU_TEMP_CRITICAL)
  5. RAM                 (> RAM_WARN_PERCENT / RAM_CRITICAL_PERCENT)
  6. Disk SSD1 + SSD2    (> DISK_WARN_PERCENT)
  7. Mikrofon            (ReSpeaker WiFi TCP 10.42.0.2:80)
  8. Heimnetz            (Router-Ping 192.168.178.1)
  9. hailo-ollama        (HTTP localhost:8000)

CoreIntegrator-Keys (registriert in core_integrator.py):
  hardware_pain  (Tension +0.7) — akut: Pipeline, ONVIF, Mic, Netz, LLM
  system_stress  (Tension +0.3) — chronisch: Temp, RAM, Disk

REGELN:
  - NIEMALS Vision-Modelle entladen (→ Crash)
  - NIEMALS GStreamer-Pipeline stoppen (außer bei echtem Crash)
  - Watchdog selbst macht KEINEN NPU-Zugriff
  - TTS mit Cooldown (kein Spam)

Singleton: get_watchdog()
Integration:
    watchdog = get_watchdog()
    watchdog.set_core_integrator(self._core_integrator)
    watchdog.set_speak_callback(self._voice_pipeline._speak)
    watchdog.configure(inference=..., camera=...)
    watchdog.start()
"""

import logging
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("MolochWatchdog")

# SHM-Frame-Pfad (identisch mit tappas_pipeline.py)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"

# --- Schwellwerte ---
FRAME_FREEZE_TIMEOUT   = 8.0    # Sekunden ohne Frame → Pipeline-Neustart
ONVIF_FAIL_THRESHOLD   = 5      # Fehler in Folge → ONVIF-Reconnect
CPU_TEMP_WARN          = 70.0   # °C — Warnung
CPU_TEMP_CRITICAL      = 80.0   # °C — LLM stoppen, Loop drosseln
RAM_WARN_PERCENT       = 85.0   # % — Warnung
RAM_CRITICAL_PERCENT   = 92.0   # % — LLM stoppen
DISK_WARN_PERCENT      = 90.0   # % — Platzmangel
CHECK_INTERVAL         = 3.0    # Sekunden zwischen Checks
ONVIF_RECONNECT_COOLDOWN  = 30.0  # Sekunden nach Reconnect-Versuch
OLLAMA_STARTUP_GRACE_S    = 60.0  # Sekunden Anlaufzeit — kein llm_dead-Pain
PIPELINE_RESTART_COOLDOWN = 30.0  # Sekunden zwischen Pipeline-Restart-Versuchen
CAMERA_CHECK_INTERVAL     = 15.0  # Cache-TTL fuer Kamera-TCP-Check (Sekunden)
CAMERA_IP                 = "192.168.178.25"  # Sonoff Kamera RTSP-IP


class MolochWatchdog:
    """Zentraler System-Watchdog — Nervensystem von M.O.L.O.C.H."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # --- Externe Referenzen (via configure() gesetzt) ---
        self._inference = None          # TappasPipeline
        self._camera = None             # SonoffCameraController
        self._camera_manager = None     # CameraManager
        self._llm_bridge = None         # LocalLLMBridge

        # --- Nervensystem: CoreIntegrator + TTS ---
        self._core = None               # CoreIntegrator (via set_core_integrator)
        self._speak_cb = None           # TTS-Callback   (via set_speak_callback)

        # Schmerz-Level pro Event: {event_type: severity 0.0–1.0}
        self._pain_levels: Dict[str, float] = {}
        self._pain_lock = threading.Lock()

        # TTS-Cooldowns: {event_type: letzter monotonic-Zeitpunkt}
        self._last_spoken: Dict[str, float] = {}

        # --- Callbacks für technische Reaktionen ---
        self._on_pipeline_restart: Optional[Callable] = None
        self._on_onvif_reconnect:  Optional[Callable] = None
        self._on_throttle:         Optional[Callable] = None
        self._on_llm_pause:        Optional[Callable] = None

        # --- Zähler (thread-safe) ---
        self._onvif_fail_count = 0
        self._onvif_lock = threading.Lock()
        self._onvif_last_reconnect = 0.0
        self._pipeline_restart_count = 0
        self._last_pipeline_restart  = 0.0   # Letzter Restart-Versuch (monotonic)
        self._pipeline_stopped_since = 0.0   # 0.0 = laeuft, >0 = gestoppt seit (monotonic)
        self._camera_reachable       = True  # Letzter bekannter Kamera-Status
        self._last_camera_check      = 0.0   # Zeitpunkt letzter TCP-Check (monotonic)
        self._throttled = False
        self._llm_paused = False

        # --- Status-Snapshot für IPC/Audit ---
        self._last_check_time = 0.0
        self._last_cpu_temp = 0.0
        self._last_ram_percent = 0.0
        self._last_frame_age = 0.0
        self._warnings: list = []

        # Startzeit — für Anlauf-Gnadenfrist bei llm_dead
        self._start_time = time.monotonic()

    # =========================================================================
    # Dependency Injection
    # =========================================================================

    def set_core_integrator(self, core):
        """CoreIntegrator setzen — Schmerz-Signale werden darüber eingespeist."""
        self._core = core

    def set_speak_callback(self, cb):
        """TTS-Callback setzen (später durch Moloch selbst generiert)."""
        self._speak_cb = cb

    def configure(self, inference=None, camera=None, camera_manager=None,
                  llm_bridge=None, on_pipeline_restart=None,
                  on_onvif_reconnect=None, on_throttle=None, on_llm_pause=None):
        """Externe Referenzen und Callbacks setzen."""
        if inference is not None:
            self._inference = inference
        if camera is not None:
            self._camera = camera
        if camera_manager is not None:
            self._camera_manager = camera_manager
        if llm_bridge is not None:
            self._llm_bridge = llm_bridge
        if on_pipeline_restart is not None:
            self._on_pipeline_restart = on_pipeline_restart
        if on_onvif_reconnect is not None:
            self._on_onvif_reconnect = on_onvif_reconnect
        if on_throttle is not None:
            self._on_throttle = on_throttle
        if on_llm_pause is not None:
            self._on_llm_pause = on_llm_pause

    # =========================================================================
    # ONVIF API — wird von camera.py aufgerufen
    # =========================================================================

    def report_onvif_error(self):
        """Von camera.py aufgerufen wenn AbsoluteMove fehlschlägt."""
        with self._onvif_lock:
            self._onvif_fail_count += 1

    def report_onvif_success(self):
        """Von camera.py aufgerufen bei erfolgreichem PTZ-Befehl."""
        with self._onvif_lock:
            self._onvif_fail_count = 0

    # =========================================================================
    # Lebenszyklus
    # =========================================================================

    def start(self):
        """Watchdog-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="MolochWatchdog"
        )
        self._thread.start()
        logger.info("[WATCHDOG] Nervensystem gestartet (Intervall %.0fs)", CHECK_INTERVAL)

    def stop(self):
        """Watchdog-Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("[WATCHDOG] Gestoppt")

    # =========================================================================
    # Haupt-Loop
    # =========================================================================

    def _watchdog_loop(self):
        """Zentraler Watchdog-Loop."""
        time.sleep(10)  # System erst stabilisieren lassen
        logger.info("[WATCHDOG] Erster Check...")

        while self._running:
            try:
                self._last_check_time = time.monotonic()
                self._warnings.clear()

                self._check_frame_freeze()
                self._check_pipeline_running()
                self._check_onvif()
                self._check_resources()
                self._check_disk()
                self._check_microphone()
                self._check_network()
                self._check_ollama()

                # Aggregierte Schmerz-Werte an CoreIntegrator senden
                self._update_core()

            except Exception as e:
                logger.error(f"[WATCHDOG] Fehler im Check-Loop: {e}")

            time.sleep(CHECK_INTERVAL)

    # =========================================================================
    # Einzelne Checks
    # =========================================================================

    def _check_frame_freeze(self):
        """Prüft ob SHM-Frame aktuell ist. Bei Freeze → Schmerz + Pipeline-Neustart."""
        frame_age = self._read_shm_frame_age()
        self._last_frame_age = frame_age

        if frame_age < 0:
            return  # SHM nicht lesbar — Pipeline noch nicht gestartet

        if frame_age > FRAME_FREEZE_TIMEOUT:
            msg = f"Frame-Freeze: {frame_age:.0f}s seit letztem Frame"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            self._set_pain("pipeline_freeze", 1.0,
                           "Meine Augen... ich sehe nichts mehr.", cooldown=300)
            now = time.monotonic()
            if (self._on_pipeline_restart
                    and now - self._last_pipeline_restart >= PIPELINE_RESTART_COOLDOWN):
                self._last_pipeline_restart = now
                self._pipeline_restart_count += 1
                try:
                    self._on_pipeline_restart()
                except Exception as e:
                    logger.error(f"[WATCHDOG] Pipeline-Neustart fehlgeschlagen: {e}")
        else:
            self._set_pain("pipeline_freeze", 0.0)

    def _check_pipeline_running(self):
        """Prüft ob TAPPAS-Pipeline noch lebt. Startet neu wenn Kamera wieder erreichbar."""
        if self._inference is None:
            return
        pipeline_ok = getattr(self._inference, '_running', True)
        if not pipeline_ok:
            now = time.monotonic()
            if self._pipeline_stopped_since == 0.0:
                self._pipeline_stopped_since = now
                logger.warning("[WATCHDOG] TAPPAS-Pipeline gestoppt — warte auf Kamera")
            stopped_for = now - self._pipeline_stopped_since
            self._warnings.append(f"TAPPAS gestoppt seit {stopped_for:.0f}s")
            self._set_pain("pipeline_dead", 0.9,
                           "Meine Augen... ich sehe nichts mehr.", cooldown=300)
            # Restart: nur wenn Cooldown abgelaufen UND Kamera TCP:554 erreichbar
            if (self._on_pipeline_restart
                    and now - self._last_pipeline_restart >= PIPELINE_RESTART_COOLDOWN
                    and self._is_camera_reachable()):
                logger.warning("[WATCHDOG] Kamera erreichbar — Pipeline-Restart nach "
                               f"{stopped_for:.0f}s Ausfall")
                self._last_pipeline_restart = now
                self._pipeline_restart_count += 1
                try:
                    self._on_pipeline_restart()
                except Exception as e:
                    logger.error(f"[WATCHDOG] Pipeline-Restart fehlgeschlagen: {e}")
        else:
            if self._pipeline_stopped_since > 0.0:
                logger.info("[WATCHDOG] Pipeline laeuft wieder")
                self._pipeline_stopped_since = 0.0
            self._set_pain("pipeline_dead", 0.0)

    def _check_onvif(self):
        """ONVIF-Fehler-Counter auswerten. Bei >N → Schmerz + Reconnect."""
        with self._onvif_lock:
            errors = self._onvif_fail_count

        if errors >= ONVIF_FAIL_THRESHOLD:
            self._set_pain("onvif_dead", 0.7,
                           "Ich kann die Kamera nicht mehr bewegen.", cooldown=300)
            now = time.monotonic()
            if now - self._onvif_last_reconnect >= ONVIF_RECONNECT_COOLDOWN:
                msg = f"ONVIF: {errors} Fehler — Reconnect"
                logger.warning(f"[WATCHDOG] {msg}")
                self._warnings.append(msg)
                self._onvif_last_reconnect = now
                if self._on_onvif_reconnect:
                    try:
                        self._on_onvif_reconnect()
                        with self._onvif_lock:
                            self._onvif_fail_count = 0
                        logger.info("[WATCHDOG] ONVIF-Reconnect erfolgreich")
                    except Exception as e:
                        logger.error(f"[WATCHDOG] ONVIF-Reconnect fehlgeschlagen: {e}")
        else:
            self._set_pain("onvif_dead", 0.0)

    def _check_resources(self):
        """CPU-Temperatur und RAM-Auslastung prüfen."""
        cpu_temp = self._read_cpu_temp()
        ram_percent = self._read_ram_percent()
        self._last_cpu_temp = cpu_temp
        self._last_ram_percent = ram_percent

        # CPU-Temperatur
        if cpu_temp >= CPU_TEMP_CRITICAL:
            msg = f"CPU KRITISCH: {cpu_temp:.1f}°C"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            severity = min(0.6, (cpu_temp - CPU_TEMP_WARN) / 15.0 * 0.6)
            self._set_pain("high_temp", severity, "Mir ist heiß.", cooldown=900)
            self._activate_throttle(True)
            self._pause_llm(True)
        elif cpu_temp >= CPU_TEMP_WARN:
            severity = min(0.4, (cpu_temp - CPU_TEMP_WARN) / 10.0 * 0.4)
            self._set_pain("high_temp", severity, "Mir ist heiß.", cooldown=900)
            self._warnings.append(f"CPU warm: {cpu_temp:.1f}°C")
        else:
            self._set_pain("high_temp", 0.0)
            if self._throttled and cpu_temp < CPU_TEMP_WARN - 5:
                self._activate_throttle(False)

        # RAM
        if ram_percent >= RAM_CRITICAL_PERCENT:
            msg = f"RAM KRITISCH: {ram_percent:.1f}%"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            self._set_pain("low_ram", 0.5, "Ich bin erschöpft.", cooldown=600)
            self._pause_llm(True)
        elif ram_percent >= RAM_WARN_PERCENT:
            self._set_pain("low_ram", 0.3, "Ich bin erschöpft.", cooldown=600)
            self._warnings.append(f"RAM hoch: {ram_percent:.1f}%")
        else:
            self._set_pain("low_ram", 0.0)
            if self._llm_paused and ram_percent < RAM_WARN_PERCENT - 5:
                self._pause_llm(False)

    def _check_disk(self):
        """Disk-Belegung beider SSDs prüfen."""
        for path, name in [("~/moloch", "SSD1"), ("/mnt/moloch-data", "SSD2")]:
            try:
                p = os.path.expanduser(path)
                if not os.path.exists(p):
                    continue
                usage = shutil.disk_usage(p)
                pct = usage.used / usage.total * 100.0
                key = f"disk_full_{name}"
                if pct > DISK_WARN_PERCENT:
                    self._set_pain(key, 0.5,
                                   f"Ich brauche mehr Platz — {name} ist fast voll.",
                                   cooldown=900)
                    self._warnings.append(f"Disk {name}: {pct:.0f}% belegt")
                else:
                    self._set_pain(key, 0.0)
            except Exception as e:
                logger.debug(f"[WATCHDOG] Disk-Check {name}: {e}")

    def _is_camera_reachable(self) -> bool:
        """TCP-Check auf RTSP-Port 554 der Kamera. Gecacht (CAMERA_CHECK_INTERVAL)."""
        now = time.monotonic()
        if now - self._last_camera_check >= CAMERA_CHECK_INTERVAL:
            self._last_camera_check = now
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                result = s.connect_ex((CAMERA_IP, 554))
                s.close()
                self._camera_reachable = (result == 0)
                if not self._camera_reachable:
                    logger.warning(f"[WATCHDOG] Kamera {CAMERA_IP}:554 nicht erreichbar")
            except Exception:
                self._camera_reachable = False
        return self._camera_reachable

    def _check_microphone(self):
        """ReSpeaker WiFi-Mic erreichbar? (HTTP Port 80 an 10.42.0.2)."""
        self._check_tcp("mic_dead", "10.42.0.2", 80,
                        "Ich höre nichts mehr.", cooldown=600)

    def _check_network(self):
        """Heimnetz erreichbar? Router-Ping."""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", "192.168.178.1"],
                capture_output=True, timeout=4
            )
            if result.returncode == 0:
                self._set_pain("network_dead", 0.0)
            else:
                self._set_pain("network_dead", 0.9,
                               "Ich bin isoliert.", cooldown=600)
        except Exception:
            self._set_pain("network_dead", 0.9,
                           "Ich bin isoliert.", cooldown=600)

    def _check_ollama(self):
        """hailo-ollama LLM-Service erreichbar? (Port 8000).

        Anlauf-Gnadenfrist: Kein Pain waehrend der ersten OLLAMA_STARTUP_GRACE_S
        Sekunden — hailo-ollama braucht ~30s bis es bereit ist.
        Danach: system_stress (Severity 0.4, ≤0.5) statt hardware_pain (0.7).
        """
        if time.monotonic() - self._start_time < OLLAMA_STARTUP_GRACE_S:
            return  # Anlauf — noch nicht pruefen
        try:
            urllib.request.urlopen("http://localhost:8000", timeout=2)
            self._set_pain("llm_dead", 0.0)
        except Exception:
            # 0.4 → faellt in system_stress-Bereich (≤0.5), kein hardware_pain-Spike
            self._set_pain("llm_dead", 0.4,
                           "Mein Verstand antwortet nicht mehr.", cooldown=600)

    # =========================================================================
    # Nervensystem — Schmerz-Signale
    # =========================================================================

    def _set_pain(self, event_type: str, severity: float,
                  message: str = None, cooldown: int = 300):
        """Schmerz-Level setzen. Loggt + löst TTS aus wenn neuer Schmerz."""
        with self._pain_lock:
            old = self._pain_levels.get(event_type, 0.0)
            self._pain_levels[event_type] = severity
            is_new_pain  = (old == 0.0 and severity > 0.0)
            is_recovered = (old > 0.0 and severity == 0.0)

        if is_new_pain:
            logger.warning(f"[WATCHDOG] Schmerz: {event_type} (Stärke={severity:.1f})")
            if message:
                self._speak_once(event_type, message, cooldown)
        elif is_recovered:
            logger.info(f"[WATCHDOG] Erholt: {event_type}")

    def _update_core(self):
        """Aggregierte Schmerz-Werte an CoreIntegrator senden.

        hardware_pain = Maximum aller akuten Schmerzen  (severity > 0.5)
        system_stress = Maximum aller chronischen Lasten (0 < severity ≤ 0.5)
        """
        if not self._core:
            return
        with self._pain_lock:
            levels = dict(self._pain_levels)

        hw_pain    = max((v for v in levels.values() if v > 0.5),         default=0.0)
        sys_stress = max((v for v in levels.values() if 0.0 < v <= 0.5),  default=0.0)

        self._core.update_inputs("watchdog", {
            "hardware_pain": hw_pain,
            "system_stress": sys_stress,
        })

    def _speak_once(self, event_type: str, text: str, cooldown: int = 300):
        """TTS nur wenn Cooldown abgelaufen und Callback vorhanden."""
        now = time.monotonic()
        if now - self._last_spoken.get(event_type, 0.0) > cooldown:
            if self._speak_cb:
                try:
                    self._speak_cb(text)
                except Exception as e:
                    logger.debug(f"[WATCHDOG] TTS-Fehler: {e}")
            self._last_spoken[event_type] = now

    def _check_tcp(self, event_type: str, host: str, port: int,
                   tts_text: str, cooldown: int = 600):
        """TCP-Verbindungstest mit 2s Timeout."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            result = s.connect_ex((host, port))
            s.close()
            if result == 0:
                self._set_pain(event_type, 0.0)
            else:
                self._set_pain(event_type, 0.8, tts_text, cooldown)
        except Exception:
            self._set_pain(event_type, 0.8, tts_text, cooldown)

    # =========================================================================
    # Technische Reaktionen (Callbacks)
    # =========================================================================

    def _activate_throttle(self, throttle: bool):
        """Perception-Loop drosseln (2Hz) oder wiederherstellen (5Hz)."""
        if throttle == self._throttled:
            return
        self._throttled = throttle
        if self._on_throttle:
            try:
                self._on_throttle(throttle)
                logger.info(f"[WATCHDOG] Loop "
                            f"{'gedrosselt (2Hz)' if throttle else 'normal (5Hz)'}")
            except Exception as e:
                logger.error(f"[WATCHDOG] Throttle-Callback Fehler: {e}")

    def _pause_llm(self, pause: bool):
        """LLM-Anfragen pausieren oder freigeben."""
        if pause == self._llm_paused:
            return
        self._llm_paused = pause
        if self._on_llm_pause:
            try:
                self._on_llm_pause(pause)
                logger.info(f"[WATCHDOG] LLM {'pausiert' if pause else 'freigegeben'}")
            except Exception as e:
                logger.error(f"[WATCHDOG] LLM-Pause-Callback Fehler: {e}")

    # =========================================================================
    # Hardware-Lese-Funktionen (kein NPU-Zugriff!)
    # =========================================================================

    @staticmethod
    def _read_shm_frame_age() -> float:
        """Frame-Age aus SHM-Header lesen. Gibt Sekunden zurück, -1 bei Fehler.

        SHM-Header: struct.pack('<IIIId', h, w, c, seq, ts)
        ts = float64 (time.monotonic()), Byte 16–23.
        """
        try:
            with open(SHM_FRAME_PATH, "rb") as f:
                header = f.read(24)
            if len(header) < 24:
                return -1.0
            _, _, _, _, ts = struct.unpack('<IIIId', header)
            if ts <= 0:
                return -1.0
            return round(time.monotonic() - ts, 1)
        except (OSError, struct.error):
            return -1.0

    @staticmethod
    def _read_cpu_temp() -> float:
        """CPU-Temperatur aus /sys lesen."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return 0.0

    @staticmethod
    def _read_ram_percent() -> float:
        """RAM-Auslastung aus /proc/meminfo lesen (kein psutil nötig)."""
        try:
            meminfo: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total     = meminfo.get("MemTotal", 1)
            available = meminfo.get("MemAvailable", total)
            return round((1.0 - available / total) * 100.0, 1)
        except Exception:
            return 0.0

    # =========================================================================
    # Status (für Audit + IPC)
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Aktuellen Watchdog-Zustand für Audit/IPC zurückgeben."""
        with self._pain_lock:
            active_pains = {k: round(v, 2) for k, v in self._pain_levels.items()
                            if v > 0.0}
        with self._onvif_lock:
            onvif_errors = self._onvif_fail_count
        return {
            "running":                 self._running,
            "cpu_temp":                self._last_cpu_temp,
            "ram_percent":             self._last_ram_percent,
            "frame_age":               self._last_frame_age,
            "onvif_consecutive_errors": onvif_errors,
            "pipeline_restarts":       self._pipeline_restart_count,
            "pipeline_stopped_since":  round(time.monotonic() - self._pipeline_stopped_since, 1)
                                       if self._pipeline_stopped_since > 0.0 else 0.0,
            "camera_reachable":        self._camera_reachable,
            "throttled":               self._throttled,
            "llm_paused":              self._llm_paused,
            "active_pains":            active_pains,
            "warnings":                list(self._warnings[-5:]),
            "last_check":              self._last_check_time,
        }


# =========================================================================
# Singleton
# =========================================================================

_instance: Optional[MolochWatchdog] = None
_instance_lock = threading.Lock()


def get_watchdog() -> MolochWatchdog:
    """Singleton — thread-safe. Wird auch von camera.py lazy importiert."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MolochWatchdog()
    return _instance
