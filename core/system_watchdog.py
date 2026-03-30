#!/usr/bin/env python3
"""
M.O.L.O.C.H. System-Watchdog — Zentraler Gesundheitsmonitor
=============================================================

Ersetzt fragmentierte Einzel-Watchdogs durch EINEN koordinierten Thread.
Prüft alle 5 Sekunden:
  1. Frame-Freeze (SHM-Timestamp)
  2. TAPPAS-Pipeline (_running Flag)
  3. ONVIF/PTZ Erreichbarkeit (Fehler-Counter)
  4. CPU-Temperatur + RAM
  5. hailo-ollama Verfügbarkeit

REGELN (aus System Contract):
  - NIEMALS Vision-Modelle entladen (→ Crash)
  - NIEMALS GStreamer-Pipeline stoppen (außer bei echtem Crash)
  - Watchdog selbst macht KEINEN NPU-Zugriff
  - Alle Logs auf INFO/WARNING Level

Singleton: get_watchdog()
"""

import logging
import os
import struct
import threading
import time
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger("MolochWatchdog")

# SHM-Frame-Pfad (gleich wie in tappas_pipeline.py)
SHM_FRAME_PATH = "/dev/shm/moloch_frame"

# Schwellwerte
FRAME_FREEZE_TIMEOUT = 30.0    # Sekunden ohne Frame → Pipeline-Neustart
ONVIF_FAIL_THRESHOLD = 5       # Fehler in Folge → ONVIF-Reconnect
CPU_TEMP_WARN = 70.0           # °C — Warnung
CPU_TEMP_CRITICAL = 80.0       # °C — LLM stoppen, Loop drosseln
RAM_WARN_PERCENT = 85.0        # % — Warnung
RAM_CRITICAL_PERCENT = 92.0    # % — LLM stoppen
CHECK_INTERVAL = 5.0           # Sekunden zwischen Checks
ONVIF_RECONNECT_COOLDOWN = 30.0  # Sekunden nach Reconnect-Versuch


class MolochWatchdog:
    """Zentraler System-Watchdog fuer M.O.L.O.C.H."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Externe Referenzen (werden via configure() gesetzt)
        self._inference = None          # TappasPipeline
        self._camera = None             # SonoffCameraController
        self._camera_manager = None     # CameraManager
        self._llm_bridge = None         # LocalLLMBridge

        # Callbacks fuer Reaktionen
        self._on_pipeline_restart: Optional[Callable] = None
        self._on_onvif_reconnect: Optional[Callable] = None
        self._on_throttle: Optional[Callable] = None    # (throttle: bool) → Loop drosseln
        self._on_llm_pause: Optional[Callable] = None   # (pause: bool) → LLM stoppen

        # Interne Zaehler
        self._onvif_fail_count = 0
        self._onvif_last_reconnect = 0.0
        self._pipeline_restart_count = 0
        self._last_frame_ts = 0.0       # Letzter bekannter SHM-Timestamp (monotonic)
        self._throttled = False
        self._llm_paused = False

        # Status (fuer IPC/Status-JSON)
        self._last_check_time = 0.0
        self._last_cpu_temp = 0.0
        self._last_ram_percent = 0.0
        self._last_frame_age = 0.0
        self._warnings: list = []

    def configure(self, inference=None, camera=None, camera_manager=None,
                  llm_bridge=None, on_pipeline_restart=None,
                  on_onvif_reconnect=None, on_throttle=None, on_llm_pause=None):
        """Externe Referenzen und Callbacks setzen. Wird von moloch_service aufgerufen."""
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

    def start(self):
        """Watchdog-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="MolochWatchdog")
        self._thread.start()
        logger.info("[WATCHDOG] Gestartet (Intervall %.0fs)", CHECK_INTERVAL)

    def stop(self):
        """Watchdog-Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("[WATCHDOG] Gestoppt")

    def get_status(self) -> Dict[str, Any]:
        """Aktuellen Watchdog-Status fuer IPC/Status-JSON."""
        return {
            "running": self._running,
            "cpu_temp": self._last_cpu_temp,
            "ram_percent": self._last_ram_percent,
            "frame_age": self._last_frame_age,
            "onvif_fail_count": self._onvif_fail_count,
            "pipeline_restarts": self._pipeline_restart_count,
            "throttled": self._throttled,
            "llm_paused": self._llm_paused,
            "warnings": list(self._warnings[-5:]),  # Letzte 5 Warnungen
            "last_check": self._last_check_time,
        }

    def report_onvif_error(self):
        """Von camera.py/tracker aufgerufen wenn AbsoluteMove fehlschlaegt."""
        self._onvif_fail_count += 1

    def report_onvif_success(self):
        """Von camera.py/tracker aufgerufen bei erfolgreichem PTZ-Befehl."""
        self._onvif_fail_count = 0

    # =========================================================================
    # Haupt-Loop
    # =========================================================================

    def _watchdog_loop(self):
        """Zentraler Watchdog-Loop — prüft alle CHECK_INTERVAL Sekunden."""
        time.sleep(10)  # 10s warten damit System hochfahren kann
        logger.info("[WATCHDOG] Erster Check in 10s...")

        while self._running:
            try:
                self._last_check_time = time.monotonic()
                self._warnings.clear()

                # 1. Frame-Freeze (SHM-Timestamp)
                self._check_frame_freeze()

                # 2. TAPPAS-Pipeline (_running Flag)
                self._check_pipeline_running()

                # 3. ONVIF/PTZ Fehler-Counter
                self._check_onvif()

                # 4. CPU-Temperatur + RAM
                self._check_resources()

                # 5. LLM-Verfuegbarkeit (optional, nur wenn konfiguriert)
                self._check_llm()

            except Exception as e:
                logger.error(f"[WATCHDOG] Fehler im Check-Loop: {e}")

            time.sleep(CHECK_INTERVAL)

    # =========================================================================
    # Einzelne Checks
    # =========================================================================

    def _check_frame_freeze(self):
        """Prüft ob SHM-Frame aktuell ist. Bei >30s → Pipeline-Neustart."""
        frame_age = self._read_shm_frame_age()
        self._last_frame_age = frame_age

        if frame_age < 0:
            # SHM nicht lesbar — kein Alarm, Pipeline evtl. noch nicht gestartet
            return

        if frame_age > FRAME_FREEZE_TIMEOUT:
            msg = f"Frame-Freeze: {frame_age:.0f}s seit letztem Frame"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)

            if self._on_pipeline_restart:
                logger.warning("[WATCHDOG] → Triggere Pipeline-Neustart")
                self._pipeline_restart_count += 1
                try:
                    self._on_pipeline_restart()
                except Exception as e:
                    logger.error(f"[WATCHDOG] Pipeline-Neustart fehlgeschlagen: {e}")

    def _check_pipeline_running(self):
        """Prüft ob TAPPAS-Pipeline noch lebt."""
        if self._inference is None:
            return
        if not getattr(self._inference, '_running', True):
            msg = "TAPPAS-Pipeline ist nicht mehr aktiv (_running=False)"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            # Nicht doppelt neustarten — _check_frame_freeze macht das bereits
            # TappasWatchdog in moloch_service.py handelt diesen Fall auch

    def _check_onvif(self):
        """Prüft ONVIF-Fehler-Counter. Bei >N Fehlern → Reconnect."""
        if self._onvif_fail_count < ONVIF_FAIL_THRESHOLD:
            return

        now = time.monotonic()
        if now - self._onvif_last_reconnect < ONVIF_RECONNECT_COOLDOWN:
            return  # Cooldown nach letztem Reconnect-Versuch

        msg = f"ONVIF: {self._onvif_fail_count} Fehler in Folge — Reconnect"
        logger.warning(f"[WATCHDOG] {msg}")
        self._warnings.append(msg)
        self._onvif_last_reconnect = now

        if self._on_onvif_reconnect:
            try:
                self._on_onvif_reconnect()
                self._onvif_fail_count = 0
                logger.info("[WATCHDOG] ONVIF-Reconnect erfolgreich")
            except Exception as e:
                logger.error(f"[WATCHDOG] ONVIF-Reconnect fehlgeschlagen: {e}")

    def _check_resources(self):
        """Prüft CPU-Temperatur und RAM-Auslastung."""
        cpu_temp = self._read_cpu_temp()
        ram_percent = self._read_ram_percent()
        self._last_cpu_temp = cpu_temp
        self._last_ram_percent = ram_percent

        # CPU-Temperatur
        if cpu_temp >= CPU_TEMP_CRITICAL:
            msg = f"CPU KRITISCH: {cpu_temp:.1f}°C — LLM pausiert, Loop gedrosselt"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            self._activate_throttle(True)
            self._pause_llm(True)
        elif cpu_temp >= CPU_TEMP_WARN:
            msg = f"CPU warm: {cpu_temp:.1f}°C"
            logger.info(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
        elif self._throttled and cpu_temp < CPU_TEMP_WARN - 5:
            # Hysterese: erst 5°C unter Warnschwelle wieder normal
            self._activate_throttle(False)

        # RAM
        if ram_percent >= RAM_CRITICAL_PERCENT:
            msg = f"RAM KRITISCH: {ram_percent:.1f}% — LLM pausiert"
            logger.warning(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
            self._pause_llm(True)
        elif ram_percent >= RAM_WARN_PERCENT:
            msg = f"RAM hoch: {ram_percent:.1f}%"
            logger.info(f"[WATCHDOG] {msg}")
            self._warnings.append(msg)
        elif self._llm_paused and ram_percent < RAM_WARN_PERCENT - 5:
            # Hysterese: erst 5% unter Warnschwelle wieder freigeben
            self._pause_llm(False)

    def _check_llm(self):
        """Prüft ob hailo-ollama erreichbar ist (wenn konfiguriert)."""
        if self._llm_bridge is None:
            return
        if not getattr(self._llm_bridge, '_ollama_available', False):
            return  # Binary nicht installiert — nichts zu prüfen

        running = getattr(self._llm_bridge, '_is_ollama_running', lambda: False)()
        if not running:
            # Nur loggen, kein Alarm — Cloud-Fallback greift automatisch
            pass  # Kein Spam bei dauerhaft ausgeschaltetem hailo-ollama

    # =========================================================================
    # Reaktionen (rufen Callbacks auf)
    # =========================================================================

    def _activate_throttle(self, throttle: bool):
        """Perception-Loop drosseln (2Hz) oder wiederherstellen (5Hz)."""
        if throttle == self._throttled:
            return
        self._throttled = throttle
        if self._on_throttle:
            try:
                self._on_throttle(throttle)
                logger.info(f"[WATCHDOG] Perception-Loop {'gedrosselt (2Hz)' if throttle else 'normal (5Hz)'}")
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
        """Frame-Age aus SHM-Header lesen. Gibt Sekunden zurueck, -1 bei Fehler."""
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
        """RAM-Auslastung aus /proc/meminfo lesen."""
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = int(parts[1].strip().split()[0])
                        meminfo[key] = val
            total = meminfo.get("MemTotal", 1)
            available = meminfo.get("MemAvailable", total)
            return round((1.0 - available / total) * 100.0, 1)
        except Exception:
            return 0.0


# =========================================================================
# Singleton
# =========================================================================

_instance: Optional[MolochWatchdog] = None

def get_watchdog() -> MolochWatchdog:
    """Globale MolochWatchdog-Instanz."""
    global _instance
    if _instance is None:
        _instance = MolochWatchdog()
    return _instance
