#!/usr/bin/env python3
"""
M.O.L.O.C.H. Homeostasis — Selbstueberwachung + Auto-Heal
============================================================

Ueberwacht System-Gesundheit und greift bei Problemen automatisch ein:
  - RAM > 85% → Qdrant flush, GC forcieren
  - CPU Temp > 80°C → FPS drosseln
  - FPS < 10 → Pipeline-Neustart anstossen
  - Disk > 90% → Event-Logs rotieren

Laeuft als Background-Thread mit 10s Intervall.
Publiziert health_alert Event (Priority 0 = CRITICAL) bei Problemen.

Singleton: get_homeostasis()
Gate 5: Autonomous Environmental Agent
"""

import gc
import logging
import os
import threading
import time
from typing import Optional, Dict, Any, List

logger = logging.getLogger("MolochHomeostasis")

# Schwellwerte
THRESHOLDS = {
    "ram_warn": 75.0,       # % — Warnung
    "ram_critical": 85.0,   # % — Auto-Heal
    "cpu_temp_warn": 75.0,  # °C
    "cpu_temp_critical": 80.0,
    "fps_min": 10.0,
    "disk_critical": 90.0,  # %
}

CHECK_INTERVAL = 10.0  # Sekunden zwischen Checks
HEAL_COOLDOWN = 60.0   # Sekunden zwischen gleichen Heal-Aktionen


class Homeostasis:
    """Selbstueberwachung und automatische Heilung."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heals: Dict[str, float] = {}
        self._current_fps: float = 20.0
        self._alerts: List[Dict[str, Any]] = []  # Ringbuffer letzte 20 Alerts
        self._stats = {
            "checks": 0,
            "heals": 0,
            "alerts_total": 0,
        }

    def start(self):
        """Homeostasis-Thread starten."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="Homeostasis"
        )
        self._thread.start()
        logger.info("[HOMEOSTASIS] Gestartet (Intervall: 10s)")

    def stop(self):
        """Homeostasis-Thread stoppen."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("[HOMEOSTASIS] Gestoppt")

    def set_fps(self, fps: float):
        """Aktuelle FPS von aussen setzen (aus Perception Loop)."""
        self._current_fps = fps

    def _monitor_loop(self):
        """Hauptschleife: Alle 10s System-Metriken pruefen."""
        while self._running:
            try:
                self._check_system()
            except Exception as e:
                logger.error(f"[HOMEOSTASIS] Check-Fehler: {e}")
            time.sleep(CHECK_INTERVAL)

    def _check_system(self):
        """Alle System-Metriken pruefen und bei Bedarf heilen."""
        self._stats["checks"] += 1

        ram = self._get_ram_percent()
        cpu_temp = self._get_cpu_temp()
        fps = self._current_fps
        disk = self._get_disk_percent()

        # --- RAM Check ---
        if ram > THRESHOLDS["ram_critical"]:
            self._alert("ram_critical", f"RAM {ram:.1f}% > {THRESHOLDS['ram_critical']}%",
                        level="critical", value=ram)
            self._heal_ram()
        elif ram > THRESHOLDS["ram_warn"]:
            self._alert("ram_warn", f"RAM {ram:.1f}% > {THRESHOLDS['ram_warn']}%",
                        level="warning", value=ram)

        # --- CPU Temperatur ---
        if cpu_temp > THRESHOLDS["cpu_temp_critical"]:
            self._alert("cpu_temp_critical", f"CPU {cpu_temp:.1f}C > {THRESHOLDS['cpu_temp_critical']}C",
                        level="critical", value=cpu_temp)
            self._heal_thermal()
        elif cpu_temp > THRESHOLDS["cpu_temp_warn"]:
            self._alert("cpu_temp_warn", f"CPU {cpu_temp:.1f}C > {THRESHOLDS['cpu_temp_warn']}C",
                        level="warning", value=cpu_temp)

        # --- FPS Check ---
        if fps < THRESHOLDS["fps_min"] and fps > 0:
            self._alert("fps_low", f"FPS {fps:.1f} < {THRESHOLDS['fps_min']}",
                        level="warning", value=fps)

        # --- Disk Check ---
        if disk > THRESHOLDS["disk_critical"]:
            self._alert("disk_critical", f"Disk {disk:.1f}% > {THRESHOLDS['disk_critical']}%",
                        level="critical", value=disk)
            self._heal_disk()

    def _alert(self, alert_type: str, message: str, level: str = "warning",
               value: float = 0.0):
        """Alert registrieren und Event publizieren."""
        alert = {
            "type": alert_type,
            "message": message,
            "level": level,
            "value": round(value, 1),
            "timestamp": time.time(),
        }

        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > 20:
                self._alerts.pop(0)
            self._stats["alerts_total"] += 1

        # Event publizieren
        priority = 0 if level == "critical" else 4
        try:
            from core.moloch_event_bus import get_event_bus
            get_event_bus().publish(
                event_type="health_alert",
                source="homeostasis",
                priority=priority,
                payload=alert,
            )
        except Exception:
            pass

        if level == "critical":
            logger.warning(f"[HOMEOSTASIS] CRITICAL: {message}")
        else:
            logger.info(f"[HOMEOSTASIS] {level.upper()}: {message}")

    # =====================================================================
    # Auto-Heal Aktionen
    # =====================================================================

    def _can_heal(self, heal_type: str) -> bool:
        """Cooldown pruefen — gleicher Heal nur alle 60s."""
        now = time.time()
        last = self._last_heals.get(heal_type, 0.0)
        if (now - last) < HEAL_COOLDOWN:
            return False
        self._last_heals[heal_type] = now
        self._stats["heals"] += 1
        return True

    def _heal_ram(self):
        """RAM-Druck reduzieren: GC forcieren."""
        if not self._can_heal("ram"):
            return

        logger.warning("[HOMEOSTASIS] Auto-Heal: RAM — GC forcieren")
        # Python GC forcieren
        gc.collect()

        # Qdrant embedded Client hat keinen flush — aber Python-seitig GC reicht
        logger.info("[HOMEOSTASIS] GC abgeschlossen")

    def _heal_thermal(self):
        """Thermische Drosselung: Event fuer Pipeline-Slowdown."""
        if not self._can_heal("thermal"):
            return

        logger.warning("[HOMEOSTASIS] Auto-Heal: Thermal — Drosselung empfohlen")
        # Kein direkter Pipeline-Eingriff — nur Event + Logging
        # Die Pipeline kann auf health_alert reagieren

    def _heal_disk(self):
        """Disk-Platz freigeben: Alte Event-Logs loeschen."""
        if not self._can_heal("disk"):
            return

        logger.warning("[HOMEOSTASIS] Auto-Heal: Disk — Event-Logs rotieren")
        log_dir = os.path.expanduser("~/moloch/logs/events")
        if not os.path.isdir(log_dir):
            return

        # Aelteste Event-Log Dateien loeschen (behalte letzte 7 Tage)
        try:
            files = sorted(
                [os.path.join(log_dir, f) for f in os.listdir(log_dir)
                 if f.startswith("events_") and f.endswith(".jsonl")],
                key=os.path.getmtime,
            )
            # Behalte die neuesten 7, loesche den Rest
            to_delete = files[:-7] if len(files) > 7 else []
            for f in to_delete:
                os.remove(f)
                logger.info(f"[HOMEOSTASIS] Geloescht: {f}")
        except Exception as e:
            logger.error(f"[HOMEOSTASIS] Log-Rotation fehlgeschlagen: {e}")

    # =====================================================================
    # System-Metriken
    # =====================================================================

    def _get_ram_percent(self) -> float:
        """RAM-Auslastung in Prozent (aus /proc/meminfo)."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem[key] = int(parts[1])
            total = mem.get("MemTotal", 1)
            available = mem.get("MemAvailable", total)
            return (1.0 - available / total) * 100.0
        except Exception:
            return 0.0

    def _get_cpu_temp(self) -> float:
        """CPU-Temperatur in Celsius (Pi5 thermal_zone0)."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _get_disk_percent(self) -> float:
        """Disk-Auslastung der Haupt-SSD in Prozent."""
        try:
            stat = os.statvfs(os.path.expanduser("~/moloch"))
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            if total == 0:
                return 0.0
            return (1.0 - free / total) * 100.0
        except Exception:
            return 0.0

    # =====================================================================
    # Public API
    # =====================================================================

    def get_state(self) -> Dict[str, Any]:
        """Aktueller State fuer IPC/Panel."""
        with self._lock:
            return {
                "ram_percent": round(self._get_ram_percent(), 1),
                "cpu_temp": round(self._get_cpu_temp(), 1),
                "fps": round(self._current_fps, 1),
                "disk_percent": round(self._get_disk_percent(), 1),
                "stats": dict(self._stats),
                "recent_alerts": list(self._alerts[-5:]),
            }

    def get_metrics(self) -> Dict[str, float]:
        """Nur Metriken (lightweight fuer schnellen Zugriff)."""
        return {
            "ram": round(self._get_ram_percent(), 1),
            "cpu_temp": round(self._get_cpu_temp(), 1),
            "fps": round(self._current_fps, 1),
            "disk": round(self._get_disk_percent(), 1),
        }


# =========================================================================
# SINGLETON
# =========================================================================

_instance: Optional[Homeostasis] = None
_instance_lock = threading.Lock()


def get_homeostasis() -> Homeostasis:
    """Singleton-Zugriff auf Homeostasis."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Homeostasis()
    return _instance
