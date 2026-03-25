#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hardware Monitor Popup — Pi5 + Hailo-10H
=======================================================

Eigenstaendiges Toplevel-Fenster fuer Hardware-Monitoring.

Sektionen:
- CPU: Temperatur (Balken + Wert), Last (%), Frequenz, Luefter
- RAM: System + MOLOCH Service RSS (mit Balken)
- NPU: Hailo-10H Temperatur, RAM pro Modell (echte HEF-Groessen), FPS
- Storage: SSD1 (/) + SSD2 (/mnt/moloch-data) mit Balken
- Uptime: System-Laufzeit

Alle 5 Sekunden aktualisiert.
Balken: gruen <60%, gelb 60-80%, rot >80%.

Importiert NUR panel_styles und tkinter.
"""

import glob as globmod
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_hardware")

# psutil optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Update-Intervall
UPDATE_MS = 5000

# Canvas-Balken
BAR_WIDTH = 280
BAR_HEIGHT = 16


def _bar_color(percent):
    """Farbe nach Auslastung: gruen <60%, gelb 60-80%, rot >80%."""
    if percent < 60:
        return STATUS_GREEN
    elif percent < 80:
        return STATUS_YELLOW
    return STATUS_RED


class HardwarePopup:
    """Hardware Monitor als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy
        self._after_id = None

        # CPU-Last Fallback
        self._prev_idle = 0
        self._prev_total = 0

        # Toplevel
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Hardware Monitor \u2014 Pi5 + Hailo-10H")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("400x860")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # GUI aufbauen
        self._build_cpu_section()
        self._build_fans_section()
        self._build_ram_section()
        self._build_npu_section()
        self._build_ssd_section()
        self._build_uptime_section()

        # Erster Update
        self._update_all()

    # =========================================================================
    # CPU Section
    # =========================================================================

    def _build_cpu_section(self):
        """CPU: Temperatur mit Balken, Last, Frequenz, Luefter."""
        section = tk.LabelFrame(
            self.win, text="CPU",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Temperatur: Wert + Balken
        row_temp = tk.Frame(section, bg=BG_FRAME)
        row_temp.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row_temp, text="Temperatur:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_temp = tk.Label(row_temp, text="--", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_temp.pack(side=tk.RIGHT)

        self._canvas_temp = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_temp.pack(padx=8, pady=(0, 3))

        # Last
        row_load = tk.Frame(section, bg=BG_FRAME)
        row_load.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_load, text="CPU Last:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_cpu = tk.Label(row_load, text="--", bg=BG_FRAME,
                                 fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_cpu.pack(side=tk.RIGHT)

        # Frequenz
        row_freq = tk.Frame(section, bg=BG_FRAME)
        row_freq.pack(fill=tk.X, padx=8, pady=(2, 5))
        tk.Label(row_freq, text="Frequenz:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_freq = tk.Label(row_freq, text="--", bg=BG_FRAME,
                                  fg=FG_WHITE, font=FONT_MONO)
        self._lbl_freq.pack(side=tk.RIGHT)


    # =========================================================================
    # Kühlung Section (Noctua + CPU-Kühler)
    # =========================================================================

    def _build_fans_section(self):
        """Noctua NF-A4x20 (GPIO18 PWM) + Pi5 CPU-Kühler."""
        section = tk.LabelFrame(
            self.win, text="K\u00fchlung",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # --- Noctua NF-A4x20 ---
        tk.Label(section, text="Noctua NF-A4x20  (GPIO18 PWM-PIO)",
                 bg=BG_FRAME, fg=FG_WHITE, font=FONT_SMALL).pack(
                 anchor=tk.W, padx=8, pady=(5, 0))

        row_noctua = tk.Frame(section, bg=BG_FRAME)
        row_noctua.pack(fill=tk.X, padx=8, pady=(2, 0))
        tk.Label(row_noctua, text="PWM:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_noctua = tk.Label(row_noctua, text="-- %", bg=BG_FRAME,
                                    fg=STATUS_GREEN, font=FONT_MONO)
        self._lbl_noctua.pack(side=tk.RIGHT)

        self._canvas_noctua = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_noctua.pack(padx=8, pady=(0, 4))

        # --- Pi5 CPU-Kühler ---
        tk.Label(section, text="CPU-K\u00fchler  (Pi5 built-in)",
                 bg=BG_FRAME, fg=FG_WHITE, font=FONT_SMALL).pack(
                 anchor=tk.W, padx=8)

        row_cpufan = tk.Frame(section, bg=BG_FRAME)
        row_cpufan.pack(fill=tk.X, padx=8, pady=(2, 0))
        tk.Label(row_cpufan, text="Status:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_cpufan = tk.Label(row_cpufan, text="---", bg=BG_FRAME,
                                    fg=FG_DIM, font=FONT_MONO)
        self._lbl_cpufan.pack(side=tk.RIGHT)

        self._canvas_cpufan = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_cpufan.pack(padx=8, pady=(0, 5))

    # =========================================================================
    # RAM Section
    # =========================================================================

    def _build_ram_section(self):
        """RAM: System + MOLOCH Service RSS."""
        section = tk.LabelFrame(
            self.win, text="RAM \u2014 Pi5 (4 GB)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # System-RAM
        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row, text="System:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_ram = tk.Label(row, text="-- / -- MB", bg=BG_FRAME,
                                 fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ram.pack(side=tk.RIGHT)

        self._canvas_ram = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ram.pack(padx=8, pady=(0, 3))

        # MOLOCH Service RSS
        row_svc = tk.Frame(section, bg=BG_FRAME)
        row_svc.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_svc, text="MOLOCH:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_service_rss = tk.Label(row_svc, text="-- MB", bg=BG_FRAME,
                                         fg=FG_DIM, font=FONT_MONO)
        self._lbl_service_rss.pack(side=tk.RIGHT)

        # Threads
        row_thr = tk.Frame(section, bg=BG_FRAME)
        row_thr.pack(fill=tk.X, padx=8, pady=(0, 5))
        tk.Label(row_thr, text="Threads:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_threads = tk.Label(row_thr, text="--", bg=BG_FRAME,
                                     fg=FG_DIM, font=FONT_MONO)
        self._lbl_threads.pack(side=tk.RIGHT)

    # =========================================================================
    # NPU Section (erweitert mit RAM + Modell-FPS)
    # =========================================================================

    def _build_npu_section(self):
        """NPU: Temperatur, Status, RAM pro Modell, FPS."""
        section = tk.LabelFrame(
            self.win, text="NPU \u2014 Hailo-10H (8 GB LPDDR4)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # Status + Temperatur auf einer Zeile
        row_status = tk.Frame(section, bg=BG_FRAME)
        row_status.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row_status, text="Status:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_status = tk.Label(row_status, text="--", bg=BG_FRAME,
                                        fg=FG_DIM, font=FONT_MONO)
        self._lbl_npu_status.pack(side=tk.RIGHT)

        # NPU Temperatur
        row_npu_temp = tk.Frame(section, bg=BG_FRAME)
        row_npu_temp.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_npu_temp, text="NPU Temp:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_temp = tk.Label(row_npu_temp, text="--", bg=BG_FRAME,
                                      fg=FG_DIM, font=FONT_MONO)
        self._lbl_npu_temp.pack(side=tk.RIGHT)

        self._canvas_npu_temp = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_npu_temp.pack(padx=8, pady=(0, 3))

        # NPU RAM (echte HEF-Groessen)
        row_npu_ram = tk.Frame(section, bg=BG_FRAME)
        row_npu_ram.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_npu_ram, text="NPU RAM:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_ram = tk.Label(row_npu_ram, text="-- / 8192 MB", bg=BG_FRAME,
                                     fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_npu_ram.pack(side=tk.RIGHT)

        self._canvas_npu_ram = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_npu_ram.pack(padx=8, pady=(0, 3))

        # Modelle + FPS + RAM (mehrzeilig, tabellarisch)
        row_models = tk.Frame(section, bg=BG_FRAME)
        row_models.pack(fill=tk.X, padx=8, pady=(2, 5))
        tk.Label(row_models, text="Modelle:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT, anchor=tk.N)
        self._lbl_npu_models = tk.Label(
            row_models, text="--", bg=BG_FRAME,
            fg=FG_DIM, font=FONT_MONO,
            wraplength=280, justify=tk.LEFT, anchor=tk.W,
        )
        self._lbl_npu_models.pack(side=tk.LEFT, padx=(10, 0))

    # =========================================================================
    # SSD Section
    # =========================================================================

    def _build_ssd_section(self):
        """SSD: System (/) + Daten (/mnt/moloch-data)."""
        section = tk.LabelFrame(
            self.win, text="Storage",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # SSD 1
        tk.Label(section, text="System-SSD (/)", bg=BG_FRAME, fg=FG_WHITE,
                 font=FONT_SMALL).pack(anchor=tk.W, padx=8, pady=(5, 0))
        row_ssd1 = tk.Frame(section, bg=BG_FRAME)
        row_ssd1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_ssd1, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_ssd1 = tk.Label(row_ssd1, text="-- / -- GB", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ssd1.pack(side=tk.RIGHT)
        self._canvas_ssd1 = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ssd1.pack(padx=8, pady=(0, 5))

        # SSD 2
        tk.Label(section, text="Daten-SSD (/mnt/moloch-data)", bg=BG_FRAME,
                 fg=FG_WHITE, font=FONT_SMALL).pack(anchor=tk.W, padx=8)
        row_ssd2 = tk.Frame(section, bg=BG_FRAME)
        row_ssd2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(row_ssd2, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_ssd2 = tk.Label(row_ssd2, text="-- / -- GB", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ssd2.pack(side=tk.RIGHT)
        self._canvas_ssd2 = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ssd2.pack(padx=8, pady=(0, 5))

    # =========================================================================
    # Uptime Section
    # =========================================================================

    def _build_uptime_section(self):
        """System-Uptime."""
        section = tk.LabelFrame(
            self.win, text="System",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(5, 10))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=5)
        tk.Label(row, text="Uptime:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_uptime = tk.Label(row, text="--", bg=BG_FRAME,
                                    fg=FG_WHITE, font=FONT_MONO)
        self._lbl_uptime.pack(side=tk.RIGHT)

    # =========================================================================
    # Daten lesen
    # =========================================================================

    def _read_cpu_temp(self):
        """CPU-Temperatur via vcgencmd."""
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                text = result.stdout.strip()
                val = text.split("=")[1].replace("'C", "")
                return float(val)
        except Exception:
            pass
        return None

    def _read_cpu_percent(self):
        """CPU-Last in Prozent."""
        if HAS_PSUTIL:
            try:
                return psutil.cpu_percent(interval=0)
            except Exception:
                pass

        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
            parts = line.split()
            vals = [int(x) for x in parts[1:]]
            idle = vals[3]
            total = sum(vals)
            diff_idle = idle - self._prev_idle
            diff_total = total - self._prev_total
            self._prev_idle = idle
            self._prev_total = total
            if diff_total > 0:
                return (1.0 - diff_idle / diff_total) * 100.0
        except Exception:
            pass
        return None

    def _read_cpu_freq(self):
        """CPU-Frequenz in MHz."""
        try:
            path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            with open(path, "r") as f:
                khz = int(f.read().strip())
            return khz / 1000
        except Exception:
            pass
        return None

    def _read_noctua_pwm(self):
        """Noctua PWM-Prozent aus /sys/class/pwm lesen.
        Sucht den Chip mit npwm != 4 (nicht den Pi5 Built-in mit npwm=4)."""
        try:
            pwm_basis = "/sys/class/pwm"
            chips = sorted(os.listdir(pwm_basis))
            for chip in chips:
                chip_pfad = f"{pwm_basis}/{chip}"
                try:
                    with open(f"{chip_pfad}/npwm") as f:
                        if int(f.read().strip()) == 4:
                            continue  # Pi5 Built-in ueberspringen
                except OSError:
                    continue
                # Dieser Chip ist der Noctua-Chip (pwm-pio)
                duty_path = f"{chip_pfad}/pwm0/duty_cycle"
                period_path = f"{chip_pfad}/pwm0/period"
                if os.path.exists(duty_path) and os.path.exists(period_path):
                    with open(duty_path) as f:
                        duty = int(f.read().strip())
                    with open(period_path) as f:
                        period = int(f.read().strip())
                    if period > 0:
                        return round(duty / period * 100)
        except Exception:
            pass
        return None

    _FAN_STAGES = {
        0: ("AUS", FG_DIM),
        1: ("Leise", STATUS_GREEN),
        2: ("Mittel", STATUS_GREEN),
        3: ("Schnell", STATUS_YELLOW),
        4: ("Voll", STATUS_RED),
    }

    def _read_fan_state(self):
        """Fan PWM-Stufe und Prozent."""
        try:
            pwm_matches = globmod.glob(
                "/sys/devices/platform/cooling_fan/hwmon/hwmon*/pwm1")
            pwm_pct = None
            if pwm_matches:
                with open(pwm_matches[0], "r") as f:
                    pwm_val = int(f.read().strip())
                pwm_pct = round(pwm_val / 255 * 100)

            cur_state = None
            max_state = None
            cs_path = "/sys/class/thermal/cooling_device0/cur_state"
            ms_path = "/sys/class/thermal/cooling_device0/max_state"
            if os.path.exists(cs_path):
                with open(cs_path, "r") as f:
                    cur_state = int(f.read().strip())
            if os.path.exists(ms_path):
                with open(ms_path, "r") as f:
                    max_state = int(f.read().strip())

            if cur_state is not None:
                return cur_state, max_state, pwm_pct
        except Exception:
            pass
        return None, None, None

    def _read_ram(self):
        """RAM benutzt/gesamt in MB."""
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                return mem.used / (1024 * 1024), mem.total / (1024 * 1024)
            except Exception:
                pass

        try:
            info = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        info[key] = int(parts[1])
            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
            used_kb = total_kb - avail_kb
            return used_kb / 1024, total_kb / 1024
        except Exception:
            pass
        return None, None

    def _read_disk(self, path):
        """Disk benutzt/gesamt in GB."""
        try:
            usage = shutil.disk_usage(path)
            used_gb = usage.used / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            return used_gb, total_gb
        except Exception:
            pass
        return None, None

    # Echte HEF-Groessen (MB, berechnet aus Dateigroesse auf Disk)
    # Dateien auf /mnt/moloch-data/hailo/models/
    _HEF_FILES = {
        "scrfd": "scrfd_10g.hef",
        "arcface": "arcface_mobilefacenet.hef",
        "yolov8m": "yolov8m_h10.hef",
        "pose": "yolov8s_pose_h10.hef",
        "hand_landmark": "hand_landmark_lite.hef",
        "face_attr": "face_attr_resnet_v1_18.hef",
    }
    _HEF_DIR = "/mnt/moloch-data/hailo/models"
    _hef_size_cache = {}  # Einmal lesen, dann cachen

    def _get_hef_size_mb(self, model_name):
        """Echte HEF-Dateigroesse in MB (gecached)."""
        if model_name in self._hef_size_cache:
            return self._hef_size_cache[model_name]
        hef_file = self._HEF_FILES.get(model_name)
        if hef_file:
            path = os.path.join(self._HEF_DIR, hef_file)
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self._hef_size_cache[model_name] = round(size_mb, 1)
                return self._hef_size_cache[model_name]
            except OSError:
                pass
        # Fallback-Schaetzung
        fallback = {"scrfd": 6, "arcface": 3, "yolov8m": 21,
                     "pose": 14, "hand_landmark": 1, "face_attr": 7}
        return fallback.get(model_name, 5)

    def _read_npu_temperature(self):
        """NPU-Temperatur via HailoRT Device (NICHT VDevice!).

        Device.control.get_chip_temperature() funktioniert parallel zur
        TAPPAS-Pipeline — kein VDevice noetig, kein Error 74.
        Laeuft im Background-Thread, blockiert nicht den Main-Loop.

        Returns:
            float oder None bei Fehler.
        """
        try:
            from hailo_platform import Device
            d = Device()
            temp_info = d.control.get_chip_temperature()
            ts0 = temp_info.ts0_temperature
            ts1 = temp_info.ts1_temperature
            d.release()
            return round(max(ts0, ts1), 1)
        except Exception as e:
            logger.debug(f"NPU Temperatur nicht lesbar: {e}")
        return None

    _cached_service_pid = None

    def _read_service_rss(self):
        """MOLOCH Service RSS + Thread-Count aus /proc.

        Cached PID — iteriert /proc nur beim ersten Aufruf oder wenn PID ungueltig.
        Returns:
            (rss_mb, thread_count) oder (None, None)
        """
        # Schneller Pfad: gecachte PID pruefen
        pid = self._cached_service_pid
        if pid is not None:
            try:
                rss_kb, threads = self._read_pid_stats(pid)
                if rss_kb > 0:
                    return rss_kb / 1024, threads
            except (OSError, ValueError):
                pass
            # PID ungueltig — Cache loeschen
            self._cached_service_pid = None

        # Langsamer Pfad: PID suchen (einmalig)
        try:
            for pid_str in os.listdir("/proc"):
                if not pid_str.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                        cmdline = f.read().decode("utf-8", errors="replace")
                    if "moloch_service" not in cmdline and "MolochService" not in cmdline:
                        continue
                    pid = int(pid_str)
                    self._cached_service_pid = pid
                    rss_kb, threads = self._read_pid_stats(pid)
                    return rss_kb / 1024, threads
                except (OSError, ValueError, IndexError):
                    continue
        except Exception:
            pass
        return None, None

    def _read_pid_stats(self, pid):
        """RSS + Thread-Count fuer eine PID lesen."""
        rss_kb = 0
        threads = 0
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
        return rss_kb, threads

    def _read_npu_status(self):
        """NPU-Status: status, models + FPS + RAM, Gesamt-RAM.

        WICHTIG: active_models und npu_stage liegen auf TOP-LEVEL im Status-JSON,
        NICHT unter einem 'npu'-Subkey!

        Returns:
            (status_text, status_color, models_text, npu_ram_mb)
        """
        status = self.service.read_status()
        if status and isinstance(status, dict):
            # Top-Level Keys (so schreibt der Service den Status)
            npu_state = status.get("npu_stage",
                        status.get("npu_sched_mode", "unbekannt"))
            active = status.get("active_models", [])

            # FPS pro Modell
            fps_dict = status.get("fps", {})
            if not isinstance(fps_dict, dict):
                fps_dict = {}

            # Modelle + FPS + RAM tabellarisch
            model_lines = []
            npu_ram = 0.0
            if isinstance(active, list) and active:
                for m in active:
                    m_str = str(m)
                    fps_val = fps_dict.get(m_str, 0)
                    ram = self._get_hef_size_mb(m_str)
                    npu_ram += ram
                    if fps_val:
                        model_lines.append(
                            f"{m_str:<12} {ram:5.1f} MB  {fps_val:4.0f} FPS")
                    else:
                        model_lines.append(
                            f"{m_str:<12} {ram:5.1f} MB  geladen")

            # TAPPAS-Modelle die nicht in active_models stehen
            # (face_attr laeuft mit, wird aber oft nicht gelistet)
            tappas_extra = {"face_attr"}
            use_tappas = os.environ.get("MOLOCH_USE_TAPPAS", "0") == "1"
            if use_tappas and active:
                for extra in tappas_extra:
                    if extra not in [str(m) for m in active]:
                        ram = self._get_hef_size_mb(extra)
                        npu_ram += ram
                        model_lines.append(
                            f"{extra:<12} {ram:5.1f} MB  (TAPPAS)")

            if not model_lines:
                model_lines = ["keine"]

            models_text = "\n".join(model_lines)
            # Status-Farbe: gruen wenn Modelle aktiv
            if active:
                color = STATUS_GREEN
            else:
                color = FG_DIM
            return npu_state, color, models_text, npu_ram

        # Fallback: /dev/hailo0
        if os.path.exists("/dev/hailo0"):
            try:
                result = subprocess.run(
                    ["lsof", "/dev/hailo0"],
                    capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and result.stdout.strip():
                    return "aktiv", STATUS_GREEN, "lsof: aktiv", 0
                return "frei", STATUS_YELLOW, "keine", 0
            except Exception:
                return "vorhanden", FG_DIM, "lsof fehlt", 0

        return "nicht erkannt", STATUS_RED, "kein /dev/hailo0", 0

    def _read_uptime(self):
        """System-Uptime aus /proc/uptime."""
        try:
            with open("/proc/uptime", "r") as f:
                secs = float(f.read().split()[0])
            days = int(secs // 86400)
            hours = int((secs % 86400) // 3600)
            mins = int((secs % 3600) // 60)
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            elif hours > 0:
                return f"{hours}h {mins}m"
            else:
                return f"{mins}m"
        except Exception:
            pass
        return None

    # =========================================================================
    # Canvas-Balken
    # =========================================================================

    def _draw_bar(self, canvas, percent):
        """Farbigen Balken zeichnen."""
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 10:
            w = BAR_WIDTH
        percent = max(0, min(100, percent))
        px = int(w * percent / 100)
        if px > 0:
            color = _bar_color(percent)
            canvas.create_rectangle(0, 0, px, BAR_HEIGHT, fill=color, outline="")

    # =========================================================================
    # Update-Loop
    # =========================================================================

    def _update_all(self):
        """Daten im Background-Thread sammeln, UI im Main-Thread updaten.

        WICHTIG: Alle blocking I/O (subprocess, /proc-Iteration, HailoRT)
        laufen im Thread. Tkinter-Widgets werden NUR via win.after(0, ...)
        im Main-Thread aktualisiert. So blockiert nichts das Preview-Video.
        """
        def _collect():
            """Alle Hardware-Werte sammeln (laeuft im Thread)."""
            data = {}
            data["temp"] = self._read_cpu_temp()
            data["cpu_pct"] = self._read_cpu_percent()
            data["freq"] = self._read_cpu_freq()
            data["noctua"] = self._read_noctua_pwm()
            data["fan"] = self._read_fan_state()
            data["ram"] = self._read_ram()
            data["svc_rss"] = self._read_service_rss()
            data["npu"] = self._read_npu_status()
            data["npu_temp"] = self._read_npu_temperature()
            data["ssd1"] = self._read_disk("/")
            data["ssd2"] = self._read_disk("/mnt/moloch-data")
            data["uptime"] = self._read_uptime()
            # UI-Update im Main-Thread einplanen
            try:
                self.win.after(0, self._apply_update, data)
            except Exception:
                pass  # Fenster bereits geschlossen

        threading.Thread(target=_collect, daemon=True).start()
        # Naechsten Zyklus planen (unabhaengig vom Thread)
        self._after_id = self.win.after(UPDATE_MS, self._update_all)

    def _apply_update(self, data):
        """UI-Widgets aktualisieren (laeuft im Tkinter Main-Thread)."""
        # CPU Temperatur + Balken
        temp = data.get("temp")
        if temp is not None:
            color = STATUS_GREEN if temp < 60 else (
                STATUS_YELLOW if temp < 75 else STATUS_RED)
            self._lbl_temp.config(text=f"{temp:.1f}\u00b0C", fg=color)
            temp_pct = max(0, min(100, (temp - 30) / 60 * 100))
            self._draw_bar(self._canvas_temp, temp_pct)
        else:
            self._lbl_temp.config(text="n/a", fg=FG_DIM)

        # CPU Last
        cpu_pct = data.get("cpu_pct")
        if cpu_pct is not None:
            color = _bar_color(cpu_pct)
            self._lbl_cpu.config(text=f"{cpu_pct:.0f}%", fg=color)
        else:
            self._lbl_cpu.config(text="n/a", fg=FG_DIM)

        # CPU Frequenz
        freq = data.get("freq")
        if freq is not None:
            self._lbl_freq.config(text=f"{freq:.0f} MHz")
        else:
            self._lbl_freq.config(text="n/a", fg=FG_DIM)

        # Noctua NF-A4x20 (GPIO18 PWM-PIO)
        noctua_pct = data.get("noctua")
        if noctua_pct is not None:
            if noctua_pct <= 30:
                n_color = STATUS_GREEN
            elif noctua_pct <= 75:
                n_color = STATUS_YELLOW
            else:
                n_color = STATUS_RED
            self._lbl_noctua.config(text=f"{noctua_pct} %", fg=n_color)
            self._draw_bar(self._canvas_noctua, noctua_pct)
        else:
            self._lbl_noctua.config(text="n/a", fg=FG_DIM)
            self._draw_bar(self._canvas_noctua, 0)

        # Pi5 CPU-Kuehler
        fan_data = data.get("fan", (None, None, None))
        cur_st, max_st, pwm_pct = fan_data
        if cur_st is not None:
            label, color = self._FAN_STAGES.get(cur_st, (f"Stufe {cur_st}", FG_WHITE))
            max_str = str(max_st) if max_st is not None else "?"
            pct_str = f"  {pwm_pct}%" if pwm_pct is not None else ""
            self._lbl_cpufan.config(
                text=f"Stufe {cur_st}/{max_str}{pct_str}  {label}", fg=color)
            bar_pct = pwm_pct if pwm_pct is not None else (cur_st / (max_st or 4) * 100)
            self._draw_bar(self._canvas_cpufan, bar_pct)
        else:
            self._lbl_cpufan.config(text="---", fg=FG_DIM)
            self._draw_bar(self._canvas_cpufan, 0)

        # RAM (System)
        ram_used, ram_total = data.get("ram", (None, None))
        if ram_used is not None and ram_total is not None and ram_total > 0:
            pct = (ram_used / ram_total) * 100
            color = _bar_color(pct)
            self._lbl_ram.config(
                text=f"{ram_used:.0f} / {ram_total:.0f} MB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ram, pct)
        else:
            self._lbl_ram.config(text="n/a", fg=FG_DIM)

        # MOLOCH Service RSS + Threads
        svc_rss, svc_threads = data.get("svc_rss", (None, None))
        if svc_rss is not None:
            if svc_rss < 300:
                rss_color = STATUS_GREEN
            elif svc_rss < 800:
                rss_color = STATUS_YELLOW
            else:
                rss_color = STATUS_RED
            self._lbl_service_rss.config(text=f"{svc_rss:.0f} MB", fg=rss_color)
            self._lbl_threads.config(text=str(svc_threads), fg=FG_WHITE)
        else:
            self._lbl_service_rss.config(text="Service nicht aktiv", fg=FG_DIM)
            self._lbl_threads.config(text="--", fg=FG_DIM)

        # NPU Status + Modelle
        npu_status, npu_color, npu_models, npu_ram = data.get("npu", ("--", FG_DIM, "--", 0))
        self._lbl_npu_status.config(text=npu_status, fg=npu_color)
        self._lbl_npu_models.config(text=npu_models, fg=npu_color)

        # NPU Temperatur
        npu_temp = data.get("npu_temp")
        if npu_temp is not None:
            t_color = STATUS_GREEN if npu_temp < 60 else (
                STATUS_YELLOW if npu_temp < 75 else STATUS_RED)
            self._lbl_npu_temp.config(text=f"{npu_temp:.1f}\u00b0C", fg=t_color)
            t_pct = max(0, min(100, (npu_temp - 20) / 70 * 100))
            self._draw_bar(self._canvas_npu_temp, t_pct)
        else:
            self._lbl_npu_temp.config(text="n/a (belegt)", fg=FG_DIM)
            self._draw_bar(self._canvas_npu_temp, 0)

        # NPU RAM Balken (8GB NPU-RAM, Modelle belegen nur ~36MB)
        npu_total_mb = 8192
        if npu_ram > 0:
            npu_pct = (npu_ram / npu_total_mb) * 100
            self._lbl_npu_ram.config(
                text=f"{npu_ram:.0f} / {npu_total_mb} MB ({npu_pct:.1f}%)",
                fg=STATUS_GREEN)
            # Mindestbreite 3% damit der Balken sichtbar ist
            bar_pct = max(3.0, npu_pct)
            self._draw_bar(self._canvas_npu_ram, bar_pct)
        else:
            self._lbl_npu_ram.config(text=f"-- / {npu_total_mb} MB", fg=FG_DIM)
            self._draw_bar(self._canvas_npu_ram, 0)

        # SSD 1
        ssd1_used, ssd1_total = data.get("ssd1", (None, None))
        if ssd1_used is not None and ssd1_total is not None and ssd1_total > 0:
            pct = (ssd1_used / ssd1_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd1.config(
                text=f"{ssd1_used:.1f} / {ssd1_total:.1f} GB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ssd1, pct)
        else:
            self._lbl_ssd1.config(text="n/a", fg=FG_DIM)

        # SSD 2
        ssd2_used, ssd2_total = data.get("ssd2", (None, None))
        if ssd2_used is not None and ssd2_total is not None and ssd2_total > 0:
            pct = (ssd2_used / ssd2_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd2.config(
                text=f"{ssd2_used:.1f} / {ssd2_total:.1f} GB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ssd2, pct)
        else:
            self._lbl_ssd2.config(text="nicht gemountet", fg=STATUS_RED)

        # Uptime
        uptime = data.get("uptime")
        if uptime:
            self._lbl_uptime.config(text=uptime)
        else:
            self._lbl_uptime.config(text="n/a", fg=FG_DIM)

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Timer stoppen, Fenster schliessen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
