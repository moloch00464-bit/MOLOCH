#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hardware Monitor Popup — Pi5 + Hailo-10H
=======================================================

Eigenstaendiges Toplevel-Fenster fuer Hardware-Monitoring.

Sektionen:
- CPU: Temperatur (Balken + Wert), Last (%), Frequenz, Luefter
- RAM: Benutzt/Gesamt MB + Balken (farbkodiert)
- NPU RAM: Hailo-10H Speicher (MB von 8 GB) + aktive Modelle mit FPS
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
        self.win.geometry("400x720")
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
        """RAM: Benutzt/Gesamt MB mit Balken."""
        section = tk.LabelFrame(
            self.win, text="RAM",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_ram = tk.Label(row, text="-- / -- MB", bg=BG_FRAME,
                                 fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ram.pack(side=tk.RIGHT)

        self._canvas_ram = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ram.pack(padx=8, pady=(0, 5))

    # =========================================================================
    # NPU Section (erweitert mit RAM + Modell-FPS)
    # =========================================================================

    def _build_npu_section(self):
        """NPU: Status, RAM-Nutzung, aktive Modelle mit FPS."""
        section = tk.LabelFrame(
            self.win, text="NPU \u2014 Hailo-10H (8 GB)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # Status
        row_status = tk.Frame(section, bg=BG_FRAME)
        row_status.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row_status, text="Status:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_status = tk.Label(row_status, text="--", bg=BG_FRAME,
                                        fg=FG_DIM, font=FONT_MONO)
        self._lbl_npu_status.pack(side=tk.RIGHT)

        # NPU RAM (geschaetzt)
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

        # Modelle + FPS (mehrzeilig)
        row_models = tk.Frame(section, bg=BG_FRAME)
        row_models.pack(fill=tk.X, padx=8, pady=(2, 5))
        tk.Label(row_models, text="Modelle:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT, anchor=tk.N)
        self._lbl_npu_models = tk.Label(
            row_models, text="--", bg=BG_FRAME,
            fg=FG_DIM, font=FONT_MONO,
            wraplength=240, justify=tk.LEFT, anchor=tk.W,
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

    def _read_npu_status(self):
        """NPU-Status: status, models + FPS, geschaetzter RAM.

        Returns:
            (status_text, status_color, models_text, npu_ram_mb)
        """
        # Modell-Groessen (MB, geschaetzt aus HEF-Dateien)
        MODEL_RAM_MB = {
            "scrfd": 6, "arcface": 3, "yolov8m": 21,
            "pose": 14, "hand_landmark": 4, "whisper": 130,
        }

        status = self.service.read_status()
        if status and isinstance(status, dict):
            npu = status.get("npu")
            if isinstance(npu, dict):
                npu_state = npu.get("status", "unbekannt")
                active = npu.get("active_models", [])

                # FPS pro Modell
                fps_dict = status.get("fps", {})
                if not isinstance(fps_dict, dict):
                    fps_dict = {}

                # Modelle + FPS formatiert
                model_lines = []
                npu_ram = 0
                if isinstance(active, list) and active:
                    for m in active:
                        m_str = str(m)
                        fps_val = fps_dict.get(m_str, 0)
                        ram = MODEL_RAM_MB.get(m_str, 5)
                        npu_ram += ram
                        if fps_val:
                            model_lines.append(f"{m_str}: {fps_val:.0f} FPS")
                        else:
                            model_lines.append(f"{m_str}: geladen")

                if not model_lines:
                    model_lines = ["keine"]

                models_text = "\n".join(model_lines)
                color = STATUS_GREEN if npu_state in ("vision", "voice", "active") else FG_DIM
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
        """Alle Hardware-Werte aktualisieren."""
        # CPU Temperatur + Balken
        temp = self._read_cpu_temp()
        if temp is not None:
            color = STATUS_GREEN if temp < 60 else (
                STATUS_YELLOW if temp < 75 else STATUS_RED)
            self._lbl_temp.config(text=f"{temp:.1f}\u00b0C", fg=color)
            # Temp-Balken: 30-90C -> 0-100%
            temp_pct = max(0, min(100, (temp - 30) / 60 * 100))
            self._draw_bar(self._canvas_temp, temp_pct)
        else:
            self._lbl_temp.config(text="n/a", fg=FG_DIM)

        # CPU Last
        cpu_pct = self._read_cpu_percent()
        if cpu_pct is not None:
            color = _bar_color(cpu_pct)
            self._lbl_cpu.config(text=f"{cpu_pct:.0f}%", fg=color)
        else:
            self._lbl_cpu.config(text="n/a", fg=FG_DIM)

        # CPU Frequenz
        freq = self._read_cpu_freq()
        if freq is not None:
            self._lbl_freq.config(text=f"{freq:.0f} MHz")
        else:
            self._lbl_freq.config(text="n/a", fg=FG_DIM)

        # Noctua NF-A4x20 (GPIO18 PWM-PIO)
        noctua_pct = self._read_noctua_pwm()
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

        # Pi5 CPU-Kühler
        cur_st, max_st, pwm_pct = self._read_fan_state()
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

        # RAM
        ram_used, ram_total = self._read_ram()
        if ram_used is not None and ram_total is not None and ram_total > 0:
            pct = (ram_used / ram_total) * 100
            color = _bar_color(pct)
            self._lbl_ram.config(
                text=f"{ram_used:.0f} / {ram_total:.0f} MB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ram, pct)
        else:
            self._lbl_ram.config(text="n/a", fg=FG_DIM)

        # NPU
        npu_status, npu_color, npu_models, npu_ram = self._read_npu_status()
        self._lbl_npu_status.config(text=npu_status, fg=npu_color)
        self._lbl_npu_models.config(text=npu_models, fg=npu_color)

        # NPU RAM Balken
        npu_total_mb = 8192
        if npu_ram > 0:
            npu_pct = (npu_ram / npu_total_mb) * 100
            color = _bar_color(npu_pct)
            self._lbl_npu_ram.config(
                text=f"~{npu_ram} / {npu_total_mb} MB ({npu_pct:.1f}%)", fg=color)
            self._draw_bar(self._canvas_npu_ram, npu_pct)
        else:
            self._lbl_npu_ram.config(text=f"-- / {npu_total_mb} MB", fg=FG_DIM)
            self._draw_bar(self._canvas_npu_ram, 0)

        # SSD 1
        ssd1_used, ssd1_total = self._read_disk("/")
        if ssd1_used is not None and ssd1_total is not None and ssd1_total > 0:
            pct = (ssd1_used / ssd1_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd1.config(
                text=f"{ssd1_used:.1f} / {ssd1_total:.1f} GB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ssd1, pct)
        else:
            self._lbl_ssd1.config(text="n/a", fg=FG_DIM)

        # SSD 2
        ssd2_used, ssd2_total = self._read_disk("/mnt/moloch-data")
        if ssd2_used is not None and ssd2_total is not None and ssd2_total > 0:
            pct = (ssd2_used / ssd2_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd2.config(
                text=f"{ssd2_used:.1f} / {ssd2_total:.1f} GB ({pct:.0f}%)", fg=color)
            self._draw_bar(self._canvas_ssd2, pct)
        else:
            self._lbl_ssd2.config(text="nicht gemountet", fg=STATUS_RED)

        # Uptime
        uptime = self._read_uptime()
        if uptime:
            self._lbl_uptime.config(text=uptime)
        else:
            self._lbl_uptime.config(text="n/a", fg=FG_DIM)

        # Naechster Update
        self._after_id = self.win.after(UPDATE_MS, self._update_all)

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Timer stoppen, Fenster schliessen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
