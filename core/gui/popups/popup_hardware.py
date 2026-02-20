#!/usr/bin/env python3
"""
M.O.L.O.C.H. Hardware Monitor Popup — Pi5
===========================================

Eigenstaendiges Toplevel-Fenster fuer Hardware-Monitoring.
Zeigt CPU, RAM, SSD und NPU Status des Raspberry Pi 5.

Sektionen:
- CPU: Temperatur (vcgencmd), Last (psutil oder /proc/stat), Frequenz
- RAM: Benutzt/Gesamt in MB, Canvas-Balken (Pi5 = 4GB)
- SSD: System-SSD (/) und Daten-SSD (/mnt/moloch-data), Canvas-Balken
- NPU: Hailo-10H Status aus ServiceProxy oder /dev/hailo0

Alle Werte alle 5 Sekunden aktualisiert.
Balken: gruen <60%, gelb 60-80%, rot >80%.

Importiert NUR panel_styles und tkinter.
"""

import logging
import os
import shutil
import subprocess
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_hardware")

# psutil optional (Fallback auf /proc)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Update-Intervall
UPDATE_MS = 5000

# Canvas-Balken Abmessungen
BAR_WIDTH = 260
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
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
            service_proxy: ServiceProxy Instanz fuer NPU-Status
        """
        self.parent = parent
        self.service = service_proxy
        self._after_id = None

        # Fuer CPU-Last Fallback (/proc/stat)
        self._prev_idle = 0
        self._prev_total = 0

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.title("Hardware Monitor \u2014 Pi5")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("380x480")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # GUI aufbauen
        self._build_cpu_section()
        self._build_ram_section()
        self._build_ssd_section()
        self._build_npu_section()

        # Erster Update sofort
        self._update_all()

    # =========================================================================
    # CPU Section
    # =========================================================================

    def _build_cpu_section(self):
        """CPU: Temperatur, Last, Frequenz."""
        section = tk.LabelFrame(
            self.win, text="CPU",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Temperatur
        row_temp = tk.Frame(section, bg=BG_FRAME)
        row_temp.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row_temp, text="Temperatur:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_temp = tk.Label(row_temp, text="--", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_temp.pack(side=tk.RIGHT)

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
    # RAM Section
    # =========================================================================

    def _build_ram_section(self):
        """RAM: Benutzt/Gesamt in MB mit Canvas-Balken."""
        section = tk.LabelFrame(
            self.win, text="RAM",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=(5, 2))
        self._lbl_ram = tk.Label(row, text="-- / -- MB", bg=BG_FRAME,
                                 fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ram.pack(side=tk.RIGHT)
        tk.Label(row, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)

        self._canvas_ram = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ram.pack(padx=8, pady=(0, 5))

    # =========================================================================
    # SSD Section
    # =========================================================================

    def _build_ssd_section(self):
        """SSD: System-SSD (/) und Daten-SSD (/mnt/moloch-data)."""
        section = tk.LabelFrame(
            self.win, text="Storage",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # System-SSD
        tk.Label(section, text="System-SSD (/)", bg=BG_FRAME, fg=FG_WHITE,
                 font=FONT_SMALL).pack(anchor=tk.W, padx=8, pady=(5, 0))

        row_ssd1 = tk.Frame(section, bg=BG_FRAME)
        row_ssd1.pack(fill=tk.X, padx=8, pady=2)
        self._lbl_ssd1 = tk.Label(row_ssd1, text="-- / -- GB", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ssd1.pack(side=tk.RIGHT)
        tk.Label(row_ssd1, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)

        self._canvas_ssd1 = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ssd1.pack(padx=8, pady=(0, 5))

        # Daten-SSD
        tk.Label(section, text="Daten-SSD (/mnt/moloch-data)", bg=BG_FRAME,
                 fg=FG_WHITE, font=FONT_SMALL).pack(anchor=tk.W, padx=8)

        row_ssd2 = tk.Frame(section, bg=BG_FRAME)
        row_ssd2.pack(fill=tk.X, padx=8, pady=2)
        self._lbl_ssd2 = tk.Label(row_ssd2, text="-- / -- GB", bg=BG_FRAME,
                                  fg=STATUS_YELLOW, font=FONT_MONO)
        self._lbl_ssd2.pack(side=tk.RIGHT)
        tk.Label(row_ssd2, text="Benutzt:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)

        self._canvas_ssd2 = tk.Canvas(
            section, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_ssd2.pack(padx=8, pady=(0, 5))

    # =========================================================================
    # NPU Section
    # =========================================================================

    def _build_npu_section(self):
        """NPU: Hailo-10H Status und aktive Modelle."""
        section = tk.LabelFrame(
            self.win, text="NPU \u2014 Hailo-10H",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(5, 10))

        row_status = tk.Frame(section, bg=BG_FRAME)
        row_status.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(row_status, text="Status:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_status = tk.Label(row_status, text="--", bg=BG_FRAME,
                                        fg=FG_DIM, font=FONT_MONO)
        self._lbl_npu_status.pack(side=tk.RIGHT)

        row_models = tk.Frame(section, bg=BG_FRAME)
        row_models.pack(fill=tk.X, padx=8, pady=(2, 5))
        tk.Label(row_models, text="Modelle:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_npu_models = tk.Label(row_models, text="--", bg=BG_FRAME,
                                        fg=FG_DIM, font=FONT_MONO,
                                        wraplength=220, justify=tk.RIGHT)
        self._lbl_npu_models.pack(side=tk.RIGHT)

    # =========================================================================
    # Daten lesen
    # =========================================================================

    def _read_cpu_temp(self):
        """CPU-Temperatur via vcgencmd measure_temp lesen."""
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                # Format: "temp=51.0'C"
                text = result.stdout.strip()
                val = text.split("=")[1].replace("'C", "")
                return float(val)
        except Exception as e:
            logger.debug(f"[HW] vcgencmd measure_temp failed: {e}")
        return None

    def _read_cpu_percent(self):
        """CPU-Last in Prozent. psutil bevorzugt, Fallback /proc/stat."""
        if HAS_PSUTIL:
            try:
                return psutil.cpu_percent(interval=0)
            except Exception:
                pass

        # Fallback: /proc/stat parsen
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
            parts = line.split()
            # user nice system idle iowait irq softirq steal
            vals = [int(x) for x in parts[1:]]
            idle = vals[3]
            total = sum(vals)
            diff_idle = idle - self._prev_idle
            diff_total = total - self._prev_total
            self._prev_idle = idle
            self._prev_total = total
            if diff_total > 0:
                return (1.0 - diff_idle / diff_total) * 100.0
        except Exception as e:
            logger.debug(f"[HW] /proc/stat lesen failed: {e}")
        return None

    def _read_cpu_freq(self):
        """CPU-Frequenz in MHz aus sysfs lesen."""
        try:
            path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            with open(path, "r") as f:
                khz = int(f.read().strip())
            return khz / 1000  # MHz
        except Exception as e:
            logger.debug(f"[HW] CPU freq lesen failed: {e}")
        return None

    def _read_ram(self):
        """RAM benutzt/gesamt in MB. psutil bevorzugt, Fallback /proc/meminfo."""
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                return mem.used / (1024 * 1024), mem.total / (1024 * 1024)
            except Exception:
                pass

        # Fallback: /proc/meminfo
        try:
            info = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        info[key] = int(parts[1])  # kB
            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
            used_kb = total_kb - avail_kb
            return used_kb / 1024, total_kb / 1024
        except Exception as e:
            logger.debug(f"[HW] /proc/meminfo lesen failed: {e}")
        return None, None

    def _read_disk(self, path):
        """Disk benutzt/gesamt in GB via shutil.disk_usage."""
        try:
            usage = shutil.disk_usage(path)
            used_gb = usage.used / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            return used_gb, total_gb
        except Exception as e:
            logger.debug(f"[HW] disk_usage({path}) failed: {e}")
        return None, None

    def _read_npu_status(self):
        """NPU-Status aus ServiceProxy oder /dev/hailo0 pruefen.

        Returns:
            (status_text, status_color, models_text)
        """
        # Erst aus ServiceProxy versuchen
        status = self.service.read_status()
        if status and isinstance(status, dict):
            npu = status.get("npu")
            if isinstance(npu, dict):
                npu_state = npu.get("status", "unbekannt")
                active = npu.get("active_models", [])
                if isinstance(active, list) and active:
                    models_text = ", ".join(str(m) for m in active)
                else:
                    models_text = "keine"

                if npu_state in ("vision", "voice", "active"):
                    return npu_state, STATUS_GREEN, models_text
                elif npu_state == "frei":
                    return "frei", STATUS_YELLOW, "keine"
                else:
                    return npu_state, FG_DIM, models_text

        # Fallback: /dev/hailo0 pruefen
        if os.path.exists("/dev/hailo0"):
            # Pruefen ob Prozesse das Device nutzen
            try:
                result = subprocess.run(
                    ["lsof", "/dev/hailo0"],
                    capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and result.stdout.strip():
                    # Prozessnamen extrahieren
                    lines = result.stdout.strip().splitlines()
                    procs = set()
                    for line in lines[1:]:  # Header ueberspringen
                        parts = line.split()
                        if parts:
                            procs.add(parts[0])
                    if procs:
                        return "aktiv", STATUS_GREEN, ", ".join(procs)
                return "frei", STATUS_YELLOW, "keine"
            except Exception:
                return "vorhanden", FG_DIM, "lsof fehlt"

        return "nicht erkannt", STATUS_RED, "kein /dev/hailo0"

    # =========================================================================
    # Canvas-Balken zeichnen
    # =========================================================================

    def _draw_bar(self, canvas, percent):
        """Canvas-Balken zeichnen mit farbiger Fuelllung."""
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
        """Alle Hardware-Werte lesen und UI aktualisieren."""
        # CPU Temperatur
        temp = self._read_cpu_temp()
        if temp is not None:
            color = STATUS_GREEN if temp < 60 else (
                STATUS_YELLOW if temp < 75 else STATUS_RED)
            self._lbl_temp.config(text=f"{temp:.1f}\u00b0C", fg=color)
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

        # RAM
        ram_used, ram_total = self._read_ram()
        if ram_used is not None and ram_total is not None and ram_total > 0:
            pct = (ram_used / ram_total) * 100
            color = _bar_color(pct)
            self._lbl_ram.config(
                text=f"{ram_used:.0f} / {ram_total:.0f} MB", fg=color)
            self._draw_bar(self._canvas_ram, pct)
        else:
            self._lbl_ram.config(text="n/a", fg=FG_DIM)

        # SSD 1: System (/)
        ssd1_used, ssd1_total = self._read_disk("/")
        if ssd1_used is not None and ssd1_total is not None and ssd1_total > 0:
            pct = (ssd1_used / ssd1_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd1.config(
                text=f"{ssd1_used:.1f} / {ssd1_total:.1f} GB", fg=color)
            self._draw_bar(self._canvas_ssd1, pct)
        else:
            self._lbl_ssd1.config(text="n/a", fg=FG_DIM)

        # SSD 2: Daten (/mnt/moloch-data)
        ssd2_used, ssd2_total = self._read_disk("/mnt/moloch-data")
        if ssd2_used is not None and ssd2_total is not None and ssd2_total > 0:
            pct = (ssd2_used / ssd2_total) * 100
            color = _bar_color(pct)
            self._lbl_ssd2.config(
                text=f"{ssd2_used:.1f} / {ssd2_total:.1f} GB", fg=color)
            self._draw_bar(self._canvas_ssd2, pct)
        else:
            self._lbl_ssd2.config(text="nicht gemountet", fg=STATUS_RED)

        # NPU
        npu_status, npu_color, npu_models = self._read_npu_status()
        self._lbl_npu_status.config(text=npu_status, fg=npu_color)
        self._lbl_npu_models.config(text=npu_models, fg=npu_color)

        # Naechster Update
        self._after_id = self.win.after(UPDATE_MS, self._update_all)

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen — Timer stoppen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
