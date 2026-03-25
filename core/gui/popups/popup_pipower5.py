#!/usr/bin/env python3
"""
M.O.L.O.C.H. PiPower5 Popup — SunFounder PiPower 5 HAT+ Monitor
=================================================================

Zeigt alle Anschluesse und Messwerte des PiPower 5 HAT+:

  Abschnitte:
  - Stromeingang:  Schraubklemme DC-In / USB-C, Spannung + Strom
  - Akku:          18650-Zellen, %, Spannung, Strom, Ladebalken
  - Ausgang Pi:    5V Ausgang, Spannung, Strom, Leistung
  - Quelle:        Netzbetrieb / Akkubetrieb, Shutdown-Status
  - Anschluesse:   Schraubklemmen-Uebersicht (statisch)

Daten werden alle 5 Sekunden aktualisiert.
Pipower5-Bibliothek liegt in eigenem venv — Zugriff via subprocess.
"""

import json
import logging
import subprocess
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_pipower5")

# PiPower5 venv Python
_PIPOWER5_PYTHON = "/opt/pipower5/venv/bin/python3"

# Aktualisierungsintervall
UPDATE_MS = 5000

# Balken-Abmessungen
BAR_WIDTH  = 300
BAR_HEIGHT = 16

# read_all() via separatem Prozess (eigenes venv)
_READ_CMD = [
    _PIPOWER5_PYTHON, "-c",
    "import json; "
    "from pipower5.pipower5_service import PiPower5; "
    "p=PiPower5(); "
    "print(json.dumps(p.read_all()))"
]


def _read_pipower5():
    """Liefert dict mit allen PiPower5-Messwerten oder None bei Fehler."""
    try:
        result = subprocess.run(
            _READ_CMD,
            capture_output=True, text=True, timeout=4
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except Exception as e:
        logger.warning("PiPower5 read_all() Fehler: %s", e)
    return None


def _bar_color(pct):
    """Balkenfarbe: gruen >60%, gelb 30-60%, rot <30% (Akkulogik)."""
    if pct >= 60:
        return STATUS_GREEN
    elif pct >= 30:
        return STATUS_YELLOW
    return STATUS_RED


def _bar_color_load(pct):
    """Balkenfarbe fuer Last: gruen <60%, gelb 60-80%, rot >80%."""
    if pct < 60:
        return STATUS_GREEN
    elif pct < 80:
        return STATUS_YELLOW
    return STATUS_RED


class PiPower5Popup:
    """PiPower5 HAT+ Monitor — eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent):
        self.parent  = parent
        self._after  = None

        # --- Fenster ---
        self.win = tk.Toplevel(parent)
        self.win.title("PiPower 5 HAT+ — Stromversorgung")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("420x680")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Sektionen aufbauen ---
        self._build_input_section()
        self._build_battery_section()
        self._build_output_section()
        self._build_status_section()
        self._build_connections_section()

        # --- Erster Datenabruf ---
        self._update()

    # =========================================================================
    # Sektion: Stromeingang
    # =========================================================================

    def _build_input_section(self):
        """Schraubklemme DC-Eingang / USB-C."""
        sec = tk.LabelFrame(
            self.win, text="⚡  Stromeingang  (DC-Schraubklemme / USB-C)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        sec.pack(fill=tk.X, padx=10, pady=(10, 4))

        # Stecker-Status
        r0 = tk.Frame(sec, bg=BG_FRAME)
        r0.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(r0, text="Stecker:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_plugged = tk.Label(r0, text="---", bg=BG_FRAME,
                                     fg=FG_DIM, font=FONT_MONO)
        self._lbl_plugged.pack(side=tk.RIGHT)

        # Eingangsspannung
        r1 = tk.Frame(sec, bg=BG_FRAME)
        r1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r1, text="Spannung:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_in_v = tk.Label(r1, text="--- V", bg=BG_FRAME,
                                   fg=FG_WHITE, font=FONT_MONO)
        self._lbl_in_v.pack(side=tk.RIGHT)

        # Eingangsstrom
        r2 = tk.Frame(sec, bg=BG_FRAME)
        r2.pack(fill=tk.X, padx=8, pady=(2, 6))
        tk.Label(r2, text="Strom:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_in_a = tk.Label(r2, text="--- mA", bg=BG_FRAME,
                                   fg=FG_WHITE, font=FONT_MONO)
        self._lbl_in_a.pack(side=tk.RIGHT)

    # =========================================================================
    # Sektion: Akku
    # =========================================================================

    def _build_battery_section(self):
        """18650-Akku: Ladestand, Spannung, Strom, Balken."""
        sec = tk.LabelFrame(
            self.win, text="🔋  Akku  (18650 · 2000 mAh · externe Klemme)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        sec.pack(fill=tk.X, padx=10, pady=4)

        # Ladestand + Balken
        r_pct = tk.Frame(sec, bg=BG_FRAME)
        r_pct.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(r_pct, text="Ladestand:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_bat_pct = tk.Label(r_pct, text="--%", bg=BG_FRAME,
                                      fg=STATUS_GREEN, font=FONT_MONO)
        self._lbl_bat_pct.pack(side=tk.RIGHT)

        self._canvas_bat = tk.Canvas(
            sec, width=BAR_WIDTH, height=BAR_HEIGHT,
            bg=BG_INPUT, highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._canvas_bat.pack(padx=8, pady=(0, 3))

        # Akkuspannung
        r1 = tk.Frame(sec, bg=BG_FRAME)
        r1.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r1, text="Spannung:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_bat_v = tk.Label(r1, text="--- V", bg=BG_FRAME,
                                    fg=FG_WHITE, font=FONT_MONO)
        self._lbl_bat_v.pack(side=tk.RIGHT)

        # Akkustrom
        r2 = tk.Frame(sec, bg=BG_FRAME)
        r2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r2, text="Strom:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_bat_a = tk.Label(r2, text="--- mA", bg=BG_FRAME,
                                    fg=FG_WHITE, font=FONT_MONO)
        self._lbl_bat_a.pack(side=tk.RIGHT)

        # Ladestatus
        r3 = tk.Frame(sec, bg=BG_FRAME)
        r3.pack(fill=tk.X, padx=8, pady=(2, 6))
        tk.Label(r3, text="Status:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_bat_status = tk.Label(r3, text="---", bg=BG_FRAME,
                                         fg=FG_DIM, font=FONT_MONO)
        self._lbl_bat_status.pack(side=tk.RIGHT)

    # =========================================================================
    # Sektion: Ausgang zum Pi
    # =========================================================================

    def _build_output_section(self):
        """5V-Ausgang: Spannung, Strom, Leistung."""
        sec = tk.LabelFrame(
            self.win, text="📤  Ausgang zum Pi  (5 V Schiene)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        sec.pack(fill=tk.X, padx=10, pady=4)

        r1 = tk.Frame(sec, bg=BG_FRAME)
        r1.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(r1, text="Spannung:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_out_v = tk.Label(r1, text="--- V", bg=BG_FRAME,
                                    fg=FG_WHITE, font=FONT_MONO)
        self._lbl_out_v.pack(side=tk.RIGHT)

        r2 = tk.Frame(sec, bg=BG_FRAME)
        r2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(r2, text="Strom:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_out_a = tk.Label(r2, text="--- mA", bg=BG_FRAME,
                                    fg=FG_WHITE, font=FONT_MONO)
        self._lbl_out_a.pack(side=tk.RIGHT)

        r3 = tk.Frame(sec, bg=BG_FRAME)
        r3.pack(fill=tk.X, padx=8, pady=(2, 6))
        tk.Label(r3, text="Leistung:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_out_w = tk.Label(r3, text="--- W", bg=BG_FRAME,
                                    fg=STATUS_GREEN, font=FONT_MONO)
        self._lbl_out_w.pack(side=tk.RIGHT)

    # =========================================================================
    # Sektion: Quelle & Status
    # =========================================================================

    def _build_status_section(self):
        """Stromquelle, Shutdown-Request."""
        sec = tk.LabelFrame(
            self.win, text="🔌  Quelle & Systemstatus",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        sec.pack(fill=tk.X, padx=10, pady=4)

        r1 = tk.Frame(sec, bg=BG_FRAME)
        r1.pack(fill=tk.X, padx=8, pady=(5, 2))
        tk.Label(r1, text="Stromquelle:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_source = tk.Label(r1, text="---", bg=BG_FRAME,
                                     fg=FG_DIM, font=FONT_MONO)
        self._lbl_source.pack(side=tk.RIGHT)

        r2 = tk.Frame(sec, bg=BG_FRAME)
        r2.pack(fill=tk.X, padx=8, pady=(2, 6))
        tk.Label(r2, text="Shutdown-Request:", bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._lbl_shutdown = tk.Label(r2, text="---", bg=BG_FRAME,
                                       fg=STATUS_GREEN, font=FONT_MONO)
        self._lbl_shutdown.pack(side=tk.RIGHT)

    # =========================================================================
    # Sektion: Anschluss-Uebersicht (statisch)
    # =========================================================================

    def _build_connections_section(self):
        """Schraubklemmen-Uebersicht — statische Referenz."""
        sec = tk.LabelFrame(
            self.win, text="ℹ️  Schraubklemmen-Referenz",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        sec.pack(fill=tk.X, padx=10, pady=(4, 10))

        verbindungen = [
            ("DC-Eingang",        "6–30 V  ·  Schraubklemme links oben"),
            ("Ext. Akkupack",     "7.4 V LiPo  ·  Schraubklemme rechts"),
            ("Int. Akku (18650)", "2 Zellen  ·  Sockel auf HAT+"),
            ("5V-Ausgang Pi",     "GPIO Pin 2/4  ·  automatisch"),
            ("USB-C Eingang",     "5 V / 3 A  ·  Buchse seitlich"),
            ("Power-Button",      "Kurzdr.=An/Status  ·  Lang=Aus"),
        ]

        for name, detail in verbindungen:
            row = tk.Frame(sec, bg=BG_FRAME)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=name, bg=BG_FRAME, fg=FG_WHITE,
                     font=FONT_SMALL, width=18, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=detail, bg=BG_FRAME, fg=FG_DIM,
                     font=FONT_SMALL, anchor=tk.W).pack(side=tk.LEFT, padx=(4, 0))

        # Letzter Abstand
        tk.Frame(sec, bg=BG_FRAME, height=4).pack()

    # =========================================================================
    # Balken zeichnen
    # =========================================================================

    def _draw_bar(self, canvas, pct, color_fn=_bar_color):
        """Farbigen Fuellbalken zeichnen."""
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 10:
            w = BAR_WIDTH
        pct = max(0.0, min(100.0, float(pct)))
        px = int(w * pct / 100)
        if px > 0:
            canvas.create_rectangle(0, 0, px, BAR_HEIGHT,
                                    fill=color_fn(pct), outline="")

    # =========================================================================
    # Update-Loop
    # =========================================================================

    def _update(self):
        """Messwerte holen und Anzeige aktualisieren."""
        data = _read_pipower5()

        if data:
            self._update_input(data)
            self._update_battery(data)
            self._update_output(data)
            self._update_status(data)
        else:
            self._show_error()

        self._after = self.win.after(UPDATE_MS, self._update)

    def _update_input(self, d):
        """Eingang aktualisieren."""
        plugged   = d.get("is_input_plugged_in", False)
        in_v_mv   = d.get("input_voltage", 0)
        in_a_ma   = d.get("input_current", 0)

        if plugged:
            self._lbl_plugged.config(text="✓ Angeschlossen", fg=STATUS_GREEN)
        else:
            self._lbl_plugged.config(text="✗ Getrennt", fg=STATUS_RED)

        in_v = in_v_mv / 1000.0
        fg_v = FG_WHITE if in_v_mv > 0 else FG_DIM
        self._lbl_in_v.config(text=f"{in_v:.3f} V", fg=fg_v)
        self._lbl_in_a.config(text=f"{in_a_ma:+d} mA", fg=fg_v)

    def _update_battery(self, d):
        """Akku-Sektion aktualisieren."""
        pct       = d.get("battery_percentage", 0)
        bat_v_mv  = d.get("battery_voltage", 0)
        bat_a_ma  = d.get("battery_current", 0)
        charging  = d.get("is_charging", False)

        # Ladestand
        color_pct = _bar_color(pct)
        self._lbl_bat_pct.config(text=f"{pct} %", fg=color_pct)
        self._draw_bar(self._canvas_bat, pct, _bar_color)

        # Spannung + Strom
        bat_v = bat_v_mv / 1000.0
        self._lbl_bat_v.config(text=f"{bat_v:.3f} V")

        if bat_a_ma > 50:
            strom_txt = f"+{bat_a_ma} mA  (laedt)"
            strom_fg  = STATUS_GREEN
        elif bat_a_ma < -50:
            strom_txt = f"{bat_a_ma} mA  (Entladung)"
            strom_fg  = STATUS_YELLOW
        else:
            strom_txt = f"{bat_a_ma} mA  (Leerlauf)"
            strom_fg  = FG_DIM
        self._lbl_bat_a.config(text=strom_txt, fg=strom_fg)

        # Ladestatus
        if charging:
            self._lbl_bat_status.config(text="⚡ Laedt", fg=STATUS_GREEN)
        elif not d.get("is_input_plugged_in", False):
            self._lbl_bat_status.config(text="🔋 Akkubetrieb", fg=STATUS_YELLOW)
        else:
            self._lbl_bat_status.config(text="✓ Voll / Bereit", fg=STATUS_GREEN)

    def _update_output(self, d):
        """Ausgang zum Pi aktualisieren."""
        out_v_mv = d.get("output_voltage", 0)
        out_a_ma = d.get("output_current", 0)

        out_v = out_v_mv / 1000.0
        out_a = out_a_ma / 1000.0
        out_w = out_v * out_a

        v_fg = STATUS_GREEN if 4.9 <= out_v <= 5.3 else STATUS_RED
        self._lbl_out_v.config(text=f"{out_v:.3f} V", fg=v_fg)
        self._lbl_out_a.config(text=f"{out_a_ma} mA")
        self._lbl_out_w.config(text=f"{out_w:.2f} W")

    def _update_status(self, d):
        """Quelle und Shutdown-Status aktualisieren."""
        # 0 = EXTERNAL, 1 = BATTERY
        src = d.get("power_source", -1)
        if src == 0:
            self._lbl_source.config(text="Netz (extern)", fg=STATUS_GREEN)
        elif src == 1:
            self._lbl_source.config(text="Akku", fg=STATUS_YELLOW)
        else:
            self._lbl_source.config(text="unbekannt", fg=FG_DIM)

        # 0=NONE, 1=LOW_BATTERY, 2=BUTTON, 3=LOW_VOLTAGE
        sd = d.get("shutdown_request", 0)
        sd_map = {
            0: ("Keiner",             STATUS_GREEN),
            1: ("Akku schwach!",      STATUS_RED),
            2: ("Button gedrueckt",   STATUS_YELLOW),
            3: ("Spannung zu niedrig",STATUS_RED),
        }
        txt, fg = sd_map.get(sd, (f"Code {sd}", FG_DIM))
        self._lbl_shutdown.config(text=txt, fg=fg)

    def _show_error(self):
        """Alle Labels auf Fehler setzen."""
        err = "FEHLER"
        for lbl in (self._lbl_plugged, self._lbl_in_v, self._lbl_in_a,
                    self._lbl_bat_pct, self._lbl_bat_v, self._lbl_bat_a,
                    self._lbl_bat_status, self._lbl_out_v, self._lbl_out_a,
                    self._lbl_out_w, self._lbl_source, self._lbl_shutdown):
            lbl.config(text=err, fg=STATUS_RED)

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Timer stoppen und Fenster schliessen."""
        if self._after is not None:
            self.win.after_cancel(self._after)
            self._after = None
        self.win.destroy()
