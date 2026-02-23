#!/usr/bin/env python3
"""
M.O.L.O.C.H. Tracker Einstellungen Popup
==========================================

Eigenstaendiges Toplevel-Fenster fuer AutonomousTracker Parameter.

Sektionen:
- Dead Zone & Coast: Dead Zone Groesse, Track Start, Coast Timeout, Coast Resume
- Tracking Dynamik: Speed, Smoothing, Pan/Tilt Gain, Max Step
- Hardware Limits: Pan/Tilt Min/Max
- Search Modus: Speed, Timeout, Home Timeout
- Home Position: Button "Aktuelle Position als Home"

Alle Aenderungen sofort live via IPC, persistent in settings.json.
Importiert NUR panel_styles, tkinter, json, os.
"""

import json
import os
import tempfile
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_YELLOW, STATUS_GREEN, ACCENT_RED, ACCENT_GREEN, ACCENT_CYAN,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

# Settings-Pfad
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")

# ============================================================================
# Slider Definitionen: (Anzeigename, key, min, max, default, schritt, einheit, tooltip)
# ============================================================================

DEADZONE_DEFS = [
    ("Dead Zone", "dead_zone_pct",
     0.05, 0.40, 0.15, 0.01, "", "Ruhezone im Bildzentrum (% vom Bild, beidseitig)"),
    ("Track Start", "track_start_pct",
     0.05, 0.40, 0.18, 0.01, "", "Ab hier startet Tracking (Hysterese ueber Dead Zone)"),
    ("Coast Timeout", "coast_stable_time",
     0.5, 10.0, 1.5, 0.5, "s", "Sekunden stabil im Dead Zone -> Kamera einfrieren"),
    ("Coast Aufwach-%", "coast_resume_pct",
     0.05, 0.30, 0.12, 0.01, "", "Abweichung zum Aufwachen aus Coast-Modus"),
]

TRACKING_DEFS = [
    ("Tracking Speed", "tracking_speed",
     0.1, 1.0, 0.7, 0.05, "", "ONVIF Geschwindigkeit (1.0 = Vollgas, nicht empfohlen)"),
    ("Min. Speed", "min_move_speed",
     0.05, 0.50, 0.15, 0.05, "", "Minimale Geschwindigkeit bei kleinen Korrekturen"),
    ("Smoothing (EMA)", "smooth_alpha",
     0.05, 0.50, 0.20, 0.05, "", "EMA Alpha (hoeher = schnellere Reaktion, mehr Ruckeln)"),
    ("Pan Gain", "pan_gain",
     0.05, 0.60, 0.25, 0.05, "", "Proportional-Regler Pan (hoeher = aggressiver)"),
    ("Tilt Gain", "tilt_gain",
     0.05, 0.60, 0.20, 0.05, "", "Proportional-Regler Tilt (hoeher = aggressiver)"),
    ("Max Step Pan", "max_step_pan",
     1.0, 15.0, 5.0, 0.5, "deg", "Maximale Pan-Korrektur pro Move"),
    ("Max Step Tilt", "max_step_tilt",
     1.0, 10.0, 3.0, 0.5, "deg", "Maximale Tilt-Korrektur pro Move"),
    ("Move Cooldown", "move_cooldown_ms",
     100, 1000, 400, 50, "ms", "Mindestpause zwischen Kamerabewegungen"),
    ("Smoothing Frames", "center_ring_buffer_size",
     3, 20, 10, 1, "", "Ring-Buffer Groesse fuer Positionsmittelung"),
]

LIMIT_DEFS = [
    ("Pan Links (min)", "pan_limit_min",
     -180.0, 0.0, -168.4, 1.0, "deg", "Linke Pan-Grenze (negativ = links)"),
    ("Pan Rechts (max)", "pan_limit_max",
     0.0, 180.0, 170.0, 1.0, "deg", "Rechte Pan-Grenze (positiv = rechts)"),
    ("Tilt Unten (min)", "tilt_limit_min",
     -90.0, 0.0, -78.0, 1.0, "deg", "Untere Tilt-Grenze (negativ = runter)"),
    ("Tilt Oben (max)", "tilt_limit_max",
     0.0, 90.0, 78.8, 1.0, "deg", "Obere Tilt-Grenze (positiv = hoch)"),
]

SEARCH_DEFS = [
    ("Search Speed", "search_speed",
     0.1, 0.8, 0.3, 0.05, "", "Patrol-Geschwindigkeit beim Suchen"),
    ("Search Timeout", "search_home_timeout",
     10, 120, 30, 5, "s", "Sekunden ohne Fund -> zurueck zu Home"),
    ("Target Lost Timeout", "target_lost_timeout",
     1.0, 15.0, 5.0, 0.5, "s", "Coasting-Zeit bevor Search startet"),
]


class TrackerPopup:
    """Tracker Einstellungen als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy
        self._save_after_id = None

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Tracker Einstellungen")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("460x700")
        self.win.resizable(False, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Scrollbare Flaeche
        self._build_scrollable()

        # Slider-Variablen: {key: tk.DoubleVar}
        self._vars = {}
        self._labels = {}

        # GUI-Sektionen
        self._build_deadzone_section()
        self._build_tracking_section()
        self._build_limits_section()
        self._build_search_section()
        self._build_home_section()
        self._build_buttons()

        # Werte laden
        self._load_current_values()

        # Scroll-Region
        self._inner.update_idletasks()
        self._canvas.config(scrollregion=self._canvas.bbox("all"))

    # =========================================================================
    # Scrollbare Flaeche
    # =========================================================================

    def _build_scrollable(self):
        """Canvas + Scrollbar + innerer Frame."""
        container = tk.Frame(self.win, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(self._canvas, bg=BG_DARK)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor=tk.NW)

        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._canvas.bind("<Configure>", _on_canvas_configure)

        def _on_inner_configure(_event):
            self._canvas.config(scrollregion=self._canvas.bbox("all"))
        self._inner.bind("<Configure>", _on_inner_configure)

        def _on_mousewheel(event):
            self._canvas.yview_scroll(-1 * (event.delta // 120 or (-1 if event.num == 4 else 1)), "units")
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._canvas.bind_all("<Button-4>", _on_mousewheel)
        self._canvas.bind_all("<Button-5>", _on_mousewheel)

    # =========================================================================
    # Dead Zone & Coast
    # =========================================================================

    def _build_deadzone_section(self):
        """Dead Zone und Coast Parameter."""
        section = tk.LabelFrame(
            self._inner, text="Dead Zone & Coast",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(10, 5))

        for name, key, vmin, vmax, default, step, unit, tip in DEADZONE_DEFS:
            self._build_slider_row(section, name, key, vmin, vmax, default, step, unit, tip)

    # =========================================================================
    # Tracking Dynamik
    # =========================================================================

    def _build_tracking_section(self):
        """Tracking Speed, Gain, Smoothing Parameter."""
        section = tk.LabelFrame(
            self._inner, text="Tracking Dynamik",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        for name, key, vmin, vmax, default, step, unit, tip in TRACKING_DEFS:
            self._build_slider_row(section, name, key, vmin, vmax, default, step, unit, tip)

    # =========================================================================
    # Hardware Limits
    # =========================================================================

    def _build_limits_section(self):
        """Pan/Tilt Hardware-Grenzen."""
        section = tk.LabelFrame(
            self._inner, text="Hardware Limits",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        for name, key, vmin, vmax, default, step, unit, tip in LIMIT_DEFS:
            self._build_slider_row(section, name, key, vmin, vmax, default, step, unit, tip)

    # =========================================================================
    # Search Modus
    # =========================================================================

    def _build_search_section(self):
        """Search/Patrol Parameter."""
        section = tk.LabelFrame(
            self._inner, text="Such-Modus",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        for name, key, vmin, vmax, default, step, unit, tip in SEARCH_DEFS:
            self._build_slider_row(section, name, key, vmin, vmax, default, step, unit, tip)

    # =========================================================================
    # Home Position
    # =========================================================================

    def _build_home_section(self):
        """Home-Position setzen und anzeigen."""
        section = tk.LabelFrame(
            self._inner, text="Home Position",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=5)

        self._lbl_home = tk.Label(
            row, text="Home: Pan 0.0 / Tilt 0.0",
            bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        self._lbl_home.pack(side=tk.LEFT)

        tk.Button(
            row, text="Aktuelle Position als Home",
            bg=ACCENT_CYAN, fg=BG_DARK, font=FONT_BUTTON,
            activebackground="#00bbdd",
            command=self._set_home_position,
        ).pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # Buttons
    # =========================================================================

    def _build_buttons(self):
        """Save und Reset Buttons."""
        btn_frame = tk.Frame(self._inner, bg=BG_DARK)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        tk.Button(
            btn_frame, text="SAVE", width=12,
            bg=ACCENT_GREEN, fg=BG_DARK, font=FONT_BUTTON,
            activebackground="#00cc55",
            command=self._force_save,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="RESET DEFAULTS", width=14,
            bg=ACCENT_RED, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground="#cc0000",
            command=self._reset_defaults,
        ).pack(side=tk.RIGHT, padx=5)

        self._lbl_feedback = tk.Label(
            btn_frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_feedback.pack(side=tk.LEFT, padx=10)

    # =========================================================================
    # Slider-Zeile
    # =========================================================================

    def _build_slider_row(self, parent, name, key, vmin, vmax,
                          default, step, unit, tooltip):
        """Eine Slider-Zeile mit Live-Wert."""
        row = tk.Frame(parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(
            row, text=name, width=20, anchor=tk.W,
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        fmt = self._format_value(default, unit, step)
        lbl = tk.Label(
            row, text=fmt, width=9, anchor=tk.E,
            bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        lbl.pack(side=tk.RIGHT)
        self._labels[key] = lbl

        resolution = step if isinstance(default, float) else int(step)
        var = tk.DoubleVar(value=default)
        self._vars[key] = var

        slider = tk.Scale(
            row, from_=vmin, to=vmax, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=lambda val, k=key, u=unit, s=step: self._on_slider_changed(k, val, u, s),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        if tooltip:
            tk.Label(
                parent, text=tooltip, anchor=tk.W,
                bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
            ).pack(fill=tk.X, padx=16, pady=(0, 2))

    @staticmethod
    def _format_value(val, unit="", step=0.01):
        """Wert formatiert als String."""
        if isinstance(val, float):
            if step >= 1:
                return f"{val:.0f}{unit}"
            elif step >= 0.1:
                return f"{val:.1f}{unit}"
            else:
                return f"{val:.2f}{unit}"
        return f"{int(val)}{unit}"

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_slider_changed(self, key, value, unit, step):
        """Slider geaendert: Label updaten, Service senden, speichern."""
        val = float(value)
        self._labels[key].config(text=self._format_value(val, unit, step))

        self.service._write_command("action", {
            "action": "set_tracker_param",
            "param": key,
            "value": val,
        })
        self._save_settings()

    def _set_home_position(self):
        """Aktuelle Kamera-Position als Home speichern."""
        # Position aus Service-Status lesen
        status = self.service.read_status()
        home_pan = 0.0
        home_tilt = 0.0

        if status and isinstance(status, dict):
            tracker = status.get("tracker", {})
            if isinstance(tracker, dict):
                cam_pos = tracker.get("camera_position", {})
                if isinstance(cam_pos, dict):
                    home_pan = cam_pos.get("pan_deg", 0.0)
                    home_tilt = cam_pos.get("tilt_deg", 0.0)

        self._lbl_home.config(
            text=f"Home: Pan {home_pan:.1f} / Tilt {home_tilt:.1f}",
            fg=STATUS_GREEN)

        # Service-Command
        self.service._write_command("action", {
            "action": "set_tracker_home",
            "pan": home_pan,
            "tilt": home_tilt,
        })

        # In settings.json speichern
        try:
            data = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
            tracker = data.get("tracker", {})
            tracker["home_pan"] = round(home_pan, 1)
            tracker["home_tilt"] = round(home_tilt, 1)
            data["tracker"] = tracker
            self._atomic_write(data)
        except Exception:
            pass

        self._lbl_feedback.config(text="Home gesetzt!", fg=ACCENT_GREEN)
        self.win.after(2000, lambda: self._lbl_feedback.config(text="", fg=FG_DIM))

    # =========================================================================
    # Werte laden
    # =========================================================================

    def _load_current_values(self):
        """Aktuelle Werte aus settings.json laden."""
        data = {}
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
        except Exception:
            pass

        tracker = data.get("tracker", {})
        if not isinstance(tracker, dict):
            tracker = {}

        all_defs = list(DEADZONE_DEFS) + list(TRACKING_DEFS) + list(LIMIT_DEFS) + list(SEARCH_DEFS)
        for name, key, vmin, vmax, default, step, unit, tip in all_defs:
            raw = tracker.get(key)
            if raw is not None:
                try:
                    val = max(vmin, min(vmax, float(raw)))
                except (TypeError, ValueError):
                    val = default
            else:
                val = default

            if key in self._vars:
                self._vars[key].set(val)
                self._labels[key].config(text=self._format_value(val, unit, step))

        # Home Position Label
        home_pan = tracker.get("home_pan", 0.0)
        home_tilt = tracker.get("home_tilt", 0.0)
        self._lbl_home.config(
            text=f"Home: Pan {home_pan:.1f} / Tilt {home_tilt:.1f}")

    # =========================================================================
    # Settings speichern (debounced 300ms)
    # =========================================================================

    def _save_settings(self):
        """Save nach 300ms Debounce."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
        self._save_after_id = self.win.after(300, self._do_save_settings)

    def _force_save(self):
        """Sofort speichern."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._do_save_settings()
        self._lbl_feedback.config(text="Gespeichert!", fg=ACCENT_GREEN)
        self.win.after(2000, lambda: self._lbl_feedback.config(text="", fg=FG_DIM))

    def _do_save_settings(self):
        """Tracker-Werte atomar in settings.json schreiben."""
        self._save_after_id = None
        try:
            data = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)

            tracker = data.get("tracker", {})
            all_defs = list(DEADZONE_DEFS) + list(TRACKING_DEFS) + list(LIMIT_DEFS) + list(SEARCH_DEFS)
            for _name, key, _vmin, _vmax, _default, _step, _unit, _tip in all_defs:
                if key in self._vars:
                    tracker[key] = round(self._vars[key].get(), 3)
            data["tracker"] = tracker

            self._atomic_write(data)
        except Exception:
            pass

    def _atomic_write(self, data):
        """Atomar in settings.json schreiben."""
        dir_path = os.path.dirname(SETTINGS_PATH)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, SETTINGS_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # =========================================================================
    # Reset Defaults
    # =========================================================================

    def _reset_defaults(self):
        """Alle Slider auf Defaults."""
        all_defs = list(DEADZONE_DEFS) + list(TRACKING_DEFS) + list(LIMIT_DEFS) + list(SEARCH_DEFS)
        for name, key, vmin, vmax, default, step, unit, tip in all_defs:
            if key in self._vars:
                self._vars[key].set(default)
                self._labels[key].config(text=self._format_value(default, unit, step))

        self._do_save_settings()
        self._lbl_feedback.config(text="Defaults geladen!", fg=STATUS_YELLOW)
        self.win.after(2000, lambda: self._lbl_feedback.config(text="", fg=FG_DIM))

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen."""
        try:
            self._canvas.unbind_all("<MouseWheel>")
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")
        except Exception:
            pass

        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
            self._save_after_id = None
            self._do_save_settings()
        self.win.destroy()
