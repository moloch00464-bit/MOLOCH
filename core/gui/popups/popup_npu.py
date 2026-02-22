#!/usr/bin/env python3
"""
M.O.L.O.C.H. NPU Einstellungen Popup — Hailo-10H
===================================================

Eigenstaendiges Toplevel-Fenster fuer NPU Modell-Schwellwerte
und Hand-Verdeckung Parameter.

Threshold Slider:
- SCRFD Erkennungsschwelle (0.1-0.9, default 0.5, Schritt 0.05)
- SCRFD Ueberlappungsfilter (0.1-0.9, default 0.4, Schritt 0.05)
- ArcFace Aehnlichkeit (0.3-0.9, default 0.6, Schritt 0.05)
- YOLOv8m Erkennungsschwelle (0.1-0.9, default 0.5, Schritt 0.05)

Hand-Verdeckung:
- Zeitlimit (1-10s), Trefferfolge (1-10), Aktualitaet (0.5-5.0s)

Aenderungen sofort via _write_command an Service gesendet
und persistent in settings.json gespeichert.

Importiert NUR panel_styles, tkinter, json, os.
"""

import json
import os
import tempfile
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_YELLOW,
    FONT_TITLE, FONT_LABEL, FONT_SMALL,
)

# Settings-Pfad
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")

# Threshold Definitionen: (Anzeigename, settings-key, min, max, default, schritt)
THRESHOLD_DEFS = [
    ("SCRFD Erkennung", "scrfd_conf", 0.1, 0.9, 0.5, 0.05),
    ("SCRFD Überlappung", "scrfd_nms", 0.1, 0.9, 0.4, 0.05),
    ("ArcFace Ähnlichkeit", "arcface_thresh", 0.3, 0.9, 0.6, 0.05),
    ("YOLOv8m Erkennung", "yolo_conf", 0.1, 0.9, 0.5, 0.05),
]

# Hand-Occlusion Definitionen: (Anzeigename, settings-key, min, max, default, schritt, einheit)
HAND_DEFS = [
    ("Zeitlimit", "timeout", 1.0, 10.0, 5.0, 0.5, "s"),
    ("Trefferfolge", "streak", 1, 10, 3, 1, ""),
    ("Aktualität", "recency", 0.5, 5.0, 2.0, 0.5, "s"),
]

# Tooltip-Beschreibungen pro Slider (key -> text)
TOOLTIPS = {
    "scrfd_conf": "Höher = weniger Fehlerkennungen, niedriger = mehr Gesichter",
    "scrfd_nms": "Filtert doppelte Erkennungen (höher = aggressiver)",
    "arcface_thresh": "Wie ähnlich muss ein Gesicht sein? (höher = strenger)",
    "yolo_conf": "Höher = weniger Fehlerkennungen, niedriger = mehr Personen",
    "timeout": "Sekunden bis Hand-Modus deaktiviert wird",
    "streak": "Frames in Folge mit Hand bevor Modus wechselt",
    "recency": "Zeitfenster für letzte Erkennung",
}


class NpuThreshPopup:
    """NPU Threshold Settings als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
            service_proxy: ServiceProxy Instanz fuer Commands
        """
        self.parent = parent
        self.service = service_proxy

        # Debounced Save Timer
        self._save_after_id = None

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("NPU Einstellungen \u2014 Hailo-10H")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("420x480")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Slider-Variablen
        self._thresh_vars = {}
        self._thresh_labels = {}
        self._hand_vars = {}
        self._hand_labels = {}

        # GUI aufbauen
        self._build_threshold_section()
        self._build_hand_section()

        # Aktuelle Werte laden
        self._load_current_values()

    # =========================================================================
    # Threshold Slider Section
    # =========================================================================

    def _build_threshold_section(self):
        """Threshold Slider fuer alle AI-Modelle."""
        section = tk.LabelFrame(
            self.win, text="Modell-Schwellwerte",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(10, 5))

        for name, key, vmin, vmax, default, step in THRESHOLD_DEFS:
            self._build_slider_row(
                section, name, key, vmin, vmax, default, step,
                self._thresh_vars, self._thresh_labels,
                self._on_threshold_changed,
                tooltip=TOOLTIPS.get(key, ""),
            )

    # =========================================================================
    # Hand-Occlusion Section
    # =========================================================================

    def _build_hand_section(self):
        """Hand-Occlusion Parameter Slider."""
        section = tk.LabelFrame(
            self.win, text="Hand-Verdeckung",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(5, 10))

        for name, key, vmin, vmax, default, step, unit in HAND_DEFS:
            self._build_slider_row(
                section, name, key, vmin, vmax, default, step,
                self._hand_vars, self._hand_labels,
                self._on_hand_changed, unit=unit,
                tooltip=TOOLTIPS.get(key, ""),
            )

    # =========================================================================
    # Slider-Zeile bauen (wiederverwendbar)
    # =========================================================================

    def _build_slider_row(self, parent, name, key, vmin, vmax, default, step,
                          var_dict, label_dict, callback, unit="", tooltip=""):
        """Eine Slider-Zeile: Label links, Slider mitte, Wert rechts."""
        row = tk.Frame(parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=3)

        # Name links
        tk.Label(
            row, text=name, width=24, anchor=tk.W,
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        # Wert-Label rechts
        fmt = f"{default:.2f}" if isinstance(default, float) else str(default)
        lbl = tk.Label(
            row, text=fmt + unit, width=7, anchor=tk.E,
            bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        lbl.pack(side=tk.RIGHT)
        label_dict[key] = lbl

        # Slider
        # Fuer int-Werte (streak) resolution als int
        resolution = step if isinstance(default, float) else int(step)
        var = tk.DoubleVar(value=default)
        var_dict[key] = var

        slider = tk.Scale(
            row, from_=vmin, to=vmax, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=lambda val, k=key, u=unit: callback(k, val, u),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Tooltip-Beschreibung unter dem Slider
        if tooltip:
            tk.Label(
                parent, text=tooltip, anchor=tk.W,
                bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
            ).pack(fill=tk.X, padx=16, pady=(0, 2))

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_threshold_changed(self, key, value, _unit=""):
        """Threshold geaendert — Label updaten, Service senden, speichern."""
        val = float(value)
        self._thresh_labels[key].config(text=f"{val:.2f}")
        self.service._write_command("action", {
            "action": "set_threshold",
            "attr": key,
            "value": val,
        })
        self._save_settings()

    def _on_hand_changed(self, key, value, unit=""):
        """Hand-Occlusion geaendert — Label updaten, Service senden, speichern."""
        val = float(value)
        # Streak als int darstellen
        if key == "streak":
            self._hand_labels[key].config(text=f"{int(val)}{unit}")
        else:
            self._hand_labels[key].config(text=f"{val:.1f}{unit}")
        self.service._write_command("action", {
            "action": "set_hand_params",
            "timeout": float(self._hand_vars["timeout"].get()),
            "streak": int(self._hand_vars["streak"].get()),
            "recency": float(self._hand_vars["recency"].get()),
        })
        self._save_settings()

    # =========================================================================
    # Werte laden
    # =========================================================================

    def _load_current_values(self):
        """Aktuelle Werte aus settings.json laden, sonst defaults."""
        thresholds = {}
        hand = {}

        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
                raw_t = data.get("thresholds")
                if isinstance(raw_t, dict):
                    thresholds = raw_t
                raw_h = data.get("hand_occlusion")
                if isinstance(raw_h, dict):
                    hand = raw_h
        except Exception:
            pass

        # Threshold Slider setzen
        for _name, key, vmin, vmax, default, _step in THRESHOLD_DEFS:
            raw = thresholds.get(key)
            if raw is not None:
                try:
                    val = max(vmin, min(vmax, float(raw)))
                except (TypeError, ValueError):
                    val = default
            else:
                val = default
            self._thresh_vars[key].set(val)
            self._thresh_labels[key].config(text=f"{val:.2f}")

        # Hand-Occlusion Slider setzen
        for _name, key, vmin, vmax, default, _step, unit in HAND_DEFS:
            raw = hand.get(key)
            if raw is not None:
                try:
                    val = max(vmin, min(vmax, float(raw)))
                except (TypeError, ValueError):
                    val = default
            else:
                val = default
            self._hand_vars[key].set(val)
            if key == "streak":
                self._hand_labels[key].config(text=f"{int(val)}{unit}")
            else:
                self._hand_labels[key].config(text=f"{val:.1f}{unit}")

    # =========================================================================
    # Settings persistent speichern (debounced 300ms)
    # =========================================================================

    def _save_settings(self):
        """Save nach 300ms Debounce (verhindert Schreibflut bei Slider)."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
        self._save_after_id = self.win.after(300, self._do_save_settings)

    def _do_save_settings(self):
        """Thresholds und Hand-Occlusion in settings.json atomar schreiben."""
        self._save_after_id = None
        try:
            # Bestehende settings.json lesen
            data = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)

            # Nur thresholds-Sektion updaten
            data["thresholds"] = {
                "scrfd_conf": round(self._thresh_vars["scrfd_conf"].get(), 2),
                "scrfd_nms": round(self._thresh_vars["scrfd_nms"].get(), 2),
                "arcface_thresh": round(self._thresh_vars["arcface_thresh"].get(), 2),
                "yolo_conf": round(self._thresh_vars["yolo_conf"].get(), 2),
            }

            # Nur hand_occlusion-Sektion updaten
            data["hand_occlusion"] = {
                "timeout": round(self._hand_vars["timeout"].get(), 1),
                "streak": int(self._hand_vars["streak"].get()),
                "recency": round(self._hand_vars["recency"].get(), 1),
            }

            # Atomar schreiben: tmp + rename
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
        except Exception:
            pass

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen — ausstehende Saves sofort ausfuehren."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
            self._save_after_id = None
            self._do_save_settings()
        self.win.destroy()
