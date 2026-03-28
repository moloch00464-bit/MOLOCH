#!/usr/bin/env python3
"""
M.O.L.O.C.H. NPU Threshold & MPO Popup — Hailo-10H + Persoenlichkeit
======================================================================

Eigenstaendiges Toplevel-Fenster mit Scrollbar fuer:

1. NPU Modell-Schwellwerte:
   - SCRFD Confidence, SCRFD NMS
   - ArcFace Aehnlichkeit
   - YOLOv8m Confidence
   - Pose Confidence
   - Hand Landmark Confidence

2. M.O.L.O.C.H. Persoenlichkeits-Dynamik (MPO 5.2/5.3):
   - Tension Decay Rate (tau)
   - Dominance Drift Speed
   - Hysterese Schwelle (Zone-Wechsel)
   - Berserker Trigger Level
   - Thermal Damping Start

3. Gestenerkennung:
   - Aktive Gesten (Checkboxen)
   - Gesten-Sensitivity

Alle Aenderungen sofort live, persistent in settings.json.
Importiert NUR panel_styles, tkinter, json, os.
"""

import json
import os
import tempfile
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_YELLOW, STATUS_GREEN, ACCENT_RED, ACCENT_GREEN,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

# Settings-Pfad
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")

# ============================================================================
# Modell-Definitionen: Per-Model Gruppierung mit Status
# (model_name, hef_info, active_default, sliders[])
#   slider: (Anzeigename, settings-sektion, settings-key,
#            min, max, default, schritt, einheit, tooltip)
# ============================================================================

MODEL_DEFS = [
    # --- TAPPAS Pipeline Modelle (immer aktiv, NPU shared) ---
    ("SCRFD 10G (Face)", "scrfd_10g.hef (5.8 MB)", True, [
        ("Confidence", "thresholds", "scrfd_conf",
         0.1, 0.9, 0.5, 0.05, "", "Gesichtserkennung (hoeher = weniger Fehlalarme)"),
        ("NMS Ueberlappung", "thresholds", "scrfd_nms",
         0.1, 0.9, 0.4, 0.05, "", "Doppel-Detections filtern (hoeher = aggressiver)"),
    ]),
    ("ArcFace MobileFaceNet", "arcface_mobilefacenet.hef (2.6 MB)", True, [
        ("Aehnlichkeit", "thresholds", "arcface_thresh",
         0.3, 0.9, 0.65, 0.05, "", "Wie aehnlich muss ein Gesicht sein? (hoeher = strenger)"),
    ]),
    ("YOLOv8m Person", "yolov8m_h10.hef (21 MB)", True, [
        ("Confidence", "thresholds", "yolo_conf",
         0.1, 0.9, 0.5, 0.05, "", "Person Detection (hoeher = weniger Fehlalarme)"),
    ]),
    ("Face Attributes", "face_attr_resnet_v1_18.hef (1.2 MB)", True, [
        # Kein eigener Threshold — laeuft immer mit SCRFD
    ]),
    ("Pose YOLOv8s", "yolov8s_pose_h10.hef (14 MB)", True, [
        ("Confidence", "thresholds", "pose_conf",
         0.1, 0.9, 0.6, 0.05, "", "Pose/Keypoint Detection Confidence"),
    ]),
    ("Person-ReID", "repvgg_a0_person_reid_512.hef (5.1 MB)", True, [
        # Kein eigener Threshold — Embedding-Matching intern
    ]),
    # --- Optionale Modelle (Valve-gated) ---
    ("Hand Landmark", "hand_landmark_lite.hef (5.3 MB)", False, [
        ("Confidence", "thresholds", "hand_conf",
         0.1, 0.9, 0.65, 0.05, "", "Hand/Gesten Detection Confidence"),
    ]),
]

# Flache Liste aller Threshold-Slider (fuer Save/Load Kompatibilitaet)
THRESHOLD_DEFS = []
for _mname, _hef, _active, sliders in MODEL_DEFS:
    THRESHOLD_DEFS.extend(sliders)

# MPO Persoenlichkeits-Dynamik (5.2/5.3)
MPO_DEFS = [
    ("Tension Decay (tau)", "mpo", "tension_tau",
     100, 600, 300, 10, "s", "Wie schnell baut sich Anspannung ab (hoeher = langsamer)"),
    ("Dominance Drift", "mpo", "dominance_drift",
     0.001, 0.050, 0.010, 0.001, "/min", "Drift-Geschwindigkeit Richtung Guardian (hoeher = schneller)"),
    ("Hysterese Schwelle", "mpo", "zone_hysteresis",
     0.05, 0.30, 0.15, 0.01, "", "Min. Dominance-Delta fuer Persoenlichkeits-Wechsel"),
    ("Berserker Trigger", "mpo", "berserker_threshold",
     0.85, 1.00, 0.95, 0.01, "", "Tension-Level fuer Berserker-Aktivierung"),
    ("Thermal Damping Start", "mpo", "thermal_damping_start",
     50, 80, 70, 1, "C", "CPU-Temperatur ab der Tension gedaempft wird"),
]

# Gesten-Definitionen: (Anzeigename, settings-key, default)
GESTURE_DEFS = [
    ("Winken", "wave_enabled", True),
    ("Daumen hoch", "thumbs_up_enabled", True),
    ("Stop/Halt", "stop_enabled", True),
    ("Zeigen", "point_enabled", True),
]


class NpuThreshPopup:
    """NPU Threshold + MPO + Gesten Popup mit Scrollbar."""

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy
        self._save_after_id = None

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("NPU & MPO Einstellungen")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("480x750")
        self.win.resizable(False, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Scrollbare Flaeche
        self._build_scrollable()

        # Slider-Variablen: {(section, key): tk.DoubleVar}
        self._vars = {}
        self._labels = {}

        # Gesten-Variablen: {key: tk.BooleanVar}
        self._gesture_vars = {}
        self._gesture_sensitivity = None

        # GUI-Sektionen aufbauen
        self._build_threshold_section()
        self._build_mpo_section()
        self._build_gesture_section()
        self._build_buttons()

        # Aktuelle Werte laden
        self._load_current_values()

        # Scroll-Region aktualisieren
        self._inner.update_idletasks()
        self._canvas.config(scrollregion=self._canvas.bbox("all"))

    # =========================================================================
    # Scrollbare Flaeche
    # =========================================================================

    def _build_scrollable(self):
        """Canvas + Scrollbar + innerer Frame fuer scrollbare Inhalte."""
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

        # Breite des inneren Frames an Canvas anpassen
        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._canvas.bind("<Configure>", _on_canvas_configure)

        # Scroll-Region aktualisieren bei Groessenaenderung
        def _on_inner_configure(_event):
            self._canvas.config(scrollregion=self._canvas.bbox("all"))
        self._inner.bind("<Configure>", _on_inner_configure)

        # Mausrad-Scrollen
        def _on_mousewheel(event):
            self._canvas.yview_scroll(-1 * (event.delta // 120 or (-1 if event.num == 4 else 1)), "units")
        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._canvas.bind_all("<Button-4>", _on_mousewheel)
        self._canvas.bind_all("<Button-5>", _on_mousewheel)

    # =========================================================================
    # NPU Modell-Sektionen mit Status-LEDs
    # =========================================================================

    def _build_threshold_section(self):
        """Per-Model Sektionen mit Status-LED, HEF-Info und Threshold-Slidern."""
        # Aktive Modelle aus settings.json laden
        active_models = []
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)
                active_models = data.get("active_models", [])
        except Exception:
            pass

        # Mapping: model_name → active (aus active_models Liste)
        active_map = {
            "SCRFD 10G (Face)": "scrfd" in active_models,
            "ArcFace MobileFaceNet": "arcface" in active_models,
            "YOLOv8m Person": "yolov8m" in active_models,
            "Face Attributes": "scrfd" in active_models,  # laeuft immer mit SCRFD
            "Pose YOLOv8s": "pose" in active_models,
            "Person-ReID": "person_reid" in active_models,
            "Hand Landmark": "hand_landmark" in active_models,
        }

        for model_name, hef_info, _default_active, sliders in MODEL_DEFS:
            is_active = active_map.get(model_name, _default_active)

            # Model-Section als LabelFrame
            section = tk.LabelFrame(
                self._inner, text=model_name,
                bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
            )
            section.pack(fill=tk.X, padx=10, pady=(5, 2))

            # Status-Zeile: LED + HEF-Info
            status_row = tk.Frame(section, bg=BG_FRAME)
            status_row.pack(fill=tk.X, padx=8, pady=(3, 2))

            led_color = ACCENT_GREEN if is_active else ACCENT_RED
            led_text = "AKTIV" if is_active else "INAKTIV"
            tk.Label(
                status_row, text="\u25CF", fg=led_color, bg=BG_FRAME,
                font=FONT_LABEL,
            ).pack(side=tk.LEFT)
            tk.Label(
                status_row, text=f" {led_text}", fg=led_color, bg=BG_FRAME,
                font=FONT_SMALL,
            ).pack(side=tk.LEFT)
            tk.Label(
                status_row, text=f"  {hef_info}", fg=FG_DIM, bg=BG_FRAME,
                font=FONT_SMALL,
            ).pack(side=tk.LEFT, padx=(10, 0))

            # Slider fuer dieses Modell (ausgegraut wenn inaktiv)
            for name, sec, key, vmin, vmax, default, step, unit, tip in sliders:
                self._build_slider_row(
                    section, name, sec, key, vmin, vmax, default, step, unit, tip,
                    enabled=is_active,
                )

    # =========================================================================
    # MPO Persoenlichkeits-Dynamik Section
    # =========================================================================

    def _build_mpo_section(self):
        """MPO Slider: Tension, Dominance, Berserker, Thermal."""
        section = tk.LabelFrame(
            self._inner, text="Persoenlichkeits-Dynamik (MPO)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        for name, sec, key, vmin, vmax, default, step, unit, tip in MPO_DEFS:
            self._build_slider_row(section, name, sec, key, vmin, vmax, default, step, unit, tip)

    # =========================================================================
    # Gesten Section
    # =========================================================================

    def _build_gesture_section(self):
        """Gestenerkennung: Checkboxen + Sensitivity Slider."""
        section = tk.LabelFrame(
            self._inner, text="Gestenerkennung",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=5)

        # Checkboxen fuer aktive Gesten
        cb_row = tk.Frame(section, bg=BG_FRAME)
        cb_row.pack(fill=tk.X, padx=8, pady=(5, 3))

        for i, (label, key, default) in enumerate(GESTURE_DEFS):
            var = tk.BooleanVar(value=default)
            self._gesture_vars[key] = var
            cb = tk.Checkbutton(
                cb_row, text=label, variable=var,
                bg=BG_FRAME, fg=FG_WHITE, selectcolor=BG_FRAME,
                activebackground=BG_FRAME, activeforeground=FG_WHITE,
                font=FONT_SMALL,
                command=self._on_gesture_changed,
            )
            cb.grid(row=0, column=i, padx=4, pady=2)

        # Sensitivity Slider
        self._gesture_sensitivity = tk.DoubleVar(value=0.5)
        self._build_slider_row(
            section, "Sensitivity", "gestures", "sensitivity",
            0.1, 1.0, 0.5, 0.05, "", "Gesten-Empfindlichkeit (hoeher = empfindlicher)")

    # =========================================================================
    # Buttons (Save + Reset)
    # =========================================================================

    def _build_buttons(self):
        """Reset Button + Feedback Label am unteren Rand."""
        btn_frame = tk.Frame(self._inner, bg=BG_DARK)
        btn_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        tk.Button(
            btn_frame, text="RESET DEFAULTS", width=14,
            bg=ACCENT_RED, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground="#cc0000",
            command=self._reset_defaults,
        ).pack(side=tk.LEFT, padx=5)

        # Feedback Label
        self._lbl_feedback = tk.Label(
            btn_frame, text="Auto-Save aktiv", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_feedback.pack(side=tk.LEFT, padx=10)

    # =========================================================================
    # Slider-Zeile bauen (generisch)
    # =========================================================================

    def _build_slider_row(self, parent, name, section, key, vmin, vmax,
                          default, step, unit, tooltip, enabled=True):
        """Eine Slider-Zeile: Label + Slider + Live-Wert. Ausgegraut wenn enabled=False."""
        row = tk.Frame(parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=2)

        fg = FG_LABEL if enabled else FG_DIM

        # Name links
        tk.Label(
            row, text=name, width=22, anchor=tk.W,
            bg=BG_FRAME, fg=fg, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        # Wert-Label rechts
        fmt = self._format_value(default, unit, step)
        lbl = tk.Label(
            row, text=fmt, width=9, anchor=tk.E,
            bg=BG_FRAME, fg=STATUS_YELLOW if enabled else FG_DIM, font=FONT_LABEL,
        )
        lbl.pack(side=tk.RIGHT)
        self._labels[(section, key)] = lbl

        # Slider
        resolution = step if isinstance(default, float) else int(step)
        var = tk.DoubleVar(value=default)
        self._vars[(section, key)] = var

        state = tk.NORMAL if enabled else tk.DISABLED
        slider = tk.Scale(
            row, from_=vmin, to=vmax, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var,
            bg=BG_FRAME, fg=FG_WHITE if enabled else FG_DIM,
            troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False, state=state,
            command=lambda val, s=section, k=key, u=unit, st=step: self._on_slider_changed(s, k, val, u, st),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Tooltip unter dem Slider (FG_DIM zu dunkel auf BG_FRAME → heller)
        if tooltip:
            tk.Label(
                parent, text=tooltip, anchor=tk.W,
                bg=BG_FRAME, fg="#999999", font=FONT_SMALL,
            ).pack(fill=tk.X, padx=16, pady=(0, 2))

    # =========================================================================
    # Wert-Formatierung
    # =========================================================================

    @staticmethod
    def _format_value(val, unit="", step=0.01):
        """Wert formatiert als String mit Einheit."""
        if isinstance(val, float):
            # Dezimalstellen aus step ableiten
            if step >= 1:
                return f"{val:.0f}{unit}"
            elif step >= 0.1:
                return f"{val:.1f}{unit}"
            elif step >= 0.01:
                return f"{val:.2f}{unit}"
            else:
                return f"{val:.3f}{unit}"
        return f"{int(val)}{unit}"

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_slider_changed(self, section, key, value, unit, step):
        """Slider geaendert: Label updaten, Service senden, speichern."""
        val = float(value)
        self._labels[(section, key)].config(
            text=self._format_value(val, unit, step))

        # Service-Command senden (je nach Sektion)
        if section == "thresholds":
            self.service._write_command("action", {
                "action": "set_threshold",
                "attr": key,
                "value": val,
            })
        elif section == "mpo":
            self.service._write_command("action", {
                "action": "set_mpo_param",
                "param": key,
                "value": val,
            })
        elif section == "gestures":
            self.service._write_command("action", {
                "action": "set_gesture_param",
                "param": key,
                "value": val,
            })

        self._save_settings()

    def _on_gesture_changed(self):
        """Gesten-Checkbox geaendert: Service senden, speichern."""
        gesture_state = {}
        for key, var in self._gesture_vars.items():
            gesture_state[key] = var.get()

        self.service._write_command("action", {
            "action": "set_gesture_params",
            **gesture_state,
        })
        self._save_settings()

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

        # Alle Slider setzen
        all_defs = list(THRESHOLD_DEFS) + list(MPO_DEFS)
        for name, sec, key, vmin, vmax, default, step, unit, tip in all_defs:
            section_data = data.get(sec, {})
            if isinstance(section_data, dict):
                raw = section_data.get(key)
            else:
                raw = None

            if raw is not None:
                try:
                    val = max(vmin, min(vmax, float(raw)))
                except (TypeError, ValueError):
                    val = default
            else:
                val = default

            var_key = (sec, key)
            if var_key in self._vars:
                self._vars[var_key].set(val)
                self._labels[var_key].config(
                    text=self._format_value(val, unit, step))

        # Gesten-Sensitivity laden
        gestures = data.get("gestures", {})
        if isinstance(gestures, dict):
            sens_key = ("gestures", "sensitivity")
            if sens_key in self._vars:
                raw = gestures.get("sensitivity", 0.5)
                try:
                    val = max(0.1, min(1.0, float(raw)))
                except (TypeError, ValueError):
                    val = 0.5
                self._vars[sens_key].set(val)
                self._labels[sens_key].config(
                    text=self._format_value(val, "", 0.05))

            # Gesten-Checkboxen laden
            for _label, key, default in GESTURE_DEFS:
                if key in self._gesture_vars:
                    val = gestures.get(key, default)
                    self._gesture_vars[key].set(bool(val))

    # =========================================================================
    # Settings persistent speichern (debounced 300ms)
    # =========================================================================

    def _save_settings(self):
        """Save nach 300ms Debounce."""
        if self._save_after_id is not None:
            self.win.after_cancel(self._save_after_id)
        self._save_after_id = self.win.after(300, self._do_save_settings)

    def _do_save_settings(self):
        """Alle Werte atomar in settings.json schreiben."""
        self._save_after_id = None
        try:
            data = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r") as f:
                    data = json.load(f)

            # Thresholds updaten
            thresholds = data.get("thresholds", {})
            for _name, sec, key, _vmin, _vmax, _default, _step, _unit, _tip in THRESHOLD_DEFS:
                var_key = (sec, key)
                if var_key in self._vars:
                    thresholds[key] = round(self._vars[var_key].get(), 3)
            data["thresholds"] = thresholds

            # MPO updaten
            mpo = data.get("mpo", {})
            for _name, sec, key, _vmin, _vmax, _default, _step, _unit, _tip in MPO_DEFS:
                var_key = (sec, key)
                if var_key in self._vars:
                    mpo[key] = round(self._vars[var_key].get(), 3)
            data["mpo"] = mpo

            # Gesten updaten
            gestures = data.get("gestures", {})
            for _label, key, _default in GESTURE_DEFS:
                if key in self._gesture_vars:
                    gestures[key] = self._gesture_vars[key].get()
            sens_key = ("gestures", "sensitivity")
            if sens_key in self._vars:
                gestures["sensitivity"] = round(self._vars[sens_key].get(), 2)
            data["gestures"] = gestures

            # Atomar schreiben
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
    # Reset Defaults
    # =========================================================================

    def _reset_defaults(self):
        """Alle Slider auf Defaults zuruecksetzen."""
        all_defs = list(THRESHOLD_DEFS) + list(MPO_DEFS)
        for name, sec, key, vmin, vmax, default, step, unit, tip in all_defs:
            var_key = (sec, key)
            if var_key in self._vars:
                self._vars[var_key].set(default)
                self._labels[var_key].config(
                    text=self._format_value(default, unit, step))

        # Gesten Defaults
        for _label, key, default in GESTURE_DEFS:
            if key in self._gesture_vars:
                self._gesture_vars[key].set(default)

        sens_key = ("gestures", "sensitivity")
        if sens_key in self._vars:
            self._vars[sens_key].set(0.5)
            self._labels[sens_key].config(text="0.50")

        self._do_save_settings()
        self._lbl_feedback.config(text="Defaults geladen!", fg=STATUS_YELLOW)
        self.win.after(2000, lambda: self._lbl_feedback.config(text="", fg=FG_DIM))

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen."""
        # Mousewheel Bindings entfernen
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
