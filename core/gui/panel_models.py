#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Models
===========================

Model Steuerung und Popup-Buttons.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Model Checkboxen: SCRFD, ArcFace, YOLOv8m, Hand Landmark, Pose
- FPS Anzeige (STATUS_YELLOW, 500ms Update)
- SAVE SETTINGS Button
- Popup-Buttons Reihe: AUDIO, HARDWARE, NPU THRESH, SETTINGS

Alle Commands via ServiceProxy._write_command().
Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON,
    BTN_ON_GREEN, BTN_OFF_DARK,
    ACCENT_GREEN, ACCENT_CYAN,
    STATUS_YELLOW,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL, FONT_MONO,
    STATUS_UPDATE_MS,
)

# Audio Popup importieren (optional, Fehler faengt Panel nicht ab)
try:
    from core.gui.popups.popup_audio import AudioPopup
except ImportError:
    AudioPopup = None

# Hardware Popup importieren (optional)
try:
    from core.gui.popups.popup_hardware import HardwarePopup
except ImportError:
    HardwarePopup = None

# NPU Thresh Popup importieren (optional, NEUES Popup mit MPO + Gesten)
try:
    from core.gui.popups.popup_npu_thresh import NpuThreshPopup
except ImportError:
    NpuThreshPopup = None

# Tracker Popup importieren (optional)
try:
    from core.gui.popups.popup_tracker import TrackerPopup
except ImportError:
    TrackerPopup = None

# Settings Popup importieren (optional)
try:
    from core.gui.popups.popup_settings import SettingsPopup
except ImportError:
    SettingsPopup = None

# Dashboard Popup importieren (optional)
try:
    from core.gui.popups.popup_dashboard import DashboardPopup
except ImportError:
    DashboardPopup = None

# Whisper Popup importieren (optional)
try:
    from core.gui.popups.popup_whisper import WhisperPopup
except ImportError:
    WhisperPopup = None


class ModelsModule:
    """Model Controls und Popup-Buttons im uebergebenen LabelFrame."""

    # TAPPAS Pipeline Modelle (immer aktiv, nicht togglebar)
    TAPPAS_MODELS = [
        ("SCRFD", "scrfd"),
        ("ArcFace", "arcface"),
        ("YOLOv8m", "yolov8m"),
    ]

    # Zusaetzliche Modelle (HEFs vorhanden, noch nicht in Pipeline)
    EXTRA_MODELS = [
        ("Pose", "pose"),
        ("Hand LM", "hand_landmark"),
    ]

    # Alle Modelle zusammen (fuer Kompatibilitaet)
    MODELS = TAPPAS_MODELS + EXTRA_MODELS

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None

        # Checkbox-Variablen
        self._model_vars = {}
        for _, key in self.MODELS:
            self._model_vars[key] = tk.BooleanVar(value=False)

        # Learner Flash State
        self._learner_flash = False

        # Popup-Callbacks (werden von panel_main spaeter gesetzt)
        self.on_popup_audio = self._open_audio_popup
        self.on_popup_hardware = self._open_hardware_popup
        self.on_popup_npu = self._open_npu_popup
        self.on_popup_tracker = self._open_tracker_popup
        self.on_popup_settings = self._open_settings_popup
        self.on_popup_dashboard = self._open_dashboard_popup
        self.on_popup_whisper = self._open_whisper_popup

        # GUI aufbauen
        self._build_pipeline_status()
        self._build_model_checkboxes()
        self._build_fps_display()
        self._build_save_button()
        self._build_popup_buttons()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # Audio Popup oeffnen
    # =========================================================================

    def _open_audio_popup(self):
        """Audio Settings Popup oeffnen."""
        if AudioPopup is not None:
            AudioPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_audio.py konnte nicht importiert werden")

    # =========================================================================
    # Hardware Popup oeffnen
    # =========================================================================

    def _open_hardware_popup(self):
        """Hardware Monitor Popup oeffnen."""
        if HardwarePopup is not None:
            HardwarePopup(self._parent, self._service)
        else:
            print("FEHLER: popup_hardware.py konnte nicht importiert werden")

    # =========================================================================
    # NPU Thresh Popup oeffnen
    # =========================================================================

    def _open_npu_popup(self):
        """NPU Thresholds Popup oeffnen."""
        if NpuThreshPopup is not None:
            NpuThreshPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_npu.py konnte nicht importiert werden")

    # =========================================================================
    # Settings Popup oeffnen
    # =========================================================================

    def _open_tracker_popup(self):
        """Tracker Settings Popup oeffnen."""
        if TrackerPopup is not None:
            TrackerPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_tracker.py konnte nicht importiert werden")

    def _open_settings_popup(self):
        """Settings Popup oeffnen."""
        if SettingsPopup is not None:
            SettingsPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_settings.py konnte nicht importiert werden")

    def _open_dashboard_popup(self):
        """NPU Dashboard Popup oeffnen."""
        if DashboardPopup is not None:
            DashboardPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_dashboard.py konnte nicht importiert werden")

    def _open_whisper_popup(self):
        """Whisper STT Monitor Popup oeffnen."""
        if WhisperPopup is not None:
            WhisperPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_whisper.py konnte nicht importiert werden")

    # =========================================================================
    # Pipeline Status (TAPPAS/Legacy Anzeige)
    # =========================================================================

    def _build_pipeline_status(self):
        """TAPPAS/Legacy Pipeline Status-Zeile."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=10, pady=(5, 0))

        tk.Label(
            row, text="Pipeline:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._lbl_pipeline = tk.Label(
            row, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_pipeline.pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # Model Checkboxen
    # =========================================================================

    def _build_model_checkboxes(self):
        """Checkbuttons: TAPPAS-Modelle (locked) + Extra-Modelle (togglebar)."""
        section = tk.LabelFrame(
            self._parent,
            text="Modelle",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Zeile 1: TAPPAS Pipeline Modelle (immer aktiv)
        row_tappas = tk.Frame(section, bg=BG_FRAME)
        row_tappas.pack(pady=(5, 0), padx=5)

        self._checkboxes = {}
        for i, (label, key) in enumerate(self.TAPPAS_MODELS):
            cb = tk.Checkbutton(
                row_tappas, text=label,
                variable=self._model_vars[key],
                bg=BG_FRAME, fg=ACCENT_GREEN,
                selectcolor=BG_FRAME,
                activebackground=BG_FRAME,
                activeforeground=ACCENT_GREEN,
                disabledforeground=ACCENT_GREEN,
                font=FONT_BUTTON,
                state=tk.DISABLED,
            )
            cb.grid(row=0, column=i, padx=6, pady=2)
            self._checkboxes[key] = cb

        # Label "TAPPAS" rechts neben den Checkboxen
        tk.Label(
            row_tappas, text="TAPPAS",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).grid(row=0, column=len(self.TAPPAS_MODELS), padx=(10, 0))

        # Zeile 2: Extra-Modelle (togglebar)
        if self.EXTRA_MODELS:
            row_extra = tk.Frame(section, bg=BG_FRAME)
            row_extra.pack(pady=(0, 5), padx=5)

            for i, (label, key) in enumerate(self.EXTRA_MODELS):
                cb = tk.Checkbutton(
                    row_extra, text=label,
                    variable=self._model_vars[key],
                    bg=BG_FRAME, fg=FG_WHITE,
                    selectcolor=BG_FRAME,
                    activebackground=BG_FRAME,
                    activeforeground=FG_WHITE,
                    font=FONT_BUTTON,
                    command=lambda k=key: self._toggle_model(k),
                )
                cb.grid(row=0, column=i, padx=6, pady=2)
                self._checkboxes[key] = cb

    def _toggle_model(self, model_key):
        """Model an/aus und Command senden."""
        enabled = self._model_vars[model_key].get()
        self._service._write_command("toggle_model", {
            "model": model_key,
            "enabled": enabled,
        })

    # =========================================================================
    # FPS Anzeige
    # =========================================================================

    def _build_fps_display(self):
        """FPS Anzeige: Total + pro Modell."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            row, text="FPS:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._lbl_fps = tk.Label(
            row, text="--",
            bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_MONO,
        )
        self._lbl_fps.pack(side=tk.LEFT, padx=5)

        # Detail-FPS (scrfd/arcface/yolo)
        self._lbl_fps_detail = tk.Label(
            row, text="",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_fps_detail.pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # SAVE SETTINGS
    # =========================================================================

    def _build_save_button(self):
        """SAVE SETTINGS Button + BLITZ Toggle + Feedback-Label."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            row, text="SAVE SETTINGS", width=16,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._save_settings,
        ).pack(side=tk.LEFT, padx=5)

        # BLITZ Toggle: Weisse LED blinkt bei Learner-Snapshot
        self._btn_blitz = tk.Button(
            row, text="BLITZ", width=8,
            bg=BTN_OFF_DARK, fg=FG_DIM, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_learner_flash,
        )
        self._btn_blitz.pack(side=tk.LEFT, padx=5)

        self._lbl_save = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save.pack(side=tk.LEFT, padx=5)

    def _toggle_learner_flash(self):
        """Learner Flash umschalten und Command senden."""
        self._learner_flash = not self._learner_flash
        self._service._write_command("toggle_learner_flash", {
            "on": self._learner_flash,
        })
        self._update_blitz_button()

    def _update_blitz_button(self):
        """BLITZ Button Farbe aktualisieren."""
        if self._learner_flash:
            self._btn_blitz.config(bg=ACCENT_CYAN, fg=BG_FRAME)
        else:
            self._btn_blitz.config(bg=BTN_OFF_DARK, fg=FG_DIM)

    def _save_settings(self):
        """Settings speichern und kurzes Feedback zeigen."""
        self._service.save_settings()
        self._lbl_save.config(text="Gespeichert!", fg=ACCENT_GREEN)
        self._parent.after(2000, lambda: self._lbl_save.config(
            text="", fg=FG_DIM
        ))

    # =========================================================================
    # Popup-Buttons
    # =========================================================================

    def _build_popup_buttons(self):
        """Popup-Buttons: AUDIO, HARDWARE, NPU THRESH, SETTINGS."""
        section = tk.LabelFrame(
            self._parent,
            text="Popups",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        popup_defs = [
            ("AUDIO", lambda: self.on_popup_audio()),
            ("HARDWARE", lambda: self.on_popup_hardware()),
            ("NPU/MPO", lambda: self.on_popup_npu()),
            ("TRACKER", lambda: self.on_popup_tracker()),
            ("WHISPER", lambda: self.on_popup_whisper()),
            ("DASHBOARD", lambda: self.on_popup_dashboard()),
            ("SETTINGS", lambda: self.on_popup_settings()),
        ]

        for i, (label, cmd) in enumerate(popup_defs):
            tk.Button(
                row, text=label, width=9,
                bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=cmd,
            ).grid(row=i // 4, column=i % 4, padx=2, pady=2)

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Status vom Service lesen: Pipeline, FPS und aktive Modelle aktualisieren."""
        status = self._service.read_status()

        if status:
            # Pipeline-Typ (TAPPAS oder Legacy)
            active_models = status.get("active_models", [])
            is_tappas = len(active_models) >= 3 and "scrfd" in active_models
            if is_tappas:
                self._lbl_pipeline.config(text="TAPPAS", fg=ACCENT_GREEN)
            else:
                self._lbl_pipeline.config(text="Legacy", fg=STATUS_YELLOW)

            # FPS Total
            try:
                fps_dict = status.get("fps", {})
                fps = float(fps_dict.get("total", 0.0)) if isinstance(fps_dict, dict) else 0.0
                self._lbl_fps.config(text=f"{fps:.1f}")

                # Detail-FPS pro Modell
                if isinstance(fps_dict, dict):
                    parts = []
                    for name in ["scrfd", "arcface", "yolov8m"]:
                        val = fps_dict.get(name, 0.0)
                        if val and float(val) > 0:
                            parts.append(f"{name}:{float(val):.0f}")
                    self._lbl_fps_detail.config(text="  ".join(parts))
            except (TypeError, ValueError):
                self._lbl_fps.config(text="--")
                self._lbl_fps_detail.config(text="")

            # Model-Checkboxen Status synchronisieren
            status_key_map = {
                "scrfd": "scrfd_active",
                "arcface": "arcface_active",
                "yolov8m": "yolo_active",
                "hand_landmark": "hand_active",
                "pose": "pose_active",
            }
            for _, key in self.MODELS:
                try:
                    status_key = status_key_map.get(key, key)
                    active = bool(status.get(status_key, False))
                    if self._model_vars[key].get() != active:
                        self._model_vars[key].set(active)
                except (TypeError, ValueError):
                    pass

            # BLITZ Button State vom Service synchronisieren
            flash_active = bool(status.get("learner_flash", False))
            if self._learner_flash != flash_active:
                self._learner_flash = flash_active
                self._update_blitz_button()

        # Widgets sofort neu zeichnen
        self._parent.update_idletasks()

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)
