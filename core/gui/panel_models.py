#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Models
===========================

Model Steuerung und Popup-Buttons.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Model Checkboxen: SCRFD, ArcFace, YOLOv8m, Hand Landmark
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


class ModelsModule:
    """Model Controls und Popup-Buttons im uebergebenen LabelFrame."""

    # Model Definitionen: (Anzeigename, Service-Key)
    MODELS = [
        ("SCRFD", "scrfd"),
        ("ArcFace", "arcface"),
        ("YOLOv8m", "yolov8m"),
        ("Hand LM", "hand_landmark"),
    ]

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

        # Popup-Callbacks (werden von panel_main spaeter gesetzt)
        self.on_popup_audio = lambda: print("TODO: popup_audio")
        self.on_popup_hardware = lambda: print("TODO: popup_hardware")
        self.on_popup_npu = lambda: print("TODO: popup_npu")
        self.on_popup_settings = lambda: print("TODO: popup_settings")

        # GUI aufbauen
        self._build_model_checkboxes()
        self._build_fps_display()
        self._build_save_button()
        self._build_popup_buttons()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # Model Checkboxen
    # =========================================================================

    def _build_model_checkboxes(self):
        """Checkbuttons fuer alle AI-Modelle."""
        section = tk.LabelFrame(
            self._parent,
            text="Modelle",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        self._checkboxes = {}
        for i, (label, key) in enumerate(self.MODELS):
            cb = tk.Checkbutton(
                row, text=label,
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
        """FPS Label in STATUS_YELLOW."""
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

    # =========================================================================
    # SAVE SETTINGS
    # =========================================================================

    def _build_save_button(self):
        """SAVE SETTINGS Button mit Feedback-Label."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            row, text="SAVE SETTINGS", width=16,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._save_settings,
        ).pack(side=tk.LEFT, padx=5)

        self._lbl_save = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save.pack(side=tk.LEFT, padx=5)

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
            ("NPU THRESH", lambda: self.on_popup_npu()),
            ("SETTINGS", lambda: self.on_popup_settings()),
        ]

        for i, (label, cmd) in enumerate(popup_defs):
            tk.Button(
                row, text=label, width=10,
                bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=cmd,
            ).grid(row=0, column=i, padx=3, pady=2)

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Status vom Service lesen: FPS und aktive Modelle aktualisieren."""
        status = self._service.read_status()

        if status:
            # FPS aktualisieren
            fps = status.get("fps")
            if fps is not None:
                self._lbl_fps.config(text=f"{fps:.1f}")
            else:
                self._lbl_fps.config(text="--")

            # Model-Checkboxen aktualisieren
            models = status.get("models", {})
            for _, key in self.MODELS:
                active = models.get(key, False)
                if self._model_vars[key].get() != active:
                    self._model_vars[key].set(active)

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)
