#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Voice
==========================

Voice-Steuerung fuer die rechte Spalte (unter TalkChat).
- Dropdown: Alle 8 Piper TTS Stimmen
- Test-Button: Spricht Testsatz mit gewaehlter Stimme
- Status-Label: Aktuelle Stimme + Zustand

Bekommt parent_frame und ServiceProxy von panel_main.
Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk
from tkinter import ttk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON, BG_INPUT,
    BTN_OFF_DARK, BTN_ON_GREEN,
    ACCENT_CYAN,
    FG_TEXT, FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL,
    STATUS_UPDATE_MS,
)


# Stimmen-Zuordnung: voice_id -> Anzeigename
VOICE_NAMES = {
    "de_DE-thorsten-high": "Thorsten (High)",
    "de_DE-thorsten-medium": "Thorsten (Medium)",
    "de_DE-thorsten-low": "Thorsten (Low)",
    "de_DE-karlsson-low": "Karlsson (Kobold)",
    "de_DE-kerstin-low": "Kerstin",
    "de_DE-ramona-low": "Ramona",
    "de_DE-pavoque-low": "Pavoque",
    "de_DE-eva_k-x_low": "Eva K.",
}


class VoiceModule:
    """Voice-Dropdown und Test-Button im uebergebenen Frame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: Frame von panel_main (rechte Spalte)
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None
        self._current_voice = "de_DE-thorsten-high"

        self._build_ui()
        self._poll_voice_state()

    def _build_ui(self):
        """Voice-Sektion aufbauen."""
        section = tk.LabelFrame(
            self._parent,
            text="Stimme",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        # Dropdown-Zeile
        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=5)

        # Voice Dropdown
        self._voice_var = tk.StringVar(value="Thorsten (High)")

        # Anzeigenamen fuer Dropdown
        display_names = list(VOICE_NAMES.values())

        self._voice_combo = ttk.Combobox(
            row,
            textvariable=self._voice_var,
            values=display_names,
            state="readonly",
            width=18,
        )
        self._voice_combo.pack(side=tk.LEFT, padx=(0, 5))
        self._voice_combo.bind("<<ComboboxSelected>>", self._on_voice_changed)

        # Test-Button
        self._btn_test = tk.Button(
            row, text="Test", width=5,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=ACCENT_CYAN,
            command=self._test_voice,
        )
        self._btn_test.pack(side=tk.LEFT, padx=(0, 5))

        # Status-Label
        self._lbl_status = tk.Label(
            row, text="",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_status.pack(side=tk.LEFT)

    def _on_voice_changed(self, event=None):
        """Stimme gewechselt — an Service senden."""
        display_name = self._voice_var.get()

        # Display-Name -> voice_id umwandeln
        voice_id = None
        for vid, name in VOICE_NAMES.items():
            if name == display_name:
                voice_id = vid
                break

        if voice_id:
            self._current_voice = voice_id
            self._service._write_command("set_voice", {"voice_id": voice_id})

    def _test_voice(self):
        """Testsatz mit aktueller Stimme sprechen."""
        self._service._write_command("voice_test", {
            "text": "Moloch ist online. Die dunkle Seite macht mehr Spass.",
        })
        self._lbl_status.config(text="Test...", fg=ACCENT_CYAN)

    def _poll_voice_state(self):
        """Voice-Status aus Service-Status synchronisieren."""
        status = self._service.read_status()

        if status:
            voice = status.get("voice", {})

            # Aktuelle Stimme synchronisieren
            svc_voice = voice.get("current_voice", "")
            if svc_voice and svc_voice != self._current_voice:
                self._current_voice = svc_voice
                display = VOICE_NAMES.get(svc_voice, svc_voice)
                self._voice_var.set(display)

            # Sprechstatus anzeigen
            whisper_status = voice.get("whisper_status", "")
            speaking = voice.get("speaking", False)
            if speaking:
                self._lbl_status.config(text="Spricht...", fg=ACCENT_CYAN)
            elif whisper_status and whisper_status != "Idle":
                self._lbl_status.config(text=whisper_status, fg=ACCENT_CYAN)
            else:
                self._lbl_status.config(text="", fg=FG_DIM)

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_voice_state)
