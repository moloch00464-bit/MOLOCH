#!/usr/bin/env python3
"""
M.O.L.O.C.H. Audio Settings Popup
===================================

Eigenstaendiges Toplevel-Fenster fuer Audio-Einstellungen.
Bekommt parent und ServiceProxy uebergeben.

Features:
- Mic Gain Slider (0-100)
- Noise Gate Slider (-60 bis 0 dB)
- AGC Checkbox (Automatic Gain Control)
- VU Meter (Canvas, gruen/gelb/rot, 100ms Update)
- MIC TEST Button (3s Aufnahme + Wiedergabe)

Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_CYAN,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

# VU Meter Update-Intervall in ms
VU_UPDATE_MS = 100

# VU Meter Schwellwerte (0.0 - 1.0)
VU_YELLOW_THRESH = 0.6
VU_RED_THRESH = 0.85


class AudioPopup:
    """Audio Settings als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
            service_proxy: ServiceProxy Instanz fuer Commands/Status
        """
        self.parent = parent
        self.service = service_proxy
        self._after_id = None

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.title("Audio Settings")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("380x470")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Variablen
        self._gain_var = tk.IntVar(value=50)
        self._noise_gate_var = tk.IntVar(value=-30)
        self._agc_var = tk.BooleanVar(value=False)

        # GUI aufbauen
        self._build_title()
        self._build_gain_slider()
        self._build_noise_gate_slider()
        self._build_agc_checkbox()
        self._build_vu_meter()
        self._build_mic_test()
        self._build_save_button()

        # Aktuelle Werte vom Service laden
        self._load_current_values()

        # VU Meter starten
        self._update_vu()

    # =========================================================================
    # Titel
    # =========================================================================

    def _build_title(self):
        """Titel-Label oben."""
        tk.Label(
            self.win, text="Audio Settings",
            bg=BG_DARK, fg=FG_WHITE, font=FONT_TITLE,
        ).pack(pady=(10, 5))

    # =========================================================================
    # Mic Gain Slider
    # =========================================================================

    def _build_gain_slider(self):
        """Horizontaler Slider fuer Mic Gain (0-100)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(
            frame, text="Mic Gain:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(anchor=tk.W)

        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X)

        self._gain_label = tk.Label(
            row, text="50", width=4,
            bg=BG_DARK, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        self._gain_label.pack(side=tk.RIGHT, padx=(5, 0))

        self._gain_slider = tk.Scale(
            row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self._gain_var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=self._on_gain_changed,
        )
        self._gain_slider.pack(fill=tk.X, expand=True)

    def _on_gain_changed(self, val):
        """Gain geaendert - Label updaten und an Service senden."""
        self._gain_label.config(text=str(int(float(val))))
        self._send_audio_settings()

    # =========================================================================
    # Noise Gate Slider
    # =========================================================================

    def _build_noise_gate_slider(self):
        """Horizontaler Slider fuer Noise Gate (-60 bis 0 dB)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        tk.Label(
            frame, text="Noise Gate (dB):", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(anchor=tk.W)

        row = tk.Frame(frame, bg=BG_DARK)
        row.pack(fill=tk.X)

        self._noise_gate_label = tk.Label(
            row, text="-30 dB", width=6,
            bg=BG_DARK, fg=STATUS_YELLOW, font=FONT_LABEL,
        )
        self._noise_gate_label.pack(side=tk.RIGHT, padx=(5, 0))

        self._noise_gate_slider = tk.Scale(
            row, from_=-60, to=0, orient=tk.HORIZONTAL,
            variable=self._noise_gate_var,
            bg=BG_FRAME, fg=FG_WHITE, troughcolor=BG_INPUT,
            highlightthickness=0, font=FONT_SMALL,
            showvalue=False,
            command=self._on_noise_gate_changed,
        )
        self._noise_gate_slider.pack(fill=tk.X, expand=True)

    def _on_noise_gate_changed(self, val):
        """Noise Gate geaendert - Label updaten und senden."""
        self._noise_gate_label.config(text=f"{int(float(val))} dB")
        self._send_audio_settings()

    # =========================================================================
    # AGC Checkbox
    # =========================================================================

    def _build_agc_checkbox(self):
        """Checkbox fuer Automatic Gain Control."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=5)

        self._agc_cb = tk.Checkbutton(
            frame, text="Automatic Gain Control (AGC)",
            variable=self._agc_var,
            bg=BG_DARK, fg=FG_WHITE,
            selectcolor=BG_FRAME,
            activebackground=BG_DARK,
            activeforeground=FG_WHITE,
            font=FONT_LABEL,
            command=self._on_agc_changed,
        )
        self._agc_cb.pack(anchor=tk.W)

    def _on_agc_changed(self):
        """AGC geaendert - an Service senden."""
        self._send_audio_settings()

    # =========================================================================
    # VU Meter
    # =========================================================================

    def _build_vu_meter(self):
        """Canvas-Balken fuer Audio-Pegel (gruen/gelb/rot)."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(10, 5))

        tk.Label(
            frame, text="Audio-Pegel:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(anchor=tk.W)

        self._vu_canvas = tk.Canvas(
            frame, height=24, bg=BG_INPUT,
            highlightthickness=1, highlightbackground=FG_DIM,
        )
        self._vu_canvas.pack(fill=tk.X, pady=(2, 0))

        # Pegel-Label rechts
        self._vu_label = tk.Label(
            frame, text="-- dB", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._vu_label.pack(anchor=tk.E)

    def _update_vu(self):
        """VU Meter alle 100ms aus status['audio']['level'] aktualisieren."""
        level = 0.0
        status = self.service.read_status()
        if status:
            audio = status.get("audio")
            if isinstance(audio, dict):
                raw = audio.get("level")
                if raw is not None and not isinstance(raw, (dict, list)):
                    try:
                        level = max(0.0, min(1.0, float(raw)))
                    except (TypeError, ValueError):
                        level = 0.0

        # Canvas zeichnen
        self._vu_canvas.delete("all")
        canvas_w = self._vu_canvas.winfo_width()
        if canvas_w < 2:
            canvas_w = 340  # Fallback beim ersten Aufruf

        bar_w = int(canvas_w * level)

        # Farbe nach Pegel
        if level >= VU_RED_THRESH:
            color = STATUS_RED
        elif level >= VU_YELLOW_THRESH:
            color = STATUS_YELLOW
        else:
            color = STATUS_GREEN

        if bar_w > 0:
            self._vu_canvas.create_rectangle(
                0, 0, bar_w, 24, fill=color, outline="",
            )

        # dB-Wert anzeigen (log-Skala, -60 dB als Minimum)
        if level > 0.001:
            import math
            db = 20 * math.log10(level)
            self._vu_label.config(text=f"{db:.1f} dB", fg=color)
        else:
            self._vu_label.config(text="-inf dB", fg=FG_DIM)

        # Naechstes Update
        self._after_id = self.win.after(VU_UPDATE_MS, self._update_vu)

    # =========================================================================
    # MIC TEST
    # =========================================================================

    def _build_mic_test(self):
        """MIC TEST Button - startet 3s Aufnahme + Wiedergabe."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(10, 15))

        self._btn_mic_test = tk.Button(
            frame, text="MIC TEST", width=14,
            bg=BG_BUTTON, fg=ACCENT_CYAN, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._on_mic_test,
        )
        self._btn_mic_test.pack()

        self._lbl_mic_status = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_mic_status.pack(pady=(3, 0))

    def _on_mic_test(self):
        """3-Sekunden Mic Test starten."""
        self._btn_mic_test.config(state=tk.DISABLED, text="Aufnahme...")
        self._lbl_mic_status.config(text="Aufnahme laeuft (3s)...", fg=STATUS_YELLOW)
        self.service._write_command("action", {"action": "mic_test"})

        # Button nach 4s wieder freigeben (3s Aufnahme + 1s Puffer)
        self.win.after(4000, self._mic_test_done)

    def _mic_test_done(self):
        """Mic Test abgeschlossen - Button wieder freigeben."""
        self._btn_mic_test.config(state=tk.NORMAL, text="MIC TEST")
        self._lbl_mic_status.config(text="Test abgeschlossen", fg=ACCENT_GREEN)
        self.win.after(3000, lambda: self._lbl_mic_status.config(text="", fg=FG_DIM))

    # =========================================================================
    # SAVE Button
    # =========================================================================

    def _build_save_button(self):
        """SAVE Button - speichert Audio-Einstellungen persistent."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        self._btn_save = tk.Button(
            frame, text="SAVE", width=14,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._on_save,
        )
        self._btn_save.pack(side=tk.LEFT)

        self._lbl_save = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save.pack(side=tk.LEFT, padx=(8, 0))

    def _on_save(self):
        """Audio-Einstellungen persistent speichern."""
        self.service._write_command("action", {
            "action": "save_settings",
            "audio": {
                "mic_gain": self._gain_var.get() / 100.0,
                "noise_gate_db": float(self._noise_gate_var.get()),
                "agc_enabled": self._agc_var.get(),
            },
        })
        self._lbl_save.config(text="Gespeichert!", fg=ACCENT_GREEN)
        self.win.after(2000, lambda: self._lbl_save.config(text="", fg=FG_DIM))

    # =========================================================================
    # Audio Settings senden
    # =========================================================================

    def _send_audio_settings(self):
        """Aktuelle Audio-Einstellungen sofort an den Service senden."""
        self.service._write_command("action", {
            "action": "set_audio",
            "mic_gain": self._gain_var.get() / 100.0,
            "noise_gate_db": float(self._noise_gate_var.get()),
            "agc_enabled": self._agc_var.get(),
        })

    def _load_current_values(self):
        """Aktuelle Werte vom Service lesen und Slider/Checkbox setzen."""
        status = self.service.read_status()
        if not status:
            return

        audio = status.get("audio")
        if not isinstance(audio, dict):
            return

        # Gain (Service: 0.0-1.0 Float → Slider: 0-100 Int)
        raw_gain = audio.get("mic_gain")
        if raw_gain is not None and not isinstance(raw_gain, (dict, list)):
            try:
                gain = int(float(raw_gain) * 100)
                gain = max(0, min(100, gain))
                self._gain_var.set(gain)
                self._gain_label.config(text=str(gain))
            except (TypeError, ValueError):
                pass

        # Noise Gate
        raw_ng = audio.get("noise_gate_db")
        if raw_ng is not None and not isinstance(raw_ng, (dict, list)):
            try:
                ng = int(float(raw_ng))
                ng = max(-60, min(0, ng))
                self._noise_gate_var.set(ng)
                self._noise_gate_label.config(text=f"{ng} dB")
            except (TypeError, ValueError):
                pass

        # AGC
        raw_agc = audio.get("agc_enabled")
        if raw_agc is not None:
            try:
                self._agc_var.set(bool(raw_agc))
            except (TypeError, ValueError):
                pass

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Fenster sauber schliessen - Timer stoppen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
