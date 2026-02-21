#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel PTZ
=======================

PTZ Steuerung und Hauptbuttons.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- D-Pad: 5 Buttons in Kreuzform (Hoch/Runter/Links/Rechts/Home)
- Quick Positions: Schreibtisch, Tuer, Fenster, Bett
- Toggle-Buttons: AUTONOM, TEACHEN
- Status-Labels mit 500ms Update via ServiceProxy

Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON,
    BTN_ON_GREEN, BTN_OFF_DARK, BTN_OFF_RED,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL,
    STATUS_UPDATE_MS,
)


class PtzModule:
    """PTZ Steuerung und Toggle-Buttons im uebergebenen LabelFrame."""

    # Quick-Position Definitionen
    POSITIONS = [
        ("Schreibtisch", "schreibtisch"),
        ("Tuer", "tuer"),
        ("Fenster", "fenster"),
        ("Bett", "bett"),
    ]

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main (frame_steuerung)
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None

        # Toggle-Zustaende (vom Service-Status aktualisiert)
        self._autonomous = False
        self._daily_learner = False

        # GUI aufbauen
        self._build_dpad()
        self._build_quick_positions()
        self._build_toggles()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # D-Pad
    # =========================================================================

    def _build_dpad(self):
        """D-Pad in Kreuzform: Hoch/Runter/Links/Rechts/Home."""
        section = tk.LabelFrame(
            self._parent,
            text="PTZ",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        grid = tk.Frame(section, bg=BG_FRAME)
        grid.pack(pady=5)

        bw, bh = 4, 2  # Button Breite/Hoehe in Zeichen

        # Zeile 0: Hoch
        tk.Button(
            grid, text="\u25B2", width=bw, height=bh,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=lambda: self._ptz_move("up"),
        ).grid(row=0, column=1, padx=2, pady=2)

        # Zeile 1: Links / Home / Rechts
        tk.Button(
            grid, text="\u25C4", width=bw, height=bh,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=lambda: self._ptz_move("left"),
        ).grid(row=1, column=0, padx=2, pady=2)

        tk.Button(
            grid, text="\u2302", width=bw, height=bh,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=lambda: self._ptz_move("home"),
        ).grid(row=1, column=1, padx=2, pady=2)

        tk.Button(
            grid, text="\u25BA", width=bw, height=bh,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=lambda: self._ptz_move("right"),
        ).grid(row=1, column=2, padx=2, pady=2)

        # Zeile 2: Runter
        tk.Button(
            grid, text="\u25BC", width=bw, height=bh,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=lambda: self._ptz_move("down"),
        ).grid(row=2, column=1, padx=2, pady=2)

    def _ptz_move(self, direction):
        """PTZ-Bewegung senden."""
        self._service._write_command("ptz_move", {"direction": direction})

    # =========================================================================
    # Quick Positions
    # =========================================================================

    def _build_quick_positions(self):
        """Buttons fuer gespeicherte Kamera-Positionen."""
        section = tk.LabelFrame(
            self._parent,
            text="Positionen",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=2)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        for i, (label, name) in enumerate(self.POSITIONS):
            tk.Button(
                row, text=label, width=10,
                bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=lambda n=name: self._ptz_goto(n),
            ).grid(row=0, column=i, padx=2, pady=2)

    def _ptz_goto(self, position):
        """Zu gespeicherter Position fahren."""
        self._service._write_command("ptz_goto", {"position": position})

    # =========================================================================
    # Toggle-Buttons + Status-Labels
    # =========================================================================

    def _build_toggles(self):
        """Toggle-Buttons: AUTONOM und TEACHEN mit Status-Labels."""
        section = tk.LabelFrame(
            self._parent,
            text="Modi",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        grid = tk.Frame(section, bg=BG_FRAME)
        grid.pack(pady=5, padx=5)

        # AUTONOM
        self._btn_autonom = tk.Button(
            grid, text="AUTONOM", width=12,
            bg=BTN_OFF_RED, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_autonomous,
        )
        self._btn_autonom.grid(row=0, column=0, padx=3, pady=2)

        # TEACHEN (Daily Learner)
        self._btn_alltag = tk.Button(
            grid, text="TEACHEN", width=10,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_daily_learner,
        )
        self._btn_alltag.grid(row=0, column=1, padx=3, pady=2)

        # Status-Labels unter den Buttons
        self._lbl_autonom = tk.Label(
            grid, text="Manuell", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_autonom.grid(row=1, column=0, pady=(0, 5))

        self._lbl_alltag = tk.Label(
            grid, text="Bereit", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_alltag.grid(row=1, column=1, pady=(0, 5))

    def _toggle_autonomous(self):
        """Autonomen Modus umschalten."""
        self._service.toggle_autonomous()

    def _toggle_daily_learner(self):
        """Daily Learner umschalten."""
        self._service.toggle_daily_learner()

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Status vom Service lesen und Button-Farben/Labels aktualisieren."""
        status = self._service.read_status()

        if status:
            # Autonomer Modus
            auto = not status.get("manual_mode", True)
            if auto != self._autonomous:
                self._autonomous = auto
                self._btn_autonom.config(
                    bg=BTN_ON_GREEN if auto else BTN_OFF_RED
                )
                self._lbl_autonom.config(
                    text="Autonom" if auto else "Manuell",
                    fg=BTN_ON_GREEN if auto else FG_DIM,
                )

            # Daily Learner
            dl = status.get("daily_learner_enabled", False)
            if dl != self._daily_learner:
                self._daily_learner = dl
                self._btn_alltag.config(
                    bg=BTN_ON_GREEN if dl else BTN_OFF_DARK
                )
                self._lbl_alltag.config(
                    text="Lernt..." if dl else "Bereit",
                    fg=BTN_ON_GREEN if dl else FG_DIM,
                )

        # Widgets sofort neu zeichnen
        self._parent.update_idletasks()

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)

    def update(self):
        """Manuelles Status-Update — liest Service-Status und aktualisiert GUI."""
        status = self._service.read_status()
        if not status:
            return

        self._autonomous = not status.get("manual_mode", True)
        self._daily_learner = status.get("daily_learner_enabled", False)

        self._btn_autonom.config(
            bg=BTN_ON_GREEN if self._autonomous else BTN_OFF_RED
        )
        self._lbl_autonom.config(
            text="Autonom" if self._autonomous else "Manuell",
            fg=BTN_ON_GREEN if self._autonomous else FG_DIM,
        )

        self._btn_alltag.config(
            bg=BTN_ON_GREEN if self._daily_learner else BTN_OFF_DARK
        )
        self._lbl_alltag.config(
            text="Lernt..." if self._daily_learner else "Bereit",
            fg=BTN_ON_GREEN if self._daily_learner else FG_DIM,
        )
