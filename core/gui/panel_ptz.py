#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel PTZ
=======================

PTZ Steuerung und Hauptbuttons.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- D-Pad: 5 Buttons in Kreuzform (Hoch/Runter/Links/Rechts/Home)
- Pan/Tilt Live-Anzeige + Moves-Zaehler
- Quick Positions: Werkstatt, Wohnzimmer
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
    ACCENT_CYAN,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
)


class PtzModule:
    """PTZ Steuerung und Toggle-Buttons im uebergebenen LabelFrame."""

    # Quick-Position Definitionen
    POSITIONS = [
        ("Werkstatt", "werkstatt"),
        ("Wohnzimmer", "wohnzimmer"),
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
        self._active_position = None

        # Restless-Score Cache
        self._restless_score = 0.0

        # Manuelle Bewegungszaehler
        self._move_count = 0

        # GUI aufbauen
        self._build_dpad()
        self._build_restless_indicator()
        self._build_quick_positions()
        self._build_toggles()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # D-Pad + Pan/Tilt Anzeige
    # =========================================================================

    def _build_dpad(self):
        """D-Pad in Kreuzform + Pan/Tilt Live-Anzeige + Moves."""
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

        # Zeile 3: Pan/Tilt Live-Anzeige
        self._lbl_pan_tilt = tk.Label(
            grid, text="Pan: ---  Tilt: ---",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_pan_tilt.grid(row=3, column=0, columnspan=3, pady=(4, 0))

        # Zeile 4: Moves-Zaehler
        self._lbl_moves = tk.Label(
            grid, text="Moves: 0  Trk: 0  Srch: 0",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_moves.grid(row=4, column=0, columnspan=3, pady=(0, 5))

    def _ptz_move(self, direction):
        """PTZ-Bewegung senden."""
        self._service._write_command("ptz_move", {"direction": direction})
        self._move_count += 1

    # =========================================================================
    # Restless-Indikator
    # =========================================================================

    def _build_restless_indicator(self):
        """Kleiner Indikator: Kamera-Bewegungsintensitaet + NPU Stage.

        Ruhig (gruen) | Moderat (gelb) | Hektisch (rot)
        NPU: IDLE (grau) | PERSON (gelb) | FACE (gruen)
        """
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=(0, 1))

        tk.Label(
            row, text="PTZ:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(5, 2))

        self._lbl_restless = tk.Label(
            row, text="Ruhig", bg=BG_FRAME, fg=STATUS_GREEN, font=FONT_SMALL,
        )
        self._lbl_restless.pack(side=tk.LEFT, padx=2)

        self._lbl_tracker_state = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_tracker_state.pack(side=tk.RIGHT, padx=5)

        # NPU Stage Zeile
        npu_row = tk.Frame(self._parent, bg=BG_FRAME)
        npu_row.pack(fill=tk.X, padx=5, pady=(0, 2))

        tk.Label(
            npu_row, text="NPU:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(5, 2))

        self._lbl_npu_stage = tk.Label(
            npu_row, text="---", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_npu_stage.pack(side=tk.LEFT, padx=2)

        self._lbl_npu_models = tk.Label(
            npu_row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_npu_models.pack(side=tk.RIGHT, padx=5)

    def _update_restless_indicator(self, status):
        """Restless-Indikator aus Status-Dict aktualisieren."""
        ptz = status.get("ptz", {})
        score = ptz.get("ptz_restless_score", 0.0)
        stage = ptz.get("ptz_stage", "idle")
        tracker_state = ptz.get("tracker_state", "idle")

        # Farbe nach Score
        if score < 0.2:
            text = "Ruhig"
            color = STATUS_GREEN
        elif score < 0.6:
            text = f"Aktiv ({score:.1f})"
            color = STATUS_YELLOW
        else:
            text = f"Hektisch ({score:.1f})"
            color = STATUS_RED

        if hasattr(self, '_lbl_restless'):
            self._lbl_restless.config(text=text, fg=color)

        # Tracker-State rechts anzeigen
        state_map = {
            "idle": "", "tracking": "Trackt", "searching": "Sucht...",
            "locked": "Locked", "frozen": "Frozen", "coast": "Coast",
            "dwell": "Dwell",
        }
        state_text = state_map.get(tracker_state, tracker_state)
        if hasattr(self, '_lbl_tracker_state'):
            self._lbl_tracker_state.config(text=state_text)

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

        self._pos_buttons = []
        for i, (label, name) in enumerate(self.POSITIONS):
            btn = tk.Button(
                row, text=label, width=14,
                bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=lambda n=name: self._ptz_goto(n),
            )
            btn.grid(row=0, column=i, padx=2, pady=2)
            self._pos_buttons.append((btn, name))

    def _ptz_goto(self, position):
        """Zu gespeicherter Position fahren mit Farb-Feedback."""
        self._service._write_command("ptz_goto", {"position": position})
        self._active_position = position
        for btn_widget, pos_name in self._pos_buttons:
            if pos_name == position:
                btn_widget.config(bg=ACCENT_CYAN)
            else:
                btn_widget.config(bg=BG_BUTTON)

    # =========================================================================
    # Toggle-Buttons + Status-Labels
    # =========================================================================

    def _build_toggles(self):
        """Toggle-Buttons: AUTONOM, ST Toggle, TEACHEN mit Status-Labels."""
        section = tk.LabelFrame(
            self._parent,
            text="Modi",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        # Zeile 1: Smart Tracking Status + Toggle
        st_row = tk.Frame(section, bg=BG_FRAME)
        st_row.pack(fill=tk.X, padx=5, pady=(5, 2))

        self._btn_st = tk.Button(
            st_row, text="ST: ---", width=10,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_smart_tracking,
        )
        self._btn_st.pack(side=tk.LEFT, padx=3)

        self._lbl_arbiter = tk.Label(
            st_row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_arbiter.pack(side=tk.LEFT, padx=5)

        # Zeile 2: AUTONOM + TEACHEN
        grid = tk.Frame(section, bg=BG_FRAME)
        grid.pack(pady=(2, 5), padx=5)

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

    def _toggle_smart_tracking(self):
        """Smart Tracking an/aus."""
        self._service.toggle_smart_tracking()

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

            # Pan/Tilt Anzeige aktualisieren
            ptz = status.get("ptz", {})
            cur_pan = ptz.get("current_pan")
            cur_tilt = ptz.get("current_tilt")
            if cur_pan is not None and cur_tilt is not None:
                self._lbl_pan_tilt.config(
                    text=f"Pan: {cur_pan:.1f}\u00b0  Tilt: {cur_tilt:.1f}\u00b0",
                    fg=FG_LABEL,
                )
            else:
                self._lbl_pan_tilt.config(text="Pan: ---  Tilt: ---", fg=FG_DIM)

            # Moves-Zaehler (manuell + tracking + search)
            trk = ptz.get("tracking_moves", 0)
            srch = ptz.get("search_moves", 0)
            self._lbl_moves.config(
                text=f"Moves: {self._move_count}  Trk: {trk}  Srch: {srch}",
            )

            # Smart Tracking Status + Arbiter
            st_on = status.get("cam_smart_tracking", False)
            arbiter_mode = status.get("ptz_arbiter_mode", "")
            self._btn_st.config(
                text=f"ST: {'AN' if st_on else 'AUS'}",
                bg=BTN_ON_GREEN if st_on else BTN_OFF_DARK,
            )
            # Arbiter-Modus als Kurztext
            arbiter_map = {
                "kamera_fuehrt": "Kamera fuehrt",
                "moloch_korrigiert": "MOLOCH korrigiert",
                "moloch_uebernimmt": "MOLOCH steuert",
            }
            arbiter_text = arbiter_map.get(arbiter_mode, arbiter_mode)
            arbiter_color = STATUS_GREEN if arbiter_mode == "kamera_fuehrt" else (
                STATUS_YELLOW if arbiter_mode == "moloch_korrigiert" else STATUS_RED
            )
            self._lbl_arbiter.config(text=arbiter_text, fg=arbiter_color)

            # NPU-Stage Anzeige
            npu_stage = status.get("npu_stage", "")
            active_models = status.get("active_models", [])
            npu_paused = status.get("npu_paused", False)
            if npu_paused:
                npu_text = "Voice"
                npu_color = STATUS_YELLOW
            elif npu_stage == "face":
                npu_text = "FACE"
                npu_color = STATUS_GREEN
            elif npu_stage == "person":
                npu_text = "PERSON"
                npu_color = STATUS_YELLOW
            elif npu_stage == "idle":
                npu_text = "IDLE"
                npu_color = FG_DIM
            else:
                npu_text = npu_stage.upper() if npu_stage else "---"
                npu_color = FG_DIM
            self._lbl_npu_stage.config(text=npu_text, fg=npu_color)
            models_text = ", ".join(active_models) if active_models else "---"
            self._lbl_npu_models.config(text=models_text)

            # Restless-Indikator aktualisieren
            self._update_restless_indicator(status)

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
