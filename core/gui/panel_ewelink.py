#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel eWeLink
============================

eWeLink Cloud Controls fuer Sonoff CAM-PT2.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- FLUTLICHT: Toggle weisse LEDs (Panel: 0=aus/IR, 2=an/Farb-Nacht)
- ERKANNT: Status-Indikator blaue LED (sledOnline, vom Service gesteuert)
- ALARM (rot toggle), SNAP (cyan einmal)
- EINPRÄGEN Button: Batch-Analyse aller Snapshots (Face + Pose Enrollment)

Alle Commands via ServiceProxy._write_command().
Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON,
    BTN_ALARM_RED, BTN_OFF_DARK, BTN_SNAP_CYAN,
    ACCENT_CYAN,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL,
)


class EwelinkModule:
    """eWeLink Cloud Controls im uebergebenen LabelFrame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy

        # Zustaende (vom Cloud-Sync aktualisiert)
        self._flutlicht_on = False  # nightVision: 0=aus, 2=an
        self._alarm_active = False
        self._erkannt_led_on = False  # blaue Status-LED (sledOnline)

        # Button-Referenzen
        self._btn_alarm = None
        self._btn_flutlicht = None
        self._btn_erkannt = None

        # GUI aufbauen
        self._build_action_buttons()

    # =========================================================================
    # Buttons: ALARM, SNAP, FLUTLICHT, ERKANNT, EINPRÄGEN
    # =========================================================================

    def _build_action_buttons(self):
        """ALARM, SNAP, FLUTLICHT, ERKANNT, EINPRÄGEN, GALERIE Buttons."""
        section = tk.LabelFrame(
            self._parent,
            text="Aktionen",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        # ALARM (rot toggle)
        self._btn_alarm = tk.Button(
            row, text="ALARM", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_alarm,
        )
        self._btn_alarm.grid(row=0, column=0, padx=3, pady=2)

        # SNAP (cyan, einmal-klick)
        tk.Button(
            row, text="SNAP", width=8,
            bg=BTN_SNAP_CYAN, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._take_snapshot,
        ).grid(row=0, column=1, padx=3, pady=2)

        # FLUTLICHT (weisse LEDs toggle, nightVision 0=aus / 2=an)
        self._btn_flutlicht = tk.Button(
            row, text="FLUTLICHT", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_flutlicht,
        )
        self._btn_flutlicht.grid(row=0, column=2, padx=3, pady=2)

        # ERKANNT (Status-Indikator, wird vom Service gesteuert)
        self._btn_erkannt = tk.Button(
            row, text="ERKANNT", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            state="disabled",
            disabledforeground=FG_WHITE,
        )
        self._btn_erkannt.grid(row=0, column=3, padx=3, pady=2)

        # EINPRÄGEN Button
        self._btn_einpraegen = tk.Button(
            row, text="EINPRÄGEN", width=10,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._start_einpraegen,
        )
        self._btn_einpraegen.grid(row=0, column=4, padx=3, pady=2)

        # GALERIE Button
        tk.Button(
            row, text="GALERIE", width=8,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._open_gallery,
        ).grid(row=0, column=5, padx=3, pady=2)

        # Status-Labels
        self._lbl_alarm = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_alarm.grid(row=1, column=0, pady=(0, 5))

        self._lbl_snap = tk.Label(
            row, text="bereit", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_snap.grid(row=1, column=1, pady=(0, 5))

        self._lbl_flutlicht = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_flutlicht.grid(row=1, column=2, pady=(0, 5))

        self._lbl_erkannt = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_erkannt.grid(row=1, column=3, pady=(0, 5))

        self._lbl_einpraegen = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_einpraegen.grid(row=1, column=4, pady=(0, 5))

        # Einpraegen Poll-State
        self._einpraegen_polling = False

    def _toggle_alarm(self):
        """Alarm an/aus."""
        self._alarm_active = not self._alarm_active
        self._update_alarm_button()
        self._service._write_command("cloud_alarm", {"on": self._alarm_active})

    def _update_alarm_button(self):
        """Alarm-Button Farbe aktualisieren."""
        if self._alarm_active:
            self._btn_alarm.config(bg=BTN_ALARM_RED)
            self._lbl_alarm.config(text="AN", fg=BTN_ALARM_RED)
        else:
            self._btn_alarm.config(bg=BTN_OFF_DARK)
            self._lbl_alarm.config(text="AUS", fg=FG_DIM)

    def _take_snapshot(self):
        """Snapshot ausloesen."""
        self._service._write_command("snapshot")
        self._lbl_snap.config(text="gespeichert", fg=BTN_SNAP_CYAN)
        # Nach 2 Sekunden zuruecksetzen
        self._parent.after(2000, lambda: self._lbl_snap.config(
            text="bereit", fg=FG_DIM
        ))

    def _toggle_flutlicht(self):
        """Weisse LEDs an/aus (nightVision 0=day/aus, 2=night/an)."""
        self._flutlicht_on = not self._flutlicht_on
        level = 2 if self._flutlicht_on else 0
        self._update_flutlicht_button()
        self._service._write_command("cloud_led", {"level": level})

    def _update_flutlicht_button(self):
        """FLUTLICHT Button Farbe aktualisieren."""
        if self._flutlicht_on:
            self._btn_flutlicht.config(bg=ACCENT_CYAN)
            self._lbl_flutlicht.config(text="AN", fg=ACCENT_CYAN)
        else:
            self._btn_flutlicht.config(bg=BTN_OFF_DARK)
            self._lbl_flutlicht.config(text="AUS", fg=FG_DIM)

    def _update_erkannt_button(self):
        """Status-Indikator: zeigt LED-State und wer erkannt wurde."""
        if self._erkannt_led_on:
            self._btn_erkannt.config(bg=ACCENT_CYAN, disabledforeground="#000000")
            self._lbl_erkannt.config(text="MARKUS", fg=ACCENT_CYAN)
        else:
            self._btn_erkannt.config(bg=BTN_OFF_DARK, disabledforeground=FG_WHITE)
            self._lbl_erkannt.config(text="---", fg=FG_DIM)

    def _start_einpraegen(self):
        """EINPRÄGEN starten: Batch-Analyse aller Snapshots via Service."""
        self._btn_einpraegen.config(state="disabled")
        self._lbl_einpraegen.config(text="starte...", fg=ACCENT_CYAN)
        self._service._write_command("einpraegen")
        # Fortschritt pollen
        if not self._einpraegen_polling:
            self._einpraegen_polling = True
            self._poll_einpraegen()

    def _poll_einpraegen(self):
        """Fortschritt des Einpraegen-Prozesses pollen."""
        status = self._service.read_status()
        if status:
            running = status.get("einpraegen_running", False)
            progress = status.get("einpraegen_progress", "")

            if running and progress:
                self._btn_einpraegen.config(text=f"EINPRÄGEN ({progress})")
                self._lbl_einpraegen.config(text="laeuft...", fg=ACCENT_CYAN)
                self._parent.after(500, self._poll_einpraegen)
            elif not running and status.get("einpraegen_done", False):
                # Fertig
                self._btn_einpraegen.config(text="EINPRÄGEN \u2713", state="normal")
                self._lbl_einpraegen.config(text="fertig", fg=ACCENT_CYAN)
                self._einpraegen_polling = False
                # Nach 5 Sekunden Button-Text zuruecksetzen
                self._parent.after(5000, self._reset_einpraegen_button)
            else:
                # Noch kein Status — weiter pollen
                self._parent.after(500, self._poll_einpraegen)
        else:
            # Kein Status — weiter pollen (Service evtl. noch nicht bereit)
            self._parent.after(1000, self._poll_einpraegen)

    def _reset_einpraegen_button(self):
        """Button-Text zurueck auf Standard."""
        self._btn_einpraegen.config(text="EINPRÄGEN", state="normal")
        self._lbl_einpraegen.config(text="", fg=FG_DIM)

    def update_from_status(self, status):
        """Vom panel_main Poll aufgerufen: ERKANNT-Indikator aktualisieren."""
        led_on = status.get("led_markus_on", False)
        if led_on != self._erkannt_led_on:
            self._erkannt_led_on = led_on
            self._update_erkannt_button()

    def _open_gallery(self):
        """Snapshot Galerie Popup oeffnen."""
        from core.gui.popups.popup_gallery import SnapshotGallery
        SnapshotGallery(self._parent)
