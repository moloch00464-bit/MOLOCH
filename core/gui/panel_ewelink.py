#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel eWeLink
============================

eWeLink Cloud Controls fuer Sonoff CAM-PT2.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- LED Section: AUS/LOW/MED/HIGH (4 Buttons, aktiver Level cyan)
- Einzel-Buttons: ALARM (rot toggle), SNAP (cyan einmal), White LED (toggle)
- SYNC Button: Holt Cloud-Status, aktualisiert Button-Farben

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

    # LED Level Namen
    LED_LEVELS = [
        ("AUS", 0),
        ("LOW", 1),
        ("MED", 2),
        ("HIGH", 3),
    ]

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy

        # Zustaende (vom Cloud-Sync aktualisiert)
        self._led_level = 0
        self._alarm_active = False
        self._white_led_on = False

        # Button-Referenzen
        self._led_btns = []
        self._btn_alarm = None
        self._btn_white_led = None

        # GUI aufbauen
        self._build_led_section()
        self._build_action_buttons()

    # =========================================================================
    # LED Section
    # =========================================================================

    def _build_led_section(self):
        """4 LED Buttons in Reihe: AUS/LOW/MED/HIGH."""
        section = tk.LabelFrame(
            self._parent,
            text="LED",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        self._led_btns = []
        for i, (label, level) in enumerate(self.LED_LEVELS):
            btn = tk.Button(
                row, text=label, width=7,
                bg=ACCENT_CYAN if level == self._led_level else BTN_OFF_DARK,
                fg=FG_WHITE, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=lambda lv=level: self._set_led(lv),
            )
            btn.grid(row=0, column=i, padx=2, pady=2)
            self._led_btns.append(btn)

    def _set_led(self, level):
        """LED Level setzen und Buttons aktualisieren."""
        self._led_level = level
        self._update_led_buttons()
        self._service._write_command("cloud_led", {"level": level})

    def _update_led_buttons(self):
        """LED Buttons farblich aktualisieren."""
        for i, (_, level) in enumerate(self.LED_LEVELS):
            bg = ACCENT_CYAN if level == self._led_level else BTN_OFF_DARK
            self._led_btns[i].config(bg=bg)

    # =========================================================================
    # Einzel-Buttons: ALARM, SNAP, White LED, SYNC
    # =========================================================================

    def _build_action_buttons(self):
        """ALARM, SNAP, White LED und SYNC Buttons."""
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

        # White LED (toggle)
        self._btn_white_led = tk.Button(
            row, text="W-LED", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_white_led,
        )
        self._btn_white_led.grid(row=0, column=2, padx=3, pady=2)

        # SYNC Button
        tk.Button(
            row, text="SYNC", width=8,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._sync_cloud_status,
        ).grid(row=0, column=3, padx=3, pady=2)

        # GALERIE Button
        tk.Button(
            row, text="GALERIE", width=8,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._open_gallery,
        ).grid(row=0, column=4, padx=3, pady=2)

        # Status-Labels
        self._lbl_alarm = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_alarm.grid(row=1, column=0, pady=(0, 5))

        self._lbl_snap = tk.Label(
            row, text="bereit", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_snap.grid(row=1, column=1, pady=(0, 5))

        self._lbl_white_led = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_white_led.grid(row=1, column=2, pady=(0, 5))

        self._lbl_sync = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_sync.grid(row=1, column=3, pady=(0, 5))

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

    def _toggle_white_led(self):
        """Weisse Status-LED an/aus."""
        self._white_led_on = not self._white_led_on
        self._update_white_led_button()
        self._service._write_command("cloud_status_led", {"on": self._white_led_on})

    def _update_white_led_button(self):
        """White LED Button Farbe aktualisieren."""
        if self._white_led_on:
            self._btn_white_led.config(bg=ACCENT_CYAN)
            self._lbl_white_led.config(text="AN", fg=ACCENT_CYAN)
        else:
            self._btn_white_led.config(bg=BTN_OFF_DARK)
            self._lbl_white_led.config(text="AUS", fg=FG_DIM)

    def _sync_cloud_status(self):
        """Cloud-Status holen und alle Buttons aktualisieren."""
        self._lbl_sync.config(text="sync...", fg=ACCENT_CYAN)
        self._service._write_command("cloud_sync")

        # Status lesen und GUI aktualisieren
        status = self._service.read_status()
        if status:
            cloud = status.get("cloud", {})
            if cloud:
                # LED Level
                led = cloud.get("led_level")
                if led is not None:
                    self._led_level = led
                    self._update_led_buttons()

                # Alarm
                alarm = cloud.get("alarm_active")
                if alarm is not None:
                    self._alarm_active = alarm
                    self._update_alarm_button()

                # White LED
                wled = cloud.get("status_led")
                if wled is not None:
                    self._white_led_on = wled
                    self._update_white_led_button()

                self._lbl_sync.config(text="OK", fg=ACCENT_CYAN)
            else:
                self._lbl_sync.config(text="leer", fg=FG_DIM)
        else:
            self._lbl_sync.config(text="fehler", fg=BTN_ALARM_RED)

        # Label nach 3 Sekunden zuruecksetzen
        self._parent.after(3000, lambda: self._lbl_sync.config(
            text="", fg=FG_DIM
        ))

    def _open_gallery(self):
        """Snapshot Galerie Popup oeffnen."""
        from core.gui.popups.popup_gallery import SnapshotGallery
        SnapshotGallery(self._parent)
