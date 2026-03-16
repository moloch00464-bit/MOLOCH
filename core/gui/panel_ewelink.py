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
    BTN_ALARM_RED, BTN_OFF_DARK, BTN_SNAP_CYAN, BTN_ON_GREEN,
    ACCENT_CYAN, ACCENT_GREEN,
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
        self._erkannt_mode = "guardian"  # Gate0 Phase 6: personality_mode

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

        # SNAP (cyan, einmal-klick) → speichert in media/snapshots/
        tk.Button(
            row, text="SNAP", width=8,
            bg=BTN_SNAP_CYAN, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._take_snapshot,
        ).grid(row=0, column=1, padx=3, pady=2)

        # TEACH (Toggle: Lernmodus AN/AUS im Service)
        self._btn_teach = tk.Button(
            row, text="TEACH", width=6,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_teach_mode,
        )
        self._btn_teach.grid(row=0, column=2, padx=2, pady=2)

        # FOTO (manueller Teach-Trigger, sofort, kein Cooldown)
        self._btn_foto = tk.Button(
            row, text="FOTO", width=5,
            bg="#9933cc", fg=FG_WHITE, font=FONT_BUTTON,
            activebackground="#7722aa",
            command=self._trigger_teach_foto,
        )
        self._btn_foto.grid(row=0, column=3, padx=2, pady=2)

        # FLUTLICHT (weisse LEDs toggle, nightVision 0=aus / 2=an)
        self._btn_flutlicht = tk.Button(
            row, text="FLUTLICHT", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._toggle_flutlicht,
        )
        self._btn_flutlicht.grid(row=0, column=4, padx=3, pady=2)

        # ERKANNT (Status-Indikator, wird vom Service gesteuert)
        self._btn_erkannt = tk.Button(
            row, text="ERKANNT", width=8,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            state="disabled",
            disabledforeground=FG_WHITE,
        )
        self._btn_erkannt.grid(row=0, column=5, padx=3, pady=2)

        # EINPRÄGEN Button
        self._btn_einpraegen = tk.Button(
            row, text="EINPRÄGEN", width=10,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._start_einpraegen,
        )
        self._btn_einpraegen.grid(row=0, column=6, padx=3, pady=2)

        # GALERIE Button
        tk.Button(
            row, text="GALERIE", width=8,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._open_gallery,
        ).grid(row=0, column=7, padx=3, pady=2)

        # Status-Labels (Zeile 1 unter den Buttons)
        self._lbl_alarm = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_alarm.grid(row=1, column=0, pady=(0, 5))

        self._lbl_snap = tk.Label(
            row, text="bereit", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_snap.grid(row=1, column=1, pady=(0, 5))

        self._lbl_teach = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_teach.grid(row=1, column=2, columnspan=2, pady=(0, 5))

        self._lbl_flutlicht = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_flutlicht.grid(row=1, column=4, pady=(0, 5))

        self._lbl_erkannt = tk.Label(
            row, text="AUS", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_erkannt.grid(row=1, column=5, pady=(0, 5))

        self._lbl_einpraegen = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_einpraegen.grid(row=1, column=6, pady=(0, 5))

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
        """Snapshot ausloesen → media/snapshots/."""
        self._service._write_command("snapshot")
        self._lbl_snap.config(text="gespeichert", fg=BTN_SNAP_CYAN)
        self._parent.after(2000, lambda: self._lbl_snap.config(
            text="bereit", fg=FG_DIM
        ))

    def _toggle_teach_mode(self):
        """Teach-Modus AN/AUS im Service umschalten."""
        self._service._write_command("teach_mode_toggle")

    def _trigger_teach_foto(self):
        """Manueller Teach-Trigger — sofort, ohne Cooldown."""
        self._service._write_command("teach_trigger")
        self._lbl_teach.config(text="\u23f3 Foto...", fg="#cc88ff")

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
        """Status-Indikator: zeigt LED-State UND personality_mode.

        Gate0 Phase 6: LED zeigt Wahrheit — stimmt mit Iris ueberein.
        Guardian + Markus = BLAU/CYAN
        Shadow = ROT
        Berserker = DUNKELROT
        """
        if self._erkannt_mode in ("shadow", "berserker"):
            # Shadow/Berserker: ROT — unabhaengig von Erkennung
            clr = BTN_ALARM_RED if self._erkannt_mode == "shadow" else "#880000"
            lbl = "SHADOW" if self._erkannt_mode == "shadow" else "BERSERKER"
            self._btn_erkannt.config(bg=clr, disabledforeground="#000000")
            self._lbl_erkannt.config(text=lbl, fg=clr)
        elif self._erkannt_led_on:
            # Guardian + Markus erkannt = BLAU
            self._btn_erkannt.config(bg=ACCENT_CYAN, disabledforeground="#000000")
            self._lbl_erkannt.config(text="MARKUS", fg=ACCENT_CYAN)
        else:
            # Guardian + niemand erkannt = dunkel
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
        """Vom panel_main Poll aufgerufen: Status-Indikatoren aktualisieren.

        - ERKANNT: LED-State + personality_mode + Face-ID
        - TEACH Button: Modus (gruen/grau) + Prozess-Status (pulsierend/gruen/rot)
        """
        led_on = status.get("led_markus_on", False)
        led_mode = status.get("led_personality_mode", "guardian")

        # Face-ID Name + ArcFace Similarity aus Status lesen
        face_id = status.get("face_id", "")
        face_sim = status.get("face_similarity", 0.0)
        face_detected = status.get("face_detected", False)

        if led_on != self._erkannt_led_on or led_mode != self._erkannt_mode:
            self._erkannt_led_on = led_on
            self._erkannt_mode = led_mode
            self._update_erkannt_button()

        # Label unter ERKANNT: Face-ID Name + ArcFace Similarity
        if face_id and face_detected:
            self._lbl_erkannt.config(
                text=f"{face_id} ({face_sim:.0%})",
                fg=ACCENT_CYAN if face_sim > 0.5 else "#ffcc00",
            )
        elif face_detected:
            self._lbl_erkannt.config(text="unbekannt", fg="#ffcc00")
        elif not self._erkannt_led_on:
            self._lbl_erkannt.config(text="---", fg=FG_DIM)

        # --- TEACH Modus Button (gruen=AN, grau=AUS) ---
        teach_on = status.get("teach_mode_enabled", False)
        if teach_on:
            self._btn_teach.config(bg=BTN_ON_GREEN, text="TEACH AN")
        else:
            self._btn_teach.config(bg=BTN_OFF_DARK, text="TEACH")

        # --- TEACH Prozess-Status (Label = Spiegel vom Service) ---
        teach = status.get("teach_result", {})
        teach_st = teach.get("status", "")

        if teach_st in ("running", "starting"):
            attempt = teach.get("attempt", 0)
            if attempt > 0:
                self._lbl_teach.config(
                    text=f"\u23f3 Versuch {attempt}/3...", fg="#ffaa00"
                )
            else:
                self._lbl_teach.config(text="\u23f3 Verarbeite...", fg="#ffaa00")
        elif teach_st == "retry":
            reason = teach.get("reason", "")
            self._lbl_teach.config(text=reason, fg="#ff6666")
        elif teach_st == "success":
            sim = teach.get("similarity", 0)
            sim_pct = int(sim * 100)
            self._lbl_teach.config(
                text=f"\u2713 Sim: {sim_pct}%", fg=ACCENT_GREEN
            )
        elif teach_st == "failed":
            reason = teach.get("reason", "Fehlgeschlagen")
            self._lbl_teach.config(text=f"\u2717 {reason}", fg="#ff4444")
        elif teach_on:
            self._lbl_teach.config(text="AN", fg=ACCENT_GREEN)
        else:
            self._lbl_teach.config(text="AUS", fg=FG_DIM)

    def _open_gallery(self):
        """Snapshot Galerie Popup oeffnen."""
        from core.gui.popups.popup_gallery import SnapshotGallery
        SnapshotGallery(self._parent)
