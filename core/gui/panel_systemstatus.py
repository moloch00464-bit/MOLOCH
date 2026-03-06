#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Systemstatus
=================================

Live-Anzeige aller Gate 1-5 Systemdaten im Hauptpanel.
Liest aus moloch_status.json via ServiceProxy.

Sektionen:
  1. Bridge + Erkennung: Bridge State, Face-ID Name+Score, Owner
  2. Innenleben: Tension-Bar, Zone, Personality, Tageszeit
  3. Trends: Approaching/Leaving, Distance, Presence Duration

Alle Daten kommen aus /dev/shm/moloch_status.json.
Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_RED, STATUS_YELLOW,
    ACCENT_CYAN, ACCENT_ORANGE, ACCENT_RED,
    FONT_LABEL, FONT_SMALL, FONT_MONO,
    STATUS_UPDATE_MS,
)

# Farben fuer Zones
ZONE_COLORS = {
    "guardian": ACCENT_CYAN,
    "shadow": ACCENT_RED,
    "berserker": STATUS_RED,
}

# Farben fuer Bridge States
BRIDGE_COLORS = {
    "idle": FG_DIM,
    "searching": STATUS_YELLOW,
    "tracking": STATUS_GREEN,
    "interaction": ACCENT_CYAN,
    "manual_override": STATUS_RED,
}


class SystemStatusModule:
    """Systemstatus-Anzeige im uebergebenen LabelFrame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main (frame_steuerung)
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None

        # GUI aufbauen
        self._build_bridge_section()
        self._build_core_section()
        self._build_trends_section()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # Sektion 1: Bridge + Erkennung
    # =========================================================================

    def _build_bridge_section(self):
        """Bridge State + Face-ID + Owner."""
        section = tk.LabelFrame(
            self._parent,
            text="Bridge / Erkennung",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Zeile 1: Bridge State + Alter
        row1 = tk.Frame(section, bg=BG_FRAME)
        row1.pack(fill=tk.X, padx=8, pady=(4, 1))

        tk.Label(
            row1, text="Bridge:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_bridge_state = tk.Label(
            row1, text="---", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_bridge_state.pack(side=tk.LEFT, padx=5)

        self._lbl_bridge_age = tk.Label(
            row1, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_bridge_age.pack(side=tk.LEFT, padx=3)

        self._lbl_bridge_decisions = tk.Label(
            row1, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_bridge_decisions.pack(side=tk.RIGHT, padx=5)

        # Zeile 2: Face-ID + Confidence
        row2 = tk.Frame(section, bg=BG_FRAME)
        row2.pack(fill=tk.X, padx=8, pady=(1, 4))

        tk.Label(
            row2, text="Face:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_face_id = tk.Label(
            row2, text="---", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_face_id.pack(side=tk.LEFT, padx=5)

        self._lbl_owner_status = tk.Label(
            row2, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_owner_status.pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # Sektion 2: Core Innenleben
    # =========================================================================

    def _build_core_section(self):
        """Tension, Zone, Personality, Tageszeit."""
        section = tk.LabelFrame(
            self._parent,
            text="Innenleben",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=2)

        # Zeile 1: Tension-Bar
        row_t = tk.Frame(section, bg=BG_FRAME)
        row_t.pack(fill=tk.X, padx=8, pady=(4, 1))

        tk.Label(
            row_t, text="Tension:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        # Canvas fuer Tension-Bar (kleiner Balken)
        self._tension_canvas = tk.Canvas(
            row_t, width=100, height=12, bg=BG_INPUT, highlightthickness=0,
        )
        self._tension_canvas.pack(side=tk.LEFT, padx=5)
        self._tension_bar = self._tension_canvas.create_rectangle(
            0, 0, 0, 12, fill=STATUS_GREEN, outline="",
        )

        self._lbl_tension_val = tk.Label(
            row_t, text="0.00", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_tension_val.pack(side=tk.LEFT, padx=3)

        self._lbl_dominance = tk.Label(
            row_t, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_dominance.pack(side=tk.RIGHT, padx=5)

        # Zeile 2: Zone + Personality + Tageszeit
        row_z = tk.Frame(section, bg=BG_FRAME)
        row_z.pack(fill=tk.X, padx=8, pady=(1, 4))

        tk.Label(
            row_z, text="Zone:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_zone = tk.Label(
            row_z, text="---", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_zone.pack(side=tk.LEFT, padx=5)

        self._lbl_personality = tk.Label(
            row_z, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_personality.pack(side=tk.LEFT, padx=10)

        self._lbl_time_period = tk.Label(
            row_z, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_time_period.pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # Sektion 3: Trends
    # =========================================================================

    def _build_trends_section(self):
        """Approaching/Leaving, Distance, Presence Duration."""
        section = tk.LabelFrame(
            self._parent,
            text="Aktivitaet",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=4)

        # Presence / Absence
        tk.Label(
            row, text="Praesenz:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_presence = tk.Label(
            row, text="---", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_presence.pack(side=tk.LEFT, padx=5)

        # Distanz
        self._lbl_distance = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_distance.pack(side=tk.LEFT, padx=10)

        # Bewegungsrichtung
        self._lbl_motion = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_motion.pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Status lesen und alle Anzeigen aktualisieren."""
        status = self._service.read_status()

        if status:
            self._update_bridge(status)
            self._update_core(status)
            self._update_trends(status)

        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)

    def _update_bridge(self, status):
        """Bridge + Face-ID aktualisieren."""
        bridge = status.get("bridge", {})

        # Bridge State
        state = bridge.get("state", "---")
        color = BRIDGE_COLORS.get(state, FG_DIM)
        self._lbl_bridge_state.config(text=state.upper(), fg=color)

        # State-Alter
        age = bridge.get("state_age_s", 0)
        if isinstance(age, (int, float)):
            if age < 60:
                self._lbl_bridge_age.config(text=f"({age:.0f}s)")
            else:
                self._lbl_bridge_age.config(text=f"({age/60:.0f}m)")
        else:
            self._lbl_bridge_age.config(text="")

        # Decisions
        dec = bridge.get("decisions", 0)
        self._lbl_bridge_decisions.config(text=f"Dec: {dec}")

        # Face-ID + ArcFace Similarity
        face_id = status.get("face_id", "")
        face_sim = status.get("face_similarity", 0.0)
        face_detected = status.get("face_detected", False)

        if face_id and face_detected:
            self._lbl_face_id.config(
                text=f"{face_id} ({face_sim:.0%})",
                fg=ACCENT_CYAN if face_sim > 0.5 else STATUS_YELLOW,
            )
        elif face_detected:
            self._lbl_face_id.config(text="unbekannt", fg=STATUS_YELLOW)
        else:
            self._lbl_face_id.config(text="---", fg=FG_DIM)

        # Owner-Status
        owner = bridge.get("owner_detected", False)
        owner_name = bridge.get("owner_name", "")
        if owner and owner_name:
            self._lbl_owner_status.config(
                text=f"Owner: {owner_name}",
                fg=ACCENT_CYAN,
            )
        else:
            self._lbl_owner_status.config(text="", fg=FG_DIM)

    def _update_core(self, status):
        """Tension, Zone, Personality, Tageszeit aktualisieren."""
        core = status.get("core", {})

        # Tension (0.0 - 1.0)
        tension = core.get("tension", 0.0)
        if not isinstance(tension, (int, float)):
            tension = 0.0
        tension = max(0.0, min(1.0, tension))

        # Tension-Bar Farbe: gruen < 0.3, gelb < 0.6, orange < 0.8, rot >= 0.8
        if tension < 0.3:
            bar_color = STATUS_GREEN
        elif tension < 0.6:
            bar_color = STATUS_YELLOW
        elif tension < 0.8:
            bar_color = ACCENT_ORANGE
        else:
            bar_color = STATUS_RED

        bar_width = int(tension * 100)
        self._tension_canvas.coords(self._tension_bar, 0, 0, bar_width, 12)
        self._tension_canvas.itemconfig(self._tension_bar, fill=bar_color)
        self._lbl_tension_val.config(text=f"{tension:.2f}", fg=bar_color)

        # Dominance
        dom = core.get("dominance", 0.0)
        if isinstance(dom, (int, float)):
            self._lbl_dominance.config(text=f"Dom: {dom:+.2f}")

        # Zone
        zone = core.get("zone", "---")
        zone_color = ZONE_COLORS.get(zone, FG_DIM)
        self._lbl_zone.config(text=zone.upper(), fg=zone_color)

        # Personality Mode
        pers = status.get("personality_mode", "")
        if pers:
            pers_color = ZONE_COLORS.get(pers, FG_LABEL)
            self._lbl_personality.config(text=f"Pers: {pers}", fg=pers_color)

        # Tageszeit
        tp = core.get("time_period", "")
        if tp:
            self._lbl_time_period.config(text=tp)

    def _update_trends(self, status):
        """Trends + Activity aktualisieren."""
        core = status.get("core", {})
        trends = core.get("trends", {})

        # Presence Duration
        pres = trends.get("presence_duration", 0.0)
        abse = trends.get("absence_duration", 0.0)
        person = trends.get("smoothed_person", False)

        if person and isinstance(pres, (int, float)) and pres > 0:
            if pres < 60:
                self._lbl_presence.config(
                    text=f"da seit {pres:.0f}s", fg=STATUS_GREEN,
                )
            else:
                self._lbl_presence.config(
                    text=f"da seit {pres/60:.0f}m", fg=STATUS_GREEN,
                )
        elif isinstance(abse, (int, float)) and abse > 0:
            if abse < 60:
                self._lbl_presence.config(
                    text=f"weg seit {abse:.0f}s", fg=FG_DIM,
                )
            else:
                self._lbl_presence.config(
                    text=f"weg seit {abse/60:.0f}m", fg=FG_DIM,
                )
        else:
            self._lbl_presence.config(text="---", fg=FG_DIM)

        # Distanz
        dist = trends.get("smoothed_distance", "")
        if dist:
            dist_map = {"close": "Nah", "medium": "Mittel", "far": "Fern"}
            dist_text = dist_map.get(dist, dist)
            self._lbl_distance.config(text=f"Dist: {dist_text}")

        # Bewegung
        approaching = trends.get("approaching", False)
        leaving = trends.get("leaving", False)
        if approaching:
            self._lbl_motion.config(text="-> Kommt", fg=STATUS_GREEN)
        elif leaving:
            self._lbl_motion.config(text="<- Geht", fg=STATUS_YELLOW)
        else:
            self._lbl_motion.config(text="", fg=FG_DIM)
