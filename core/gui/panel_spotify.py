#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Spotify
=============================

Spotify-Steuerung fuer das Panel.
- Aktueller Track + Artist
- Play/Pause/Skip/Previous Buttons
- Auto-DJ Toggle (Zone-basiert)
- Volume Slider
- Smart-Buttons: Aehnlich, Top Tracks, Neue Musik

Bekommt parent_frame und ServiceProxy von panel_main.
Importiert NUR panel_styles und tkinter.
"""

import tkinter as tk
from tkinter import ttk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON, BG_INPUT,
    BTN_OFF_DARK, BTN_ON_GREEN, BTN_ON_ORANGE,
    ACCENT_CYAN,
    FG_TEXT, FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL, FONT_MONO,
    STATUS_UPDATE_MS, STATUS_GREEN, STATUS_RED,
)


class SpotifyModule:
    """Spotify-Steuerung im uebergebenen Frame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: Frame von panel_main
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None
        self._volume_var = tk.IntVar(value=50)
        self._auto_dj_active = False

        self._build_ui()
        self._poll_spotify_state()

    def _build_ui(self):
        """Spotify-Sektion aufbauen."""
        section = tk.LabelFrame(
            self._parent,
            text="SPOTIFY",
            bg=BG_FRAME,
            fg=ACCENT_CYAN,
            font=FONT_LABEL,
            padx=5, pady=3,
        )
        section.pack(fill="x", padx=5, pady=(5, 2))

        # --- Track-Anzeige ---
        self._track_label = tk.Label(
            section,
            text="Kein Track",
            bg=BG_INPUT,
            fg=FG_TEXT,
            font=FONT_MONO,
            anchor="w",
            padx=4, pady=2,
        )
        self._track_label.pack(fill="x", pady=(2, 1))

        self._artist_label = tk.Label(
            section,
            text="",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=FONT_SMALL,
            anchor="w",
            padx=4,
        )
        self._artist_label.pack(fill="x", pady=(0, 3))

        # --- Transport Buttons ---
        transport_frame = tk.Frame(section, bg=BG_FRAME)
        transport_frame.pack(fill="x", pady=2)

        btn_cfg = dict(
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BTN_OFF_DARK, activeforeground=FG_WHITE,
            relief="flat", bd=0, padx=6, pady=2,
        )

        tk.Button(
            transport_frame, text="<<", command=self._on_previous, **btn_cfg,
        ).pack(side="left", padx=1)

        self._btn_play = tk.Button(
            transport_frame, text="PLAY", command=self._on_toggle,
            bg=BTN_ON_GREEN, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BTN_ON_GREEN, activeforeground=FG_WHITE,
            relief="flat", bd=0, padx=10, pady=2,
        )
        self._btn_play.pack(side="left", padx=1)

        tk.Button(
            transport_frame, text=">>", command=self._on_skip, **btn_cfg,
        ).pack(side="left", padx=1)

        # Auto-DJ Button
        self._btn_auto_dj = tk.Button(
            transport_frame, text="AUTO-DJ", command=self._on_auto_dj_toggle,
            bg=BTN_OFF_DARK, fg=FG_DIM, font=FONT_BUTTON,
            activebackground=BTN_OFF_DARK, activeforeground=FG_WHITE,
            relief="flat", bd=0, padx=6, pady=2,
        )
        self._btn_auto_dj.pack(side="right", padx=1)

        # --- Volume ---
        vol_frame = tk.Frame(section, bg=BG_FRAME)
        vol_frame.pack(fill="x", pady=(2, 1))

        tk.Label(
            vol_frame, text="VOL", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side="left")

        self._vol_scale = tk.Scale(
            vol_frame,
            from_=0, to=100,
            orient="horizontal",
            variable=self._volume_var,
            bg=BG_FRAME, fg=FG_TEXT,
            troughcolor=BG_INPUT,
            highlightthickness=0,
            sliderrelief="flat",
            showvalue=False,
            length=120,
            command=self._on_volume_change,
        )
        self._vol_scale.pack(side="left", fill="x", expand=True, padx=3)

        self._vol_label = tk.Label(
            vol_frame, text="50%", bg=BG_FRAME, fg=FG_DIM,
            font=FONT_SMALL, width=4,
        )
        self._vol_label.pack(side="left")

        # --- Smart Buttons ---
        smart_frame = tk.Frame(section, bg=BG_FRAME)
        smart_frame.pack(fill="x", pady=(2, 3))

        smart_cfg = dict(
            bg=BG_BUTTON, fg=FG_DIM, font=FONT_SMALL,
            activebackground=BTN_OFF_DARK, activeforeground=FG_WHITE,
            relief="flat", bd=0, padx=4, pady=1,
        )

        tk.Button(
            smart_frame, text="Aehnlich", command=self._on_similar, **smart_cfg,
        ).pack(side="left", padx=1)

        tk.Button(
            smart_frame, text="Top", command=self._on_top_tracks, **smart_cfg,
        ).pack(side="left", padx=1)

        tk.Button(
            smart_frame, text="Neu", command=self._on_new_music, **smart_cfg,
        ).pack(side="left", padx=1)

        # Zone-Label
        self._zone_label = tk.Label(
            smart_frame, text="", bg=BG_FRAME, fg=FG_DIM,
            font=FONT_SMALL,
        )
        self._zone_label.pack(side="right", padx=2)

    # === Callbacks ===

    def _on_toggle(self):
        self._service.spotify_toggle()

    def _on_skip(self):
        self._service.spotify_skip()

    def _on_previous(self):
        self._service.spotify_previous()

    def _on_auto_dj_toggle(self):
        self._service.spotify_auto_dj("toggle")

    def _on_volume_change(self, val):
        vol = int(float(val))
        self._vol_label.config(text=f"{vol}%")
        self._service.spotify_volume(vol)

    def _on_similar(self):
        self._service.spotify_similar()

    def _on_top_tracks(self):
        self._service.spotify_top_tracks()

    def _on_new_music(self):
        self._service.spotify_new_music()

    # === Status-Polling ===

    def _poll_spotify_state(self):
        """Spotify-Status aus Service lesen und UI aktualisieren."""
        try:
            status = self._service.read_status()
            spotify = status.get("spotify", {})

            if spotify:
                # Track-Anzeige
                track = spotify.get("current_track")
                if track:
                    name = track.get("track", "?")
                    artist = track.get("artist", "?")
                    is_playing = track.get("is_playing", False)

                    # Track-Name kuerzen wenn zu lang
                    max_len = 30
                    if len(name) > max_len:
                        name = name[:max_len - 1] + "\u2026"

                    self._track_label.config(
                        text=name,
                        fg=FG_TEXT if is_playing else FG_DIM,
                    )
                    self._artist_label.config(text=artist)

                    # Play/Pause Button
                    if is_playing:
                        self._btn_play.config(text="PAUSE", bg=BTN_ON_ORANGE)
                    else:
                        self._btn_play.config(text="PLAY", bg=BTN_ON_GREEN)

                    # Volume Sync (nur wenn User nicht gerade schiebt)
                    vol = track.get("volume", 0)
                    if abs(self._volume_var.get() - vol) > 5:
                        self._volume_var.set(vol)
                        self._vol_label.config(text=f"{vol}%")
                else:
                    self._track_label.config(text="Kein Track", fg=FG_DIM)
                    self._artist_label.config(text="")
                    self._btn_play.config(text="PLAY", bg=BTN_ON_GREEN)

                # Auto-DJ Status
                auto_dj = spotify.get("auto_dj", False)
                auto_dj_zone = spotify.get("auto_dj_zone", "")
                if auto_dj:
                    self._btn_auto_dj.config(
                        bg=BTN_ON_GREEN, fg=FG_WHITE,
                        text=f"DJ:{auto_dj_zone.upper()}" if auto_dj_zone else "AUTO-DJ",
                    )
                    self._auto_dj_active = True
                else:
                    self._btn_auto_dj.config(
                        bg=BTN_OFF_DARK, fg=FG_DIM, text="AUTO-DJ",
                    )
                    self._auto_dj_active = False
            else:
                self._track_label.config(text="Spotify nicht verbunden", fg=FG_DIM)
                self._artist_label.config(text="")

            # Zone-Label aus Core State
            core = status.get("core", {})
            zone = core.get("zone", "")
            if zone:
                zone_colors = {
                    "guardian": STATUS_GREEN,
                    "shadow": "#aa44ff",
                    "berserker": "#ff4444",
                }
                self._zone_label.config(
                    text=zone.upper(),
                    fg=zone_colors.get(zone, FG_DIM),
                )

        except Exception:
            pass

        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_spotify_state)
