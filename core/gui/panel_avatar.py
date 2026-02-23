#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Avatar — Das Auge
======================================

HAL-9000 inspiriertes mechanisches Auge als Core State Visualisierung.
Tkinter Canvas Widget, 300x300 Pixel.

Mapping:
  TENSION  -> Ring-Farbe + Glow (Guardian=Blau, Shadow=Lila, Berserker=Rot)
  ATTENTION -> Pupille + Blinzeln
  PRESENCE  -> Gesamthelligkeit

Rein passiv — liest NUR Core State, schreibt NICHTS.
"""

import tkinter as tk
import math
import random
import time

from core.gui.panel_styles import BG_DARK, BG_FRAME, FG_DIM, FONT_SMALL


# =============================================================================
# Farb-Konstanten pro Zone
# =============================================================================
COLOR_GUARDIAN = "#00BFFF"   # Blau
COLOR_SHADOW = "#9933FF"     # Lila
COLOR_BERSERKER = "#FF0000"  # Rot

# Abgedunkelte Versionen fuer Ring-Aussen
COLOR_GUARDIAN_DIM = "#005577"
COLOR_SHADOW_DIM = "#441177"
COLOR_BERSERKER_DIM = "#770000"

# Iris-Farben (heller als Ring)
COLOR_GUARDIAN_IRIS = "#33DDFF"
COLOR_SHADOW_IRIS = "#BB66FF"
COLOR_BERSERKER_IRIS = "#FF3333"

# Animation
ANIM_INTERVAL_MS = 50     # 20 FPS Render
BLINK_DURATION_MS = 150   # Blinzel-Dauer
BERSERKER_FLASH_MS = 200  # Rot-Flash Dauer

# Canvas-Groesse
AVATAR_SIZE = 300
CENTER = AVATAR_SIZE // 2  # 150


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Lineares Interpolieren zwischen zwei Hex-Farben."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _scale_color(color: str, factor: float) -> str:
    """Farbe um Faktor skalieren (Helligkeit)."""
    r = min(255, int(int(color[1:3], 16) * factor))
    g = min(255, int(int(color[3:5], 16) * factor))
    b = min(255, int(int(color[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


class AvatarModule:
    """
    M.O.L.O.C.H. Avatar-Auge — Core State Visualisierung.

    Wird in einen parent_frame eingebettet.
    Liest Core State via service_proxy.read_status()["core"].
    """

    def __init__(self, parent_frame, service_proxy):
        self._parent = parent_frame
        self._service = service_proxy
        self._running = False
        self._after_id = None

        # --- Core State Cache ---
        self._tension = 0.0
        self._attention = 0.5
        self._presence = 0.0
        self._zone = "guardian"

        # --- Animation State ---
        self._tick = 0
        self._blink_progress = 0.0   # 0=offen, 1=geschlossen
        self._blinking = False
        self._blink_opening = False
        self._next_blink_tick = self._random_blink_tick()
        self._flash_until = 0.0       # Berserker-Flash Zeitstempel
        self._pupil_dx = 0.0          # Mikro-Bewegung X Offset
        self._pupil_dy = 0.0          # Mikro-Bewegung Y Offset
        self._target_dx = 0.0
        self._target_dy = 0.0
        self._pulse_phase = 0.0       # Ring-Puls Phase (0..2*pi)

        # --- Canvas ---
        self._canvas = tk.Canvas(
            parent_frame,
            width=AVATAR_SIZE,
            height=AVATAR_SIZE,
            bg="#000000",
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(padx=5, pady=5)

        # Zone-Label unter dem Auge
        self._zone_label = tk.Label(
            parent_frame,
            text="OFFLINE",
            bg=BG_FRAME,
            fg=FG_DIM,
            font=FONT_SMALL,
        )
        self._zone_label.pack(pady=(0, 5))

        # Initialer Draw
        self._draw_eye()

    def _random_blink_tick(self) -> int:
        """Naechsten Blink-Zeitpunkt berechnen (4-9 Sekunden = 80-180 Ticks)."""
        return self._tick + random.randint(80, 180)

    def _get_zone_colors(self):
        """Aktive Farben basierend auf Zone und Tension."""
        if self._zone == "berserker":
            return COLOR_BERSERKER, COLOR_BERSERKER_DIM, COLOR_BERSERKER_IRIS
        elif self._zone == "shadow":
            # Uebergang: interpoliere zwischen Guardian und Shadow basierend auf Tension
            t = max(0.0, (self._tension - 0.4) / 0.35)  # 0.4->0, 0.75->1
            main = _lerp_color(COLOR_GUARDIAN, COLOR_SHADOW, t)
            dim = _lerp_color(COLOR_GUARDIAN_DIM, COLOR_SHADOW_DIM, t)
            iris = _lerp_color(COLOR_GUARDIAN_IRIS, COLOR_SHADOW_IRIS, t)
            return main, dim, iris
        else:
            return COLOR_GUARDIAN, COLOR_GUARDIAN_DIM, COLOR_GUARDIAN_IRIS

    def _draw_eye(self):
        """Komplettes Auge neu zeichnen."""
        c = self._canvas
        c.delete("all")

        # Presence steuert Gesamthelligkeit (0.15 minimum damit nie ganz unsichtbar)
        brightness = 0.15 + self._presence * 0.85

        color_main, color_dim, color_iris = self._get_zone_colors()

        # Berserker-Flash uebersteuert
        now = time.monotonic()
        if now < self._flash_until:
            brightness = 1.0
            color_main = "#FF4444"
            color_iris = "#FFAAAA"

        # Puls-Faktor (Ring pulsiert)
        pulse_speed = 1.0 + self._tension * 3.0  # Hoeher bei Tension
        self._pulse_phase += pulse_speed * 0.1
        pulse_scale = 1.0 + math.sin(self._pulse_phase) * 0.02
        if self._zone == "berserker":
            pulse_scale = 1.0 + math.sin(self._pulse_phase) * 0.04  # Aggressiver

        # === GLOW-EFFEKT (mehrere transparente Kreise) ===
        glow_intensity = brightness * (0.3 + self._tension * 0.7)
        glow_color = _scale_color(color_main, glow_intensity * 0.3)
        for i in range(4, 0, -1):
            r = int(130 * pulse_scale) + i * 12
            gc = _scale_color(color_main, glow_intensity * (0.08 + i * 0.04))
            c.create_oval(
                CENTER - r, CENTER - r, CENTER + r, CENTER + r,
                fill=gc, outline="", width=0,
            )

        # === AEUSSERER RING ===
        ring_outer_r = int(120 * pulse_scale)
        ring_color = _scale_color(color_dim, brightness)
        c.create_oval(
            CENTER - ring_outer_r, CENTER - ring_outer_r,
            CENTER + ring_outer_r, CENTER + ring_outer_r,
            fill="", outline=ring_color, width=3,
        )

        # === INNERER RING ===
        ring_inner_r = int(95 * pulse_scale)
        ring_inner_color = _scale_color(color_main, brightness * 0.7)
        c.create_oval(
            CENTER - ring_inner_r, CENTER - ring_inner_r,
            CENTER + ring_inner_r, CENTER + ring_inner_r,
            fill="", outline=ring_inner_color, width=2,
        )

        # === Ring-Segmente (mechanischer Look) ===
        # 8 kleine Striche auf dem aeusseren Ring
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = CENTER + math.cos(angle) * (ring_outer_r - 8)
            y1 = CENTER + math.sin(angle) * (ring_outer_r - 8)
            x2 = CENTER + math.cos(angle) * (ring_outer_r + 8)
            y2 = CENTER + math.sin(angle) * (ring_outer_r + 8)
            c.create_line(x1, y1, x2, y2, fill=ring_color, width=1)

        # === IRIS (gefuellter Kreis) ===
        iris_r = int(55 * brightness)
        iris_color = _scale_color(color_iris, brightness * 0.5)
        # Iris-Position (Mikro-Bewegung)
        ix = CENTER + self._pupil_dx
        iy = CENTER + self._pupil_dy
        c.create_oval(
            ix - iris_r, iy - iris_r, ix + iris_r, iy + iris_r,
            fill=iris_color, outline=_scale_color(color_main, brightness * 0.6), width=1,
        )

        # === PUPILLE (innerer Kern) ===
        # Attention steuert Pupillengroesse: hoch=gross, niedrig=klein
        base_pupil = 15 + self._attention * 20  # 15-35 Pixel Radius
        if now < self._flash_until:
            base_pupil = 10  # Berserker-Flash: Pupille verengt
        pupil_r = int(base_pupil * brightness)
        pupil_color = _scale_color(color_main, brightness)
        c.create_oval(
            ix - pupil_r, iy - pupil_r, ix + pupil_r, iy + pupil_r,
            fill=pupil_color, outline="", width=0,
        )

        # Pupillen-Highlight (Lichtreflex)
        hl_r = max(3, pupil_r // 3)
        hl_x = ix - pupil_r * 0.3
        hl_y = iy - pupil_r * 0.3
        hl_color = _scale_color("#FFFFFF", brightness * 0.6)
        c.create_oval(
            hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r,
            fill=hl_color, outline="", width=0,
        )

        # === BLINZELN (Augenlider von oben und unten) ===
        if self._blink_progress > 0.0:
            lid_height = int(AVATAR_SIZE * 0.5 * self._blink_progress)
            lid_color = "#000000"
            # Oberes Lid
            c.create_rectangle(
                0, 0, AVATAR_SIZE, lid_height,
                fill=lid_color, outline="", width=0,
            )
            # Unteres Lid
            c.create_rectangle(
                0, AVATAR_SIZE - lid_height, AVATAR_SIZE, AVATAR_SIZE,
                fill=lid_color, outline="", width=0,
            )
            # Lid-Kanten (mechanisch)
            lid_edge = _scale_color(color_dim, brightness * 0.5)
            if lid_height > 5:
                c.create_line(0, lid_height, AVATAR_SIZE, lid_height,
                              fill=lid_edge, width=2)
                c.create_line(0, AVATAR_SIZE - lid_height,
                              AVATAR_SIZE, AVATAR_SIZE - lid_height,
                              fill=lid_edge, width=2)

        # === HALBES AUGE bei niedriger Attention ===
        if self._attention < 0.3 and self._blink_progress == 0.0:
            # Lider halb geschlossen
            droop = int((0.3 - self._attention) / 0.3 * AVATAR_SIZE * 0.25)
            lid_color = "#000000"
            lid_edge = _scale_color(color_dim, brightness * 0.4)
            c.create_rectangle(0, 0, AVATAR_SIZE, droop,
                               fill=lid_color, outline="")
            c.create_rectangle(0, AVATAR_SIZE - droop, AVATAR_SIZE, AVATAR_SIZE,
                               fill=lid_color, outline="")
            if droop > 3:
                c.create_line(0, droop, AVATAR_SIZE, droop,
                              fill=lid_edge, width=1)
                c.create_line(0, AVATAR_SIZE - droop, AVATAR_SIZE, AVATAR_SIZE - droop,
                              fill=lid_edge, width=1)

        # Zone-Label aktualisieren
        zone_text = {
            "guardian": "GUARDIAN",
            "shadow": "SHADOW",
            "berserker": "BERSERKER",
        }.get(self._zone, "OFFLINE")
        self._zone_label.config(
            text=f"{zone_text} | T:{self._tension:.2f} A:{self._attention:.2f} P:{self._presence:.2f}",
            fg=_scale_color(color_main, max(0.4, brightness)),
        )

    def _update_animation(self):
        """Animation-Tick: Blinzeln, Mikro-Bewegung, Status lesen."""
        if not self._running:
            return

        self._tick += 1

        # --- Core State lesen (alle 10 Ticks = 500ms) ---
        if self._tick % 10 == 0:
            self._read_core_state()

        # --- Blinzel-Logik ---
        if self._blinking:
            if not self._blink_opening:
                # Auge schliesst sich
                self._blink_progress += 0.25  # 4 Frames zum Schliessen
                if self._blink_progress >= 1.0:
                    self._blink_progress = 1.0
                    self._blink_opening = True
            else:
                # Auge oeffnet sich
                self._blink_progress -= 0.25
                if self._blink_progress <= 0.0:
                    self._blink_progress = 0.0
                    self._blinking = False
                    self._blink_opening = False
                    self._next_blink_tick = self._random_blink_tick()
        elif self._tick >= self._next_blink_tick:
            # Hohe Attention = weniger Blinzeln (Chance zu ueberspringen)
            if self._attention > 0.7 and random.random() < 0.5:
                self._next_blink_tick = self._random_blink_tick()
            else:
                self._blinking = True
                self._blink_opening = False
                # Niedrige Attention = langsameres Blinzeln
                if self._attention < 0.3:
                    self._blink_progress = 0.0  # Startet langsamer

        # --- Mikro-Bewegung ---
        if self._tick % 20 == 0:
            # Neues Ziel alle 1 Sekunde
            self._target_dx = random.uniform(-2, 2)
            self._target_dy = random.uniform(-2, 2)

        # Smooth interpolation zum Ziel
        self._pupil_dx += (self._target_dx - self._pupil_dx) * 0.1
        self._pupil_dy += (self._target_dy - self._pupil_dy) * 0.1

        # --- Berserker-Flash bei Zonenuebergang ---
        # (wird in _read_core_state ausgeloest)

        # --- Zeichnen ---
        self._draw_eye()

        # Naechster Tick
        self._after_id = self._parent.after(ANIM_INTERVAL_MS, self._update_animation)

    def _read_core_state(self):
        """Core State aus Status-JSON lesen."""
        status = self._service.read_status()
        if not status:
            return

        core = status.get("core", {})
        if not core:
            return

        old_zone = self._zone

        self._tension = float(core.get("tension", 0.0))
        self._attention = float(core.get("attention", 0.5))
        self._presence = float(core.get("presence", 0.0))
        self._zone = core.get("zone", "guardian")

        # Berserker-Flash bei Eintritt in Berserker-Zone
        if self._zone == "berserker" and old_zone != "berserker":
            self._flash_until = time.monotonic() + BERSERKER_FLASH_MS / 1000.0

    def start(self):
        """Animation starten."""
        if self._running:
            return
        self._running = True
        self._update_animation()

    def stop(self):
        """Animation stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None
