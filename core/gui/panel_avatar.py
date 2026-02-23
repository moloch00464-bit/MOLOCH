#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Avatar — Das Auge
======================================

HAL-9000 inspiriertes mechanisches Auge als Core State Visualisierung.
Tkinter Canvas Widget, 300x300 Pixel. Dark Wave Aesthetik.

Mapping:
  TENSION   -> Ring-Farbe + Glow (Guardian=Blau, Shadow=Weiss/Lila, Berserker=Rot)
  ATTENTION -> Pupillengroesse + Blinzeln
  PRESENCE  -> Gesamthelligkeit

Rein passiv — liest NUR Core State, schreibt NICHTS.
"""

import tkinter as tk
import math
import random
import time

from core.gui.panel_styles import BG_FRAME, FG_DIM


# =============================================================================
# Farb-Konstanten pro Zone (kraeftig, leuchtend)
# =============================================================================
COLOR_GUARDIAN = "#00AAFF"        # Leuchtendes Blau
COLOR_SHADOW = "#FFFFFF"          # Helles Weiss
COLOR_BERSERKER = "#FF2200"       # Intensives Rot

COLOR_GUARDIAN_DIM = "#005588"
COLOR_SHADOW_DIM = "#9977AA"      # Lila-Schimmer
COLOR_BERSERKER_DIM = "#881100"

COLOR_GUARDIAN_IRIS = "#44CCFF"
COLOR_SHADOW_IRIS = "#EEDDFF"     # Leichter Lila-Ton
COLOR_BERSERKER_IRIS = "#FF4433"

# Hintergrund
BG_AVATAR = "#0A0A14"            # Sehr dunkles Blau
GRID_COLOR = "#111120"            # Subtile Grid-Linien

# Animation
ANIM_INTERVAL_MS = 50
BERSERKER_FLASH_MS = 200

# Canvas-Groesse
AVATAR_SIZE = 300
CENTER = AVATAR_SIZE // 2


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
    factor = max(0.0, factor)
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

        # --- Core State ---
        self._tension = 0.0
        self._attention = 0.5
        self._presence = 0.0
        self._zone = "guardian"
        self._status_text = "Idle"

        # --- Smooth Animation (interpoliert zum Ziel) ---
        self._s_tension = 0.0
        self._s_attention = 0.5
        self._s_presence = 0.0

        # --- Animation State ---
        self._tick = 0
        self._blink_progress = 0.0
        self._blinking = False
        self._blink_opening = False
        self._next_blink_tick = self._random_blink_tick()
        self._flash_until = 0.0
        self._pupil_dx = 0.0
        self._pupil_dy = 0.0
        self._target_dx = 0.0
        self._target_dy = 0.0
        self._pulse_phase = 0.0

        # --- Canvas ---
        self._canvas = tk.Canvas(
            parent_frame,
            width=AVATAR_SIZE,
            height=AVATAR_SIZE,
            bg=BG_AVATAR,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(padx=5, pady=5)

        # --- Info-Bereich unter dem Auge ---
        info = tk.Frame(parent_frame, bg=BG_FRAME)
        info.pack(fill="x", padx=5, pady=(0, 3))

        self._zone_label = tk.Label(
            info, text="OFFLINE", bg=BG_FRAME, fg=FG_DIM,
            font=("Helvetica", 11, "bold"),
        )
        self._zone_label.pack()

        self._bars_canvas = tk.Canvas(
            info, width=200, height=36, bg=BG_FRAME,
            highlightthickness=0, bd=0,
        )
        self._bars_canvas.pack(pady=(2, 0))

        self._status_label = tk.Label(
            info, text="Idle", bg=BG_FRAME, fg=FG_DIM,
            font=("Helvetica", 9),
        )
        self._status_label.pack(pady=(0, 2))

        # Initialer Draw
        self._draw_eye()
        self._draw_bars()

    # =========================================================================
    # Hilfsfunktionen
    # =========================================================================

    def _random_blink_tick(self) -> int:
        """Naechsten Blink-Zeitpunkt (4-9 Sekunden = 80-180 Ticks bei 20 FPS)."""
        return self._tick + random.randint(80, 180)

    def _get_zone_colors(self):
        """Farben fuer aktive Zone. Gibt (main, dim, iris) zurueck."""
        if self._zone == "berserker":
            return COLOR_BERSERKER, COLOR_BERSERKER_DIM, COLOR_BERSERKER_IRIS
        elif self._zone == "shadow":
            t = max(0.0, (self._s_tension - 0.4) / 0.35)
            main = _lerp_color(COLOR_GUARDIAN, COLOR_SHADOW, t)
            dim = _lerp_color(COLOR_GUARDIAN_DIM, COLOR_SHADOW_DIM, t)
            iris = _lerp_color(COLOR_GUARDIAN_IRIS, COLOR_SHADOW_IRIS, t)
            return main, dim, iris
        return COLOR_GUARDIAN, COLOR_GUARDIAN_DIM, COLOR_GUARDIAN_IRIS

    # =========================================================================
    # Zeichnen — Hauptfunktion
    # =========================================================================

    def _draw_eye(self):
        """Komplettes Auge zeichnen (1 Frame)."""
        c = self._canvas
        c.delete("all")

        # Smooth Interpolation der Werte
        self._s_tension += (self._tension - self._s_tension) * 0.12
        self._s_attention += (self._attention - self._s_attention) * 0.12
        self._s_presence += (self._presence - self._s_presence) * 0.12

        brightness = 0.4 + self._s_presence * 0.6
        color_main, color_dim, color_iris = self._get_zone_colors()

        # Berserker-Flash
        now = time.monotonic()
        flash = now < self._flash_until
        if flash:
            brightness = 1.0
            color_main = "#FF4444"
            color_iris = "#FFAAAA"

        # Puls-Faktor
        speed = 1.0 + self._s_tension * 3.0
        self._pulse_phase += speed * 0.1
        pulse = 1.0 + math.sin(self._pulse_phase) * 0.02
        if self._zone == "berserker":
            pulse = 1.0 + math.sin(self._pulse_phase) * 0.06

        # --- Hintergrund-Grid (Tron-Style) ---
        for x in range(0, AVATAR_SIZE, 25):
            c.create_line(x, 0, x, AVATAR_SIZE, fill=GRID_COLOR)
        for y in range(0, AVATAR_SIZE, 25):
            c.create_line(0, y, AVATAR_SIZE, y, fill=GRID_COLOR)

        # --- Glow (mehrere Schichten, staerker bei Berserker) ---
        glow_str = brightness * (0.4 + self._s_tension * 0.6)
        layers = 7 if self._zone == "berserker" else 5
        for i in range(layers, 0, -1):
            r = int(130 * pulse) + i * 16
            gc = _scale_color(color_main, glow_str * (0.04 + i * 0.025))
            c.create_oval(CENTER - r, CENTER - r, CENTER + r, CENTER + r,
                          fill=gc, outline="")

        # --- Aeusserer Ring (dick, mechanisch) ---
        ro = int(120 * pulse)
        ring_c = _scale_color(color_main, brightness * 0.8)
        c.create_oval(CENTER - ro, CENTER - ro, CENTER + ro, CENTER + ro,
                      fill="", outline=ring_c, width=4)

        # 12 Tick-Marks wie Uhr-Striche
        for i in range(12):
            angle = math.radians(i * 30 - 90)  # 12 Uhr = oben
            tl = 12 if i % 3 == 0 else 6       # Viertelstunden laenger
            tw = 2 if i % 3 == 0 else 1
            x1 = CENTER + math.cos(angle) * (ro - tl)
            y1 = CENTER + math.sin(angle) * (ro - tl)
            x2 = CENTER + math.cos(angle) * (ro + 5)
            y2 = CENTER + math.sin(angle) * (ro + 5)
            tc = _scale_color(color_main, brightness * 0.55)
            c.create_line(x1, y1, x2, y2, fill=tc, width=tw)

        # --- Innerer Ring (pulsiert mit Tension) ---
        ip = 1.0 + math.sin(self._pulse_phase * 1.5) * 0.02 * (1 + self._s_tension * 2)
        ri = int(95 * ip)
        ic = _scale_color(color_main, brightness * 0.55)
        c.create_oval(CENTER - ri, CENTER - ri, CENTER + ri, CENTER + ri,
                      fill="", outline=ic, width=2)

        # --- Iris (mit Strahlen-Muster) ---
        ir = int(62 * brightness)
        ix = CENTER + self._pupil_dx
        iy = CENTER + self._pupil_dy

        # Iris Grundkreis
        iris_fill = _scale_color(color_iris, brightness * 0.4)
        iris_edge = _scale_color(color_main, brightness * 0.7)
        c.create_oval(ix - ir, iy - ir, ix + ir, iy + ir,
                      fill=iris_fill, outline=iris_edge, width=2)

        # 16 radiale Strahlen (leichte Rotation fuer lebendigen Look)
        pupil_ratio = 0.2 + self._s_attention * 0.4
        pupil_inner = ir * pupil_ratio + 3
        for j in range(16):
            a = math.radians(j * 22.5 + self._tick * 0.3)
            r_in = pupil_inner
            r_out = ir - 3
            if r_in < r_out:
                rx1 = ix + math.cos(a) * r_in
                ry1 = iy + math.sin(a) * r_in
                rx2 = ix + math.cos(a) * r_out
                ry2 = iy + math.sin(a) * r_out
                ray_c = _scale_color(color_iris, brightness * 0.22)
                c.create_line(rx1, ry1, rx2, ry2, fill=ray_c, width=1)

        # --- Pupille (20-60% der Iris je nach Attention) ---
        pr = int(ir * pupil_ratio)
        if flash:
            pr = int(ir * 0.15)
        c.create_oval(ix - pr, iy - pr, ix + pr, iy + pr,
                      fill="#000000", outline="")

        # Lichtreflex oben rechts (Hauptreflex)
        hr = max(3, pr // 3)
        hx, hy = ix + pr * 0.3, iy - pr * 0.35
        hc = _scale_color("#FFFFFF", brightness * 0.75)
        c.create_oval(hx - hr, hy - hr, hx + hr, hy + hr,
                      fill=hc, outline="")

        # Kleiner Reflex unten links
        h2r = max(2, pr // 5)
        h2x, h2y = ix - pr * 0.2, iy + pr * 0.25
        h2c = _scale_color("#FFFFFF", brightness * 0.3)
        c.create_oval(h2x - h2r, h2y - h2r, h2x + h2r, h2y + h2r,
                      fill=h2c, outline="")

        # --- Augenlider ---
        if self._blink_progress > 0.01:
            self._draw_lids(c, brightness, color_main)
        elif self._s_attention < 0.3:
            self._draw_droopy(c, brightness, color_main)

    # =========================================================================
    # Zeichnen — Augenlider
    # =========================================================================

    def _draw_lids(self, c, brightness, color_main):
        """Blinzel-Lider mit gewoelbter Kante (Halbkreis-Effekt)."""
        h = int(AVATAR_SIZE * 0.5 * self._blink_progress)
        if h < 2:
            return

        curve = min(15, h // 2)

        # Oberes Lid: Polygon mit flacher Oberkante, gewoelbter Unterkante
        pts_top = [0, 0, AVATAR_SIZE, 0]
        for px in range(AVATAR_SIZE, -1, -10):
            dx = (px - CENTER) / CENTER
            cy = h + int(curve * (1.0 - dx * dx))
            pts_top.extend([px, cy])
        c.create_polygon(*pts_top, fill=BG_AVATAR, outline="")

        # Unteres Lid: Polygon mit flacher Unterkante, gewoelbter Oberkante
        base = AVATAR_SIZE - h
        pts_bot = [0, AVATAR_SIZE, AVATAR_SIZE, AVATAR_SIZE]
        for px in range(AVATAR_SIZE, -1, -10):
            dx = (px - CENTER) / CENTER
            cy = base - int(curve * (1.0 - dx * dx))
            pts_bot.extend([px, cy])
        c.create_polygon(*pts_bot, fill=BG_AVATAR, outline="")

        # Leuchtende Kanten (smooth spline)
        edge = _scale_color(color_main, brightness * 0.45)

        top_edge = []
        for px in range(0, AVATAR_SIZE + 1, 8):
            dx = (px - CENTER) / CENTER
            cy = h + int(curve * (1.0 - dx * dx))
            top_edge.extend([px, cy])
        if len(top_edge) >= 4:
            c.create_line(*top_edge, fill=edge, width=2, smooth=True)

        bot_edge = []
        for px in range(0, AVATAR_SIZE + 1, 8):
            dx = (px - CENTER) / CENTER
            cy = base - int(curve * (1.0 - dx * dx))
            bot_edge.extend([px, cy])
        if len(bot_edge) >= 4:
            c.create_line(*bot_edge, fill=edge, width=2, smooth=True)

    def _draw_droopy(self, c, brightness, color_main):
        """Halb-geschlossene Lider bei niedriger Attention."""
        droop = int((0.3 - self._s_attention) / 0.3 * AVATAR_SIZE * 0.22)
        if droop < 3:
            return

        c.create_rectangle(0, 0, AVATAR_SIZE, droop,
                           fill=BG_AVATAR, outline="")
        c.create_rectangle(0, AVATAR_SIZE - droop, AVATAR_SIZE, AVATAR_SIZE,
                           fill=BG_AVATAR, outline="")

        edge = _scale_color(color_main, brightness * 0.3)
        c.create_line(0, droop, AVATAR_SIZE, droop, fill=edge, width=1)
        c.create_line(0, AVATAR_SIZE - droop, AVATAR_SIZE, AVATAR_SIZE - droop,
                      fill=edge, width=1)

    # =========================================================================
    # Zeichnen — Info-Balken
    # =========================================================================

    def _draw_bars(self):
        """T/A/P Balken-Visualisierung unter dem Auge."""
        bc = self._bars_canvas
        bc.delete("all")

        color_main, _, _ = self._get_zone_colors()
        bar_w, bar_h = 130, 8
        x0, y0 = 35, 4

        for i, (label, val) in enumerate([
            ("T", self._s_tension),
            ("A", self._s_attention),
            ("P", self._s_presence),
        ]):
            y = y0 + i * 12
            # Label
            bc.create_text(x0 - 12, y + bar_h // 2, text=label,
                           fill="#777777", font=("Helvetica", 8, "bold"),
                           anchor="e")
            # Hintergrund-Balken
            bc.create_rectangle(x0, y, x0 + bar_w, y + bar_h,
                                fill="#14142a", outline="#2a2a44")
            # Fuell-Balken
            fw = int(bar_w * max(0.0, min(1.0, val)))
            if fw > 0:
                bc.create_rectangle(x0, y, x0 + fw, y + bar_h,
                                    fill=_scale_color(color_main, 0.65),
                                    outline="")
            # Wert rechts
            bc.create_text(x0 + bar_w + 10, y + bar_h // 2,
                           text=f"{val:.2f}", fill="#555555",
                           font=("Helvetica", 7), anchor="w")

    # =========================================================================
    # Animation Loop
    # =========================================================================

    def _update_animation(self):
        """Animation-Tick: Blinzeln, Mikro-Bewegung, Status lesen."""
        if not self._running:
            return

        self._tick += 1

        # Core State lesen (alle 10 Ticks = 500ms)
        if self._tick % 10 == 0:
            self._read_core_state()

        # Blinzel-Logik
        if self._blinking:
            if not self._blink_opening:
                self._blink_progress += 0.25
                if self._blink_progress >= 1.0:
                    self._blink_progress = 1.0
                    self._blink_opening = True
            else:
                self._blink_progress -= 0.25
                if self._blink_progress <= 0.0:
                    self._blink_progress = 0.0
                    self._blinking = False
                    self._blink_opening = False
                    self._next_blink_tick = self._random_blink_tick()
        elif self._tick >= self._next_blink_tick:
            if self._attention > 0.7 and random.random() < 0.5:
                self._next_blink_tick = self._random_blink_tick()
            else:
                self._blinking = True
                self._blink_opening = False

        # Mikro-Bewegung (Pupille wandert langsam)
        if self._tick % 20 == 0:
            self._target_dx = random.uniform(-3, 3)
            self._target_dy = random.uniform(-3, 3)
        self._pupil_dx += (self._target_dx - self._pupil_dx) * 0.1
        self._pupil_dy += (self._target_dy - self._pupil_dy) * 0.1

        # Zeichnen
        self._draw_eye()
        self._draw_bars()

        self._after_id = self._parent.after(ANIM_INTERVAL_MS, self._update_animation)

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_from_status(self, status: dict):
        """Core State aus uebergebenem Status-Dict lesen (aufgerufen von panel_main)."""
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

        # Status-Text aus Detections (falls vorhanden)
        detections = status.get("detections", {})
        faces = detections.get("faces", []) if isinstance(detections, dict) else []
        persons = detections.get("persons", 0) if isinstance(detections, dict) else 0

        if faces:
            known = [f for f in faces if isinstance(f, dict)
                     and f.get("name", "unknown") != "unknown"]
            if known:
                names = ", ".join(f["name"].capitalize() for f in known[:3])
                self._status_text = f"{names} erkannt"
            else:
                self._status_text = f"{len(faces)} Gesicht(er)"
        elif persons:
            self._status_text = f"{persons} Person(en)"
        elif self._presence > 0.3:
            self._status_text = "Suche..."
        else:
            self._status_text = "Idle"

        # Labels aktualisieren
        color_main, _, _ = self._get_zone_colors()
        bright = max(0.5, 0.4 + self._s_presence * 0.6)
        zone_name = {"guardian": "GUARDIAN", "shadow": "SHADOW",
                     "berserker": "BERSERKER"}.get(self._zone, "OFFLINE")
        self._zone_label.config(text=zone_name,
                                fg=_scale_color(color_main, bright))
        self._status_label.config(text=self._status_text,
                                  fg=_scale_color(color_main, bright * 0.6))

        # Berserker-Flash bei Eintritt in Berserker-Zone
        if self._zone == "berserker" and old_zone != "berserker":
            self._flash_until = time.monotonic() + BERSERKER_FLASH_MS / 1000.0

    def _read_core_state(self):
        """Core State aus Status-JSON lesen (Fallback, falls poll nicht aktiv)."""
        status = self._service.read_status()
        self.update_from_status(status)

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
