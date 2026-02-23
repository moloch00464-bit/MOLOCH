#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Avatar — Das Auge (v2)
============================================

HAL-9000 inspiriertes mechanisches Auge als Core State Visualisierung.
Tkinter Canvas Widget, 300x300 Pixel. Dark Wave Aesthetik.

Blueprint 5.2/5.3 Mapping:
  TENSION   -> Ring-Puls-Geschwindigkeit + Glow-Intensitaet
  DOMINANCE -> Farbe (Guardian=Blau, Neutral=Weiss, Shadow=Rot)
  CPU-TEMP  -> Puls verlangsamt sich bei Hitze

Hysterese: Farbwechsel erst bei 0.15 dominance-Delta.
300ms kubische Interpolation fuer alle visuellen Aenderungen.

Rein passiv — liest NUR Core State, schreibt NICHTS.
"""

import tkinter as tk
import math
import random
import time

from core.gui.panel_styles import BG_FRAME, FG_DIM


# =============================================================================
# Farb-Konstanten (Dominance-Achse)
# =============================================================================
COLOR_GUARDIAN = "#00AAFF"        # Leuchtendes Blau (+1.0)
COLOR_NEUTRAL = "#FFFFFF"         # Weiss (0.0)
COLOR_SHADOW = "#FF2200"          # Intensives Rot (-1.0)

COLOR_GUARDIAN_DIM = "#005588"
COLOR_NEUTRAL_DIM = "#888888"
COLOR_SHADOW_DIM = "#881100"

COLOR_GUARDIAN_IRIS = "#44CCFF"
COLOR_NEUTRAL_IRIS = "#DDDDDD"
COLOR_SHADOW_IRIS = "#FF4433"

# Berserker Override
COLOR_BERSERKER = "#FF2200"
COLOR_BERSERKER_DIM = "#881100"
COLOR_BERSERKER_IRIS = "#FF4433"

# Hintergrund
BG_AVATAR = "#0A0A14"
GRID_COLOR = "#111120"

# Animation
ANIM_INTERVAL_MS = 50
BERSERKER_FLASH_MS = 200

# Canvas-Groesse
AVATAR_SIZE = 300
CENTER = AVATAR_SIZE // 2

# Hysterese fuer visuelle Dominance
VISUAL_DOMINANCE_HYSTERESIS = 0.15


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


def _dominance_to_colors(dominance: float):
    """Dominance-Wert (-1 bis +1) in Farben umrechnen.

    -1.0 = Shadow (Rot), 0.0 = Neutral (Weiss), +1.0 = Guardian (Blau).
    Smooth Interpolation ueber den gesamten Bereich.
    """
    if dominance >= 0:
        # Neutral -> Guardian (0 bis +1)
        t = dominance
        main = _lerp_color(COLOR_NEUTRAL, COLOR_GUARDIAN, t)
        dim = _lerp_color(COLOR_NEUTRAL_DIM, COLOR_GUARDIAN_DIM, t)
        iris = _lerp_color(COLOR_NEUTRAL_IRIS, COLOR_GUARDIAN_IRIS, t)
    else:
        # Neutral -> Shadow (0 bis -1)
        t = -dominance
        main = _lerp_color(COLOR_NEUTRAL, COLOR_SHADOW, t)
        dim = _lerp_color(COLOR_NEUTRAL_DIM, COLOR_SHADOW_DIM, t)
        iris = _lerp_color(COLOR_NEUTRAL_IRIS, COLOR_SHADOW_IRIS, t)
    return main, dim, iris


class AvatarModule:
    """
    M.O.L.O.C.H. Avatar-Auge v2 — Core State Visualisierung.

    Blueprint 5.2/5.3: 2 Achsen (tension + dominance) + CPU-Temp.
    Wird in einen parent_frame eingebettet.
    Liest Core State via service_proxy.read_status()["core"].
    """

    def __init__(self, parent_frame, service_proxy):
        self._parent = parent_frame
        self._service = service_proxy
        self._running = False
        self._after_id = None

        # --- Core State (Zielwerte) ---
        self._tension = 0.0
        self._dominance = 0.5        # -1 bis +1
        self._cpu_temp = 0.0         # normalisiert 0-1
        self._cpu_temp_celsius = 0.0
        self._zone = "guardian"
        self._status_text = "Idle"

        # --- Smooth Animation (interpoliert zum Ziel, ~300ms Settling) ---
        self._s_tension = 0.0
        self._s_dominance = 0.5

        # --- Visuelle Dominance mit Hysterese ---
        self._visual_dominance = 0.5
        self._dominance_at_last_change = 0.5

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

        self._state_label = tk.Label(
            info, text="T:0.00 | D:+0.50", bg=BG_FRAME, fg=FG_DIM,
            font=("Helvetica", 9),
        )
        self._state_label.pack(pady=(1, 0))

        self._status_label = tk.Label(
            info, text="Idle", bg=BG_FRAME, fg=FG_DIM,
            font=("Helvetica", 9),
        )
        self._status_label.pack(pady=(0, 2))

        # Initialer Draw
        self._draw_eye()

    # =========================================================================
    # Hilfsfunktionen
    # =========================================================================

    def _random_blink_tick(self) -> int:
        """Naechsten Blink-Zeitpunkt (4-9 Sekunden = 80-180 Ticks bei 20 FPS)."""
        return self._tick + random.randint(80, 180)

    def _get_zone_colors(self):
        """Farben basierend auf visueller Dominance. Berserker ueberschreibt."""
        if self._zone == "berserker":
            return COLOR_BERSERKER, COLOR_BERSERKER_DIM, COLOR_BERSERKER_IRIS
        return _dominance_to_colors(self._visual_dominance)

    # =========================================================================
    # Zeichnen — Hauptfunktion
    # =========================================================================

    def _draw_eye(self):
        """Komplettes Auge zeichnen (1 Frame)."""
        c = self._canvas
        c.delete("all")

        # Smooth Interpolation (~300ms Settling bei 20 FPS, rate=0.25)
        smooth_rate = 0.25
        self._s_tension += (self._tension - self._s_tension) * smooth_rate
        self._s_dominance += (self._dominance - self._s_dominance) * smooth_rate

        # Visuelle Dominance mit Hysterese
        if abs(self._dominance - self._dominance_at_last_change) > VISUAL_DOMINANCE_HYSTERESIS:
            self._dominance_at_last_change = self._dominance
        # Smooth Interpolation der visuellen Dominance zum Ziel
        self._visual_dominance += (self._dominance_at_last_change - self._visual_dominance) * smooth_rate

        # Basis-Helligkeit (immer mindestens 0.5, tension erhoeht)
        brightness = 0.5 + self._s_tension * 0.5
        color_main, color_dim, color_iris = self._get_zone_colors()

        # Berserker-Flash
        now = time.monotonic()
        flash = now < self._flash_until
        if flash:
            brightness = 1.0
            color_main = "#FF4444"
            color_iris = "#FFAAAA"

        # === Puls-Faktor: Geschwindigkeit von tension, verlangsamt bei CPU-Hitze ===
        speed = 1.0 + self._s_tension * 3.0
        # CPU-Temp Verlangsamung
        if self._cpu_temp > 0.7:
            speed *= 0.85   # 15% langsamer
        if self._cpu_temp > 0.9:
            speed *= 0.6    # Auge wird "muede"

        self._pulse_phase += speed * 0.1
        pulse = 1.0 + math.sin(self._pulse_phase) * 0.02
        if self._zone == "berserker":
            pulse = 1.0 + math.sin(self._pulse_phase) * 0.06

        # === Glow-Intensitaet von tension (staerker bei hoher Tension) ===
        glow_str = brightness * (0.4 + self._s_tension * 0.6)

        # --- Hintergrund-Grid (Tron-Style) ---
        for x in range(0, AVATAR_SIZE, 25):
            c.create_line(x, 0, x, AVATAR_SIZE, fill=GRID_COLOR)
        for y in range(0, AVATAR_SIZE, 25):
            c.create_line(0, y, AVATAR_SIZE, y, fill=GRID_COLOR)

        # --- Glow (mehrere Schichten) ---
        layers = 7 if self._zone == "berserker" else 5
        for i in range(layers, 0, -1):
            r = int(130 * pulse) + i * 16
            gc = _scale_color(color_main, glow_str * (0.04 + i * 0.025))
            c.create_oval(CENTER - r, CENTER - r, CENTER + r, CENTER + r,
                          fill=gc, outline="")

        # --- Aeusserer Ring (Pulsiert mit Tension) ---
        ro = int(120 * pulse)
        ring_c = _scale_color(color_main, brightness * 0.8)
        c.create_oval(CENTER - ro, CENTER - ro, CENTER + ro, CENTER + ro,
                      fill="", outline=ring_c, width=4)

        # 12 Tick-Marks
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            tl = 12 if i % 3 == 0 else 6
            tw = 2 if i % 3 == 0 else 1
            x1 = CENTER + math.cos(angle) * (ro - tl)
            y1 = CENTER + math.sin(angle) * (ro - tl)
            x2 = CENTER + math.cos(angle) * (ro + 5)
            y2 = CENTER + math.sin(angle) * (ro + 5)
            tc = _scale_color(color_main, brightness * 0.55)
            c.create_line(x1, y1, x2, y2, fill=tc, width=tw)

        # --- Innerer Ring (pulsiert staerker mit Tension) ---
        ip = 1.0 + math.sin(self._pulse_phase * 1.5) * 0.02 * (1 + self._s_tension * 2)
        ri = int(95 * ip)
        ic = _scale_color(color_main, brightness * 0.55)
        c.create_oval(CENTER - ri, CENTER - ri, CENTER + ri, CENTER + ri,
                      fill="", outline=ic, width=2)

        # --- Iris (mit Strahlen-Muster) ---
        ir = int(62 * brightness)
        ix = CENTER + self._pupil_dx
        iy = CENTER + self._pupil_dy

        iris_fill = _scale_color(color_iris, brightness * 0.4)
        iris_edge = _scale_color(color_main, brightness * 0.7)
        c.create_oval(ix - ir, iy - ir, ix + ir, iy + ir,
                      fill=iris_fill, outline=iris_edge, width=2)

        # 16 radiale Strahlen
        pupil_ratio = 0.3 + self._s_tension * 0.3
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

        # --- Pupille (30-60% der Iris, Tension = engere Pupille) ---
        pr = int(ir * pupil_ratio)
        if flash:
            pr = int(ir * 0.15)
        c.create_oval(ix - pr, iy - pr, ix + pr, iy + pr,
                      fill="#000000", outline="")

        # Lichtreflex oben rechts
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
        elif self._cpu_temp > 0.85:
            # "Muede" bei hoher CPU-Temp (statt attention-basiert)
            self._draw_droopy(c, brightness, color_main, self._cpu_temp)

    # =========================================================================
    # Zeichnen — Augenlider
    # =========================================================================

    def _draw_lids(self, c, brightness, color_main):
        """Blinzel-Lider mit gewoelbter Kante."""
        h = int(AVATAR_SIZE * 0.5 * self._blink_progress)
        if h < 2:
            return

        curve = min(15, h // 2)

        pts_top = [0, 0, AVATAR_SIZE, 0]
        for px in range(AVATAR_SIZE, -1, -10):
            dx = (px - CENTER) / CENTER
            cy = h + int(curve * (1.0 - dx * dx))
            pts_top.extend([px, cy])
        c.create_polygon(*pts_top, fill=BG_AVATAR, outline="")

        base = AVATAR_SIZE - h
        pts_bot = [0, AVATAR_SIZE, AVATAR_SIZE, AVATAR_SIZE]
        for px in range(AVATAR_SIZE, -1, -10):
            dx = (px - CENTER) / CENTER
            cy = base - int(curve * (1.0 - dx * dx))
            pts_bot.extend([px, cy])
        c.create_polygon(*pts_bot, fill=BG_AVATAR, outline="")

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

    def _draw_droopy(self, c, brightness, color_main, cpu_temp):
        """Halb-geschlossene Lider bei hoher CPU-Temperatur — Auge wird 'muede'."""
        # Stärke proportional zur CPU-Temp ueber 0.85
        droop_factor = (cpu_temp - 0.85) / 0.15  # 0.85->0, 1.0->1
        droop = int(max(0, droop_factor) * AVATAR_SIZE * 0.22)
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
            # Bei hoher Tension seltener blinzeln
            if self._tension > 0.7 and random.random() < 0.5:
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

        self._after_id = self._parent.after(ANIM_INTERVAL_MS, self._update_animation)

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_from_status(self, status: dict):
        """Core State v2 aus uebergebenem Status-Dict lesen."""
        if not status:
            return

        core = status.get("core", {})
        if not core:
            return

        old_zone = self._zone

        self._tension = float(core.get("tension", 0.0))
        self._dominance = float(core.get("dominance", 0.5))
        self._cpu_temp = float(core.get("cpu_temp_norm", 0.0))
        self._cpu_temp_celsius = float(core.get("cpu_temp", 0.0))
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
        elif self._tension > 0.1:
            self._status_text = "Suche..."
        else:
            self._status_text = "Idle"

        # CPU-Temp in Status
        cpu_str = f"CPU: {self._cpu_temp_celsius:.0f}°C" if self._cpu_temp_celsius > 0 else ""

        # Labels aktualisieren
        color_main, _, _ = self._get_zone_colors()
        bright = max(0.5, 0.5 + self._s_tension * 0.5)

        zone_name = {"guardian": "GUARDIAN", "shadow": "SHADOW",
                     "berserker": "BERSERKER"}.get(self._zone, "OFFLINE")
        self._zone_label.config(text=zone_name,
                                fg=_scale_color(color_main, bright))

        # Zeile 2: T + D Werte
        self._state_label.config(
            text=f"T:{self._tension:.2f} | D:{self._dominance:+.2f}",
            fg=_scale_color(color_main, bright * 0.7),
        )

        # Zeile 3: CPU + Status
        status_parts = []
        if cpu_str:
            status_parts.append(cpu_str)
        status_parts.append(self._status_text)
        self._status_label.config(
            text=" | ".join(status_parts),
            fg=_scale_color(color_main, bright * 0.6),
        )

        # Berserker-Flash bei Eintritt
        if self._zone == "berserker" and old_zone != "berserker":
            self._flash_until = time.monotonic() + BERSERKER_FLASH_MS / 1000.0

    def _read_core_state(self):
        """Core State aus Status-JSON lesen (Fallback)."""
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
