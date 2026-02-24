#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Avatar — Das Auge (v3 PyGame)
===================================================

HAL-9000 Auge mit PyGame Off-Screen Rendering in Tkinter eingebettet.
Nutzt echtes Alpha-Blending und Anti-Aliasing via pygame.gfxdraw.

Rendering-Pipeline: PyGame Surface -> PIL Image -> ImageTk -> Tkinter Label
Kein separates Fenster — alles inline im Panel.

Blueprint 5.2/5.3 Mapping:
  TENSION   -> Ring-Puls-Geschwindigkeit + Glow-Intensitaet + Pupillengroesse
  DOMINANCE -> Farbe (Guardian=Blau, Neutral=Weiss, Shadow=Rot)
  CPU-TEMP  -> Puls verlangsamt sich bei Hitze, Auge wird "muede"

Hysterese: Farbwechsel erst bei 0.15 dominance-Delta.
300ms Interpolation fuer alle visuellen Aenderungen.

Rein passiv — liest NUR Core State, schreibt NICHTS.
"""

import tkinter as tk
import math
import random
import time
import logging

from core.gui.panel_styles import BG_FRAME, FG_DIM

# PyGame + PIL (graceful fallback)
try:
    import pygame
    import pygame.gfxdraw
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Music Visualizer Konstante (muss mit music_visualizer.py uebereinstimmen)
MAX_VISUAL_AMP = 0.15


# =============================================================================
# Konstanten
# =============================================================================

AVATAR_SIZE = 300
CX = AVATAR_SIZE // 2
CY = AVATAR_SIZE // 2
ANIM_INTERVAL_MS = 33   # ~30 FPS

# Hintergrund
BG_AVATAR = "#0A0A14"
BG_RGB = (10, 10, 20)
GRID_RGB = (17, 17, 32)

# Zone-Farben (RGB)
COL_GUARDIAN = (0, 170, 255)
COL_NEUTRAL = (255, 255, 255)
COL_SHADOW = (255, 34, 0)
COL_BERSERKER = (255, 34, 0)

# Hysterese
DOM_HYSTERESIS = 0.15


# =============================================================================
# Farb-Hilfsfunktionen
# =============================================================================

def _lerp(c1, c2, t):
    """Lineare Interpolation zwischen zwei RGB-Tupeln."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _scale(color, factor):
    """Farbe um Helligkeitsfaktor skalieren."""
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def _dom_color(d):
    """Dominance -> Hauptfarbe."""
    if d >= 0:
        return _lerp(COL_NEUTRAL, COL_GUARDIAN, d)
    return _lerp(COL_NEUTRAL, COL_SHADOW, -d)


def _dom_iris(d):
    """Dominance -> Iris-Farbe (heller)."""
    if d >= 0:
        return _lerp((220, 220, 220), (68, 204, 255), d)
    return _lerp((220, 220, 220), (255, 68, 51), -d)


def _rgb_to_hex(rgb):
    """RGB-Tupel in Hex-String."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _hex_scale(color_hex, factor):
    """Hex-Farbe skalieren."""
    factor = max(0.0, factor)
    r = min(255, int(int(color_hex[1:3], 16) * factor))
    g = min(255, int(int(color_hex[3:5], 16) * factor))
    b = min(255, int(int(color_hex[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# Glow Renderer
# =============================================================================

def _render_glow(radius, color, intensity):
    """Weichen Glow als Alpha-Surface rendern.

    Quadratischer Alpha-Falloff von Zentrum nach aussen.
    Das ist der Hauptvorteil gegenueber Tkinter Canvas.
    """
    size = radius * 2 + 4
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    step = max(3, radius // 25)
    for r in range(radius, 0, -step):
        t = r / radius
        alpha = int(intensity * 255 * (1.0 - t) * (1.0 - t) * 0.15)
        alpha = min(255, max(0, alpha))
        if alpha > 0:
            pygame.gfxdraw.filled_circle(surf, cx, cy, r, (*color, alpha))
    return surf


# =============================================================================
# AvatarModule
# =============================================================================

class AvatarModule:
    """
    M.O.L.O.C.H. Avatar-Auge v3 — PyGame Off-Screen in Tkinter.

    Rendering: PyGame Surface -> PIL Image -> ImageTk -> Tkinter Label.
    Gleiches Interface wie v2 (start/stop/update_from_status).
    """

    def __init__(self, parent_frame, service_proxy):
        self._parent = parent_frame
        self._service = service_proxy
        self._running = False
        self._after_id = None
        self._logger = logging.getLogger("AvatarPyGame")

        # PyGame + PIL pruefen
        if not _PYGAME_OK or not _PIL_OK:
            self._logger.error("PyGame oder PIL nicht verfuegbar")
            tk.Label(parent_frame, text="Avatar: PyGame/PIL fehlt",
                     bg=BG_FRAME, fg=FG_DIM).pack(pady=20)
            self._enabled = False
            return

        self._enabled = True

        # Off-Screen Surface (kein Display noetig)
        self._surface = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE))

        # --- Core State (Zielwerte) ---
        self._tension = 0.0
        self._dominance = 0.5
        self._cpu_temp = 0.0
        self._cpu_temp_celsius = 0.0
        self._zone = "guardian"
        self._status_text = "Idle"

        # --- Smooth Interpolation (~300ms Settling) ---
        self._s_tension = 0.0
        self._s_dominance = 0.5
        self._s_cpu = 0.0

        # --- Visuelle Dominance mit Hysterese ---
        self._visual_dom = 0.5
        self._dom_at_last_change = 0.5

        # --- Animation State ---
        self._tick = 0
        self._pulse_phase = 0.0
        self._blink_progress = 0.0
        self._blinking = False
        self._blink_opening = False
        self._next_blink_tick = random.randint(100, 250)
        self._flash_until = 0.0
        self._pupil_dx = 0.0
        self._pupil_dy = 0.0
        self._target_dx = 0.0
        self._target_dy = 0.0
        self._last_time = time.monotonic()

        # --- Voice State (TTS/PTT) ---
        self._tts_active = False
        self._ptt_active = False
        self._s_voice_speak = 0.0   # Smooth 0-1 fuer Sprech-Modus
        self._s_voice_listen = 0.0  # Smooth 0-1 fuer Zuhoer-Modus
        self._speech_phase = 0.0    # Eigene Phase fuer Sprech-Pulsation

        # --- Music Reactive State (Daten kommen via update_from_status) ---
        self._s_music_bass = 0.0
        self._s_music_mid = 0.0
        self._s_music_high = 0.0
        self._s_music_rms = 0.0
        self._t_music_bass = 0.0   # Zielwerte aus Status-JSON
        self._t_music_mid = 0.0
        self._t_music_high = 0.0
        self._t_music_rms = 0.0
        self._music_active = False

        # --- Glow Cache ---
        self._glow_surface = None
        self._glow_key = None

        # --- Tkinter: Label fuer Bild-Anzeige ---
        self._photo = None
        self._label = tk.Label(
            parent_frame,
            bg=BG_AVATAR,
            width=AVATAR_SIZE,
            height=AVATAR_SIZE,
        )
        self._label.pack(padx=5, pady=5)

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

        # --- Track-Anzeige (Spotify, unter dem Auge) ---
        self._track_label = tk.Label(
            parent_frame, text="", bg=BG_FRAME, fg="#1DB954",
            font=("Helvetica", 9), anchor="w",
        )
        # Wird erst gepacked wenn Musik laeuft
        self._track_text = ""
        self._track_scroll_offset = 0
        self._track_scroll_pause = 0
        self._last_track_name = ""
        self._track_visible = False

    # =========================================================================
    # Rendering — Hauptfunktion
    # =========================================================================

    def _render(self):
        """Kompletten Frame auf self._surface rendern.

        Blueprint 5.4: Core State + Music Reactive + Industrial Aesthetic.
        Musik ADDIERT sich zum Core-Puls, ueberschreibt nie.
        """
        s = self._surface
        s.fill(BG_RGB)

        now = time.monotonic()
        flash = now < self._flash_until

        # === Music TTS Dim (Musik 40% leiser wenn Moloch spricht) ===
        music_dim = 1.0 - self._s_voice_speak * 0.4

        # === Farben bestimmen ===
        if flash:
            main_c = (255, 68, 68)
            iris_c = (255, 170, 170)
            brightness = 1.0
        elif self._zone == "berserker":
            main_c = COL_BERSERKER
            iris_c = (255, 68, 51)
            brightness = 0.6 + self._s_tension * 0.4
        else:
            main_c = _dom_color(self._visual_dom)
            iris_c = _dom_iris(self._visual_dom)
            brightness = 0.5 + self._s_tension * 0.5

        pulse = 1.0 + math.sin(self._pulse_phase) * 0.025
        # Voice-Puls
        pulse += self._s_voice_speak * 0.035 * math.sin(self._speech_phase)
        if self._zone == "berserker":
            pulse = 1.0 + math.sin(self._pulse_phase) * 0.07

        # === Music: Bass → Outer Ring Modulation ===
        bass_mod = self._s_music_bass * music_dim
        if self._zone == "guardian":
            # Smooth Sinus
            bass_offset = bass_mod * 0.08 / MAX_VISUAL_AMP * math.sin(
                self._pulse_phase * 2.0)
        elif self._zone == "berserker":
            # Snap: schneller Anstieg, kein negativer Anteil
            bass_offset = bass_mod * 0.08 / MAX_VISUAL_AMP * max(
                0.0, math.sin(self._pulse_phase * 3.0))
        else:
            # Shadow: leichte Phase-Asymmetrie
            bass_offset = bass_mod * 0.08 / MAX_VISUAL_AMP * (
                math.sin(self._pulse_phase * 2.0) - 0.3 * math.sin(
                    self._pulse_phase * 4.1))

        # === Grid (Tron-Style) ===
        for x in range(0, AVATAR_SIZE, 25):
            pygame.draw.line(s, GRID_RGB, (x, 0), (x, AVATAR_SIZE))
        for y in range(0, AVATAR_SIZE, 25):
            pygame.draw.line(s, GRID_RGB, (0, y), (AVATAR_SIZE, y))

        # === Scan-Line (Industrial CRT Effekt, kaum sichtbar) ===
        scan_y = int((self._tick * 2) % AVATAR_SIZE)
        scan_surf = pygame.Surface((AVATAR_SIZE, 2), pygame.SRCALPHA)
        scan_surf.fill((*main_c, 12))
        s.blit(scan_surf, (0, scan_y))

        # === Glow (Alpha-Blending) ===
        glow_str = brightness * (0.5 + self._s_tension * 0.5)
        glow_str *= 1.0 + self._s_voice_speak * 0.7
        glow_str *= 1.0 + self._s_voice_listen * 0.3
        # Music: RMS Volume moduliert Glow 20-100%
        if self._s_music_rms > 0.001:
            glow_music = 0.2 + (self._s_music_rms / MAX_VISUAL_AMP) * 0.8
            glow_str *= glow_music * music_dim
        glow_r = int(130 * pulse)
        glow_key = (main_c, int(glow_str * 4), glow_r // 8)
        if glow_key != self._glow_key:
            self._glow_surface = _render_glow(glow_r, main_c, glow_str)
            self._glow_key = glow_key
        if self._glow_surface:
            gr = self._glow_surface.get_rect(center=(CX, CY))
            s.blit(self._glow_surface, gr)

        # === Aeusserer Halo-Ring (dezent, hauchduenn) ===
        halo_r = int(100 * (pulse + bass_offset * 0.3))
        halo_c = _scale(main_c, brightness * 0.15)
        if halo_r > 5:
            pygame.gfxdraw.aacircle(s, CX, CY, halo_r, halo_c)

        # === Aeusserer Ring (Anti-Aliased, 3px dick + Bass-Puls) ===
        ro = int(90 * (pulse + bass_offset))
        ro += int(self._s_voice_speak * 2.5 * math.sin(self._speech_phase * 3.1))
        ro = max(10, ro)
        rc = _scale(main_c, brightness * 0.8)
        rc_dim = _scale(main_c, brightness * 0.4)
        for dr in (-1, 0, 1):
            r = ro + dr
            if r > 2:
                c = rc if dr == 0 else rc_dim
                pygame.gfxdraw.aacircle(s, CX, CY, r, c)

        # === 32 Tick-Marks (Industrial, mit High-Energy Flicker) ===
        for i in range(32):
            angle = math.radians(i * (360.0 / 32.0) - 90)
            if i % 8 == 0:
                tl, tw, bri_f = 12, 2, 0.7
            elif i % 4 == 0:
                tl, tw, bri_f = 7, 1, 0.5
            else:
                tl, tw, bri_f = 4, 1, 0.3

            # High-Energy Micro-Flicker ±3%
            flicker = 0.0
            if self._s_music_high > 0.005:
                tick_phase = self._pulse_phase * 4.0 + i * 0.5
                flicker = (self._s_music_high / MAX_VISUAL_AMP) * 0.03 * math.sin(
                    tick_phase) * music_dim
            tl_mod = tl * (1.0 + flicker)

            x1 = CX + math.cos(angle) * (ro - tl_mod)
            y1 = CY + math.sin(angle) * (ro - tl_mod)
            x2 = CX + math.cos(angle) * (ro + 4)
            y2 = CY + math.sin(angle) * (ro + 4)
            tc = _scale(main_c, brightness * bri_f)
            pygame.draw.line(s, tc, (int(x1), int(y1)), (int(x2), int(y2)), tw)

        # === Mittlerer Ring (semi-transparent, pulsiert mit Mid-Energy) ===
        mid_mod = self._s_music_mid * music_dim
        mid_ring_r = int(81 * pulse)
        mid_ring_r += int((mid_mod / MAX_VISUAL_AMP) * 2.0 * math.sin(
            self._pulse_phase * 1.3))
        mid_ring_r = max(5, mid_ring_r)
        mid_ring_c = _scale(main_c, brightness * 0.3)
        pygame.gfxdraw.aacircle(s, CX, CY, mid_ring_r, mid_ring_c)

        # === Innerer Ring (pulsiert staerker mit Tension) ===
        inner_amp = 0.025 * (1 + self._s_tension * 2)
        inner_amp *= 1.0 - self._s_voice_listen * 0.6
        ip = 1.0 + math.sin(self._pulse_phase * 1.5) * inner_amp
        ri = int(72 * ip)
        ri += int(self._s_voice_speak * 2.0 * math.sin(self._speech_phase * 2.7))
        ri = max(5, ri)
        ic = _scale(main_c, brightness * 0.55)
        pygame.gfxdraw.aacircle(s, CX, CY, ri, ic)

        # === Iris mit Micro-Ringen (Sci-Fi Textur) + Mid-Energy ===
        mid_iris = (mid_mod / MAX_VISUAL_AMP) * 0.05 * math.sin(
            self._pulse_phase * 1.7) if MAX_VISUAL_AMP > 0 else 0
        ir = int(50 * (0.8 + brightness * 0.2 + mid_iris))
        ir = max(8, ir)
        ix = int(CX + self._pupil_dx)
        iy = int(CY + self._pupil_dy)

        # Iris Base Fill (dunkel)
        pygame.gfxdraw.filled_circle(s, ix, iy, ir,
                                     _scale(iris_c, brightness * 0.25))

        # Konzentrische Micro-Ringe (6 Ringe, Industrial-Textur)
        for ring_i in range(6):
            ring_r = ir - ring_i * 6
            if ring_r <= 5:
                break
            ring_alpha = brightness * (0.5 - ring_i * 0.07)
            ring_c = _scale(iris_c, max(0.05, ring_alpha))
            pygame.gfxdraw.aacircle(s, ix, iy, ring_r, ring_c)

        # Iris Gradient Overlay (4 Stufen, semi-transparent)
        for r in range(ir, max(ir - 4, 0), -1):
            t = (ir - r) / 4.0
            c = _lerp(_scale(iris_c, brightness * 0.35),
                      _scale(main_c, brightness * 0.55), t)
            pygame.gfxdraw.filled_circle(s, ix, iy, r, c)

        # Iris-Rand (Anti-Aliased)
        pygame.gfxdraw.aacircle(s, ix, iy, ir,
                                _scale(main_c, brightness * 0.7))

        # === 16 radiale Strahlen (langsame Rotation) ===
        pupil_ratio = 0.25 + self._s_tension * 0.35
        pupil_inner = ir * pupil_ratio + 3
        ray_c = _scale(iris_c, brightness * 0.25)
        for j in range(16):
            a = math.radians(j * 22.5 + self._tick * 0.12)
            r_in = pupil_inner
            r_out = ir - 3
            if r_in < r_out:
                rx1 = ix + math.cos(a) * r_in
                ry1 = iy + math.sin(a) * r_in
                rx2 = ix + math.cos(a) * r_out
                ry2 = iy + math.sin(a) * r_out
                pygame.draw.aaline(s, ray_c, (rx1, ry1), (rx2, ry2))

        # === Pupille (Tension + Voice) ===
        voice_pupil = pupil_ratio
        voice_pupil += self._s_voice_speak * 0.06 * math.sin(
            self._speech_phase * 1.5)
        voice_pupil *= 1.0 - self._s_voice_listen * 0.35
        pr = int(ir * voice_pupil)
        if flash:
            pr = int(ir * 0.12)
        pr = max(2, pr)
        pygame.gfxdraw.filled_circle(s, ix, iy, pr, (0, 0, 0))
        pygame.gfxdraw.aacircle(s, ix, iy, pr, (0, 0, 0))

        # Hauptreflex (oben rechts)
        hr = max(3, pr // 3)
        hx = int(ix + pr * 0.3)
        hy = int(iy - pr * 0.35)
        hc = _scale((255, 255, 255), brightness * 0.8)
        pygame.gfxdraw.filled_circle(s, hx, hy, hr, hc)
        pygame.gfxdraw.aacircle(s, hx, hy, hr, hc)

        # Kleiner Reflex (unten links)
        h2r = max(2, pr // 5)
        h2x = int(ix - pr * 0.2)
        h2y = int(iy + pr * 0.25)
        h2c = _scale((255, 255, 255), brightness * 0.35)
        pygame.gfxdraw.filled_circle(s, h2x, h2y, h2r, h2c)

        # === Augenlider ===
        if self._blink_progress > 0.01:
            self._render_lids(s, brightness, main_c)
        elif self._s_cpu > 0.85:
            self._render_droopy(s, brightness, main_c)

    # =========================================================================
    # Lider
    # =========================================================================

    def _render_lids(self, s, brightness, color):
        """Blinzel-Lider mit geschwungener Leucht-Kante."""
        h = int(AVATAR_SIZE * 0.5 * self._blink_progress)
        if h < 2:
            return
        pygame.draw.rect(s, BG_RGB, (0, 0, AVATAR_SIZE, h))
        pygame.draw.rect(s, BG_RGB, (0, AVATAR_SIZE - h, AVATAR_SIZE, h))

        edge_c = _scale(color, brightness * 0.5)
        curve = min(10, h // 3)
        top_pts = []
        bot_pts = []
        for px in range(0, AVATAR_SIZE + 1, 4):
            dx = (px - CX) / CX
            c_off = int(curve * (1.0 - dx * dx))
            top_pts.append((px, h + c_off))
            bot_pts.append((px, (AVATAR_SIZE - h) - c_off))
        if len(top_pts) > 1:
            pygame.draw.lines(s, edge_c, False, top_pts, 2)
        if len(bot_pts) > 1:
            pygame.draw.lines(s, edge_c, False, bot_pts, 2)

    def _render_droopy(self, s, brightness, color):
        """Halb-geschlossene Lider bei hoher CPU-Temperatur."""
        droop_factor = (self._s_cpu - 0.85) / 0.15
        droop = int(max(0, droop_factor) * AVATAR_SIZE * 0.22)
        if droop < 3:
            return
        pygame.draw.rect(s, BG_RGB, (0, 0, AVATAR_SIZE, droop))
        pygame.draw.rect(s, BG_RGB, (0, AVATAR_SIZE - droop, AVATAR_SIZE, droop))
        edge = _scale(color, brightness * 0.3)
        pygame.draw.line(s, edge, (0, droop), (AVATAR_SIZE, droop), 1)
        pygame.draw.line(s, edge,
                         (0, AVATAR_SIZE - droop), (AVATAR_SIZE, AVATAR_SIZE - droop), 1)

    # =========================================================================
    # Surface -> Tkinter Konvertierung
    # =========================================================================

    def _blit_to_tkinter(self):
        """PyGame Surface -> PIL Image -> ImageTk -> Label."""
        data = pygame.image.tostring(self._surface, 'RGB')
        img = Image.frombytes('RGB', (AVATAR_SIZE, AVATAR_SIZE), data)
        self._photo = ImageTk.PhotoImage(img)
        self._label.config(image=self._photo)

    # =========================================================================
    # Animation Loop
    # =========================================================================

    def _update_animation(self):
        """Ein Animation-Frame: Update State -> Render -> Display."""
        if not self._running:
            return

        now = time.monotonic()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now
        self._tick += 1

        # Core State lesen (alle 15 Ticks ~ 500ms bei 30 FPS)
        if self._tick % 15 == 0:
            self._read_core_state()

        # Smooth Interpolation (rate=dt*6 ~ 300ms Settling)
        rate = min(1.0, dt * 6.0)
        self._s_tension += (self._tension - self._s_tension) * rate
        self._s_dominance += (self._dominance - self._s_dominance) * rate
        self._s_cpu += (self._cpu_temp - self._s_cpu) * rate

        # Music Data interpolieren (Zielwerte kommen via update_from_status)
        if self._music_active:
            self._s_music_bass += (self._t_music_bass - self._s_music_bass) * rate
            self._s_music_mid += (self._t_music_mid - self._s_music_mid) * rate
            self._s_music_high += (self._t_music_high - self._s_music_high) * rate
            self._s_music_rms += (self._t_music_rms - self._s_music_rms) * rate
        else:
            # Smooth Decay wenn keine Musik
            self._s_music_bass *= 0.95
            self._s_music_mid *= 0.95
            self._s_music_high *= 0.95
            self._s_music_rms *= 0.95

        # Voice State Interpolation (smooth fuer weiche Uebergaenge)
        speak_target = 1.0 if self._tts_active else 0.0
        listen_target = 1.0 if (self._ptt_active and not self._tts_active) else 0.0
        self._s_voice_speak += (speak_target - self._s_voice_speak) * rate
        self._s_voice_listen += (listen_target - self._s_voice_listen) * rate

        # Visuelle Dominance mit Hysterese
        if abs(self._dominance - self._dom_at_last_change) > DOM_HYSTERESIS:
            self._dom_at_last_change = self._dominance
        self._visual_dom += (self._dom_at_last_change - self._visual_dom) * rate

        # Puls-Phase: Geschwindigkeit von Tension + Voice, gedaempft bei CPU-Hitze
        speed = 1.0 + self._s_tension * 3.0
        speed *= 1.0 + self._s_voice_speak * 2.5   # Schneller beim Sprechen
        if self._s_voice_listen > 0.3:
            speed *= 0.6                            # Ruhiger beim Zuhoeren
        if self._s_cpu > 0.7:
            speed *= 0.85
        if self._s_cpu > 0.9:
            speed *= 0.6
        self._pulse_phase += speed * dt * math.tau / 3.0

        # Speech-Phase (eigene Frequenz fuer Sprech-Pulsation, ~3.5 Hz)
        self._speech_phase += self._s_voice_speak * dt * math.tau * 3.5

        # Blinzel-Logik (pausiert beim Sprechen — Auge "spricht dich an")
        blink_speed = 4.0 * dt
        speaking = self._s_voice_speak > 0.5

        if speaking and self._blinking:
            # Beim Sprechen: Auge schnell oeffnen
            self._blink_progress -= blink_speed * 3
            if self._blink_progress <= 0.0:
                self._blink_progress = 0.0
                self._blinking = False
                self._blink_opening = False
        elif self._blinking:
            if not self._blink_opening:
                self._blink_progress += blink_speed
                if self._blink_progress >= 1.0:
                    self._blink_progress = 1.0
                    self._blink_opening = True
            else:
                self._blink_progress -= blink_speed
                if self._blink_progress <= 0.0:
                    self._blink_progress = 0.0
                    self._blinking = False
                    self._blink_opening = False
                    self._next_blink_tick = self._tick + random.randint(100, 250)
        elif not speaking and self._tick >= self._next_blink_tick:
            if self._s_tension > 0.7 and random.random() < 0.5:
                self._next_blink_tick = self._tick + random.randint(100, 250)
            else:
                self._blinking = True
                self._blink_opening = False

        # Mikro-Bewegung Pupille
        if self._tick % 60 == 0:
            self._target_dx = random.uniform(-4, 4)
            self._target_dy = random.uniform(-4, 4)
        self._pupil_dx += (self._target_dx - self._pupil_dx) * 0.03
        self._pupil_dy += (self._target_dy - self._pupil_dy) * 0.03

        # Track-Text Scrolling (alle 2 Ticks bei 30 FPS)
        if self._track_visible and self._track_text and self._tick % 2 == 0:
            max_vis = 35
            if len(self._track_text) > max_vis:
                if self._track_scroll_pause > 0:
                    self._track_scroll_pause -= 1
                    self._track_label.config(text=self._track_text[:max_vis])
                else:
                    self._track_scroll_offset += 1
                    if self._track_scroll_offset > len(self._track_text) - max_vis + 10:
                        self._track_scroll_offset = 0
                        self._track_scroll_pause = 45  # ~3s bei halber Rate
                    start = max(0, min(self._track_scroll_offset,
                                       len(self._track_text) - max_vis))
                    self._track_label.config(
                        text=self._track_text[start:start + max_vis])
            else:
                self._track_label.config(text=self._track_text)

        # Rendern + Display
        self._render()
        self._blit_to_tkinter()

        self._after_id = self._parent.after(ANIM_INTERVAL_MS, self._update_animation)

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_from_status(self, status: dict):
        """Core State v2 aus Panel-Status aktualisieren."""
        if not self._enabled or not status:
            return

        core = status.get("core", {})
        if not core:
            return

        # Voice State lesen (TTS/PTT aus voice_pipeline)
        voice = status.get("voice", {})
        self._tts_active = bool(voice.get("speaking", False))
        self._ptt_active = bool(voice.get("recording", False))

        old_zone = self._zone

        self._tension = float(core.get("tension", 0.0))
        self._dominance = float(core.get("dominance", 0.5))
        self._cpu_temp = float(core.get("cpu_temp_norm", 0.0))
        self._cpu_temp_celsius = float(core.get("cpu_temp", 0.0))
        self._zone = core.get("zone", "guardian")

        # Status-Text aus Detections
        det = status.get("detections", {})
        if isinstance(det, dict):
            faces = det.get("faces", [])
            persons = det.get("persons", 0)
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

        # Info-Labels aktualisieren
        main_hex = _rgb_to_hex(_dom_color(self._visual_dom))
        bright = max(0.5, 0.5 + self._s_tension * 0.5)

        zone_name = {"guardian": "GUARDIAN", "shadow": "SHADOW",
                     "berserker": "BERSERKER"}.get(self._zone, "OFFLINE")
        self._zone_label.config(text=zone_name,
                                fg=_hex_scale(main_hex, bright))

        self._state_label.config(
            text=f"T:{self._tension:.2f} | D:{self._dominance:+.2f}",
            fg=_hex_scale(main_hex, bright * 0.7),
        )

        cpu_str = f"CPU: {self._cpu_temp_celsius:.0f}\u00b0C" if self._cpu_temp_celsius > 0 else ""
        parts = []
        if cpu_str:
            parts.append(cpu_str)
        # Voice-Modus anzeigen
        if self._tts_active:
            parts.append("Spricht...")
        elif self._ptt_active:
            parts.append("Hoert zu...")
        else:
            parts.append(self._status_text)
        self._status_label.config(
            text=" | ".join(parts),
            fg=_hex_scale(main_hex, bright * 0.6),
        )

        # Berserker Flash bei Eintritt
        if self._zone == "berserker" and old_zone != "berserker":
            self._flash_until = time.monotonic() + 0.3

        # Music-Daten aus Status-JSON lesen
        music = status.get("music", {})
        self._music_active = bool(music.get("active", False))
        if self._music_active:
            self._t_music_bass = float(music.get("bass", 0.0))
            self._t_music_mid = float(music.get("mid", 0.0))
            self._t_music_high = float(music.get("high", 0.0))
            self._t_music_rms = float(music.get("rms", 0.0))
        else:
            self._t_music_bass = 0.0
            self._t_music_mid = 0.0
            self._t_music_high = 0.0
            self._t_music_rms = 0.0

        # Track-Anzeige aktualisieren
        spotify = status.get("spotify", {})
        track_info = spotify.get("current_track")
        if track_info and track_info.get("is_playing"):
            artist = track_info.get("artist", "")
            track_name = track_info.get("track", "")
            display_text = f"\u266b {artist} \u2014 {track_name}"

            if display_text != self._last_track_name:
                self._last_track_name = display_text
                self._track_text = display_text
                self._track_scroll_offset = 0
                self._track_scroll_pause = 45

            # Zone-Farbe
            track_colors = {
                "guardian": "#00aaff",
                "shadow": "#ff2200",
                "berserker": "#ff2200",
            }
            self._track_label.config(
                fg=track_colors.get(self._zone, "#1DB954"))

            if not self._track_visible:
                self._track_label.pack(fill="x", padx=5, pady=(0, 2))
                self._track_visible = True
        else:
            if self._track_visible:
                self._track_label.pack_forget()
                self._track_visible = False
                self._last_track_name = ""

    def _read_core_state(self):
        """Core State via ServiceProxy lesen (Fallback)."""
        status = self._service.read_status()
        self.update_from_status(status)

    # =========================================================================
    # Start / Stop
    # =========================================================================

    def start(self):
        """Animation starten."""
        if not self._enabled or self._running:
            return
        self._running = True
        self._last_time = time.monotonic()
        self._update_animation()
        self._logger.info("Avatar PyGame gestartet (off-screen -> Tkinter)")

    def stop(self):
        """Animation stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None
