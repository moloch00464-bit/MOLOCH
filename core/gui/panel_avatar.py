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
    Staerker als vorher (0.35 statt 0.15) fuer sichtbaren Effekt.
    """
    size = radius * 2 + 4
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    step = max(3, radius // 25)
    for r in range(radius, 0, -step):
        t = r / radius
        alpha = int(intensity * 255 * (1.0 - t) * (1.0 - t) * 0.35)
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
        self._beat_detected = False

        # --- Dual-Modus: Idle vs Music Eye ---
        self._music_blend = 0.0    # 0=Idle, 1=Music (smooth crossfade)
        self._beat_flash_val = 0.0
        self._shockwave_r = 0.0
        self._shockwave_alpha = 0.0

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
        """Kompletten Frame rendern — Dual-Modus: Idle vs Music Eye.

        _music_blend steuert Crossfade (0=Idle, 1=Music).
        Idle: Ruhiges HAL-9000 Auge, sanfter Puls, Industrial Clean.
        Music: Dramatisch, pumpt mit Bass, Farb-Flash, Shockwaves.
        """
        s = self._surface
        s.fill(BG_RGB)

        now = time.monotonic()
        flash = now < self._flash_until
        tick = self._tick
        pp = self._pulse_phase
        mb = self._music_blend  # 0=Idle, 1=Music

        # === Music Werte (0-1 normalisiert, mit mb skaliert) ===
        music_dim = 1.0 - self._s_voice_speak * 0.4
        bass_f = min(1.0, self._s_music_bass / MAX_VISUAL_AMP) * music_dim
        mid_f = min(1.0, self._s_music_mid / MAX_VISUAL_AMP) * music_dim
        high_f = min(1.0, self._s_music_high / MAX_VISUAL_AMP) * music_dim
        rms_f = min(1.0, self._s_music_rms / MAX_VISUAL_AMP) * music_dim
        beat = self._beat_detected

        # === Farben ===
        if flash:
            main_c = (255, 68, 68)
            iris_c = (255, 170, 170)
            bri = 1.0
        elif self._zone == "berserker":
            main_c = (255, 20, 0)
            iris_c = (255, 68, 51)
            bri = 0.7 + self._s_tension * 0.3
        else:
            main_c = _dom_color(self._visual_dom)
            iris_c = _dom_iris(self._visual_dom)
            bri = 0.55 + self._s_tension * 0.45

        # MUSIC: Helligkeit pulsiert mit RMS (nur bei Music-Blend)
        bri = min(1.0, bri + rms_f * 0.4 * mb)

        # Beat-Flash Decay
        beat_trigger = 1.0 if (beat and mb > 0.3) else 0.0
        self._beat_flash_val = max(beat_trigger, self._beat_flash_val * 0.7)
        bf = self._beat_flash_val

        # Beat-Flash hellt Farbe auf
        if bf > 0.1:
            main_c = _lerp(main_c, (255, 255, 255), bf * 0.5)
            iris_c = _lerp(iris_c, (255, 255, 255), bf * 0.3)

        # Puls (Idle: sanft, Music: staerker)
        idle_pulse = 1.0 + math.sin(pp) * 0.03
        music_pulse = 1.0 + math.sin(pp) * 0.06
        pulse = idle_pulse + (music_pulse - idle_pulse) * mb
        pulse += self._s_voice_speak * 0.04 * math.sin(self._speech_phase)
        if self._zone == "berserker":
            pulse = 1.0 + math.sin(pp) * 0.08

        # === BASS: Ring-Pump (nur bei Music, ±30%) ===
        bass_pump = 0.0
        if mb > 0.05:
            if self._zone == "guardian":
                bass_pump = bass_f * 0.30 * (0.5 + 0.5 * math.sin(pp * 2.0))
            elif self._zone == "berserker":
                bass_pump = bass_f * 0.35 * max(0.0, math.sin(pp * 3.0))
            else:
                bass_pump = bass_f * 0.30 * (
                    0.5 + 0.5 * math.sin(pp * 2.0)
                    - 0.15 * math.sin(pp * 4.1))
            bass_pump *= mb  # Skaliert mit Crossfade
            bass_pump += bf * 0.15

        # === Grid (Idle: dunkel, Music: pulsiert) ===
        grid_bri = 20 + int(rms_f * 15 * mb)
        grid_c = (grid_bri, grid_bri, grid_bri + 15)
        for x in range(0, AVATAR_SIZE, 30):
            pygame.draw.line(s, grid_c, (x, 0), (x, AVATAR_SIZE))
        for y in range(0, AVATAR_SIZE, 30):
            pygame.draw.line(s, grid_c, (0, y), (AVATAR_SIZE, y))

        # === HUD Brackets (Idle: dezent, Music: leuchten bei Beat) ===
        bk_bri = bri * (0.25 + 0.1 * mb + bf * 0.65)
        bk_len = 25
        bk_c = _scale(main_c, bk_bri)
        m = 8
        bk_w = 2 if bf > 0.3 else 1
        for (bx, by, dx, dy) in [
            (m, m, 1, 1), (AVATAR_SIZE-m, m, -1, 1),
            (m, AVATAR_SIZE-m, 1, -1), (AVATAR_SIZE-m, AVATAR_SIZE-m, -1, -1),
        ]:
            pygame.draw.line(s, bk_c, (bx, by), (bx + bk_len * dx, by), bk_w)
            pygame.draw.line(s, bk_c, (bx, by), (bx, by + bk_len * dy), bk_w)

        # === Scan-Line (Idle: langsam, Music: schnell + heller) ===
        scan_speed = 1.5 + rms_f * 3.0 * mb
        scan_y = int((tick * scan_speed) % AVATAR_SIZE)
        scan_alpha = 20 + int((10 + rms_f * 40) * mb)
        scan_surf = pygame.Surface((AVATAR_SIZE, 2 + int(mb)), pygame.SRCALPHA)
        scan_surf.fill((*main_c, scan_alpha))
        s.blit(scan_surf, (0, scan_y))
        if mb > 0.3:
            scan_y2 = (AVATAR_SIZE - int((tick * 0.7) % AVATAR_SIZE)) % AVATAR_SIZE
            scan_surf2 = pygame.Surface((AVATAR_SIZE, 1), pygame.SRCALPHA)
            scan_surf2.fill((*main_c, 15))
            s.blit(scan_surf2, (0, scan_y2))

        # === Glow (Idle: dezent, Music: STARK) ===
        glow_str = bri * (0.5 + self._s_tension * 0.3)
        glow_str *= 1.0 + self._s_voice_speak * 0.8
        glow_str *= 1.0 + self._s_voice_listen * 0.3
        glow_str *= 1.0 + rms_f * 2.5 * mb  # Music-Boost
        glow_str += bf * 1.5  # Beat-Explosion
        glow_r = int((130 + int(bass_f * 30 * mb)) * pulse)
        glow_key = (main_c, int(glow_str * 4), glow_r // 4)
        if glow_key != self._glow_key:
            self._glow_surface = _render_glow(glow_r, main_c, glow_str)
            self._glow_key = glow_key
        if self._glow_surface:
            gr = self._glow_surface.get_rect(center=(CX, CY))
            s.blit(self._glow_surface, gr)

        # === MUSIC: Shockwave-Ring bei Beat ===
        if beat and mb > 0.3:
            self._shockwave_r = 90.0
            self._shockwave_alpha = 200.0
        if self._shockwave_alpha > 5:
            sw_r = int(self._shockwave_r)
            sw_a = int(self._shockwave_alpha)
            if 5 < sw_r < AVATAR_SIZE // 2:
                sw_surf = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(sw_surf, CX, CY, sw_r, (*main_c, sw_a))
                if sw_r > 6:
                    pygame.gfxdraw.aacircle(sw_surf, CX, CY, sw_r - 1,
                                            (*main_c, sw_a // 2))
                s.blit(sw_surf, (0, 0))
            self._shockwave_r += 4.0
            self._shockwave_alpha *= 0.82

        # === Halo-Ringe (Idle: statisch, Music: pumpt mit Bass) ===
        halo_r1 = int((110 + bass_pump * 15) * pulse)
        halo_r2 = int((105 + bass_pump * 10) * pulse)
        halo_bri1 = bri * (0.2 + bass_f * 0.35 * mb)
        halo_bri2 = bri * (0.1 + bass_f * 0.18 * mb)
        if halo_r1 > 5:
            pygame.gfxdraw.aacircle(s, CX, CY, min(halo_r1, 148),
                                    _scale(main_c, halo_bri1))
        if halo_r2 > 5:
            pygame.gfxdraw.aacircle(s, CX, CY, min(halo_r2, 148),
                                    _scale(main_c, halo_bri2))

        # === Segmentierter Deko-Ring (Music: schneller) ===
        seg_r = int((100 + bass_pump * 8) * pulse)
        seg_speed = 0.15 + bass_f * 0.5 * mb
        seg_bri = bri * (0.15 + bass_f * 0.3 * mb)
        seg_c = _scale(main_c, seg_bri)
        for seg in range(4):
            start_a = seg * 90 + 10 + tick * seg_speed
            end_a = start_a + 60
            for a_deg in range(int(start_a), int(end_a), 2):
                a = math.radians(a_deg)
                px = int(CX + math.cos(a) * seg_r)
                py = int(CY + math.sin(a) * seg_r)
                if 0 <= px < AVATAR_SIZE and 0 <= py < AVATAR_SIZE:
                    s.set_at((px, py), seg_c)

        # === AEUSSERER RING (Idle: 3px, Music: 3-8px pumpt) ===
        ro = int(90 * (pulse + bass_pump))
        ro += int(self._s_voice_speak * 3.0 * math.sin(self._speech_phase * 3.1))
        ro = max(12, min(ro, 140))
        ring_thick = 3 + int(bass_f * 5 * mb)
        half_t = ring_thick // 2
        ring_bri_boost = 1.0 + bass_f * 0.5 * mb
        rc_bright = _scale(main_c, min(1.0, bri * ring_bri_boost))
        for dr in range(-half_t, half_t + 1):
            r = ro + dr
            if r > 2:
                dist = abs(dr) / max(1, half_t)
                ring_fade = 1.0 - dist * 0.6
                pygame.gfxdraw.aacircle(s, CX, CY, r,
                                        _scale(rc_bright, ring_fade))

        # === TICKMARKS (Idle: statisch, Music: ±40% Flicker) ===
        for i in range(40):
            angle = math.radians(i * 9.0 - 90)
            if i % 10 == 0:
                tl_in, tl_out, tw, bri_f = 18, 10, 2, 0.9
            elif i % 5 == 0:
                tl_in, tl_out, tw, bri_f = 12, 6, 2, 0.6
            elif i % 2 == 0:
                tl_in, tl_out, tw, bri_f = 6, 4, 1, 0.4
            else:
                tl_in, tl_out, tw, bri_f = 4, 2, 1, 0.2

            flicker = 0.0
            flicker_bri = 1.0
            if mb > 0.1 and high_f > 0.02:
                tick_phase = pp * 7.0 + i * 1.1
                flicker = high_f * 0.40 * math.sin(tick_phase) * mb
                flicker_bri = 1.0 + high_f * 0.6 * abs(math.sin(tick_phase)) * mb
            tl_in_m = tl_in * (1.0 + flicker)

            x1 = CX + math.cos(angle) * (ro - tl_in_m)
            y1 = CY + math.sin(angle) * (ro - tl_in_m)
            x2 = CX + math.cos(angle) * (ro + tl_out)
            y2 = CY + math.sin(angle) * (ro + tl_out)
            tc = _scale(main_c, min(1.0, bri * bri_f * flicker_bri))
            pygame.draw.line(s, tc, (int(x1), int(y1)), (int(x2), int(y2)), tw)

        # === Mittlerer Ring (Idle: stabil, Music: Mid-Pump) ===
        mid_mod = mid_f * 8.0 * math.sin(pp * 1.3) * mb
        mid_r1 = int(78 * pulse + mid_mod)
        mid_r2 = int(75 * pulse + mid_f * 5.0 * math.sin(pp * 1.7) * mb)
        mid_r1 = max(5, min(mid_r1, 130))
        mid_r2 = max(5, min(mid_r2, 130))
        mid_ring_bri = bri * (0.35 + 0.1 * mb + mid_f * 0.35 * mb)
        pygame.gfxdraw.aacircle(s, CX, CY, mid_r1, _scale(main_c, mid_ring_bri))
        pygame.gfxdraw.aacircle(s, CX, CY, mid_r2,
                                _scale(main_c, mid_ring_bri * 0.6))

        # === Innerer Ring ===
        inner_amp = 0.03 * (1 + self._s_tension * 2.5)
        inner_amp *= 1.0 - self._s_voice_listen * 0.6
        ip = 1.0 + math.sin(pp * 1.5) * inner_amp
        ri = int(68 * ip)
        ri += int(self._s_voice_speak * 2.5 * math.sin(self._speech_phase * 2.7))
        ri = max(5, ri)
        inner_bri = bri * (0.55 + 0.1 * mb + mid_f * 0.25 * mb)
        pygame.gfxdraw.aacircle(s, CX, CY, ri, _scale(main_c, inner_bri))
        pygame.gfxdraw.aacircle(s, CX, CY, ri - 1,
                                _scale(main_c, inner_bri * 0.5))

        # === Mini-Ticks am inneren Ring ===
        for i in range(8):
            angle = math.radians(i * 45 + tick * 0.08)
            x1 = CX + math.cos(angle) * (ri - 4)
            y1 = CY + math.sin(angle) * (ri - 4)
            x2 = CX + math.cos(angle) * (ri + 3)
            y2 = CY + math.sin(angle) * (ri + 3)
            pygame.draw.line(s, _scale(main_c, bri * 0.5),
                             (int(x1), int(y1)), (int(x2), int(y2)), 1)

        # === IRIS (Idle: stabil, Music: ±20% Mid-Reaktion) ===
        mid_iris = mid_f * 0.20 * math.sin(pp * 1.7) * mb
        ir = int(50 * (0.82 + bri * 0.18 + mid_iris))
        ir = max(10, min(ir, 65))
        ix = int(CX + self._pupil_dx)
        iy = int(CY + self._pupil_dy)

        iris_base_bri = bri * (0.2 + rms_f * 0.15 * mb)
        pygame.gfxdraw.filled_circle(s, ix, iy, ir, _scale(iris_c, iris_base_bri))

        # Konzentrische Micro-Ringe
        for ring_i in range(8):
            ring_r = ir - ring_i * 5
            if ring_r <= 4:
                break
            ring_bri = bri * (0.55 - ring_i * 0.05 + rms_f * 0.15 * mb)
            pygame.gfxdraw.aacircle(s, ix, iy, ring_r,
                                    _scale(iris_c, max(0.08, ring_bri)))

        # Iris Rand-Gradient
        for dr in range(6):
            r = ir - dr
            if r <= 4:
                break
            t = dr / 6.0
            c = _lerp(_scale(main_c, bri * (0.75 + rms_f * 0.25 * mb)),
                      _scale(iris_c, bri * 0.3), t)
            pygame.gfxdraw.aacircle(s, ix, iy, r, c)

        # === Radiale Iris-Strahlen ===
        pupil_ratio = 0.22 + self._s_tension * 0.35
        pupil_inner = ir * pupil_ratio + 4
        for j in range(24):
            a = math.radians(j * 15.0 + tick * 0.1)
            r_in = pupil_inner
            r_out = ir - 4
            if r_in < r_out:
                ray_bri = (0.2 + 0.1 * math.sin(pp + j * 0.5) + rms_f * 0.15 * mb
                           if j % 3 == 0 else 0.12 + rms_f * 0.08 * mb)
                ray_c = _scale(iris_c, bri * ray_bri)
                rx1 = ix + math.cos(a) * r_in
                ry1 = iy + math.sin(a) * r_in
                rx2 = ix + math.cos(a) * r_out
                ry2 = iy + math.sin(a) * r_out
                pygame.draw.aaline(s, ray_c, (rx1, ry1), (rx2, ry2))

        # === Pupille (Music: kontrahiert bei Bass) ===
        voice_pupil = pupil_ratio
        voice_pupil += self._s_voice_speak * 0.07 * math.sin(
            self._speech_phase * 1.5)
        voice_pupil *= 1.0 - self._s_voice_listen * 0.35
        voice_pupil *= 1.0 - bass_f * 0.25 * mb
        pr = int(ir * voice_pupil)
        if flash:
            pr = int(ir * 0.12)
        pr = max(3, pr)

        pygame.gfxdraw.filled_circle(s, ix, iy, pr, (2, 2, 5))
        pupil_edge_bri = bri * (0.3 + bf * 0.4)
        pygame.gfxdraw.aacircle(s, ix, iy, pr, _scale(main_c, pupil_edge_bri))

        # Reflexe
        hr = max(3, pr // 3 + 1)
        hx = int(ix + pr * 0.28)
        hy = int(iy - pr * 0.33)
        pygame.gfxdraw.filled_circle(s, hx, hy, hr,
                                     _scale((255, 255, 255), bri * 0.9))
        pygame.gfxdraw.aacircle(s, hx, hy, hr,
                                _scale((255, 255, 255), bri * 0.7))
        h2r = max(2, pr // 5)
        h2x = int(ix - pr * 0.22)
        h2y = int(iy + pr * 0.28)
        pygame.gfxdraw.filled_circle(s, h2x, h2y, h2r,
                                     _scale((255, 255, 255), bri * 0.4))

        # === MUSIC: Beat-Flash Overlay ===
        if bf > 0.15:
            flash_surf = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA)
            flash_surf.fill((*main_c, int(bf * 70)))
            s.blit(flash_surf, (0, 0))

        # === Augenlider ===
        if self._blink_progress > 0.01:
            self._render_lids(s, bri, main_c)
        elif self._s_cpu > 0.85:
            self._render_droopy(s, bri, main_c)

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
        else:
            # Music Data JEDEN anderen Frame lesen (~33ms) fuer Sync
            # Binary IPC ist <0.1ms, kein Performance-Problem
            self._read_music_fast()

        # Smooth Interpolation (rate=dt*6 ~ 300ms Settling)
        rate = min(1.0, dt * 6.0)
        self._s_tension += (self._tension - self._s_tension) * rate
        self._s_dominance += (self._dominance - self._s_dominance) * rate
        self._s_cpu += (self._cpu_temp - self._s_cpu) * rate

        # Music Data: DIREKT uebernehmen, KEINE zweite Interpolation!
        # Visualizer liefert bereits EMA-geglattete Werte via Binary IPC.
        # Doppeltes Smoothing war die Ursache fuer Sync-Probleme.
        if self._music_active:
            self._s_music_bass = self._t_music_bass
            self._s_music_mid = self._t_music_mid
            self._s_music_high = self._t_music_high
            self._s_music_rms = self._t_music_rms
        else:
            # Smooth Decay wenn keine Musik (nur hier, nicht bei aktiver Musik)
            self._s_music_bass *= 0.92
            self._s_music_mid *= 0.92
            self._s_music_high *= 0.92
            self._s_music_rms *= 0.92

        # Dual-Modus Crossfade: 0→1 bei Musik, 1→0 bei Stille (~1s)
        blend_target = 1.0 if self._music_active else 0.0
        blend_rate = min(1.0, dt * 2.0)  # ~500ms Uebergang
        self._music_blend += (blend_target - self._music_blend) * blend_rate

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

    def _read_music_fast(self):
        """Music-Daten direkt aus Binary-IPC lesen (~0.1ms, kein JSON-Parse)."""
        try:
            import struct as _struct
            with open("/dev/shm/moloch_music.bin", "rb") as f:
                raw = f.read(22)
            if len(raw) >= 22:
                rms, bass, mid, high, ts, active, beat = _struct.unpack("=5f2B", raw)
                self._music_active = bool(active)
                if self._music_active:
                    self._t_music_bass = bass
                    self._t_music_mid = mid
                    self._t_music_high = high
                    self._t_music_rms = rms
                    self._beat_detected = bool(beat)
                else:
                    self._t_music_bass = 0.0
                    self._t_music_mid = 0.0
                    self._t_music_high = 0.0
                    self._t_music_rms = 0.0
                    self._beat_detected = False
        except Exception:
            pass

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
