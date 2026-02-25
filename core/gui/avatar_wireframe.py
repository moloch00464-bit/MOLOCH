#!/usr/bin/env python3
"""
M.O.L.O.C.H. Wireframe Avatar — Max Headroom Style
=====================================================

Prozedurales Low-Poly Drahtgitter-Gesicht im Retro-Cyberpunk Stil.
Kein rundes Auge — ein Wireframe-Face wie Max Headroom trifft Ghost in the Shell.

Rendering-Pipeline: PyGame Surface -> PIL Image -> ImageTk -> Tkinter Label
Kein separates Fenster — alles inline im Panel (identisch zu panel_avatar.py).

Persoenlichkeits-Modi:
  GUARDIAN  -> Blaue/Cyan Linien, perfekte Symmetrie, stabiler Refresh
  SHADOW   -> Rote/Violette Linien, Jitter, Scanline-Verschiebung
  BERSERKER -> Polygon-Reduktion, Heavy Glitching, knallrot + weiss

Music-Reaktion (via Binary IPC /dev/shm/moloch_music.bin):
  Bass -> Grid-Puls, Scanline-Helligkeit
  Mid  -> Vertex-Displacement (Gesicht atmet)
  High -> Glitch-Frequenz, Noise

Rein passiv — liest NUR Core State, schreibt NICHTS.
"""

import tkinter as tk
import math
import random
import time
import logging
import struct as _struct

from core.gui.panel_styles import BG_FRAME, FG_DIM

# Graceful Imports
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

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

logger = logging.getLogger("WireframeAvatar")

# Music Visualizer Konstante
MAX_VISUAL_AMP = 0.15


# =============================================================================
# Konstanten
# =============================================================================

AVATAR_SIZE = 300
CX = AVATAR_SIZE // 2
CY = AVATAR_SIZE // 2 - 8       # Gesicht etwas hoeher (Platz fuer Track)
ANIM_INTERVAL_MS = 33            # ~30 FPS

FACE_SCALE = 135                 # Pixel pro normalisierte Einheit
FOV = 2.5                        # Perspektiv-Faktor

# Hintergrund
BG_RGB = (8, 8, 18)
GRID_RGB = (15, 18, 35)

# Zone-Farben (RGB)
COL_GUARDIAN = (0, 170, 255)
COL_SHADOW = (255, 34, 0)
COL_SHADOW_VIOLET = (180, 20, 80)
COL_BERSERKER = (255, 0, 0)
COL_NEUTRAL = (255, 255, 255)

# Hysterese
DOM_HYSTERESIS = 0.15

# Edge-Gewichte
EDGE_BRIGHT = 2      # Kontur, Augen, Mund
EDGE_NORMAL = 1      # Struktur
EDGE_DIM = 0         # Kreuzversteifung


# =============================================================================
# Farb-Hilfsfunktionen
# =============================================================================

def _lerp_c(c1, c2, t):
    """Lineare Interpolation zwischen zwei RGB-Tupeln."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _scale_c(color, factor):
    """Farbe um Helligkeitsfaktor skalieren."""
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def _dom_color(d):
    """Dominance -> Hauptfarbe."""
    if d >= 0:
        return _lerp_c(COL_NEUTRAL, COL_GUARDIAN, d)
    return _lerp_c(COL_NEUTRAL, COL_SHADOW, -d)


def _rgb_to_hex(rgb):
    """RGB-Tupel -> Hex-String."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _hex_scale(color_hex, factor):
    """Hex-Farbe skalieren."""
    factor = max(0.0, factor)
    r = min(255, int(int(color_hex[1:3], 16) * factor))
    g = min(255, int(int(color_hex[3:5], 16) * factor))
    b = min(255, int(int(color_hex[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# Gesichts-Mesh
# =============================================================================

def _build_face_mesh():
    """Low-Poly Human Face Mesh erstellen.

    Returns:
        vertices: np.ndarray shape (N, 3) — Basis-Positionen
        edges: list of (i, j, weight) — Kanten mit Gewicht
        groups: dict — Vertex-Gruppen fuer Animation
    """
    verts = np.array([
        # === KONTUR (0-16) ===
        [0.0,   -0.88, 0.02],     # 0:  Scheitel Mitte
        [-0.22, -0.85, 0.04],     # 1:  Scheitel links
        [0.22,  -0.85, 0.04],     # 2:  Scheitel rechts
        [-0.42, -0.70, 0.06],     # 3:  Stirn links
        [0.0,   -0.73, 0.08],     # 4:  Stirn Mitte
        [0.42,  -0.70, 0.06],     # 5:  Stirn rechts
        [-0.52, -0.45, 0.08],     # 6:  Schlaefe links
        [0.52,  -0.45, 0.08],     # 7:  Schlaefe rechts
        [-0.56, -0.15, 0.10],     # 8:  Wangenknochen links
        [0.56,  -0.15, 0.10],     # 9:  Wangenknochen rechts
        [-0.48, 0.10, 0.07],      # 10: Wange links
        [0.48,  0.10, 0.07],      # 11: Wange rechts
        [-0.38, 0.30, 0.04],      # 12: Kiefer links
        [0.38,  0.30, 0.04],      # 13: Kiefer rechts
        [-0.20, 0.45, 0.02],      # 14: Kinn links
        [0.20,  0.45, 0.02],      # 15: Kinn rechts
        [0.0,   0.52, 0.0],       # 16: Kinn Spitze

        # === BRAUENWULST (17-22) ===
        [-0.40, -0.40, 0.14],     # 17: Braue links aussen
        [-0.27, -0.46, 0.16],     # 18: Braue links Bogen
        [-0.12, -0.42, 0.16],     # 19: Braue links innen
        [0.12,  -0.42, 0.16],     # 20: Braue rechts innen
        [0.27,  -0.46, 0.16],     # 21: Braue rechts Bogen
        [0.40,  -0.40, 0.14],     # 22: Braue rechts aussen

        # === LINKES AUGE (23-28) ===
        [-0.36, -0.30, 0.12],     # 23: Aussen-Winkel
        [-0.27, -0.37, 0.14],     # 24: Oberlid
        [-0.14, -0.30, 0.14],     # 25: Innen-Winkel
        [-0.27, -0.23, 0.12],     # 26: Unterlid
        [-0.29, -0.31, 0.17],     # 27: Pupille links
        [-0.23, -0.31, 0.17],     # 28: Pupille rechts

        # === RECHTES AUGE (29-34) ===
        [0.14,  -0.30, 0.14],     # 29: Innen-Winkel
        [0.27,  -0.37, 0.14],     # 30: Oberlid
        [0.36,  -0.30, 0.12],     # 31: Aussen-Winkel
        [0.27,  -0.23, 0.12],     # 32: Unterlid
        [0.23,  -0.31, 0.17],     # 33: Pupille links
        [0.29,  -0.31, 0.17],     # 34: Pupille rechts

        # === NASE (35-41) ===
        [0.0,   -0.36, 0.18],     # 35: Nasenruecken oben
        [0.0,   -0.18, 0.22],     # 36: Nasenruecken Mitte
        [0.0,   -0.04, 0.24],     # 37: Nasenspitze
        [-0.08, -0.01, 0.20],     # 38: Spitze links
        [0.08,  -0.01, 0.20],     # 39: Spitze rechts
        [-0.11, 0.03, 0.16],      # 40: Nasenfluegel links
        [0.11,  0.03, 0.16],      # 41: Nasenfluegel rechts

        # === MUND OBEN (42-46) ===
        [-0.22, 0.15, 0.14],      # 42: Mundwinkel links
        [-0.11, 0.12, 0.17],      # 43: Oberlippe links
        [0.0,   0.10, 0.18],      # 44: Oberlippe Mitte (Amorbogen)
        [0.11,  0.12, 0.17],      # 45: Oberlippe rechts
        [0.22,  0.15, 0.14],      # 46: Mundwinkel rechts

        # === MUND UNTEN (47-49) ===
        [-0.11, 0.20, 0.16],      # 47: Unterlippe links
        [0.0,   0.22, 0.17],      # 48: Unterlippe Mitte
        [0.11,  0.20, 0.16],      # 49: Unterlippe rechts

        # === INNERE STRUKTUR (50-57) ===
        [-0.32, -0.10, 0.12],     # 50: Nasolabial links oben
        [-0.26, 0.08, 0.11],      # 51: Nasolabial links unten
        [0.32,  -0.10, 0.12],     # 52: Nasolabial rechts oben
        [0.26,  0.08, 0.11],      # 53: Nasolabial rechts unten
        [-0.38, -0.05, 0.11],     # 54: Mitte-Gesicht links
        [0.38,  -0.05, 0.11],     # 55: Mitte-Gesicht rechts
        [-0.16, 0.36, 0.04],      # 56: Kinnbereich links
        [0.16,  0.36, 0.04],      # 57: Kinnbereich rechts

        # === HALS (58-60) ===
        [-0.18, 0.62, -0.02],     # 58: Hals links
        [0.18,  0.62, -0.02],     # 59: Hals rechts
        [0.0,   0.68, -0.04],     # 60: Hals Mitte

        # === STIRN-DETAIL (61-63) ===
        [-0.32, -0.56, 0.10],     # 61: Stirn links-Mitte
        [0.0,   -0.58, 0.12],     # 62: Stirn Zentrum
        [0.32,  -0.56, 0.10],     # 63: Stirn rechts-Mitte
    ], dtype=np.float32)

    # Kanten: (Start-Index, End-Index, Gewicht)
    edges = [
        # --- Kontur (BRIGHT) ---
        (0, 1, EDGE_BRIGHT), (0, 2, EDGE_BRIGHT),
        (1, 3, EDGE_BRIGHT), (2, 5, EDGE_BRIGHT),
        (3, 6, EDGE_BRIGHT), (5, 7, EDGE_BRIGHT),
        (6, 8, EDGE_BRIGHT), (7, 9, EDGE_BRIGHT),
        (8, 10, EDGE_BRIGHT), (9, 11, EDGE_BRIGHT),
        (10, 12, EDGE_BRIGHT), (11, 13, EDGE_BRIGHT),
        (12, 14, EDGE_BRIGHT), (13, 15, EDGE_BRIGHT),
        (14, 16, EDGE_BRIGHT), (15, 16, EDGE_BRIGHT),

        # --- Brauenwulst (BRIGHT) ---
        (17, 18, EDGE_BRIGHT), (18, 19, EDGE_BRIGHT),
        (20, 21, EDGE_BRIGHT), (21, 22, EDGE_BRIGHT),
        (6, 17, EDGE_NORMAL), (7, 22, EDGE_NORMAL),
        (19, 35, EDGE_NORMAL), (20, 35, EDGE_NORMAL),

        # --- Linkes Auge (BRIGHT) ---
        (23, 24, EDGE_BRIGHT), (24, 25, EDGE_BRIGHT),
        (25, 26, EDGE_BRIGHT), (26, 23, EDGE_BRIGHT),
        (27, 28, EDGE_BRIGHT),
        (17, 23, EDGE_NORMAL), (19, 25, EDGE_NORMAL),

        # --- Rechtes Auge (BRIGHT) ---
        (29, 30, EDGE_BRIGHT), (30, 31, EDGE_BRIGHT),
        (31, 32, EDGE_BRIGHT), (32, 29, EDGE_BRIGHT),
        (33, 34, EDGE_BRIGHT),
        (20, 29, EDGE_NORMAL), (22, 31, EDGE_NORMAL),

        # --- Nase (BRIGHT) ---
        (35, 36, EDGE_BRIGHT), (36, 37, EDGE_BRIGHT),
        (37, 38, EDGE_BRIGHT), (37, 39, EDGE_BRIGHT),
        (38, 40, EDGE_NORMAL), (39, 41, EDGE_NORMAL),
        (40, 41, EDGE_NORMAL),

        # --- Mund (BRIGHT) ---
        (42, 43, EDGE_BRIGHT), (43, 44, EDGE_BRIGHT),
        (44, 45, EDGE_BRIGHT), (45, 46, EDGE_BRIGHT),
        (42, 47, EDGE_BRIGHT), (47, 48, EDGE_BRIGHT),
        (48, 49, EDGE_BRIGHT), (49, 46, EDGE_BRIGHT),

        # --- Nasolabial-Falten (NORMAL) ---
        (50, 51, EDGE_NORMAL), (52, 53, EDGE_NORMAL),
        (8, 50, EDGE_DIM), (40, 51, EDGE_DIM),
        (51, 42, EDGE_NORMAL),
        (9, 52, EDGE_DIM), (41, 53, EDGE_DIM),
        (53, 46, EDGE_NORMAL),

        # --- Mitte-Gesicht Struktur (DIM) ---
        (8, 54, EDGE_DIM), (54, 50, EDGE_DIM),
        (9, 55, EDGE_DIM), (55, 52, EDGE_DIM),

        # --- Kinnbereich (NORMAL) ---
        (12, 56, EDGE_NORMAL), (13, 57, EDGE_NORMAL),
        (56, 14, EDGE_NORMAL), (57, 15, EDGE_NORMAL),
        (56, 16, EDGE_DIM), (57, 16, EDGE_DIM),

        # --- Hals (DIM) ---
        (14, 58, EDGE_DIM), (16, 60, EDGE_DIM), (15, 59, EDGE_DIM),
        (58, 60, EDGE_DIM), (59, 60, EDGE_DIM),

        # --- Stirn-Detail (NORMAL) ---
        (3, 61, EDGE_NORMAL), (61, 62, EDGE_NORMAL),
        (62, 63, EDGE_NORMAL), (63, 5, EDGE_NORMAL),
        (1, 61, EDGE_DIM), (4, 62, EDGE_DIM), (2, 63, EDGE_DIM),
        (61, 18, EDGE_DIM), (63, 21, EDGE_DIM),

        # --- Kreuzversteifung (DIM) ---
        (3, 4, EDGE_DIM), (4, 5, EDGE_DIM),
        (23, 50, EDGE_DIM), (31, 52, EDGE_DIM),
        (26, 50, EDGE_DIM), (32, 52, EDGE_DIM),
        (10, 51, EDGE_DIM), (11, 53, EDGE_DIM),
        (10, 42, EDGE_DIM), (11, 46, EDGE_DIM),
    ]

    # Vertex-Gruppen fuer Animation
    groups = {
        "mouth_lower": np.array([47, 48, 49]),
        "mouth_upper": np.array([43, 44, 45]),
        "mouth_corners": np.array([42, 46]),
        "chin_follow": np.array([14, 15, 16, 56, 57]),
        "neck": np.array([58, 59, 60]),
        "eye_upper_l": np.array([24]),
        "eye_lower_l": np.array([26]),
        "eye_upper_r": np.array([30]),
        "eye_lower_r": np.array([32]),
        "pupil_l": np.array([27, 28]),
        "pupil_r": np.array([33, 34]),
        "brow_l": np.array([17, 18, 19]),
        "brow_r": np.array([20, 21, 22]),
        "nose_tip": np.array([37, 38, 39]),
        # Leuchtende Vertices (groesserer Glow)
        "hot": np.array([27, 28, 33, 34, 37, 44, 48, 24, 30, 0, 16]),
    }

    return verts, edges, groups


# =============================================================================
# Projektion + Rotation
# =============================================================================

def _project(x, y, z, cx, cy, scale, fov):
    """3D -> 2D Perspektiv-Projektion."""
    depth = fov / (fov + z)
    sx = cx + x * scale * depth
    sy = cy + y * scale * depth
    return sx, sy, depth


def _rotation_matrix_y(angle):
    """Y-Achsen Rotation (Kopf links/rechts)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def _rotation_matrix_x(angle):
    """X-Achsen Rotation (Kopf hoch/runter)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)


# =============================================================================
# Pre-Rendered Surfaces
# =============================================================================

def _create_scanline_overlay(size):
    """Statische CRT-Scanlines (einmalig berechnet)."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    for y in range(0, size, 3):
        pygame.draw.line(surf, (0, 0, 0, 40), (0, y), (size - 1, y))
    return surf


def _create_vignette(size):
    """Radiale Vignette (einmalig berechnet)."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    # Numpy-beschleunigt
    y_arr, x_arr = np.ogrid[:size, :size]
    r = np.sqrt((x_arr.astype(float) - cx) ** 2
                + (y_arr.astype(float) - cy) ** 2) / (size * 0.5)
    alpha = np.clip((r - 0.50) / 0.50 * 160, 0, 160).astype(np.uint8)
    try:
        arr = pygame.surfarray.pixels_alpha(surf)
        arr[:] = alpha.T
        del arr
    except Exception:
        # Fallback ohne surfarray: pixelweises Zeichnen
        for y in range(size):
            for x in range(size):
                a = int(alpha[y, x])
                if a > 5:
                    surf.set_at((x, y), (0, 0, 0, a))
    return surf


# =============================================================================
# WireframeAvatarModule
# =============================================================================

class WireframeAvatarModule:
    """
    M.O.L.O.C.H. Wireframe Avatar — Max Headroom Style.

    PyGame Off-Screen Rendering -> Tkinter.
    Gleiches Interface wie AvatarModule (start/stop/update_from_status).
    """

    def __init__(self, parent_frame, service_proxy):
        self._parent = parent_frame
        self._service = service_proxy
        self._running = False
        self._after_id = None
        self._logger = logging.getLogger("WireframeAvatar")

        # Dependencies pruefen
        if not (_PYGAME_OK and _PIL_OK and _NUMPY_OK):
            missing = []
            if not _PYGAME_OK:
                missing.append("pygame")
            if not _PIL_OK:
                missing.append("PIL")
            if not _NUMPY_OK:
                missing.append("numpy")
            self._logger.error(f"Fehlende Module: {', '.join(missing)}")
            tk.Label(parent_frame, text=f"Avatar: {', '.join(missing)} fehlt",
                     bg=BG_FRAME, fg=FG_DIM).pack(pady=20)
            self._enabled = False
            return

        self._enabled = True

        # Off-Screen Surface
        self._surface = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE))

        # Mesh laden
        self._base_verts, self._edges, self._groups = _build_face_mesh()
        self._num_verts = len(self._base_verts)

        # Pre-Rendered Overlays
        self._scanline_surf = _create_scanline_overlay(AVATAR_SIZE)
        self._vignette_surf = _create_vignette(AVATAR_SIZE)

        # --- Core State (Zielwerte) ---
        self._tension = 0.0
        self._dominance = 0.5
        self._cpu_temp = 0.0
        self._cpu_temp_celsius = 0.0
        self._zone = "guardian"
        self._status_text = "Idle"

        # --- Smooth Interpolation ---
        self._s_tension = 0.0
        self._s_dominance = 0.5
        self._s_cpu = 0.0
        self._visual_dom = 0.5
        self._dom_at_last_change = 0.5

        # --- Animation State ---
        self._tick = 0
        self._pulse_phase = 0.0
        self._last_time = time.monotonic()

        # Blinzeln
        self._blink_progress = 0.0
        self._blinking = False
        self._blink_opening = False
        self._next_blink_tick = random.randint(100, 250)

        # Pupillen-Mikrobewegung
        self._pupil_dx = 0.0
        self._pupil_dy = 0.0
        self._target_pdx = 0.0
        self._target_pdy = 0.0

        # Kopf-Rotation
        self._head_yaw = 0.0    # Y-Achse (links/rechts)
        self._head_pitch = 0.0  # X-Achse (hoch/runter)

        # --- Voice State ---
        self._tts_active = False
        self._ptt_active = False
        self._s_voice_speak = 0.0
        self._s_voice_listen = 0.0
        self._speech_phase = 0.0

        # --- Music Reactive State ---
        self._s_music_bass = 0.0
        self._s_music_mid = 0.0
        self._s_music_high = 0.0
        self._s_music_rms = 0.0
        self._t_music_bass = 0.0
        self._t_music_mid = 0.0
        self._t_music_high = 0.0
        self._t_music_rms = 0.0
        self._music_active = False
        self._beat_detected = False
        self._music_blend = 0.0

        # Beat-Flash
        self._beat_flash = 0.0

        # --- Glitch State (Shadow/Berserker) ---
        self._glitch_y = 0
        self._glitch_h = 0
        self._glitch_dx = 0
        self._glitch_frames = 0

        # Berserker Edge-Dropout
        self._edge_mask = None     # Bool-Array, True = zeichnen
        self._edge_mask_ttl = 0

        # Flash
        self._flash_until = 0.0

        # --- Projected Vertices Cache ---
        self._proj_x = np.zeros(self._num_verts, dtype=np.float32)
        self._proj_y = np.zeros(self._num_verts, dtype=np.float32)
        self._proj_depth = np.zeros(self._num_verts, dtype=np.float32)

        # --- Tkinter GUI ---
        self._photo = None
        self._label = tk.Label(
            parent_frame, bg="#080812",
            width=AVATAR_SIZE, height=AVATAR_SIZE,
        )
        self._label.pack(padx=5, pady=5)

        # Info-Bereich
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

        # Track-Anzeige
        self._track_label = tk.Label(
            parent_frame, text="", bg=BG_FRAME, fg="#1DB954",
            font=("Helvetica", 9), anchor="w",
        )
        self._track_text = ""
        self._track_scroll_offset = 0
        self._track_scroll_pause = 0
        self._last_track_name = ""
        self._track_visible = False

    # =========================================================================
    # Rendering — Hauptfunktion
    # =========================================================================

    def _render(self):
        """Kompletten Frame rendern."""
        s = self._surface
        s.fill(BG_RGB)

        now = time.monotonic()
        flash = now < self._flash_until
        tick = self._tick
        mb = self._music_blend

        # Musik-Werte normalisiert (0-1)
        music_dim = 1.0 - self._s_voice_speak * 0.4
        bass_f = min(1.0, self._s_music_bass / MAX_VISUAL_AMP) * music_dim
        mid_f = min(1.0, self._s_music_mid / MAX_VISUAL_AMP) * music_dim
        high_f = min(1.0, self._s_music_high / MAX_VISUAL_AMP) * music_dim
        rms_f = min(1.0, self._s_music_rms / MAX_VISUAL_AMP) * music_dim

        # Hauptfarbe
        if flash:
            main_c = (255, 68, 68)
            bri = 1.0
        elif self._zone == "berserker":
            main_c = COL_BERSERKER
            bri = 0.7 + self._s_tension * 0.3
        else:
            main_c = _dom_color(self._visual_dom)
            bri = 0.55 + self._s_tension * 0.45

        # Musik-Helligkeit
        bri = min(1.0, bri + rms_f * 0.3 * mb)

        # Beat-Flash
        beat_trigger = 1.0 if (self._beat_detected and mb > 0.3) else 0.0
        self._beat_flash = max(beat_trigger, self._beat_flash * 0.72)
        bf = self._beat_flash

        if bf > 0.1:
            main_c = _lerp_c(main_c, (255, 255, 255), bf * 0.5)

        # =====================================================================
        # 1) HINTERGRUND: Grid
        # =====================================================================
        grid_bri = 18 + int(bass_f * 20 * mb)
        grid_c = (grid_bri, grid_bri, grid_bri + 12)

        # Perspektiv-Grid (konvergierend)
        spacing = 30
        for x in range(0, AVATAR_SIZE, spacing):
            pygame.draw.line(s, grid_c, (x, 0), (x, AVATAR_SIZE))
        for y in range(0, AVATAR_SIZE, spacing):
            pygame.draw.line(s, grid_c, (0, y), (AVATAR_SIZE, y))

        # =====================================================================
        # 2) HINTERGRUND: Moving Scanline
        # =====================================================================
        scan_speed = 1.5 + rms_f * 3.0 * mb
        scan_y = int((tick * scan_speed) % AVATAR_SIZE)
        scan_alpha = 18 + int((8 + rms_f * 35) * mb)
        scan_surf = pygame.Surface((AVATAR_SIZE, 2), pygame.SRCALPHA)
        scan_surf.fill((*main_c, min(255, scan_alpha)))
        s.blit(scan_surf, (0, scan_y))

        # Zweite Scanline (nur bei Musik)
        if mb > 0.3:
            sy2 = (AVATAR_SIZE - int((tick * 0.7) % AVATAR_SIZE)) % AVATAR_SIZE
            scan2 = pygame.Surface((AVATAR_SIZE, 1), pygame.SRCALPHA)
            scan2.fill((*main_c, 12))
            s.blit(scan2, (0, sy2))

        # =====================================================================
        # 3) MESH TRANSFORMIEREN
        # =====================================================================
        verts = self._base_verts.copy()

        # --- Mund-Sync (TTS oder Musik) ---
        mouth_open = 0.0
        if self._s_voice_speak > 0.1:
            # TTS: Mund oeffnet mit RMS
            mouth_open = rms_f * 1.5 * self._s_voice_speak
            mouth_open += 0.15 * self._s_voice_speak * abs(math.sin(
                self._speech_phase * 2.3))
        elif mb > 0.1:
            # Musik: Subtile Mundbewegung mit Bass
            mouth_open = bass_f * 0.3 * mb

        mouth_open = min(1.0, mouth_open)
        if mouth_open > 0.01:
            dy_lower = mouth_open * 0.10
            dy_chin = mouth_open * 0.04
            g = self._groups
            verts[g["mouth_lower"], 1] += dy_lower
            verts[g["chin_follow"], 1] += dy_chin
            verts[g["neck"], 1] += dy_chin * 0.5
            # Oberlippe leicht hoch
            verts[g["mouth_upper"], 1] -= mouth_open * 0.02
            # Mundwinkel etwas breiter
            verts[g["mouth_corners"][0], 0] -= mouth_open * 0.02
            verts[g["mouth_corners"][1], 0] += mouth_open * 0.02

        # --- Blinzeln ---
        if self._blink_progress > 0.01:
            blink_dy = self._blink_progress * 0.07
            g = self._groups
            verts[g["eye_upper_l"], 1] += blink_dy
            verts[g["eye_lower_l"], 1] -= blink_dy
            verts[g["eye_upper_r"], 1] += blink_dy
            verts[g["eye_lower_r"], 1] -= blink_dy
            # Pupillen unsichtbar bei vollem Blinzeln
            if self._blink_progress > 0.8:
                verts[g["pupil_l"], 1] = verts[g["eye_upper_l"][0], 1]
                verts[g["pupil_r"], 1] = verts[g["eye_upper_r"][0], 1]

        # --- Pupillen-Mikrobewegung ---
        g = self._groups
        verts[g["pupil_l"], 0] += self._pupil_dx * 0.025
        verts[g["pupil_l"], 1] += self._pupil_dy * 0.015
        verts[g["pupil_r"], 0] += self._pupil_dx * 0.025
        verts[g["pupil_r"], 1] += self._pupil_dy * 0.015

        # --- Musik: Atem-Effekt (Mid → Gesicht skaliert leicht) ---
        if mid_f > 0.02 and mb > 0.05:
            breathe = 1.0 + mid_f * 0.04 * math.sin(
                self._pulse_phase * 1.3) * mb
            verts[:, 0] *= breathe
            verts[:, 1] *= breathe

        # --- Kopf-Rotation ---
        yaw_angle = self._head_yaw
        pitch_angle = self._head_pitch

        if abs(yaw_angle) > 0.001 or abs(pitch_angle) > 0.001:
            rot = np.eye(3, dtype=np.float32)
            if abs(yaw_angle) > 0.001:
                rot = rot @ _rotation_matrix_y(yaw_angle)
            if abs(pitch_angle) > 0.001:
                rot = rot @ _rotation_matrix_x(pitch_angle)
            verts = verts @ rot.T

        # --- Personality Jitter ---
        if self._zone == "shadow":
            jitter = np.random.uniform(-0.012, 0.012,
                                       size=(self._num_verts, 2))
            verts[:, :2] += jitter.astype(np.float32)
        elif self._zone == "berserker":
            jitter = np.random.uniform(-0.030, 0.030,
                                       size=(self._num_verts, 2))
            verts[:, :2] += jitter.astype(np.float32)

        # --- High-Freq Noise (Musik) ---
        if high_f > 0.05 and mb > 0.1:
            noise_amp = high_f * 0.015 * mb
            noise = np.random.uniform(-noise_amp, noise_amp,
                                      size=(self._num_verts, 2))
            verts[:, :2] += noise.astype(np.float32)

        # =====================================================================
        # 4) PROJEKTION 3D -> 2D
        # =====================================================================
        for i in range(self._num_verts):
            sx, sy, depth = _project(
                verts[i, 0], verts[i, 1], verts[i, 2],
                CX, CY, FACE_SCALE, FOV)
            self._proj_x[i] = sx
            self._proj_y[i] = sy
            self._proj_depth[i] = depth

        # --- Glitch-Band (Shadow/Berserker) ---
        if self._glitch_frames > 0:
            self._glitch_frames -= 1
            gy_lo = self._glitch_y
            gy_hi = self._glitch_y + self._glitch_h
            for i in range(self._num_verts):
                py = self._proj_y[i]
                if gy_lo <= py <= gy_hi:
                    self._proj_x[i] += self._glitch_dx

        # =====================================================================
        # 5) KANTEN ZEICHNEN
        # =====================================================================
        edge_count = len(self._edges)

        # Berserker: Edge-Dropout (zufaellig Kanten weglassen)
        if self._zone == "berserker":
            if self._edge_mask is None or self._edge_mask_ttl <= 0:
                self._edge_mask = np.random.random(edge_count) > 0.15
                self._edge_mask_ttl = random.randint(2, 8)
            self._edge_mask_ttl -= 1

        for idx, (i, j, weight) in enumerate(self._edges):
            # Berserker Dropout
            if (self._zone == "berserker"
                    and self._edge_mask is not None
                    and not self._edge_mask[idx]):
                continue

            x1, y1 = self._proj_x[i], self._proj_y[i]
            x2, y2 = self._proj_x[j], self._proj_y[j]

            # Clip gegen Fensterrand
            if (x1 < -10 or x1 > AVATAR_SIZE + 10
                    or y1 < -10 or y1 > AVATAR_SIZE + 10):
                if (x2 < -10 or x2 > AVATAR_SIZE + 10
                        or y2 < -10 or y2 > AVATAR_SIZE + 10):
                    continue

            # Tiefe-basierte Helligkeit (naeher = heller)
            avg_depth = (self._proj_depth[i] + self._proj_depth[j]) * 0.5
            depth_bri = 0.4 + avg_depth * 0.6

            # Gewicht → Helligkeit
            if weight == EDGE_BRIGHT:
                w_bri = 1.0
                line_w = 2
            elif weight == EDGE_NORMAL:
                w_bri = 0.65
                line_w = 1
            else:
                w_bri = 0.35
                line_w = 1

            # Shadow: gelegentliches Linien-Flackern
            if self._zone == "shadow" and random.random() < 0.03:
                w_bri *= random.uniform(0.2, 0.6)

            final_bri = bri * depth_bri * w_bri
            edge_c = _scale_c(main_c, final_bri)

            # Beat: Dicke Kanten bei Beat
            if bf > 0.3 and weight == EDGE_BRIGHT:
                line_w = 3

            if line_w >= 2:
                pygame.draw.line(s, edge_c,
                                 (int(x1), int(y1)), (int(x2), int(y2)),
                                 line_w)
            else:
                pygame.draw.aaline(s, edge_c,
                                   (x1, y1), (x2, y2))

        # =====================================================================
        # 6) VERTEX GLOW-PUNKTE
        # =====================================================================
        hot_set = set(self._groups["hot"].tolist())

        for i in range(self._num_verts):
            px = int(self._proj_x[i])
            py = int(self._proj_y[i])

            if px < 0 or px >= AVATAR_SIZE or py < 0 or py >= AVATAR_SIZE:
                continue

            depth_bri = 0.4 + self._proj_depth[i] * 0.6
            is_hot = i in hot_set

            if is_hot:
                # Groesserer Glow fuer wichtige Punkte
                glow_r = 4 + int(bf * 3)
                glow_alpha = int(min(255, (80 + bf * 120) * bri * depth_bri))
                glow_c = (*main_c, max(0, min(255, glow_alpha)))
                if glow_r > 1:
                    glow_surf = pygame.Surface(
                        (glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(
                        glow_surf, glow_r + 1, glow_r + 1, glow_r, glow_c)
                    s.blit(glow_surf,
                           (px - glow_r - 1, py - glow_r - 1))
                # Heller Kern
                dot_r = 2
                dot_c = _scale_c(main_c, min(1.0, bri * depth_bri * 1.2))
                pygame.gfxdraw.filled_circle(s, px, py, dot_r, dot_c)
            else:
                # Normaler Vertex-Punkt
                dot_c = _scale_c(main_c, bri * depth_bri * 0.6)
                s.set_at((px, py), dot_c)
                if bri * depth_bri > 0.5:
                    # 2px Punkt fuer bessere Sichtbarkeit
                    if px + 1 < AVATAR_SIZE:
                        s.set_at((px + 1, py), dot_c)

        # =====================================================================
        # 7) BEAT SHOCKWAVE
        # =====================================================================
        if bf > 0.15:
            sw_r = int(80 + (1.0 - bf) * 60)
            sw_alpha = int(bf * 100)
            if 5 < sw_r < AVATAR_SIZE // 2 and sw_alpha > 3:
                sw_surf = pygame.Surface(
                    (AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(
                    sw_surf, CX, CY, sw_r, (*main_c, sw_alpha))
                s.blit(sw_surf, (0, 0))

        # =====================================================================
        # 8) BERSERKER: Weisse Blitze
        # =====================================================================
        if self._zone == "berserker" and random.random() < 0.08:
            flash_y = random.randint(0, AVATAR_SIZE - 2)
            flash_h = random.randint(1, 3)
            flash_surf = pygame.Surface(
                (AVATAR_SIZE, flash_h), pygame.SRCALPHA)
            flash_surf.fill((255, 255, 255, random.randint(30, 100)))
            s.blit(flash_surf, (0, flash_y))

        # =====================================================================
        # 9) PIXEL-NOISE (Shadow/Berserker, Musik-abhaengig)
        # =====================================================================
        noise_chance = 0.0
        if self._zone == "shadow":
            noise_chance = 0.002 + high_f * 0.008 * mb
        elif self._zone == "berserker":
            noise_chance = 0.005 + high_f * 0.015 * mb

        if noise_chance > 0:
            noise_count = int(AVATAR_SIZE * AVATAR_SIZE * noise_chance)
            noise_count = min(noise_count, 200)  # CPU-Limit
            for _ in range(noise_count):
                nx = random.randint(0, AVATAR_SIZE - 1)
                ny = random.randint(0, AVATAR_SIZE - 1)
                nc = _scale_c(main_c, random.uniform(0.3, 0.8))
                s.set_at((nx, ny), nc)

        # =====================================================================
        # 10) POST-EFFECTS: Scanlines + Vignette
        # =====================================================================
        s.blit(self._scanline_surf, (0, 0))
        s.blit(self._vignette_surf, (0, 0))

        # =====================================================================
        # 11) HUD Brackets (Ecken)
        # =====================================================================
        bk_bri = bri * (0.25 + bf * 0.5)
        bk_c = _scale_c(main_c, bk_bri)
        bk_len = 22
        m = 6
        bk_w = 2 if bf > 0.3 else 1
        for (bx, by, dx, dy) in [
            (m, m, 1, 1),
            (AVATAR_SIZE - m, m, -1, 1),
            (m, AVATAR_SIZE - m, 1, -1),
            (AVATAR_SIZE - m, AVATAR_SIZE - m, -1, -1),
        ]:
            pygame.draw.line(s, bk_c, (bx, by),
                             (bx + bk_len * dx, by), bk_w)
            pygame.draw.line(s, bk_c, (bx, by),
                             (bx, by + bk_len * dy), bk_w)

        # =====================================================================
        # 12) BEAT FLASH OVERLAY
        # =====================================================================
        if bf > 0.2:
            flash_surf = pygame.Surface(
                (AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA)
            flash_surf.fill((*main_c, int(bf * 45)))
            s.blit(flash_surf, (0, 0))

    # =========================================================================
    # Surface -> Tkinter
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
        """Ein Animation-Frame: State Update -> Render -> Display."""
        if not self._running:
            return

        now = time.monotonic()
        dt = min(now - self._last_time, 0.1)
        self._last_time = now
        self._tick += 1

        # Core State lesen (alle 15 Ticks ~ 500ms)
        if self._tick % 15 == 0:
            self._read_core_state()
        else:
            self._read_music_fast()

        # Smooth Interpolation
        rate = min(1.0, dt * 6.0)
        self._s_tension += (self._tension - self._s_tension) * rate
        self._s_dominance += (self._dominance - self._s_dominance) * rate
        self._s_cpu += (self._cpu_temp - self._s_cpu) * rate

        # Music-Daten direkt uebernehmen (bereits EMA-geglaettet)
        if self._music_active:
            self._s_music_bass = self._t_music_bass
            self._s_music_mid = self._t_music_mid
            self._s_music_high = self._t_music_high
            self._s_music_rms = self._t_music_rms
        else:
            self._s_music_bass *= 0.92
            self._s_music_mid *= 0.92
            self._s_music_high *= 0.92
            self._s_music_rms *= 0.92

        # Crossfade Idle ↔ Music
        blend_target = 1.0 if self._music_active else 0.0
        blend_rate = min(1.0, dt * 2.0)
        self._music_blend += (blend_target - self._music_blend) * blend_rate

        # Voice Interpolation
        speak_target = 1.0 if self._tts_active else 0.0
        listen_target = (1.0 if (self._ptt_active and not self._tts_active)
                         else 0.0)
        self._s_voice_speak += (speak_target - self._s_voice_speak) * rate
        self._s_voice_listen += (listen_target - self._s_voice_listen) * rate

        # Visuelle Dominance mit Hysterese
        if abs(self._dominance - self._dom_at_last_change) > DOM_HYSTERESIS:
            self._dom_at_last_change = self._dominance
        self._visual_dom += (self._dom_at_last_change - self._visual_dom) * rate

        # Puls-Phase
        speed = 1.0 + self._s_tension * 3.0
        speed *= 1.0 + self._s_voice_speak * 2.0
        if self._s_voice_listen > 0.3:
            speed *= 0.6
        if self._s_cpu > 0.8:
            speed *= 0.7
        self._pulse_phase += speed * dt * math.tau / 3.0

        # Speech-Phase
        self._speech_phase += self._s_voice_speak * dt * math.tau * 3.5

        # Kopf-Rotation (subtil, Musik-abhaengig)
        mb = self._music_blend
        bass_f = min(1.0, self._s_music_bass / MAX_VISUAL_AMP)
        yaw_target = math.sin(now * 0.3) * 0.06
        yaw_target += math.sin(now * 0.7) * 0.03
        if mb > 0.1:
            yaw_target += bass_f * 0.04 * math.sin(
                self._pulse_phase * 0.5) * mb
        pitch_target = math.sin(now * 0.2) * 0.03
        if mb > 0.1:
            pitch_target += bass_f * 0.03 * math.sin(
                self._pulse_phase * 0.7) * mb

        head_rate = min(1.0, dt * 3.0)
        self._head_yaw += (yaw_target - self._head_yaw) * head_rate
        self._head_pitch += (pitch_target - self._head_pitch) * head_rate

        # Blinzel-Logik
        blink_speed = 4.0 * dt
        speaking = self._s_voice_speak > 0.5

        if speaking and self._blinking:
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
                    self._next_blink_tick = (
                        self._tick + random.randint(100, 250))
        elif not speaking and self._tick >= self._next_blink_tick:
            if self._s_tension > 0.7 and random.random() < 0.5:
                self._next_blink_tick = self._tick + random.randint(100, 250)
            else:
                self._blinking = True
                self._blink_opening = False

        # Pupillen-Mikrobewegung
        if self._tick % 60 == 0:
            self._target_pdx = random.uniform(-4, 4)
            self._target_pdy = random.uniform(-3, 3)
        self._pupil_dx += (self._target_pdx - self._pupil_dx) * 0.03
        self._pupil_dy += (self._target_pdy - self._pupil_dy) * 0.03

        # Glitch-Band triggern (Shadow/Berserker)
        if self._zone == "shadow":
            if random.random() < 0.025:
                self._glitch_y = random.randint(20, AVATAR_SIZE - 40)
                self._glitch_h = random.randint(8, 25)
                self._glitch_dx = random.randint(-12, 12)
                self._glitch_frames = random.randint(2, 5)
        elif self._zone == "berserker":
            if random.random() < 0.08:
                self._glitch_y = random.randint(10, AVATAR_SIZE - 30)
                self._glitch_h = random.randint(15, 40)
                self._glitch_dx = random.randint(-20, 20)
                self._glitch_frames = random.randint(1, 4)

        # Track-Text Scrolling
        if self._track_visible and self._track_text and self._tick % 2 == 0:
            max_vis = 35
            if len(self._track_text) > max_vis:
                if self._track_scroll_pause > 0:
                    self._track_scroll_pause -= 1
                    self._track_label.config(
                        text=self._track_text[:max_vis])
                else:
                    self._track_scroll_offset += 1
                    if (self._track_scroll_offset
                            > len(self._track_text) - max_vis + 10):
                        self._track_scroll_offset = 0
                        self._track_scroll_pause = 45
                    start = max(0, min(self._track_scroll_offset,
                                       len(self._track_text) - max_vis))
                    self._track_label.config(
                        text=self._track_text[start:start + max_vis])
            else:
                self._track_label.config(text=self._track_text)

        # Rendern + Display
        self._render()
        self._blit_to_tkinter()

        self._after_id = self._parent.after(
            ANIM_INTERVAL_MS, self._update_animation)

    # =========================================================================
    # State Updates
    # =========================================================================

    def update_from_status(self, status: dict):
        """Core State aus Panel-Status aktualisieren."""
        if not self._enabled or not status:
            return

        core = status.get("core", {})
        if not core:
            return

        # Voice State
        voice = status.get("voice", {})
        self._tts_active = bool(voice.get("speaking", False))
        self._ptt_active = bool(voice.get("recording", False))

        old_zone = self._zone

        self._tension = float(core.get("tension", 0.0))
        self._dominance = float(core.get("dominance", 0.5))
        self._cpu_temp = float(core.get("cpu_temp_norm", 0.0))
        self._cpu_temp_celsius = float(core.get("cpu_temp", 0.0))
        self._zone = core.get("zone", "guardian")

        # Status-Text
        det = status.get("detections", {})
        if isinstance(det, dict):
            faces = det.get("faces", [])
            persons = det.get("persons", 0)
            if faces:
                known = [f for f in faces if isinstance(f, dict)
                         and f.get("name", "unknown") != "unknown"]
                if known:
                    names = ", ".join(
                        f["name"].capitalize() for f in known[:3])
                    self._status_text = f"{names} erkannt"
                else:
                    self._status_text = f"{len(faces)} Gesicht(er)"
            elif persons:
                self._status_text = f"{persons} Person(en)"
            elif self._tension > 0.1:
                self._status_text = "Suche..."
            else:
                self._status_text = "Idle"

        # Info-Labels
        main_hex = _rgb_to_hex(_dom_color(self._visual_dom))
        bright = max(0.5, 0.5 + self._s_tension * 0.5)

        zone_name = {"guardian": "GUARDIAN", "shadow": "SHADOW",
                     "berserker": "BERSERKER"}.get(self._zone, "OFFLINE")
        self._zone_label.config(
            text=zone_name, fg=_hex_scale(main_hex, bright))

        self._state_label.config(
            text=f"T:{self._tension:.2f} | D:{self._dominance:+.2f}",
            fg=_hex_scale(main_hex, bright * 0.7),
        )

        cpu_str = (f"CPU: {self._cpu_temp_celsius:.0f}\u00b0C"
                   if self._cpu_temp_celsius > 0 else "")
        parts = []
        if cpu_str:
            parts.append(cpu_str)
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

        # Music-Daten
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

        # Track-Anzeige
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
        """Music-Daten aus Binary-IPC lesen (~0.1ms)."""
        try:
            with open("/dev/shm/moloch_music.bin", "rb") as f:
                raw = f.read(22)
            if len(raw) >= 22:
                rms, bass, mid, high, ts, active, beat = _struct.unpack(
                    "=5f2B", raw)
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
        """Core State via ServiceProxy lesen."""
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
        self._logger.info("Wireframe Avatar gestartet")

    def stop(self):
        """Animation stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None


# =============================================================================
# Standalone-Modus (Test/Demo)
# =============================================================================

def _run_standalone():
    """Eigenstaendiges PyGame-Fenster fuer Test/Demo."""
    if not _PYGAME_OK or not _NUMPY_OK:
        print("FEHLER: pygame und numpy werden benoetigt")
        return

    pygame.init()
    size = 640
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("M.O.L.O.C.H. Wireframe Avatar — Demo")
    clock = pygame.time.Clock()

    # Mesh laden
    base_verts, edges, groups = _build_face_mesh()
    num_verts = len(base_verts)

    # Pre-Rendered Surfaces
    scanline_surf = _create_scanline_overlay(size)
    vignette_surf = _create_vignette(size)

    # State
    zone = "guardian"
    tension = 0.3
    tick = 0
    pulse_phase = 0.0
    mouth_open = 0.0
    blink_progress = 0.0
    blinking = False
    blink_opening = False
    next_blink = random.randint(100, 250)
    speaking = False
    speech_phase = 0.0

    scale = int(size * 0.45)
    cx, cy = size // 2, size // 2 - 20

    font = pygame.font.SysFont("monospace", 16)

    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        tick += 1
        now = time.monotonic()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    zone = "guardian"
                elif event.key == pygame.K_2:
                    zone = "shadow"
                elif event.key == pygame.K_3:
                    zone = "berserker"
                elif event.key == pygame.K_SPACE:
                    speaking = not speaking

        # Farbe
        if zone == "guardian":
            main_c = COL_GUARDIAN
        elif zone == "shadow":
            main_c = COL_SHADOW
        else:
            main_c = COL_BERSERKER

        # Simulierte Musik
        sim_bass = 0.08 + 0.07 * abs(math.sin(now * 2.0))
        sim_mid = 0.05 + 0.05 * abs(math.sin(now * 3.1))
        sim_high = 0.03 + 0.03 * abs(math.sin(now * 5.7))
        sim_rms = 0.06 + 0.06 * abs(math.sin(now * 1.5))
        sim_beat = (math.sin(now * 2.0) > 0.95)

        bass_f = min(1.0, sim_bass / MAX_VISUAL_AMP)
        mid_f = min(1.0, sim_mid / MAX_VISUAL_AMP)
        high_f = min(1.0, sim_high / MAX_VISUAL_AMP)
        rms_f = min(1.0, sim_rms / MAX_VISUAL_AMP)

        # Puls
        speed = 1.0 + tension * 3.0
        if speaking:
            speed *= 2.5
            speech_phase += dt * math.tau * 3.5
        pulse_phase += speed * dt * math.tau / 3.0

        # Mund
        if speaking:
            mouth_target = 0.5 + 0.3 * abs(math.sin(speech_phase * 2.3))
        else:
            mouth_target = bass_f * 0.2
        mouth_open += (mouth_target - mouth_open) * min(1.0, dt * 12.0)

        # Blinzeln
        blink_spd = 4.0 * dt
        if blinking:
            if not blink_opening:
                blink_progress += blink_spd
                if blink_progress >= 1.0:
                    blink_progress = 1.0
                    blink_opening = True
            else:
                blink_progress -= blink_spd
                if blink_progress <= 0.0:
                    blink_progress = 0.0
                    blinking = False
                    blink_opening = False
                    next_blink = tick + random.randint(100, 250)
        elif tick >= next_blink:
            blinking = True
            blink_opening = False

        # Vertices transformieren
        verts = base_verts.copy()

        # Mund
        if mouth_open > 0.01:
            verts[groups["mouth_lower"], 1] += mouth_open * 0.10
            verts[groups["chin_follow"], 1] += mouth_open * 0.04
            verts[groups["mouth_upper"], 1] -= mouth_open * 0.02

        # Blinzeln
        if blink_progress > 0.01:
            bdy = blink_progress * 0.07
            verts[groups["eye_upper_l"], 1] += bdy
            verts[groups["eye_lower_l"], 1] -= bdy
            verts[groups["eye_upper_r"], 1] += bdy
            verts[groups["eye_lower_r"], 1] -= bdy

        # Atem
        breathe = 1.0 + mid_f * 0.04 * math.sin(pulse_phase * 1.3)
        verts[:, 0] *= breathe
        verts[:, 1] *= breathe

        # Kopf-Rotation
        yaw = math.sin(now * 0.3) * 0.08
        pitch = math.sin(now * 0.2) * 0.04
        rot = _rotation_matrix_y(yaw) @ _rotation_matrix_x(pitch)
        verts = verts @ rot.T

        # Personality Jitter
        if zone == "shadow":
            jitter = np.random.uniform(-0.012, 0.012,
                                       size=(num_verts, 2))
            verts[:, :2] += jitter.astype(np.float32)
        elif zone == "berserker":
            jitter = np.random.uniform(-0.030, 0.030,
                                       size=(num_verts, 2))
            verts[:, :2] += jitter.astype(np.float32)

        # High Noise
        if high_f > 0.05:
            noise = np.random.uniform(-high_f * 0.015, high_f * 0.015,
                                      size=(num_verts, 2))
            verts[:, :2] += noise.astype(np.float32)

        # Projektion
        proj = np.zeros((num_verts, 3), dtype=np.float32)
        for i in range(num_verts):
            sx, sy, depth = _project(
                verts[i, 0], verts[i, 1], verts[i, 2],
                cx, cy, scale, FOV)
            proj[i] = [sx, sy, depth]

        # Glitch (Shadow/Berserker)
        glitch_active = False
        glitch_y_lo = 0
        glitch_y_hi = 0
        glitch_dx = 0
        if zone in ("shadow", "berserker"):
            chance = 0.03 if zone == "shadow" else 0.08
            if random.random() < chance:
                glitch_active = True
                glitch_y_lo = random.randint(50, size - 100)
                glitch_y_hi = glitch_y_lo + random.randint(15, 40)
                glitch_dx = random.randint(-15, 15)

        if glitch_active:
            for i in range(num_verts):
                if glitch_y_lo <= proj[i, 1] <= glitch_y_hi:
                    proj[i, 0] += glitch_dx

        # ===== RENDER =====
        screen.fill(BG_RGB)

        # Grid
        grid_bri = 18 + int(bass_f * 20)
        grid_c = (grid_bri, grid_bri, grid_bri + 12)
        for x in range(0, size, 40):
            pygame.draw.line(screen, grid_c, (x, 0), (x, size))
        for y in range(0, size, 40):
            pygame.draw.line(screen, grid_c, (0, y), (size, y))

        # Moving Scanline
        scan_y = int((tick * 2.0) % size)
        scan_surf = pygame.Surface((size, 2), pygame.SRCALPHA)
        scan_surf.fill((*main_c, 25))
        screen.blit(scan_surf, (0, scan_y))

        bri = 0.7 + tension * 0.3

        # Beat-Flash
        bf = 0.0
        if sim_beat:
            bf = 0.8

        # Berserker Edge Mask
        edge_mask = None
        if zone == "berserker":
            edge_mask = np.random.random(len(edges)) > 0.15

        # Kanten zeichnen
        for idx, (i, j, weight) in enumerate(edges):
            if (zone == "berserker" and edge_mask is not None
                    and not edge_mask[idx]):
                continue

            x1, y1 = proj[i, 0], proj[i, 1]
            x2, y2 = proj[j, 0], proj[j, 1]

            avg_d = (proj[i, 2] + proj[j, 2]) * 0.5
            d_bri = 0.4 + avg_d * 0.6
            w_bri = {EDGE_BRIGHT: 1.0, EDGE_NORMAL: 0.65,
                     EDGE_DIM: 0.35}.get(weight, 0.5)

            if zone == "shadow" and random.random() < 0.03:
                w_bri *= random.uniform(0.2, 0.6)

            edge_c = _scale_c(main_c, bri * d_bri * w_bri)
            lw = 2 if weight == EDGE_BRIGHT else 1
            if bf > 0.3 and weight == EDGE_BRIGHT:
                lw = 3

            if lw >= 2:
                pygame.draw.line(screen, edge_c,
                                 (int(x1), int(y1)), (int(x2), int(y2)), lw)
            else:
                pygame.draw.aaline(screen, edge_c, (x1, y1), (x2, y2))

        # Vertex-Punkte
        hot_set = set(groups["hot"].tolist())
        for i in range(num_verts):
            px, py = int(proj[i, 0]), int(proj[i, 1])
            if 0 <= px < size and 0 <= py < size:
                d_bri = 0.4 + proj[i, 2] * 0.6
                if i in hot_set:
                    gr = 5
                    ga = int(min(255, 90 * bri * d_bri))
                    gs = pygame.Surface((gr * 2 + 2, gr * 2 + 2),
                                        pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(
                        gs, gr + 1, gr + 1, gr, (*main_c, ga))
                    screen.blit(gs, (px - gr - 1, py - gr - 1))
                    dc = _scale_c(main_c, min(1.0, bri * d_bri * 1.2))
                    pygame.gfxdraw.filled_circle(screen, px, py, 2, dc)
                else:
                    dc = _scale_c(main_c, bri * d_bri * 0.6)
                    screen.set_at((px, py), dc)

        # Noise (Shadow/Berserker)
        if zone in ("shadow", "berserker"):
            nc = 50 if zone == "shadow" else 120
            for _ in range(nc):
                nx = random.randint(0, size - 1)
                ny = random.randint(0, size - 1)
                screen.set_at((nx, ny),
                              _scale_c(main_c, random.uniform(0.3, 0.8)))

        # Berserker weisse Blitze
        if zone == "berserker" and random.random() < 0.1:
            fy = random.randint(0, size - 3)
            fs = pygame.Surface((size, 2), pygame.SRCALPHA)
            fs.fill((255, 255, 255, random.randint(30, 90)))
            screen.blit(fs, (0, fy))

        # Post-Effects
        screen.blit(scanline_surf, (0, 0))
        screen.blit(vignette_surf, (0, 0))

        # HUD
        bk_c = _scale_c(main_c, 0.3)
        bk_len = 30
        m = 8
        for (bx, by, dx, dy) in [
            (m, m, 1, 1), (size - m, m, -1, 1),
            (m, size - m, 1, -1), (size - m, size - m, -1, -1),
        ]:
            pygame.draw.line(screen, bk_c, (bx, by),
                             (bx + bk_len * dx, by), 1)
            pygame.draw.line(screen, bk_c, (bx, by),
                             (bx, by + bk_len * dy), 1)

        # Info-Text
        zone_text = font.render(
            f"[{zone.upper()}]  1=Guardian 2=Shadow 3=Berserker "
            f"SPACE={'SPEAK' if not speaking else 'STOP'}",
            True, _scale_c(main_c, 0.5))
        screen.blit(zone_text, (10, size - 25))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    _run_standalone()
