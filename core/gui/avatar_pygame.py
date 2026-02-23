#!/usr/bin/env python3
"""
M.O.L.O.C.H. Avatar-Auge — PyGame Edition
============================================

Hardwarebeschleunigtes HAL-9000 Auge mit echtem Alpha-Blending.
Standalone-Prozess, liest State aus /dev/shm/moloch_status.json.

Blueprint 5.2/5.3 Mappings:
  TENSION   -> Puls-Geschwindigkeit + Glow-Intensitaet + Pupillengroesse
  DOMINANCE -> Farbe (Blau +1 <-> Weiss 0 <-> Rot -1)
  CPU-TEMP  -> Puls verlangsamt sich bei Hitze, Auge wird "muede"

Steuerung:
  ESC / Q   -> Beenden
  F         -> Fullscreen Toggle
  R         -> Fenster-Position Reset

Standalone: python3 core/gui/avatar_pygame.py
"""

import pygame
import pygame.gfxdraw
import json
import math
import time
import random
import os
import sys

# =============================================================================
# Konstanten
# =============================================================================

EYE_SIZE = 400
INFO_HEIGHT = 80
WIDTH = EYE_SIZE
HEIGHT = EYE_SIZE + INFO_HEIGHT
CX = EYE_SIZE // 2
CY = EYE_SIZE // 2
FPS = 60

STATUS_PATH = "/dev/shm/moloch_status.json"

# Hintergrund
BG = (10, 10, 20)
GRID = (17, 17, 32)
INFO_BG = (12, 12, 24)

# Zone-Farben (RGB)
COL_GUARDIAN = (0, 170, 255)      # Leuchtendes Blau
COL_NEUTRAL = (255, 255, 255)     # Weiss
COL_SHADOW = (255, 34, 0)        # Intensives Rot
COL_BERSERKER = (255, 34, 0)

# Hysterese
DOM_HYSTERESIS = 0.15


# =============================================================================
# Farb-Hilfsfunktionen
# =============================================================================

def lerp_color(c1, c2, t):
    """Lineare Interpolation zwischen zwei RGB-Tupeln."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def scale_color(color, factor):
    """Farbe um Helligkeitsfaktor skalieren."""
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def dominance_color(d):
    """Dominance-Wert (-1 bis +1) in Hauptfarbe umrechnen."""
    if d >= 0:
        return lerp_color(COL_NEUTRAL, COL_GUARDIAN, d)
    return lerp_color(COL_NEUTRAL, COL_SHADOW, -d)


def dominance_iris_color(d):
    """Iris-Farbe aus Dominance ableiten (heller als Hauptfarbe)."""
    if d >= 0:
        return lerp_color((220, 220, 220), (68, 204, 255), d)
    return lerp_color((220, 220, 220), (255, 68, 51), -d)


# =============================================================================
# Glow-Renderer
# =============================================================================

def render_glow(radius, color, intensity):
    """Weichen Glow als Alpha-Surface rendern.

    Quadratischer Alpha-Falloff von Zentrum nach aussen.
    Das ist der Hauptvorteil gegenueber Tkinter.
    """
    size = radius * 2 + 4
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2
    step = max(3, radius // 25)  # ~25 Ringe: Performance auf Pi5
    for r in range(radius, 0, -step):
        t = r / radius
        # Quadratischer Falloff: hell im Zentrum, sanft am Rand
        alpha = int(intensity * 255 * (1.0 - t) * (1.0 - t) * 0.15)
        alpha = min(255, max(0, alpha))
        if alpha > 0:
            pygame.gfxdraw.filled_circle(surf, cx, cy, r, (*color, alpha))
    return surf


# =============================================================================
# MolochEye — Haupt-Klasse
# =============================================================================

class MolochEye:
    """PyGame-basiertes M.O.L.O.C.H. Avatar-Auge."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("M.O.L.O.C.H.")
        self.clock = pygame.time.Clock()
        self._fullscreen = False

        # Fonts
        self.font_zone = pygame.font.SysFont("Helvetica", 18, bold=True)
        self.font_info = pygame.font.SysFont("Helvetica", 14)
        self.font_small = pygame.font.SysFont("Helvetica", 12)

        # === State (Zielwerte aus Status-Datei) ===
        self.tension = 0.0
        self.dominance = 0.5
        self.cpu_temp = 0.0
        self.cpu_temp_celsius = 0.0
        self.zone = "guardian"
        self.berserker_active = False
        self.status_text = "Idle"

        # === Smooth Interpolation (~300ms Settling bei 60 FPS) ===
        self.s_tension = 0.0
        self.s_dominance = 0.5
        self.s_cpu = 0.0

        # === Visuelle Dominance mit Hysterese ===
        self.visual_dom = 0.5
        self.dom_at_last_change = 0.5

        # === Animation State ===
        self.tick = 0
        self.pulse_phase = 0.0
        self.blink_progress = 0.0
        self.blinking = False
        self.blink_opening = False
        self.next_blink_tick = random.randint(180, 420)
        self.flash_until = 0.0
        self.pupil_dx = 0.0
        self.pupil_dy = 0.0
        self.target_dx = 0.0
        self.target_dy = 0.0

        # === Glow Surface Cache ===
        self._glow_surface = None
        self._glow_color_key = None

    # =========================================================================
    # Status lesen
    # =========================================================================

    def read_status(self):
        """State aus Shared Memory lesen (/dev/shm/moloch_status.json)."""
        try:
            with open(STATUS_PATH) as f:
                status = json.load(f)
        except Exception:
            return

        core = status.get("core", {})
        if core:
            old_zone = self.zone
            self.tension = float(core.get("tension", 0.0))
            self.dominance = float(core.get("dominance", 0.5))
            self.cpu_temp = float(core.get("cpu_temp_norm", 0.0))
            self.cpu_temp_celsius = float(core.get("cpu_temp", 0.0))
            self.zone = core.get("zone", "guardian")
            self.berserker_active = core.get("berserker_active", False)

            # Berserker Flash bei Eintritt
            if self.zone == "berserker" and old_zone != "berserker":
                self.flash_until = time.monotonic() + 0.3

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
                    self.status_text = f"{names} erkannt"
                else:
                    self.status_text = f"{len(faces)} Gesicht(er)"
            elif persons:
                self.status_text = f"{persons} Person(en)"
            elif self.tension > 0.1:
                self.status_text = "Suche..."
            else:
                self.status_text = "Idle"

    # =========================================================================
    # Update (Animation)
    # =========================================================================

    def update(self, dt):
        """Animation-State aktualisieren. dt in Sekunden."""
        self.tick += 1

        # Kubische Interpolation (~300ms Settling)
        # Bei 60 FPS: rate=6*dt ≈ 0.1/Frame, 95% nach ~30 Frames = 500ms
        rate = min(1.0, dt * 6.0)
        self.s_tension += (self.tension - self.s_tension) * rate
        self.s_dominance += (self.dominance - self.s_dominance) * rate
        self.s_cpu += (self.cpu_temp - self.s_cpu) * rate

        # Visuelle Dominance mit Hysterese
        if abs(self.dominance - self.dom_at_last_change) > DOM_HYSTERESIS:
            self.dom_at_last_change = self.dominance
        self.visual_dom += (self.dom_at_last_change - self.visual_dom) * rate

        # Puls-Phase: Geschwindigkeit von Tension, gedaempft bei CPU-Hitze
        speed = 1.0 + self.s_tension * 3.0
        if self.s_cpu > 0.7:
            speed *= 0.85
        if self.s_cpu > 0.9:
            speed *= 0.6
        self.pulse_phase += speed * dt * math.tau / 3.0  # ~1 Zyklus pro 3s

        # === Blinzel-Logik ===
        blink_speed = 4.0 * dt  # ~4 pro Sekunde = 250ms Blinzeln
        if self.blinking:
            if not self.blink_opening:
                self.blink_progress += blink_speed
                if self.blink_progress >= 1.0:
                    self.blink_progress = 1.0
                    self.blink_opening = True
            else:
                self.blink_progress -= blink_speed
                if self.blink_progress <= 0.0:
                    self.blink_progress = 0.0
                    self.blinking = False
                    self.blink_opening = False
                    self.next_blink_tick = self.tick + random.randint(180, 420)
        elif self.tick >= self.next_blink_tick:
            # Bei hoher Tension seltener blinzeln
            if self.s_tension > 0.7 and random.random() < 0.5:
                self.next_blink_tick = self.tick + random.randint(180, 420)
            else:
                self.blinking = True
                self.blink_opening = False

        # === Mikro-Bewegung (Pupille) ===
        if self.tick % 90 == 0:
            self.target_dx = random.uniform(-5, 5)
            self.target_dy = random.uniform(-5, 5)
        self.pupil_dx += (self.target_dx - self.pupil_dx) * 0.03
        self.pupil_dy += (self.target_dy - self.pupil_dy) * 0.03

    # =========================================================================
    # Zeichnen
    # =========================================================================

    def draw(self):
        """Einen kompletten Frame rendern."""
        self.screen.fill(BG)

        now = time.monotonic()
        flash = now < self.flash_until

        # === Farben bestimmen ===
        if flash:
            main_c = (255, 68, 68)
            iris_c = (255, 170, 170)
            brightness = 1.0
        elif self.zone == "berserker":
            main_c = COL_BERSERKER
            iris_c = (255, 68, 51)
            brightness = 0.6 + self.s_tension * 0.4
        else:
            main_c = dominance_color(self.visual_dom)
            iris_c = dominance_iris_color(self.visual_dom)
            brightness = 0.5 + self.s_tension * 0.5

        pulse = 1.0 + math.sin(self.pulse_phase) * 0.025
        if self.zone == "berserker":
            pulse = 1.0 + math.sin(self.pulse_phase) * 0.07

        # === Grid ===
        for x in range(0, EYE_SIZE, 25):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, EYE_SIZE))
        for y in range(0, EYE_SIZE, 25):
            pygame.draw.line(self.screen, GRID, (0, y), (EYE_SIZE, y))

        # === Glow (Alpha-Blending — DER Unterschied zu Tkinter) ===
        glow_str = brightness * (0.5 + self.s_tension * 0.5)
        glow_r = int(170 * pulse)
        # Glow-Surface cachen (grober Key = weniger Regenerationen)
        color_key = (main_c, int(glow_str * 4), glow_r // 8)
        if color_key != self._glow_color_key:
            self._glow_surface = render_glow(glow_r, main_c, glow_str)
            self._glow_color_key = color_key
        if self._glow_surface:
            gr = self._glow_surface.get_rect(center=(CX, CY))
            self.screen.blit(self._glow_surface, gr)

        # === Aeusserer Ring (Anti-Aliased, dick) ===
        ro = int(120 * pulse)
        rc = scale_color(main_c, brightness * 0.8)
        rc_dim = scale_color(main_c, brightness * 0.4)
        # 3-Pixel Ring fuer dicken Look
        for dr in (-1, 0, 1):
            r = ro + dr
            if r > 2:
                c = rc if dr == 0 else rc_dim
                pygame.gfxdraw.aacircle(self.screen, CX, CY, r, c)

        # === 12 Tick-Marks ===
        for i in range(12):
            angle = math.radians(i * 30 - 90)
            tl = 14 if i % 3 == 0 else 7
            tw = 2 if i % 3 == 0 else 1
            x1 = CX + math.cos(angle) * (ro - tl)
            y1 = CY + math.sin(angle) * (ro - tl)
            x2 = CX + math.cos(angle) * (ro + 6)
            y2 = CY + math.sin(angle) * (ro + 6)
            tc = scale_color(main_c, brightness * 0.6)
            pygame.draw.line(self.screen, tc,
                             (int(x1), int(y1)), (int(x2), int(y2)), tw)

        # === Innerer Ring (pulsiert staerker) ===
        ip = 1.0 + math.sin(self.pulse_phase * 1.5) * 0.025 * (1 + self.s_tension * 2)
        ri = int(95 * ip)
        ic = scale_color(main_c, brightness * 0.55)
        pygame.gfxdraw.aacircle(self.screen, CX, CY, ri, ic)

        # === Iris ===
        ir = int(65 * (0.8 + brightness * 0.2))
        ix = int(CX + self.pupil_dx)
        iy = int(CY + self.pupil_dy)

        # Iris Gradient (mehrere Kreise von aussen nach innen)
        for r in range(ir, max(ir - 10, 0), -1):
            t = (ir - r) / 10.0
            c = lerp_color(scale_color(iris_c, brightness * 0.35),
                           scale_color(main_c, brightness * 0.65), t)
            pygame.gfxdraw.filled_circle(self.screen, ix, iy, r, c)
        # Iris Rand (Anti-Aliased)
        pygame.gfxdraw.aacircle(self.screen, ix, iy, ir,
                                scale_color(main_c, brightness * 0.7))

        # === 16 radiale Strahlen ===
        pupil_ratio = 0.25 + self.s_tension * 0.35
        pupil_inner = ir * pupil_ratio + 3
        ray_c = scale_color(iris_c, brightness * 0.25)
        for j in range(16):
            a = math.radians(j * 22.5 + self.tick * 0.12)
            r_in = pupil_inner
            r_out = ir - 3
            if r_in < r_out:
                rx1 = ix + math.cos(a) * r_in
                ry1 = iy + math.sin(a) * r_in
                rx2 = ix + math.cos(a) * r_out
                ry2 = iy + math.sin(a) * r_out
                pygame.draw.aaline(self.screen, ray_c,
                                   (rx1, ry1), (rx2, ry2))

        # === Pupille ===
        pr = int(ir * pupil_ratio)
        if flash:
            pr = int(ir * 0.12)
        pygame.gfxdraw.filled_circle(self.screen, ix, iy, pr, (0, 0, 0))
        pygame.gfxdraw.aacircle(self.screen, ix, iy, pr, (0, 0, 0))

        # Hauptreflex (oben rechts)
        hr = max(3, pr // 3)
        hx = int(ix + pr * 0.3)
        hy = int(iy - pr * 0.35)
        hc = scale_color((255, 255, 255), brightness * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, hx, hy, hr, hc)
        pygame.gfxdraw.aacircle(self.screen, hx, hy, hr, hc)

        # Kleiner Reflex (unten links)
        h2r = max(2, pr // 5)
        h2x = int(ix - pr * 0.2)
        h2y = int(iy + pr * 0.25)
        h2c = scale_color((255, 255, 255), brightness * 0.35)
        pygame.gfxdraw.filled_circle(self.screen, h2x, h2y, h2r, h2c)

        # === Augenlider ===
        if self.blink_progress > 0.01:
            self._draw_lids(brightness, main_c)
        elif self.s_cpu > 0.85:
            self._draw_droopy(brightness, main_c)

        # === Info-Bar ===
        self._draw_info(main_c, brightness)

        pygame.display.flip()

    # =========================================================================
    # Lider
    # =========================================================================

    def _draw_lids(self, brightness, color):
        """Blinzel-Lider mit geschwungener Kante."""
        h = int(EYE_SIZE * 0.5 * self.blink_progress)
        if h < 2:
            return

        # Oberes Lid
        pygame.draw.rect(self.screen, BG, (0, 0, EYE_SIZE, h))
        # Unteres Lid
        pygame.draw.rect(self.screen, BG, (0, EYE_SIZE - h, EYE_SIZE, h))

        # Geschwungene Leucht-Kanten
        edge_c = scale_color(color, brightness * 0.5)
        curve = min(12, h // 3)

        top_pts = []
        bot_pts = []
        for px in range(0, EYE_SIZE + 1, 3):
            dx = (px - CX) / CX
            c_off = int(curve * (1.0 - dx * dx))
            top_pts.append((px, h + c_off))
            bot_pts.append((px, (EYE_SIZE - h) - c_off))

        if len(top_pts) > 1:
            pygame.draw.lines(self.screen, edge_c, False, top_pts, 2)
        if len(bot_pts) > 1:
            pygame.draw.lines(self.screen, edge_c, False, bot_pts, 2)

    def _draw_droopy(self, brightness, color):
        """Halb-geschlossene Lider bei hoher CPU-Temperatur."""
        droop_factor = (self.s_cpu - 0.85) / 0.15
        droop = int(max(0, droop_factor) * EYE_SIZE * 0.22)
        if droop < 3:
            return
        pygame.draw.rect(self.screen, BG, (0, 0, EYE_SIZE, droop))
        pygame.draw.rect(self.screen, BG, (0, EYE_SIZE - droop, EYE_SIZE, droop))
        edge = scale_color(color, brightness * 0.3)
        pygame.draw.line(self.screen, edge, (0, droop), (EYE_SIZE, droop), 1)
        pygame.draw.line(self.screen, edge,
                         (0, EYE_SIZE - droop), (EYE_SIZE, EYE_SIZE - droop), 1)

    # =========================================================================
    # Info-Bar
    # =========================================================================

    def _draw_info(self, color, brightness):
        """Info-Bereich unter dem Auge: Zone, T/D, CPU, Status."""
        y0 = EYE_SIZE
        pygame.draw.rect(self.screen, INFO_BG, (0, y0, WIDTH, INFO_HEIGHT))

        # Zeile 1: Zone in Farbe
        zone_name = {"guardian": "GUARDIAN", "shadow": "SHADOW",
                     "berserker": "BERSERKER"}.get(self.zone, "OFFLINE")
        zone_c = scale_color(color, max(0.6, brightness))
        zone_surf = self.font_zone.render(zone_name, True, zone_c)
        zone_r = zone_surf.get_rect(centerx=WIDTH // 2, top=y0 + 6)
        self.screen.blit(zone_surf, zone_r)

        # Zeile 2: T + D Werte
        td = f"T:{self.tension:.2f} | D:{self.dominance:+.2f}"
        td_c = scale_color(color, brightness * 0.7)
        td_surf = self.font_info.render(td, True, td_c)
        td_r = td_surf.get_rect(centerx=WIDTH // 2, top=y0 + 30)
        self.screen.blit(td_surf, td_r)

        # Zeile 3: CPU + Status
        parts = []
        if self.cpu_temp_celsius > 0:
            parts.append(f"CPU: {self.cpu_temp_celsius:.0f}\u00b0C")
        parts.append(self.status_text)
        status = " | ".join(parts)
        st_c = scale_color(color, brightness * 0.5)
        st_surf = self.font_small.render(status, True, st_c)
        st_r = st_surf.get_rect(centerx=WIDTH // 2, top=y0 + 52)
        self.screen.blit(st_surf, st_r)

    # =========================================================================
    # Hauptschleife
    # =========================================================================

    def run(self):
        """Hauptschleife: 60 FPS, liest Status alle 500ms."""
        running = True
        last_status_read = 0.0

        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_f:
                        self._fullscreen = not self._fullscreen
                        if self._fullscreen:
                            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

            # Status alle 500ms lesen
            now = time.monotonic()
            if now - last_status_read > 0.5:
                self.read_status()
                last_status_read = now

            self.update(dt)
            self.draw()

        pygame.quit()


# =============================================================================
# Standalone-Start
# =============================================================================

if __name__ == "__main__":
    # Moloch Pfad sicherstellen
    moloch_path = os.path.expanduser("~/moloch")
    if moloch_path not in sys.path:
        sys.path.insert(0, moloch_path)

    eye = MolochEye()
    eye.run()
