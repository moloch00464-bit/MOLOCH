#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Preview
===========================

Kamera-Preview Modul. Zeigt den Live-Stream im Canvas an.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Aufloesung waehlbar: 640x360, 800x450, 960x540, 1280x720
- 28ms Update-Intervall (35 FPS Ziel), NEAREST Resize
- Frame-Skip wenn Verarbeitung laenger als 28ms dauert
- BGR->RGB Konvertierung, Resize auf gewaehlte Preview-Groesse
- FPS-Zaehler oben rechts als gelbes Overlay
- "Kein Signal" bei fehlendem Frame
"""

import tkinter as tk
import struct
import time
import os

from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_INPUT, BG_DARK, FG_DIM, FG_TEXT, STATUS_YELLOW, FONT_MONO, FONT_SMALL,
    SHM_FRAME,
)

# Verfuegbare Aufloesungen (Label -> (width, height))
RESOLUTIONS = [
    ("640x360", 640, 360),
    ("800x450", 800, 450),
    ("960x540", 960, 540),
    ("1280x720", 1280, 720),
]

# Festes Update-Intervall: 28ms = ~35 FPS Ziel
UPDATE_INTERVAL_MS = 28


class PreviewModule:
    """Kamera-Preview im uebergebenen LabelFrame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main (frame_kamera)
            service_proxy: ServiceProxy Instanz fuer read_frame()
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._running = False
        self._after_id = None

        # Aktuelle Preview-Groesse (Standard: 640x360)
        self._preview_w = RESOLUTIONS[0][1]
        self._preview_h = RESOLUTIONS[0][2]

        # FPS-Zaehler
        self._frame_times = []
        self._fps = 0.0

        # Zeitstempel fuer Frame-Skip Logik
        self._last_update_start = 0.0

        # --- Aufloesung-Selector oberhalb des Canvas ---
        self._res_frame = tk.Frame(parent_frame, bg=BG_DARK)
        self._res_frame.pack(padx=5, pady=(5, 0), fill=tk.X)

        tk.Label(
            self._res_frame, text="Aufloesung:", bg=BG_DARK,
            fg=FG_TEXT, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=(0, 5))

        self._res_var = tk.StringVar(value=RESOLUTIONS[0][0])
        self._res_menu = tk.OptionMenu(
            self._res_frame, self._res_var,
            *[r[0] for r in RESOLUTIONS],
            command=self._on_resolution_changed,
        )
        self._res_menu.config(
            bg=BG_INPUT, fg=FG_TEXT, font=FONT_SMALL,
            highlightthickness=0, bd=1, relief=tk.FLAT,
        )
        self._res_menu["menu"].config(bg=BG_INPUT, fg=FG_TEXT, font=FONT_SMALL)
        self._res_menu.pack(side=tk.LEFT)

        # --- Canvas ---
        self._canvas = tk.Canvas(
            parent_frame,
            width=self._preview_w,
            height=self._preview_h,
            bg=BG_INPUT,
            highlightthickness=0,
        )
        self._canvas.pack(padx=5, pady=5)

        # Schwarzes Startbild
        self._photo = ImageTk.PhotoImage(
            Image.new('RGB', (self._preview_w, self._preview_h), (0, 0, 0))
        )

        # Canvas-Items (Reihenfolge = Z-Order)
        self._image_id = self._canvas.create_image(
            0, 0, anchor=tk.NW, image=self._photo
        )
        self._nosignal_id = self._canvas.create_text(
            self._preview_w // 2, self._preview_h // 2,
            text="Kein Signal",
            fill=FG_DIM,
            font=FONT_MONO,
        )
        self._fps_id = self._canvas.create_text(
            self._preview_w - 5, 5,
            anchor=tk.NE,
            text="0.0 FPS",
            fill=STATUS_YELLOW,
            font=FONT_MONO,
        )

    def _read_shm_frame(self):
        """
        Frame direkt aus SHM lesen mit korrektem Header-Parsing.

        SHM-Format (geschrieben von moloch_service._write_shm):
          16 Byte Header:
            h   (uint32 LE) = Bildhoehe (numpy rows)
            w   (uint32 LE) = Bildbreite (numpy cols)
            c   (uint32 LE) = Kanaele (3 = BGR)
            seq (uint32 LE) = Sequenznummer
          Danach: h * w * c Bytes BGR Pixeldaten

        Returns:
            (width, height, raw_bytes) oder None bei Fehler
        """
        try:
            if not os.path.exists(SHM_FRAME):
                return None
            with open(SHM_FRAME, "rb") as f:
                header = f.read(16)
                if len(header) < 16:
                    return None
                h, w, c, _seq = struct.unpack("<IIII", header)
                expected = w * h * c
                if expected == 0 or expected > 10_000_000:
                    return None
                raw = f.read(expected)
                if len(raw) < expected:
                    return None
            return (w, h, raw)
        except OSError:
            return None

    def _on_resolution_changed(self, selection):
        """Aufloesung gewechselt — Canvas und Overlay-Positionen anpassen."""
        for label, w, h in RESOLUTIONS:
            if label == selection:
                self._preview_w = w
                self._preview_h = h
                break

        # Canvas-Groesse anpassen
        self._canvas.config(width=self._preview_w, height=self._preview_h)

        # Overlay-Positionen neu setzen
        self._canvas.coords(
            self._nosignal_id,
            self._preview_w // 2, self._preview_h // 2,
        )
        self._canvas.coords(
            self._fps_id,
            self._preview_w - 5, 5,
        )

    def _update(self):
        """Einen Frame lesen, konvertieren und anzeigen."""
        if not self._running:
            return

        now_start = time.monotonic()

        # Frame-Skip: wenn letzter Update laenger als 28ms gedauert hat,
        # diesen Frame ueberspringen und direkt naechsten planen
        elapsed_since_last = (now_start - self._last_update_start) * 1000
        if self._last_update_start > 0 and elapsed_since_last < UPDATE_INTERVAL_MS * 0.5:
            # Zu frueh — ueberspringen
            self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)
            return

        self._last_update_start = now_start

        result = self._read_shm_frame()

        if result is not None:
            shm_w, shm_h, raw = result

            # Bild mit Originalgroesse aus SHM rekonstruieren (BGR raw)
            img = Image.frombytes('RGB', (shm_w, shm_h), raw)
            # B und R tauschen (BGR -> RGB)
            b, g, r = img.split()
            img = Image.merge('RGB', (r, g, b))

            # Auf gewaehlte Preview-Aufloesung resizen
            if img.size != (self._preview_w, self._preview_h):
                img = img.resize((self._preview_w, self._preview_h), Image.BILINEAR)

            # Anzeigen
            self._photo = ImageTk.PhotoImage(img)
            self._canvas.itemconfig(self._image_id, image=self._photo)
            self._canvas.itemconfig(self._nosignal_id, state='hidden')

            # FPS berechnen (Frames der letzten Sekunde zaehlen)
            now = time.monotonic()
            self._frame_times.append(now)
            cutoff = now - 1.0
            self._frame_times = [t for t in self._frame_times if t > cutoff]
            self._fps = len(self._frame_times)
        else:
            # Kein Frame — "Kein Signal" anzeigen
            self._canvas.itemconfig(self._nosignal_id, state='normal')
            self._fps = 0.0

        # FPS-Overlay aktualisieren und nach vorne
        self._canvas.itemconfig(self._fps_id, text=f"{self._fps:.1f} FPS")
        self._canvas.tag_raise(self._fps_id)

        # Frame-Skip Logik: wenn Verarbeitung > 28ms, naechsten Frame ueberspringen
        processing_time = (time.monotonic() - now_start) * 1000
        if processing_time > UPDATE_INTERVAL_MS:
            # Verarbeitung war zu lang — uebernaechstes Intervall planen
            self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)
        else:
            # Normal: restliche Zeit bis zum naechsten Intervall warten
            wait = max(1, UPDATE_INTERVAL_MS - int(processing_time))
            self._after_id = self._parent.after(wait, self._update)

    def start(self):
        """Preview-Loop starten."""
        if self._running:
            return
        self._running = True
        self._frame_times = []
        self._last_update_start = 0.0
        self._update()

    def stop(self):
        """Preview-Loop stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None
