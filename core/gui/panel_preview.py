#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Preview
===========================

Kamera-Preview Modul. Zeigt den Live-Stream im Canvas an.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Canvas 640x360, 15 FPS Update-Loop
- BGR->RGB Konvertierung, Resize auf Preview-Groesse
- FPS-Zaehler oben rechts als gelbes Overlay
- "Kein Signal" bei fehlendem Frame
"""

import tkinter as tk
import time

from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_INPUT, FG_DIM, STATUS_YELLOW, FONT_MONO,
    PREVIEW_W, PREVIEW_H, PREVIEW_FPS,
)


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

        # Update-Intervall: 1000ms / 15 FPS = 66ms
        self._interval_ms = 1000 // PREVIEW_FPS

        # FPS-Zaehler
        self._frame_times = []
        self._fps = 0.0

        # Canvas erstellen
        self._canvas = tk.Canvas(
            parent_frame,
            width=PREVIEW_W,
            height=PREVIEW_H,
            bg=BG_INPUT,
            highlightthickness=0,
        )
        self._canvas.pack(padx=5, pady=5)

        # Schwarzes Startbild
        self._photo = ImageTk.PhotoImage(
            Image.new('RGB', (PREVIEW_W, PREVIEW_H), (0, 0, 0))
        )

        # Canvas-Items (Reihenfolge = Z-Order)
        self._image_id = self._canvas.create_image(
            0, 0, anchor=tk.NW, image=self._photo
        )
        self._nosignal_id = self._canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="Kein Signal",
            fill=FG_DIM,
            font=FONT_MONO,
        )
        self._fps_id = self._canvas.create_text(
            PREVIEW_W - 5, 5,
            anchor=tk.NE,
            text="0.0 FPS",
            fill=STATUS_YELLOW,
            font=FONT_MONO,
        )

    def _update(self):
        """Einen Frame lesen, konvertieren und anzeigen."""
        if not self._running:
            return

        result = self._service.read_frame()

        if result is not None:
            width, height, channels, raw = result

            # BGR raw -> PIL Image (wird als RGB interpretiert, Kanaele sind aber BGR)
            img = Image.frombytes('RGB', (width, height), raw)
            # B und R tauschen
            b, g, r = img.split()
            img = Image.merge('RGB', (r, g, b))

            # Resize falls noetig
            if img.size != (PREVIEW_W, PREVIEW_H):
                img = img.resize((PREVIEW_W, PREVIEW_H), Image.BILINEAR)

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

        # Naechsten Update planen
        self._after_id = self._parent.after(self._interval_ms, self._update)

    def start(self):
        """Preview-Loop starten."""
        if self._running:
            return
        self._running = True
        self._frame_times = []
        self._update()

    def stop(self):
        """Preview-Loop stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None
