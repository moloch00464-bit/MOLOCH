#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Preview
===========================

Kamera-Preview Modul. Zeigt den Live-Stream im Canvas an.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Aufloesung waehlbar: SD 640x360, HD 800x450, HD+ 960x540, Full (960 fit)
- Max Canvas-Groesse 960x540, groessere Aufloesungen werden eingepasst
- 33ms Update-Intervall (30 FPS Ziel), BILINEAR Resize
- Frame-Skip wenn Verarbeitung laenger als 28ms dauert
- SHM liefert RGB direkt (kein BGR-Umweg), Resize auf gewaehlte Preview-Groesse
- FPS-Zaehler oben rechts als gelbes Overlay
- "Kein Signal" bei fehlendem Frame
"""

import tkinter as tk
import struct
import time
import os
import logging

import numpy as np
from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_INPUT, BG_DARK, FG_DIM, FG_TEXT, STATUS_YELLOW, FONT_MONO, FONT_SMALL,
    SHM_FRAME,
)

# Verfuegbare Aufloesungen (Label, Resize-Breite, Resize-Hoehe)
RESOLUTIONS = [
    ("SD 640x360", 640, 360),
    ("HD 800x450", 800, 450),
    ("HD+ 960x540", 960, 540),
    ("Full (960 fit)", 1280, 720),
]

# Maximale Canvas-Groesse (Fenster darf nicht groesser als Bildschirm werden)
MAX_CANVAS_W = 960
MAX_CANVAS_H = 540

# Festes Update-Intervall: 33ms = 30 FPS Ziel (Gate1: Preview-Latenz unter 100ms)
UPDATE_INTERVAL_MS = 33


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

        # Resize-Ziel (Aufloesung aus Dropdown)
        self._resize_w = RESOLUTIONS[0][1]
        self._resize_h = RESOLUTIONS[0][2]

        # Canvas-Groesse (gekappt auf MAX_CANVAS)
        self._canvas_w = min(self._resize_w, MAX_CANVAS_W)
        self._canvas_h = min(self._resize_h, MAX_CANVAS_H)

        # FPS-Zaehler
        self._frame_times = []
        self._fps = 0.0

        # SHM Sequenznummer — gleicher Frame = kein Neuzeichnen
        self._last_seq = -1

        # Fehler-Zaehler fuer Diagnostik
        self._error_count = 0
        self._logger = logging.getLogger("Preview")

        # Zeitstempel fuer Frame-Skip Logik
        self._last_update_start = 0.0

        # Watchdog: Letzter erfolgreicher Render (Gate0 Phase 9)
        self.last_render_time = 0.0

        # Lag-Diagnostik: 30s Logging, alle 2s ein Eintrag
        self._lag_log_start = 0.0
        self._lag_log_next = 0.0
        self._lag_log_active = False

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
            width=self._canvas_w,
            height=self._canvas_h,
            bg=BG_INPUT,
            highlightthickness=0,
        )
        self._canvas.pack(padx=5, pady=5)

        # Schwarzes Startbild
        self._photo = ImageTk.PhotoImage(
            Image.new('RGB', (self._canvas_w, self._canvas_h), (0, 0, 0))
        )

        # Canvas-Items (Reihenfolge = Z-Order)
        self._image_id = self._canvas.create_image(
            0, 0, anchor=tk.NW, image=self._photo
        )
        self._nosignal_id = self._canvas.create_text(
            self._canvas_w // 2, self._canvas_h // 2,
            text="Kein Signal",
            fill=FG_DIM,
            font=FONT_MONO,
        )
        self._fps_id = self._canvas.create_text(
            self._canvas_w - 5, 5,
            anchor=tk.NE,
            text="0.0 FPS",
            fill=STATUS_YELLOW,
            font=FONT_MONO,
        )

    def _read_shm_frame(self):
        """
        Frame direkt aus SHM lesen mit korrektem Header-Parsing.

        SHM-Format (geschrieben von tappas_pipeline._write_shm_frame):
          24 Byte Header:
            h   (uint32 LE) = Bildhoehe (numpy rows)
            w   (uint32 LE) = Bildbreite (numpy cols)
            c   (uint32 LE) = Kanaele (3 = RGB)
            seq (uint32 LE) = Sequenznummer
            ts  (float64 LE) = time.monotonic() Zeitstempel
          Danach: h * w * c Bytes RGB Pixeldaten

        Returns:
            (width, height, seq, ts, raw_bytes) oder None bei Fehler
        """
        try:
            if not os.path.exists(SHM_FRAME):
                return None
            with open(SHM_FRAME, "rb") as f:
                header = f.read(24)
                if len(header) >= 24:
                    # Neuer 24-Byte Header (mit Timestamp)
                    h, w, c, seq, ts = struct.unpack("<IIIId", header)
                elif len(header) >= 16:
                    # Alter 16-Byte Header (ohne Timestamp) — Rueckwaertskompatibel
                    h, w, c, seq = struct.unpack("<IIII", header[:16])
                    ts = 0.0
                else:
                    return None
                expected = w * h * c
                if expected == 0 or expected > 10_000_000:
                    return None
                raw = f.read(expected)
                if len(raw) < expected:
                    return None
            return (w, h, seq, ts, raw)
        except OSError:
            return None

    def _on_resolution_changed(self, selection):
        """Aufloesung gewechselt — Canvas und Overlay-Positionen anpassen."""
        for label, w, h in RESOLUTIONS:
            if label == selection:
                self._resize_w = w
                self._resize_h = h
                break

        # Canvas-Groesse gekappt auf Maximum
        self._canvas_w = min(self._resize_w, MAX_CANVAS_W)
        self._canvas_h = min(self._resize_h, MAX_CANVAS_H)

        # Canvas-Groesse anpassen
        self._canvas.config(width=self._canvas_w, height=self._canvas_h)

        # Overlay-Positionen neu setzen
        self._canvas.coords(
            self._nosignal_id,
            self._canvas_w // 2, self._canvas_h // 2,
        )
        self._canvas.coords(
            self._fps_id,
            self._canvas_w - 5, 5,
        )

    def _update(self):
        """Einen Frame lesen, konvertieren und anzeigen.

        KRITISCH: try/except um den GESAMTEN Body. Wenn hier eine Exception
        durchrutscht, stirbt der after()-Chain und das Bild friert ein.
        """
        if not self._running:
            return

        try:
            now_start = time.monotonic()

            # Frame-Skip: zu frueh seit letztem Update
            elapsed_since_last = (now_start - self._last_update_start) * 1000
            if self._last_update_start > 0 and elapsed_since_last < UPDATE_INTERVAL_MS * 0.5:
                self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)
                return

            self._last_update_start = now_start

            result = self._read_shm_frame()

            if result is not None:
                shm_w, shm_h, seq, shm_ts, raw = result

                # Gleicher Frame wie letztes Mal — kein Neuzeichnen noetig
                if seq == self._last_seq:
                    self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)
                    return
                self._last_seq = seq

                # Lag-Diagnostik: Latenz SHM-Write → Preview-Read
                now_mono = time.monotonic()
                lag_ms = (now_mono - shm_ts) * 1000.0 if shm_ts > 0 else -1.0

                if self._lag_log_active and now_mono < self._lag_log_start + 30.0:
                    if now_mono >= self._lag_log_next:
                        self._logger.info(
                            f"[LAG] seq={seq} lag={lag_ms:.1f}ms fps={self._fps:.1f}"
                        )
                        self._lag_log_next = now_mono + 2.0
                elif self._lag_log_active:
                    self._lag_log_active = False
                    self._logger.info("[LAG] 30s Lag-Logging beendet")

                # SHM ist bereits RGB (TAPPAS schreibt RGB direkt)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((shm_h, shm_w, 3))
                img = Image.fromarray(arr)

                # Auf Canvas-Groesse resizen (gekappt auf MAX_CANVAS)
                if img.size != (self._canvas_w, self._canvas_h):
                    img = img.resize((self._canvas_w, self._canvas_h), Image.BILINEAR)

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

                # Fehler-Zaehler zuruecksetzen nach erfolgreichem Frame
                self._error_count = 0
                self.last_render_time = time.monotonic()
            else:
                # Kein Frame — "Kein Signal" anzeigen
                self._canvas.itemconfig(self._nosignal_id, state='normal')
                self._fps = 0.0

            # FPS-Overlay aktualisieren und nach vorne
            self._canvas.itemconfig(self._fps_id, text=f"{self._fps:.1f} FPS")
            self._canvas.tag_raise(self._fps_id)

            # Naechsten Frame planen
            processing_time = (time.monotonic() - now_start) * 1000
            if processing_time > UPDATE_INTERVAL_MS:
                self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)
            else:
                wait = max(1, UPDATE_INTERVAL_MS - int(processing_time))
                self._after_id = self._parent.after(wait, self._update)

        except Exception as e:
            # KRITISCH: after-Chain MUSS weiterlaufen, sonst Bild-Freeze!
            self._error_count += 1
            if self._error_count <= 3 or self._error_count % 100 == 0:
                self._logger.warning(
                    f"[WARNUNG] Preview frame_error count={self._error_count} err={e}"
                )
            self._after_id = self._parent.after(UPDATE_INTERVAL_MS, self._update)

    def start(self):
        """Preview-Loop starten. Startet 30s Lag-Diagnostik."""
        if self._running:
            return
        self._running = True
        self._frame_times = []
        self._last_update_start = 0.0
        # Lag-Logging: 30 Sekunden, alle 2 Sekunden ein Eintrag
        now = time.monotonic()
        self._lag_log_start = now
        self._lag_log_next = now
        self._lag_log_active = True
        self._logger.info("[LAG] 30s Lag-Logging gestartet")
        self._update()

    def stop(self):
        """Preview-Loop stoppen."""
        self._running = False
        if self._after_id is not None:
            self._parent.after_cancel(self._after_id)
            self._after_id = None
