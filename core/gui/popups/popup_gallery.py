#!/usr/bin/env python3
"""
M.O.L.O.C.H. Snapshot Galerie Popup
=====================================

Eigenstaendiges Toplevel-Fenster fuer Snapshot-Verwaltung.
Liest JPGs aus ~/moloch/snapshots/, zeigt Thumbnail-Grid.

Features:
- Thumbnail-Grid (3 Spalten, 150x112px)
- Klick auf Thumbnail oeffnet Vollbild
- Dateiname + Datum unter jedem Thumbnail
- Refresh-Button
- Loeschen-Button pro Bild (mit Bestaetigung)

Importiert NUR panel_styles und tkinter.
"""

import logging
import os
import time
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON,
    BTN_ALARM_RED, BTN_OFF_DARK, BTN_SNAP_CYAN,
    ACCENT_CYAN,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

logger = logging.getLogger("moloch.popup_gallery")

SNAP_DIR = os.path.expanduser("~/moloch/snapshots")
THUMB_W = 150
THUMB_H = 112
COLUMNS = 3


class SnapshotGallery:
    """Snapshot Galerie als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent):
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
        """
        self.parent = parent
        self._thumb_refs = []  # ImageTk Referenzen halten (GC-Schutz)

        # Verzeichnis sicherstellen
        os.makedirs(SNAP_DIR, exist_ok=True)

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.title("Snapshot Galerie")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("540x600")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Header mit Refresh-Button
        header = tk.Frame(self.win, bg=BG_DARK)
        header.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            header, text="Snapshots", bg=BG_DARK, fg=FG_WHITE,
            font=FONT_TITLE,
        ).pack(side=tk.LEFT, padx=5)

        self._lbl_count = tk.Label(
            header, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_count.pack(side=tk.LEFT, padx=10)

        tk.Button(
            header, text="REFRESH", width=10,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._load_snapshots,
        ).pack(side=tk.RIGHT, padx=5)

        # Scrollbarer Bereich
        container = tk.Frame(self.win, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._grid_frame = tk.Frame(self._canvas, bg=BG_DARK)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._grid_frame, anchor=tk.NW
        )

        self._grid_frame.bind("<Configure>", self._on_grid_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        # Mausrad-Scrolling
        self._canvas.bind_all("<Button-4>", lambda e: self._canvas.yview_scroll(-3, "units"))
        self._canvas.bind_all("<Button-5>", lambda e: self._canvas.yview_scroll(3, "units"))

        # Snapshots laden
        self._load_snapshots()

    def _on_grid_configure(self, event):
        """Scroll-Region aktualisieren wenn Grid sich aendert."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Grid-Frame Breite an Canvas anpassen."""
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _load_snapshots(self):
        """Snapshots aus Verzeichnis laden und Grid aufbauen."""
        # Alte Widgets entfernen
        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        # JPGs sammeln (neueste zuerst)
        files = []
        if os.path.isdir(SNAP_DIR):
            for f in os.listdir(SNAP_DIR):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(SNAP_DIR, f)
                    mtime = os.path.getmtime(path)
                    files.append((path, f, mtime))
        files.sort(key=lambda x: x[2], reverse=True)

        self._lbl_count.config(text=f"{len(files)} Bilder")

        if not files:
            tk.Label(
                self._grid_frame, text="Keine Snapshots vorhanden",
                bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=COLUMNS, pady=40)
            return

        # Thumbnail-Grid aufbauen
        for idx, (path, fname, mtime) in enumerate(files):
            row = idx // COLUMNS
            col = idx % COLUMNS
            self._build_thumbnail(row, col, path, fname, mtime)

    def _build_thumbnail(self, row, col, path, fname, mtime):
        """Einzelnes Thumbnail mit Label und Loeschen-Button."""
        cell = tk.Frame(self._grid_frame, bg=BG_FRAME, padx=3, pady=3)
        cell.grid(row=row, column=col, padx=4, pady=4, sticky=tk.N)

        # Thumbnail laden
        try:
            img = Image.open(path)
            img.thumbnail((THUMB_W, THUMB_H))
            photo = ImageTk.PhotoImage(img)
            self._thumb_refs.append(photo)

            lbl_img = tk.Label(cell, image=photo, bg=BG_FRAME, cursor="hand2")
            lbl_img.pack(padx=2, pady=2)
            lbl_img.bind("<Button-1>", lambda e, p=path: self._open_fullsize(p))
        except Exception as e:
            logger.warning(f"Thumbnail Fehler fuer {fname}: {e}")
            tk.Label(
                cell, text="[Fehler]", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
                width=20, height=7,
            ).pack(padx=2, pady=2)

        # Dateiname
        tk.Label(
            cell, text=fname, bg=BG_FRAME, fg=FG_LABEL,
            font=FONT_SMALL, wraplength=THUMB_W,
        ).pack()

        # Datum
        date_str = time.strftime("%d.%m.%Y %H:%M", time.localtime(mtime))
        tk.Label(
            cell, text=date_str, bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack()

        # Loeschen-Button
        tk.Button(
            cell, text="X", width=3,
            bg=BTN_OFF_DARK, fg=BTN_ALARM_RED, font=FONT_SMALL,
            activebackground=BG_FRAME,
            command=lambda p=path, f=fname: self._delete_snapshot(p, f),
        ).pack(pady=(2, 0))

    def _open_fullsize(self, path):
        """Bild in voller Groesse in neuem Toplevel oeffnen."""
        try:
            img = Image.open(path)
        except Exception as e:
            logger.error(f"Bild oeffnen fehlgeschlagen: {e}")
            return

        full_win = tk.Toplevel(self.win)
        full_win.title(os.path.basename(path))
        full_win.configure(bg=BG_DARK)

        # Bildgroesse begrenzen auf 1024x768 max
        max_w, max_h = 1024, 768
        w, h = img.size
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        # Referenz halten
        full_win._photo_ref = photo

        full_win.geometry(f"{img.size[0]}x{img.size[1]}")
        full_win.resizable(False, False)

        tk.Label(full_win, image=photo, bg=BG_DARK).pack()

    def _delete_snapshot(self, path, fname):
        """Snapshot loeschen mit Bestaetigung."""
        confirm = messagebox.askyesno(
            "Snapshot loeschen",
            f"Wirklich loeschen?\n{fname}",
            parent=self.win,
        )
        if not confirm:
            return
        try:
            os.remove(path)
            logger.info(f"Snapshot geloescht: {fname}")
        except Exception as e:
            logger.error(f"Loeschen fehlgeschlagen: {e}")
            return
        # Grid neu laden
        self._load_snapshots()

    def _on_close(self):
        """Fenster schliessen, Mausrad-Bindings entfernen."""
        try:
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        self._thumb_refs.clear()
        self.win.destroy()
