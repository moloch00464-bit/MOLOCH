#!/usr/bin/env python3
"""
M.O.L.O.C.H. Snapshot Galerie Popup
=====================================

Eigenstaendiges Toplevel-Fenster mit 2 Tabs (ttk.Notebook):
  Tab 1: Snapshots — ~/moloch/snapshots/ (manuelle Snap-Button Fotos)
  Tab 2: Teachen  — /mnt/moloch-data/daily/<YYYY-MM-DD>/ (DailyLearner)

Features:
- Thumbnail-Grid (3 Spalten, 150x112px)
- Klick auf Thumbnail oeffnet Vollbild
- Dateiname + Datum unter jedem Thumbnail
- Refresh-Button pro Tab
- Loeschen-Button pro Bild (mit Bestaetigung)
- Scrollbar bei vielen Bildern (Canvas + Scrollbar Pattern)
- Thumbnails in Background-Thread laden (kein GUI-Freeze)

Importiert NUR panel_styles und tkinter. KEIN Import von moloch_service.
"""

import json
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON,
    BTN_ALARM_RED, BTN_OFF_DARK, BTN_SNAP_CYAN,
    ACCENT_CYAN,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

logger = logging.getLogger("moloch.popup_gallery")

# Pfade
SNAP_DIR = os.path.expanduser("~/moloch/snapshots")
DAILY_DIR = "/mnt/moloch-data/daily"

THUMB_W = 150
THUMB_H = 112
COLUMNS = 3


class _ScrollableGrid:
    """Wiederverwendbarer scrollbarer Thumbnail-Grid Container."""

    def __init__(self, parent):
        self._thumb_refs = []
        self._parent = parent

        container = tk.Frame(parent, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        self._scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grid_frame = tk.Frame(self.canvas, bg=BG_DARK)
        self._canvas_window = self.canvas.create_window(
            (0, 0), window=self.grid_frame, anchor=tk.NW
        )

        self.grid_frame.bind("<Configure>", self._on_grid_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_grid_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._canvas_window, width=event.width)

    def clear(self):
        """Alle Widgets und Thumbnail-Referenzen entfernen."""
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

    def add_thumb_ref(self, ref):
        """ImageTk Referenz halten (GC-Schutz)."""
        self._thumb_refs.append(ref)


class SnapshotGallery:
    """Snapshot Galerie als eigenstaendiges Toplevel mit 2 Tabs."""

    def __init__(self, parent):
        self._parent = parent
        self._loading = False

        # Toplevel erstellen
        self.win = tk.Toplevel(parent)
        self.win.title("M.O.L.O.C.H. Galerie")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("560x650")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ttk Style fuer dunkle Tabs
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=BG_DARK)
        style.configure("Dark.TNotebook.Tab",
                        background=BG_FRAME, foreground=FG_LABEL,
                        padding=[10, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", FG_WHITE)])

        # Notebook (Tabs)
        self._notebook = ttk.Notebook(self.win, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Snapshots (manuelle Fotos)
        self._tab_snapshots = tk.Frame(self._notebook, bg=BG_DARK)
        self._notebook.add(self._tab_snapshots, text="Snapshots")
        self._grid_snapshots = self._build_tab(self._tab_snapshots, self._load_snapshots)

        # Tab 2: Teachen (DailyLearner)
        self._tab_teachen = tk.Frame(self._notebook, bg=BG_DARK)
        self._notebook.add(self._tab_teachen, text="Teachen")
        self._grid_teachen, self._day_var, self._day_menu = self._build_teachen_tab()

        # Mausrad-Scrolling (nur fuer aktiven Tab)
        self.win.bind("<Button-4>", self._on_scroll_up)
        self.win.bind("<Button-5>", self._on_scroll_down)

        # Beide Tabs laden (im Background)
        self._load_snapshots()
        self._load_teachen()

    # =========================================================================
    # Tab-Builder
    # =========================================================================

    def _build_tab(self, tab_frame, refresh_callback):
        """Standard-Tab mit Header + ScrollableGrid bauen."""
        header = tk.Frame(tab_frame, bg=BG_DARK)
        header.pack(fill=tk.X, padx=5, pady=(5, 0))

        tk.Button(
            header, text="REFRESH", width=10,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=refresh_callback,
        ).pack(side=tk.RIGHT, padx=5)

        grid = _ScrollableGrid(tab_frame)
        return grid

    def _build_teachen_tab(self):
        """Teachen-Tab mit Tages-Dropdown + ScrollableGrid."""
        header = tk.Frame(self._tab_teachen, bg=BG_DARK)
        header.pack(fill=tk.X, padx=5, pady=(5, 0))

        tk.Label(
            header, text="Tag:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT, padx=5)

        # Tage ermitteln
        days = self._get_daily_days()
        day_var = tk.StringVar(value=days[0] if days else "(leer)")

        day_menu = tk.OptionMenu(
            header, day_var, *(days if days else ["(leer)"]),
            command=lambda _: self._load_teachen(),
        )
        day_menu.config(bg=BG_BUTTON, fg=FG_WHITE, font=FONT_SMALL,
                        activebackground=BG_FRAME, highlightthickness=0)
        day_menu.pack(side=tk.LEFT, padx=5)

        tk.Button(
            header, text="REFRESH", width=10,
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._refresh_teachen,
        ).pack(side=tk.RIGHT, padx=5)

        grid = _ScrollableGrid(self._tab_teachen)
        return grid, day_var, day_menu

    # =========================================================================
    # Datenladen
    # =========================================================================

    def _get_daily_days(self):
        """Verfuegbare Tage aus /mnt/moloch-data/daily/ ermitteln."""
        if not os.path.isdir(DAILY_DIR):
            return []
        days = []
        for d in os.listdir(DAILY_DIR):
            full = os.path.join(DAILY_DIR, d)
            if os.path.isdir(full) and d.startswith("20"):
                days.append(d)
        days.sort(reverse=True)
        return days

    def _load_snapshots(self):
        """Tab Snapshots: Fotos aus ~/moloch/snapshots/ laden."""
        grid = self._grid_snapshots
        grid.clear()

        def _bg_load():
            os.makedirs(SNAP_DIR, exist_ok=True)
            files = []
            if os.path.isdir(SNAP_DIR):
                for f in os.listdir(SNAP_DIR):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        path = os.path.join(SNAP_DIR, f)
                        mtime = os.path.getmtime(path)
                        files.append((path, f, mtime))
            files.sort(key=lambda x: x[2], reverse=True)

            # Zurueck im Main-Thread
            try:
                self.win.after(0, lambda: self._populate_grid(
                    grid, files, f"Snapshots ({len(files)})", 0))
            except tk.TclError:
                pass

        threading.Thread(target=_bg_load, daemon=True).start()

    def _load_teachen(self):
        """Tab Teachen: Bilder aus ausgewaehltem Tag laden."""
        grid = self._grid_teachen
        grid.clear()

        day = self._day_var.get()
        if not day or day == "(leer)":
            self._show_empty(grid, "Keine Teachen-Daten")
            self._update_tab_title(1, "Teachen (0)")
            return

        day_dir = os.path.join(DAILY_DIR, day)

        def _bg_load():
            files = []
            if os.path.isdir(day_dir):
                for f in os.listdir(day_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        path = os.path.join(day_dir, f)
                        mtime = os.path.getmtime(path)
                        # Metadaten-JSON laden wenn vorhanden
                        json_path = path.rsplit(".", 1)[0] + ".json"
                        meta = None
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, "r") as jf:
                                    meta = json.load(jf)
                            except Exception:
                                pass
                        files.append((path, f, mtime, meta))
            files.sort(key=lambda x: x[2], reverse=True)

            try:
                self.win.after(0, lambda: self._populate_teachen_grid(
                    grid, files, len(files)))
            except tk.TclError:
                pass

        threading.Thread(target=_bg_load, daemon=True).start()

    def _refresh_teachen(self):
        """Teachen-Tab komplett refreshen (inkl. Tage-Dropdown)."""
        days = self._get_daily_days()
        menu = self._day_menu["menu"]
        menu.delete(0, "end")
        for d in (days if days else ["(leer)"]):
            menu.add_command(label=d, command=lambda v=d: (self._day_var.set(v), self._load_teachen()))
        if days and self._day_var.get() not in days:
            self._day_var.set(days[0])
        self._load_teachen()

    # =========================================================================
    # Grid Population
    # =========================================================================

    def _populate_grid(self, grid, files, tab_title, tab_index):
        """Standard-Grid mit Thumbnails fuellen (Snapshots-Tab)."""
        grid.clear()
        self._update_tab_title(tab_index, tab_title)

        if not files:
            self._show_empty(grid, "Keine Snapshots vorhanden")
            return

        for idx, (path, fname, mtime) in enumerate(files):
            row = idx // COLUMNS
            col = idx % COLUMNS
            self._build_thumbnail(grid, row, col, path, fname, mtime)

    def _populate_teachen_grid(self, grid, files, count):
        """Teachen-Grid mit Thumbnails + Metadaten fuellen."""
        grid.clear()
        self._update_tab_title(1, f"Teachen ({count})")

        if not files:
            self._show_empty(grid, "Keine Teachen-Bilder fuer diesen Tag")
            return

        for idx, (path, fname, mtime, meta) in enumerate(files):
            row = idx // COLUMNS
            col = idx % COLUMNS
            self._build_thumbnail(grid, row, col, path, fname, mtime, meta=meta)

    def _build_thumbnail(self, grid, row, col, path, fname, mtime, meta=None):
        """Einzelnes Thumbnail mit Label und Loeschen-Button."""
        cell = tk.Frame(grid.grid_frame, bg=BG_FRAME, padx=3, pady=3)
        cell.grid(row=row, column=col, padx=4, pady=4, sticky=tk.N)

        # Thumbnail laden
        try:
            img = Image.open(path)
            img.thumbnail((THUMB_W, THUMB_H))
            photo = ImageTk.PhotoImage(img)
            grid.add_thumb_ref(photo)

            lbl_img = tk.Label(cell, image=photo, bg=BG_FRAME, cursor="hand2")
            lbl_img.pack(padx=2, pady=2)
            lbl_img.bind("<Button-1>", lambda e, p=path: self._open_fullsize(p))
        except Exception as e:
            logger.warning(f"Thumbnail Fehler fuer {fname}: {e}")
            tk.Label(
                cell, text="[Fehler]", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
                width=20, height=7,
            ).pack(padx=2, pady=2)

        # Dateiname (gekuerzt)
        display = fname if len(fname) <= 25 else fname[:22] + "..."
        tk.Label(
            cell, text=display, bg=BG_FRAME, fg=FG_LABEL,
            font=FONT_SMALL, wraplength=THUMB_W,
        ).pack()

        # Datum
        date_str = time.strftime("%d.%m.%Y %H:%M", time.localtime(mtime))
        tk.Label(
            cell, text=date_str, bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack()

        # Metadaten (Teachen-Tab)
        if meta:
            name = meta.get("name", "?")
            conf = meta.get("confidence", 0)
            meta_text = f"{name} ({conf:.0%})"
            tk.Label(
                cell, text=meta_text, bg=BG_FRAME, fg=ACCENT_CYAN, font=FONT_SMALL,
            ).pack()

        # Loeschen-Button
        tk.Button(
            cell, text="X", width=3,
            bg=BTN_OFF_DARK, fg=BTN_ALARM_RED, font=FONT_SMALL,
            activebackground=BG_FRAME,
            command=lambda p=path, f=fname: self._delete_file(p, f),
        ).pack(pady=(2, 0))

    # =========================================================================
    # Aktionen
    # =========================================================================

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

        max_w, max_h = 1024, 768
        w, h = img.size
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        full_win._photo_ref = photo
        full_win.geometry(f"{img.size[0]}x{img.size[1]}")
        full_win.resizable(False, False)
        tk.Label(full_win, image=photo, bg=BG_DARK).pack()

    def _delete_file(self, path, fname):
        """Datei loeschen mit Bestaetigung."""
        confirm = messagebox.askyesno(
            "Loeschen",
            f"Wirklich loeschen?\n{fname}",
            parent=self.win,
        )
        if not confirm:
            return
        try:
            os.remove(path)
            # Zugehoerige JSON-Metadaten auch loeschen
            json_path = path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(json_path):
                os.remove(json_path)
            logger.info(f"Geloescht: {fname}")
        except Exception as e:
            logger.error(f"Loeschen fehlgeschlagen: {e}")
            return

        # Aktuellen Tab neu laden
        tab_idx = self._notebook.index(self._notebook.select())
        if tab_idx == 0:
            self._load_snapshots()
        elif tab_idx == 1:
            self._load_teachen()

    # =========================================================================
    # Hilfsfunktionen
    # =========================================================================

    def _show_empty(self, grid, text):
        """Leer-Hinweis im Grid anzeigen."""
        tk.Label(
            grid.grid_frame, text=text,
            bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL,
        ).grid(row=0, column=0, columnspan=COLUMNS, pady=40)

    def _update_tab_title(self, index, title):
        """Tab-Titel mit Anzahl aktualisieren."""
        try:
            self._notebook.tab(index, text=title)
        except Exception:
            pass

    def _on_scroll_up(self, event):
        """Mausrad hoch fuer aktiven Tab."""
        grid = self._get_active_grid()
        if grid:
            grid.canvas.yview_scroll(-3, "units")

    def _on_scroll_down(self, event):
        """Mausrad runter fuer aktiven Tab."""
        grid = self._get_active_grid()
        if grid:
            grid.canvas.yview_scroll(3, "units")

    def _get_active_grid(self):
        """ScrollableGrid des aktiven Tabs zurueckgeben."""
        try:
            idx = self._notebook.index(self._notebook.select())
        except Exception:
            return None
        if idx == 0:
            return self._grid_snapshots
        elif idx == 1:
            return self._grid_teachen
        return None

    def _on_close(self):
        """Fenster schliessen."""
        self.win.destroy()
