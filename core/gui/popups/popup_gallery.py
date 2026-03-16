#!/usr/bin/env python3
"""
M.O.L.O.C.H. Snapshot Galerie Popup — v4.0
============================================

4 Tabs:
  Tab 1: Personen  — ~/moloch/media/snapshots/ (Person-Unterordner: markus, franzi, ...)
  Tab 2: Enrollment — ~/moloch/media/faces/ (Training/Embedding Bilder)
  Tab 3: Captures  — ~/moloch/media/snapshots/ (manuell + Auto-Crops)
  Tab 4: Teach     — ~/moloch/media/teach/ (Teach-Bilder)

Features:
- Thumbnails 100x100px, 4 Spalten
- Max 50 pro Seite, Blättern mit Prev/Next
- Klick = Vollbild
- Dateiname + Datum + Bildgröße
- Refresh pro Tab
- Löschen-Button
- Suchleiste (filtert Dateiname)
- Background-Threading (kein GUI-Freeze)

Importiert NUR panel_styles + tkinter. KEIN Import von moloch_service.
"""

import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON,
    BTN_ALARM_RED, BTN_OFF_DARK, BTN_SNAP_CYAN,
    ACCENT_CYAN,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL,
)

logger = logging.getLogger("moloch.popup_gallery")

# Zentrale Bildpfade
MEDIA_DIR     = os.path.expanduser("~/moloch/media")
SNAPSHOTS_DIR = os.path.join(MEDIA_DIR, "snapshots")   # Person-Unterordner (symlink → ~/moloch/snapshots/)
FACES_DIR     = os.path.join(MEDIA_DIR, "faces")        # Enrollment-Bilder
TEACH_DIR     = os.path.join(MEDIA_DIR, "teach")        # Teach-Bilder (neu)

# Tab-Verzeichnisse in Reihenfolge
TAB_DIRS  = [SNAPSHOTS_DIR, FACES_DIR, SNAPSHOTS_DIR, TEACH_DIR]
TAB_NAMES = ["Personen", "Enrollment", "Captures", "Teach"]

THUMB_W  = 100
THUMB_H  = 100
COLUMNS  = 4
PAGE_SIZE = 50


class _ScrollableGrid:
    """Scrollbarer Thumbnail-Grid Container."""

    def __init__(self, parent):
        self._thumb_refs = []

        container = tk.Frame(parent, bg=BG_DARK)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(container, bg=BG_DARK, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grid_frame = tk.Frame(self.canvas, bg=BG_DARK)
        self._win = self.canvas.create_window((0, 0), window=self.grid_frame, anchor=tk.NW)

        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(
            self._win, width=e.width))

    def clear(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

    def add_ref(self, ref):
        self._thumb_refs.append(ref)


def _collect_images(base_dir, query=None):
    """Alle Bilder aus base_dir sammeln (inkl. Unterordner).
    Gibt Liste von (path, fname, mtime, person) zurück, sortiert nach mtime desc."""
    result = []
    if not os.path.isdir(base_dir):
        return result

    entries = os.scandir(base_dir)
    for entry in entries:
        if entry.is_dir():
            # Personen-Unterordner
            person = entry.name
            try:
                for f in os.scandir(entry.path):
                    if f.is_file() and f.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        if query and query not in f.name.lower() and query not in person.lower():
                            continue
                        result.append((f.path, f.name, f.stat().st_mtime, person))
            except OSError:
                pass
        elif entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
            if query and query not in entry.name.lower():
                continue
            result.append((entry.path, entry.name, entry.stat().st_mtime, ""))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


class SnapshotGallery:
    """Galerie-Popup mit 4 Tabs: Personen, Enrollment, Captures, Teach."""

    def __init__(self, parent):
        self._parent = parent
        self._search_timer = None

        # Alle Verzeichnisse anlegen falls nicht vorhanden
        for d in set(TAB_DIRS):
            os.makedirs(d, exist_ok=True)

        # Seitenzähler + Dateiliste pro Tab (4 Tabs)
        self._pages = [0, 0, 0, 0]
        self._all_files = [[], [], [], []]

        # Toplevel
        self.win = tk.Toplevel(parent)
        self.win.attributes("-topmost", True)
        self.win.transient(parent)
        self.win.title("M.O.L.O.C.H. Galerie")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("600x680")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)

        # Suchleiste
        search_frame = tk.Frame(self.win, bg=BG_DARK)
        search_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        tk.Label(search_frame, text="Suche:", bg=BG_DARK, fg=FG_LABEL,
                 font=FONT_LABEL).pack(side=tk.LEFT, padx=(5, 3))
        self._search_var = tk.StringVar()
        self._search_entry = tk.Entry(
            search_frame, textvariable=self._search_var,
            bg=BG_FRAME, fg=FG_WHITE, insertbackground=FG_WHITE,
            font=FONT_LABEL, width=25)
        self._search_entry.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        self._search_var.trace_add("write", self._on_search_changed)
        tk.Button(search_frame, text="X", width=3,
                  bg=BG_BUTTON, fg=FG_LABEL, font=FONT_SMALL,
                  command=lambda: self._search_var.set("")).pack(side=tk.LEFT, padx=3)

        # Notebook
        style = ttk.Style()
        style.configure("Dark.TNotebook", background=BG_DARK)
        style.configure("Dark.TNotebook.Tab", background=BG_FRAME,
                        foreground=FG_LABEL, padding=[10, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", FG_WHITE)])

        self._notebook = ttk.Notebook(self.win, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Tab 0: Personen
        self._tab_frames = []
        self._grids = []
        self._pagers = []  # (prev_btn, next_btn, page_lbl) pro Tab

        for tab_name in TAB_NAMES:
            frame = tk.Frame(self._notebook, bg=BG_DARK)
            self._notebook.add(frame, text=tab_name)
            self._tab_frames.append(frame)

        self._build_tab_ui()

        # Mausrad-Scrolling
        self.win.bind("<Button-4>", self._scroll_up)
        self.win.bind("<Button-5>", self._scroll_down)

        # Initialer Load aller 4 Tabs im Background
        for i in range(len(TAB_NAMES)):
            self._load_tab(i)

    # =========================================================================
    # UI Aufbau
    # =========================================================================

    def _build_tab_ui(self):
        """Header + Grid + Pager für alle 4 Tabs bauen."""
        tab_dirs = TAB_DIRS
        refresh_cbs = [lambda i=i: self._refresh(i) for i in range(len(TAB_NAMES))]

        for i, (frame, base_dir, cb) in enumerate(zip(self._tab_frames, tab_dirs, refresh_cbs)):
            # Header
            header = tk.Frame(frame, bg=BG_DARK)
            header.pack(fill=tk.X, padx=5, pady=(5, 0))
            tk.Label(header, text=base_dir.replace(os.path.expanduser("~"), "~"),
                     bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL).pack(side=tk.LEFT, padx=5)
            tk.Button(header, text="REFRESH", width=9,
                      bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
                      activebackground=BG_FRAME,
                      command=cb).pack(side=tk.RIGHT, padx=5)

            # Grid
            grid = _ScrollableGrid(frame)
            self._grids.append(grid)

            # Pager
            pager_frame = tk.Frame(frame, bg=BG_DARK)
            pager_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            prev_btn = tk.Button(pager_frame, text="< Zurück", width=9,
                                 bg=BG_BUTTON, fg=FG_LABEL, font=FONT_SMALL,
                                 state=tk.DISABLED,
                                 command=lambda idx=i: self._page_prev(idx))
            prev_btn.pack(side=tk.LEFT, padx=5)
            page_lbl = tk.Label(pager_frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL)
            page_lbl.pack(side=tk.LEFT, expand=True)
            next_btn = tk.Button(pager_frame, text="Weiter >", width=9,
                                 bg=BG_BUTTON, fg=FG_LABEL, font=FONT_SMALL,
                                 state=tk.DISABLED,
                                 command=lambda idx=i: self._page_next(idx))
            next_btn.pack(side=tk.RIGHT, padx=5)
            self._pagers.append((prev_btn, next_btn, page_lbl))

    # =========================================================================
    # Laden
    # =========================================================================

    def _load_tab(self, tab_idx, query=None):
        """Tab im Background laden."""
        base_dir = TAB_DIRS[tab_idx]

        def _bg():
            # Verzeichnis anlegen falls nicht vorhanden
            os.makedirs(base_dir, exist_ok=True)
            files = _collect_images(base_dir, query)
            try:
                self.win.after(0, lambda: self._on_files_loaded(tab_idx, files))
            except tk.TclError:
                pass

        threading.Thread(target=_bg, daemon=True).start()

    def _on_files_loaded(self, tab_idx, files):
        """Wird im GUI-Thread aufgerufen nach Background-Load."""
        self._all_files[tab_idx] = files
        self._pages[tab_idx] = 0
        self._render_page(tab_idx)

    def _refresh(self, tab_idx):
        query = self._search_var.get().strip().lower() or None
        self._load_tab(tab_idx, query)

    # =========================================================================
    # Pagination
    # =========================================================================

    def _page_prev(self, tab_idx):
        if self._pages[tab_idx] > 0:
            self._pages[tab_idx] -= 1
            self._render_page(tab_idx)

    def _page_next(self, tab_idx):
        total = len(self._all_files[tab_idx])
        if (self._pages[tab_idx] + 1) * PAGE_SIZE < total:
            self._pages[tab_idx] += 1
            self._render_page(tab_idx)

    def _render_page(self, tab_idx):
        """Aktuelle Seite im Grid anzeigen."""
        files = self._all_files[tab_idx]
        page = self._pages[tab_idx]
        grid = self._grids[tab_idx]
        prev_btn, next_btn, page_lbl = self._pagers[tab_idx]

        grid.clear()

        total = len(files)
        # Tab-Titel mit Bildanzahl aktualisieren
        try:
            self._notebook.tab(tab_idx, text=f"{TAB_NAMES[tab_idx]} ({total})")
        except Exception:
            pass

        if total == 0:
            tk.Label(grid.grid_frame, text="Keine Bilder vorhanden",
                     bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL).grid(
                row=0, column=0, columnspan=COLUMNS, pady=40)
            page_lbl.config(text="")
            prev_btn.config(state=tk.DISABLED)
            next_btn.config(state=tk.DISABLED)
            return

        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        page_files = files[start:end]

        total_pages = (total - 1) // PAGE_SIZE + 1
        page_lbl.config(text=f"Seite {page + 1}/{total_pages}  ({start + 1}–{end} von {total})")
        prev_btn.config(state=tk.NORMAL if page > 0 else tk.DISABLED)
        next_btn.config(state=tk.NORMAL if end < total else tk.DISABLED)

        for idx, (path, fname, mtime, person) in enumerate(page_files):
            row = idx // COLUMNS
            col = idx % COLUMNS
            self._build_thumb(grid, tab_idx, row, col, path, fname, mtime, person)

    # =========================================================================
    # Thumbnail Aufbau
    # =========================================================================

    def _build_thumb(self, grid, tab_idx, row, col, path, fname, mtime, person):
        """Einzelnes Thumbnail-Widget."""
        cell = tk.Frame(grid.grid_frame, bg=BG_FRAME, padx=2, pady=2)
        cell.grid(row=row, column=col, padx=3, pady=3, sticky=tk.N)

        # Bild laden
        try:
            img = Image.open(path)
            orig_w, orig_h = img.size
            img.thumbnail((THUMB_W, THUMB_H), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            grid.add_ref(photo)
            lbl_img = tk.Label(cell, image=photo, bg=BG_FRAME, cursor="hand2")
            lbl_img.pack(padx=2, pady=2)
            lbl_img.bind("<Button-1>", lambda e, p=path: self._open_fullsize(p))
        except Exception:
            tk.Label(cell, text="[Fehler]", bg=BG_FRAME, fg=FG_DIM,
                     font=FONT_SMALL, width=14, height=6).pack(padx=2, pady=2)
            orig_w, orig_h = 0, 0

        # Person-Label (nur bei Unterordner-Quellen)
        if person:
            tk.Label(cell, text=person, bg=BG_FRAME, fg=ACCENT_CYAN,
                     font=FONT_SMALL).pack()

        # Dateiname (gekürzt)
        short = fname if len(fname) <= 18 else fname[:15] + "..."
        tk.Label(cell, text=short, bg=BG_FRAME, fg=FG_LABEL,
                 font=FONT_SMALL).pack()

        # Datum + Größe
        date_str = time.strftime("%d.%m.%y %H:%M", time.localtime(mtime))
        size_str = f" {orig_w}×{orig_h}" if orig_w else ""
        tk.Label(cell, text=f"{date_str}{size_str}", bg=BG_FRAME, fg=FG_DIM,
                 font=FONT_SMALL).pack()

        # Löschen-Button
        tk.Button(cell, text="X", width=3,
                  bg=BTN_OFF_DARK, fg=BTN_ALARM_RED, font=FONT_SMALL,
                  activebackground=BG_FRAME,
                  command=lambda p=path, ti=tab_idx: self._delete(p, ti)
                  ).pack(pady=(2, 0))

    # =========================================================================
    # Aktionen
    # =========================================================================

    def _open_fullsize(self, path):
        try:
            img = Image.open(path)
        except Exception as e:
            logger.error(f"Bild öffnen fehlgeschlagen: {e}")
            return
        win = tk.Toplevel(self.win)
        win.title(os.path.basename(path))
        win.configure(bg=BG_DARK)
        max_w, max_h = 1024, 768
        w, h = img.size
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        win._photo_ref = photo
        win.geometry(f"{img.size[0]}x{img.size[1]}")
        win.resizable(False, False)
        tk.Label(win, image=photo, bg=BG_DARK).pack()

    def _delete(self, path, tab_idx):
        try:
            os.remove(path)
            # Zugehörige JSON-Meta auch löschen
            json_path = path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(json_path):
                os.remove(json_path)
        except Exception as e:
            logger.error(f"Löschen fehlgeschlagen: {e}")
            return
        # Aus Cache entfernen + Seite neu rendern
        self._all_files[tab_idx] = [
            f for f in self._all_files[tab_idx] if f[0] != path]
        self._render_page(tab_idx)

    # =========================================================================
    # Suche
    # =========================================================================

    def _on_search_changed(self, *_):
        if self._search_timer:
            self.win.after_cancel(self._search_timer)
        self._search_timer = self.win.after(300, self._apply_search)

    def _apply_search(self):
        query = self._search_var.get().strip().lower() or None
        tab_idx = self._notebook.index(self._notebook.select())
        self._load_tab(tab_idx, query)

    # =========================================================================
    # Scrolling + Tab-Wechsel
    # =========================================================================

    def _scroll_up(self, _event):
        grid = self._get_active_grid()
        if grid:
            grid.canvas.yview_scroll(-3, "units")

    def _scroll_down(self, _event):
        grid = self._get_active_grid()
        if grid:
            grid.canvas.yview_scroll(3, "units")

    def _get_active_grid(self):
        try:
            idx = self._notebook.index(self._notebook.select())
            return self._grids[idx]
        except Exception:
            return None

    def _on_tab_changed(self, _event):
        """Bei Tab-Wechsel: Falls noch nicht geladen, jetzt laden."""
        try:
            idx = self._notebook.index(self._notebook.select())
        except Exception:
            return
        if not self._all_files[idx]:
            self._load_tab(idx)
