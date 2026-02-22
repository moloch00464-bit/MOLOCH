#!/usr/bin/env python3
"""
M.O.L.O.C.H. Settings Popup — JSON Viewer
==========================================

Eigenstaendiges Toplevel-Fenster fuer Settings-Verwaltung.
Zeigt config/settings.json als lesbares JSON mit Bearbeitungsfunktionen.

Sektionen:
- Readonly Text-Widget: JSON-Inhalt formatiert (json.dumps indent=2)
- RELOAD Button: Liest settings.json neu ein
- BACKUP Button: Kopiert settings.json nach settings_YYYY-MM-DD_HHMMSS.bak
- RESET Button (rot): Setzt auf Defaults zurueck (mit Bestaetigung)
- Info-Label: Dateipfad und letzte Aenderungszeit

Importiert NUR panel_styles, tkinter, json, os, shutil, time.
"""

import json
import logging
import os
import shutil
import time
import tkinter as tk
from tkinter import messagebox

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    ACCENT_GREEN, ACCENT_RED,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_settings")

# Pfad zur settings.json
SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "settings.json"
)
SETTINGS_PATH = os.path.normpath(SETTINGS_PATH)

# Default-Werte fuer Reset
DEFAULTS = {
    "version": 1,
    "thresholds": {
        "scrfd_conf": 0.5,
        "scrfd_nms": 0.4,
        "arcface_thresh": 0.6,
        "yolo_conf": 0.5,
    },
    "hand_occlusion": {
        "timeout": 5.0,
        "streak": 3,
        "recency": 2.0,
    },
    "audio": {
        "mic_gain": 1.0,
        "noise_gate_db": -60,
        "agc_enabled": False,
    },
}


class SettingsPopup(tk.Toplevel):
    """Settings JSON Viewer/Manager als eigenstaendiges Toplevel."""

    def __init__(self, parent, service_proxy):
        super().__init__(parent)
        self._service = service_proxy

        self.attributes('-topmost', True)
        self.transient(parent)
        self.title("Settings — config/settings.json")
        self.configure(bg=BG_DARK)
        self.geometry("520x560")
        self.resizable(True, True)

        self._build_gui()
        self._load_and_display()

    # =========================================================================
    # GUI aufbauen
    # =========================================================================

    def _build_gui(self):
        """Alle GUI-Elemente erstellen."""
        # Titel
        tk.Label(
            self, text="Settings — config/settings.json",
            bg=BG_DARK, fg=FG_WHITE, font=FONT_TITLE,
        ).pack(padx=10, pady=(10, 5))

        # JSON Text-Widget mit Scrollbar
        text_frame = tk.Frame(self, bg=BG_DARK)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._text = tk.Text(
            text_frame,
            bg=BG_INPUT, fg=FG_WHITE, font=FONT_MONO,
            insertbackground=FG_WHITE,
            selectbackground="#334466",
            wrap=tk.NONE,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=8, pady=8,
        )
        self._text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._text.yview)

        # Horizontale Scrollbar
        h_scroll = tk.Scrollbar(text_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self._text.config(xscrollcommand=h_scroll.set)
        h_scroll.config(command=self._text.xview)

        # Button-Reihe
        btn_frame = tk.Frame(self, bg=BG_DARK)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(
            btn_frame, text="RELOAD", width=10,
            bg=BG_FRAME, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_DARK,
            command=self._reload,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame, text="BACKUP", width=10,
            bg=BG_FRAME, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_DARK,
            command=self._backup,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame, text="RESET", width=10,
            bg=ACCENT_RED, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground="#cc0000",
            command=self._reset,
        ).pack(side=tk.RIGHT, padx=3)

        # Feedback-Label
        self._lbl_feedback = tk.Label(
            self, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_feedback.pack(padx=10, pady=2)

        # Info-Label unten: Dateipfad und letzte Aenderung
        self._lbl_info = tk.Label(
            self, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
            anchor=tk.W,
        )
        self._lbl_info.pack(fill=tk.X, padx=10, pady=(0, 10))

    # =========================================================================
    # JSON laden und anzeigen
    # =========================================================================

    def _load_and_display(self):
        """settings.json lesen, formatiert anzeigen, Info-Label aktualisieren."""
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            formatted = f"FEHLER: {SETTINGS_PATH} nicht gefunden!"
            logger.error("settings.json nicht gefunden: %s", SETTINGS_PATH)
        except json.JSONDecodeError as e:
            formatted = f"FEHLER: Ungültiges JSON!\n{e}"
            logger.error("settings.json JSON-Fehler: %s", e)

        # Text-Widget aktualisieren
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", formatted)
        self._text.config(state=tk.DISABLED)

        # Info-Label: Pfad + letzte Aenderung
        self._update_info_label()

    def _update_info_label(self):
        """Info-Label mit Dateipfad und Aenderungszeit aktualisieren."""
        try:
            mtime = os.path.getmtime(SETTINGS_PATH)
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            self._lbl_info.config(
                text=f"{SETTINGS_PATH}  |  Letzte Änderung: {mtime_str}"
            )
        except OSError:
            self._lbl_info.config(text=f"{SETTINGS_PATH}  |  Datei nicht gefunden")

    def _show_feedback(self, text, color=ACCENT_GREEN):
        """Feedback-Text kurz anzeigen, nach 3s ausblenden."""
        self._lbl_feedback.config(text=text, fg=color)
        self.after(3000, lambda: self._lbl_feedback.config(text="", fg=FG_DIM))

    # =========================================================================
    # RELOAD
    # =========================================================================

    def _reload(self):
        """settings.json neu einlesen und anzeigen."""
        self._load_and_display()
        self._show_feedback("Neu geladen.")
        logger.info("Settings neu geladen")

    # =========================================================================
    # BACKUP
    # =========================================================================

    def _backup(self):
        """settings.json als Backup mit Zeitstempel kopieren."""
        if not os.path.exists(SETTINGS_PATH):
            self._show_feedback("Keine settings.json vorhanden!", color=ACCENT_RED)
            return

        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        backup_name = f"settings_{timestamp}.bak"
        backup_path = os.path.join(os.path.dirname(SETTINGS_PATH), backup_name)

        try:
            shutil.copy2(SETTINGS_PATH, backup_path)
            self._show_feedback(f"Backup: {backup_name}")
            logger.info("Settings Backup erstellt: %s", backup_path)
        except OSError as e:
            self._show_feedback(f"Backup fehlgeschlagen: {e}", color=ACCENT_RED)
            logger.error("Backup fehlgeschlagen: %s", e)

    # =========================================================================
    # RESET
    # =========================================================================

    def _reset(self):
        """Auf Defaults zuruecksetzen mit Bestaetigung."""
        confirmed = messagebox.askyesno(
            "Reset bestätigen",
            "Settings wirklich auf Defaults zurücksetzen?\n\n"
            "Die aktuelle settings.json wird überschrieben!",
            parent=self,
        )
        if not confirmed:
            return

        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULTS, f, indent=2, ensure_ascii=False)
                f.write("\n")
            self._load_and_display()
            self._show_feedback("Defaults wiederhergestellt.", color=ACCENT_GREEN)
            logger.info("Settings auf Defaults zurueckgesetzt")
        except OSError as e:
            self._show_feedback(f"Reset fehlgeschlagen: {e}", color=ACCENT_RED)
            logger.error("Reset fehlgeschlagen: %s", e)
