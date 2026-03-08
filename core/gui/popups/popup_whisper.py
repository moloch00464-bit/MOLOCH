#!/usr/bin/env python3
"""
M.O.L.O.C.H. Whisper Popup — STT Monitor + Einstellungen
==========================================================

Eigenstaendiges Toplevel-Fenster fuer Whisper Speech-to-Text.

Oben: Echtzeit-Pegelanzeige (WiFi-Mic Amplitude, 100ms Update)
Mitte: Scrollbares Log mit Whisper-Erkennungen (Timestamp, Dauer, Text)
Unten: Einstellungen (VAD, Modell-Info, Manueller Test)

Liest Whisper-Ergebnisse aus dem Service-Status (voice.whisper_results).
Pegelanzeige liest wifi_mic Pegel aus dem Status.
Importiert NUR panel_styles und tkinter.
"""

import json
import logging
import os
import time
import tkinter as tk
from tkinter import scrolledtext

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_BUTTON, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    ACCENT_GREEN, ACCENT_CYAN,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_BUTTON, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_whisper")

# Konstanten
LEVEL_UPDATE_MS = 100   # Pegel-Update alle 100ms
STATUS_UPDATE_MS = 500  # Status-Polling alle 500ms
LEVEL_WIDTH = 300
LEVEL_HEIGHT = 20
SETTINGS_PATH = os.path.expanduser("~/moloch/config/settings.json")


class WhisperPopup:
    """Whisper STT Monitor als eigenstaendiges Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        """
        Args:
            parent: Parent-Widget (fuer Toplevel)
            service_proxy: ServiceProxy Instanz fuer Commands/Status
        """
        self.parent = parent
        self.service = service_proxy
        self._after_level = None
        self._after_status = None
        self._seen_results = set()  # IDs der bereits angezeigten Ergebnisse
        self._vad_enabled = tk.BooleanVar(value=True)

        # Settings laden
        self._load_settings()

        # Fenster erstellen
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Whisper STT — M.O.L.O.C.H.")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("460x520")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # GUI aufbauen
        self._build_level_meter()
        self._build_status_section()
        self._build_log_section()
        self._build_settings_section()
        self._build_buttons()

        # Updates starten
        self._update_level()
        self._update_status()

    # =========================================================================
    # Pegelanzeige (oben)
    # =========================================================================

    def _build_level_meter(self):
        """Echtzeit-Pegelanzeige mit Canvas-Balken."""
        frame = tk.LabelFrame(
            self.win, text="Mikrofon-Pegel",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        row = tk.Frame(frame, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=5)

        self._canvas_level = tk.Canvas(
            row, width=LEVEL_WIDTH, height=LEVEL_HEIGHT,
            bg=BG_INPUT, highlightthickness=0,
        )
        self._canvas_level.pack(side=tk.LEFT, padx=(0, 8))

        self._lbl_db = tk.Label(
            row, text="-- dB", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
            width=8, anchor=tk.E,
        )
        self._lbl_db.pack(side=tk.LEFT)

    def _update_level(self):
        """Pegel-Balken aktualisieren (100ms)."""
        if not self.win.winfo_exists():
            return

        rms_db = -60.0
        try:
            status = self.service.read_status()
            if status:
                voice = status.get("voice", {})
                wifi = voice.get("wifi_mic", {})
                rms_db = wifi.get("rms_db", -60.0)
        except Exception:
            pass

        # dB in Balkenlänge umrechnen (-60dB..0dB → 0..LEVEL_WIDTH)
        normalized = max(0.0, min(1.0, (rms_db + 60.0) / 60.0))
        bar_width = int(normalized * LEVEL_WIDTH)

        # Farbe nach Pegel
        if normalized > 0.8:
            color = STATUS_RED
        elif normalized > 0.5:
            color = STATUS_YELLOW
        elif normalized > 0.15:
            color = ACCENT_GREEN
        else:
            color = FG_DIM

        self._canvas_level.delete("all")
        if bar_width > 0:
            self._canvas_level.create_rectangle(
                0, 0, bar_width, LEVEL_HEIGHT, fill=color, outline="",
            )

        self._lbl_db.config(text=f"{rms_db:.0f} dB", fg=color)

        self._after_level = self.win.after(LEVEL_UPDATE_MS, self._update_level)

    # =========================================================================
    # Status-Anzeige
    # =========================================================================

    def _build_status_section(self):
        """Aufnahme-Status + Whisper-Backend Anzeige."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        # Status-Zeile
        tk.Label(
            frame, text="Status:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))

        self._lbl_status = tk.Label(
            frame, text="Warte auf PTT...", bg=BG_DARK, fg=FG_DIM,
            font=FONT_MONO,
        )
        self._lbl_status.grid(row=0, column=1, sticky=tk.W)

        # Backend-Zeile
        tk.Label(
            frame, text="Backend:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).grid(row=1, column=0, sticky=tk.W, padx=(0, 5))

        self._lbl_backend = tk.Label(
            frame, text="--", bg=BG_DARK, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_backend.grid(row=1, column=1, sticky=tk.W)

        # Audio-Source
        tk.Label(
            frame, text="Quelle:", bg=BG_DARK, fg=FG_LABEL, font=FONT_LABEL,
        ).grid(row=2, column=0, sticky=tk.W, padx=(0, 5))

        self._lbl_source = tk.Label(
            frame, text="--", bg=BG_DARK, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_source.grid(row=2, column=1, sticky=tk.W)

    # =========================================================================
    # Erkennungs-Log (Mitte)
    # =========================================================================

    def _build_log_section(self):
        """Scrollbares Textfeld fuer Whisper-Erkennungen."""
        frame = tk.LabelFrame(
            self.win, text="Letzte Erkennungen",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._txt_log = scrolledtext.ScrolledText(
            frame, height=8, wrap=tk.WORD,
            bg=BG_INPUT, fg=FG_WHITE, font=FONT_MONO,
            insertbackground=FG_WHITE,
            selectbackground=ACCENT_CYAN,
            state=tk.DISABLED,
            borderwidth=0, highlightthickness=0,
        )
        self._txt_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tag fuer Timestamp-Farbe
        self._txt_log.tag_configure("ts", foreground=FG_DIM)
        self._txt_log.tag_configure("dur", foreground=STATUS_YELLOW)
        self._txt_log.tag_configure("text", foreground=FG_WHITE)

    def _add_log_entry(self, timestamp: float, duration_ms: float, text: str):
        """Neue Erkennung oben ins Log einfuegen."""
        ts_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        dur_str = f"({duration_ms / 1000:.1f}s)"

        self._txt_log.config(state=tk.NORMAL)
        # Am Anfang einfuegen
        line = f"[{ts_str}] {dur_str} \"{text}\"\n"
        self._txt_log.insert("1.0", line)

        # Max 50 Zeilen behalten
        content = self._txt_log.get("1.0", tk.END)
        lines = content.split("\n")
        if len(lines) > 52:
            self._txt_log.delete(f"{51}.0", tk.END)

        self._txt_log.config(state=tk.DISABLED)

    # =========================================================================
    # Einstellungen (unten)
    # =========================================================================

    def _build_settings_section(self):
        """Einstellungen: VAD, Modell-Info."""
        frame = tk.LabelFrame(
            self.win, text="Einstellungen",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        frame.pack(fill=tk.X, padx=8, pady=4)

        # Zeile 1: Modell-Info (nur Anzeige, Hailo HEF nicht wechselbar)
        row1 = tk.Frame(frame, bg=BG_FRAME)
        row1.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(
            row1, text="Modell:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._lbl_model = tk.Label(
            row1, text="whisper-base (Hailo NPU HEF)",
            bg=BG_FRAME, fg=ACCENT_CYAN, font=FONT_SMALL,
        )
        self._lbl_model.pack(side=tk.LEFT, padx=5)

        # Zeile 2: VAD Checkbox
        row2 = tk.Frame(frame, bg=BG_FRAME)
        row2.pack(fill=tk.X, padx=5, pady=2)

        tk.Checkbutton(
            row2, text="VAD (Voice Activity Detection)",
            variable=self._vad_enabled,
            bg=BG_FRAME, fg=FG_WHITE, font=FONT_LABEL,
            selectcolor=BG_INPUT,
            activebackground=BG_FRAME, activeforeground=FG_WHITE,
            command=self._on_vad_toggle,
        ).pack(side=tk.LEFT)

        self._lbl_vad_info = tk.Label(
            row2, text="webrtcvad, Aggressivitaet 2",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_vad_info.pack(side=tk.LEFT, padx=10)

        # Zeile 3: Audio-Preprocessing Info
        row3 = tk.Frame(frame, bg=BG_FRAME)
        row3.pack(fill=tk.X, padx=5, pady=(2, 5))

        tk.Label(
            row3, text="Preprocessing:",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        tk.Label(
            row3, text="DC-Offset + Normalisierung -3dBFS",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # Buttons (unten)
    # =========================================================================

    def _build_buttons(self):
        """Manuell testen + Einstellungen speichern."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Button(
            frame, text="Manuell testen (3s)",
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._manual_test,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            frame, text="Einstellungen speichern",
            bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._save_settings,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            frame, text="Log loeschen",
            bg=BG_BUTTON, fg=FG_DIM, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._clear_log,
        ).pack(side=tk.RIGHT, padx=4)

        self._lbl_save_status = tk.Label(
            frame, text="", bg=BG_DARK, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save_status.pack(side=tk.RIGHT, padx=4)

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _update_status(self):
        """Status vom Service lesen und GUI aktualisieren (500ms)."""
        if not self.win.winfo_exists():
            return

        try:
            status = self.service.read_status()
            if status:
                voice = status.get("voice", {})

                # Whisper-Status
                ws = voice.get("whisper_status", "Idle")
                if ws == "Aufnahme...":
                    self._lbl_status.config(text="Aufnahme laeuft", fg=STATUS_RED)
                elif ws == "Transkribiere...":
                    self._lbl_status.config(text="Whisper verarbeitet...", fg=STATUS_YELLOW)
                elif ws == "Denke...":
                    self._lbl_status.config(text="Claude denkt...", fg=ACCENT_CYAN)
                elif ws == "Spreche...":
                    self._lbl_status.config(text="TTS spricht...", fg=ACCENT_GREEN)
                else:
                    self._lbl_status.config(text="Warte auf PTT...", fg=FG_DIM)

                # Backend
                backend = voice.get("whisper_backend", "nicht geladen")
                self._lbl_backend.config(text=backend)

                # Audio-Source
                source = voice.get("audio_source", "usb")
                if source == "wifi":
                    self._lbl_source.config(text="ESP32 WiFi-Mic", fg=ACCENT_GREEN)
                else:
                    self._lbl_source.config(text="USB ReSpeaker", fg=STATUS_YELLOW)

                # Neue Whisper-Ergebnisse pruefen
                results = voice.get("whisper_results", [])
                for r in results:
                    rid = r.get("id", 0)
                    if rid not in self._seen_results:
                        self._seen_results.add(rid)
                        self._add_log_entry(
                            timestamp=r.get("ts", time.time()),
                            duration_ms=r.get("duration_ms", 0),
                            text=r.get("text", ""),
                        )

        except Exception as e:
            logger.debug(f"[WHISPER-POPUP] Status-Fehler: {e}")

        self._after_status = self.win.after(STATUS_UPDATE_MS, self._update_status)

    # =========================================================================
    # Actions
    # =========================================================================

    def _manual_test(self):
        """3s Aufnahme ohne PTT starten — via IPC Command."""
        self.service._write_command("action", {
            "action": "whisper_test",
            "duration_s": 3,
        })
        self._lbl_status.config(text="Manueller Test (3s)...", fg=STATUS_YELLOW)

    def _on_vad_toggle(self):
        """VAD an/aus toggle — via IPC Command."""
        enabled = self._vad_enabled.get()
        self.service._write_command("action", {
            "action": "whisper_vad",
            "enabled": enabled,
        })
        logger.info(f"[WHISPER-POPUP] VAD {'an' if enabled else 'aus'}")

    def _clear_log(self):
        """Erkennungs-Log leeren."""
        self._txt_log.config(state=tk.NORMAL)
        self._txt_log.delete("1.0", tk.END)
        self._txt_log.config(state=tk.DISABLED)
        self._seen_results.clear()

    def _save_settings(self):
        """Einstellungen in settings.json speichern."""
        try:
            settings = {}
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    settings = json.load(f)

            settings["whisper"] = {
                "vad_enabled": self._vad_enabled.get(),
                "model": "whisper-base",
                "preprocessing": {
                    "dc_offset_removal": True,
                    "normalize_dbfs": -3.0,
                },
            }

            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            self._lbl_save_status.config(text="Gespeichert!", fg=ACCENT_GREEN)
            self.win.after(2000, lambda: self._lbl_save_status.config(text=""))
            logger.info("[WHISPER-POPUP] Settings gespeichert")
        except Exception as e:
            self._lbl_save_status.config(text=f"Fehler: {e}", fg=STATUS_RED)
            logger.error(f"[WHISPER-POPUP] Settings speichern: {e}")

    def _load_settings(self):
        """Einstellungen aus settings.json laden."""
        try:
            if os.path.exists(SETTINGS_PATH):
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                ws = settings.get("whisper", {})
                self._vad_enabled.set(ws.get("vad_enabled", True))
        except Exception:
            pass

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _on_close(self):
        """Popup schliessen — Timer stoppen."""
        if self._after_level:
            try:
                self.win.after_cancel(self._after_level)
            except Exception:
                pass
        if self._after_status:
            try:
                self.win.after_cancel(self._after_status)
            except Exception:
                pass
        self.win.destroy()
