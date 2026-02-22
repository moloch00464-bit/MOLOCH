#!/usr/bin/env python3
"""
M.O.L.O.C.H. 4.0 - Panel Main
===============================

Hauptfenster des modularen Panels.
- Tkinter Fenster mit 3-Spalten-Layout
- ServiceProxy fuer IPC zum Backend-Service
- Platzhalter-Frames fuer Module (Preview, Steuerung, Chat)
- Status-Bar mit Service-Status

Importiert NUR panel_styles. Keine anderen Module.
"""

import tkinter as tk
from tkinter import ttk
import json
import struct
import os
import logging
import glob as glob_mod
from typing import Optional, Dict, Any

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_TEXT, FG_LABEL, FG_DIM, FG_WHITE,
    STATUS_GREEN, STATUS_RED, STATUS_YELLOW,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
    PREVIEW_W, PREVIEW_H, STATUS_UPDATE_MS,
    SHM_FRAME, SHM_STATUS,
)

# --- Modul-Imports (graceful fallback) ---
try:
    from core.gui.panel_preview import PreviewModule
    _PREVIEW_OK = True
except Exception:
    _PREVIEW_OK = False

try:
    from core.gui.panel_ptz import PtzModule
    _PTZ_OK = True
except Exception:
    _PTZ_OK = False

try:
    from core.gui.panel_ewelink import EwelinkModule
    _EWELINK_OK = True
except Exception:
    _EWELINK_OK = False

try:
    from core.gui.panel_models import ModelsModule
    _MODELS_OK = True
except Exception:
    _MODELS_OK = False

try:
    from core.gui.panel_talk_chat import TalkChatModule
    _TALKCHAT_OK = True
except Exception:
    _TALKCHAT_OK = False


# =============================================================================
# ServiceProxy — IPC zum M.O.L.O.C.H. Backend
# =============================================================================

class ServiceProxy:
    """
    Kommunikation mit dem M.O.L.O.C.H. Service ueber Dateisystem-IPC.

    Commands: /tmp/moloch_cmd_NNNN.json (nummeriert)
    Status:   /dev/shm/moloch_status.json (gelesen alle 500ms)
    Frame:    /dev/shm/moloch_frame (12-byte Header + BGR raw)
    """

    CMD_DIR = "/tmp"
    CMD_PREFIX = "moloch_cmd_"

    def __init__(self):
        self.logger = logging.getLogger("ServiceProxy")
        self._cmd_counter = self._find_next_cmd_number()
        self._last_status: Dict[str, Any] = {}
        self._last_status_mtime: float = 0.0

    def _find_next_cmd_number(self) -> int:
        """Naechste freie Command-Nummer ermitteln."""
        pattern = os.path.join(self.CMD_DIR, f"{self.CMD_PREFIX}*.json")
        existing = glob_mod.glob(pattern)
        if not existing:
            return 1
        # Hoechste Nummer finden
        max_num = 0
        for path in existing:
            basename = os.path.basename(path)
            # moloch_cmd_0001.json -> 0001
            try:
                num_str = basename.replace(self.CMD_PREFIX, "").replace(".json", "")
                num = int(num_str)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
        return max_num + 1

    def _write_command(self, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Command als nummerierte JSON-Datei schreiben.
        Format: Flaches JSON mit "action" als Key.
        params werden direkt ins Dict gemischt (nicht verschachtelt).

        Args:
            action: Action-Name (z.B. 'toggle_model')
            params: Optionale Parameter (werden flach ins Dict gemischt)

        Returns:
            True wenn geschrieben
        """
        cmd_file = os.path.join(
            self.CMD_DIR,
            f"{self.CMD_PREFIX}{self._cmd_counter:04d}.json"
        )
        payload = {"action": action}
        if params:
            payload.update(params)

        try:
            with open(cmd_file, "w") as f:
                json.dump(payload, f)
            self.logger.info(f"CMD #{self._cmd_counter:04d}: {action}")
            self._cmd_counter += 1
            return True
        except Exception as e:
            self.logger.error(f"Command schreiben fehlgeschlagen: {e}")
            return False

    # ----- Command-Methoden -----

    def toggle_model(self, model_name: str):
        """Modell an/aus schalten."""
        self._write_command("toggle_model", {"model": model_name})

    def force_models(self, models: Dict[str, bool]):
        """Mehrere Modelle gleichzeitig setzen."""
        self._write_command("force_models", {"models": models})

    def toggle_smart_tracking(self):
        """Smart Tracking an/aus."""
        self._write_command("toggle_smart_tracking")

    def toggle_autonomous(self):
        """Autonomen Modus an/aus."""
        self._write_command("toggle_autonomous")

    def reload_face_db(self):
        """Face-Datenbank neu laden."""
        self._write_command("reload_face_db")

    def set_threshold(self, model: str, value: float):
        """Schwellwert fuer ein Modell setzen."""
        self._write_command("set_threshold", {"model": model, "value": value})

    def save_settings(self):
        """Aktuelle Settings speichern."""
        self._write_command("save_settings")

    def toggle_daily_learner(self):
        """Daily Learner an/aus."""
        self._write_command("toggle_daily_learner")

    # ----- Status lesen -----

    def read_status(self) -> Dict[str, Any]:
        """
        Service-Status aus Shared Memory lesen.
        Cached anhand mtime — liest nur bei Aenderung.

        Returns:
            Status-Dict oder leeres Dict bei Fehler
        """
        try:
            if not os.path.exists(SHM_STATUS):
                return {}
            mtime = os.path.getmtime(SHM_STATUS)
            if mtime == self._last_status_mtime:
                return self._last_status
            with open(SHM_STATUS, "r") as f:
                self._last_status = json.load(f)
            self._last_status_mtime = mtime
            return self._last_status
        except (json.JSONDecodeError, OSError) as e:
            self.logger.debug(f"Status lesen fehlgeschlagen: {e}")
            return self._last_status

    # ----- Frame lesen -----

    def read_frame(self) -> Optional[bytes]:
        """
        Kamera-Frame aus Shared Memory lesen.

        Format: 12 Byte Header (width u32 LE, height u32 LE, channels u32 LE)
                danach width * height * channels Bytes BGR raw

        Returns:
            (width, height, channels, raw_bytes) oder None bei Fehler
        """
        try:
            if not os.path.exists(SHM_FRAME):
                return None
            with open(SHM_FRAME, "rb") as f:
                header = f.read(12)
                if len(header) < 12:
                    return None
                width, height, channels = struct.unpack("<III", header)
                expected_size = width * height * channels
                if expected_size == 0 or expected_size > 10_000_000:
                    return None
                raw = f.read(expected_size)
                if len(raw) < expected_size:
                    return None
            return (width, height, channels, raw)
        except OSError:
            return None


# =============================================================================
# MolochPanel — Hauptfenster
# =============================================================================

class MolochPanel:
    """
    M.O.L.O.C.H. 4.0 Hauptfenster.

    3-Spalten-Layout:
      Links:  Kamera-Preview (640x360)
      Mitte:  Steuerung
      Rechts: Kommunikation

    Module registrieren sich ueber register_module().
    """

    def __init__(self):
        self.logger = logging.getLogger("MolochPanel")

        # Service-Proxy
        self.service = ServiceProxy()

        # Registrierte Module
        self._modules: Dict[str, Any] = {}

        # Hauptfenster
        self.root = tk.Tk()
        self.root.title("M.O.L.O.C.H. 4.0")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # Layout aufbauen
        self._build_layout()

        # Module laden
        self._load_modules()

        # Fenster-Schliessen abfangen
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status-Polling starten
        self._poll_status()

    def _build_layout(self):
        """3-Spalten-Layout mit Status-Bar aufbauen."""

        # Hauptcontainer
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Spalten konfigurieren
        main_frame.columnconfigure(0, weight=0)  # Kamera: feste Breite
        main_frame.columnconfigure(1, weight=3)  # Steuerung: mehr Platz
        main_frame.columnconfigure(2, weight=2)  # Chat: weniger Platz
        main_frame.rowconfigure(0, weight=1)

        # --- Spalte Links: Kamera Preview ---
        self.frame_kamera = tk.LabelFrame(
            main_frame,
            text="Kamera",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_TITLE,
        )
        self.frame_kamera.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        # Platzhalter-Label fuer Preview
        self._preview_placeholder = tk.Label(
            self.frame_kamera,
            text="Kein Signal",
            bg=BG_INPUT,
            fg=FG_DIM,
            font=FONT_MONO,
            width=PREVIEW_W // 8,
            height=PREVIEW_H // 16,
        )
        self._preview_placeholder.pack(padx=5, pady=5)

        # --- Spalte Mitte: Steuerung ---
        self.frame_steuerung = tk.LabelFrame(
            main_frame,
            text="Steuerung",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_TITLE,
        )
        self.frame_steuerung.grid(row=0, column=1, sticky="nsew", padx=3)

        # Platzhalter
        tk.Label(
            self.frame_steuerung,
            text="Module laden...",
            bg=BG_FRAME,
            fg=FG_DIM,
            font=FONT_SMALL,
        ).pack(pady=20)

        # --- Spalte Rechts: Kommunikation ---
        self.frame_chat = tk.LabelFrame(
            main_frame,
            text="Kommunikation",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_TITLE,
        )
        self.frame_chat.grid(row=0, column=2, sticky="nsew", padx=(3, 0))

        # Platzhalter
        tk.Label(
            self.frame_chat,
            text="Module laden...",
            bg=BG_FRAME,
            fg=FG_DIM,
            font=FONT_SMALL,
        ).pack(pady=20)

        # --- Status-Bar unten ---
        self.status_bar = tk.Label(
            self.root,
            text="Service: nicht verbunden",
            bg=BG_DARK,
            fg=FG_DIM,
            font=FONT_SMALL,
            anchor="w",
        )
        self.status_bar.pack(fill=tk.X, padx=5, pady=(0, 3))

    def _load_modules(self):
        """Module in die Frames einstecken. Fallback-Label bei Fehler."""

        # Platzhalter entfernen
        self._preview_placeholder.destroy()
        for w in self.frame_steuerung.winfo_children():
            w.destroy()
        for w in self.frame_chat.winfo_children():
            w.destroy()

        # (A) Preview -> frame_kamera
        self._preview = None
        if _PREVIEW_OK:
            try:
                self._preview = PreviewModule(self.frame_kamera, self.service)
                self._preview.start()
                self.logger.info("Modul geladen: Preview")
            except Exception as e:
                self.logger.error(f"Preview fehlgeschlagen: {e}")
        if self._preview is None:
            tk.Label(self.frame_kamera, text="Modul nicht geladen: Preview",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=20)

        # (B) PTZ -> frame_steuerung (oben)
        if _PTZ_OK:
            try:
                PtzModule(self.frame_steuerung, self.service)
                self.logger.info("Modul geladen: PTZ")
            except Exception as e:
                self.logger.error(f"PTZ fehlgeschlagen: {e}")
                tk.Label(self.frame_steuerung, text="Modul nicht geladen: PTZ",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)
        else:
            tk.Label(self.frame_steuerung, text="Modul nicht geladen: PTZ",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

        # (C) eWeLink -> frame_steuerung (unter PTZ)
        self._ewelink = None
        if _EWELINK_OK:
            try:
                self._ewelink = EwelinkModule(self.frame_steuerung, self.service)
                self.logger.info("Modul geladen: eWeLink")
            except Exception as e:
                self.logger.error(f"eWeLink fehlgeschlagen: {e}")
                tk.Label(self.frame_steuerung, text="Modul nicht geladen: eWeLink",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)
        else:
            tk.Label(self.frame_steuerung, text="Modul nicht geladen: eWeLink",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

        # (D) Models -> frame_steuerung (unter eWeLink)
        if _MODELS_OK:
            try:
                ModelsModule(self.frame_steuerung, self.service)
                self.logger.info("Modul geladen: Models")
            except Exception as e:
                self.logger.error(f"Models fehlgeschlagen: {e}")
                tk.Label(self.frame_steuerung, text="Modul nicht geladen: Models",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)
        else:
            tk.Label(self.frame_steuerung, text="Modul nicht geladen: Models",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

        # (E) TalkChat -> frame_chat
        if _TALKCHAT_OK:
            try:
                TalkChatModule(self.frame_chat, self.service)
                self.logger.info("Modul geladen: TalkChat")
            except Exception as e:
                self.logger.error(f"TalkChat fehlgeschlagen: {e}")
                tk.Label(self.frame_chat, text="Modul nicht geladen: TalkChat",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=20)
        else:
            tk.Label(self.frame_chat, text="Modul nicht geladen: TalkChat",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=20)

    def _on_close(self):
        """Fenster schliessen: Preview stoppen, dann beenden."""
        if self._preview is not None:
            try:
                self._preview.stop()
            except Exception:
                pass
        self.root.destroy()

    def _poll_status(self):
        """Status vom Service pollen (alle STATUS_UPDATE_MS Millisekunden)."""
        status = self.service.read_status()
        if status:
            # FPS ist ein dict mit scrfd/arcface/yolov8m/hand_landmark/total
            try:
                fps_dict = status.get("fps", {})
                fps = float(fps_dict.get("total", 0.0)) if isinstance(fps_dict, dict) else 0.0
            except (TypeError, ValueError):
                fps = 0.0

            # Modus aus autonomous_mode ableiten
            auto = not status.get("manual_mode", True)
            mode = "AUTONOM" if auto else "MANUELL"

            self.status_bar.config(
                text=f"Service: aktiv | FPS: {fps:.1f} | Modus: {mode}",
                fg=STATUS_GREEN,
            )

            # ERKANNT-Indikator im eWeLink-Modul aktualisieren
            if self._ewelink is not None:
                self._ewelink.update_from_status(status)
        else:
            self.status_bar.config(
                text="Service: nicht verbunden",
                fg=FG_DIM,
            )

        # Widgets sofort neu zeichnen
        self.root.update_idletasks()

        # Naechster Poll
        self.root.after(STATUS_UPDATE_MS, self._poll_status)

    def register_module(self, name: str, frame: tk.Frame):
        """
        Modul registrieren.

        Args:
            name: Modulname (z.B. 'preview', 'ptz', 'chat')
            frame: Der Frame in dem das Modul lebt
        """
        self._modules[name] = frame
        self.logger.info(f"Modul registriert: {name}")

    def get_frame(self, column: str) -> tk.LabelFrame:
        """
        Frame fuer eine Spalte holen.

        Args:
            column: 'kamera', 'steuerung' oder 'chat'

        Returns:
            Der LabelFrame der Spalte
        """
        frames = {
            "kamera": self.frame_kamera,
            "steuerung": self.frame_steuerung,
            "chat": self.frame_chat,
        }
        return frames.get(column)

    def run(self):
        """Hauptschleife starten."""
        self.logger.info("M.O.L.O.C.H. 4.0 Panel gestartet")
        self.root.mainloop()


# =============================================================================
# main()
# =============================================================================

def main():
    """M.O.L.O.C.H. 4.0 Panel starten."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    panel = MolochPanel()
    panel.run()


if __name__ == "__main__":
    main()
