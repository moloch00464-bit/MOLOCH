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
import time
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

try:
    from core.gui.panel_voice import VoiceModule
    _VOICE_OK = True
except Exception:
    _VOICE_OK = False

try:
    from core.gui.panel_avatar import AvatarModule
    _AVATAR_OK = True
except Exception:
    _AVATAR_OK = False

try:
    from core.gui.panel_spotify import SpotifyModule
    _SPOTIFY_OK = True
except Exception:
    _SPOTIFY_OK = False

try:
    from core.gui.panel_systemstatus import SystemStatusModule
    _SYSTEMSTATUS_OK = True
except Exception:
    _SYSTEMSTATUS_OK = False


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

    def toggle_teachen(self):
        """Teachen an/aus."""
        self._write_command("toggle_teachen")

    # ----- Spotify Commands -----

    def spotify_play(self, uri: str = None):
        """Spotify Play (optional mit Track-URI)."""
        params = {"uri": uri} if uri else None
        self._write_command("spotify_play", params)

    def spotify_pause(self):
        """Spotify Pause."""
        self._write_command("spotify_pause")

    def spotify_toggle(self):
        """Spotify Play/Pause umschalten."""
        self._write_command("spotify_toggle")

    def spotify_skip(self):
        """Naechster Track."""
        self._write_command("spotify_skip")

    def spotify_previous(self):
        """Vorheriger Track."""
        self._write_command("spotify_previous")

    def spotify_volume(self, volume: int):
        """Volume setzen (0-100)."""
        self._write_command("spotify_volume", {"volume": volume})

    def spotify_search(self, query: str):
        """Suchen und abspielen."""
        self._write_command("spotify_search", {"query": query})

    def spotify_mood(self, zone: str):
        """Musik passend zur Zone spielen."""
        self._write_command("spotify_mood", {"zone": zone})

    def spotify_artist(self, name: str):
        """Musik von Artist spielen."""
        self._write_command("spotify_artist", {"artist": name})

    def spotify_auto_dj(self, state: str = "toggle"):
        """Auto-DJ steuern. state: 'on', 'off', 'toggle'."""
        self._write_command("spotify_auto_dj", {"state": state})

    def spotify_shuffle(self, state: bool = True):
        """Shuffle ein/ausschalten."""
        self._write_command("spotify_shuffle", {"state": state})

    def spotify_similar(self):
        """Aehnliche Musik zum aktuellen Track spielen."""
        self._write_command("spotify_similar")

    def spotify_top_tracks(self):
        """Markus' Top Tracks spielen."""
        self._write_command("spotify_top_tracks")

    def spotify_new_music(self):
        """Neue Musik entdecken basierend auf Profil."""
        self._write_command("spotify_new_music")

    def spotify_from_year(self, year: int):
        """Tracks aus bestimmtem Jahr spielen."""
        self._write_command("spotify_from_year", {"year": year})

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

        Format: 24 Byte Header (h, w, c, seq als uint32 LE + ts als float64 LE)
                danach h * w * c Bytes RGB raw

        Returns:
            (width, height, channels, raw_bytes) oder None bei Fehler
        """
        try:
            if not os.path.exists(SHM_FRAME):
                return None
            with open(SHM_FRAME, "rb") as f:
                header = f.read(24)
                if len(header) >= 24:
                    height, width, channels, _seq, _ts = struct.unpack("<IIIId", header)
                elif len(header) >= 12:
                    width, height, channels = struct.unpack("<III", header[:12])
                else:
                    return None
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

        # Fenster maximiert starten (1920x1080 Vollbild)
        try:
            self.root.attributes('-zoomed', True)
        except tk.TclError:
            self.root.geometry("1920x1050+0+0")

        # Layout aufbauen
        self._build_layout()

        # Module laden
        self._load_modules()

        # Fenster-Schliessen abfangen
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Status-Polling starten
        self._poll_status()

        # Watchdog starten (Gate0 Phase 9) — erster Check nach 10s (Startup-Grace)
        self._watchdog_last_poll = time.monotonic()
        self.root.after(10000, self._watchdog)

    def _build_layout(self):
        """3-Spalten-Layout mit Status-Bar aufbauen."""

        # Hauptcontainer
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Spalten konfigurieren — Chat schmal, Steuerung bekommt Platz
        main_frame.columnconfigure(0, weight=0)  # Kamera: feste Breite
        main_frame.columnconfigure(1, weight=1, minsize=400)  # Steuerung: flexibel
        main_frame.columnconfigure(2, weight=0, minsize=260)  # Chat: kompakt, fest
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

        # (A2) Avatar -> frame_kamera (unter Preview)
        self._avatar = None
        if _AVATAR_OK:
            try:
                self._avatar = AvatarModule(self.frame_kamera, self.service)
                self._avatar.start()
                self.logger.info("Modul geladen: Avatar")
            except Exception as e:
                self.logger.error(f"Avatar fehlgeschlagen: {e}")
        if self._avatar is None:
            tk.Label(self.frame_kamera, text="Modul nicht geladen: Avatar",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

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

        # (C2) Systemstatus -> frame_steuerung (unter eWeLink)
        self._systemstatus = None
        if _SYSTEMSTATUS_OK:
            try:
                self._systemstatus = SystemStatusModule(self.frame_steuerung, self.service)
                self.logger.info("Modul geladen: Systemstatus")
            except Exception as e:
                self.logger.error(f"Systemstatus fehlgeschlagen: {e}")

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

        # (F) Voice -> frame_chat (unter TalkChat)
        if _VOICE_OK:
            try:
                VoiceModule(self.frame_chat, self.service)
                self.logger.info("Modul geladen: Voice")
            except Exception as e:
                self.logger.error(f"Voice fehlgeschlagen: {e}")
                tk.Label(self.frame_chat, text="Modul nicht geladen: Voice",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)
        else:
            tk.Label(self.frame_chat, text="Modul nicht geladen: Voice",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

        # (G) Spotify -> frame_chat (unter Voice)
        if _SPOTIFY_OK:
            try:
                SpotifyModule(self.frame_chat, self.service)
                self.logger.info("Modul geladen: Spotify")
            except Exception as e:
                self.logger.error(f"Spotify fehlgeschlagen: {e}")
                tk.Label(self.frame_chat, text="Modul nicht geladen: Spotify",
                         bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)
        else:
            tk.Label(self.frame_chat, text="Modul nicht geladen: Spotify",
                     bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack(pady=5)

    def _on_close(self):
        """Fenster schliessen: Preview + Avatar stoppen, dann beenden."""
        if self._preview is not None:
            try:
                self._preview.stop()
            except Exception:
                pass
        if self._avatar is not None:
            try:
                self._avatar.stop()
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

            # Bridge State + Face-ID + Zone fuer Status-Bar
            bridge = status.get("bridge", {})
            bridge_state = bridge.get("state", "?")
            face_id = status.get("face_id", "")
            face_sim = status.get("face_similarity", 0.0)
            zone = status.get("core", {}).get("zone", "")
            tension = status.get("core", {}).get("tension", 0.0)

            face_str = f"{face_id}({face_sim:.0%})" if face_id else "---"
            zone_str = zone.upper() if zone else "?"

            self.status_bar.config(
                text=(f"Service: aktiv | FPS: {fps:.1f} | {mode} | "
                      f"Bridge: {bridge_state} | Face: {face_str} | "
                      f"Zone: {zone_str} | T: {tension:.2f}"),
                fg=STATUS_GREEN,
            )

            # Module mit aktuellem Status aktualisieren
            if self._ewelink is not None:
                self._ewelink.update_from_status(status)
            if self._avatar is not None:
                self._avatar.update_from_status(status)
        else:
            self.status_bar.config(
                text="Service: nicht verbunden",
                fg=FG_DIM,
            )

        # Naechster Poll
        self.root.after(STATUS_UPDATE_MS, self._poll_status)

    def _watchdog(self):
        """Panel-Watchdog: Prueft ob Module noch rendern (Gate0 Phase 9).

        Laeuft alle 5 Sekunden. Warnt wenn Preview oder Avatar > 2s nicht
        gerendert hat. Versucht after-Chain neu zu starten wenn moeglich.
        """
        try:
            now = time.monotonic()

            # Preview pruefen
            if self._preview is not None and hasattr(self._preview, 'last_render_time'):
                last = self._preview.last_render_time
                if last > 0 and (now - last) > 2.0:
                    self.logger.warning(
                        f"[WARNUNG] Panel render_timeout modul=preview "
                        f"stale_sec={now - last:.1f}"
                    )
                    # after-Chain neu starten
                    if self._preview._running:
                        try:
                            self._preview._after_id = self._preview._parent.after(
                                100, self._preview._update
                            )
                            self.logger.info("[WATCHDOG] Preview after-Chain neugestartet")
                        except Exception:
                            pass

            # Avatar pruefen
            if self._avatar is not None and hasattr(self._avatar, 'last_render_time'):
                last = self._avatar.last_render_time
                if last > 0 and (now - last) > 2.0:
                    self.logger.warning(
                        f"[WARNUNG] Panel render_timeout modul=avatar "
                        f"stale_sec={now - last:.1f}"
                    )
                    # after-Chain neu starten
                    if self._avatar._running:
                        try:
                            self._avatar._after_id = self._avatar._parent.after(
                                100, self._avatar._update_animation
                            )
                            self.logger.info("[WATCHDOG] Avatar after-Chain neugestartet")
                        except Exception:
                            pass

            # Status-Poll selbst pruefen (wenn _poll_status haengt)
            poll_gap = now - self._watchdog_last_poll
            if poll_gap > 3.0:
                self.logger.warning(
                    f"[WARNUNG] Panel poll_timeout stale_sec={poll_gap:.1f}"
                )

            self._watchdog_last_poll = now

        except Exception as e:
            self.logger.error(f"[WATCHDOG] Fehler: {e}")

        # Naechster Check in 5 Sekunden
        self.root.after(5000, self._watchdog)

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
