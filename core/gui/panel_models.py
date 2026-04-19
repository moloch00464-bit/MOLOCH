#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Models
===========================

Model Steuerung und Popup-Buttons.
Bekommt parent_frame (LabelFrame) und ServiceProxy von panel_main.

- Model Checkboxen: SCRFD, ArcFace, YOLOv11m, Hand Landmark, Pose
- FPS Anzeige (STATUS_YELLOW, 500ms Update)
- SAVE SETTINGS Button
- Popup-Buttons Reihe: AUDIO, HARDWARE, NPU THRESH, SETTINGS

Alle Commands via ServiceProxy._write_command().
Importiert NUR panel_styles und tkinter.
"""

import json
import os
import tempfile
import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON,
    BTN_ON_GREEN, BTN_OFF_DARK,
    ACCENT_GREEN, ACCENT_CYAN,
    STATUS_YELLOW, STATUS_GREEN,
    FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL, FONT_MONO,
    STATUS_UPDATE_MS,
)

# Pfade fuer LLM-Profile und Settings
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
_SETTINGS_PATH = os.path.join(_CONFIG_DIR, "settings.json")
_LLM_PROFILES_PATH = os.path.join(_CONFIG_DIR, "llm_profiles.json")
_CAPS_PATH = os.path.join(_CONFIG_DIR, "system_capabilities.json")

# Audio Popup importieren (optional, Fehler faengt Panel nicht ab)
try:
    from core.gui.popups.popup_audio import AudioPopup
except ImportError:
    AudioPopup = None

# Hardware Popup importieren (optional)
try:
    from core.gui.popups.popup_hardware import HardwarePopup
except ImportError:
    HardwarePopup = None

# NPU Thresh Popup importieren (optional, NEUES Popup mit MPO + Gesten)
try:
    from core.gui.popups.popup_npu_thresh import NpuThreshPopup
except ImportError:
    NpuThreshPopup = None

# Tracker Popup importieren (optional)
try:
    from core.gui.popups.popup_tracker import TrackerPopup
except ImportError:
    TrackerPopup = None

# Settings Popup importieren (optional)
try:
    from core.gui.popups.popup_settings import SettingsPopup
except ImportError:
    SettingsPopup = None

# Dashboard Popup importieren (optional)
try:
    from core.gui.popups.popup_dashboard import DashboardPopup
except ImportError:
    DashboardPopup = None

# Whisper Popup importieren (optional)
try:
    from core.gui.popups.popup_whisper import WhisperPopup
except ImportError:
    WhisperPopup = None

# Supervisor Popup importieren (optional)
try:
    from core.gui.popups.popup_supervisor import SupervisorPopup
except ImportError:
    SupervisorPopup = None

# PiPower5 Popup importieren (optional)
try:
    from core.gui.popups.popup_pipower5 import PiPower5Popup
except ImportError:
    PiPower5Popup = None


class _Tooltip:
    """Leichter Tkinter-Tooltip — Toplevel-Popup beim Hover ueber ein Widget.

    Erscheint mit kleinem Delay (350ms) damit kein Flackern bei schnellem Drueberfahren.
    Bricht bei Leave/Click sauber ab.
    """

    def __init__(self, widget, text: str, delay_ms: int = 350, wrap_px: int = 360):
        self._widget = widget
        self._text = text
        self._delay = delay_ms
        self._wrap = wrap_px
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _evt=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self._widget.winfo_rootx() + 16
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        try:
            lbl = tk.Label(
                self._tip, text=self._text, justify=tk.LEFT,
                bg="#1f1f1f", fg=FG_WHITE, font=FONT_SMALL,
                relief=tk.SOLID, borderwidth=1,
                wraplength=self._wrap, padx=8, pady=6,
            )
            lbl.pack()
        except Exception:
            self._hide()

    def _hide(self, _evt=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def update_text(self, text: str):
        self._text = text


class ModelsModule:
    """Model Controls und Popup-Buttons im uebergebenen LabelFrame."""

    # TAPPAS Pipeline Modelle (immer aktiv, nicht togglebar)
    TAPPAS_MODELS = [
        ("SCRFD", "scrfd"),
        ("ArcFace", "arcface"),
        ("YOLOv11m", "yolov8m"),
        ("FaceAttr", "faceattr"),
    ]

    # Zusaetzliche Modelle (HEFs vorhanden, togglebar)
    EXTRA_MODELS = [
        ("Pose", "pose"),
        ("Hand LM", "hand_landmark"),
        ("Person-ReID", "person_reid"),
    ]

    # Alle Modelle zusammen (fuer Kompatibilitaet)
    MODELS = TAPPAS_MODELS + EXTRA_MODELS

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: LabelFrame von panel_main
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None

        # Checkbox-Variablen
        self._model_vars = {}
        for _, key in self.MODELS:
            self._model_vars[key] = tk.BooleanVar(value=False)

        # Popup-Callbacks (werden von panel_main spaeter gesetzt)
        self.on_popup_audio = self._open_audio_popup
        self.on_popup_hardware = self._open_hardware_popup
        self.on_popup_npu = self._open_npu_popup
        self.on_popup_tracker = self._open_tracker_popup
        self.on_popup_settings = self._open_settings_popup
        self.on_popup_dashboard = self._open_dashboard_popup
        self.on_popup_whisper = self._open_whisper_popup
        self.on_popup_supervisor = self._open_supervisor_popup
        self.on_popup_pipower5 = self._open_pipower5_popup

        # LLM-Modus: Profil-Variable + Referenz auf Buttons
        self._llm_profile_var = tk.StringVar(value="")
        self._llm_btn_refs = {}   # key → Button-Widget

        # GUI aufbauen
        self._build_pipeline_status()
        self._build_npu_status()
        self._build_model_checkboxes()
        self._build_fps_display()
        self._build_llm_profile_section()
        self._build_save_button()
        self._build_popup_buttons()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # Audio Popup oeffnen
    # =========================================================================

    def _open_audio_popup(self):
        """Audio Settings Popup oeffnen."""
        if AudioPopup is not None:
            AudioPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_audio.py konnte nicht importiert werden")

    # =========================================================================
    # Hardware Popup oeffnen
    # =========================================================================

    def _open_hardware_popup(self):
        """Hardware Monitor Popup oeffnen."""
        if HardwarePopup is not None:
            HardwarePopup(self._parent, self._service)
        else:
            print("FEHLER: popup_hardware.py konnte nicht importiert werden")

    # =========================================================================
    # NPU Thresh Popup oeffnen
    # =========================================================================

    def _open_npu_popup(self):
        """NPU Thresholds Popup oeffnen."""
        if NpuThreshPopup is not None:
            NpuThreshPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_npu.py konnte nicht importiert werden")

    # =========================================================================
    # Settings Popup oeffnen
    # =========================================================================

    def _open_tracker_popup(self):
        """Tracker Settings Popup oeffnen."""
        if TrackerPopup is not None:
            TrackerPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_tracker.py konnte nicht importiert werden")

    def _open_settings_popup(self):
        """Settings Popup oeffnen."""
        if SettingsPopup is not None:
            SettingsPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_settings.py konnte nicht importiert werden")

    def _open_dashboard_popup(self):
        """NPU Dashboard Popup oeffnen."""
        if DashboardPopup is not None:
            DashboardPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_dashboard.py konnte nicht importiert werden")

    def _open_whisper_popup(self):
        """Whisper STT Monitor Popup oeffnen."""
        if WhisperPopup is not None:
            WhisperPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_whisper.py konnte nicht importiert werden")

    def _open_supervisor_popup(self):
        """Supervisor Dashboard Popup oeffnen."""
        if SupervisorPopup is not None:
            SupervisorPopup(self._parent, self._service)
        else:
            print("FEHLER: popup_supervisor.py konnte nicht importiert werden")

    def _open_pipower5_popup(self):
        """PiPower5 HAT+ Monitor Popup oeffnen."""
        if PiPower5Popup is not None:
            PiPower5Popup(self._parent)
        else:
            print("FEHLER: popup_pipower5.py konnte nicht importiert werden")

    # =========================================================================
    # Pipeline Status (TAPPAS/Legacy Anzeige)
    # =========================================================================

    def _build_pipeline_status(self):
        """TAPPAS/Legacy Pipeline Status-Zeile."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=10, pady=(5, 0))

        tk.Label(
            row, text="Pipeline:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._lbl_pipeline = tk.Label(
            row, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_pipeline.pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # NPU Scheduler + Tracking Status
    # =========================================================================

    def _build_npu_status(self):
        """NPU Scheduler Status + Tracking Source Anzeige."""
        section = tk.LabelFrame(
            self._parent,
            text="NPU Status",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 2))

        # Zeile 1: Aktive Modelle (Scheduler)
        row1 = tk.Frame(section, bg=BG_FRAME)
        row1.pack(fill=tk.X, padx=8, pady=(4, 1))

        tk.Label(
            row1, text="Aktiv:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_npu_models = tk.Label(
            row1, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
        )
        self._lbl_npu_models.pack(side=tk.LEFT, padx=5)

        # Zeile 2: Tracking Source
        row2 = tk.Frame(section, bg=BG_FRAME)
        row2.pack(fill=tk.X, padx=8, pady=(1, 4))

        self._lbl_tracking_src = tk.Label(
            row2, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_tracking_src.pack(side=tk.LEFT)

    def _update_npu_status_display(self, status):
        """NPU Scheduler Mode + Tracking Source aus Status aktualisieren."""
        # Szenario-Anzeige (Perception Router)
        sched = status.get("npu_sched_mode", "")
        _SCENARIO_DISPLAY = {
            "IDLE":     ("IDLE — YOLO only", FG_DIM),
            "FERN":     ("FERN — YOLO+ReID+Pose", STATUS_YELLOW),
            "MITTEL":   ("MITTEL — YOLO+SCRFD+ArcFace+Pose", ACCENT_CYAN),
            "NAH":      ("NAH — SCRFD+ArcFace+Hand", ACCENT_GREEN),
            "RUECKEN":  ("RUECKEN — YOLO+ReID+Pose", STATUS_YELLOW),
            "MULTI":    ("MULTI — Alle aktiv", ACCENT_GREEN),
            "NACHT":    ("NACHT — Schlafmodus", FG_DIM),
        }
        if sched in _SCENARIO_DISPLAY:
            text, color = _SCENARIO_DISPLAY[sched]
            self._lbl_npu_models.config(text=text, fg=color)
        elif sched in ("ALL_ACTIVE", "YOLO_SCRFD", "YOLO_ONLY"):
            # Legacy-Fallback
            legacy = {"ALL_ACTIVE": ("YOLO+SCRFD+ArcFace", ACCENT_GREEN),
                       "YOLO_SCRFD": ("YOLO+SCRFD", STATUS_YELLOW),
                       "YOLO_ONLY": ("YOLO only", FG_DIM)}
            text, color = legacy[sched]
            self._lbl_npu_models.config(text=text, fg=color)
        elif not sched:
            active = status.get("active_models", [])
            if active:
                self._lbl_npu_models.config(text=" + ".join(active), fg=ACCENT_CYAN)

        # Tracking Source: PTZ Arbiter Mode
        arbiter = status.get("ptz_arbiter_mode", "")
        if arbiter == "MOLOCH_AUTONOM":
            self._lbl_tracking_src.config(
                text="\U0001f535 Moloch Tracking aktiv", fg=ACCENT_CYAN,
            )
        elif arbiter == "MOLOCH_MANUELL":
            self._lbl_tracking_src.config(
                text="\U0001f7e1 Manuell uebernommen", fg=STATUS_YELLOW,
            )
        elif arbiter:
            self._lbl_tracking_src.config(text=arbiter, fg=FG_DIM)

    # =========================================================================
    # Model Checkboxen
    # =========================================================================

    def _build_model_checkboxes(self):
        """Checkbuttons: TAPPAS-Modelle (locked) + Extra-Modelle (togglebar)."""
        section = tk.LabelFrame(
            self._parent,
            text="Modelle",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Zeile 1: TAPPAS Pipeline Modelle (immer aktiv)
        row_tappas = tk.Frame(section, bg=BG_FRAME)
        row_tappas.pack(pady=(5, 0), padx=5)

        self._checkboxes = {}
        for i, (label, key) in enumerate(self.TAPPAS_MODELS):
            cb = tk.Checkbutton(
                row_tappas, text=label,
                variable=self._model_vars[key],
                bg=BG_FRAME, fg=ACCENT_GREEN,
                selectcolor=BG_FRAME,
                activebackground=BG_FRAME,
                activeforeground=ACCENT_GREEN,
                disabledforeground=ACCENT_GREEN,
                font=FONT_BUTTON,
                state=tk.DISABLED,
            )
            cb.grid(row=0, column=i, padx=6, pady=2)
            self._checkboxes[key] = cb

        # Label "TAPPAS" rechts neben den Checkboxen
        tk.Label(
            row_tappas, text="TAPPAS",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).grid(row=0, column=len(self.TAPPAS_MODELS), padx=(10, 0))

        # Zeile 2: Extra-Modelle (togglebar)
        if self.EXTRA_MODELS:
            row_extra = tk.Frame(section, bg=BG_FRAME)
            row_extra.pack(pady=(0, 5), padx=5)

            for i, (label, key) in enumerate(self.EXTRA_MODELS):
                cb = tk.Checkbutton(
                    row_extra, text=label,
                    variable=self._model_vars[key],
                    bg=BG_FRAME, fg=FG_WHITE,
                    selectcolor=BG_FRAME,
                    activebackground=BG_FRAME,
                    activeforeground=FG_WHITE,
                    font=FONT_BUTTON,
                    command=lambda k=key: self._toggle_model(k),
                )
                cb.grid(row=0, column=i, padx=6, pady=2)
                self._checkboxes[key] = cb

    def _toggle_model(self, model_key):
        """Model an/aus und Command senden."""
        enabled = self._model_vars[model_key].get()
        self._service._write_command("toggle_model", {
            "model": model_key,
            "enabled": enabled,
        })

    # =========================================================================
    # FPS Anzeige
    # =========================================================================

    def _build_fps_display(self):
        """FPS Anzeige: Total + pro Modell."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            row, text="FPS:", bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        ).pack(side=tk.LEFT)

        self._lbl_fps = tk.Label(
            row, text="--",
            bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_MONO,
        )
        self._lbl_fps.pack(side=tk.LEFT, padx=5)

        # Detail-FPS (scrfd/arcface/yolo)
        self._lbl_fps_detail = tk.Label(
            row, text="",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_fps_detail.pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # LLM-Modus Sektion
    # =========================================================================

    def _load_llm_profiles(self):
        """Laedt config/llm_profiles.json. Gibt (profiles_dict, active_key) zurueck.
        Bei fehlender Datei: leeres Dict + None.
        """
        if not os.path.exists(_LLM_PROFILES_PATH):
            return {}, None
        try:
            with open(_LLM_PROFILES_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            profiles = data.get("profiles", {})
            active = data.get("active", None)
            return profiles, active
        except Exception:
            return {}, None

    def _read_active_from_settings(self):
        """Liest llm_profile Key aus settings.json (ueberschreibt profiles.json active)."""
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            return s.get("llm_profile", None)
        except Exception:
            return None

    def _write_llm_profile_to_settings(self, profile_key: str):
        """Schreibt llm_profile atomar in settings.json (NEVER #6: atomic write)."""
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["llm_profile"] = profile_key
            fd, tmp = tempfile.mkstemp(dir=_CONFIG_DIR, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                os.replace(tmp, _SETTINGS_PATH)
            except OSError:
                # NTFS-Fallback
                with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception as e:
            print(f"[panel_models] LLM-Profil schreiben fehlgeschlagen: {e}")

    def _build_llm_profile_section(self):
        """LLM-Modus Sektion: aktuelles Profil + Radio-Buttons pro Profil."""
        self._llm_section = tk.LabelFrame(
            self._parent,
            text="LLM-Modus",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        self._llm_section.pack(fill=tk.X, padx=5, pady=(2, 2))

        # Status-Zeile: "aktiv: chat"
        status_row = tk.Frame(self._llm_section, bg=BG_FRAME)
        status_row.pack(fill=tk.X, padx=8, pady=(4, 2))

        tk.Label(
            status_row, text="aktiv:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)

        self._lbl_llm_active = tk.Label(
            status_row, text="--", bg=BG_FRAME, fg=ACCENT_CYAN, font=FONT_MONO,
        )
        self._lbl_llm_active.pack(side=tk.LEFT, padx=5)

        # Tentakel-Status (eigene Zeile): zeigt Ollama auf Markus-Rechner
        tentacle_row = tk.Frame(self._llm_section, bg=BG_FRAME)
        tentacle_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            tentacle_row, text="Tentakel:", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        ).pack(side=tk.LEFT)
        self._lbl_tentacle = tk.Label(
            tentacle_row, text="--", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_tentacle.pack(side=tk.LEFT, padx=5)

        # Profil-Buttons (werden beim ersten _update_llm_section befuellt)
        self._llm_btn_frame = tk.Frame(self._llm_section, bg=BG_FRAME)
        self._llm_btn_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Anzeige-Namen fuer Profil-Keys
        self._llm_display_names = {
            "chat":         "Chat",
            "introspect":   "Introspect",
            "technical":    "Technical",
            "dark":         "Dark",
            "multi_person": "Multi",
        }

        # Einmalig aufbauen
        self._llm_last_profile_keys = []
        self._update_llm_section()

    def _update_llm_section(self):
        """Profile aus JSON laden, Buttons ggf. neu aufbauen, aktives Profil highlighten."""
        profiles, active_from_file = self._load_llm_profiles()

        # settings.json ueberschreibt active
        active_from_settings = self._read_active_from_settings()
        active = active_from_settings or active_from_file

        profile_keys = list(profiles.keys())

        # Buttons nur neu bauen wenn sich Profile geaendert haben
        if profile_keys != self._llm_last_profile_keys:
            # Alte Buttons loeschen
            for w in self._llm_btn_frame.winfo_children():
                w.destroy()
            self._llm_btn_refs = {}

            if not profile_keys:
                # Platzhalter wenn profiles.json noch nicht existiert
                tk.Label(
                    self._llm_btn_frame,
                    text="LLM-Profile noch nicht initialisiert",
                    bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
                ).pack(side=tk.LEFT, pady=2)
            else:
                # Profile-Beschreibungen — was jeder Modus tatsaechlich tut.
                # Kurz und konkret, keine Marketing-Sprache.
                profile_help = {
                    "chat":         "Normale Konversation. Kurz, frech, dunkel-humorvoll. Default-Modus fuer alltaegliche Fragen.",
                    "introspect":   "Selbstreflexion mit Live-Vision + Inner State (Person, Zone, Tension, Dominance). Persoenlich, oft poetisch. Fuer 'Wie geht es dir?', 'Was siehst du?'.",
                    "technical":    "Sachlich-praezise Antworten. Fakten zuerst, Persoenlichkeit minimal. Fuer technische Fragen, Diagnosen, Erklaerungen.",
                    "dark":         "Berserker-Modus. Ein Satz, scharf, trocken-bissig. Hoechste Temperatur — kann unvorhersehbar sein.",
                    "multi_person": "Mehrere Personen vor der Kamera unterscheiden ('markus macht X, rebecca macht Y'). Mit Live-Vision-Kontext.",
                }
                for i, key in enumerate(profile_keys):
                    display = self._llm_display_names.get(key, key.capitalize())
                    meta = profiles[key]
                    max_tok = meta.get("max_tokens", "?")
                    temp = meta.get("temperature", "?")
                    live = "ja" if meta.get("include_live_context") else "nein"
                    desc = profile_help.get(key, "")
                    tooltip_txt = (
                        f"{display}  ({key})\n"
                        f"---\n"
                        f"{desc}\n"
                        f"---\n"
                        f"max_tokens={max_tok}  |  temperature={temp}  |  Live-Kontext: {live}"
                    )
                    btn = tk.Button(
                        self._llm_btn_frame,
                        text=display,
                        width=8,
                        bg=BG_BUTTON,
                        fg=FG_LABEL,
                        font=FONT_BUTTON,
                        activebackground=BG_FRAME,
                        command=lambda k=key: self._select_llm_profile(k),
                    )
                    btn.grid(row=0, column=i, padx=2, pady=2)
                    self._llm_btn_refs[key] = btn
                    # Echtes Tooltip-Popup (Toplevel) — kein Button-Text-Swap mehr.
                    _Tooltip(btn, tooltip_txt)

            self._llm_last_profile_keys = profile_keys

        # Aktives Profil highlighten + Label setzen
        if active:
            self._llm_profile_var.set(active)
            self._lbl_llm_active.config(text=active, fg=ACCENT_CYAN)
            for key, btn in self._llm_btn_refs.items():
                display = self._llm_display_names.get(key, key.capitalize())
                if key == active:
                    btn.config(bg=BTN_ON_GREEN, fg=FG_WHITE)
                else:
                    btn.config(bg=BG_BUTTON, fg=FG_LABEL, text=display)
        else:
            self._lbl_llm_active.config(text="--", fg=FG_DIM)

        # Tentakel-Status aus system_capabilities.json lesen
        self._update_tentacle_status()

    def _update_tentacle_status(self):
        """Tentakel-Anzeige aktualisieren. Quelle: system_capabilities.json + settings.json."""
        enabled = True
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            enabled = bool((s.get("tentacle_llm", {}) or {}).get("enabled", True))
        except Exception:
            pass
        if not enabled:
            self._lbl_tentacle.config(text="deaktiviert", fg=FG_DIM)
            return
        try:
            with open(_CAPS_PATH, "r", encoding="utf-8") as fh:
                caps = json.load(fh)
            t = (caps.get("tentacle_llm", {}) or {})
            reachable = bool(t.get("reachable", False))
            model = t.get("model") or ""
        except Exception:
            reachable = False
            model = ""
        if reachable:
            label = f"online ({model})" if model else "online"
            self._lbl_tentacle.config(text=label, fg=STATUS_GREEN)
        else:
            self._lbl_tentacle.config(text="offline — nur NPU", fg=FG_DIM)

    def _select_llm_profile(self, profile_key: str):
        """Profil auswaehlen: settings.json schreiben + GUI sofort aktualisieren."""
        self._write_llm_profile_to_settings(profile_key)
        self._llm_profile_var.set(profile_key)
        self._lbl_llm_active.config(text=profile_key, fg=ACCENT_CYAN)
        # Buttons neu highlighten
        for key, btn in self._llm_btn_refs.items():
            display = self._llm_display_names.get(key, key.capitalize())
            if key == profile_key:
                btn.config(bg=BTN_ON_GREEN, fg=FG_WHITE)
            else:
                btn.config(bg=BG_BUTTON, fg=FG_LABEL, text=display)

    # =========================================================================
    # SAVE SETTINGS
    # =========================================================================

    def _build_save_button(self):
        """SAVE SETTINGS Button + BLITZ Toggle + Feedback-Label."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=2)

        tk.Button(
            row, text="SAVE SETTINGS", width=16,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._save_settings,
        ).pack(side=tk.LEFT, padx=5)

        self._lbl_save = tk.Label(
            row, text="", bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_save.pack(side=tk.LEFT, padx=5)

    def _save_settings(self):
        """Settings speichern und kurzes Feedback zeigen."""
        self._service.save_settings()
        self._lbl_save.config(text="Gespeichert!", fg=ACCENT_GREEN)
        self._parent.after(2000, lambda: self._lbl_save.config(
            text="", fg=FG_DIM
        ))

    # =========================================================================
    # Popup-Buttons
    # =========================================================================

    def _build_popup_buttons(self):
        """Popup-Buttons: AUDIO, HARDWARE, NPU THRESH, SETTINGS."""
        section = tk.LabelFrame(
            self._parent,
            text="Popups",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=(2, 5))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        popup_defs = [
            ("AUDIO", lambda: self.on_popup_audio()),
            ("HARDWARE", lambda: self.on_popup_hardware()),
            ("NPU/MPO", lambda: self.on_popup_npu()),
            ("TRACKER", lambda: self.on_popup_tracker()),
            ("WHISPER", lambda: self.on_popup_whisper()),
            ("DASHBOARD", lambda: self.on_popup_dashboard()),
            ("SUPERVISOR", lambda: self.on_popup_supervisor()),
            ("PIPOWER5", lambda: self.on_popup_pipower5()),
            ("SETTINGS", lambda: self.on_popup_settings()),
        ]

        for i, (label, cmd) in enumerate(popup_defs):
            tk.Button(
                row, text=label, width=9,
                bg=BG_BUTTON, fg=FG_LABEL, font=FONT_BUTTON,
                activebackground=BG_FRAME,
                command=cmd,
            ).grid(row=i // 5, column=i % 5, padx=2, pady=2)

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Status vom Service lesen: Pipeline, FPS und aktive Modelle aktualisieren."""
        status = self._service.read_status()

        if status:
            # Pipeline-Typ (TAPPAS oder Legacy)
            active_models = status.get("active_models", [])
            is_tappas = len(active_models) >= 3 and "scrfd" in active_models
            if is_tappas:
                self._lbl_pipeline.config(text="TAPPAS", fg=ACCENT_GREEN)
            else:
                self._lbl_pipeline.config(text="Legacy", fg=STATUS_YELLOW)

            # FPS Total
            try:
                fps_dict = status.get("fps", {})
                fps = float(fps_dict.get("total", 0.0)) if isinstance(fps_dict, dict) else 0.0
                self._lbl_fps.config(text=f"{fps:.1f}")

                # Detail-FPS pro Modell
                if isinstance(fps_dict, dict):
                    parts = []
                    for name in ["scrfd", "arcface", "yolov8m", "faceattr", "pose"]:
                        val = fps_dict.get(name, 0.0)
                        if val and float(val) > 0:
                            parts.append(f"{name}:{float(val):.0f}")
                    self._lbl_fps_detail.config(text="  ".join(parts))
            except (TypeError, ValueError):
                self._lbl_fps.config(text="--")
                self._lbl_fps_detail.config(text="")

            # Model-Checkboxen Status synchronisieren
            status_key_map = {
                "scrfd": "scrfd_active",
                "arcface": "arcface_active",
                "yolov8m": "yolo_active",
                "faceattr": "faceattr_active",
                "hand_landmark": "hand_active",
                "pose": "pose_active",
                "person_reid": "person_reid_active",
            }

            # TAPPAS-Modelle: aktiv wenn in active_models Liste
            for _, key in self.TAPPAS_MODELS:
                try:
                    active = key in active_models or \
                             bool(status.get(status_key_map.get(key, key), False))
                    if self._model_vars[key].get() != active:
                        self._model_vars[key].set(active)
                except (TypeError, ValueError):
                    pass

            # Extra-Modelle: aus Status-Keys
            for _, key in self.EXTRA_MODELS:
                try:
                    status_key = status_key_map.get(key, f"{key}_active")
                    active = bool(status.get(status_key, False))
                    if self._model_vars[key].get() != active:
                        self._model_vars[key].set(active)
                except (TypeError, ValueError):
                    pass

            # NPU Scheduler + Tracking Status aktualisieren
            self._update_npu_status_display(status)

        # LLM-Profil Anzeige aktualisieren (liest profiles.json + settings.json)
        self._update_llm_section()

        # Widgets sofort neu zeichnen
        self._parent.update_idletasks()

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)
