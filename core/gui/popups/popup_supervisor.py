#!/usr/bin/env python3
"""
M.O.L.O.C.H. Supervisor Dashboard Popup
=========================================

Toplevel-Fenster das alle 2 Sekunden localhost:5000/moloch/diagnostics abfragt.

Sektionen:
- Ampel oben: Basiert auf Health Score (0-100)
  90-100=GRUEN, 75-89=GELB, 50-74=ORANGE, unter 50=ROT
- System-Zeile: FPS, CPU-Temp, RAM%, Luefterstufe, Uptime
- Core-Zeile: Bridge-State, Face-ID, Tension, Mood
- Nervensystem: 5 Pipeline-Verbindungen mit Farb-Status
- Warnungen-Liste (aus "warnungen" Array + Pipeline-Alerts)
- Letzte 5 Events vom Event-Bus

Pollt NUR wenn offen, stoppt beim Schliessen.
Importiert NUR panel_styles und tkinter + urllib.
"""

import json
import logging
import tkinter as tk
import urllib.request
from typing import Dict, Any, List

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_YELLOW, STATUS_RED,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
)

logger = logging.getLogger("moloch.popup_supervisor")

# Poll-Intervall in ms
POLL_MS = 2000

# Diagnostics API URL
DIAG_URL = "http://localhost:5000/moloch/diagnostics"

# Ampel-Groesse
AMPEL_RADIUS = 20

# Orange fuer mittlere Warnstufe (50-74 Score)
STATUS_ORANGE = "#ff8800"

# Pipeline-Labels (Key -> Anzeigename)
PIPELINE_LABELS = {
    "vision_core": "Vision \u2192 Core",
    "core_bridge": "Core \u2192 Bridge",
    "bridge_tracker": "Bridge \u2192 Tracker",
    "esp_audio": "ESP \u2192 Audio",
    "feedback_loop": "Feedback Loop",
}

# Farben pro Status
_STATUS_COLORS = {
    "OK": STATUS_GREEN,
    "DEGRADED": STATUS_YELLOW,
    "BROKEN": STATUS_RED,
    "MISSING": STATUS_RED,
}


class SupervisorPopup:
    """Supervisor Dashboard als Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy
        self._after_id = None
        self._event_history: List[str] = []

        # Toplevel
        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Supervisor Dashboard")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("500x620")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # GUI aufbauen
        self._build_ampel()
        self._build_system_zeile()
        self._build_core_zeile()
        self._build_nervensystem()
        self._build_warnungen()
        self._build_events()

        # Erster Poll
        self._poll()

    # =========================================================================
    # Ampel (oben) — jetzt Score-basiert
    # =========================================================================

    def _build_ampel(self):
        """Ampel-Kreis oben mit Health Score."""
        frame = tk.Frame(self.win, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self._ampel_canvas = tk.Canvas(
            frame, width=AMPEL_RADIUS * 2 + 4, height=AMPEL_RADIUS * 2 + 4,
            bg=BG_DARK, highlightthickness=0,
        )
        self._ampel_canvas.pack(side=tk.LEFT)

        self._lbl_ampel = tk.Label(
            frame, text="Verbinde...",
            bg=BG_DARK, fg=FG_DIM, font=FONT_TITLE,
        )
        self._lbl_ampel.pack(side=tk.LEFT, padx=10)

        # Health Score rechts
        self._lbl_score = tk.Label(
            frame, text="",
            bg=BG_DARK, fg=FG_DIM, font=FONT_LABEL,
        )
        self._lbl_score.pack(side=tk.RIGHT, padx=10)

        # Initial grau
        self._draw_ampel(FG_DIM)

    def _draw_ampel(self, color: str):
        """Ampel-Kreis zeichnen."""
        c = self._ampel_canvas
        c.delete("all")
        r = AMPEL_RADIUS
        c.create_oval(2, 2, r * 2 + 2, r * 2 + 2, fill=color, outline=FG_DIM, width=1)

    # =========================================================================
    # System-Zeile: FPS, CPU, RAM, Luefter, Uptime
    # =========================================================================

    def _build_system_zeile(self):
        """Einzeilige System-Metriken."""
        section = tk.LabelFrame(
            self.win, text="System",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        self._lbl_system = tk.Label(
            section, text="---",
            bg=BG_FRAME, fg=FG_WHITE, font=FONT_MONO,
            anchor=tk.W, justify=tk.LEFT,
        )
        self._lbl_system.pack(fill=tk.X, padx=8, pady=5)

    # =========================================================================
    # Core-Zeile: Bridge, Face, Tension, Mood
    # =========================================================================

    def _build_core_zeile(self):
        """Einzeilige Core-State Metriken."""
        section = tk.LabelFrame(
            self.win, text="Core",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        self._lbl_core = tk.Label(
            section, text="---",
            bg=BG_FRAME, fg=FG_WHITE, font=FONT_MONO,
            anchor=tk.W, justify=tk.LEFT,
        )
        self._lbl_core.pack(fill=tk.X, padx=8, pady=5)

    # =========================================================================
    # Nervensystem — 5 Pipeline-Verbindungen
    # =========================================================================

    def _build_nervensystem(self):
        """5 Pipeline-Zeilen mit farbigem Status."""
        section = tk.LabelFrame(
            self.win, text="Nervensystem",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        self._pipe_labels: Dict[str, tk.Label] = {}

        for key, display_name in PIPELINE_LABELS.items():
            row = tk.Frame(section, bg=BG_FRAME)
            row.pack(fill=tk.X, padx=8, pady=1)

            # Pipeline-Name links
            tk.Label(
                row, text=f"  {display_name}:",
                bg=BG_FRAME, fg=FG_LABEL, font=FONT_MONO,
                anchor=tk.W, width=22,
            ).pack(side=tk.LEFT)

            # Status rechts (wird live aktualisiert)
            lbl = tk.Label(
                row, text="---",
                bg=BG_FRAME, fg=FG_DIM, font=FONT_MONO,
                anchor=tk.W,
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._pipe_labels[key] = lbl

    # =========================================================================
    # Warnungen-Liste
    # =========================================================================

    def _build_warnungen(self):
        """Warnungen aus dem diagnostics-Array + Pipeline-Alerts."""
        section = tk.LabelFrame(
            self.win, text="Warnungen",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        self._lbl_warnungen = tk.Label(
            section, text="---",
            bg=BG_FRAME, fg=STATUS_GREEN, font=FONT_SMALL,
            anchor=tk.W, justify=tk.LEFT, wraplength=460,
        )
        self._lbl_warnungen.pack(fill=tk.X, padx=8, pady=5)

    # =========================================================================
    # Events (letzte 5)
    # =========================================================================

    def _build_events(self):
        """Letzte 5 Events vom Event-Bus."""
        section = tk.LabelFrame(
            self.win, text="Events (letzte 5)",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 10))

        self._txt_events = tk.Text(
            section, height=5, width=58,
            bg=BG_INPUT, fg=FG_WHITE, font=FONT_MONO,
            state=tk.DISABLED, wrap=tk.WORD,
            highlightthickness=0, borderwidth=0,
        )
        self._txt_events.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

    # =========================================================================
    # API abfragen
    # =========================================================================

    def _fetch_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics-API per HTTP abfragen. Timeout 1s."""
        try:
            req = urllib.request.Request(DIAG_URL)
            with urllib.request.urlopen(req, timeout=1) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug(f"Diagnostics-Abfrage fehlgeschlagen: {e}")
            return {}

    # =========================================================================
    # Update-Loop
    # =========================================================================

    def _poll(self):
        """Diagnostics pollen und GUI aktualisieren."""
        data = self._fetch_diagnostics()

        if data:
            self._update_gui(data)
        else:
            self._draw_ampel(FG_DIM)
            self._lbl_ampel.config(text="Keine Verbindung", fg=FG_DIM)
            self._lbl_score.config(text="", fg=FG_DIM)

        # Naechster Poll
        self._after_id = self.win.after(POLL_MS, self._poll)

    def _update_gui(self, data: Dict[str, Any]):
        """GUI mit neuen Daten aktualisieren."""
        warnungen = list(data.get("warnungen", []))

        # --- Nervensystem auswerten ---
        nervensystem = data.get("nervensystem", {})
        pipelines = nervensystem.get("pipelines", {})
        health_score = nervensystem.get("health_score", -1)
        pipeline_alerts = []

        # Pipeline-Zeilen aktualisieren
        for key, lbl in self._pipe_labels.items():
            info = pipelines.get(key, {})
            status = info.get("status", "BROKEN")
            detail = info.get("detail", "Unbekannt")
            color = _STATUS_COLORS.get(status, FG_DIM)
            lbl.config(text=f"{status}  ({detail})", fg=color)

            # Alerts fuer BROKEN/MISSING Pipelines
            display_name = PIPELINE_LABELS.get(key, key)
            if status == "BROKEN":
                pipeline_alerts.append(f"{display_name}: {detail}")
            elif status == "MISSING":
                pipeline_alerts.append(f"{display_name}: FEHLT — {detail}")

        # --- Ampel basierend auf Health Score ---
        if health_score >= 90:
            ampel_color = STATUS_GREEN
            ampel_text = f"Alles OK (Score: {health_score})"
            ampel_fg = STATUS_GREEN
        elif health_score >= 75:
            ampel_color = STATUS_YELLOW
            ampel_text = f"Warnung (Score: {health_score})"
            ampel_fg = STATUS_YELLOW
        elif health_score >= 50:
            ampel_color = STATUS_ORANGE
            ampel_text = f"Degradiert (Score: {health_score})"
            ampel_fg = STATUS_ORANGE
        elif health_score >= 0:
            ampel_color = STATUS_RED
            ampel_text = f"KRITISCH (Score: {health_score})"
            ampel_fg = STATUS_RED
        else:
            # Kein Nervensystem-Daten (alte API ohne nervensystem-Feld)
            # Fallback auf Warnungen-basierte Ampel
            ampel_color, ampel_text, ampel_fg = self._fallback_ampel(warnungen)

        self._draw_ampel(ampel_color)
        self._lbl_ampel.config(text=ampel_text, fg=ampel_fg)
        self._lbl_score.config(
            text=f"Health: {health_score}/100" if health_score >= 0 else "",
            fg=ampel_fg,
        )

        # --- System-Zeile ---
        fps = data.get("fps", 0.0)
        cpu_temp = data.get("cpu_temp", 0.0)
        ram_pct = data.get("ram_percent", 0.0)
        luefter = data.get("luefter_stufe", 0)
        uptime = data.get("uptime", "?")

        self._lbl_system.config(
            text=f"FPS: {fps:.1f}  |  CPU: {cpu_temp:.0f}\u00b0C  |  "
                 f"RAM: {ram_pct:.0f}%  |  L\u00fcfter: {luefter}  |  Up: {uptime}"
        )

        # --- Core-Zeile ---
        bridge = data.get("bridge_state", "?")
        face_id = data.get("face_id", None)
        face_sim = data.get("face_similarity", 0.0)
        tension = data.get("tension", 0.0)
        mood = data.get("mood", "?")

        face_str = f"{face_id}({face_sim:.0%})" if face_id else "---"
        self._lbl_core.config(
            text=f"Bridge: {bridge}  |  Face: {face_str}  |  "
                 f"Tension: {tension:.2f}  |  Mood: {mood}"
        )

        # --- Warnungen (System + Pipeline-Alerts) ---
        alle_warnungen = warnungen + pipeline_alerts
        if alle_warnungen:
            warn_text = "\n".join(f"\u26a0 {w}" for w in alle_warnungen)
            kritisch_keywords = ["offline", "Pipeline nicht", "kritisch", "heiss", "BROKEN", "FEHLT"]
            ist_kritisch = any(
                any(kw in w for kw in kritisch_keywords)
                for w in alle_warnungen
            )
            warn_color = STATUS_RED if ist_kritisch else STATUS_YELLOW
            self._lbl_warnungen.config(text=warn_text, fg=warn_color)
        else:
            self._lbl_warnungen.config(text="Keine Warnungen", fg=STATUS_GREEN)

        # --- Events ---
        events = data.get("recent_events", [])
        if events:
            for ev in events:
                ev_str = str(ev) if not isinstance(ev, str) else ev
                if ev_str not in self._event_history:
                    self._event_history.append(ev_str)

        self._event_history = self._event_history[-5:]

        self._txt_events.config(state=tk.NORMAL)
        self._txt_events.delete("1.0", tk.END)
        if self._event_history:
            self._txt_events.insert(tk.END, "\n".join(self._event_history))
        else:
            self._txt_events.insert(tk.END, "(keine Events)")
        self._txt_events.config(state=tk.DISABLED)

    @staticmethod
    def _fallback_ampel(warnungen: List[str]):
        """Fallback-Ampel wenn kein Nervensystem-Score vorhanden."""
        if not warnungen:
            return STATUS_GREEN, "Alles OK", STATUS_GREEN
        kritisch_keywords = ["offline", "Pipeline nicht", "kritisch", "heiss"]
        ist_kritisch = any(
            any(kw in w for kw in kritisch_keywords) for w in warnungen
        )
        if ist_kritisch:
            return STATUS_RED, f"KRITISCH ({len(warnungen)})", STATUS_RED
        return STATUS_YELLOW, f"Warnung ({len(warnungen)})", STATUS_YELLOW

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Timer stoppen, Fenster schliessen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
