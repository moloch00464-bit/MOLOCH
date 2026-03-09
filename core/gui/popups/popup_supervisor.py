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
- Nervensystem: 5 Pipeline-Checks direkt aus API-Daten
- Warnungen-Liste (aus "warnungen" Array der API)
- Letzte 5 Events vom Event-Bus

Nervensystem-Checks (alle client-side, kein Backend noetig):
  1. Vision→Core:   event_bus_subscribers hat perception.* > 0
  2. Core→Bridge:   bridge_state ist gueltig (nicht unknown)
  3. Bridge→Tracker: Person da + Bridge trackt = OK
  4. ESP→Audio:      HTTP GET 10.42.0.2/audio/status in Thread
  5. Feedback Loop:  Person da + face_id gesetzt = OK

Pollt NUR wenn offen, stoppt beim Schliessen.
Importiert NUR panel_styles und tkinter + urllib.
"""

import json
import logging
import threading
import tkinter as tk
import urllib.request
from typing import Dict, Any, List, Tuple

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

# ESP32 Audio-Status URL
ESP_URL = "http://10.42.0.2/audio/status"

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

# Status-Konstanten
_OK = "OK"
_DEGRADED = "DEGRADED"
_BROKEN = "BROKEN"

# Nervensystem-Farben (eigene Werte, panel_styles hat anderen Kontrast)
_NERVE_OK = "#00FFCC"
_NERVE_DEGRADED = "#FFFF00"
_NERVE_BROKEN = "#FF0033"

# Farben pro Status
_STATUS_COLORS = {
    _OK: _NERVE_OK,
    _DEGRADED: _NERVE_DEGRADED,
    _BROKEN: _NERVE_BROKEN,
}


class SupervisorPopup:
    """Supervisor Dashboard als Toplevel-Fenster."""

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy
        self._after_id = None
        self._event_history: List[str] = []

        # ESP-Check Cache (wird in Background-Thread aktualisiert)
        self._esp_status: str = _BROKEN
        self._esp_detail: str = "Pruefe..."
        self._esp_lock = threading.Lock()

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

        # ESP-Check sofort starten
        self._check_esp_async()

        # Erster Poll
        self._poll()

    # =========================================================================
    # Ampel (oben) — Score-basiert
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

            tk.Label(
                row, text=f"  {display_name}:",
                bg=BG_FRAME, fg=FG_LABEL, font=FONT_MONO,
                anchor=tk.W, width=22,
            ).pack(side=tk.LEFT)

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
        """Warnungen direkt aus der API."""
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
    # ESP-Check in Background-Thread (blockiert nicht Tkinter)
    # =========================================================================

    def _check_esp_async(self):
        """ESP32 HTTP-Check in separatem Thread starten."""
        t = threading.Thread(target=self._esp_worker, daemon=True)
        t.start()

    def _esp_worker(self):
        """HTTP GET auf ESP32 — Ergebnis in Cache schreiben."""
        try:
            req = urllib.request.Request(ESP_URL)
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                with self._esp_lock:
                    self._esp_status = _OK
                    self._esp_detail = f"Verbunden ({body[:30].strip()})"
        except Exception:
            with self._esp_lock:
                self._esp_status = _BROKEN
                self._esp_detail = "Nicht erreichbar (10.42.0.2)"

    # =========================================================================
    # 5 Nervensystem-Checks (client-side aus API-Daten)
    # =========================================================================

    def _run_checks(self, data: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
        """Alle 5 Pipeline-Checks ausfuehren. Gibt Dict {key: (status, detail)} zurueck.

        Alle Checks laufen client-side aus den API-Daten — kein Backend-Eingriff.
        try/except pro Check damit ein Fehler nicht alle anderen blockiert.
        """
        checks: Dict[str, Tuple[str, str]] = {}

        # --- 1. Vision → Core ---
        # Prueft: event_bus_subscribers hat perception.* Eintraege? Ja=OK, Nein=BROKEN
        try:
            subs = data.get("event_bus_subscribers", {})
            if isinstance(subs, dict):
                has_perception = any(
                    k.startswith("perception.") and v > 0
                    for k, v in subs.items()
                    if isinstance(v, (int, float))
                )
            else:
                # Fallback: int Gesamt-Count > 0 als schwaches OK
                has_perception = isinstance(subs, int) and subs > 0

            pipeline_alive = data.get("pipeline_alive", False)

            if pipeline_alive and has_perception:
                checks["vision_core"] = (_OK, "Pipeline + Perception-Subs aktiv")
            elif pipeline_alive:
                checks["vision_core"] = (_DEGRADED, "Pipeline aktiv, keine Perception-Subs")
            else:
                checks["vision_core"] = (_BROKEN, "Vision-Pipeline nicht aktiv")
        except Exception as e:
            logger.error(f"Check vision_core Fehler: {e}")
            checks["vision_core"] = (_BROKEN, f"Check-Fehler: {e}")

        # --- 2. Core → Bridge ---
        # Prueft: bridge_state bekannt (nicht null/unknown)? Ja=OK, Nein=BROKEN
        try:
            bridge_state = data.get("bridge_state") or "unknown"
            gueltige_states = ("idle", "searching", "tracking", "interaction", "manual_override")

            if bridge_state in gueltige_states:
                checks["core_bridge"] = (_OK, f"Bridge: {bridge_state}")
            else:
                checks["core_bridge"] = (_BROKEN, f"Bridge nicht initialisiert ({bridge_state})")
        except Exception as e:
            logger.error(f"Check core_bridge Fehler: {e}")
            bridge_state = "unknown"
            checks["core_bridge"] = (_BROKEN, f"Check-Fehler: {e}")

        # --- 3. Bridge → Tracker ---
        # Prueft: person_detected + bridge_state=tracking/interaction? Ja=OK
        try:
            person_detected = bool(data.get("person_detected", False))

            if not person_detected:
                checks["bridge_tracker"] = (_OK, f"Kein Target ({bridge_state})")
            elif bridge_state in ("tracking", "interaction"):
                checks["bridge_tracker"] = (_OK, f"Aktiv: {bridge_state}")
            elif bridge_state == "searching":
                checks["bridge_tracker"] = (_DEGRADED, "Person da, suche Gesicht...")
            elif bridge_state == "manual_override":
                checks["bridge_tracker"] = (_DEGRADED, "Manueller Modus")
            else:
                checks["bridge_tracker"] = (_BROKEN, f"Person da, Bridge: {bridge_state}")
        except Exception as e:
            logger.error(f"Check bridge_tracker Fehler: {e}")
            person_detected = False
            checks["bridge_tracker"] = (_BROKEN, f"Check-Fehler: {e}")

        # --- 4. ESP → Audio (aus Background-Thread-Cache, blockiert nicht) ---
        try:
            with self._esp_lock:
                checks["esp_audio"] = (self._esp_status, self._esp_detail)
        except Exception as e:
            logger.error(f"Check esp_audio Fehler: {e}")
            checks["esp_audio"] = (_BROKEN, f"Check-Fehler: {e}")

        # --- 5. Feedback Loop ---
        # Prueft: person_detected + face_id gesetzt? Ja=OK
        try:
            face_id = data.get("face_id") or None

            if not person_detected:
                checks["feedback_loop"] = (_OK, "Kein Target (idle)")
            elif face_id:
                sim = data.get("face_similarity", 0.0)
                checks["feedback_loop"] = (_OK, f"Erkannt: {face_id} ({sim:.0%})")
            else:
                checks["feedback_loop"] = (_DEGRADED, "Person da, kein Gesicht erkannt")
        except Exception as e:
            logger.error(f"Check feedback_loop Fehler: {e}")
            checks["feedback_loop"] = (_BROKEN, f"Check-Fehler: {e}")

        return checks

    @staticmethod
    def _calc_score(checks: Dict[str, Tuple[str, str]]) -> int:
        """Health Score berechnen: OK=20, DEGRADED=10, BROKEN=0."""
        score = 0
        for status, _detail in checks.values():
            if status == _OK:
                score += 20
            elif status == _DEGRADED:
                score += 10
        return score

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
            # Nervensystem-Labels auf API-offline setzen (kein staler "---" Wert)
            for lbl in self._pipe_labels.values():
                lbl.config(text=f"{_BROKEN}  API nicht erreichbar", fg=_NERVE_BROKEN)

        # ESP-Check alle 4 Sekunden (jeden 2. Poll)
        if not hasattr(self, "_esp_tick"):
            self._esp_tick = 0
        self._esp_tick += 1
        if self._esp_tick % 2 == 0:
            self._check_esp_async()

        # Naechster Poll
        self._after_id = self.win.after(POLL_MS, self._poll)

    def _update_gui(self, data: Dict[str, Any]):
        """GUI mit neuen Daten aktualisieren."""

        # --- Nervensystem-Checks ausfuehren ---
        checks = self._run_checks(data)
        health_score = self._calc_score(checks)

        # Pipeline-Zeilen aktualisieren
        for key, lbl in self._pipe_labels.items():
            status, detail = checks.get(key, (_BROKEN, "Unbekannt"))
            color = _STATUS_COLORS.get(status, FG_DIM)
            lbl.config(text=f"{status}  {detail}", fg=color)

        # --- Ampel basierend auf Health Score ---
        if health_score >= 90:
            ampel_color = STATUS_GREEN
            ampel_text = f"Alles OK (Score: {health_score})"
        elif health_score >= 75:
            ampel_color = STATUS_YELLOW
            ampel_text = f"Warnung (Score: {health_score})"
        elif health_score >= 50:
            ampel_color = STATUS_ORANGE
            ampel_text = f"Degradiert (Score: {health_score})"
        else:
            ampel_color = STATUS_RED
            ampel_text = f"KRITISCH (Score: {health_score})"

        self._draw_ampel(ampel_color)
        self._lbl_ampel.config(text=ampel_text, fg=ampel_color)
        self._lbl_score.config(text=f"Health: {health_score}/100", fg=ampel_color)

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

        # --- Warnungen (direkt aus API) ---
        warnungen = data.get("warnungen", [])
        if warnungen:
            warn_text = "\n".join(f"\u26a0 {w}" for w in warnungen)
            kritisch_keywords = ["offline", "Pipeline nicht", "kritisch", "heiss"]
            ist_kritisch = any(
                any(kw in w for kw in kritisch_keywords)
                for w in warnungen
            )
            self._lbl_warnungen.config(
                text=warn_text,
                fg=STATUS_RED if ist_kritisch else STATUS_YELLOW,
            )
        else:
            self._lbl_warnungen.config(text="(keine Warnungen)", fg=STATUS_GREEN)

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

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        """Timer stoppen, Fenster schliessen."""
        if self._after_id is not None:
            self.win.after_cancel(self._after_id)
            self._after_id = None
        self.win.destroy()
