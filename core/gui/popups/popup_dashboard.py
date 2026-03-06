#!/usr/bin/env python3
"""
M.O.L.O.C.H. NPU Dashboard Popup — Gate 1
============================================

Zeigt Live-Daten:
- FPS pro Modell (SCRFD, ArcFace, YOLOv8m)
- Event Bus Rate + Silence-Level
- Action Bridge State + Alter
- Letzte 5 Decisions aus Ringbuffer

Aktualisiert alle 500ms via ServiceProxy.read_status().
"""

import time
import tkinter as tk

from core.gui.panel_styles import (
    BG_DARK, BG_FRAME, BG_INPUT,
    FG_WHITE, FG_LABEL, FG_DIM,
    STATUS_GREEN, STATUS_RED, STATUS_YELLOW,
    ACCENT_CYAN,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO,
)


class DashboardPopup:
    """NPU Dashboard als eigenstaendiges Toplevel-Fenster."""

    REFRESH_MS = 500

    def __init__(self, parent, service_proxy):
        self.parent = parent
        self.service = service_proxy

        self.win = tk.Toplevel(parent)
        self.win.attributes('-topmost', True)
        self.win.transient(parent)
        self.win.title("Dashboard \u2014 NPU / Bridge / Bus")
        self.win.configure(bg=BG_DARK)
        self.win.geometry("460x420")
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_fps_section()
        self._build_bus_section()
        self._build_bridge_section()
        self._build_decisions_section()

        self._poll()

    # =========================================================================
    # FPS pro Modell
    # =========================================================================

    def _build_fps_section(self):
        section = tk.LabelFrame(
            self.win, text="FPS pro Modell",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=(10, 3))

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=4)

        self._fps_labels = {}
        for name in ("scrfd", "arcface", "yolov8m", "total"):
            col = tk.Frame(row, bg=BG_FRAME)
            col.pack(side=tk.LEFT, expand=True)
            tk.Label(col, text=name.upper(), bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack()
            lbl = tk.Label(col, text="--", bg=BG_FRAME, fg=STATUS_GREEN, font=FONT_TITLE)
            lbl.pack()
            self._fps_labels[name] = lbl

    # =========================================================================
    # Event Bus Stats
    # =========================================================================

    def _build_bus_section(self):
        section = tk.LabelFrame(
            self.win, text="Event Bus",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=4)

        self._bus_labels = {}
        for key, label in [("total_published", "Published"),
                           ("total_delivered", "Delivered"),
                           ("total_silenced", "Silenced"),
                           ("silence_level", "Silence")]:
            col = tk.Frame(row, bg=BG_FRAME)
            col.pack(side=tk.LEFT, expand=True)
            tk.Label(col, text=label, bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack()
            lbl = tk.Label(col, text="--", bg=BG_FRAME, fg=ACCENT_CYAN, font=FONT_LABEL)
            lbl.pack()
            self._bus_labels[key] = lbl

    # =========================================================================
    # Action Bridge State
    # =========================================================================

    def _build_bridge_section(self):
        section = tk.LabelFrame(
            self.win, text="Action Bridge",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=10, pady=3)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=8, pady=4)

        self._bridge_labels = {}
        for key, label in [("state", "State"), ("state_age_s", "Alter"),
                           ("decisions", "Decisions"), ("prev_state", "Vorher")]:
            col = tk.Frame(row, bg=BG_FRAME)
            col.pack(side=tk.LEFT, expand=True)
            tk.Label(col, text=label, bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL).pack()
            lbl = tk.Label(col, text="--", bg=BG_FRAME, fg=STATUS_YELLOW, font=FONT_LABEL)
            lbl.pack()
            self._bridge_labels[key] = lbl

    # =========================================================================
    # Letzte 5 Decisions
    # =========================================================================

    def _build_decisions_section(self):
        section = tk.LabelFrame(
            self.win, text="Letzte Decisions",
            bg=BG_FRAME, fg=FG_LABEL, font=FONT_LABEL,
        )
        section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 10))

        self._decisions_text = tk.Text(
            section, bg=BG_INPUT, fg=FG_WHITE, font=FONT_MONO,
            height=8, wrap=tk.NONE, state=tk.DISABLED,
            highlightthickness=0, borderwidth=0,
        )
        self._decisions_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # =========================================================================
    # Polling
    # =========================================================================

    def _poll(self):
        """Status vom Service lesen und Anzeige aktualisieren."""
        try:
            status = self.service.read_status()
            if status:
                self._update_fps(status.get("fps", {}))
                self._update_bus(status.get("bus_stats", {}), status.get("silence_level", 0))
                self._update_bridge(status.get("bridge", {}))
                self._update_decisions(status.get("bridge_decisions", []))
        except Exception:
            pass

        if self.win.winfo_exists():
            self.win.after(self.REFRESH_MS, self._poll)

    def _update_fps(self, fps: dict):
        for key, lbl in self._fps_labels.items():
            val = fps.get(key, 0)
            lbl.config(text=f"{val:.1f}")
            lbl.config(fg=STATUS_GREEN if val > 5 else STATUS_RED)

    def _update_bus(self, stats: dict, silence: int):
        for key, lbl in self._bus_labels.items():
            if key == "silence_level":
                names = {0: "NORMAL", 1: "REDUCED", 2: "SILENT"}
                lbl.config(text=names.get(silence, str(silence)))
                lbl.config(fg=STATUS_GREEN if silence == 0 else STATUS_YELLOW)
            else:
                lbl.config(text=str(stats.get(key, 0)))

    def _update_bridge(self, bridge: dict):
        state_colors = {
            "idle": FG_DIM, "searching": STATUS_YELLOW,
            "tracking": STATUS_GREEN, "interaction": ACCENT_CYAN,
            "manual_override": STATUS_RED,
        }
        for key, lbl in self._bridge_labels.items():
            val = bridge.get(key, "--")
            if key == "state_age_s" and isinstance(val, (int, float)):
                lbl.config(text=f"{val:.0f}s")
            else:
                lbl.config(text=str(val))
            if key == "state":
                lbl.config(fg=state_colors.get(str(val), FG_LABEL))

    def _update_decisions(self, decisions: list):
        self._decisions_text.config(state=tk.NORMAL)
        self._decisions_text.delete("1.0", tk.END)
        for d in reversed(decisions):
            ts = d.get("timestamp", 0)
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            line = (f"{t_str} [{d.get('old_state', '?')}->{d.get('new_state', '?')}] "
                    f"{d.get('thought', '')}\n")
            self._decisions_text.insert(tk.END, line)
        self._decisions_text.config(state=tk.DISABLED)

    # =========================================================================
    # Schliessen
    # =========================================================================

    def _on_close(self):
        self.win.destroy()
