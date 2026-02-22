#!/usr/bin/env python3
"""
M.O.L.O.C.H. Panel Talk/Chat
==============================

Kommunikationsmodul fuer die rechte Spalte.
Bekommt parent_frame und ServiceProxy von panel_main.

- Chat-Display: Scrollbares Text-Widget (readonly, Zeitstempel)
- Eingabezeile mit Send-Button
- Push-to-Talk (PTT) Button mit Hold-Logik
- Whisper STT Status-Anzeige
- Voice Output Toggle

Alle Commands via ServiceProxy._write_command().
Importiert NUR panel_styles und tkinter.
"""

import time
import tkinter as tk

from core.gui.panel_styles import (
    BG_FRAME, BG_BUTTON, BG_INPUT,
    BTN_OFF_DARK,
    ACCENT_RED, ACCENT_CYAN, ACCENT_GREEN,
    STATUS_YELLOW,
    FG_TEXT, FG_WHITE, FG_LABEL, FG_DIM,
    FONT_BUTTON, FONT_LABEL, FONT_SMALL, FONT_MONO,
    STATUS_UPDATE_MS,
)


class TalkChatModule:
    """Chat, PTT und Voice Controls im uebergebenen Frame."""

    def __init__(self, parent_frame, service_proxy):
        """
        Args:
            parent_frame: Frame von panel_main (rechte Spalte)
            service_proxy: ServiceProxy Instanz
        """
        self._parent = parent_frame
        self._service = service_proxy
        self._after_id = None

        # Zustaende
        self._ptt_active = False
        self._voice_enabled = tk.BooleanVar(value=False)
        self._last_whisper_status = ""

        # GUI aufbauen
        self._build_chat_display()
        self._build_input_row()
        self._build_ptt_section()
        self._build_voice_section()

        # Status-Polling starten
        self._poll_status()

    # =========================================================================
    # Chat-Display
    # =========================================================================

    def _build_chat_display(self):
        """Scrollbares Text-Widget fuer Chat-Nachrichten."""
        chat_frame = tk.Frame(self._parent, bg=BG_FRAME)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))

        # Scrollbar
        scrollbar = tk.Scrollbar(chat_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text-Widget (readonly)
        self._txt_chat = tk.Text(
            chat_frame,
            bg=BG_INPUT, fg=FG_TEXT, font=FONT_MONO,
            wrap=tk.WORD,
            insertbackground=FG_TEXT,
            selectbackground=ACCENT_CYAN,
            yscrollcommand=scrollbar.set,
            state=tk.DISABLED,
            height=12,
        )
        self._txt_chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._txt_chat.yview)

    def add_message(self, sender, text):
        """
        Nachricht ins Chat-Display einfuegen.

        Args:
            sender: Absender-Name (z.B. "Markus", "MOLOCH", "System")
            text: Nachrichtentext
        """
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {sender}: {text}\n"

        self._txt_chat.config(state=tk.NORMAL)
        self._txt_chat.insert(tk.END, line)
        self._txt_chat.see(tk.END)
        self._txt_chat.config(state=tk.DISABLED)

    # =========================================================================
    # Eingabezeile
    # =========================================================================

    def _build_input_row(self):
        """Entry + Send-Button."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=2)

        self._entry = tk.Entry(
            row,
            bg=BG_INPUT, fg=FG_TEXT, font=FONT_MONO,
            insertbackground=FG_TEXT,
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._entry.bind("<Return>", lambda e: self._send_message())

        tk.Button(
            row, text="Send", width=6,
            bg=BG_BUTTON, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=BG_FRAME,
            command=self._send_message,
        ).pack(side=tk.RIGHT)

    def _send_message(self):
        """Text aus Entry an Service senden und Feld leeren."""
        text = self._entry.get().strip()
        if not text:
            return

        self._service._write_command("chat_message", {"text": text})
        self.add_message("Markus", text)
        self._entry.delete(0, tk.END)

    # =========================================================================
    # Push-to-Talk
    # =========================================================================

    def _build_ptt_section(self):
        """PTT Button mit Hold-Logik und Whisper-Status."""
        section = tk.LabelFrame(
            self._parent,
            text="Push-to-Talk",
            bg=BG_FRAME,
            fg=FG_LABEL,
            font=FONT_LABEL,
        )
        section.pack(fill=tk.X, padx=5, pady=2)

        row = tk.Frame(section, bg=BG_FRAME)
        row.pack(pady=5, padx=5)

        # PTT Button — Hold-Logik via Bind
        self._btn_ptt = tk.Button(
            row, text="PTT", width=12, height=2,
            bg=BTN_OFF_DARK, fg=FG_WHITE, font=FONT_BUTTON,
            activebackground=ACCENT_RED,
        )
        self._btn_ptt.pack(side=tk.LEFT, padx=(0, 10))
        self._btn_ptt.bind("<ButtonPress-1>", self._ptt_press)
        self._btn_ptt.bind("<ButtonRelease-1>", self._ptt_release)

        # Whisper Status Label
        self._lbl_whisper = tk.Label(
            row, text="Idle",
            bg=BG_FRAME, fg=FG_DIM, font=FONT_SMALL,
        )
        self._lbl_whisper.pack(side=tk.LEFT)

    def _ptt_press(self, event):
        """PTT gedrueckt — Aufnahme starten."""
        self._ptt_active = True
        self._btn_ptt.config(bg=ACCENT_RED)
        self._service._write_command("ptt_start")

    def _ptt_release(self, event):
        """PTT losgelassen — Aufnahme stoppen."""
        self._ptt_active = False
        self._btn_ptt.config(bg=BTN_OFF_DARK)
        self._service._write_command("ptt_stop")

    # =========================================================================
    # Voice Output Toggle
    # =========================================================================

    def _build_voice_section(self):
        """Voice Output Checkbutton."""
        row = tk.Frame(self._parent, bg=BG_FRAME)
        row.pack(fill=tk.X, padx=5, pady=(2, 5))

        tk.Checkbutton(
            row, text="Sprache AN/AUS",
            variable=self._voice_enabled,
            bg=BG_FRAME, fg=FG_WHITE,
            selectcolor=BG_FRAME,
            activebackground=BG_FRAME,
            activeforeground=FG_WHITE,
            font=FONT_BUTTON,
            command=self._toggle_voice_output,
        ).pack(side=tk.LEFT, padx=5)

    def _toggle_voice_output(self):
        """Voice Output umschalten."""
        enabled = self._voice_enabled.get()
        self._service._write_command("toggle_voice_output", {
            "enabled": enabled,
        })

    # =========================================================================
    # Status-Polling
    # =========================================================================

    def _poll_status(self):
        """Whisper-Status und Voice-Toggle aus Service-Status aktualisieren."""
        status = self._service.read_status()

        if status:
            # Whisper STT Status
            voice = status.get("voice", {})
            whisper_state = voice.get("whisper_status", "Idle")
            if whisper_state != self._last_whisper_status:
                self._last_whisper_status = whisper_state
                color_map = {
                    "Idle": FG_DIM,
                    "Aufnahme...": ACCENT_RED,
                    "Transkribiere...": STATUS_YELLOW,
                    "Denke...": ACCENT_CYAN,
                    "Spreche...": ACCENT_GREEN,
                    "Fehler": ACCENT_RED,
                }
                self._lbl_whisper.config(
                    text=whisper_state,
                    fg=color_map.get(whisper_state, FG_DIM),
                )

            # Chat-Messages vom Service abholen und anzeigen
            messages = voice.get("messages", [])
            for msg in messages:
                sender = msg.get("sender", "?")
                text = msg.get("text", "")
                if text:
                    self.add_message(sender, text)

            # Voice Output Toggle synchronisieren
            voice_on = voice.get("voice_enabled", False)
            if self._voice_enabled.get() != voice_on:
                self._voice_enabled.set(voice_on)

        # Naechster Poll
        self._after_id = self._parent.after(STATUS_UPDATE_MS, self._poll_status)
