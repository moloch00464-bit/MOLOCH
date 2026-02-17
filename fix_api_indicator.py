#!/usr/bin/env python3
"""Fix: API-Indikator + Token-Zaehler im Panel.

1. Chat-Nachrichten zeigen [API] oder [Intern] Herkunft
2. Token-Zaehler (kumulativ) im Panel
3. _chat_with_claude gibt Usage-Info zurueck
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: Token Counter Attribut in __init__ (nach claude_client)
old_client = '''        self.claude_client = None'''

new_client = '''        self.claude_client = None
        self._api_tokens_in = 0     # Kumulierte Input Tokens
        self._api_tokens_out = 0    # Kumulierte Output Tokens
        self._api_calls = 0         # Anzahl API Calls'''

if old_client in code:
    code = code.replace(old_client, new_client, 1)
    print('FIX 1: Token Counter Attribute - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Token-Zaehler in der Statusleiste (neben PTT Status)
# Suche nach dem PTT Status Label
old_ptt_status = '''        self.ptt_status = tk.Label('''

# Wir fuegen das Token-Label nach dem ptt_status Block ein
# Suche eine passende Stelle in der Voice-Sektion
old_voice_section = '''        # Interne Statistik
        self._cal_stats = {}'''

new_voice_section = '''        # Interne Statistik
        self._cal_stats = {}

    def _update_api_indicator(self):
        """API Token-Zaehler im Chat anzeigen."""
        if hasattr(self, '_api_indicator'):
            cost_approx = (self._api_tokens_in * 3 + self._api_tokens_out * 15) / 1_000_000
            self._api_indicator.config(
                text=f"API: {self._api_calls} Calls | "
                     f"{self._api_tokens_in + self._api_tokens_out:,} Tokens | "
                     f"~${cost_approx:.3f}")'''

if old_voice_section in code:
    code = code.replace(old_voice_section, new_voice_section)
    print('FIX 2: _update_api_indicator Methode - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: API-Label im Chat-Bereich (nach Chat-Eingabe)
old_chat_entry = '''        self._chat_entry.bind("<Return>", self._on_chat_enter)'''

new_chat_entry = '''        self._chat_entry.bind("<Return>", self._on_chat_enter)
        # API Indikator unter Chat
        self._api_indicator = tk.Label(
            chat_frame, text="API: 0 Calls | 0 Tokens | ~$0.000",
            bg="#0a0a14", fg="#555555", font=("Courier", 8), anchor=tk.W)
        self._api_indicator.pack(fill=tk.X, padx=5)'''

if old_chat_entry in code:
    code = code.replace(old_chat_entry, new_chat_entry)
    print('FIX 3: API Indikator Label - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# FIX 4: _chat_with_claude - Token Usage tracken + [API] Tag
old_response_text = '''            text = response.content[0].text

            # Extract [REMEMBER:] tags'''

new_response_text = '''            text = response.content[0].text

            # Token Usage tracken
            if hasattr(response, 'usage'):
                self._api_tokens_in += response.usage.input_tokens
                self._api_tokens_out += response.usage.output_tokens
                self._api_calls += 1
                logger.info(f"[API] Tokens: in={response.usage.input_tokens} "
                            f"out={response.usage.output_tokens} "
                            f"total={self._api_tokens_in + self._api_tokens_out}")
                self.root.after(0, self._update_api_indicator)

            # Extract [REMEMBER:] tags'''

if old_response_text in code:
    code = code.replace(old_response_text, new_response_text)
    print('FIX 4: Token Usage Tracking - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# FIX 5: Chat-Anzeige mit [API] Tag
old_chat_display = '''                self._append_chat(f"M.O.L.O.C.H.: {response}", "moloch")'''

# Nur den ersten Treffer (PTT Pipeline), nicht den Chat-Input Handler
new_chat_display = '''                self._append_chat(f"M.O.L.O.C.H. [API]: {response}", "moloch")'''

if old_chat_display in code:
    code = code.replace(old_chat_display, new_chat_display, 1)
    print('FIX 5: [API] Tag bei Voice Response - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

# FIX 6: Personality Engine Antworten mit [Intern] Tag
# Suche nach dem Personality speak_event Pattern
old_intern = '''                self._append_chat(f"M.O.L.O.C.H.: {opinion}", "moloch")'''
new_intern = '''                self._append_chat(f"M.O.L.O.C.H. [Intern]: {opinion}", "moloch")'''

if old_intern in code:
    code = code.replace(old_intern, new_intern)
    print('FIX 6: [Intern] Tag bei Voice Opinion - OK')
    fixes += 1
else:
    print('FIX 6: ANCHOR NOT FOUND (voice opinion)')

# FIX 7: Chat-Eingabe Antwort mit [API] Tag
old_chat_response = '''            self._append_chat(f"M.O.L.O.C.H.: {response}", "moloch")
            if self.tts'''

# Die zweite Stelle ist der Text-Chat Handler
if old_chat_response in code:
    code = code.replace(old_chat_response,
        '''            self._append_chat(f"M.O.L.O.C.H. [API]: {response}", "moloch")
            if self.tts''', 1)
    print('FIX 7: [API] Tag bei Chat Response - OK')
    fixes += 1
else:
    print('FIX 7: ANCHOR NOT FOUND (chat response)')

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\nPanel: {fixes} Fixes. Syntax OK.')

if fixes < 4:
    print(f'!!! INCOMPLETE: {fixes} !!!')
    sys.exit(1)

print('\n=== API INDIKATOR KOMPLETT ===')
print('[API] Tag bei Claude-Antworten')
print('[Intern] Tag bei Personality Engine Antworten')
print('Token-Zaehler unter Chat')
