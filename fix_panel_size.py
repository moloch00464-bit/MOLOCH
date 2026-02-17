#!/usr/bin/env python3
"""Fix: Panel Fenstergeosse + Layout straffer.

1. Explizite Startgroesse (volle Breite, 90% Hoehe)
2. Main padding kleiner
3. Preview etwas kleiner
4. Perception Tabs: Notebook kompakter
"""
import sys

panel = '/home/molochzuhause/moloch/core/gui/moloch_unified_panel.py'
with open(panel) as f:
    code = f.read()

fixes = 0

# FIX 1: Explizite Fenstergeosse nach root.resizable
old_resize = '''        self.root.resizable(True, True)'''

new_resize = '''        self.root.resizable(True, True)
        # Starte maximiert (volle Breite, volle Hoehe)
        try:
            self.root.attributes('-zoomed', True)
        except Exception:
            self.root.geometry("1920x1050+0+0")'''

if old_resize in code:
    code = code.replace(old_resize, new_resize)
    print('FIX 1: Fenster maximiert - OK')
    fixes += 1
else:
    print('FIX 1: ANCHOR NOT FOUND!')

# FIX 2: Main padding kleiner
old_padding = '''        main = ttk.Frame(self.root, padding=8)'''
new_padding = '''        main = ttk.Frame(self.root, padding=2)'''

if old_padding in code:
    code = code.replace(old_padding, new_padding)
    print('FIX 2: Main padding 2 - OK')
    fixes += 1
else:
    print('FIX 2: ANCHOR NOT FOUND!')

# FIX 3: Preview kleiner (640x360 -> 560x315) fuer mehr Platz rechts
old_prev = '''    PREVIEW_W = 640
    PREVIEW_H = 360'''
new_prev = '''    PREVIEW_W = 560
    PREVIEW_H = 315'''

if old_prev in code:
    code = code.replace(old_prev, new_prev)
    print('FIX 3: Preview 560x315 - OK')
    fixes += 1
else:
    print('FIX 3: ANCHOR NOT FOUND!')

# FIX 4: Perception Tabs Font kleiner
old_tab = '''        self._perc_nb = ttk.Notebook(parent)'''
new_tab = '''        self._perc_nb = ttk.Notebook(parent)
        # Kompaktere Tab-Font
        import tkinter.font as tkfont
        tab_font = tkfont.Font(family="Helvetica", size=8)
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Helvetica", 8), padding=[3, 1])'''

if old_tab in code:
    code = code.replace(old_tab, new_tab, 1)
    print('FIX 4: Tab-Font kleiner - OK')
    fixes += 1
else:
    print('FIX 4: ANCHOR NOT FOUND!')

# FIX 5: Talk/Chat Bereich: PTT Button kleiner
old_ptt_btn = '''            font=("Helvetica", 16, "bold"), width=10, height=3,'''
new_ptt_btn = '''            font=("Helvetica", 14, "bold"), width=8, height=2,'''

if old_ptt_btn in code:
    code = code.replace(old_ptt_btn, new_ptt_btn)
    print('FIX 5: PTT Button kleiner - OK')
    fixes += 1
else:
    print('FIX 5: ANCHOR NOT FOUND!')

# FIX 6: Chat Text Height kleiner (weniger Zeilen)
old_chat = '''        self.chat_text = tk.Text(chat, height=12,'''
new_chat = '''        self.chat_text = tk.Text(chat, height=8,'''

if old_chat in code:
    code = code.replace(old_chat, new_chat)
    print('FIX 6: Chat 8 Zeilen - OK')
    fixes += 1
else:
    # Versuche andere Hoehe
    for h in range(6, 20):
        alt = f'self.chat_text = tk.Text(chat, height={h},'
        if alt in code:
            code = code.replace(alt, 'self.chat_text = tk.Text(chat, height=8,')
            print(f'FIX 6: Chat {h} -> 8 Zeilen - OK')
            fixes += 1
            break
    else:
        print('FIX 6: ANCHOR NOT FOUND!')

with open(panel, 'w') as f:
    f.write(code)

compile(open(panel).read(), panel, 'exec')
print(f'\n{fixes} Fixes. Syntax OK.')
