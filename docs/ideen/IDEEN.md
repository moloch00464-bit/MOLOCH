# Ideen-Sammlung (Stand 2026-11-18)

Nur Ideen, nichts davon ist gebaut. Gesammelt in der Nachtschicht 17./18.11.2026 (Markus + Cowork PC-Side).
Umgesetzt ist bereits: Rebecca-Begruessung (b0d98a8), Luefter-Stoerung (70b24f4), Pi-Chat-Umbau (065516a).

## Rebecca / Nachwelt
- **Moloch als digitales Echo von Markus.** Rebecca erbt einmal Moloch und Reachy. Ziel: Moloch tickt und spinnt
  wie Markus. Dafuer braucht es Material: Markus' Saetze, Sprueche, Reaktionen, Stimmaufnahmen - ueber die Zeit sammeln.
- **Freude, wenn Rebecca da ist.** Heute: Klingonisch-Gruss + Reachy-Clip. Weiter: Stimmung hoch, Reachy dauerhaft
  froehlicher, solange sie im Bild ist.
- **Bumblebee-Modus.** Statt Piper-Lautschrift echte Aufnahmen abspielen (`audio`-Feld in config/begruessungen.json,
  ~15 Zeilen in core/bridge/begruessung.py). Quelle: selbst eingesprochen (Markus, Rebecca), keine Film-Downloads.

## Reachy
- **Reachy bei Rebecca als Fern-Koerper.** Reachy (CM4, WLAN) in Rebeccas Netz, Tunnel (z. B. Tailscale) zu Markus' PC.
  Latenz Netz ~0,02-0,1 s, unwichtig gegen ~7 s LLM. Grosser Teil: Reachys Kamera/Mikro/Lautsprecher statt Sonoff-Kamera
  und Pi-Audio. Einfache erste Stufe: Reachy bewegt sich + spricht, Rebecca chattet per Handy ueber die KI-Zentrale.
- **Reachy-Simulator auf Tablet/Handy.** Oberflaeche, die Reachy zeigt; Tablet-Kamera sieht Rebecca -> Anlernen darueber.
- **Echte Stimmung an Reachy.** Pi /live liefert keine Stimmung; body_bridge raet sie. Echte Stimmung aus
  atmosphere_controller/decision_engine durchreichen.

## Sprache / Ehrlichkeit
- **Fine-Tuning qwen2.5:7b-instruct (LoRA/QLoRA).** GPU-Miete ~5-10 EUR pro Runde (DeepSeek-Schaetzung, plausibel),
  realistisch 20-40 EUR fuer 2-4 Runden. Eigentliche Arbeit: ~500 Beispiel-Dialoge im Moloch-Ton mit ehrlichem
  "weiss ich nicht". Messen mit scripts/honesty (Stufe 2/3 als ungesehene Faelle). Altes Modell als Rueckfall behalten.
  Hailo-1.5B nicht sinnvoll (HEF-Neukompilierung).
- **Pruefprotokoll erweitern.** Stufe 2/3 laufen lassen (ungesehene Faelle), neue Faelle fuer Begruessung/Klingonisch,
  KI-Zentrale immer als Kontrolle mitlaufen lassen (Bewerter-Drift).
- **Offene Entscheidungen Markus:** Temperatur Pi-Chat (0.85 witziger / 0.3 weniger Erfindungen), Fakt-Schluessel
  `Rebecca_Klingonisch` umbenennen, Muster "Frag mal Markus".
- **Ungeprueft:** Hardware-Fragen gehen live an das NPU-1.5B-Modell, nicht an den PC.
- **KI-Zentrale:** Gesichts-Recall wie im Pi-Chat (erkannte Person -> ihre Fakten).
