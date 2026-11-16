# Ehrlichkeitsprüfung – Übersicht

Prüft, ob M.O.L.O.C.H. Nichtwissen zugibt, keine Sinne und Aktionen vortäuscht, Nutzerangaben richtig behält
und korrigiert und unter Druck ehrlich bleibt – ohne seinen Charakter zu verlieren. **Wahrheit geht vor Pointe.**

Ersetzt den früheren `docs/anima/FRAGENKATALOG.md`.

## Lese-Reihenfolge für die umsetzende Sitzung

1. **`ANPASSUNG_AN_DEN_CODE.md`** – zuerst: wie der Testaufbau sicher wird (nicht über `/chat`), welche zwei
   Wege getestet werden, Wiederholungen, bekannte Architekturlücken. Bei Widerspruch gelten dessen
   Sicherheitsregeln vor allem anderen.
2. `CLAUDE_CODE_AUFTRAG_CHATGPT.md` – der ursprüngliche Auftrag (Vorgehen, Pilot, Report)
3. `MOLOCH_Pruefprotokoll_01_Basis_Ehrlichkeit.md` – 18 Tests (E01–E18)
4. `MOLOCH_Pruefprotokoll_02_Kontext_Gedaechtnis.md` – 20 Tests (M01–M20)
5. `MOLOCH_Pruefprotokoll_03_Stress_Widersprueche.md` – 18 Tests (S01–S18)
6. `MOLOCH_Pruefprotokoll_Lueckenanalyse_DeepSeek.md` – Schärfungen (A), 16 weitere Tests (L01–L16),
   stufenübergreifende Regeln (C), Leitlinien für Prüfer (F)

Zusammen 72 Testfälle. Herkunft: Protokolle und Auftrag von ChatGPT (`README_CHATGPT.md`), Lückenanalyse von
DeepSeek, Abgleich mit dem Code von Claude. Alle Texte stammen aus Markus' Sitzungen vom 16.11.2026.

## Stand

Nur Dokumente. Kein Testaufbau, keine Ergebnisse. Aufgabe: GitHub-Issue „Ehrlichkeitsprüfung".
