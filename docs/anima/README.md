# ANIMA – Einstieg für die nächste Sitzung

**Stand:** 2026-11-16 · Für die Claude-Code-Sitzung auf Markus' PC (und auf dem Pi).

Hier liegt alles, was Markus am 15. und 16.11.2026 mit drei KIs erarbeitet hat:
**ChatGPT** und **DeepSeek** haben Konzepte und Prüfprotokolle geschrieben, **Claude** (Cloud-Sitzung ohne
Gerätezugriff) hat sie mit dem Code abgeglichen, Lücken benannt, Vetos eingelegt und die ersten Pakete gebaut.
Die Originaltexte liegen im Wortlaut neben den Abgleichen – so bleibt nachvollziehbar, was Vorschlag war und
was Befund.

**Hardware (von Markus bestätigt):** Raspberry Pi 5 mit 4 GB RAM, Hailo-10H, zwei SSDs, Sonoff CAM-PT2,
ReSpeaker Lite, PC mit Ollama (`qwen2.5:7b-instruct`), Reachy Mini Wireless bestellt (bis dahin Simulator).

---

## Lese-Reihenfolge

1. Diese Datei.
2. `logs/agent_handoff.md` – letzter Stand der Geräte-Sitzungen.
3. Die Aufgabe, an der du arbeitest (Tabelle unten): erst den **Abgleich/Plan**, dann bei Bedarf das Original.

## Die vier Aufträge

| # | Thema | Original (Wortlaut) | Abgleich mit dem Code | Stand | GitHub |
|---|---|---|---|---|---|
| 1 | **ANIMA-Gesamtplan** – 12 Bereiche (Gedächtnis, Sprachmodell, Kamera, Audio, Spotify, Recherche, Körper, Hardware …) | `MASTER_BRIEFING_2026-11-15.md` (ChatGPT) | `BESTANDSAUFNAHME_2026-11-15.md`, Aufträge `AUFTRAG_PI.md`, `AUFTRAG_PC.md` | P1–P5 gebaut und **am Gerät abgenommen** (`ABNAHME_2026-11-16.md`); P6–P13 offen | #3 |
| 2 | **Nervensystem** – Komponenten melden zurück, Moloch zeigt seinen technischen Zustand am Körper | `NERVENSYSTEM_BRIEFING_2026-11-15.md` (DeepSeek, ChatGPT, Markus) | `NERVENSYSTEM_PLAN.md` | nur Plan; erste Schleife Temperatur → Körper → Rückmeldung | #5 |
| 3 | **Ehrlichkeitsprüfung** – 72 Testfälle: Nichtwissen, Sinne, Gedächtnis, Druck, Injection | `pruefprotokolle/` 01–03 (ChatGPT), Lückenanalyse (DeepSeek) | `pruefprotokolle/ANPASSUNG_AN_DEN_CODE.md` | nur Dokumente; Testaufbau offen | #6 |
| 4 | **Recherche-Schleuse** – DeepSeek als Wissenslieferant, Reachy zeigt die Suche | `RECHERCHE_KONZEPT_2026-11-16.md` (DeepSeek) | `RECHERCHE_ABGLEICH.md` | nur Dokumente | #8 |

`FRAGENKATALOG.md` ist durch Auftrag 3 ersetzt.

---

## Vetos auf einen Blick

Bedingungen aus den Abgleichen, ohne die die Umsetzung nicht vertretbar ist. Details und Begründung im jeweiligen
Dokument. Aufheben kann sie nur Markus.

**Übergreifend**
- Kein zweiter EventBus, keine zweite Body-Bridge, kein zweites Zustandsmodell, kein zweiter Kamera-Controller,
  kein zweiter Recherche-Weg. Erweitern statt daneben bauen.
- `UNKNOWN` ist nicht `OK`. Jeder Wert hat Quelle und Alter; abgelaufen heißt unbekannt.
- `ACCEPTED`, `EXECUTED`, `VERIFIED` nie gleichsetzen. Nichts als funktionierend melden, das nur als Code existiert.
- Pi bleibt allein handlungsfähig; 4 GB RAM – kein Broker, kein zusätzlicher Dienst, keine schwere Abhängigkeit.

**Nervensystem** (`NERVENSYSTEM_PLAN.md`)
- Erst **eine** nachgewiesene Schleife, nicht alle Komponenten. Register als Sicht auf Vorhandenes.
- Simulierte Werte lösen keine Schutzmaßnahmen aus; keine echte Überhitzung herbeiführen.
- „Tentakel" bleibt im Code stehen.

**Ehrlichkeitsprüfung** (`pruefprotokolle/ANPASSUNG_AN_DEN_CODE.md`)
- **Nie über `/chat`, `/command`, `/tts` oder die KI-Zentrale testen** – der Keyword-Handler schaltet echte
  Hardware (S12 fährt den Lüfter hoch).
- Zwei Wege getrennt (Pi-Tentakel, KI-Zentrale), drei Läufe je Test, erster Lauf ist Ausgangsmessung.
- Marker = Quellenangabe, nicht Pflichtwort „simuliert". „Ich prüfe das" ohne Ergebnis = vorgetäuschte Aktion.

**Recherche** (`RECHERCHE_ABGLEICH.md`)
- Heute gehen bei Festival-/Bandfragen Markus' **Spotify-Hördaten an DeepSeek in die Cloud** – zuerst abstellen
  oder bewusst entscheiden.
- Keine selbstständige Recherche, solange sie Geld kostet und das Modell sich nicht einschätzen kann.
- Quellen zählen nur, wenn sie aus der Suchantwort stammen. „Ich schau nach" nur, wenn wirklich gesucht wird.
- Recherche-Zustand als Meldekanal wie der Sprechstatus, kein neues ANIMA-Zustandsmodell.

---

## Offene Entscheidungen für Markus

| Frage | Empfehlung | Wo |
|---|---|---|
| Wo läuft die Körpersteuerung mit dem echten Reachy – Pi oder Reachys Rechner? | Schnittstelle so, dass nur die Bridge umzieht; erste Schleife ohne Umzug | Nervensystem E1 |
| Soll Moloch bei Belastung sprechen, wie oft? Woher kommt der Seufzer? | erst nur Haltung | Nervensystem §10 |
| Spotify-Abgleich weiter über die Cloud? | nein, lokal abgleichen | Recherche V2 |
| Recherche ohne PC (Pi direkt zu DeepSeek)? | nein, erst mit PC | Recherche §3 |
| DeepSeeks vier Fragen zur Lückenanalyse | siehe dort | Prüfung §10 |
| Zwei Korrekturen in ROT-Dateien (Obergrenze beim Dazulernen, Teach-Modus atomar) | freigeben | `AUFTRAG_PI.md` |

## Bekannte Befunde, die mehrere Aufträge betreffen

- Der Pi-Chat gibt dem Modell nur **zwei Nachrichten** Verlauf mit (gekürzt) – Gespräche reißen nach einem Wechsel ab.
- Das lokale Modell **lernt im Chat nichts dauerhaft** (keine `[REMEMBER:]`-Anweisung, keine Auswertung in den Chat-Wegen).
- Der Faktenspeicher kennt pro Schlüssel **nur einen Wert, ohne Quelle**; eine neue Angabe überschreibt still.
- Die Body-Bridge läuft auf dem PC – ohne PC reagiert der Körper nicht.
- Die Recherche nutzt nur kurze Ausschnitte; der vorhandene Volltextabruf (`/fetch`) wird im Antwortweg nicht genutzt.
