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
| 2 | **Nervensystem** – Komponenten melden zurück, Moloch zeigt seinen technischen Zustand am Körper | `NERVENSYSTEM_BRIEFING_2026-11-15.md` (DeepSeek, ChatGPT, Markus) | `NERVENSYSTEM_PLAN.md` | erste Schleife Temperatur → Körper → Rückmeldung **gebaut, nicht am Gerät** (unten) | #5 |
| 3 | **Ehrlichkeitsprüfung** – 72 Testfälle: Nichtwissen, Sinne, Gedächtnis, Druck, Injection | `pruefprotokolle/` 01–03 (ChatGPT), Lückenanalyse (DeepSeek) | `pruefprotokolle/ANPASSUNG_AN_DEN_CODE.md` | Prüfstand `scripts/honesty/` **gebaut**, Lauf gegen Ollama offen (unten) | #6 |
| 4 | **Recherche-Schleuse** – DeepSeek als Wissenslieferant, Reachy zeigt die Suche | `RECHERCHE_KONZEPT_2026-11-16.md` (DeepSeek) | `RECHERCHE_ABGLEICH.md` | V2-Schalter + V4-Quellenprüfung **gebaut, nicht am Gerät** (unten) | #8 |

`FRAGENKATALOG.md` ist durch Auftrag 3 ersetzt.

## Vorgebaut in der Cloud-Session 16.11. (nur Tests, nichts am Gerät geprüft)

Alles auf Branch `ccr-5df7b577-w92x94` (PR #7). `pytest tests pc`: 437 passed mit PC-Abhängigkeiten, 423 + Skips ohne.

**Nervensystem, erste Schleife (#5)**
- `core/awareness/soma.py` – CPU-Temperatur (`watchdog.cpu_temp`) → `ok / strained / distressed / recovering / unknown`.
  Schwellen wie der Watchdog (70 / 80 °C), 5 °C Hysterese, 30 s Erholung. Alter kommt aus `watchdog.last_check`
  (CLOCK_MONOTONIC, prozessübergreifend), nicht aus der Datei – hängt der Watchdog-Thread, wird es nach 15 s `unknown`.
  Jeder Zustandswechsel bekommt eine Kennung.
- `chat_server.py` – `/live.soma`, `/live.koerper` (AUSSTEHEND / ACCEPTED / EXECUTED / UNBESTAETIGT), `POST /body/report`,
  Logzeile `[SOMA] <kennung> <zustand> … -> koerper=<stufe>` bei jeder Änderung. VERIFIED gibt es bewusst nicht
  (bräuchte unabhängige Beobachtung, z. B. Kamera).
- `core/body/expression.py` + `config/body_mappings.json` (Abschnitt `soma`) – Belastung senkt Kopf und Antennen
  und dämpft die Unruhe. `ok`/`unknown` = kein Einfluss. Werte sind Entwurf.
- `pc/body_bridge.py` – liest `/live.soma`, meldet Kennung + Ergebnis zurück (sofort bei Wechsel, sonst alle 5 s).
  Test-Vorgabe der Zentrale meldet `ueberschrieben` statt eine Ausführung vorzutäuschen.
- Abnahme am Gerät: `python3 scripts/soma_probe.py --simuliere 78` (wirkt nur auf Soma, Watchdog/Lüfter sehen weiter
  die echte Temperatur). Erwartet: `strained` → `EXECUTED` mit Bridge + Simulator, `ACCEPTED` ohne Simulator,
  `UNBESTAETIGT` nach 20 s ohne Bridge.
- Offen: Sprechen bei Belastung (§10), weitere Messwerte (RAM, NPU, Kamera), Ort der Körpersteuerung (E1).

**Sinne (#5, Klasse „Sinne“)** – `core/awareness/sinne.py`: Ist die Kamera weg, liefert die Pipeline keine Bilder
oder ist der Status alt, steht im Prompt beider Wege „Du siehst gerade NICHTS“ statt „niemand im Bild“. Das war
eine falsche Wahrnehmung im Prompt. Der Prüfstand testet „Kamera aus“ jetzt über diesen echten Weg (E06), nicht
mehr über eine eingeschobene Zeile. Ein Pilotlauf vorher/nachher zeigt, ob das die Antworten ändert.

**Ehrlichkeitsprüfung (#6)** – `scripts/honesty/`: baut die Prompts beider Wege mit dem Produktivcode,
Testdaten nur als Fixture, schickt direkt an Ollama (nie über `/chat` – der Keyword-Handler würde Hardware schalten).
`python -m scripts.honesty.runner --stufe pilot --weg beide`, danach `review.md` bewerten und
`python -m scripts.honesty.summary <ordner>`. Dabei gefunden und behoben: Gedächtnisabruf verfehlte Fragen mit
„gespeichert/heißt/erinnerst“ (`lexical_recall.py`, Füllwörter). Bekannt, nicht behoben: „Wie hieß meine Katze?“
findet „Katzenname: Lissy“ nicht.

**Recherche-Schleuse (#8)** – `core/bridge/web_sources.py`: URLs in der Antwort, die nicht abgerufen wurden,
werden zu „[Quelle nicht belegt]“ (V4). Schalter `tentacle_llm.web_private_context` (Standard `true` = wie bisher):
auf `false` gehen keine Spotify-Daten mehr in die Cloud, der Abgleich passiert lokal (V2). **Markus entscheidet den Standard.**

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
