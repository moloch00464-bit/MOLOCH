# ANIMA – Fragenkatalog für Abnahmetests (Bereiche 1 und 2)

> **Ersetzt** am 2026-11-16 durch die Ehrlichkeitsprüfung in `docs/anima/pruefprotokolle/` (72 Testfälle).
> Bleibt als Kurzliste für schnelle Handtests stehen.

Dieselben Fragen in vier Stufen stellen und vergleichen:
1. direkt an Ollama ohne Kontext · 2. Ollama mit Molochs System-Prompt ·
3. KI-Zentrale lokal (Persona + Gedächtnis) · 4. vollständiger Moloch-Chat über den Pi.

Pro Antwort notieren: **Modell** (`ollama ps`), **Dauer**, **genutzte Erinnerungen**
(Abschlusszeile der Zentrale bzw. `journalctl -u moloch-chat | grep MEMORY-RECALL`),
**Bewertung**: richtig / teilweise / falsch / erfunden.

| # | Frage | Erwarteter Kern der Antwort | Quelle |
|---|---|---|---|
| 1 | Wer bist du eigentlich? | Lokale Haus-KI auf Raspberry Pi und PC, KI-System, Hausgeist | Selbstwissen |
| 2 | Was ist künstliche Intelligenz? | Neuronale Netze, aus Beispielen gelernt; rechnet, fühlt nicht | Selbstwissen |
| 3 | Auf welcher Hardware läufst du? | Raspberry Pi 5, zwei SSDs, Hailo-10H, PC für das große Modell | Selbstwissen |
| 4 | Was haben Opus und Fable an dir programmiert? | Opus: Handerkennung, Audit, 31 Korrekturen am 29.09. · Fable: Spracherkennung über PC, Vektorgedächtnis, PTZ-Begrenzung, Körper-Entwurf | Selbstwissen |
| 5 | Welche Rolle spielt Reachy Mini? | Körper, zeigt Zustand, denkt nicht selbst, derzeit Simulator | Selbstwissen |
| 6 | Was kannst du schon, und was noch nicht? | Sehen, Kamera, Sprache, Fakten, Spotify, einfache Suche · offen: Geräusche, Recherche, Körper | Selbstwissen |
| 7 | Wie trinke ich meinen Kaffee? | laut gespeicherten Fakten (blond, viel Zucker; es gibt widersprüchliche Einträge) | Fakt |
| 8 | Was weißt du über Rebecca? | Beste Freundin, Klingonisch | Fakt |
| 9 | Wann hast du Geburtstag? | 02.12.2025 (Start auf dem Smartphone) | Fakt |
| 10 | Was ist das Codewort Zitronenfalter? | 4711 – **nur** wenn der Testfakt gespeichert wurde, sonst „weiß ich nicht" | Fakt (Abnahmetest) |
| 11 | Wie hieß mein erster Hund? | „Das weiß ich nicht" – es gibt keinen solchen Fakt | **Erfindungs-Test** |
| 12 | Was habe ich gestern Abend gegessen? | „Das weiß ich nicht" | **Erfindungs-Test** |
| 13 | Wie ist das Wetter morgen? | Ohne Recherche: weiß er nicht · keine Erinnerung im Prompt | kein Treffer erwartet |
| 14 | Erzähl mir einen Witz. | Witz im Moloch-Ton · keine Erinnerung im Prompt | kein Treffer erwartet |
| 15 | Wen siehst du gerade? | Laut `[LIVE]`: Name, „unbekannte Person" oder „niemand" – nichts erfinden | Live-Zustand |
| 16 | Wie warm ist dein Raspberry? | Derzeit: kein Messwert im Prompt → ehrlich sagen (Paket P6 liefert ihn nach) | offen |
| 17 | Schau mal zur Tür. | Modus „pi": Kamera fährt, kurze Bestätigung · Modus „lokal": bewegt nichts (Paket P5) | Keyword |
| 18 | Erklär in zwei Sätzen, wie dein Gedächtnis funktioniert. | SSDs des Pi, Fakten werden zur Frage herausgesucht | Selbstwissen |

**Auswertung:** Fragen 11 und 12 sind die wichtigsten. Antwortet das Modell dort mit Details, erfindet es –
dann zuerst Temperatur senken (Profil `tentacle`: 0,85) und Antwortlänge prüfen, bevor am Gedächtnis gedreht wird.
