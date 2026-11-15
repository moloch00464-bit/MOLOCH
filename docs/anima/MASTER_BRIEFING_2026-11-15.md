# M.O.L.O.C.H. 5.0 – ANIMA: Master-Briefing für Claude Code

> Abgelegt am 2026-11-15, Wortlaut von Markus (erarbeitet mit ChatGPT).
> Das ist der Auftrag, nicht der Stand. Stand und Plan: `docs/anima/BESTANDSAUFNAHME_2026-11-15.md`.

Vollständiges Entwicklungs-, Architektur- und Prüfbriefing

Datum: 15. November 2026
Projekt: M.O.L.O.C.H. – lokale, verkörperte Haus-KI
Repository: `moloch00464-bit/MOLOCH`
Bisher untersuchter Branch: `deepseek_architecture_overhaul`
Umfang: 12 Entwicklungsbereiche, gemeinsame Systemarchitektur, Integrationstests und Prioritätenplanung

---

## 0. Grundsätzlicher Auftrag an Claude Code

Dieses Briefing fasst die bisherigen Entwicklungsbesprechungen zu M.O.L.O.C.H. zusammen.
Es enthält konkrete Nutzerbeobachtungen, technische Verdachtsmomente, bereits bekannte Komponenten und neue Entwicklungsziele.

In den vergangenen Tagen wurde M.O.L.O.C.H. durch mehrere KI-Entwicklungssitzungen erheblich erweitert, insbesondere durch Opus und Fable. Ein Teil dieser Arbeiten ist möglicherweise noch nicht abgeschlossen.

**Wichtige Anweisung:** Bitte zuerst den tatsächlichen aktuellen Entwicklungsstand überprüfen. Nicht davon ausgehen, dass ältere Fehlerbeschreibungen noch zutreffen. Bereits implementierte Funktionen dürfen nicht unnötig neu geschrieben werden. Die nachfolgenden Punkte sind überwiegend Prüf- und Entwicklungsaufträge, keine bereits nachgewiesenen Defekte.

### Übergeordnetes Entwicklungsziel

M.O.L.O.C.H. soll keine Ansammlung voneinander getrennter KI-Programme sein. Er soll ein zusammenhängendes lokales System werden, das:

- seine Umgebung visuell wahrnimmt,
- Geräusche und Sprache erkennen kann,
- Gespräche versteht und sinnvoll beantwortet,
- auf sein Langzeitgedächtnis zugreift,
- seine eigene Hardware kennt und überwacht,
- Informationen selbstständig im Internet recherchiert,
- Musik nach persönlichen Vorlieben auswählt,
- seine Aufmerksamkeit und sein Verhalten situationsabhängig steuert,
- seine inneren Zustände über einen Roboterkörper sichtbar ausdrückt,
- seine eigenen Bewegungen beobachten, überprüfen und schrittweise kalibrieren kann.

Der zentrale M.O.L.O.C.H.-Core bleibt die koordinierende Instanz.

### Bestehende Hardwarearchitektur

**Raspberry Pi 5** – Wahrnehmungsverarbeitung, NPU-Steuerung, Kameraverfolgung, Hardwareüberwachung, Langzeitgedächtnis, Sensorintegration, Ereignisverarbeitung, autonomer Grundbetrieb, lokales kleines Sprachmodell bei ausgeschaltetem PC. Die tatsächlich verbaute RAM-Kapazität am Pi muss überprüft werden, da ältere Dokumentationen widersprüchliche Angaben enthalten.

**Hailo-10H-NPU** – 40 TOPS bei INT4, 20 TOPS bei INT8, voraussichtlich 8 GB eigener NPU-Speicher, Bildverarbeitung und unterstützte KI-Inferenzmodelle, vorhandene YOLO-, Gesichtserkennungs-, Pose- und weitere Modelle. Die konkrete Modulvariante und Softwarekompatibilität am Gerät überprüfen.

**Massenspeicher** – zwei SSDs, vermutlich jeweils rund 500 GB. Eine ältere README nennt: SSD 1 465 GiB ext4, SSD 2 477 GiB NTFS, Daten-Mountpoint `/mnt/moloch-data`. Diese Angaben müssen mit `lsblk`, `findmnt` und `df` am laufenden System abgeglichen werden.

**PC** – leistungsfähiger Rechner mit Ollama und verschiedenen lokalen Sprachmodellen. Übernimmt umfangreichere Sprachverarbeitung, komplexe Analyseaufgaben, bei Bedarf Recherche und Planung, Training neuer Erkennungsmodelle, rechenintensive Auswertungen. Der PC soll nicht zum zweiten vollständigen Gedächtnisspeicher werden.

**Weitere Komponenten** – Sonoff-CAM-PT2 mit ONVIF/RTSP und PTZ, Seeed Studio ReSpeaker Lite Voice Assistant Kit, Spotify-Anbindung, Netzwerkverbindung zwischen Pi und PC, Reachy Mini Wireless mit internem Compute Module 4, vorhandene Lüfter und Spannungsversorgung, Guardian-, Shadow-, Berserker-, Mood-, Tension- und Dominance-Zustände.

---

## 1. Langzeitgedächtnis und Ollama verbinden

**Problembeobachtung.** Moloch kennt teilweise Informationen über seine Persönlichkeit und Identität, beantwortet aber bestimmte Fragen über seine Vergangenheit und eigene Entwicklung nicht zuverlässig. Beispielsweise kann er wissen, dass er der Hausgeist ist, ohne erklären zu können, welche Funktionen Opus und Fable programmiert haben. Es besteht der Verdacht, dass die lokale PC-KI nicht vollständig auf das Langzeitgedächtnis zugreift.

**Bekannte Codebereiche.** `core/longterm_memory.py`, `core/memory/persistent_memory.py`, `core/memory/vector_memory.py`, `core/memory/embedding_provider.py`, `core/autonomy/local_llm_bridge.py`, `core/bridge/chat_server.py`. Es existieren bereits Recall-Mechanismen und Gedächtnisfunktionen. Einzelne Embedding- und Recall-Schalter sind in untersuchten Konfigurationen standardmäßig deaktiviert. Ob sie im aktuellen System durch andere Einstellungen aktiviert werden, ist noch offen.

**Prüfauftrag.** Den gesamten Gedächtnisfluss überprüfen: Nutzerfrage → Gedächtnissuche → relevante Erinnerung → Kontextaufbereitung → Ollama → Antwort. Folgende Fragen beantworten:

- Wo liegen tatsächlich die Erinnerungen?
- Welche Daten werden auf welcher SSD gespeichert?
- Läuft Qdrant?
- Sind Embeddings aktiv?
- Funktioniert die semantische Suche?
- Welche Chat-Pfade verwenden die Gedächtnissuche?
- Erhält Ollama die gefundenen Informationen?
- Werden Informationen korrekt in den Prompt eingebettet?
- Gibt es mehrere konkurrierende Gedächtnisschnittstellen?
- Welche Erinnerungen bleiben bei ausgeschaltetem PC verfügbar?

**Zielarchitektur.** Die Pi-SSDs bleiben der maßgebliche dauerhafte Gedächtnisspeicher. Der PC ruft ausschließlich relevante Erinnerungen über eine definierte Schnittstelle ab. Keine vollständige Spiegelung der Gedächtnisdaten auf den PC. Die Verarbeitung soll zwischen folgenden Wissensarten unterscheiden: persönliche Fakten, episodische Erinnerungen, Gesprächsverlauf, Projektchronik, Selbstwissen, aktuelle Sensorinformationen, extern recherchierte Informationen. Auch das kleine Pi-Sprachmodell soll auf verfügbare Erinnerungen zugreifen können.

**Abnahmetest.** Eine neue, eindeutige Testinformation ausschließlich auf der Pi-Seite speichern. Anschließend über den PC-Chat abfragen. Im Protokoll nachweisen: 1. Erinnerung wurde gesucht. 2. Treffer wurde gefunden. 3. Inhalt wurde an Ollama übergeben. 4. Antwort berücksichtigt die Information korrekt. Keine erfundenen Erinnerungen zulassen.

---

## 2. Lokale Sprachmodelle und Antwortqualität

**Problembeobachtung.** Moloch liefert teilweise konfuse, unpassende oder halluzinierte Antworten. Die betreffenden Fragen wurden direkt in den Chat geschrieben. Die Spracherkennung ist deshalb als Ursache dieser konkreten Fehler weitgehend ausgeschlossen. Der PC zeigte nach den Fragen zusätzliche Rechenlast, was jedoch noch keinen eindeutigen Nachweis für das tatsächlich verwendete Modell darstellt.

**Prüfauftrag.** Tatsächlich laufendes Ollama-Modell, PC-/Pi-Modellrouting, Cloud-Fallbacks, aktive System-Prompts, Persönlichkeitsprofile, Kontextfenster, Temperatureinstellungen, Tokenbegrenzungen, Antwortlängenlimits, Gedächtnisinjektion, Zusammenspiel von Gespräch und Wahrnehmung. Unterschiedliche ältere Modellkonfigurationen wurden gefunden, darunter Qwen-, Dolphin-Llama- und Dolphin-Mistral-Einträge. Die tatsächliche Laufzeitkonfiguration ist entscheidend.

**Vergleichstests.** Dieselben Fragen stellen: 1. direkt an Ollama ohne Zusatzkontext, 2. Ollama mit Molochs System-Prompt, 3. Ollama mit Persönlichkeit und Gedächtnis, 4. vollständiger Moloch-Chat. Antwortqualität und Laufzeit vergleichen.

**Selbstwissen verbessern.** Moloch soll wissen: was künstliche Intelligenz ist; dass er selbst ein KI-System ist; welche Komponenten zu ihm gehören; welche seiner Fähigkeiten bereits funktionieren; welche Erweiterungen geplant sind; welche Rolle Reachy Mini übernimmt; welche Entwicklungsbeiträge Opus und Fable geleistet haben, soweit dokumentiert. Sein Charakter soll erhalten bleiben. Er darf frech, trocken und humorvoll antworten, soll aber zuerst die Frage korrekt verstehen und beantworten.

**Widersprüchliche Identitätsanweisungen.** Bestehende Dateien wie `config/system_prompt.txt` und `config/moloch_identity.json` überprüfen. Alte Bezeichnungen und Persönlichkeitsdirektiven können voneinander abweichen. Eine konsistente, versionierte Identitätsbeschreibung schaffen. Keine Persönlichkeit durch starre Kurzantworten so stark einschränken, dass die sachliche Verständlichkeit verloren geht.

**Abnahmetest.** Festgelegten Fragenkatalog testen. Modellname, Prompt-Version, Antwortdauer und relevante Kontextquellen protokollieren.

---

## 3. Chat und Kamerasteuerung zusammenführen

**Ausgangsproblem.** Molochs Chat, Sonoff-Kamera und Wahrnehmungssystem scheinen teilweise unabhängig voneinander zu funktionieren. Gesprächsinhalte beeinflussen die Kamerasteuerung möglicherweise nicht ausreichend. Kamerabeobachtungen werden nicht immer in Antworten berücksichtigt.

**Prüfauftrag.** EventBus, Chat-Server, Kamera-Controller, PTZ-Arbiter, Wahrnehmungszustand, Personentracking, Intent-Erkennung, aktuelle Kameradaten im LLM-Kontext. Nachweisen, ob die Verarbeitung in beide Richtungen funktioniert.

**Gewünschtes Verhalten.**
- Beispiel A – Nutzer: „Moloch, schau mal zur Tür." Moloch erkennt die Absicht und richtet die Kamera kontrolliert auf die bekannte Türposition.
- Beispiel B – Nutzer: „Was siehst du gerade?" Moloch verwendet aktuelle Bild- und Erkennungsinformationen.
- Beispiel C – Eine Person betritt den Kamerabereich. Molochs gemeinsamer Situationszustand wird aktualisiert. Diese Beobachtung kann eine passende Gesprächsreaktion beeinflussen.

**Sicherheitsanforderungen.** Kameraaktionen müssen durch die vorhandene Prioritätssteuerung laufen. Nicht jede beiläufige Äußerung darf eine Bewegung auslösen. Keine konkurrierenden PTZ-Controller einführen.

**Abnahmetest.** Chatbefehl auslösen, Kamerabewegung messen, Beobachtung zurückführen und nachweisen, dass sie eine anschließende Antwort beeinflusst.

---

## 4. Hailo-NPU: Bounding-Boxen und Raumkalibrierung

**Ausgangslage.** Die Hailo-NPU verarbeitet verschiedene Bildmodelle. Bounding-Boxen und Körpermerkmale müssen zuverlässiger mit tatsächlichen Positionen und Entfernungen verknüpft werden.

**Prüfauftrag.** Bounding-Box-Normalisierung, Modell- und Kameraauflösung, Pan-/Tilt-Winkel, Zoom, Kamerahöhe, Kameraausrichtung, unterschiedliche Personengrößen, sitzende und stehende Personen, Änderungen durch Perspektive, Qualität der Entfernungsschätzung.

**Gewünschtes Konzept.** Für jeden Raum ein eigenes Kalibrierprofil. Darin speichern: Kamera-ID, Raum-ID, Montageposition, Montagehöhe, Ausrichtung, Referenzpunkte, bekannte Entfernungen, Zoomzustände, Pan-/Tilt-Bezug, Kalibrierungsqualität. Referenztests beispielsweise bei 0,3 m, 0,5 m, 1,5 m, 3,0 m. Bounding-Box-Größe allein ist keine ausreichend präzise Entfernungsmessung. Bei Bedarf zusätzliche Abstandssensoren berücksichtigen.

**Abnahmetest.** Entfernungsschätzungen mit real gemessenen Abständen vergleichen. Abweichungen und Unsicherheiten protokollieren.

---

## 5. ReSpeaker Lite und dauerhafte Audiowahrnehmung

**Vorhandene Hardware.** Seeed Studio ReSpeaker Lite Voice Assistant Kit. Exakte Geräteversion, Firmware und Verbindung prüfen.

**Gewünschtes Verhalten.** Moloch soll nicht nur gesprochene Kommandos verstehen. Er soll auch allgemeine Geräuschereignisse bemerken und in seinen Situationszustand übernehmen können.

**Prüfauftrag.** USB-Anbindung an Pi prüfen, eventuell bestehende ESP32-WLAN-Audiostrecke vergleichen, Firmware überprüfen, ALSA-/PipeWire-Konfiguration prüfen, Sprachaktivitätserkennung testen, STT-Anbindung prüfen, Echo-Unterdrückung prüfen, Audio-Latenzen messen, CPU-/RAM-Belastung kontrollieren, EventBus-Anbindung testen.

**Ziel.** Eine ressourcenschonende Audio-Pipeline: Mikrofon → Aktivitätserkennung → Ereignisklassifikation → zentraler Zustand → mögliche Reaktion. Aufwendige Spracherkennung nur bei passenden Auslösern verwenden. Keine dauerhafte Aufzeichnung sämtlicher Gespräche. Stummschaltung respektieren. Rückkopplung durch Molochs eigene Sprachausgabe verhindern.

**Abnahmetest.** Ein akustisches Testereignis muss im zentralen Moloch-Zustand erscheinen und dort weiterverarbeitet werden können.

---

## 6. Spotify und Molochs persönlicher DJ

**Ausgangslage.** Spotify-Steuerung und Auto-DJ-Funktionen sind bereits in der Software vorgesehen. Sie sollen intelligenter und zuverlässiger werden.

**Prüfauftrag.** Spotify-Controller, Playlist-Funktionen, Musikgedächtnis, Track-Index, Wiedergabesteuerung, Nutzerbefehle, zustandsabhängige Musikauswahl, vorhandene Spotify-Hördaten. Untersuchen, welche Spotify-API-Funktionen tatsächlich verfügbar sind.

**Gewünschte Verbesserung.** Moloch soll: Lieblingskünstler berücksichtigen, Genres unterscheiden, Musikvorlieben lernen, passende neue Künstler entdecken, Wiederholungen reduzieren, abgelehnte Titel berücksichtigen, Musikwünsche direkt umsetzen, Playlists situationsabhängig auswählen, nach Möglichkeit angenehmere Übergänge erreichen, auf Wunsch seine Musikauswahl begründen. Guardian-, Shadow- und Berserker-Zustände dürfen die Auswahl beeinflussen. Sie sollen aber nicht zu hektischen Titelwechseln führen. Nutzerbefehle haben Vorrang.

**Abnahmetest.** Mehrere DJ-Sitzungen mit bekannten Musikpräferenzen und dokumentierten Rückmeldungen testen.

---

## 7. Internetrecherche erweitern

**Ausgangsproblem.** Die vorhandene Recherche scheint teilweise nur wenige Suchergebnisse und kurze Treffertexte auszuwerten. Für umfangreiche Fragen ist das nicht ausreichend.

**Gewünschte Funktion.** Auf Befehle wie „Recherchiere das.", „Schau im Internet nach.", „Hol dir aktuelle Informationen.", „Vergleiche verschiedene Quellen." soll Moloch eine passende Internetrecherche durchführen können.

**Prüfauftrag.** Search-Proxy untersuchen, Suchtrefferbegrenzung prüfen, Suchanbieter prüfen, Volltextabruf untersuchen, Fehlerbehandlung überprüfen, LLM-Integration testen, Quellenverwaltung prüfen.

**Gewünschte Verbesserung.** Moloch soll: unterschiedliche Suchbegriffe formulieren, mehrere Suchanfragen durchführen, relevante Webseiten lesen, offizielle Primärquellen bevorzugen, mehrere unabhängige Quellen vergleichen, Aktualität berücksichtigen, Widersprüche erkennen, Ergebnisse verständlich zusammenfassen, Quellen nachvollziehbar angeben, bei fehlgeschlagenen Suchanfragen alternative Wege versuchen. Keine erfundenen Quellen. Webseiten dürfen keine unautorisierten Systemaktionen auslösen.

**Abnahmetest.** Mehrere anspruchsvolle Recherchefragen durchführen und die gefundenen Quellen sowie die Antwortqualität dokumentieren.

---

## 8. Persönliche akustische Bibliothek

**Neue Entwicklungsaufgabe.** Moloch soll typische Geräusche seiner Wohnung erkennen lernen. Dafür soll eine persönliche Hörbibliothek entstehen.

**Praktisches Training.** Der Nutzer erzeugt gezielt Geräusche: Wohnungstür öffnen, Wohnungstür schließen, Zimmertüren bewegen, Schlüsselbund, Schritte im Flur, Schritte im Wohnzimmer, Schubladen, Fenster, Haushaltsgeräte. Die Geräusche werden mit dem ReSpeaker aufgenommen.

**Technisches Konzept.** Vortrainierte Geräuscherkennungsmodelle untersuchen. Beispielsweise YAMNet oder vergleichbare geeignete Audio-Embedding-Modelle. Falls sinnvoll, einen kleinen eigenen Klassifikator ergänzen.

**Trainingsablauf.** 1. Hörprobe aufnehmen. 2. Geräusch beschriften. 3. Mehrere Varianten sammeln. 4. Hintergrundgeräusche und Gegenbeispiele erfassen. 5. Modell anpassen. 6. Mit neuen Aufnahmen testen. 7. Ergebnis in den EventBus einbinden.

**Gewünschtes Verhalten.** Moloch erkennt beispielsweise ein Geräusch, das seiner Wohnungstür ähnelt. Er kann diese Information gegebenenfalls mit der Kamerawahrnehmung verbinden. Aus einem Geräusch allein darf er keine bestimmte Person identifizieren.

**Speicherung.** Hörproben und trainierte Geräuschprofile lokal auf der Pi-SSD speichern. Keine unnötige dauerhafte Speicherung privater Gespräche.

**Abnahmetest.** Neue Geräuschaufnahmen mit nachvollziehbarer Trefferquote klassifizieren. Fehlalarme dokumentieren.

---

## 9. Reachy Mini: Sprechbewegungen und Simulator

**Konkrete Nutzerbeobachtung.** Im Simulator haben sich Kamera und Reachy bereits bewegt. Während gesprochener Antworten zeigte Reachy jedoch keine erkennbare zusätzliche Sprechbewegung.

**Bereits vorhandener Code.** In `core/body/expression.py` existiert ein `speaking`-Zustand, der leichte Kopfbewegungen erzeugen kann. Die `pc/body_bridge.py` übernimmt Zustandsinformationen. Es ist jedoch noch nicht bewiesen, dass alle TTS-Wege korrekt an den Körpercontroller angebunden sind.

**Prüfauftrag.** PC-TTS untersuchen, Pi-TTS untersuchen, Browser-Sprachausgabe prüfen, Sprechbeginn erfassen, Sprechende erfassen, `speaking`-Status verfolgen, BodyController untersuchen, Simulator-Ansteuerung testen.

**Gewünschtes Verhalten.** Während Moloch spricht, soll Reachy leicht nicken, Kopf neigen, Antennen bewegen, Betonungen unterstützen, passende emotionale Bewegungen zeigen. Die Animationen sollen zur Stimmung passen. Keine hektischen Dauerbewegungen. Die Bewegung muss möglichst mit der tatsächlichen Audiowiedergabe synchronisiert sein. Bei einem Stoppbefehl muss sie ebenfalls enden.

**Abnahmetest.** Eine hörbare Antwort im Simulator muss eine sichtbare Sprechbewegung erzeugen. Anschließend kehrt Reachy kontrolliert in seine normale Haltung zurück.

---

## 10. Gemeinsamer Core: Wahrnehmung, Gedächtnis und Körper

**Übergeordnetes Ziel.** Alle vorhandenen Komponenten sollen miteinander kommunizieren. Es soll keinen isolierten Chat, keine isolierte Kamera und kein isoliertes Gedächtnis geben. Stattdessen einen gemeinsamen Situationszustand.

**Benötigte Informationen.** Aktuelle Kamerabeobachtungen, akustische Ereignisse, Gesprächskontext, relevante Erinnerungen, Hardwarezustand, aktuelle Persönlichkeit, Tension, Mood, Aufmerksamkeit, Körperzustand, aktive Nutzerbefehle. Jede Information soll nach Möglichkeit einen Zeitstempel, eine Quelle, eine Gültigkeitsdauer und gegebenenfalls eine Erkennungssicherheit besitzen.

**Beispielablauf.**
1. ReSpeaker erkennt ein Türgeräusch.
2. Geräuschbibliothek klassifiziert es.
3. EventBus meldet das Ereignis.
4. Moloch entscheidet, ob eine visuelle Überprüfung sinnvoll ist.
5. Sonoff-Kamera bewegt sich kontrolliert.
6. Hailo verarbeitet das Bild.
7. Situationszustand wird aktualisiert.
8. Bei Bedarf werden Erinnerungen geladen.
9. Ollama formuliert eine passende Antwort.
10. Reachy zeigt Aufmerksamkeit und spricht gegebenenfalls.

Nicht jede Beobachtung soll zu einer gesprochenen Reaktion führen. Moloch soll häufig auch einfach still beobachten.

**Reachy-Bibliotheken.** Die vollständigen verfügbaren offiziellen Reachy-Mini-Bewegungsbibliotheken berücksichtigen: emotionale Gesten, Antennenbewegungen, Kopfhaltungen, Körperbewegungen, Tanzbewegungen, Bewegungssequenzen, spätere eigene Erweiterungen. Die Bewegungsbibliotheken sollen möglichst auf dem Reachy-CM4 laufen. Molochs Core bleibt auf Pi und PC. Die vorhandene MOLOCH_REACHY_BRIDGE wiederverwenden.

**Prioritäten.** 1. Sicherheitsstopp. 2. Direkte Nutzersteuerung. 3. Notwendige Wahrnehmungsreaktion. 4. Gesprächsausdruck. 5. Emotionale Bewegung. 6. Idle-Verhalten. 7. Tanz und Show.

**Abnahmetest.** Eine simulierte Situation muss nachvollziehbar durch sämtliche beteiligten Komponenten laufen.

---

## 11. Hardware-Selbstwissen und Systemüberwachung

**Grundidee.** Moloch soll seine eigene Hardware kennen und deren tatsächlichen Zustand überwachen. Eine dokumentierte Komponente ist nicht automatisch eine tatsächlich vorhandene oder funktionierende Komponente.

**Hardware-Inventar.** Zu überprüfen: Raspberry Pi 5, tatsächlicher Pi-RAM, Hailo-10H, NPU-Speichervariante, beide SSDs, Sonoff-Kamera, ReSpeaker, Noctua-Lüfter, Power-Management-Platine, Spannungsversorgung, Netzwerkverbindungen, PC und Ollama, Reachy Mini.

**Hailo-10H: Technische Daten (Herstellerangaben).** 40 TOPS bei INT4, 20 TOPS bei INT8, Modulvarianten mit 4 oder 8 GB eigenem Speicher, typische Leistungsaufnahme etwa 2,5 W, M.2 mit PCIe-3.0-x4-Unterstützung, Unterstützung für Computer Vision und geeignete generative KI-Modelle. Die tatsächliche PCIe-Anbindung kann durch Raspberry Pi und Adapter eingeschränkt sein.

**Dokumentationsfehler.** Eine ältere Moloch-README nennt für den Hailo-10H 26 TOPS. Das entspricht der älteren Hailo-8-Leistung und muss bei bestätigtem Hailo-10H korrigiert werden. Pi-RAM und NPU-RAM getrennt dokumentieren.

**Softwareprüfung.** HailoRT-Version, Treiberversion, Firmware, TAPPAS-Version, GStreamer-Kompatibilität, Model-Zoo-Kompatibilität, HEF-Kompatibilität, aktive Inferenzmodelle. Keine automatischen Updates funktionierender Treiber durchführen.

**Vorhandene Diagnosekomponenten.** `context/system_autonomy.py`, `core/hardware/hailo_manager.py`, `scripts/self_diagnosis.py`. Bereits vorhandene Funktionen wiederverwenden.

**Wichtiger Befund.** Die Prüfung auf `/dev/hailo0` bestätigt lediglich die Existenz des Geräteknotens. Sie ist kein vollständiger Nachweis einer funktionierenden NPU-Inferenz. Der vorhandene Selbstdiagnosetest muss darauf überprüft werden, ob tatsächlich ein Modell inferiert und dessen Ergebnis kontrolliert wird.

**NPU-Selbstüberwachung.** Soweit technisch verfügbar: exakter Chiptyp, Firmware, Treiber, NPU-Erreichbarkeit, aktives Modell, Inferenz-FPS, Verarbeitungsfehler, NPU-Temperatur, PCIe-Zustand, Ressourcenbelegung, fehlgeschlagene Modellinitialisierungen. Nicht verfügbare Messwerte ausdrücklich als unbekannt kennzeichnen.

**Geeignete Diagnosebefehle prüfen.** `hailortcli scan`, `hailortcli fw-control identify --extended`, `hailortcli monitor`, `hailortcli --help`, `lspci -vv`, `lsmod | grep hailo`, `dmesg | grep -i hailo`. Nur passende, möglichst nicht eingreifende Diagnoseabfragen verwenden.

**Raspberry-Pi-Überwachung.** CPU-Temperatur, CPU-Auslastung, RAM-Auslastung, Unterspannung, Throttling, Systemlaufzeit, Lüfterstatus, SSD-Kapazität, Datenträgergesundheit, Netzwerkstatus.

**Noctua-Lüfter.** Anschlussart überprüfen. Bei einem einfachen zweipoligen Lüfter ist eine direkte Drehzahlmessung möglicherweise nicht verfügbar. Wenn Tachosignal oder geeignete Steuer- beziehungsweise Messhardware vorhanden ist, diese nutzen. Keine erfundenen Drehzahlwerte.

**Power-Management.** Die tatsächliche Power-Platine identifizieren. Prüfen, ob folgende Daten auslesbar sind: Eingangsspannung, Ausgangsspannung, Stromaufnahme, Temperatur, Akkuzustand (falls vorhanden), Fehler- und Abschaltsignale. Nur real unterstützte Messwerte anzeigen.

**Selbstdiagnose über Chat.** Beispiele: „Moloch, wie warm ist dein Raspberry?", „Wie geht es deinem Hailo-Chip?", „Wie viel Speicher hast du noch?", „Funktioniert deine Kamera?" Moloch soll aktuelle Diagnosewerte abfragen und verständlich antworten.

**Proaktive Warnungen.** Relevante Ereignisse: kritische Temperatur, Unterspannung, SSD fast voll, NPU-Ausfall, Kameraverbindung verloren, wiederholte Pipeline-Abstürze, PC-Verbindung ausgefallen. Bei kleineren Problemen nur protokollieren. Bei kritischen Problemen kurz informieren.

**Sicherheitsregel.** Keine unkontrollierten Neustarts, Prozessabbrüche oder Firmwareaktionen. Besonders die bestehende Hailo-Ressourcenverwaltung auf potenziell riskante Force-Release-Mechanismen prüfen. Funktionierende Prozesse dürfen durch Diagnosen nicht unnötig beendet werden.

**Abnahmetest.** Moloch erkennt seine Hardware, beschreibt ihren Zustand korrekt und unterscheidet bekannte, gemessene und unbekannte Werte. Die grundlegende Überwachung muss auch ohne PC funktionieren.

---

## 12. Reachy Mini: eigenes Hailo-Erkennungsmodell, visuelle Selbstwahrnehmung und Selbstkalibrierung

**Neue Projektidee.** Moloch erhält mit Reachy Mini einen physischen Ausdruckskörper. Gleichzeitig verfügt er über eine externe Sonoff-PTZ-Kamera. Dadurch entsteht eine interessante Möglichkeit: Reachy kann über die Sonoff-Kamera seinen eigenen Körper von außen beobachten. Er könnte beispielsweise eine Antenne bewegen, diese Bewegung im Kamerabild verfolgen und mit dem ausgeführten Bewegungsbefehl vergleichen. Das soll zunächst als kontrollierte visuelle Selbstkontrolle und spielerisches Spiegelverhalten entwickelt werden. Nicht als Behauptung menschlichen Selbstbewusstseins.

### 12.1 Eigenes Reachy-Erkennungsmodell entwickeln

Für den Hailo-10H soll ein spezialisiertes Erkennungsmodell vorbereitet werden. Das Modell soll Reachy im Bild identifizieren und möglichst einzelne Körperteile unterscheiden.

Gewünschte Erkennungsklassen: gesamter Reachy Mini, Kopf, Gesichtsbereich beziehungsweise Vorderseite, Körper beziehungsweise Sockel, linke Antenne, rechte Antenne, Antennenspitzen, Antennenansatzpunkte.

Zusätzlich gewünschte geometrische Eigenschaften: Kopfdrehung, Kopfneigung, Antennenstellung, Sichtbarkeit der Körperteile, Position im Kamerabild, Orientierung zur Kamera.

Dabei berücksichtigen, dass linke und rechte Antenne je nach Blickrichtung im Kamerabild vertauscht erscheinen können. Die interne Körperkoordinate muss maßgeblich bleiben.

### 12.2 Geeignete Modellerkennung auswählen

Prüfen, ob ein gewöhnlicher Objektdetektor genügt. Für dünne Antennen könnten Keypoint-Detection, Pose Estimation oder Segmentierung geeigneter sein. Mögliche Ansätze: kleines spezialisiertes YOLO-Modell, Keypoint-Modell, Segmentierungsmodell, Kombination aus Objekterkennung und geometrischer Nachverarbeitung. Die Auswahl soll anhand der Genauigkeit, Modellgröße und tatsächlichen Hailo-10H-Kompatibilität erfolgen. Nicht automatisch das größte Modell verwenden.

### 12.3 Eigenes Trainingsprogramm vorbereiten

Claude soll einen nachvollziehbaren Trainingsablauf entwerfen.

**Datensammlung.** Reachy aus unterschiedlichen Blickwinkeln aufnehmen: von vorne, von hinten, von links, von rechts, schräg von oben, schräg von unten, mit verschiedenen Kopfhaltungen, mit unterschiedlichen Antennenstellungen, bei verschiedenen Entfernungen, bei wechselnden Lichtbedingungen, vor unterschiedlichen Hintergründen. Die Sonoff-Kamera soll möglichst für einen wesentlichen Teil der Trainingsaufnahmen verwendet werden. Dadurch entspricht die Bildqualität späteren realen Einsatzbedingungen.

**Datenbeschriftung.** Die relevanten Körperteile müssen korrekt markiert werden. Ein benutzerfreundlicher Trainingsablauf wäre wünschenswert. Zusätzlich können die vom Reachy-SDK gemeldeten Soll- und Ist-Stellungen als Metadaten verwendet werden, sofern verfügbar und zeitlich korrekt zugeordnet.

**Training.** Das Training findet auf einem geeigneten Rechner statt. Der Hailo-10H selbst ist kein gewöhnliches On-Device-Trainingssystem für solche Modelle. Vorgeschlagener Ablauf: Aufnahmen → Annotationen → Training am PC → Validierung → Hailo-kompatible Konvertierung und Kompilierung → HEF-Modell → Inferenz auf Hailo-10H. Vor der Implementierung die aktuelle Hailo-10H-Toolchain und unterstützten Modelloperationen überprüfen.

**Trainingsdatenqualität.** Trainings-, Validierungs- und Testdaten sauber trennen. Nicht dieselben nahezu identischen Videoframes in Training und Test verwenden. Auch Aufnahmen ohne Reachy und mit teilweise verdecktem Reachy berücksichtigen. Eine eventuell verfügbare Simulator-Datengenerierung kann ergänzen, ersetzt aber keine realen Testaufnahmen.

### 12.4 Hailo-Integration

Das fertige Modell soll auf dem Pi mit dem Hailo-10H inferiert werden können. Prüfen: HEF-Kompatibilität, Modellgröße, Inferenzgeschwindigkeit, Arbeitsspeicher, GStreamer-Anbindung, Ressourcenkonflikte, Kompatibilität zu bestehenden Vision-Modellen.

**Wichtige Einschränkung.** Das zusätzliche Reachy-Modell darf die stabile Personen- und Gesichtserkennung nicht unnötig beeinträchtigen. Eine bedarfsgesteuerte Aktivierung ist sinnvoll. Beispielsweise wird das Reachy-Modell nur aktiviert, wenn Moloch seinen Körper im Kamerabild suchen oder überprüfen möchte.

### 12.5 Reachy muss sich als eigener Körper identifizieren

Ein erkanntes Reachy-Objekt ist nicht automatisch der eigene Roboter. Moloch soll eine stabile Zuordnung zwischen seinem bekannten Roboterkörper und der visuellen Erkennung herstellen. Dazu kombinieren: bekannte Reachy-Hardwareidentität, vorhandene Roboterschnittstelle, aktuelle Bewegungsbefehle, gemeldete Gelenkpositionen, Kamerabeobachtungen, zeitliche Synchronisation. Wenn mehrere ähnliche Roboter oder Bilder von Reachy sichtbar sind, darf keine unbegründete Selbstzuordnung erfolgen.

### 12.6 Visueller Bewegungsabgleich

Das eigentliche Kernstück des Projekts. Beispiel: Moloch befiehlt Reachy, die linke Antenne anzuheben. Das System erfasst: 1. Bewegungsbefehl, 2. gewünschte Antennenstellung, 3. vom Reachy-SDK gemeldete Stellung, 4. beobachtete Veränderung im Sonoff-Kamerabild, 5. Abweichung zwischen Erwartung und Beobachtung. Auf dieser Grundlage soll er erkennen können, ob die Bewegung wie erwartet ausgeführt wurde. Auch Kopfbewegungen sollen überprüft werden können.

### 12.7 Visuelle Selbstkalibrierung

Mit mehreren kontrollierten Bewegungen könnte Moloch prüfen, wie sein Körper von außen erscheint. Beispielsweise: Kopf nach links drehen, Kopf nach rechts drehen, Kopf nach oben neigen, Antennen aufrichten, Antennen absenken, neutrale Haltung einnehmen. Die Sonoff-Kamera erfasst diese Bewegungen. Das System kann daraus eine Zuordnung zwischen Gelenkstellungen und sichtbaren Körperpositionen verbessern. Diese Kalibrierung soll schrittweise und nachvollziehbar erfolgen. Keine unkontrollierte Selbstveränderung von Bewegungsgrenzen.

### 12.8 Gemeinsames Raumkoordinatensystem

Moloch soll langfristig wissen: wo die Sonoff-Kamera installiert ist, welche Kamera welche Identität besitzt, wo Reachy steht, welche Blickrichtung die Kameras haben, wo sich Reachy im Kamerabild befindet. Dazu ein gemeinsames Raum- und Kamerakoordinatensystem vorbereiten. Mögliche technische Grundlagen: OpenCV-Kamerakalibrierung, extrinsische Kameraparameter, bekannte Referenzpunkte, ArUco- oder AprilTag-Markierungen während der Kalibrierung, PTZ-Positionsdaten, bekannte Raumgeometrie. Eine einzelne Bounding-Box reicht für eine vollständige dreidimensionale Lokalisierung nicht aus.

### 12.9 Mehrkamerasystem

Reachy besitzt eine eigene Kopfkamera. Zusätzlich existiert die externe Sonoff-Kamera. Beide Blickrichtungen sollen später im gemeinsamen Wahrnehmungssystem verfügbar sein. Die externe Kamera kann Reachy beobachten. Reachys eigene Kamera sieht seine Umgebung. Ein gemeinsames räumliches Modell kann beide Perspektiven miteinander verknüpfen. Das ist auch eine Vorbereitung für einen späteren mobilen Roboterkörper. Wichtig: Fortbewegung durch die Wohnung ist derzeit keine Priorität. Zunächst geht es um Erkennung, Selbstbeobachtung, Kamerazuordnung und kontrollierte Bewegungsrückmeldung.

### 12.10 Spielerisches Spiegelverhalten

Neben der technischen Kalibrierung soll Moloch daraus eine kleine Show-Funktion entwickeln können. Beispiel: Reachy wird von der Sonoff-Kamera erfasst. Er erkennt die eigene Körperposition. Dann bewegt er bewusst eine Antenne. Die Kamera beobachtet die Veränderung. Moloch vergleicht die Bewegung mit seiner Erwartung. Bei erfolgreicher Erkennung kann er eine passende Geste oder einen kurzen Kommentar erzeugen.

Mögliche Verhaltensweisen: neugieriges Kopfneigen, Antennen bewusst aufrichten, beide Antennen unterschiedlich positionieren, Körperhaltung überprüfen, kleine Überraschungsreaktion, kurzer humorvoller Kommentar, zufriedene Geste nach erfolgreicher Kalibrierung. Die Reaktion soll auf echten Beobachtungsdaten beruhen. Nicht ausschließlich eine fest vorprogrammierte Animation abspielen.

### 12.11 Rechenverteilung

**Raspberry Pi mit Hailo** – möglichst: Reachy-Erkennung, Körperteil-Erkennung, einfache Positionsverfolgung, Vergleich mit bekannten Bewegungsdaten, Ereignisausgabe.

**PC** – bei Bedarf: Modelltraining, umfangreichere Bildanalyse, Optimierung der Kalibrierung, anspruchsvollere räumliche Auswertung, komplexere Verhaltensentscheidungen.

**Reachy-CM4** – möglichst: lokale Bewegungsbibliotheken, Ausführung von Bewegungssequenzen, Gelenkstatus, Sicherheitsbegrenzungen, Kommunikation mit der Body-Bridge.

Ein dauerhaft laufendes großes Sprachmodell soll für einfache Bewegungsvergleiche nicht erforderlich sein.

### 12.12 Sicherheit und Stabilität

Gelenkgrenzen einhalten. Keine hektischen Bewegungen. Keine unkontrollierten Kalibrierexperimente. Keine Selbstkalibrierung bei unsicherer Erkennung. Not-Aus respektieren. Bewegungen abbrechen können. Kameraverlust erkennen. Keine Korrektur aus nur einem unsicheren Einzelbild. Veränderungen an Kalibrierparametern nachvollziehbar dokumentieren.

### 12.13 Konkrete Abnahmetests

- **Test A: Körpererkennung.** Sonoff zeigt auf Reachy. Das Modell erkennt Reachy korrekt.
- **Test B: Körperteile.** Kopf, Körper und Antennen werden unter definierten Bedingungen unterschieden.
- **Test C: Bewegungsabgleich.** Reachy bewegt eine Antenne. Die Kamera erkennt die Veränderung.
- **Test D: Fehlererkennung.** Eine angeforderte Bewegung wird nicht oder anders sichtbar ausgeführt. Das System erkennt die Abweichung, soweit die Beobachtungsqualität dafür ausreicht.
- **Test E: Spiegel-Show.** Moloch erkennt seinen Körper, löst eine kontrollierte Bewegung aus, beobachtet das Ergebnis und reagiert darauf.
- **Test F: PC ausgeschaltet.** Die einfache Körpererkennung und der Bewegungsabgleich funktionieren nach Möglichkeit weiterhin lokal über Pi, Hailo und Reachy.
- **Test G: Ressourcenprüfung.** Das zusätzliche Modell darf bestehende stabile Wahrnehmungsfunktionen nicht unkontrolliert beeinträchtigen.

---

## 13. Gemeinsame Architekturregeln

Diese Regeln gelten über alle zwölf Aufgaben hinweg.

**Eine zentrale Zustandsverwaltung.** Nicht für jede neue Funktion ein zusätzliches unabhängiges Zustandsmodell erschaffen. Bestehende State-Engines und den EventBus untersuchen und soweit sinnvoll wiederverwenden.

**Gemeinsame Ereignisstruktur.** Ereignisse sollen möglichst enthalten: Ereignistyp, Quelle, Zeitstempel, relevante Nutzdaten, Sicherheit der Erkennung, Gültigkeitsdauer, Priorität, zugehörige Kamera oder Hardware, Bearbeitungsstatus.

**Saubere Trennung von Beobachtung und Interpretation.** Beispiel – Beobachtung: „Objekt mit hoher Sicherheit als Reachy erkannt." Interpretation: „Das könnte mein eigener Roboterkörper sein." Bestätigung: „Bekannte Bewegungsdaten stimmen mit der beobachteten Bewegung überein." Diese Stufen nicht ungeprüft gleichsetzen.

**Ressourcenverwaltung.** Hailo, CPU, RAM und PC sollen nicht mit konkurrierenden Aufgaben überlastet werden. Schwere Modelle nur bei Bedarf aktivieren. Bestehende 25–28-FPS-Wahrnehmung nach Möglichkeit erhalten und messbar prüfen.

**Ausfallsicherheit.** Wenn PC, Kamera, NPU oder Gedächtnis zeitweise ausfallen, darf Moloch keine Zustände erfinden. Er muss mit eingeschränkten Funktionen weiterarbeiten können, soweit technisch und sicher möglich.

**Datenschutz.** Private Kamerabilder und Audioaufnahmen bleiben grundsätzlich lokal. Keine unnötige Übertragung nach außen. Aktive Aufnahme- und Überwachungszustände müssen kontrollierbar sein.

---

## 14. Prioritätenplan für die nächsten Entwicklungssitzungen

Claude soll zunächst einen realistischen Arbeitsplan erstellen. Nicht alle zwölf Bereiche gleichzeitig vollständig umbauen.

- **Priorität 1: Grundlagen stabilisieren** – tatsächlich laufendes Ollama-Modell bestimmen, Antwortqualität prüfen, Gedächtnisabruf nachweisen, Hardwarestatus überprüfen, widersprüchliche Konfigurationen identifizieren.
- **Priorität 2: Bestehende Systeme verbinden** – Chat und Kamera, Chat und Langzeitgedächtnis, TTS und Reachy-Bewegung, Audioereignisse und EventBus, gemeinsamen Situationszustand vervollständigen.
- **Priorität 3: Wahrnehmung ausbauen** – persönliche Geräuschbibliothek, Raumkalibrierung, Mehrkamerazuordnung, Reachy-Körpererkennung.
- **Priorität 4: Intelligente Funktionen verbessern** – Internetrecherche, Spotify-DJ, Projektwissen, Selbstwissen, kontextabhängige Reaktionen.
- **Priorität 5: Reachy-Selbstbeobachtung** – eigene Trainingsdaten, spezialisiertes Hailo-Modell, Erkennung der Körperteile, Bewegungsabgleich, Selbstkalibrierung, spielerisches Spiegelverhalten.

---

## 15. Konkrete Arbeitsweise für Claude Code

- **Phase A: Aktuellen Code untersuchen.** Den neuesten relevanten Repository-Stand prüfen. Bereits vorhandene Änderungen von Opus und Fable berücksichtigen. Vorhandene Tests und Dokumentationen auswerten.
- **Phase B: Laufzeit überprüfen.** Nicht nur Code lesen. Auf dem tatsächlichen System prüfen: Welche Prozesse laufen? Welche Modelle sind aktiv? Welche Dienste sind erreichbar? Welche Datenbanken werden benutzt? Welche Hardware wird erkannt? Welche Ereignisse werden wirklich verarbeitet?
- **Phase C: Fehler bestätigen.** Für jeden gefundenen Fehler einen reproduzierbaren Nachweis erstellen. Fehlerhafte Konfiguration, fehlende Verbindung und noch nicht implementierte Funktion voneinander unterscheiden.
- **Phase D: Reparatur planen.** Kleine, nachvollziehbare Änderungen bevorzugen. Bestehende Komponenten wiederverwenden. Große Architekturänderungen nur mit technischer Begründung vorbereiten.
- **Phase E: Implementierung.** Geänderte Module gezielt testen. Keine stabilen Systeme unnötig beeinträchtigen. Konfigurationen und Schnittstellen dokumentieren.
- **Phase F: Integrationstests.** Vollständige Verarbeitungsketten testen. Beispielsweise: ReSpeaker → Geräuscherkennung → EventBus → Kamera → Hailo → Gedächtnis → Ollama → TTS → Reachy. Und: Reachy-Bewegungsbefehl → Körperbewegung → Sonoff-Beobachtung → Reachy-Modell auf Hailo → Bewegungsabgleich → Reaktion.
- **Phase G: Abschlussbericht.** Für jeden der zwölf Entwicklungsbereiche dokumentieren: bereits vorhanden | nachweislich funktionsfähig | fehlerhaft | fehlend | Änderung vorgeschlagen | umgesetzt | erfolgreich getestet. Offene Punkte ausdrücklich als offen kennzeichnen.

---

## 16. Wichtige Sicherheits- und Entwicklungsregeln

1. Keine vorhandenen Erinnerungen löschen.
2. Keine vollständige Übertragung der Pi-SSDs auf den PC.
3. Keine riskanten Firmwareänderungen ohne ausdrückliche Freigabe.
4. Keine unnötigen Änderungen an stabilen Hailo-Pipelines.
5. Keine unkontrollierten autonomen Bewegungen.
6. Keine erfundenen Hardwaremesswerte.
7. Keine konkurrierenden Kamera-Controller.
8. Keine zweite parallele Gedächtnisarchitektur ohne Begründung.
9. Keine ungesicherten automatischen Reparaturmaßnahmen.
10. Keine Behauptung, dass eine Funktion arbeitet, wenn nur ihr Code existiert.
11. Kein automatisches Umschreiben von Bewegungsgrenzen durch die visuelle Selbstkalibrierung.
12. Keine umfangreichen Implementierungsarbeiten ohne vorherige Bestandsaufnahme und nachvollziehbaren Plan.

---

## 17. Abschließendes Projektziel

M.O.L.O.C.H. soll eine lokale, zusammenhängende und verkörperte KI werden. Der Raspberry Pi übernimmt die Wahrnehmung, Steuerung und dauerhafte Gedächtnisverwaltung. Der Hailo-10H verarbeitet geeignete neuronale Modelle. Der PC übernimmt leistungsfähigere Sprach- und Analyseaufgaben. Reachy Mini wird zur physischen Ausdrucksebene. Die Sonoff-Kamera dient nicht nur der Beobachtung der Wohnung, sondern perspektivisch auch der Beobachtung von Reachys eigenem Körper. Das ReSpeaker-Mikrofon stellt akustische Wahrnehmung bereit. Das Langzeitgedächtnis verbindet aktuelle Ereignisse mit vergangenen Erfahrungen. Molochs Charakter und Verhaltenszustände beeinflussen seine Ausdrucksweise, ohne die technische Zuverlässigkeit zu beeinträchtigen.

Langfristiges Ziel: Moloch kann sehen, hören, sprechen, sich erinnern, recherchieren, Musik auswählen, seine Hardware überwachen und sich durch Reachy ausdrücken. Darüber hinaus kann er seinen eigenen Roboterkörper erkennen, Bewegungen von außen beobachten und die sichtbaren Ergebnisse mit seinen Bewegungsbefehlen vergleichen. Diese Selbstbeobachtung soll sowohl technische Kalibrierung als auch spielerisches Verhalten ermöglichen.

Dabei geht es nicht um die Behauptung menschlichen Bewusstseins, sondern um ein überprüfbares, integriertes System aus Wahrnehmung, Körpermodell, Gedächtnis, Entscheidung und Rückmeldung.

Die eigentliche Aufgabe besteht darin, aus vielen einzelnen funktionierenden Komponenten ein einziges verlässlich zusammenarbeitendes M.O.L.O.C.H.-System zu machen.

---

*Ende des Master-Briefings – 15. November 2026. Alle zwölf Entwicklungsbereiche enthalten. Nächster Schritt für Claude Code: erst Bestandsaufnahme und Prioritätenplan, danach gezielte Umsetzung mit überprüfbaren Ergebnissen.*
