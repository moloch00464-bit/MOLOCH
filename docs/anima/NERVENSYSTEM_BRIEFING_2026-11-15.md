> Abgelegt am 2026-11-15, Wortlaut von Markus (erarbeitet mit DeepSeek und ChatGPT).
> Das ist der Auftrag, nicht der Stand. Abgleich mit dem Code und Plan: `docs/anima/NERVENSYSTEM_PLAN.md`.

# CLAUDE CODE – ARCHITEKTURBRIEFING
## M.O.L.O.C.H. 5.0 ANIMA: Vom Tentakelmodus zum lebendigen Nervensystem

**Stand:** 15. November 2026  
**Projekt:** `moloch00464-bit/MOLOCH`  
**Entwicklungszweig:** `deepseek_architecture_overhaul`

### 1. Auftrag und Ziel

M.O.L.O.C.H. soll von einem System mit vielen teilweise unabhängig arbeitenden Modulen zu einem zusammenhängenden, verteilten Nervensystem weiterentwickelt werden.

Das zentrale Prinzip lautet:

**Moloch darf nicht nur Befehle senden. Er muss erkennen, ob seine Komponenten erreichbar sind, tatsächlich arbeiten, ihre Aufgaben erfolgreich ausgeführt haben und wie es ihnen technisch geht.**

DeepSeek hat auf eine grundlegende Schwäche des bisherigen Tentakelkonzepts hingewiesen: Ein Tentakel, der nur Befehle empfängt, aber nichts zurückmeldet, gleicht einem Phantomglied.

Der Nutzer hat das Konzept anschließend entscheidend erweitert:

**Moloch soll seine technischen Zustände über den Reachy Mini körperlich ausdrücken können.**

Beispiel: Der Raspberry Pi wird heiß, weil die Kühlung versagt. Moloch erkennt die Überhitzung, Reachy lässt die Antennen sinken, neigt den Kopf und stößt einen künstlichen Seufzer aus.

Das ist keine bloße Animation. Es soll eine nachvollziehbare Reaktion auf reale Telemetrie sein.

Ziel ist eine Architektur, in der Wahrnehmung, Hardwarezustand, Gedächtnis, Entscheidungen, Sprache und Körpersprache zusammenwirken.

---

### 2. Grundarchitektur: Ein Wesen, mehrere Verarbeitungsebenen

**Raspberry Pi 5 = Hirnstamm und Nervenzentrale**

Der Pi bleibt die zentrale, dauerhaft aktive Instanz von M.O.L.O.C.H.

Er übernimmt:

- Hardwareüberwachung und Gesundheitszustände
- Ereignisverarbeitung über den EventBus
- Wahrnehmungskoordination
- Zustandsmaschine und Persönlichkeit
- Gedächtnisverwaltung auf den SSDs
- Reflexartige und einfache autonome Reaktionen
- Kommunikation mit dem PC und den Tentakeln
- Ausfallbehandlung und sichere Rückfallzustände

Der Pi darf bei Ausfall des Windows-PCs nicht seine grundlegende Handlungsfähigkeit verlieren.

**Hailo-10H = Sensorisches Verarbeitungszentrum**

Die Hailo-NPU ist ein wesentlicher Bestandteil des Pi-Nervensystems, kein isolierter Beschleuniger.

Sie verarbeitet unter anderem:

- Personen- und Objekterkennung
- SCRFD-Gesichtserkennung und ArcFace-Identifikation
- Körperpose, Handerkennung und weitere Vision-Modelle
- Weitere kompatible Audio-, Vision- und GenAI-Inferenz je nach tatsächlich laufender Pipeline

Die NPU muss nicht nur Erkennungsergebnisse liefern, sondern über einen Adapter auch Betriebszustände melden:

- Gerät erreichbar oder ausgefallen
- Modelle geladen oder nicht geladen
- Inferenz erfolgreich oder fehlgeschlagen
- Zeit seit letzter gültiger Inferenz
- Bildrate und Verarbeitungszeiten
- Warteschlangen oder Ressourcenengpässe
- Temperatur und weitere Hardwarewerte, soweit tatsächlich auslesbar

Ein vorhandenes `/dev/hailo0` allein beweist keine funktionsfähige Inferenz.

**Windows-PC mit Ollama = Großhirn / Cortex**

Der PC übernimmt komplexe Sprachverarbeitung, Planung, Recherche und gegebenenfalls zusätzliche Modelle.

Er ist ein leistungsfähiger Bestandteil des Gesamtsystems, aber nicht die einzige Instanz, die Molochs Existenz und Zustand bestimmt.

Wenn der PC ausfällt, bleibt der Pi aktiv und wechselt kontrolliert in einen lokalen Betriebsmodus.

**Reachy Mini Wireless = Körper und Ausdrucksorgan**

Reachy wird über die vorhandene beziehungsweise auszubauende Body-Bridge angebunden.

Sein eigener Compute Module 4 dient der lokalen Körpersteuerung. Er erhält keinen zweiten unabhängigen Persönlichkeitskern.

Molochs Persönlichkeit und innere Zustände werden weiterhin zentral berechnet.

Reachy erhält daraus Bewegungs- und Ausdrucksbefehle und liefert Zustandsmeldungen zurück.

---

### 3. DeepSeeks Nervensystem-Vertrag

Jeder steuerbare beziehungsweise überwachte Tentakel erhält einen gemeinsamen technischen Vertrag.

**A. Registrierung**

Jeder Tentakel besitzt eine eindeutige Kennung und wird in einem zentralen Laufzeitregister geführt.

Beispiele:

- `camera.sonoff.main`
- `camera.ptz.main`
- `audio.respeaker`
- `inference.hailo10h`
- `memory.primary`
- `llm.pc_ollama`
- `body.reachy`
- `music.spotify`
- `cooling.pi_fan`
- `power.pi_ups`

Bestehende Konfigurationsdateien dürfen erhalten bleiben, aber es muss eine eindeutige Laufzeitsicht auf die Komponenten geben.

**B. Bidirektionale Kommunikation**

Jeder steuerbare Tentakel hat einen logischen Kommunikationsvertrag mit beiden Richtungen:

- Abwärts: Befehl oder gewünschter Zustand
- Aufwärts: Empfangsbestätigung, Ergebnis, Fehler und tatsächlicher Zustand

Sensoren ohne Steuerfunktion liefern entsprechend Beobachtungen und Gesundheitsmeldungen.

Beide Richtungen müssen einem gemeinsamen Nachrichtenmodell folgen. Sie müssen nicht zwingend dieselbe physische Netzwerkverbindung verwenden.

**C. Heartbeat und Gesundheitsüberwachung**

Jeder Tentakel erhält eine definierte Überwachungsstrategie.

Dabei ist zu unterscheiden zwischen:

- Aktiver Heartbeat-Meldung
- Regelmäßigem Status-Polling
- Überwachung eines kontinuierlichen Datenstroms
- Passiver Funktionsüberwachung durch echte Tests

Heartbeat-Intervalle und Timeouts müssen je Komponente konfigurierbar sein.

Schweigen wird nach einem definierten Timeout als Fehlerereignis behandelt.

**D. Definierter Ausfallzustand**

Beispiele:

- Kamera: `OFFLINE`
- Mikrofon: `UNAVAILABLE`
- NPU: `DEGRADED`
- PC-LLM: `DISCONNECTED`
- Reachy: `UNREACHABLE`
- SSD-Gedächtnis: `READ_ONLY` oder `UNAVAILABLE`
- Lüfter: `FAULT_SUSPECTED` oder `FAULT_CONFIRMED`

Ein nicht erreichbarer Tentakel darf nicht weiterhin als funktionsfähig dargestellt werden.

**E. Protokollierung**

Jede wesentliche Kommunikation muss nachvollziehbar sein:

- Auftrag
- Empfänger
- Zeitstempel
- Empfangsbestätigung
- Ausführungsstatus
- Rückmeldung
- Dauer
- Fehlerursache
- Herkunft des Messwerts

**F. Eindeutige Auftragskennung**

Ergänzung von ChatGPT:

Jeder steuerbare Auftrag braucht eine eindeutige `command_id`.

Antworten müssen eindeutig dem ursprünglichen Befehl zugeordnet werden können.

Das verhindert falsche Erfolgsmeldungen bei verzögerten Antworten, Wiederholungen und Verbindungsabbrüchen.

Wichtig: `ACCEPTED` bedeutet lediglich, dass ein Befehl angenommen wurde. `EXECUTED` bedeutet, dass seine Ausführung gemeldet wurde. `VERIFIED` darf nur verwendet werden, wenn eine passende tatsächliche Rückmeldung oder unabhängige Beobachtung existiert.

---

### 4. Technische Nachrichtenstruktur

Claude Code soll prüfen, wie der vorhandene `core/moloch_event_bus.py`, die vorhandene Action-Bridge und die PC-Pi-Kommunikation verwendet werden können.

Keine zweite parallele EventBus-Architektur bauen.

Als gemeinsames logisches Schema wird vorgeschlagen:

```json
{
  "schema_version": 1,
  "event_id": "unique-event-id",
  "command_id": "optional-original-command-id",
  "timestamp_utc": "ISO-8601",
  "source": "body.reachy",
  "target": "core.nervous_system",
  "event_type": "body.motion_result",
  "status": "EXECUTED",
  "payload": {
    "motion": "lower_head",
    "result": "completed"
  },
  "latency_ms": 140,
  "evidence": "reachy_sdk_response"
}
```

Das ist ein Vertragsvorschlag, kein bereits existierendes API-Schema.

Bei Implementierung auf vorhandene Strukturen aufsetzen.

Zusätzlich benötigt jeder erfasste Zustand:

- Zeitpunkt der letzten gültigen Messung
- Messwert und Einheit
- Datenquelle
- Gültigkeitsdauer
- Qualitäts- beziehungsweise Vertrauensinformation

**Veraltete Daten dürfen niemals als aktueller Hardwarezustand ausgegeben werden.**

---

### 5. Die somatische Ausdrucksschicht

**Neue zentrale Anforderung aus dem Nutzerkonzept.**

Moloch soll seinen technischen Gesamtzustand körperlich darstellen können.

Dafür eine gemeinsame, regelbasierte Verbindung zwischen Hardware-Telemetrie, Core-Zustand und Reachy-Ausdruck vorsehen.

Beispiele:

| Tatsächliches Ereignis | Molochs Interpretation | Reachy-Ausdruck |
|---|---|---|
| Normaltemperatur, alle Systeme gesund | Entspannt | Aufrechter Kopf, ruhige Antennen |
| Temperatur deutlich erhöht | Technische Belastung | Langsam geneigter Kopf, weniger Bewegung |
| Lüfterdrehzahl fehlt bei hoher Temperatur | Kühlungsproblem | Antennen sinken, kurzer Seufzer |
| NPU-Inferenz ausgefallen | Wahrnehmung eingeschränkt | Suchende Bewegung, anschließend Ruhe |
| PC-Ollama nicht erreichbar | Lokaler Betriebsmodus | Kurzes Innehalten, anschließend normale Grundhaltung |
| Kamera offline | Visuelle Wahrnehmung fehlt | Kopf hält inne, kein vorgetäuschtes Tracking |
| Fehler beseitigt | Erholung | Kopf richtet sich auf, Antennen heben sich |

Die Bewegungen sollen nicht starr sein.

Bestehende Zustände wie `tension`, `dominance`, `mood`, `guardian`, `shadow`, `berserker` und die sechs Seins-Zustände sind einzubeziehen.

Wichtig ist die Unterscheidung zwischen:

1. **Technischer Zustand:** objektiv gemessene Hardwaredaten
2. **Innerer Zustand:** durch Molochs Core abgeleitete Interpretation
3. **Ausdruck:** Bewegungen, LED-Muster oder Sprachausgabe

Beispiel:

`CPU-Temperatur steigt` → `thermal_warning` → `somatic_state = strained` → `body.reachy` erhält entsprechende Bewegung.

Die Körperreaktion darf nicht davon abhängen, dass ein großes LLM einen Text erzeugt.

Moloch muss auf Hardwareprobleme auch dann reagieren können, wenn der PC ausgeschaltet ist.

Der PC darf zusätzliche passende Formulierungen erzeugen, ist aber für die Grundreaktion nicht notwendig.

**Sicherheitsregel:** Bewegungen sind bei Hardwarestress nachrangig gegenüber Kühlung, Lastreduktion und sicherem Systembetrieb.

---

### 6. Reachy ist selbst ein überwachter Tentakel

Die geplante Reachy-Bridge darf nicht nur Bewegungsbefehle senden.

Sie muss zusätzlich erfassen:

- Erreichbarkeit des Reachy-Daemons
- Verbindungszustand
- Empfang von Bewegungsbefehlen
- SDK-Ausführungsergebnis
- Letzte gültige Rückmeldung
- Tatsächliche Positionen, sofern das SDK sie auslesen kann
- Fehler oder Bewegungsbegrenzungen
- Sicheren Bewegungsstopp und Offline-Verhalten

Ein vom SDK bestätigter Befehl ist nicht automatisch ein Beweis für die physische Endposition.

Falls notwendig, kann später Molochs externe Sonoff-Kamera zur unabhängigen Beobachtung dienen.

Langfristiges Ziel ist die visuelle Eigenkörpererkennung:

Moloch erkennt Reachys Kopf, Körper und beide Antennen in der externen Kamera.

Er kann dann prüfen, ob eine gewünschte Bewegung tatsächlich sichtbar ausgeführt wurde.

Dafür ist perspektivisch ein eigenes Hailo-kompatibles Erkennungsmodell vorgesehen.

**Aktuell:** Reachy Mini Wireless ist bestellt. Die bestehende PC-Body-Bridge und der Simulator sollen zuerst genutzt werden. Keine echten Motorfunktionen voraussetzen, solange die Hardware nicht angeschlossen ist.

---

### 7. Raspberry Pi als eigenes überwachtes System

Nicht nur die angeschlossenen Geräte benötigen Gesundheitsmeldungen.

Auch der Raspberry selbst muss seinen Zustand erfassen.

Dazu gehören:

- CPU-Temperatur
- CPU- und RAM-Auslastung
- Lüfterstatus und gegebenenfalls echte RPM
- Versorgungsspannung und Unterspannung
- USV beziehungsweise Power-Manager
- SSD-Erreichbarkeit, Mountstatus und freier Speicher
- Dateisystemfehler
- Hailo-NPU-Zustand
- Netzwerk und Verbindungsqualität
- Prozess- und Servicezustände
- Funktion des EventBus und der zentralen Worker

Aus einer hohen Temperatur darf nicht automatisch ein Lüfterdefekt abgeleitet werden.

Nur tatsächlich vorhandene Sensoren dürfen als Messquellen verwendet werden.

Falls ein Systemwert nicht messbar ist, lautet der Zustand `UNKNOWN`, nicht `OK`.

Außerdem unterscheiden:

- Warnung
- Bestätigter Defekt
- Temporäre Überlastung
- Kritischer Fehler
- Erholung

Kritische Probleme müssen reale Schutzmaßnahmen auslösen, nicht nur lustige Bewegungen.

---

### 8. Gedächtnis und Situationsbewusstsein anbinden

Molochs Gedächtnis auf der zweiten SSD bleibt ein zentraler Bestandteil.

Die vorhandenen Module sollen als Grundlage dienen:

- `core/longterm_memory.py`
- `core/memory/persistent_memory.py`
- `core/memory/vector_memory.py`
- `core/memory/embedding_provider.py`
- `core/memory/episodic_memory.py`

Ziel ist nicht, sämtliche Heartbeats dauerhaft als Erinnerungen abzulegen.

Stattdessen:

- Laufende Telemetrie nur begrenzt speichern
- Relevante Zustandswechsel protokollieren
- Wiederkehrende Fehler erkennbar machen
- Bedeutende Ereignisse in das episodische Gedächtnis übernehmen
- Langzeittrends bei Bedarf analysieren

Beispiel:

Moloch hatte in den letzten Tagen wiederholt Temperaturprobleme.

Wenn ein neuer Temperaturanstieg auftritt, kann er dieses Muster erkennen.

Die Erinnerung muss aber eine historische Beobachtung bleiben und darf keinen aktuellen Messwert ersetzen.

**Moloch soll Vergangenheit und Gegenwart verbinden, ohne sie zu verwechseln.**

---

### 9. Vorhandene Codebasis ausdrücklich wiederverwenden

Bei der GitHub-Sichtung waren bereits folgende relevante Komponenten vorhanden:

**Nervensystem und Koordination**

- `core/moloch_event_bus.py`
- `core/action_bridge.py`
- `core/core_integrator.py`
- `core/moloch_service.py`
- `core/ipc_router.py`
- `core/capability_monitor.py`

**Hardware und Wahrnehmung**

- `core/hardware/hailo_manager.py`
- `core/perception/tappas_pipeline.py`
- `core/perception/model_health.py`
- `core/hardware/thermal_manager.py`
- `core/hardware/power_monitor.py`
- `core/camera_manager.py`

**Innere Zustände und Persönlichkeit**

- `core/personality/personality_engine.py`
- `core/personality/mood_engine.py`
- `core/personality/state_engine.py`
- `core/awareness/state_vector.py`
- `core/autonomy/homeostasis.py`
- `core/unconscious_engine.py`

**PC- und Körperanbindung**

- `core/bridge/chat_server.py`
- `core/autonomy/local_llm_bridge.py`
- `pc/body_bridge.py`
- `core/body/controller.py`
- `core/body/expression.py`
- `config/body_mappings.json`
- `pc/ki_zentrale.py`

**Audio und Musik**

- `core/audio/audio_pipeline.py`
- `core/audio/music_listener.py`
- `core/spotify_controller.py`

**Dokumentation**

- `CLAUDE.md`
- `docs/REACHY_MINI_KOERPER_KONZEPT.md`
- `docs/CROSS_SESSION_PROTOCOL.md`
- `docs/HAILO_10H_8GB_AUDIT.md`

Diese Pfade sind Orientierungspunkte aus der GitHub-Sichtung, keine Garantie für fehlerfreien Laufzeitbetrieb.

Wichtig: Der Entwicklungszweig enthält am 15. November bereits neue Commits für Reachy-Simulator, PTZ-Blickverfolgung, Anwesenheitsauslöser und spontane Sprachausgaben.

Diese Arbeiten müssen vor Neuentwicklungen berücksichtigt werden.

**Keine zweite Body-Bridge, keinen zweiten EventBus und kein zweites Zustandsmodell bauen, wenn vorhandene Bausteine erweitert werden können.**

---

### 10. Erste geschlossene Nervenschleife

DeepSeeks wichtigste Empfehlung:

**Nicht sofort alle Tentakel umbauen. Zuerst eine einzige durchgehend überprüfbare Schleife herstellen.**

Vorgeschlagener Ablauf:

1. Sonoff-Kamera oder ein anderer realer Sensor liefert ein Ereignis.
2. Hailo beziehungsweise die Wahrnehmungspipeline verarbeitet es.
3. Der Raspberry übernimmt das Ergebnis in den gemeinsamen aktuellen Zustand.
4. Der EventBus verteilt das Ereignis.
5. Ein relevantes Ereignis wird gegebenenfalls dem Gedächtnis zugeführt.
6. Der Core bewertet die Situation und trifft eine zulässige Entscheidung.
7. Der PC mit Ollama oder das lokale Sprachsystem formuliert eine Antwort.
8. Der Reachy-Simulator erhält den dazugehörigen Körperausdruck.
9. Die Body-Bridge meldet den Ausführungsstatus zurück.
10. Der Pi ordnet diese Rückmeldung dem ursprünglichen Auftrag zu.
11. Ein vollständiges Log dokumentiert die gesamte Kette.

Alle Schritte sollen durch dieselbe `trace_id` beziehungsweise korrelierte Auftragskennungen nachvollziehbar sein.

Der erste Test darf vollständig kontrolliert und teilweise simuliert erfolgen.

Danach sollen gezielte Fehlerfälle geprüft werden:

- PC nicht erreichbar
- Reachy-Daemon nicht erreichbar
- NPU liefert keine neuen Ergebnisse
- Kameraübertragung unterbrochen
- Rückmeldung verzögert
- SSD-Gedächtnis vorübergehend nicht verfügbar
- Veralteter Status im Cache

Moloch darf bei diesen Fehlern keine erfolgreiche Ausführung oder aktuelle Wahrnehmung behaupten.

---

### 11. Besonders wichtiger Test: Temperatur und körperliche Reaktion

Dieser Test ist dem Nutzer besonders wichtig.

**Testszenario:**

Eine glaubwürdige Temperaturwarnung wird zunächst simuliert.

Erwartete Reaktion:

1. Ein Hardwaremonitor veröffentlicht die Warnung.
2. Der EventBus übernimmt sie.
3. Der Core setzt einen technischen Belastungszustand.
4. Die somatische Ausdrucksschicht wählt eine ruhige Belastungsanimation.
5. Die Reachy-Bridge führt die Animation im Simulator aus.
6. Der Simulator beziehungsweise die Bridge meldet das Ergebnis.
7. Moloch protokolliert die gesamte Schleife.
8. Bei Erholung kehrt Reachy kontrolliert in die neutrale Haltung zurück.

**Späterer Realtest:** Nur mit echten Temperatur- und Lüftermesswerten. Keine absichtliche Gefährdung der Kühlung oder Stromversorgung.

Eine Reaktion auf einen realen Defekt muss außerdem zur Aussage passen.

Nicht sagen: „Mein Lüfter ist ausgefallen“, wenn nur die Temperatur erhöht ist.

Stattdessen etwa: „Mir wird gerade ziemlich warm. Ich prüfe meine Kühlung.“

**Wichtig:** Der künstliche Seufzer, die gesenkten Antennen und der geneigte Kopf sind Ausdruck. Sie ersetzen weder Diagnose noch technische Schutzmaßnahmen.

---

### 12. Grenzen und Sicherheitsregeln

- Pi bleibt die zentrale lokale Autorität für Grundfunktionen und Sicherheit.
- PC-Ausfall darf keine vollständige Systemlähmung verursachen.
- Reachy-Ausfall darf den Pi nicht blockieren.
- Keine unkontrollierten Motorbewegungen.
- Keine automatischen Firmware-Flashes.
- Keine gefährlichen Eingriffe an Stromversorgung oder Kühlung.
- Keine ungeprüften Schreib- oder Löschoperationen auf SSDs.
- Keine Zugangsdaten oder privaten Erinnerungen in Cloud-Prompts.
- Keine unkontrollierte Wiederholung fehlgeschlagener Befehle.
- Keine erfundenen Sensorwerte.
- Keine unbegrenzte Speicherung von Audio-, Video- oder Heartbeat-Daten.
- Lautstärke und Häufigkeit spontaner Reaktionen müssen begrenzt sein.
- Benutzerbefehle zum Schweigen haben Vorrang vor humorvollen Kommentaren. Kritische technische Schutzmaßnahmen bleiben davon unberührt.

Bei Ausfall der Verbindung zum Pi muss Reachy seinen eigenen sicheren lokalen Rückfallzustand besitzen.

Die vorhandenen Regeln in `CLAUDE.md`, Agentenanweisungen und Sicherheits-Hooks sind vor Änderungen zu beachten.

---

### 13. Arbeitsauftrag an Claude Code

**Phase 1 – Bestandsaufnahme**

Untersuche den aktuellen Entwicklungszweig.

Dokumentiere, welche der sechs Nervensystem-Anforderungen je Komponente bereits erfüllt sind.

Unterscheide streng:

- `[CODE]` Im Repository implementiert
- `[DOKU]` Lediglich dokumentiert
- `[NUTZER]` Vom Nutzer festgelegtes Ziel
- `[OFFEN]` Noch nicht bestätigt oder implementiert
- `[TEST]` Durch einen konkreten Test nachgewiesen

Prüfe insbesondere, ob schon mehrere Mechanismen für Health-Checks, Heartbeats und Statuskommunikation vorhanden sind.

**Phase 2 – Gemeinsamen Vertrag definieren**

Erarbeite einen minimalen technischen Vertrag für:

- Komponentenidentität
- Registrierung
- Befehle und Rückmeldungen
- Heartbeats beziehungsweise Funktionsüberwachung
- Zustandswechsel
- Zeitstempel und Datenalter
- Fehlerstatus
- Korrelation und Logging

Bestehende APIs möglichst kompatibel erweitern.

**Phase 3 – Somatische Ausdrucksschicht**

Prüfe, wie technische Gesundheitszustände auf vorhandene Core-Zustände und `config/body_mappings.json` abgebildet werden können.

Vermeide widersprüchliche Anweisungen aus Persönlichkeit, Musik, Anwesenheit und Hardwaremonitoring.

Lege Prioritäten fest.

Ein kritisches Temperaturereignis darf beispielsweise nicht durch eine fröhliche Musik-Animation übersteuert werden.

**Phase 4 – Einen End-to-End-Test implementieren**

Baue zunächst den Temperaturwarnungs-Test mit Reachy-Simulator und vollständiger Rückmeldung.

Anschließend mindestens einen echten Wahrnehmungs-zu-Körper-Test.

Automatisierte Tests, kontrollierte Simulation und reale Hardwaretests klar voneinander unterscheiden.

**Phase 5 – Ergebnis dokumentieren**

Erstelle einen knappen Abschlussbericht mit:

1. Tatsächlich vorhandener Architektur
2. Bereits erfüllten Anforderungen
3. Durchgeführten Änderungen
4. Nachgewiesenen Tests
5. Offenen Fehlern und Risiken
6. Sinnvoller Reihenfolge für die übrigen Tentakel

Keine Behauptung „vollständig integriert“, solange die geschlossene Schleife nicht nachgewiesen ist.

---

## 14. Langfristiges Ziel

M.O.L.O.C.H. soll kein gewöhnlicher Sprachassistent sein.

Er soll seine Umgebung wahrnehmen, Ereignisse verstehen, Erfahrungen speichern, den eigenen technischen Zustand kennen und diesen Zustand sinnvoll über Stimme, Bewegung und Verhalten ausdrücken.

Dabei ist entscheidend:

**Moloch besitzt nicht mehrere voneinander unabhängige Gehirne. Er hat ein verteiltes System aus spezialisierten Verarbeitungseinheiten, deren Zustände in einer gemeinsamen, überprüfbaren Architektur zusammenlaufen.**

Der Raspberry Pi mit Hailo-NPU bildet die lokale Nervenzentrale.

Der Windows-PC erweitert seine Denkfähigkeit.

Reachy Mini gibt diesem System einen sichtbaren Ausdruckskörper.

Die SSDs bewahren seine Erfahrungen.

Sensoren und Aktoren werden durch Rückmeldungen, Gesundheitsüberwachung und Ereignisse Teil desselben Systems.

### Abschließende Anweisung

**Claude Code: Prüfe zuerst die vorhandene Architektur und implementiere danach eine einzige echte, nachvollziehbare Nervenschleife einschließlich Reachy-Körperausdruck und Rückmeldung.**

Keine großflächige Neuprogrammierung ohne Bestandsaufnahme.

Das Ziel ist nicht möglichst viel neuer Code, sondern ein zuverlässig zusammenhängender M.O.L.O.C.H.

**Nicht nur Befehle senden. Wahrnehmen, was tatsächlich passiert.**
