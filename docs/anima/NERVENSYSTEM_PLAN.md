# ANIMA – Vom Tentakel zum Nervensystem: Plan

**Stand:** 2026-11-15 · **Auftrag:** `NERVENSYSTEM_BRIEFING_2026-11-15.md` · **Übersicht:** Issue #3
**Für:** die umsetzende Sitzung (Opus 5.5) auf Pi und PC

## Wozu dieses Dokument da ist

Es ist ein **Plan, kein Bauplan.** Es hält fest, was das Briefing will, was im Code dafür schon da ist,
welche Tatsachen der Architektur eine Lösung beachten muss und welche Entscheidungen offen sind.
Wo eine Lösung vorgeschlagen wird, steht „Vorschlag" davor – die umsetzende Sitzung soll sie prüfen,
verwerfen oder verbessern. Verbindlich sind nur die Abschnitte **Grundsätze**, **Abnahme** und **Nicht tun**.

**Herkunft der Aussagen.** Geschrieben in einer Cloud-Sitzung ohne Zugriff auf Pi, PC, NPU und Simulator.

| Marke | Bedeutung |
|---|---|
| `[CODE]` | Im Repository gelesen (Stand `deepseek_architecture_overhaul` @ `d39e03a` + PR #4). Sagt nichts über den Betrieb. |
| `[DOKU]` | Steht nur in Dokumenten oder Handoffs. |
| `[NUTZER]` | Von Markus festgelegt. |
| `[OFFEN]` | Nicht geklärt oder nicht vorhanden. |
| `[TEST]` | Durch einen Test belegt – in diesem Dokument kommt das **nirgends** vor. Alles hier ist ungeprüft am Gerät. |

---

## 1. Der Kern in vier Sätzen

Moloch schickt heute an vielen Stellen Befehle los und nimmt an, dass sie wirken.
Er soll stattdessen wissen, ob eine Komponente erreichbar ist, ob sie arbeitet, ob ein Auftrag ausgeführt wurde und wie alt sein Wissen darüber ist.
Dieses Wissen soll er zeigen können – über den Reachy-Körper, auch wenn der PC aus ist. `[NUTZER]`
Gebaut wird zuerst **eine** durchgehend nachweisbare Schleife (Temperatur → Körper → Rückmeldung), nicht ein Umbau aller Komponenten.

Dass das Problem echt ist, zeigen drei Befunde der letzten Tage `[CODE]`:
Agent-Tools für Kamera und LED sendeten monatelang Aktionen, die der Service nicht kannte;
der NPU-Selbsttest meldete Erfolg, ohne zu rechnen;
der Körper nickte nur bei einem von mehreren Sprechwegen.

---

## 2. Was es schon gibt

**Wichtigste Erkenntnis der Bestandsaufnahme:** Es existieren bereits mehrere Überwachungs- und Meldewege.
Die Aufgabe ist vor allem **zusammenführen und ehrlich machen**, nicht neu bauen.

| Baustein | Datei | Was er tut `[CODE]` | Was ihm zum Nervensystem fehlt |
|---|---|---|---|
| **System-Watchdog** (nennt sich selbst „Nervensystem") | `core/system_watchdog.py` | Prüft alle 3 s: Frame-Stillstand, Pipeline, ONVIF, CPU-Temperatur (70/80 °C), RAM, Platten, Mikrofon, Heimnetz, hailo-ollama. Meldet „Schmerz" an den CoreIntegrator (`hardware_pain`, `system_stress`) → wirkt auf Tension und Zone. | Kein Zeitstempel/Alter je Wert, kein `UNKNOWN`, keine Sicht pro Komponente, kein Weg zum Körper |
| **Thermal-Manager** | `core/hardware/thermal_manager.py` | Zustände `stable / warming / warning / throttling / critical` mit Glättung und 5 °C Hysterese, Lüfterstufe lesen/schreiben, Ereignisse, eigene Sprachausgabe | Unklar, ob er im Service überhaupt gestartet wird: gefunden wurde nur die Nutzung für Lüfter-PWM `[OFFEN]`. Temperatur wird an **drei** Stellen gelesen (hier, Watchdog, CoreIntegrator) |
| **Power-Monitor** | `core/hardware/power_monitor.py` | PiPower5 über I²C: Netz/Akku, Akkustand, Verbrauch, alle 10 s; wird im Service gestartet, meldet an den CoreIntegrator | Fehlt die Bibliothek, ist er still deaktiviert – das ist `UNKNOWN`, wird aber nicht so ausgewiesen |
| **Modell-Gesundheit** | `core/perception/model_health.py` | Pro Modell: FPS, Latenz, Fehler- und Drop-Rate, „hängt"-Erkennung | Deckt Briefing-Punkt „NPU meldet Betriebszustände" weitgehend ab; nicht als Komponente `inference.hailo10h` sichtbar |
| **Worker-Gesundheit** | Status-JSON `worker_health` | Je Worker: läuft, Inferenzen, Fehler, letzte Dauer | Wie oben |
| **Capability-Monitor** | `core/capability_monitor.py` | Scan beim Start: Module, Modelle, Hardware; veröffentlicht `capability.changed` | Nur beim Start, keine laufende Sicht |
| **Audit-System** | `core/audit/` (20 Auditoren) | Aggregiert Schichten nach `/dev/shm/audit_state.json` (`green / warn / red`, Alarmstufe) | Prüfbericht, kein Laufzeitzustand; langsamer Takt |
| **EventBus** | `core/moloch_event_bus.py` | Ereignis = `priority, timestamp, event_type, source, payload`; Dedup; schreibt jedes Ereignis mit Sequenznummer nach `/dev/shm/event_bus.jsonl` und tageweise auf die SSD | Nur innerhalb eines Prozesses zugestellt; keine `command_id`, keine Gültigkeit |
| **Situationszustand** | `GET /live` im Chat-Server | Liefert bereits: `watchdog` (Temperatur, RAM, Kamera erreichbar, throttled, Warnungen), `power`, `worker_health`, `core`, `frame_age`, `ptz`, `speech` | Faktisch die „Nervenbahn" zum PC. Kein Alter je Feld, kein Komponentenstatus |
| **Rückkanal-Muster** | `core/agent/tools/vision.py` + `npu_extras.run_request` | Auftrag mit `request_id` als IPC-Datei, Ergebnis als `/dev/shm/moloch_npu_result_<id>.json`, mit Timeout | Genau das verlangte Korrelationsmuster – existiert nur für OCR |
| **Körper** | `core/body/` + `pc/body_bridge.py` | Controller mit Not-Aus, Prioritäten, TTL, Grenzen; Bridge liest `/live`, schreibt jede Sekunde `body_status.json` (Quelle, Zustand, Pose, Verbindung, Ergebnis) | Status bleibt auf dem PC, geht nicht zum Pi zurück. Technischer Zustand fließt nicht in die Haltung |
| **Sprechstatus** | `POST /speech/state`, `/dev/shm/moloch_speech.json` | Ein Zustand für mehrere Sprechwege (PR #4) | Vorbild für „eine Stelle, viele Melder" |
| **Hardware-Inventar** | `scripts/anima_inventory.py` (PR #4) | Liest einmalig, was wirklich da ist; `unbekannt` statt Schätzung | Skript, kein Dienst – aber die Lesefunktionen sind wiederverwendbar |

### Abgleich mit dem Vertrag des Briefings (A–F)

| Komponente | A Kennung | B Hin + Rück | C Überwachung | D Ausfallzustand | E Protokoll | F Auftrags-ID |
|---|---|---|---|---|---|---|
| Kamera-Stream | `[OFFEN]` | – (Sensor) | `[CODE]` Frame-Alter, Watchdog | `[CODE]` teilweise (`camera_reachable`) | `[CODE]` EventBus | – |
| Kamera-PTZ | `[OFFEN]` | `[CODE]` hin; Rück nur als Position im Status | `[CODE]` ONVIF-Fehlerzähler | `[OFFEN]` | `[CODE]` Log | `[OFFEN]` |
| Mikrofon | `[OFFEN]` | – | `[CODE]` TCP-Prüfung | `[OFFEN]` | `[CODE]` | – |
| NPU / Modelle | `[OFFEN]` | – | `[CODE]` Modell- und Worker-Gesundheit | `[CODE]` „hängt" | `[CODE]` | `[CODE]` nur OCR |
| Gedächtnis | `[OFFEN]` | – | `[CODE]` Auditor | `[OFFEN]` (`READ_ONLY` gibt es nicht) | `[CODE]` | – |
| PC-Sprachmodell | `[OFFEN]` | `[CODE]` Anfrage/Antwort | `[CODE]` Probe, Backoff | `[CODE]` Fallback-Kette | `[CODE]` Provider im Ergebnis | `[OFFEN]` |
| Körper (Reachy) | `[OFFEN]` | `[CODE]` hin; Rück bleibt auf dem PC | `[CODE]` Verbindung, Reconnect | `[CODE]` `offline`, Not-Aus | `[CODE]` `body_status.json` | `[OFFEN]` |
| Spotify | `[OFFEN]` | `[CODE]` hin | `[CODE]` Auditor | `[OFFEN]` | `[CODE]` | `[OFFEN]` |
| Lüfter | `[OFFEN]` | `[CODE]` Stufe schreiben/lesen | `[OFFEN]` Drehzahl unbekannt | `[OFFEN]` | `[CODE]` thermal.log | `[OFFEN]` |
| Strom / USV | `[OFFEN]` | – | `[CODE]` 10 s Polling | `[OFFEN]` | `[CODE]` | – |

Lesart: Überwachung (C) ist breit vorhanden. Es fehlen durchgehend eine **einheitliche Kennung und Sicht (A)**,
ein **ehrlicher Ausfallzustand samt `UNKNOWN` (D)** und die **Zuordnung von Rückmeldungen zu Aufträgen (F)**.

---

## 3. Tatsachen, an denen jede Lösung vorbei muss

1. **4 GB RAM auf dem Pi.** `[NUTZER]` Kein zusätzlicher Dienst, kein Broker, keine neue schwere Abhängigkeit.
2. **Mehrere Prozesse.** `[CODE]` Hauptservice (`moloch.service`) und Chat-Server (`moloch-chat.service`) sind getrennt.
   Der EventBus stellt nur im eigenen Prozess zu. Zwischen den Prozessen laufen Befehlsdateien `/tmp/moloch_cmd_*.json`
   (alle 200 ms gelesen) und Zustandsdateien in `/dev/shm/`. Der PC hängt per HTTP am Chat-Server.
   „Der EventBus verteilt das Ereignis" gilt deshalb nicht systemweit.
3. **Die Body-Bridge läuft auf dem PC.** `[CODE]` Ohne PC gibt es heute keinen Körper. Die Forderung
   „Körperreaktion auch bei ausgeschaltetem PC" `[NUTZER]` ist mit dem Simulator nicht erfüllbar und verlangt für
   den echten Reachy eine Entscheidung (Abschnitt 6, E1).
4. **Der Service pollt mit 5 Hz, der Watchdog alle 3 s.** `[CODE]` `POLL_HZ` nicht erhöhen (CLAUDE.md).
5. **Der EventBus schreibt etwa 1 GB Telemetrie pro Tag.** `[DOKU]` Zusätzliche Protokollierung braucht eine Obergrenze;
   nie nach `/run` oder tmpfs ohne Begrenzung (Login-Loop vom 11.05.).
6. **ROT-Dateien** `moloch_service.py`, `core_integrator.py`, `ipc_router.py`, `voice_pipeline.py`, `hailo_manager.py`:
   einzeln, mit Backup-Tag, je höchstens 50 Zeilen. Die Hailo-Pipeline bleibt unangetastet.
7. **Der echte Reachy ist bestellt, nicht da.** `[NUTZER]` Alles Körperliche läuft vorerst gegen den Simulator.
   Grenzwerte in `config/body_mappings.json` sind Annahmen.
8. **Lüfterdrehzahl:** Der Thermal-Manager liest eine Lüfter*stufe* (`cooling_device0/cur_state`). Ob ein Tacho-Signal
   vorhanden ist, ist `[OFFEN]` – `scripts/anima_inventory.py` sucht danach. Ohne Tacho ist „Lüfter defekt" nicht feststellbar.

---

## 4. Grundsätze (verbindlich)

1. **`UNKNOWN` ist nicht `OK`.** Nicht messbar wird als unbekannt geführt und ausgesprochen.
2. **Jeder Zustand hat ein Alter.** Wer einen Wert liefert, liefert Zeitpunkt und Gültigkeit mit.
   Abgelaufene Werte gelten als unbekannt, nicht als letzter Stand.
3. **Drei Stufen, nie gleichsetzen:** `ACCEPTED` (angenommen) · `EXECUTED` (Ausführung gemeldet) ·
   `VERIFIED` (durch tatsächliche Rückmeldung oder unabhängige Beobachtung belegt).
4. **Messung → Deutung → Ausdruck** bleiben getrennte Schritte. „Temperatur 78 °C" ist eine Messung,
   „belastet" eine Deutung, „Antennen sinken" ein Ausdruck. Kein Schritt darf den vorigen ersetzen.
5. **Aussage passt zur Messung.** Erhöhte Temperatur heißt nicht „Lüfter ausgefallen".
6. **Ausdruck ersetzt keinen Schutz.** Bei Hardwarestress gehen Kühlung, Lastreduktion und sicherer Betrieb vor Bewegung.
7. **Kein zweiter EventBus, keine zweite Body-Bridge, kein zweites Zustandsmodell, kein zweiter Kamera-Controller.**
8. **Der Pi bleibt allein handlungsfähig.** PC-Ausfall und Reachy-Ausfall dürfen ihn nicht blockieren.
9. **Schweigen ist ein Befund.** Bleibt eine Meldung länger als ihr Zeitlimit aus, ist das ein Zustandswechsel, kein Nichts.

---

## 5. Die erste Schleife: Temperatur → Körper → Rückmeldung

Das ist der Arbeitsauftrag. Alles Weitere erst, wenn diese Schleife samt Fehlerfällen nachgewiesen ist.

### Ziel in einem Satz

Eine (zunächst simulierte) Temperaturwarnung führt nachvollziehbar zu einer ruhigen Belastungshaltung im
Reachy-Simulator, die Ausführung kommt beim Pi an und wird dem Auslöser zugeordnet; bei Erholung kehrt der
Körper in die Grundhaltung zurück – alles unter einer gemeinsamen Kennung im Log.

### Die Kette und was je Glied vermutlich nötig ist

| # | Glied | Vorhanden `[CODE]` | Lücke | Vorschlag |
|---|---|---|---|---|
| 1 | Messung | Watchdog liest CPU-Temperatur alle 3 s; Thermal-Manager hat Zustände mit Hysterese | Drei Leser, unklar welcher gilt | Einen zur Quelle erklären. Der Thermal-Manager hat die bessere Logik (Glättung, Hysterese); der Watchdog läuft sicher. Entscheidung E2 |
| 2 | Deutung | CoreIntegrator bekommt `system_stress` → Tension | Tension mischt Technik und Stimmung; der Körper kann „heiß" nicht von „gereizt" unterscheiden | Eigener, kleiner **somatischer Zustand** neben Tension: z. B. `ok / strained / distressed / recovering / unknown`, mit Ursache, Zeitstempel, Gültigkeit |
| 3 | Bereitstellung | Status-JSON → `/live` | Feld fehlt | Neues Feld in Status-JSON und `/live` (z. B. `soma`). Bestehende Felder unverändert lassen |
| 4 | Ausdruck | `compute_pose()` mischt Zustand, Zone, Mood, Tension, Dominance, Blick | Kein Kanal für Technik | Somatischen Zustand als weiteren Kanal in `core/body/expression.py` + `config/body_mappings.json` (Antennen tiefer, Kopf geneigt, weniger Amplitude). Reine Funktion, testbar ohne Hardware |
| 5 | Vorrang | Controller kennt Not-Aus > manuell > Dialog > Grundausdruck | Technik hat keinen Rang; Musik/Anwesenheit könnten übersteuern | Rangfolge festlegen (Entscheidung E3). Belastung dämpft eher, als dass sie eine eigene Bewegung erzwingt |
| 6 | Ausführung | Body-Bridge → Simulator | – | unverändert |
| 7 | Rückmeldung | Bridge schreibt `body_status.json` lokal auf dem PC | Kommt nicht beim Pi an | Bridge meldet an den Pi (Muster wie `POST /speech/state`): welcher somatische Zustand ausgedrückt wurde, Ergebnis des Controllers, Zeitpunkt |
| 8 | Zuordnung | – | Keine gemeinsame Kennung | Der Pi vergibt beim Zustandswechsel eine Kennung; sie wandert über `/live` zur Bridge und in deren Rückmeldung zurück. Stufen: `ACCEPTED` (Bridge hat den Zustand gelesen), `EXECUTED` (Controller meldet `sent`). `VERIFIED` gibt es im Simulator nur, wenn dessen gemeldete Pose zurückgelesen wird – sonst ehrlich bei `EXECUTED` bleiben |
| 9 | Protokoll | EventBus + JSONL | – | Jeden Schritt als Ereignis mit derselben Kennung; nur Zustands**wechsel**, keine Dauermeldungen |
| 10 | Erholung | Hysterese im Thermal-Manager | – | Rückweg genauso durchlaufen und protokollieren |

### Auslöser für den Test

Keine echte Überhitzung herbeiführen. `[NUTZER]` Vorschlag: eine Test-Vorgabe mit Ablaufzeit, wie sie
`body_override.json` und `/dev/shm/moloch_test_face_attr_override.json` schon vormachen – eine Datei mit
vorgegebener Temperatur und `until`, die nur der Messschritt liest. Im Log muss erkennbar sein, dass der Wert
simuliert war (`quelle: simulation`); ein simulierter Wert darf keine echten Schutzmaßnahmen auslösen, die
den Betrieb stören (LLM stoppen, Pipeline drosseln). Alternative: Auslösung über einen vorhandenen Testweg
(`/api/test/run`, `/sim_run/…`). Bitte prüfen, was es dort schon gibt, bevor etwas Neues entsteht.

### Abnahme (verbindlich)

**Stufe 1 – automatisch, ohne Hardware** (`pytest`, läuft in der CI):
- Zustandsfolge `ok → strained → recovering → ok` aus einer Temperaturfolge, mit Hysterese (kein Flattern an der Schwelle).
- Abgelaufener Messwert → `unknown`, nicht letzter Stand.
- `compute_pose()` zeigt im Belastungszustand tiefere Antennen und geringere Amplitude; bleibt in den Grenzen.
- Rangfolge: Belastung wird nicht von Musik-/Anwesenheits-Ausdruck übersteuert; Not-Aus und manuell schlagen alles.
- Rückmeldung mit falscher oder alter Kennung wird nicht als Erfolg gewertet.

**Stufe 2 – kontrollierte Simulation** (Pi + PC + Simulator, Temperatur vorgegeben):
- Im Simulator sichtbar: Haltung sinkt, nach Erholung richtet sie sich auf.
- Auf dem Pi ein zusammenhängendes Protokoll mit einer Kennung: Messung → Deutung → gelesen → ausgeführt → Erholung.
- Fehlerfälle, jeweils mit erwartetem Verhalten:

  | Fehlerfall | Erwartet |
  |---|---|
  | PC aus | Pi führt den somatischen Zustand weiter; Körper `UNREACHABLE`; keine Erfolgsmeldung |
  | Simulator/Daemon aus | Bridge meldet `offline`; Pi wertet den Ausdruck als nicht ausgeführt |
  | Rückmeldung kommt nicht | Nach Zeitlimit Zustand „unbestätigt", kein stilles Weiterlaufen |
  | Rückmeldung kommt verspätet | Wird dem alten Auftrag zugeordnet oder verworfen, nie dem neuen gutgeschrieben |
  | Status-JSON veraltet | `/live` weist den Wert als veraltet/unbekannt aus |
  | Chat-Server neu gestartet | Schleife fängt sich ohne Eingriff |

Werkzeug: `python3 scripts/soma_probe.py --simuliere 78` (wirkt nur auf Soma, Watchdog/Lüfter sehen die echte Temperatur).

**Stufe 3 – echte Messung:** Erst sinnvoll, wenn Stufe 2 steht. Sprachliche Reaktion nach Grundsatz 5.

- **3a – Kühlung kontrolliert reduziert** (Idee Markus, 16.11.): Gehäuselüfter (Noctua) aus, Pi-Lüfter regelt allein,
  normale Last, **keine** Stresslast. Parallel `python3 scripts/soma_probe.py` **ohne** `--simuliere` laufen lassen.
  - Geprüft wird: Regelt der Pi-Lüfter sauber? Kommt der echte Messwert frisch in `/live.soma` an
    (`quelle=watchdog.cpu_temp`, Alter < 15 s)?
  - Erwartet: Der Pi-Kühler (`cooling_device0`) springt an, er lief mit Noctua laut Markus noch nie. Dann zeigt
    `/live.soma.kuehlung` `verdacht_noctua: true`, im Log steht `[SOMA] Kuehlung: ...`. Der Zustand bleibt `ok`.
    Ein `strained` ist kein Testfehler, sondern ein Befund: Der Pi-Kühler allein reicht nicht.
  - Ehrlichkeit: Moloch kennt nur das Symptom (Pi-Kühler läuft), nicht die Ursache. Die Noctua-Drehzahl wird
    nicht gemessen. Er darf „Verdacht auf den Noctua“ sagen, nicht „der Noctua ist kaputt“. Mit dem Tacho-Kabel
    des NF-A4x20 an einem GPIO wäre es ein Beweis statt eines Verdachts (später, Hardware).
  - Vorher ohne Risiko simulieren: `soma_probe.py --simuliere 55 --pi-kuehler 2`.
  - **Abbruch:** bei 75 °C oder nach 30 min Noctua wieder an. Ab 80 °C stoppt der Watchdog das LLM und drosselt,
    ab etwa 85 °C drosselt der Pi selbst. Das wäre dann kein Test mehr, sondern eine Störung.
- **3b – Normalbetrieb:** reale Werte ohne Eingriff, über Tage. Prüft Hysterese und Erholung unter echten Bedingungen.

Reihenfolge: zuerst Stufe 2 (Simulation, ohne Risiko, zeigt, ob die Kette steht), danach 3a. So bleibt der Pi kühl,
und die Logik ist trotzdem vollständig geprüft.

Im Abschlussbericht je Punkt angeben, auf welcher Stufe er belegt ist. Nicht „vollständig integriert" schreiben,
solange Stufe 2 nicht durchgelaufen ist.

---

## 6. Offene Entscheidungen

Mit Empfehlung, aber ausdrücklich offen. Wo Markus entscheiden muss, steht es dabei.

**E1 – Wo läuft die Körpersteuerung, wenn der echte Reachy da ist?** *(Markus)*
Heute PC. Für „reagiert auch ohne PC" muss sie auf den Pi oder auf Reachys eigenen Rechner.
`core/body/` ist dafür bereits ohne PC-Abhängigkeit gebaut; die Bridge ist eine dünne Schicht.
Empfehlung: Für die erste Schleife nichts verschieben. Die Schnittstelle so wählen, dass ein Umzug nur
die Bridge betrifft. Reachys sicherer Rückfallzustand bei Verbindungsverlust gehört auf Reachy selbst.

**E2 – Welche Stelle ist die Quelle für die Temperatur?**
Erst klären, ob der Thermal-Manager im Betrieb läuft (`journalctl`, `logs/thermal.log`). Danach eine Quelle
festlegen und die anderen Leser darauf verweisen oder dokumentiert stehen lassen. Nicht drei Schwellensätze pflegen.

**E3 – Rang des technischen Ausdrucks im Körper.**
Vorschlag: Not-Aus > direkte Steuerung > **kritischer technischer Zustand** > Wahrnehmungsreaktion > Gespräch >
**technische Belastung als Dämpfung aller unteren Ebenen** > Stimmung > Idle > Show.
Belastung wirkt also eher als Regler (weniger Amplitude, tiefere Grundhaltung) denn als eigene Animation –
das vermeidet Widersprüche zwischen Persönlichkeit, Musik, Anwesenheit und Technik.

**E4 – Wie sieht das Register aus?**
Vorschlag: keine neue Datenbank, sondern eine **Sicht**: eine Funktion, die aus Watchdog, Modell-Gesundheit,
Power-Monitor, Body-Rückmeldung und Probes je Komponente einen Eintrag baut
(`id`, `zustand`, `seit`, `letzte_gültige_meldung`, `gültig_bis`, `quelle`, `ursache`).
Kennungen wie im Briefing (`camera.sonoff.main`, `inference.hailo10h`, `body.reachy`, `llm.pc_ollama`, …).
Ausgabe als ein Feld in `/live` oder eigener Endpunkt. Zeitlimits je Komponente in einer Konfigurationsdatei
unter `config/` (nicht `settings.json`).
Zu prüfen: Lässt sich das auf dem Capability-Monitor oder dem Audit-Zustand aufsetzen, statt eine dritte Sicht zu schaffen?

**E5 – Wie viel Schema?**
Das Briefing schlägt elf Felder vor. Das vorhandene Ereignis hat `priority, timestamp, event_type, source, payload`.
Vorschlag für die erste Schleife: in `payload` nur ergänzen, was gebraucht wird – Kennung der Kette, Status,
Gültigkeit, Herkunft des Messwerts. Kein Umbau bestehender Ereignisse.

**E6 – Korrelation über Prozessgrenzen.**
Vorhandenes Muster: `request_id` + Ergebnisdatei in `/dev/shm/` (OCR). Für Pi ↔ PC: Kennung im HTTP-Körper.
Vorschlag: dieses Muster verallgemeinern, statt den EventBus prozessübergreifend zu machen.

**E7 – Namen.**
„Tentakel" steht in Konfiguration und Code (`tentacle_llm`, Profil `tentacle`, Agent `tentacle`, `/live.tentakel_enabled`).
Empfehlung: nicht umbenennen. „Nervensystem" ist das Konzept, die Komponenten bleiben Tentakel. Eine Umbenennung
kostet viel und ändert am Verhalten nichts. *(Markus, falls er es anders will)*

**E8 – Was wandert ins Gedächtnis?**
Nur Zustandswechsel und Häufungen, keine Heartbeats. Vorschlag: einfache Zählung je Ursache und Tag in einer
kleinen Datei; bei Wiederholung ein Fakt über den vorhandenen Weg (`add_fact`), als historische Beobachtung
formuliert („hatte am … dreimal Temperaturwarnung"). Episodisches Gedächtnis ist derzeit deaktiviert `[CODE]`.

---

## 7. Reihenfolge nach der ersten Schleife

Erst wenn Abschnitt 5 auf Stufe 2 abgenommen ist. Jede Komponente einzeln, mit denselben drei Abnahmestufen.

1. **Körper als überwachte Komponente** – Rückmeldung der Bridge wird regulärer Zustand `body.reachy`
   (erreichbar, letzter Befehl, Fehler). Fällt großteils aus der ersten Schleife ab.
2. **PC-Sprachmodell** – `llm.pc_ollama` mit `DISCONNECTED`; der Wechsel in den lokalen Betrieb wird Zustand und Ausdruck.
3. **NPU** – `inference.hailo10h` aus Modell- und Worker-Gesundheit: Zeit seit letzter gültiger Inferenz statt Geräteknoten.
4. **Kamera** – Stream-Alter und PTZ: gefahrene gegen gemeldete Position (`ACCEPTED` → `EXECUTED` → `VERIFIED` über die Positionsrückmeldung).
   Stand 16.11.: Keine neue Messung bauen. Die Kamera wird schon doppelt geprüft: vom Watchdog (`camera_reachable`,
   steht in `/live`) und vom Hardware-Auditor (eigener Ping und RTSP). Soma liest nur `watchdog.camera_reachable`,
   die beiden Pings später zusammenlegen. **Zuerst:** Die Zeile „Kamera nicht verfügbar seit …“ in den Prompt
   beider Antwortwege. Heute erfährt das Sprachmodell nicht, dass es blind ist; die Ehrlichkeitsprüfung simuliert
   diese Zeile nur (E06). Erst danach der Körperausdruck. Eine Ersatzkamera (Reachy) gibt es noch nicht. Am Pi
   bräuchte sie NPU-Slots (8/8 belegt), also würde sie auf dem PC oder einem Jetson ausgewertet.
5. **Wahrnehmung → Körper** – der im Briefing verlangte zweite End-to-End-Test: echte Erkennung führt zu Ausdruck mit Rückmeldung.
6. **Strom, Lüfter, SSD, Mikrofon, Spotify** – je nachdem, was das Inventar als messbar ausweist.
7. **Verlauf und Gedächtnis** (E8).
8. **Visuelle Selbstbeobachtung** – erst mit echtem Reachy und freiem NPU-Platz (Issue #3, P13).

Überschneidung mit Issue #3: Dieser Plan ersetzt **P8** (Situationszustand), enthält **P6** (Hardware-Fragen
bekommen ihre Werte aus derselben Sicht) und liefert die Grundlage für **P13**. **P4** (Sprechstatus) ist ein
bereits gebautes Beispiel für das Meldemuster.

---

## 8. Arbeitsteilung

| Seite | Aufgaben der ersten Schleife |
|---|---|
| **Pi** | Messquelle klären (E2); somatischen Zustand ableiten; Feld in Status-JSON und `/live`; Endpunkt für die Körper-Rückmeldung; Zuordnung und Protokoll; Test-Vorgabe; Tests Stufe 1 für die Zustandslogik |
| **PC** | Body-Bridge liest den somatischen Zustand und meldet zurück; Karte in der KI-Zentrale zeigt Zustand und Bestätigungsstufe; Simulator-Abnahme Stufe 2 |
| **Beide** | Kanal für den somatischen Zustand in `core/body/expression.py` und `config/body_mappings.json` (Agent `body`); Rangfolge (E3) |

Ablauf nach dem Cowork-Protokoll in `CLAUDE.md`: Lead postet `task_cowork_nervensystem_schleife1`, die andere
Seite bestätigt, dann wird gearbeitet. Der Pi-Teil muss stehen, bevor der PC-Teil abgenommen werden kann.

Vor dem Start: PR #4 mergen und das Laufzeit-Inventar ziehen (`docs/anima/AUFTRAG_PI.md`, Schritt 1–2).
Mehrere offene Punkte dieses Plans (Thermal-Manager aktiv? Tacho vorhanden? PiPower5 lesbar?) beantwortet das Inventar.

---

## 9. Nicht tun

- Keinen Nachrichten-Broker, keinen zusätzlichen Dienst, keine neue schwere Bibliothek auf dem Pi.
- Den EventBus nicht umbauen und nicht prozessübergreifend machen.
- Keine Heartbeat-Threads pro Komponente – vorhandene Takte nutzen (Watchdog 3 s, Status 5 Hz, Power 10 s).
- `POLL_HZ` nicht erhöhen. Hailo-Pipeline, PTZ-Vorzeichen, ArcFace-Schwelle nicht anfassen.
- Keine Schutzmaßnahme durch simulierte Werte auslösen. Keine echte Überhitzung, kein Eingriff in Kühlung oder Strom.
- `hailo_manager._force_release_device()` nicht aus Überwachung oder Diagnose heraus aufrufen.
- Keine automatischen Wiederholungen fehlgeschlagener Befehle ohne Obergrenze.
- Keine Dauerprotokollierung von Messwerten; nur Wechsel.
- Keine Umbenennung „Tentakel" → „Nerv" im Code (E7).
- Nicht alle Komponenten gleichzeitig umbauen.
- Nichts als funktionierend melden, das nur als Code existiert.

---

## 10. Fragen an Markus

1. **E1:** Soll die Körpersteuerung mit dem echten Reachy auf den Pi oder auf Reachys eigenen Rechner?
2. **E7:** Bleibt es beim Wort „Tentakel" im Code?
3. Soll Moloch bei technischer Belastung **sprechen** („Mir wird warm"), oder zunächst nur die Haltung ändern?
   Der Thermal-Manager hat bereits eine Sprachausgabe; wie oft sie höchstens kommen darf, ist festzulegen.
4. Woher kommt der „Seufzer" – Sprachausgabe des Pi, Lautsprecher des Reachy oder PC?
5. Ist am Lüfter ein Tacho-Signal angeschlossen? (Beantwortet sonst das Inventar.)
