# ANIMA – Bestandsaufnahme und Arbeitsplan

**Stand:** 2026-11-15 · **Basis:** `deepseek_architecture_overhaul` @ `d39e03a` + Branch `ccr-5df7b577-w92x94`
**Auftrag:** `docs/anima/MASTER_BRIEFING_2026-11-15.md` · **Nächste Schritte:** `AUFTRAG_PI.md`, `AUFTRAG_PC.md` · **Hardware:** Raspberry Pi 5, 4 GB RAM (von Markus bestätigt)

## Wie dieser Bericht zu lesen ist

Erstellt in einer Cloud-Sitzung **ohne Zugriff auf Pi, PC, NPU, Kamera oder Mikrofon**.
Deshalb gibt es drei Stufen, die nicht vermischt werden:

| Stufe | Bedeutung |
|---|---|
| **Code** | Im Repository vorhanden und gelesen. Sagt nichts darüber, ob es läuft. |
| **Test** | Durch automatische Tests ohne Hardware belegt (`python3 -m pytest tests pc`: 361 bestanden, 19 übersprungen). |
| **Gerät** | Am laufenden System nachgewiesen. **In dieser Sitzung für keinen Punkt möglich.** Wo ein Handoff einen Nachweis nennt, steht die Quelle dabei. |

Phase B des Briefings (Laufzeit prüfen) ist damit **offen** und steht als Prüfliste in Abschnitt 4.

---

## 1. Übersicht der zwölf Bereiche

| # | Bereich | Vorhanden (Code) | Am Gerät belegt | Befund | In dieser Sitzung | Offen |
|---|---|---|---|---|---|---|
| 1 | Gedächtnis ↔ Ollama | ja | Speichern ja (Handoff 15.11.: `/memory/turn`, `/history`); Abruf nein | **Lücke bestätigt:** lokale Modelle bekamen nur 3 Zeilen Crew, keine Fakten (Cloud-Pfad bekam alle) | Wort-Recall, Selbstwissen, `GET /memory/recall`, Anbindung KI-Zentrale | Abnahmetest am Gerät |
| 2 | Sprachmodelle, Antwortqualität | ja | Kaskade `local_first` getestet (Handoff 15.11.) | Modellname in Konfiguration widersprüchlich; Selbstwissen fehlte | Selbstwissen (22 Einträge) | Vergleichstests, Identitätsdateien zusammenführen |
| 3 | Chat ↔ Kamera | teilweise | unbekannt | **Zwei Lücken bestätigt:** lokaler PC-Chat führte keine Befehle aus; „schau **mal** zur Tür" wurde nirgends erkannt | `POST /command`, Vorabfrage in der Zentrale, Füllwörter in der Erkennung | Abnahme am Gerät; Bildbeschreibung |
| 4 | BBox, Raumkalibrierung | minimal | nein | Entfernung = 3 Stufen aus BBox-Fläche, kein Kalibrierprofil | – | Konzept + Messreihe |
| 5 | ReSpeaker, Dauer-Audio | Sprache ja, Geräusche nein | Sprache teilweise | Keine Geräuschereignisse im EventBus | – | VAD/Ereignis-Pipeline |
| 6 | Spotify-DJ | ja | unbekannt (03.10.: Refresh-Token widerrufen) | Auto-DJ und Zonen-Empfehlung vorhanden, kein Lernen aus Ablehnung | – | Token prüfen, dann DJ-Logik |
| 7 | Internetrecherche | ja | ja (PC-Tests, Handoffs) | **Bestätigt:** 3 Treffer × 120 Zeichen Ausschnitt | – | Mehrstufige Recherche |
| 8 | Akustische Bibliothek | nein | nein | Neu zu bauen | – | Aufnahme-Werkzeug, Modellwahl |
| 9 | Reachy Sprechbewegung | teilweise | Simulator bewegt sich (Markus, 15.11.) | **Lücke bestätigt:** `speaking` nur bei proaktiven Sätzen, nicht bei Chat-Antworten | `POST /speech/state`, Meldung aus der Sprachausgabe der Zentrale | Abnahme am Simulator; Pi-`/tts` und Cockpit melden noch nicht |
| 10 | Gemeinsamer Core | teilweise | unbekannt | EventBus + `/live` + StateEngine existieren; kein einheitliches Ereignisformat mit TTL | – | Ereignisschema, Situationszustand |
| 11 | Hardware-Selbstwissen | teilweise | unbekannt | NPU-Selbsttest rechnete nicht und öffnete zweites VDevice; README nannte 26 TOPS | NPU-Selbsttest, README, Selbstwissen | Inventar am Gerät, Chat-Diagnose |
| 12 | Reachy-Selbstbeobachtung | nein | nein | Neu; alle 8 NPU-Plätze belegt | – | Trainingsablauf, Slot-Frage |

---

## 2. Bereiche im Einzelnen

### 1 · Langzeitgedächtnis und Ollama

**Antworten auf die Prüffragen (Stufe Code):**

- **Wo liegen die Erinnerungen?** `core/longterm_memory.py`: `/mnt/moloch-data/memory/` (`facts.json`, `identity.json`, `core_state.json`, `conversations/`). Dazu `data/memory/user_knowledge.json` im Repo (155 Fakten, wird beim Start zusammengeführt) und `context/personal_context.json`.
- **Welche SSD?** `/mnt/moloch-data` = SSD 2. README sagt NTFS, CLAUDE.md sagt ext4 seit Umformatierung. Am Gerät prüfen.
- **Läuft Qdrant? Sind Embeddings aktiv?** Im Code vorgesehen (Port 6333). `settings.json` enthält **keinen** Schlüssel `embedding_bridge` → Embeddings und Vektor-Recall sind **aus**. Ob Qdrant läuft: nur am Gerät feststellbar.
- **Welche Chat-Pfade nutzen das Gedächtnis?**

  | Pfad | Gedächtnis im Prompt (vor dieser Sitzung) |
  |---|---|
  | Pi `/chat` → Tentakel/Kaskade (PC-Ollama) | `get_memory_context_minimal`: nur Markus, Rebecca, Genesis |
  | Pi `/chat` → DeepSeek | dasselbe |
  | Voice-Pipeline ohne lokales Modell, Cloud | `get_memory_context`: Identität, **alle** Fakten, Spotify-Profil, letzte 20 Nachrichten |
  | Voice-Pipeline mit lokalem Modell | `get_memory_context_minimal` |
  | KI-Zentrale lokal (PC → Ollama direkt) | Persona aus `/system_prompt` + eigener Verlauf, **kein** Gedächtnisabruf |
  | Agent-Tool `read_memory` | rief `recall()` auf, das leer zurückgab |

- **Erhält Ollama die Informationen?** Die gelernten Fakten: **nein.** Es gibt nur zwei Stufen – alles (für Cloud-Modelle mit großem Kontext) oder fast nichts (für lokale Modelle, damit der Prompt klein bleibt). Seit die Antworten lokal kommen, fehlt das Gedächtnis. Das erklärt die Beobachtung im Briefing.
- **Konkurrierende Schnittstellen?** Zwei Speicher (`persistent_memory` + `longterm_memory`), die sich gegenseitig abgleichen; eine Abrufstelle (`MolochMemory`). Keine dritte eingeführt.
- **Ohne PC verfügbar?** Alles, was auf dem Pi liegt. Der neue Wort-Recall braucht weder PC noch Qdrant.

**Umgesetzt (Stufe Test):**

- `core/memory/lexical_recall.py`: Abruf über gemeinsame Wörter, lokal, 0,6 ms pro Frage bei 177 Einträgen.
- `config/self_knowledge.json`: 22 Einträge Selbstwissen und Projektchronik.
- `MolochMemory.recall_hits()` / `recall()`: Treffer mit Quelle (`fakt`, `selbstwissen`) und Modus (`lexical`, `vector`).
- `GET /memory/recall?q=…` im Chat-Server: Schnittstelle für den PC, loggt jeden Abruf als `[MEMORY-RECALL]`.
- `pc/ki_zentrale.py`: lokaler Chat holt pro Frage die Treffer und hängt sie als `[GEDAECHTNIS]`-Block an.
- Wissensarten laut Briefing: persönliche Fakten ✔, Selbstwissen ✔, Projektchronik ✔ (als Selbstwissen), Gesprächsverlauf ✔ (vorhanden), Sensorik ✔ (`/live`). **Fehlt:** episodische Erinnerungen (EpisodicMemory ist aus RAM-Gründen deaktiviert), Rechercheergebnisse als eigene Art.

**Prozessgrenze:** Hauptservice (Voice) und Chat-Server halten die Fakten je im RAM. Vor jedem Abruf wird `facts.json` übernommen, wenn sie sich geändert hat (höchstens alle 10 s). Ein per Sprache gelernter Fakt ist damit im Chat ohne Neustart abrufbar.

**Grenzen des Wort-Recalls:** findet nur, was ein Wort mit der Frage teilt. „Was trinke ich morgens?" findet den Kaffee-Fakt nicht, wenn darin „Kaffee" steht und in der Frage nicht. Dafür ist die Vektorsuche da – erst einschalten, wenn der Deutsch-Test (`scripts/check_after_merge.py`) besteht.

**Abnahmetest am Gerät:**

```bash
# 1. Fakt nur auf dem Pi speichern (eigener Prozess - der Chat-Server uebernimmt ihn binnen 10 s)
cd ~/moloch && python3 -c "from core.longterm_memory import get_memory; \\
  get_memory().add_fact('Test_Codewort', 'Zitronenfalter 4711', source='abnahmetest')"
# 2. Abruf direkt
curl -s 'localhost:9100/memory/recall?q=Was%20ist%20das%20Codewort%20Zitronenfalter'
# 3. In der KI-Zentrale (lokal) fragen: "Was ist das Codewort Zitronenfalter?"
#    Erwartung: Antwort nennt 4711, Abschlusszeile zeigt "1 Erinnerung genutzt (fakt)"
# 4. Nachweis im Log
journalctl -u moloch-chat --since -5min | grep MEMORY-RECALL
# 5. Aufraeumen: get_memory().remove_fact('Test_Codewort')
```

### 2 · Lokale Sprachmodelle und Antwortqualität

- **Routing (Code):** `llm_mode = "kaskade"`. Seit 15.11. `local_first`: PC-Modell antwortet, DeepSeek nur bei Ausfall (Handoff, am Gerät getestet: 16–39 s, Fallback 3,6 s). `web_model = "api_deepseek"` schickt Web-Recherche weiter direkt in die Cloud.
- **Widerspruch Modellname:** `settings.tentacle_llm.model = "dolphin-llama3:8b"`, Handoff spricht von „PC-qwen", KI-Zentrale nimmt `haupthirn` aus ihrer Rollen-Datei, Standard `qwen2.5:7b-instruct`. Welches Modell wirklich antwortet, zeigt nur `ollama ps` auf dem PC.
- **Profile (`config/llm_profiles.json`):** `tentacle` 160 Token, Temperatur 0,85, System-Prompt 2423 Zeichen; `chat` 120 Token; `dark` 40 Token. 160 Token bei Temperatur 0,85 auf einem 7B-Modell ist ein plausibler Grund für wirre Antworten – Hypothese, nicht belegt.
- **Identitätsdateien:** `config/system_prompt.txt` (434 Bytes, nur noch vom alten GUI-Panel gelesen), `config/moloch_identity.json` (wird ins Gedächtnis gemischt), `llm_profiles.json` (trägt den wirksamen System-Prompt). Drei Quellen, nur eine wirkt im Chat.
- **Umgesetzt:** Selbstwissen als abrufbare Einträge statt als weiterer fester Prompt-Text – so wächst der System-Prompt nicht (Latenz!) und das Modell bekommt nur, was zur Frage passt.
- **Offen:** die vier Vergleichstests des Briefings; `system_prompt.txt` stilllegen oder zusammenführen; Temperatur/Token am Gerät ausprobieren.

Fragenkatalog für den Abnahmetest: `docs/anima/FRAGENKATALOG.md`.

### 3 · Chat und Kamerasteuerung

- **Chat → Kamera (Code):** Pi-`/chat` ruft den Keyword-Handler vor dem LLM auf. „schau zur Tür" → `ptz_goto` mit benannter Position, Arbiter auf MANUELL. Läuft über die vorhandene Prioritätssteuerung, kein zweiter Controller.
- **Kamera → Chat (Code):** `_build_local_context_snippet` gibt erkanntes Gesicht, Person, Zone in den Prompt; KI-Zentrale hängt `[LIVE]` an.
- **Lücke:** Der **lokale** Modus der KI-Zentrale spricht direkt mit Ollama. Dort greift kein Keyword-Handler – „schau zur Tür" bewegt nichts. Der Modus „pi" der Zentrale geht über `/chat` und kann es.
- **Zweite Lücke (per Test belegt):** Die Erkennung verlangte die Wendung wörtlich. „Moloch, schau mal zur Tür" – der Beispielsatz des Briefings – traf nicht, weil „mal" dazwischen steht. Das galt auch für den Pi-Chat und die Sprache.
- **Umgesetzt (Stufe Test):** Füllwörter werden vor dem Abgleich entfernt, die Wendung bleibt Pflicht („die Tür klemmt" bewegt nichts). `POST /command` fragt nur den Keyword-Handler; die Zentrale ruft das im lokalen Modus vor Ollama auf.
- **Abnahme am Gerät:** In der Zentrale (lokal) „Moloch, schau mal zur Tür" → Kamera fährt, Antwort „Blick zur Tuer.", Abschlusszeile „Befehl (Pi)". Danach „Die Tür klemmt" → keine Bewegung.
- **Offen:** „Was siehst du?" liefert derzeit nur Klassen und Gesichts-ID; eine Bildbeschreibung läuft laut Entscheidung vom 05.10. über den PC (moondream), Stand unbekannt.

### 4 · Bounding-Boxen und Raumkalibrierung

- **Code:** `estimate_distance()` in `perception_frame.py` liefert `close` / `medium` / `far` aus dem Flächenanteil der Box. Keine Meter, keine Kamerahöhe, kein Zoom, keine Unterscheidung sitzend/stehend. `core/awareness/room_map.py` kennt fünf Pan-Zonen (Tür, Schreibtisch, Mitte, Sofa, Fenster) mit festen Winkeln. Ein DepthWorker (scdepthv3) existiert.
- **Offen:** Kalibrierprofil pro Raum (`config/rooms/<raum>.json`), Messreihe bei 0,3 / 0,5 / 1,5 / 3,0 m, Schätzung aus Körperhöhe über Pose-Keypoints statt Boxfläche. Braucht Messwerte von Markus.

### 5 · ReSpeaker und Audiowahrnehmung

- **Code:** Zwei Wege – USB und ESP32-WLAN (`audio_source: "auto"`). Sprache: Whisper über PC-Bridge, Fallback NPU. Ereignisse `audio.speaking_start/end`, `mic.mode_changed`, Musik-Beat. **Keine** Aktivitäts- oder Geräuschereignisse.
- **Bekannt offen (Handoff 13.06.):** ReSpeaker lieferte zeitweise 0 Byte; Diagnose-Logging eingebaut, Ergebnis nicht dokumentiert.
- **Offen:** VAD-Stufe, Ereignis `audio.sound` in den EventBus, Stummschaltung, Sperre während eigener Sprachausgabe (`_speaking` existiert in der Voice-Pipeline).

### 6 · Spotify-DJ

- **Code:** `core/spotify_controller.py` mit `auto_dj_*` und `get_recommendations_for_zone`; `core/music/music_memory.py`; 10 Spotify-Tools im Tool-Katalog.
- **Risiko:** Am 03.10. meldete der Service „Refresh token revoked". Ohne gültigen Token ist jede DJ-Arbeit Blindflug.
- **Offen:** Token prüfen; Ablehnungen merken; Wiederholungssperre; Mindestspieldauer gegen hektische Wechsel; Begründung der Auswahl. Die Spotify-Empfehlungs-API ist für neue Apps eingeschränkt – am Konto prüfen, was noch geht.

### 7 · Internetrecherche

- **Code:** `pc/search_proxy.py` (DuckDuckGo-Scrape, `/search`, `/fetch` mit Volltext, Cache). `_fetch_search_context` nimmt **höchstens 3 Treffer mit je 120 Zeichen**. `pc/auto_researcher.py` existiert. Schutz gegen interne Adressen (`url_guard`) seit 29.09.
- **Befund bestätigt:** Für mehr als Stichwortfragen reicht das nicht. `/fetch` wird im Antwortpfad nicht genutzt.
- **Offen:** Recherche-Ablauf: 2–3 Suchformulierungen → Treffer zusammenführen → 2–3 Seiten per `/fetch` lesen → Antwort mit Quellenliste. Webseiteninhalt bleibt Daten (kein Befehl).

### 8 · Akustische Bibliothek

- **Code:** nichts vorhanden.
- **Vorschlag:** Embedding-Modell auf dem PC (YAMNet o. ä.), kleiner Klassifikator, Proben auf der Pi-SSD. Zuerst das Aufnahme- und Beschriftungswerkzeug, weil ohne Daten keine Modellwahl begründbar ist. Hängt an Bereich 5.

### 9 · Reachy: Sprechbewegungen

- **Code:** `compute_pose()` nickt bei `speaking=True`. `pc/body_bridge.py` liest `/live.speech` und setzt `speaking`, wenn der Pi einen **proaktiven** Satz spricht.
- **Lücke bestätigt:** Chat-Antworten (PC-TTS, Browser-Sprachausgabe der Zentrale, Pi-`/tts`) setzen `/live.speech` nicht. Die Voice-Pipeline veröffentlicht `audio.speaking_start/end`, aber nur für ihren eigenen Weg.
- **Umgesetzt (Stufe Test):** `POST /speech/state` auf dem Pi schreibt denselben Zustand wie die Voice-Pipeline. Die Browser-Sprachausgabe der Zentrale meldet Beginn, Ende und Abbruch; alle 10 s wird aufgefrischt, nach 30 s ohne Meldung verfällt der Zustand.
- **Abnahme am Simulator:** In der Zentrale mit eingeschalteter Sprachausgabe eine Frage stellen → Kopf nickt, solange gesprochen wird, danach Ruhe. „Stopp" beendet beides.
- **Offen:** Das Cockpit (`/tts` liefert Audio an den Browser) meldet noch nicht. Die Nick-Bewegung (2° bei 2 Hz) ist ungetunt – am Simulator ansehen, ob sie zu hektisch wirkt.

### 10 · Gemeinsamer Core

- **Code:** EventBus (`core/event_bus.py`), StateEngine (6 Zustände), `state_vector`, `/live`, `/api/state/current`, PTZ-Arbiter, `core/world/world_state.py`.
- **Befund:** Die Teile existieren, aber es gibt kein einheitliches Ereignisformat mit Quelle, Zeitstempel, Gültigkeit und Sicherheit. `/live` ist faktisch der gemeinsame Situationszustand.
- **Vorschlag:** `/live` zum offiziellen Situationszustand erklären und um die fehlenden Felder erweitern, statt ein neues Zustandsmodell zu bauen. Ereignisschema als Konvention auf dem bestehenden EventBus.
- Prioritäten des Briefings decken sich mit `core/body/controller.py` (Not-Aus > manuell > Dialog > Grundausdruck); „Tanz und Show" fehlt als unterste Stufe.

### 11 · Hardware-Selbstwissen

- **Pi-RAM: 4 GB** (von Markus am 15.11.2026 bestätigt; README nannte fälschlich 8 GB, korrigiert).
- **Widersprüche in der Doku:** SSD 2: README NTFS, CLAUDE.md ext4. TOPS: README 26 (korrigiert auf 40).
- **Code:** `core/system_watchdog.py` prüft Frame-Stillstand, Pipeline, ONVIF, RAM/CPU, Platte, Mikrofon, Netz, Ollama. Tools `get_npu_status`, `moloch_status_summary`, `thermal_set_tension_pwm`. `hailo_manager._force_release_device()` beendet fremde Prozesse am NPU-Gerät – schont eigene Prozesse, bleibt aber ein Eingriff; nicht aus Diagnosen heraus aufrufen.
- **Umgesetzt:** `scripts/self_diagnosis.py::test_hailo_npu` weist jetzt echte Inferenz nach (Zähler der Worker laufen weiter) und öffnet bei laufendem Service kein VDevice mehr.
- **Offen:** Inventar am Gerät; Unterspannung/Throttling (`vcgencmd get_throttled`); Lüfter-Tacho; Power-Platine („Pico Power 5" laut Gedächtnis-Fakt, Auslesbarkeit unbekannt); Chat-Fragen wie „Wie warm ist dein Raspberry?" mit echten Messwerten beantworten.

### 12 · Reachy-Selbstbeobachtung

- **Code:** nichts vorhanden. Voraussetzung `core/body/` und Body-Bridge existieren.
- **Harte Randbedingung:** `HAILO_MAX_NETWORK_GROUPS = 8`, alle acht belegt. Ein Reachy-Modell braucht einen freien Platz – bedarfsgesteuertes Umschalten ist Pflicht, nicht Kür.
- **Reihenfolge:** (1) echtes Gerät vorhanden? (2) Aufnahmen mit der Sonoff, (3) Modellwahl an Daten, (4) Hailo-Toolchain für den 10H prüfen, (5) Bewegungsabgleich. Vor dem ersten Schritt ist Code dafür Spekulation.

---

## 3. Arbeitsplan

Jedes Paket ist für sich mergebar, hat einen Abnahmetest und nennt, was am Gerät geprüft werden muss.

| Paket | Inhalt | Bereich | Größe | Braucht Gerät | Status |
|---|---|---|---|---|---|
| **P1** | Gedächtnisabruf: Wort-Recall, Selbstwissen, `/memory/recall`, Zentrale | 1, 2 | mittel | Abnahme | **gebaut, ungeprüft am Gerät** |
| **P2** | NPU-Selbsttest ehrlich, README | 11 | klein | Abnahme | **gebaut, ungeprüft am Gerät** |
| **P3** | Laufzeit-Inventar: Skript, das Phase B abarbeitet und einen Bericht schreibt | 1, 2, 11 | klein | ja | **gebaut** (`scripts/anima_inventory.py`), am Gerät laufen lassen |
| **P4** | Ein Sprechstatus für alle TTS-Wege → Reachy nickt bei jeder Antwort | 9 | klein–mittel | Simulator | **Zentrale gebaut, ungeprüft am Simulator**; Pi-`/tts`/Cockpit offen |
| **P5** | Lokaler Chat der Zentrale erkennt Befehle (Kamera, Musik) über den Pi | 3 | klein | ja | **gebaut, ungeprüft am Gerät** |
| **P6** | Hardware-Fragen im Chat mit Messwerten (Temperatur, Speicher, NPU) | 11 | mittel | ja | offen |
| **P7** | Recherche mehrstufig mit Volltext und Quellen | 7 | mittel | PC | offen |
| **P8** | `/live` als Situationszustand, Ereignisschema | 10 | mittel | ja | offen |
| **P9** | Audio-Ereignisse (VAD) in den EventBus | 5 | mittel | ja | offen |
| **P10** | Spotify-DJ: Ablehnungen, Wiederholungssperre | 6 | mittel | Konto | offen, Token zuerst |
| **P11** | Raumkalibrierung: Profil + Messreihe | 4 | mittel | ja + Messung | offen |
| **P12** | Geräuschbibliothek: Aufnahme-Werkzeug | 8 | mittel | ja | offen, nach P9 |
| **P13** | Reachy-Erkennungsmodell und Bewegungsabgleich | 12 | groß | ja + Roboter | offen, zuletzt |

**Empfohlene Reihenfolge:** P1/P2 am Gerät abnehmen → P3 → P4 → P5 → P6 → P7 → P8 → Rest.
Begründung: P3 beantwortet die offenen Laufzeitfragen für alle anderen Pakete. P4 und P5 schließen bestätigte Lücken mit wenig Code. P13 hängt an Hardware, Daten und einem freien NPU-Platz.

---

## 4. Prüfliste für die nächste Sitzung am Gerät (Phase B)

**Auf dem Pi**

```bash
free -h; lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT; findmnt /mnt/moloch-data; df -h / /mnt/moloch-data
python3 --version; cat /proc/device-tree/model
hailortcli fw-control identify --extended        # Chip, Firmware (nur lesend)
systemctl is-active moloch moloch-chat-https qdrant 2>/dev/null; ss -ltnp | grep -E '6333|9100|9443|8000'
python3 scripts/check_after_merge.py            # Importe, Status, Qdrant, Embedding, Deutsch-Test
python3 scripts/self_diagnosis.py               # inkl. neuem NPU-Test
python3 scripts/face_db_doctor.py               # Face-DB
vcgencmd measure_temp; vcgencmd get_throttled
curl -s localhost:9100/live | python3 -m json.tool | head -60
curl -s 'localhost:9100/memory/recall?q=Wer%20bist%20du%20eigentlich'
```

**Auf dem PC**

```
ollama ps            # welches Modell ist geladen
ollama list          # welche Modelle gibt es (all-minilm fuer Embeddings?)
```

Ergebnisse bitte in `logs/agent_handoff.md` – daraus wird in diesem Bericht „Gerät".

---

## 5. Was bewusst nicht getan wurde

- Keine Änderung an `settings.json`, an der Hailo-Pipeline, am PTZ-Pfad oder an ROT-Dateien.
- Vektorsuche nicht eingeschaltet (ungetestet gegen echtes Ollama; Modell ist englisch trainiert).
- Kein zweites Gedächtnis, kein zweites Zustandsmodell, kein zweiter Kamera-Controller.
- Keine Erinnerungen gelöscht oder verändert.
- Kein Code für Bereiche, deren Voraussetzungen ungeklärt sind (8, 12).
