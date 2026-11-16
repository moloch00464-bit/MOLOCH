# Recherche-Schleuse – Abgleich mit dem Code und Einwände

**Stand:** 2026-11-16 · **Zum Konzept:** `RECHERCHE_KONZEPT_2026-11-16.md` (DeepSeek)
**Gelesen, nicht am Gerät geprüft.** Basis `deepseek_architecture_overhaul` @ `ee0f3bb`.

Das Konzept ist im Grundsatz richtig – *DeepSeek liefert, Moloch entscheidet* – und erklärt, woher die Testfälle
L01, L12 und L13 der Ehrlichkeitsprüfung stammen. Es übersieht aber, dass es schon einen Recherche-Weg gibt.
Dieses Dokument hält fest, was vorhanden ist, wo das Konzept hakt und wo ich ein **Veto** einlege.
Vetos sind Bedingungen, ohne die ich die Umsetzung nicht für vertretbar halte; Markus entscheidet.

---

## 1. Was es schon gibt

| Baustein | Ort | Was er tut |
|---|---|---|
| Suchdienst | `pc/search_proxy.py`, PC :11650 | DuckDuckGo-Scrape ohne Schlüssel. `POST /search` (Titel, Ausschnitt, URL), `POST /fetch` (Volltext einer Seite), Cache, 180 s Sperre pro gleicher Anfrage, Schutz gegen interne Adressen (`pc/url_guard.py`) |
| Erkenner im Pi-Chat | `core/bridge/chat_server.py` | `_WEB_LIVE_KEYWORDS` („recherchier", „schau nach", „such ", „finde heraus" …), Festival- und Band-Muster, `_classify_prompt_type()` → `web_research` |
| Web-Pfad im Pi-Chat | `chat_server.py` `/chat`, ~Z. 2540–2730 | holt Suchergebnisse (Ausschnitt 200 Zeichen), baut `WEB-RESULTS` + Anweisung „nur aus den Ergebnissen, nichts raten, echte URLs nennen" und schickt das an **DeepSeek in der Cloud**, weil `settings.tentacle_llm.web_model = "api_deepseek"` |
| Schutz bei leerer Suche | `chat_server.py` ~Z. 2680 | Liefert der Suchdienst nichts, geht die Frage **nicht** in die Cloud; Moloch sagt ehrlich, dass nichts kam (eingebaut nach einer erfundenen WGT-Bandzahl am 09.05.). Gutes Vorbild für das Knowledge Gate |
| Web-Kontext im Tentakel-Pfad | `local_llm_bridge._fetch_search_context()` | höchstens 3 Treffer, Titel 80, Ausschnitt 120 Zeichen |
| Auto-Rechercheur | `pc/auto_researcher.py` | täglicher Abgleich (Hailo-Releases, Ollama-Modelle, LoRA) → Vorschläge in `staging/research_proposals/`, Cockpit-Reiter „Forschung". Nicht für Nutzerfragen |
| PC-Agent | `pc/agent/` | `deepseek_client.py` (Schlüssel aus `config/api_keys.json` oder `DEEPSEEK_API_KEY`), `token_budget.py`, `safety.py` |

**Folge:** Ein Research Gateway entsteht nicht neu, sondern aus diesen Teilen. Ein zweiter Recherche-Weg neben dem
vorhandenen wäre genau die Doppelung, die alle drei ANIMA-Dokumente ausschließen.

---

## 2. Vetos

### V1 – Kein zweiter Recherche-Weg
Das Gateway wird aus `search_proxy.py` und dem vorhandenen Web-Pfad des Chat-Servers entwickelt. Der Erkenner
(`_WEB_LIVE_KEYWORDS`, `_classify_prompt_type`) bleibt die eine Stelle, die über „Recherche ja/nein" entscheidet.
`auto_researcher.py` bleibt getrennt (anderer Zweck), nutzt aber dieselbe Schleuse, falls er je Nutzerfragen beantwortet.

### V2 – Heute verlassen private Daten das Haus – zuerst abstellen
Der vorhandene Web-Pfad hängt bei Festival- und Bandfragen (wenn ein Spotify-Ausschnitt vorliegt) **Markus' Spotify-Hördaten** (`spotify_ctx`,
Künstler mit Wiedergabezahlen) an den Prompt und schickt ihn an DeepSeek in der Cloud
(`chat_server.py`, `cross_ref_block` → `augmented` → `_generate_deepseek`). Das widerspricht Abschnitt 8 des Konzepts
(„keine ungefilterten privaten Daten hochladen").
**Bedingung:** Bevor die Schleuse erweitert wird, gilt für alles, was nach außen geht:
- nur die Suchanfrage und die öffentlichen Suchergebnisse, **keine** Gedächtnis-Treffer, keine Hördaten,
  keine Live-Daten, keine Namen aus dem Haushalt;
- der Abgleich mit privaten Daten (z. B. „welche dieser Bands höre ich?") passiert **lokal** nach der Recherche.

Das ist eine Entscheidung für Markus: Er hat den Spotify-Abgleich vermutlich bewusst gewollt. Aber er sollte
wissen, dass er dafür seine Hördaten an einen Cloud-Anbieter gibt.

### V3 – Keine selbstständige Recherche, solange sie Geld kostet und das Modell sich nicht einschätzen kann
„Reicht vorhandenes Wissen?" ist der schwierigste Schritt, und das Konzept sagt nicht, wie er entschieden wird.
Ein 7B-Modell kann nicht zuverlässig erkennen, was es nicht weiß.
**Bedingung für die erste Ausbaustufe:** Recherche nur
1. auf ausdrücklichen Wunsch (vorhandener Erkenner: „recherchier", „schau nach", „such" …), und
2. bei eindeutig zeitabhängigen Fragen (vorhandene Muster).
Selbst ausgelöste Recherche aus internen Zuständen erst, wenn Budget, Abbruch und Protokoll stehen und Markus
sie freigibt.

### V4 – Das Knowledge Gate verspricht nur, was es prüfen kann
Prüft dasselbe kleine Modell oder DeepSeek selbst, kontrolliert es sich selbst. Mindestens und ehrlich machbar:
- **Quellen zählen nur, wenn sie aus der API-/Suchantwort stammen**, nicht aus dem Antworttext des Modells.
  Eine URL im Antworttext, die nicht in den Suchergebnissen steht, wird entfernt oder als unbelegt markiert.
- Die Antwort muss erkennbar zur Frage passen (Begriffe der Frage kommen vor; das „Bär statt Auto"-Beispiel).
- Alles andere: `verified_by_core: false`, und so wird es auch gesagt.
Keine Konfidenz-Prozentzahlen, keine Behauptung „geprüft", wenn nur formal gefiltert wurde.

### V5 – Kein neues Zustandsmodell für ANIMA
Die zehn Zustände des Konzepts (`IDLE` … `ANSWERING`) wären ein weiteres Zustandsmodell neben Seins-Zustand,
Zone, Mood und dem geplanten somatischen Zustand (`NERVENSYSTEM_PLAN.md`).
**Bedingung:** Recherche wird ein **Meldekanal** wie der Sprechstatus:
Pi schreibt einen kleinen Zustand (z. B. `research: {phase, request_id, seit, gültig_bis}`) → `/live` → Body-Bridge
→ ein zusätzlicher Kanal in `core/body/expression.py`. Vorbild: `POST /speech/state` und `/live.speech`.
Phasen auf das Nötige beschränken: `läuft`, `prüft`, `gefunden`, `unsicher`, `nichts`, `fehler`. Nach Ablauf der
Gültigkeit verfällt der Zustand von selbst – kein „hängendes RESEARCHING" (Testfall L13).

### V6 – „Ich schau mal nach" nur, wenn wirklich nachgeschaut wird
Die Ankündigung ist nur ehrlich, wenn das Gateway den Auftrag **angenommen** hat (`ACCEPTED` im Sinne des
Nervensystem-Plans). Ohne laufende Recherche ist sie eine vorgetäuschte Aktion. Das löst den Einwand gegen
DeepSeeks Prüfer-Leitlinie F3 (`pruefprotokolle/ANPASSUNG_AN_DEN_CODE.md`, Abschnitt 8).

### V7 – Kosten hart begrenzen, bevor etwas automatisch läuft
Tagesbudget, höchstens N Aufrufe pro Frage, Abbruch, Cache. `pc/agent/token_budget.py` existiert – prüfen und
wiederverwenden. Bei erschöpftem Budget ehrliche Meldung (Testfall L12), keine Ausweichrecherche über einen
anderen Anbieter ohne Freigabe.

---

## 3. Weitere Lücken (keine Vetos)

- **Wo läuft das Gateway?** Der Suchdienst sitzt auf dem PC. Ist der PC aus, gibt es heute keine Suche.
  Optionen: (a) Recherche nur mit PC, ehrlich sagen „geht gerade nicht"; (b) Pi geht selbst zu DeepSeek.
  Empfehlung (a) für den Anfang – weniger Wege, weniger Schlüssel auf dem Pi. *(Markus)*
- **DuckDuckGo oder DeepSeek-Suche?** Der vorhandene Weg nutzt DuckDuckGo-Scraping ohne Schlüssel (kostenlos,
  aber anfällig für Layout-Änderungen). Das Konzept schlägt DeepSeeks eigene Websuche vor. Beides als Anbieter
  hinter einer Schnittstelle, so wie das Konzept es selbst verlangt.
- **API-Angaben ungeprüft.** `deepseek_web_search`, die Search-/Fetch-Endpunkte und 0,75 USD je 1.000 Anfragen
  sind auf den 16.11.2026 datiert. Die umsetzende Sitzung muss die Dokumentation selbst lesen.
- **Volltext wird nicht genutzt.** `/fetch` existiert, der Antwortweg nimmt nur Ausschnitte (120–200 Zeichen).
  Das ist der größte Qualitätshebel und gehört in die erste Ausbaustufe (deckt sich mit Paket P7 in #3).
- **Gedächtnis.** Das Konzept will geprüfte Rechercheergebnisse mit Quelle und Abrufzeit speichern. Der heutige
  Faktenspeicher kennt weder Quelle noch Gültigkeit und überschreibt still (siehe Ehrlichkeitsprüfung,
  Abschnitt 6). Recherche-Ergebnisse deshalb vorerst **nicht** als Fakten speichern; erst wenn der Speicher
  Herkunft und Zeit tragen kann.
- **Sprachverständnis vor der Suche** (Abschnitt 6 des Konzepts): richtig, aber Phase D. Für den Anfang reicht die
  Rückfrage bei sehr kurzen oder mehrdeutigen Anfragen.
- **Murmeln** („Hm … mal sehen"): Antworten dauern schon ohne Recherche 15–57 s. Höchstens ein Zwischenruf pro
  Recherche, Ruhemodus beachten.

---

## 4. Vorschlag für die erste Ausbaustufe

Klein, ohne neue Abhängigkeit, alles auf vorhandenen Teilen:

1. **V2 umsetzen:** keine privaten Daten im Cloud-Prompt des Web-Pfads; Abgleich mit Hördaten lokal.
2. **Volltext:** 2–3 Treffer per `/fetch` lesen, gekürzt, mit Quelle.
3. **Quellenprüfung (V4):** URLs in der Antwort gegen die tatsächlich abgerufenen prüfen.
4. **Recherche-Status** als Meldekanal nach V5, zunächst nur in `/live` und im Log; Körper danach.
5. **Budget (V7)** aus `pc/agent/token_budget.py`.
6. **Tests ohne Netz:** Mock-Anbieter für Treffer, keine Treffer, Zeitüberschreitung, widersprüchliche Quellen,
   Webseite mit eingebetteter Anweisung, Budget erschöpft. Die Testfälle L01, L12, L13 der Ehrlichkeitsprüfung
   werden damit ausführbar.

Danach erst: DeepSeek-Websuche als zweiter Anbieter, Reachy-Gesten, Sprachverständnis, Gedächtnisübernahme.
