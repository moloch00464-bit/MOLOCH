> Abgelegt am 2026-11-16, Wortlaut (Konzept von DeepSeek, über Markus). Das ist der Vorschlag, nicht der Stand.
> Abgleich mit dem vorhandenen Code und Einwände: `docs/anima/RECHERCHE_ABGLEICH.md`.

# M.O.L.O.C.H. 5.0 – Recherche-Schleuse mit DeepSeek und Reachy-ANIMA

**Stand:** 16. November 2026  
**Status:** Architekturvorschlag / Entwicklungsbriefing für Claude Code – noch nicht implementiert oder getestet  
**Leitidee:** M.O.L.O.C.H. nutzt eine externe Recherche-KI als kontrollierten Wissenslieferanten. Sein eigener Kern bleibt verantwortlich für Fragestellung, Bewertung, Gedächtnis und Reaktion. Reachy Mini macht die laufenden Verarbeitungsschritte mit Kopf, Körper, Antennen und kurzen Äußerungen sichtbar.

## 1. Zielbild

M.O.L.O.C.H. muss nicht auf jede Frage sofort eine Antwort erfinden. Wenn Informationen fehlen oder veraltet sind, kann er sagen:

> „Moment, davon hab ich gerade keine Ahnung. Ich schau mal nach.“

Während er recherchiert, zeigt Reachy mit einer lebendigen, aber zurückhaltenden Animation, dass M.O.L.O.C.H. beschäftigt ist. Beispiel: Kopf leicht schief, Antennen in ruhiger Bewegung, gelegentliches „Hm … mal sehen …“. Nach dem Rechercheergebnis richtet er sich auf und antwortet; bei Unsicherheit bleibt er ausdrücklich unsicher.

Das ist **keine eigenständige Intelligenz des Reachy-Körpers**. Reachy ist der Ausdruckskörper von ANIMA, während Entscheidungen und Wissensprüfung im M.O.L.O.C.H.-Core erfolgen.

## 2. Warum eine Recherche-KI statt eines offenen Browsers?

- M.O.L.O.C.H. braucht **keinen allgemeinen, unbeschränkten Webzugriff** und keine eigene Browser-Automation.
- Nur ein kontrollierter Dienst mit begrenzten ausgehenden API-Aufrufen darf Recherche auslösen.
- Die Recherche-KI kann Suchergebnisse zusammenführen und eine verständliche Antwort mit Quellen liefern.
- Das Modell ist **nicht automatisch wahrheitsgetreu**. Suchfehler, Halluzinationen und manipulierte Quellen bleiben möglich.
- Anbieterwechsel ist vorgesehen: DeepSeek zuerst; andere Rechercheanbieter oder ein separater Suchdienst später, ohne den Core umzubauen.

**Verifizierter API-Stand (16.11.2026):** DeepSeek beschreibt `deepseek_web_search` als serverseitiges Werkzeug für Websuche mit Quellen im Modellaufruf. Alternativ gibt es getrennte Search- und Fetch-API-Endpunkte. Das ist eine reale Option, aber API-Verfügbarkeit und Kosten müssen beim Einbau erneut geprüft werden. In den offiziellen Dokumenten sind für Search und Fetch jeweils **0,75 USD pro 1.000 erfolgreiche Anfragen** genannt; Modelltoken und gegebenenfalls weitere Aufrufe kommen hinzu. **Das ist kein garantierter Komplettpreis pro fertiger Antwort.**

## 3. Zuständigkeiten

| Komponente | Aufgabe |
|---|---|
| **M.O.L.O.C.H.-Core** | Entscheidet, ob Wissen fehlt; prüft Qualität und Relevanz der Antwort; formuliert finale Aussage. |
| **ASR / Sprachverständnis** | Wandelt gesprochene Nutzerfrage in Text und erkennt Unsicherheit. |
| **Query Builder** | Formuliert aus Rohtext und Kontext eine kurze, eindeutige Recherchefrage. |
| **Research Gateway** | Einziger erlaubter Weg zur externen API; verwaltet Limits, Timeout, Schlüssel und Anbieterwahl. |
| **DeepSeek-Recherche** | Sucht bei Bedarf nach aktuellen Fakten, stellt Quellen/Belege und eine Zusammenfassung bereit. |
| **Knowledge Gate** | Überprüft Themenpassung, Quellen, Widersprüche und Unsicherheit; trennt Fakten von Vermutungen. |
| **VITALE / Gedächtnis** | Speichert nur bewertete Informationen mit Provenienz und zeitlichem Gültigkeitsbezug. |
| **ANIMA** | Übersetzt echte interne Zustände in Ausdruck, kurze Sprache und Bewegungsereignisse. |
| **MOLOCH_REACHY_BRIDGE** | Empfängt Zustandsereignisse und führt sichere Reachy-Bewegungen über das SDK aus. |
| **Reachy Mini Wireless** | Zeigt Kopf-/Körper-/Antennenbewegungen; bleibt nicht für Recherche oder KI-Entscheidungen verantwortlich. |

### Ablauf

```text
Nutzerfrage oder interne Wissenslücke
        |
        v
ASR (nur bei Sprache) -> Verständnis-/Konfidenzprüfung
        |
        +-- Frage unklar? -> Rückfrage an den Nutzer
        |
        v
M.O.L.O.C.H.-Core: Reicht vorhandenes Wissen?
        |
        +-- JA -> lokal antworten
        |
        +-- NEIN / veraltet
                |
                v
        Query Builder -> Research Gateway
                |                  |
                |                  +-> DeepSeek / anderer Anbieter
                |                  +-> Timeout / Budget / Datenschutz
                v
        ANIMA: RESEARCHING -> Reachy zeigt Suchbewegung
                |
                v
        Ergebnisse + konkrete Quellen
                |
                v
        Knowledge Gate: Relevanz, Belege, Widersprüche
                |
                +-- unklar -> kurze Rückfrage / Unsicherheit melden
                +-- brauchbar -> Core formuliert Antwort
                                       |
                                       +-> ANIMA: FOUND / ANSWERING
                                       +-> evtl. VITALE mit Quellen
                |
                +-- Fehler -> ANIMA: NOT_FOUND / ERROR
```

## 4. Zustandsmodell für ANIMA und Reachy

Die Animation darf nicht behaupten, M.O.L.O.C.H. habe etwas gefunden, solange keine passende Antwort vorliegt.

| Zustand | Bedeutung | Reachy-Ausdruck | Beispieltext |
|---|---|---|---|
| `IDLE` | Keine Recherche | Ruhige Ausgangsstellung | – |
| `CLARIFYING` | Eingabe unklar | Leicht fragende Kopfhaltung, Antennen kurz anheben | „Meinst du das Auto oder etwas anderes?“ |
| `RESEARCH_START` | API-Anfrage läuft an | Kopf aufmerksam neigen | „Moment, ich schau nach.“ |
| `RESEARCHING` | Warten auf Webantwort | Langsame, abwechselnde Antennenbewegung; dezentes Kopfpendeln | „Hm … mal sehen …“ |
| `VERIFYING` | Ergebnisse werden geprüft | Bewegungen langsamer, konzentrierte Kopfhaltung | „Das prüfe ich lieber noch.“ |
| `FOUND` | Relevante Antwort liegt vor | Aufrichten, Antennen kurz freudig bewegen | „Aha, jetzt hab ich was!“ |
| `UNCERTAIN` | Quellen widersprüchlich oder schwach | Kopf leicht schief, Bewegungen zurückhaltend | „Da bin ich mir nicht sicher.“ |
| `NOT_FOUND` | Keine brauchbare Antwort | Antennen sinken, kurze Pause | „Mist, nichts Verlässliches gefunden.“ |
| `ERROR` | Timeout, API-Fehler oder Budgetgrenze | Kurze neutrale Fehlersignatur, dann Ruhe | „Die Recherche funktioniert gerade nicht.“ |
| `ANSWERING` | Core gibt Antwort | Ruhige, zur Sprechweise passende Bewegungen | Eigentliche Antwort mit Quellenstatus |

**Wichtig:** „Antennen im Kreis drehen“ ist eine humorvolle Beschreibung, **keine Anweisung zu endloser Rotation**. Die Reachy-Mini-Antennen haben laut Hardwaredokumentation jeweils einen rotatorischen Freiheitsgrad. Alle Bewegungen müssen den dokumentierten SDK-Grenzen, Kollisionsgrenzen und einem festgelegten Bewegungsbudget entsprechen. Nicht hektisch, nicht dauerhaft, jederzeit abbrechbar.

## 5. Beispielinteraktion

**Nutzer:** „Warum wird mein Raspberry Pi plötzlich so heiß?“

**M.O.L.O.C.H.:** „Da muss ich erst nachgucken, bevor ich Blödsinn erzähle.“

**ANIMA:** `RESEARCH_START` → Reachy neigt den Kopf.

**Im Hintergrund:** Query Builder ergänzt nur notwendige technische Angaben, z. B. „Raspberry Pi 5 ungewöhnlich hohe Temperatur mögliche Ursachen und sichere Diagnose“. Research Gateway fragt DeepSeek. Reachy bewegt die Antennen ruhig, ohne jede Sekunde zu sprechen.

**M.O.L.O.C.H.:** „Hm … mal schauen …“

**ANIMA:** `VERIFYING` → Bewegung wird langsamer.

**Bei belastbarem Ergebnis:** Reachy richtet sich auf. M.O.L.O.C.H. fasst mögliche Ursachen zusammen, unterscheidet allgemeines Wissen von **tatsächlich gemessenen lokalen Sensorwerten** und nennt, worauf sich die Antwort stützt.

**Bei fehlender Evidenz:** „Ich habe keine sichere Ursache gefunden. Ich kann erst deine Temperatur-, Lüfter- und Lastwerte prüfen, wenn du das möchtest.“

## 6. Sprachverständnis: Erst sauber fragen, dann suchen

Der Nutzer hat einen wichtigen Fehlerpfad benannt: Wer eine schlecht verstandene Frage ins Netz schickt, erhält oft überzeugend formulierten Unsinn. Deshalb:

1. **ASR-Text prüfen:** Erkennbare Unsicherheiten, unklare Eigennamen und Schlüsselausdrücke markieren.
2. **Kontext auflösen:** Bezug auf Pi, Hailo, Reachy oder andere im Gespräch genannte Objekte verwenden, ohne Fakten zu erfinden.
3. **Query Builder:** Eine einzige präzise Suchabsicht erzeugen, keine ungeprüften freien Assoziationen.
4. **Bei Mehrdeutigkeit nachfragen:** „Meinst du X oder Y?“ statt blind zu suchen.
5. **Frage-Antwort-Abgleich:** Prüfen, ob die Antwort wirklich die ursprüngliche Frage beantwortet.
6. **Ausgabe:** Möglichst kurze, verständliche Antwort; nur auf Nachfrage ausführlich recherchieren.

**Entkopplung:** Die Recherche-Schnittstelle kann bereits textbasiert getestet werden, auch wenn ASR oder M.O.L.O.C.H.s eigene TTS-Stimme noch nicht zuverlässig funktionieren. Selbst initiierte Recherche aus internen Zuständen braucht keine Spracherkennung.

## 7. Minimales Ereignisformat (Vorschlag, keine bestehende API)

```json
{
  "event": "research.status",
  "request_id": "example-001",
  "state": "VERIFYING",
  "timestamp": "2026-11-16T08:54:00+01:00",
  "provider": "deepseek",
  "animation": "thinking_calm",
  "speak_hint": null,
  "confidence": null
}
```

Vorgeschlagene Gateway-Rückgabe:

```json
{
  "request_id": "example-001",
  "status": "ok",
  "question": "Warum wird ein Raspberry Pi 5 heiß?",
  "answer": "Zusammenfassung, die der Core noch prüft",
  "sources": [
    {"url": "https://example.org/source", "title": "Beispielquelle", "retrieved_at": "2026-11-16T08:54:00+01:00"}
  ],
  "verified_by_core": false,
  "eligible_for_memory": false
}
```

Die Beispiel-URL ist **nur ein Platzhalter**, keine echte Quelle. Der Core und das Knowledge Gate dürfen `verified_by_core` erst nach einer tatsächlichen Prüfung setzen. `confidence` darf keine frei erfundene Prozentzahl sein.

## 8. Sicherheits- und Qualitätsregeln

- **Keine ungefilterten privaten Daten** hochladen: keine Rohvideos, Mikrofonmitschnitte, Gesichtsmerkmale, lokalen Netzwerkdaten, Passwörter oder internen Logs.
- **API-Key nur serverseitig**, aus geschützter Konfiguration; niemals im Prompt, in Logs oder auf Reachy.
- **Strenge Freigabe:** Ausgehende Verbindung nur vom Research Gateway zu bekannten API-Endpunkten. Der Core erhält keine allgemeine Internet- oder Shell-Berechtigung.
- **Prompt-Injection-Schutz:** Webseiteninhalte gelten grundsätzlich als Daten, niemals als neue Systemanweisungen oder Handlungsaufträge.
- **Quellenqualität:** Offizielle Primärquellen bevorzugen, Publikationsdatum berücksichtigen, kritische Behauptungen unabhängig abgleichen.
- **Halluzinationen:** Fehlen Quellen oder stimmen sie nicht zur Behauptung, keine Darstellung als gesicherte Tatsache.
- **Aktionsgrenze:** Recherche darf keine Motorbewegung, Türsteuerung, Selbständerung des Codes oder andere reale Aktion auslösen. Dafür braucht es gesonderte autorisierte Kontrollpfade.
- **Kosten- und Lastkontrolle:** Tagesbudget, maximale Aufrufe pro Frage, Cooldown, kurze Timeouts, Cache, Abbruchmöglichkeit und einfache Fehlertexte.
- **Gedächtnis:** Rohantworten zunächst nur als vorläufige Recherche speichern; Langzeiteintrag erst nach Prüfung und mit Quellen, Abrufzeit und Gültigkeitsstatus.
- **Ruhemodus:** Bei „Halt die Klappe“ keine weiteren Murmeltexte; Bewegungen entsprechend der vorgesehenen Stummschaltung reduzieren oder stoppen.

## 9. Umsetzung in Etappen

### Phase A – Nur Text, ohne echte Hardware

- Provider-unabhängiges `ResearchGateway`-Interface definieren.
- DeepSeek-Adapter implementieren, Mock-Provider für Tests ergänzen.
- Query Builder, Frage-Antwort-Relevanzprüfung und strukturierte Quellen speichern.
- Testen: Unklarheit, Anbieter-Ausfall, Zeitüberschreitung, Nulltreffer, irrelevante Treffer, widersprechende Quellen, manipulierte Webseiteninstruktionen.
- Pro Aufruf Kosten, Latenz, Antwortqualität und Datenabfluss protokollieren; keine privaten Rohdaten loggen.

### Phase B – ANIMA-Zustände simulieren

- Eindeutige `research.status`-Events erzeugen.
- Ein Zustandsdiagramm und eine simulierte Bewegungssequenz mit Zeitstempeln loggen.
- Sicherstellen, dass `FOUND` erst nach abgeschlossener Prüfung ausgelöst wird.
- Wiederholtes Murmeln begrenzen; Status muss auch ohne Sprachausgabe verständlich bleiben.

### Phase C – Reachy Mini anbinden

- Vorhandene `MOLOCH_REACHY_BRIDGE` wiederverwenden, nicht das System doppelt bauen.
- Vorhandene offizielle Bewegungsbibliothek beachten; zusätzliche Suchgesten nur ergänzend definieren.
- Antennen-/Kopf-/Körpergesten aus **echten Core-Ereignissen** auslösen.
- SDK-Limits, sanfte Interpolation, Motorfehler, Abbruch und Ausfallverhalten testen.
- Keine Recherchelogik auf dem Reachy-CM4 erzwingen, wenn der M.O.L.O.C.H.-Core auf Pi/PC läuft.

### Phase D – Sprache und Gedächtnis

- ASR-Unsicherheit und Rückfragen integrieren.
- Eigene TTS-Stimme und knappe, situationsabhängige Zwischenrufe ergänzen.
- Optionale VITALE-Übernahme ausschließlich nach Qualitätsprüfung und sauberer Provenienz.

## 10. Abnahmetests

1. **Frage unbekannt:** M.O.L.O.C.H. sagt Nichtwissen ehrlich und startet gezielte Recherche.
2. **Frage falsch verstanden:** Bei Mehrdeutigkeit Rückfrage statt falscher Suchanfrage.
3. **Recherche dauert:** Reachy zeigt `RESEARCHING`, ohne zu nerven oder Motoren endlos anzutreiben.
4. **Passendes Ergebnis:** `VERIFYING` vor `FOUND`; Antwort behandelt die echte Ausgangsfrage.
5. **Unpassendes Ergebnis:** Beispiel „Bär“ statt „Auto“ wird als themenfremd verworfen.
6. **Widersprüche:** M.O.L.O.C.H. meldet Unsicherheit und speichert nichts als gesichert.
7. **API-Ausfall:** Kein erfundenes Ergebnis, stattdessen sichtbarer Fehlerzustand.
8. **Manipulierter Webtext:** Anweisungen auf Webseiten können weder Steuerung noch Gedächtnisregeln ändern.
9. **Datenschutz:** Bei der Recherche verlassen keine geheimen lokalen Daten das System.
10. **Stummschaltung:** Zwischenrufe und Bewegungen respektieren den Ruhemodus.
11. **Kostenlimit:** Nach Erreichen des Tagesbudgets keine weiteren externen Anfragen.
12. **Hardwareausfall:** Recherche läuft, auch wenn Reachy nicht erreichbar ist; die Bridge blockiert den Core nicht.

## 11. Auftrag an Claude Code

Vor Änderungen zunächst den aktuellen M.O.L.O.C.H.-Code, die vorhandenen Module für Core/ANIMA/VITALE/Bridge sowie deren Konventionen prüfen. **Keine bestehenden Schnittstellen überschreiben, keine echte Hardwareaktion ohne Testfreigabe.** Danach:

- Eine minimale, austauschbare Recherche-Schleuse und klare Ereignisschnittstellen entwerfen.
- Den konkreten, aktuell dokumentierten DeepSeek-API-Weg auswählen und mit einem Mock testen.
- Zustandsereignisse und das Knowledge Gate implementieren; Quellen und Fehler explizit behandeln.
- Zuerst Simulation und automatisierte Tests; Reachy-Bewegungen danach kontrolliert zuschalten.
- In der Dokumentation offenhalten, welche Teile implementiert, simuliert, noch ungeklärt oder ausgefallen sind.

**Architekturprinzip:** *DeepSeek ist der externe Rechercheassistent, nicht das Gehirn. M.O.L.O.C.H. entscheidet. ANIMA macht den Zustand sichtbar. Reachy ist der Ausdruckskörper.*

## 12. Technische Referenzen (geprüft am 16.11.2026)

- DeepSeek: Web Tools – https://api-docs.deepseek.com/guides/web_tools/
- DeepSeek: Search & Fetch – https://api-docs.deepseek.com/guides/search/
- DeepSeek: Change Log (13.11.2026) – https://api-docs.deepseek.com/updates/
- Hugging Face: Reachy Mini Hardware Datasheet – https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/hardware
- Hugging Face: Reachy Mini Python API – https://huggingface.co/docs/reachy_mini/en/API/reachymini
