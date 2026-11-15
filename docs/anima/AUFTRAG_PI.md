# ANIMA – Auftrag an die Pi-Sitzung

**Von:** Cloud-Sitzung 2026-11-15 · **Übersicht:** Issue #3 · **Hintergrund:** `BESTANDSAUFNAHME_2026-11-15.md`

Hardware laut Markus: **Raspberry Pi 5 mit 4 GB RAM.** Alles hier ist für 4 GB ausgelegt.

Die Cloud-Sitzung hat keinen Zugriff auf den Pi. Alles unter „gebaut" ist nur durch Tests ohne
Hardware belegt. Deine Aufgabe: übernehmen, am Gerät abnehmen, Ergebnisse festhalten, dann weiterbauen.

## Schritt 1 – Übernehmen

```bash
cd ~/moloch && git status                       # muss sauber sein
git fetch origin && git tag before_anima_p1_p5
git pull                                        # nach Merge des PR
find ~/moloch -type d -name __pycache__ -exec rm -rf {} +
sudo systemctl restart moloch moloch-chat
python3 ~/moloch/moloch_audit.py --auto         # bei FAIL: git reset --hard before_anima_p1_p5
python3 -m pytest tests -q                      # erwartet: 0 failed
```

Kein Reboot nötig (nur Python und Konfigurationsdateien, kein systemd, kein Treiber).

## Schritt 2 – Inventar (Phase B)

```bash
python3 scripts/anima_inventory.py              # schreibt logs/anima_inventar_<datum>.md
python3 scripts/check_after_merge.py
python3 scripts/self_diagnosis.py
python3 scripts/face_db_doctor.py
git add -f logs/anima_inventar_*.md && git commit -m "docs(anima): Laufzeit-Inventar" && git push
```

Im Inventar steht dann, was bisher nur vermutet ist: Dateisystem von SSD 2, ob Qdrant läuft,
welches Ollama-Modell geladen ist, NPU-Identität, Unterspannung, Lüfter-Tacho.

## Schritt 3 – Abnahme der gebauten Pakete

| Paket | Prüfung | Bestanden wenn |
|---|---|---|
| P1 Gedächtnis | `curl -s 'localhost:9100/memory/recall?q=Wer%20bist%20du%20eigentlich'` | `count` ≥ 1, Quelle `selbstwissen` |
| P1 Gedächtnis | Testfakt speichern (Befehl in der Bestandsaufnahme, Bereich 1), nach 10 s `…/memory/recall?q=Was%20ist%20das%20Codewort%20Zitronenfalter` | Treffer `Test_Codewort`, ohne Neustart |
| P1 Gedächtnis | `/chat` mit „Was haben Opus und Fable an dir programmiert?" | Antwort nennt Inhalte aus dem Selbstwissen, erfindet nichts |
| P2 NPU-Test | `python3 scripts/self_diagnosis.py` bei laufendem Service | „NPU rechnet (Service): … Inferenzen in 2s", **kein** Error 74 in `dmesg` |
| P5 Befehle | `curl -s -X POST localhost:9100/command -H 'Content-Type: application/json' -d '{"text":"Moloch, schau mal zur Tür"}'` | `handled: true`, Kamera fährt zur Tür |
| P5 Befehle | dasselbe mit „Die Tür klemmt schon wieder" | `handled: false`, keine Bewegung |
| P4 Sprechstatus | `curl -s -X POST localhost:9100/speech/state -H 'Content-Type: application/json' -d '{"speaking":true,"text":"Test"}'`, dann `curl -s localhost:9100/live` | `speech.speaking: true`, `event: "chat"` |

Danach Testfakt entfernen und P4 zurücksetzen (`{"speaking":false}`).

**Beobachten:** Der Wort-Recall hängt bis zu drei Zeilen an den Prompt des Tentakel-Pfads. Werden die
Antworten dadurch schlechter: `MOLOCH_MEMORY_RECALL_LEXICAL=0` in `~/.profile` und in Issue #3 melden.

## Schritt 4 – Ergebnisse festhalten

- `logs/agent_handoff.md`: pro Prüfung bestanden / nicht bestanden, mit Ausgabe.
- Issue #3: Haken setzen bei abgenommenen Paketen.
- `docs/anima/BESTANDSAUFNAHME_2026-11-15.md`: Spalte „Am Gerät belegt" nachtragen.
- Der PC-Sitzung per Mailbox Bescheid geben (`info_cowork_anima_pi_abnahme`).

## Schritt 5 – Weiterbauen (Pi-Territorium)

In dieser Reihenfolge, jedes Paket einzeln mergebar, mit Abnahmetest:

1. **P6 Hardware-Fragen mit Messwerten** (Bereich 11). „Wie warm ist dein Raspberry?" soll einen echten
   Messwert liefern. Vorschlag: Funktion, die Temperatur, RAM, Plattenplatz, Unterspannung, NPU-Zähler
   als kurze Zeilen liefert (Quellen wie in `scripts/anima_inventory.py`), und sie bei passender Frage
   wie den Recall an den Prompt hängen. Nicht messbar = „unbekannt", nie schätzen.
   Agent: watchdog. Kein Zugriff auf `_force_release_device`.
2. **P8 `/live` als Situationszustand** (Bereich 10). Vorhandene Felder mit Zeitstempel, Quelle und
   Gültigkeit versehen, kein neues Zustandsmodell. Agent: bridge + awareness (nacheinander).
3. **P9 Audio-Ereignisse** (Bereich 5). Erst klären, ob der ReSpeaker stabil liefert (0-Byte-Befund vom
   13.06.). Dann VAD → Ereignis `audio.sound` im EventBus, gesperrt während eigener Sprachausgabe.
   Agent: tentacle / voice. `voice_pipeline.py` ist ROT.
4. **P11 Raumkalibrierung** (Bereich 4). Braucht Messreihe von Markus (0,3 / 0,5 / 1,5 / 3,0 m).

Nicht anfangen ohne Rücksprache: P13 (Reachy-Modell) – alle acht NPU-Plätze sind belegt.

## Offene Nebenbefunde in ROT-Dateien (brauchen Markus' Nicken)

- `tappas_pipeline._continuous_learn`: Namensvergleich mit `.lower()` – bei Namen mit Großbuchstaben
  greift die Obergrenze von 50 Einträgen nie.
- `moloch_service` Teach-Modus (~Z. 512): schreibt `face_embeddings.json` nicht atomar und fest unter „Markus".
- Face-DB: zwei unverträgliche Sätze für Markus, 34 Einträge „nicht". Vorgehen: Handoff vom 14.11.
