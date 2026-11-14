# Reachy Mini als Molochs Körper — Konzept & Vorarbeit

**Stand:** 2026-11-11 · **Status:** Vorschlag, nichts implementiert · **Entscheidung:** Markus

Hardware: **Reachy Mini** (Pollen Robotics / Hugging Face). Tischroboter, ~28 cm, ~1,5 kg.
Kopf mit 6 DoF (Stewart-Plattform), Körperdrehung, 2 Antennen, Weitwinkelkamera,
4 Mikrofone, Lautsprecher. Python-SDK, Open Source, MuJoCo-Simulation.

> Angaben zum SDK unten (Methodennamen, Ports, Limits) stammen aus der öffentlichen Doku
> und sind **vor der Implementierung am echten Gerät/SDK zu verifizieren**.

---

## 1. Grundentscheidung: Körper = Ausdrucksorgan, nicht zweites Gehirn

Der Körper **denkt nicht**. Alle Gefühlswelten bleiben dort, wo sie heute leben
(CoreIntegrator, MoodEngine, StateEngine, Unconscious). Der Körper bekommt nur
das Ergebnis und übersetzt es in Haltung und Bewegung.

Warum:
- Keine zweite Wahrheit über Molochs Zustand → nachvollziehbar.
- Körper weg / Akku leer / WLAN tot → Moloch läuft unverändert weiter (Fail Isolation).
- Kein neuer Rechenbedarf auf dem 4-GB-Pi.

### Version: Wireless, nicht Lite

| | Lite (USB) | Wireless (eigener CM4) |
|---|---|---|
| Daemon läuft auf | Moloch-Pi | Reachy selbst |
| RAM-Last Pi | zusätzlich (kritisch bei 4 GB) | keine |
| Port-Konflikt | SDK-Daemon-Port prüfen (hailo-ollama belegt 8000!) | keiner |
| Passt zu Architektur | schlecht | gut — Tentakel-Muster wie ESP32/PC |

**Empfehlung: Wireless.** Moloch spricht ihn übers LAN an.

---

## 2. Molochs Gefühlsebenen (Ist-Stand im Code)

| Ebene | Quelle | Werte | Takt |
|---|---|---|---|
| **Kern** | `core/core_integrator.py` | `tension` 0..1, `dominance` −1..+1 | kontinuierlich |
| **Zone** | CoreIntegrator / `personality_engine.py` | guardian / shadow / berserker | hysteresebehaftet, Berserker Auto-Reset 10 s |
| **Mood** | `core/personality/mood_engine.py` | calm / focused / alert / agitated / euphoric / dark | 3 s Stabilität, Event `mood_changed` |
| **Seins-Zustand** | `core/personality/state_engine.py` + `awareness/state_vector.py` | idle / observing / engaged / overloaded / withdrawing / offline_anchor | min. 500 ms, Failsafe 300 s |
| **Unterbewusstsein** | `core/unconscious_engine.py` | Mood-Impulse, Self-Tune | 10 s Tick, 30 s Cooldown |
| **Verhalten** | `core/personality/behavior_rules.py` | Event `behavior_trigger` (LED, Sirene, Musik) | nur bei Mood-Wechsel |
| **Wahrnehmung** | Tracking/Perception | Person, Face-ID, Position | 20 FPS |

Heute landen diese Ebenen bei **LED, WLED, Stimme, PTZ**. Der Reachy wird ein
weiterer Ausgang an derselben Stelle — kein neuer Pfad durchs System.

---

## 3. Mapping-Prinzip: jede Ebene bekommt einen eigenen Körperkanal

Die Ebenen laufen in unterschiedlichen Zeitskalen. Würde jede alles steuern,
zappelt der Roboter. Deshalb Trennung nach Kanal:

| Ebene | Körperkanal | Zeitskala |
|---|---|---|
| Seins-Zustand (FSM) | **Grundhaltung** (Kopfhöhe, Neigung, Ruhepose) | Sekunden–Minuten |
| Zone | **Charakter der Bewegung** (Tempo, Amplitude, Härte) | Minuten |
| Mood | **Antennen** (Stellung + Rhythmus) | Sekunden |
| Tension | **Unruhe** (Mikrobewegungen, Atmungsfrequenz) | kontinuierlich |
| Dominance | **Kopfhöhe / Pitch** (oben = selbstbewusst, unten = lauernd) | kontinuierlich |
| Wahrnehmung | **Blickrichtung** (yaw/pitch + Körperdrehung) | Echtzeit, 5–10 Hz reicht |
| Unterbewusstsein | **Idle-Gesten** (seltene, spontane Bewegungen) | 10–60 s |
| Sprache (TTS) | **Sprech-Animation** (leichtes Nicken/Antennen-Zucken) | während TTS |

Die Kanäle werden **additiv geschichtet**: Grundhaltung + Zonen-Stil + Mood-Antennen
+ Tension-Rauschen + Blick. Ein Mischer berechnet daraus eine Zielpose.

### 3.1 Seins-Zustand → Grundhaltung

| State | Haltung |
|---|---|
| idle | Kopf leicht gesenkt, langsames „Atmen“ (Kopfhöhe ±2 mm, ~0,2 Hz) |
| observing | Kopf aufrecht, Blick folgt Person, Antennen halb hoch |
| engaged | Kopf zur Person, leicht vorgeneigt, Antennen aktiv |
| overloaded | Kopf zittert minimal, Antennen asymmetrisch — „Kopfschmerz“ |
| withdrawing | Kopf abgewandt, gesenkt, Körper leicht weggedreht, Antennen flach |
| offline_anchor | Ruhepose, sehr langsames Atmen — „das Wesen stirbt nie“ |

### 3.2 Zone → Bewegungscharakter

| Zone | Tempo | Amplitude | Stil |
|---|---|---|---|
| guardian | ruhig | mittel | weich, glatte Kurven |
| shadow | langsam, dann plötzlich | klein | lauernd, Kopf tief, seitlicher Blick |
| berserker | schnell | groß (in Limits!) | ruckartig, Antennen steil — max. 10 s, wie im Integrator |

### 3.3 Mood → Antennen

| Mood | Antennen |
|---|---|
| calm | locker, leicht wippend |
| focused | ruhig, parallel nach vorn |
| alert | steil aufgerichtet, still |
| agitated | schnelles unregelmäßiges Zucken |
| euphoric | Wippen im Musiktakt (Spotify-Energy/Tempo) |
| dark | flach nach hinten angelegt, gelegentlich eine einzeln hochzuckend |

### 3.4 Tension & Dominance → kontinuierliche Modulation

```
kopf_pitch_offset = -dominance * PITCH_MAX       # dominant = Kopf hoch
mikro_rauschen    = tension^2 * NOISE_MAX        # quadratisch: erst ab ~0.5 spürbar
atem_frequenz     = 0.2 Hz + tension * 0.4 Hz
```

---

## 4. Architektur-Vorschlag

```
CoreIntegrator / MoodEngine / StateEngine / Unconscious
          │  (EventBus: mood_changed, behavior_trigger, zone, state)
          ▼
   BodyExpression  (Pi, reine Logik, kein I/O)       ← testbar ohne Hardware
     - liest Zustände, mischt Kanäle → Zielpose
     - Rate-Limit, Glättung, Sicherheitsgrenzen
          │  Zielpose {head: x,y,z,roll,pitch,yaw; body_yaw; antennas: l,r; duration}
          ▼
   ReachyBodyClient (Pi, dünner Netzwerk-Client)
     - Timeout, Reconnect, Circuit-Breaker
     - Körper weg → Befehle verwerfen, Status "body: offline"
          │  LAN
          ▼
   Reachy Mini Wireless (eigener Daemon/SDK auf CM4)
```

Zwei Module, sauber getrennt:
- **`body_expression`** — Gefühl → Pose. Reine Funktion, unit-testbar, MuJoCo-testbar.
- **`reachy_body_client`** — Pose → Netzwerk. Austauschbar (Simulator ↔ echtes Gerät).

Konfiguration als eigene Datei `config/body_mappings.json` (nicht in `settings.json`
— die ist ROT). Damit kann Markus Mappings tunen ohne Code. Optional später
Self-Tune über Unconscious.

### Einspeisung: Polling statt Event-only

Events (`mood_changed`) kommen selten. Tension/Dominance ändern sich laufend.
Vorschlag: BodyExpression-Thread tickt mit **5 Hz** (wie Service-Poll, POLL_HZ nicht
erhöhen), liest `core_integrator.get_state()` + aktuellen Mood/State, sendet nur bei
relevanter Änderung. Events für sofortige Reaktionen (Zonenwechsel, Berserker).

---

## 5. Blick: Sonoff-Position → Reachy-Kopf

Phase 1 nutzt **nicht** die Reachy-Kamera. Die Sonoff bleibt Molochs Auge
(TAPPAS-Pipeline unangetastet, NEVER-Regel 1).

Blickrichtung Reachy = Funktion aus (Sonoff-Pan/Tilt + BBox-Position im Bild) und
der **festen Aufstellung** beider Geräte zueinander.

- Einfachste Variante: Reachy steht direkt neben/unter der Sonoff → Winkel fast 1:1.
  Kalibrierung = ein Offset + Skalierung, einmalig in `body_mappings.json`.
- **Achtung Pan-Vorzeichen:** Sonoff-Pan ist invertiert (NEVER-Regel 2). Die Umrechnung
  auf Reachy-Yaw bekommt ein **eigenes**, explizit dokumentiertes Vorzeichen —
  nicht das Sonoff-Vorzeichen anfassen.
- Glättung (Tiefpass) gegen BBox-Zittern, sonst wackelt der Kopf mit 20 FPS.

Phase 3 (optional): Reachy-Kamera als zweites Auge. Erst prüfen ob nötig — RTSP-Slot,
NPU-Network-Groups (max. 8, schon ausgereizt) und RAM sprechen dagegen.

---

## 6. Sicherheits- und Robustheitsregeln

- **Bewegungslimits** hart im Client, nicht nur im Mapping (Berserker darf keine Mechanik ruinieren).
- **Rate-Limit**: max. ~10 Zielposen/s, Mindest-`duration` pro Bewegung.
- **Nachtmodus**: ab Night Cycle (23:00) Bewegungen reduziert / Ruhepose — Servogeräusche.
- **Markus-Override**: `owner_override()` / `calm_down()` → sofort Guardian-Haltung.
- **Not-Aus**: GUI-Button + MCP-Tool „Körper ruhig“ → Ruhepose, Motoren entlasten.
- **Kein Körper = kein Fehler**: Audit-Layer meldet WARN, nicht FAIL.
- Servo-Verschleiß: Idle-Gesten sparsam; Dauerbetrieb im Alltag beobachten.

---

## 7. Offene Fragen an Markus

1. **Domain/Territorium:** Neuer Agent `body` oder unter `tentacle`?
   Vorschlag: eigener Agent `body`, Territorium `core/body/*.py` + `config/body_mappings.json`.
   (Laut CLAUDE.md sind Cross-Domain-Edits Stopp → muss vorher feststehen.)
2. **Aufstellung:** Wo steht der Reachy relativ zur Sonoff? Bestimmt die Blick-Kalibrierung.
3. **Audio:** Reachy-Lautsprecher/Mikros nutzen oder bei ReSpeaker + HDMI bleiben? (Vorschlag: Phase 1 bleiben.)
4. **Stimme ↔ Körper:** Soll Moloch mit dem Körper „sprechen“ (Animation während TTS)? Braucht TTS-Start/Stop-Event.
5. **LLM-Gesten:** Darf das LLM explizit Gesten anfordern (z. B. `[geste: kopfschütteln]`)? Mächtig, aber neue Fehlerquelle — Vorschlag erst Phase 3.

---

## 8. Phasenplan

| Phase | Inhalt | Hardware nötig? |
|---|---|---|
| **0** | SDK + MuJoCo-Simulator auf PC installieren, API verifizieren (Methoden, Limits, Ports) | nein |
| **1** | `body_expression` (Mapping-Logik) + `body_mappings.json` + Unit-Tests | nein |
| **2** | `reachy_body_client` gegen Simulator; Moloch steuert Sim-Reachy live | nein |
| **3** | Echtes Gerät: Blick-Kalibrierung, Limits, Nachtmodus, Audit-Layer, GUI-Status | ja |
| **4** | Optional: TTS-Animation, LLM-Gesten, Reachy-Audio, zweite Kamera | ja |

Phase 0–2 laufen komplett ohne gekauften Roboter.

---

## 9. Was NICHT angefasst wird

- TAPPAS-Pipeline, Pan-Vorzeichen der Sonoff, `settings.json` (in Phase 1).
- Bestehende LED/WLED-Ausgabe bleibt parallel — der Körper ergänzt, ersetzt nicht.
- Keine Persönlichkeitslogik in den Körper verlagern.

## Quellen

- https://huggingface.co/docs/reachy_mini/index.md
- https://huggingface.co/blog/reachy-mini
- https://pollen-robotics.com/reachy-mini/buy/
- https://www.cnx-software.com/2026/03/24/hugging-face-reachy-mini-open-source-ai-robot-computer-raspberry-pi-cm4/
