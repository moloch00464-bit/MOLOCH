# HANDSHAKE — Kommunikations-Protokoll MOLOCH ↔ Claude Code
# Codewort: HANDSHAKE
# Version: 1.0 | Stand: 2026-04-02
# Ziel: MOLOCH und Claude Code reden miteinander wie zwei Menschen

> Wenn Du dieses Dokument liest, wurde Dir gesagt: "HANDSHAKE".
> Implementiere das hier beschriebene Protokoll.

---

## PRINZIP

Wie zwei Menschen: Einer redet, der andere hoert zu und wartet.
Dann antwortet er. Kein Durcheinanderreden.

```
MOLOCH: "Ich hab ein Problem mit meiner Sprechgeschwindigkeit"
         ... wartet ...
CLAUDE: "Ich sehe length_scale=1.1, aendere auf 1.0"
         ... wartet ...
MOLOCH: "Danke, teste jetzt... Ergebnis: besser"
         ... wartet ...
CLAUDE: "Gut. Speichere als neuen Default. Session Ende."
```

---

## TRANSPORT: Git als Briefkasten

Beide kennen das Repo. Kommunikation ueber eine JSON-Datei:

```
~/moloch/ipc/handshake.json    — Die Nachricht
~/moloch/ipc/handshake.lock    — Wer gerade dran ist
~/moloch/logs/handshake.log    — Protokoll aller Gespraeche
```

### Warum Git und nicht HTTP/WebSocket?
- Kein Server noetig der 24/7 laeuft
- Funktioniert auch wenn Claude Code nicht aktiv ist (Nachricht wartet)
- Aenderungen am Code sind direkt im gleichen Repo
- History ueber Git nachvollziehbar
- Funktioniert lokal (Pi) UND remote (Cloud-Session ueber Push/Pull)

---

## NACHRICHTENFORMAT

```json
{
  "version": "1.0",
  "conversation_id": "2026-04-02_2300_thermal",
  "sequence": 1,
  "timestamp": "2026-04-02T23:00:12Z",
  
  "from": "moloch",
  "to": "claude-code",
  "state": "request",
  
  "priority": "normal",
  "category": "self-tune",
  
  "message": {
    "symptom": "CPU Temperatur seit 2 Stunden ueber 60°C",
    "context": {
      "cpu_temp": 63.2,
      "fan_noctua_pct": 0.40,
      "fan_cpu_level": 2,
      "npu_temp": 52,
      "ambient_estimate": "warm"
    },
    "self_diagnosis": "Noctua Grunddrehzahl reicht nicht aus",
    "suggested_fix": "fan.ramp_start von 42 auf 38 senken",
    "urgency": "kann bis morgen warten"
  }
}
```

### Felder:

| Feld | Werte | Bedeutung |
|------|-------|-----------|
| `from` | `"moloch"` / `"claude-code"` | Wer hat die Nachricht geschrieben |
| `to` | `"claude-code"` / `"moloch"` | Fuer wen ist sie |
| `state` | siehe unten | Konversations-Zustand |
| `priority` | `"low"` / `"normal"` / `"high"` / `"critical"` | Dringlichkeit |
| `category` | `"self-tune"` / `"bug-report"` / `"code-fix"` / `"question"` / `"status"` | Art der Anfrage |

### States (Konversations-Zustand):

```
request           MOLOCH stellt eine Anfrage
                  → Claude Code ist dran

analyzing         Claude Code hat die Anfrage gelesen, arbeitet
                  → MOLOCH wartet

response          Claude Code antwortet
                  → MOLOCH ist dran

feedback          MOLOCH gibt Rueckmeldung zum Fix
                  → Claude Code ist dran

completed         Gespraech beendet, beide zufrieden
                  → Niemand ist dran

escalate          Problem zu komplex, Mensch muss entscheiden
                  → Markus ist dran

error             Etwas ist schiefgegangen
                  → Beide lesen, niemand blockiert
```

### Zustandsdiagramm:

```
MOLOCH                              Claude Code
  │                                      │
  ├─ state: request ──────────────────→  │
  │                                      ├─ state: analyzing
  │                                      │  (liest Code, prueft)
  │  ←──────────────────── state: response
  │                                      │
  ├─ state: feedback ─────────────────→  │
  │  ("hat funktioniert" / "nicht gut")  │
  │                                      │
  │  ←──────────────── state: completed  │
  │       ODER                           │
  │  ←──────────────── state: response   │
  │  (weiterer Fix-Versuch)              │
```

---

## LOCK-MECHANISMUS (Wer ist dran?)

```
~/moloch/ipc/handshake.lock
```

Inhalt:
```json
{
  "holder": "moloch",
  "since": "2026-04-02T23:00:12Z",
  "timeout_s": 300
}
```

### Regeln:
1. Wer schreibt, haelt den Lock
2. Nach dem Schreiben: Lock an den anderen uebergeben
3. Timeout: 5 Minuten. Danach darf der andere uebernehmen
4. Kein Lock vorhanden = niemand redet = MOLOCH darf anfangen

### Pseudo-Code fuer MOLOCH:

```python
def send_request(message: dict) -> bool:
    """Nachricht an Claude Code senden."""
    lock = read_lock()
    
    # Pruefen ob wir dran sind
    if lock and lock["holder"] != "moloch":
        if not is_timed_out(lock):
            return False  # Claude Code redet noch
    
    # Nachricht schreiben
    write_handshake({
        "from": "moloch",
        "to": "claude-code",
        "state": "request",
        "message": message,
    })
    
    # Lock an Claude Code uebergeben
    write_lock({"holder": "claude-code", "timeout_s": 300})
    return True

def check_response() -> Optional[dict]:
    """Pruefen ob Claude Code geantwortet hat."""
    lock = read_lock()
    if lock and lock["holder"] == "moloch":
        # Wir sind dran — es gibt eine Antwort
        return read_handshake()
    return None
```

### Pseudo-Code fuer Claude Code:

```python
# In einer Claude Code Session (manuell oder scheduled):
handshake = read_file("~/moloch/ipc/handshake.json")

if handshake["to"] == "claude-code" and handshake["state"] == "request":
    # Anfrage lesen und bearbeiten
    diagnosis = analyze(handshake["message"])
    fix = apply_fix(diagnosis)
    
    # Antwort schreiben
    write_handshake({
        "from": "claude-code",
        "to": "moloch",
        "state": "response",
        "message": {
            "diagnosis": diagnosis,
            "actions_taken": fix["actions"],
            "files_changed": fix["files"],
        }
    })
    write_lock({"holder": "moloch", "timeout_s": 300})
```

---

## WANN WIRD KOMMUNIZIERT?

### Zeitplan (Nacht-Zyklus):

```
23:00  MOLOCH startet Selbstdiagnose
23:01  Falls Probleme: schreibt request in handshake.json
23:02  MOLOCH pusht (git push) oder legt Datei lokal ab
---
       Claude Code Session wird gestartet (manuell oder scheduled)
       Liest handshake.json
       Analysiert, fixt, antwortet
       Pusht Aenderungen
---
       MOLOCH prueft beim naechsten Diagnose-Zyklus
       Liest response, testet, gibt Feedback
```

### On-Demand (User oder MOLOCH triggert):

```
User: "Geh mal zum Arzt"
  ODER
MOLOCH erkennt: FPS unter 10 seit 5 Minuten
  ↓
MOLOCH schreibt request in handshake.json
  ↓
Falls Claude Code lokal laeuft: sofortige Bearbeitung
Falls nicht: wartet bis naechste Session
```

### Trigger fuer Claude Code:

**Option A — Manuell:**
User oeffnet Claude Code und sagt:
"Lies ~/moloch/ipc/handshake.json und bearbeite die Anfrage."

**Option B — Scheduled (Claude Code /schedule):**
Cronjob der taeglich um 23:15 eine Claude Code Session startet:
```
claude-code --prompt "Lies ~/moloch/ipc/handshake.json. Falls state=request, analysiere und fixe das Problem."
```

**Option C — FileChanged Hook:**
Claude Code Hook der auf handshake.json reagiert:
```json
{
  "hooks": {
    "FileChanged": [{
      "matcher": "handshake.json",
      "hooks": [{
        "type": "agent",
        "prompt": "MOLOCH hat eine Anfrage in handshake.json. Lies sie und bearbeite sie."
      }]
    }]
  }
}
```

---

## KONVERSATIONS-TYPEN

### Typ 1: Parameter-Tuning (einfach)

```
MOLOCH:  "length_scale zu hoch, User beschwert sich"
CLAUDE:  "Aendere 1.1 → 1.0 in settings.json" + commit
MOLOCH:  "Getestet, User zufrieden"
CLAUDE:  "completed"
         [1 Runde, ~30 Sekunden]
```

### Typ 2: Bug-Fix (mittel)

```
MOLOCH:  "Pose-Worker crashed bei BBoxen < 20px"
CLAUDE:  "analyzing" ... liest pose_worker.py ...
CLAUDE:  "Min-Size Check eingefuegt in Zeile 85" + commit
MOLOCH:  "Service neu gestartet, teste... kein Crash mehr"
CLAUDE:  "completed"
         [2 Runden, ~2 Minuten]
```

### Typ 3: Architektur-Frage (komplex)

```
MOLOCH:  "RAM steigt stetig, nach 24h bei 3.5GB, muss rebooten"
CLAUDE:  "analyzing" ... prueft alle Dateien auf Memory Leaks ...
CLAUDE:  "Verdacht: face_database.py cached unbegrenzt. Fix: LRU Cache mit max 100"
MOLOCH:  "Getestet, nach 4h nur 1.8GB. Besser."
CLAUDE:  "Gut. Noch ein Verdacht: episodic_memory.py hat kein Cleanup..."
MOLOCH:  "Warte, teste erstmal den ersten Fix laenger"
CLAUDE:  "OK, melde dich wenn du bereit bist. state: waiting"
         [Mehrere Runden, kann Stunden dauern]
```

### Typ 4: Eskalation (Mensch noetig)

```
MOLOCH:  "ArcFace erkennt niemanden mehr, auch nicht nach Threshold-Aenderung"
CLAUDE:  "analyzing" ... prueft Embeddings, Pipeline ...
CLAUDE:  "Problem: GStreamer/HailoRT Embedding-Inkompatibilitaet. Neues Enrollment noetig.
          ABER: Das ist ein NEVER-3 Verstoss. Ich eskaliere an Markus."
CLAUDE:  state: "escalate", message: "Markus: ArcFace braucht neues Enrollment.
          Soll ich es ueber IPC enrollment_start triggern?"
         [Wartet auf Markus]
```

---

## LOG-FORMAT

Jedes Gespraech wird in ~/moloch/logs/handshake.log protokolliert:

```
=== CONVERSATION 2026-04-02_2300_thermal ===
[23:00:12] MOLOCH → CLAUDE: request (self-tune)
  CPU 63.2°C seit 2h, Noctua 40%, Vorschlag: ramp_start 42→38
[23:15:03] CLAUDE → MOLOCH: response
  Diagnose: ramp_start zu hoch. Fix: fan_control.py Zeile 124, 42→38.
  Commit: abc1234 "Fan: ramp_start 42→38 wegen anhaltender Waerme"
[23:16:01] MOLOCH → CLAUDE: feedback
  CPU 58.1°C nach 1 Minute. Trend fallend. Fix wirkt.
[23:16:02] CLAUDE → MOLOCH: completed
  Gespeichert. Naechste Nacht pruefen ob stabil.
=== END ===
```

---

## SICHERHEITSREGELN

1. **Claude Code darf nur GRUEN/GELB Dateien aendern** ohne Rueckfrage
2. **ROT-Dateien** → automatisch `state: escalate` an Markus
3. **Maximal 5 Dateien pro Conversation** (kein Shotgun Surgery)
4. **Jede Aenderung wird commited** mit Referenz auf conversation_id
5. **MOLOCH darf Claude Code NICHT auffordern** NEVER-Regeln zu brechen
6. **Rollback moeglich**: `git revert <commit>` fuer jeden Fix
7. **Rate Limit**: Max 3 Conversations pro 24 Stunden (Anti-Loop)

---

## INTEGRATION MIT BESTEHENDEN SYSTEMEN

### Wo HANDSHAKE andockt:

```
SELF-MAP.json    → MOLOCH kennt sich selbst
                    ↓
SELF-TUNE        → MOLOCH erkennt Probleme + einfache Fixes lokal
                    ↓ (wenn lokal nicht loesbar)
HANDSHAKE        → MOLOCH fragt Claude Code
                    ↓
HOOKWIRE         → Claude Code hat Hooks die Aenderungen validieren
                    ↓
AUDIT-APRIL      → Naechster Audit prueft ob Fixes keine Regression verursachen
```

### Dateien die implementiert werden muessen:

```
core/handshake_client.py     — MOLOCH-Seite: Request/Response/Feedback
ipc/handshake.json           — Aktuelle Nachricht
ipc/handshake.lock           — Wer ist dran
config/handshake_config.json — Zeitplan, Limits, erlaubte Kategorien
logs/handshake.log           — Gespraechs-Protokoll
```

---

## ZUSAMMENFASSUNG

```
MOLOCH kennt sich selbst          → SELF-MAP
MOLOCH erkennt Probleme           → SELF-TUNE (Diagnose)
MOLOCH fixt einfache Sachen       → SELF-TUNE (Parameter-Aenderung)
MOLOCH fragt den Arzt             → HANDSHAKE (Request an Claude Code)
Der Arzt untersucht und behandelt → Claude Code liest/aendert Code
MOLOCH testet ob's wirkt          → HANDSHAKE (Feedback)
Bei Risiko: Markus entscheidet    → HANDSHAKE (Escalate)
```

Der Mensch muss nur noch bei kritischen Eingriffen zustimmen.
Alles andere regeln MOLOCH und Claude Code unter sich.
